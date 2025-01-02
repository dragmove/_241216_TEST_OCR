from flask import Flask, request, jsonify
import torch
import certifi
import os
from flask_cors import CORS
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
import json
import spacy
from transformers import LayoutLMForTokenClassification, LayoutLMTokenizer
from transformers import DonutProcessor, VisionEncoderDecoderModel
import torch
from PIL import Image
import re

os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

app = Flask(__name__)
CORS(app)


@app.route("/doctr", methods=["GET"])
def doctr():
    print("/doctr")

    imgPath = request.args.get("imgPath", type=str, default="")
    print("[/doctr] imgPath :", imgPath)

    model = ocr_predictor(
        det_arch="db_resnet50", reco_arch="crnn_vgg16_bn", pretrained=True
    )

    # 이미지에서 텍스트 추출 (예시로 영수증 이미지 사용)
    doc = DocumentFile.from_images(imgPath)

    # Analyze
    result = model(doc)
    print("[/doctr] result :", result)

    """
    for page in result.pages:
        for block in page.blocks:
            for line in block.lines:
                print(line.words)

    # @see https://github.com/mindee/doctr?tab=readme-ov-file#dealing-with-rotated-documents
    # matplotlib과 mplcursors lib 사용하여, model prediction을 visualize + mouse interaction 가능
    # result.show()
    """

    # 출력
    output_data = {}
    for page_index, page in enumerate(result.pages):
        output_data["blocks"] = []

        for block_index, block in enumerate(page.blocks):
            block_data = {"block_index": block_index + 1, "lines": []}

            for line_index, line in enumerate(block.lines):
                line_data = {"line_index": line_index + 1, "words": []}

                for word in line.words:
                    word_data = {"word": word.value, "confidence": word.confidence}
                    line_data["words"].append(word_data)

                block_data["lines"].append(line_data)

            output_data["blocks"].append(block_data)

    output = json.dumps(output_data, ensure_ascii=False, indent=4)
    # print("output :", output)

    # @see https://github.com/mindee/doctr?tab=readme-ov-file#dealing-with-rotated-documents
    # matplotlib과 mplcursors lib 사용하여, model prediction을 visualize + mouse interaction 가능
    # result.show()

    outputJson = {}
    if isinstance(output, str):
        outputJson = json.loads(output)

    print('outputJson["blocks"] :', outputJson["blocks"])

    words = []
    # data["blocks"]의 각 블록을 순회
    for block in outputJson["blocks"]:
        for line in block["lines"]:
            for word_info in line["words"]:
                word = word_info["word"]  # 각 word 정보에서 "word" 값 추출
                words.append(word)  # 추출한 단어를 리스트에 추가

    combinedWord = " ".join(words)
    print("combinedWord :", combinedWord)

    # 구조화 1. NER with spaCy
    # entities = getEntitiesByNLP(combinedWord)
    # print("entities :", entities)

    # 구조화 2. layoutLM (NLP 딥러닝 모델)
    # getResultByLayoutLM(combinedWord) # 훈련된 모델이 아니라서 잘 안 된다.

    # FIXME: ing
    # 구조화 3. pre-trained 특화 모델 사용
    getPretrainedNaverDonut(imgPath, combinedWord)

    return jsonify(output), 200


def getEntitiesByNLP(text):
    # NER with spaCy
    nlp = spacy.load("en_core_web_sm")

    doc = nlp(text)

    # 추출된 엔티티 출력 (NER)
    for ent in doc.ents:
        print(f"{ent.text} - {ent.label_}")

    # 정리 case 1. { text, label } entity 구조로 정리
    entities = []
    for ent in doc.ents:
        entities.append({"text": ent.text, "label": ent.label_})
    return entities

    # 정리 case 2. { label1: [text1, text2], label2: [text1, text2, ...], ... } dictionary 구조로 정리
    # entity_dict = {}
    # for ent in doc.ents:
    #     if ent.label_ not in entity_dict:
    #         entity_dict[ent.label_] = []
    #     entity_dict[ent.label_].append(ent.text)

    # return entity_dict


def getPretrainedNaverDonut(image_path, text):
    # 영수증과 invoice 특화된 pre-trained model을 사용해서 결과를 보자.

    # @see https://huggingface.co/naver-clova-ix/donut-base
    # @see https://huggingface.co/docs/transformers/main/en/model_doc/donut
    processor = DonutProcessor.from_pretrained(
        "naver-clova-ix/donut-base-finetuned-cord-v2"
    )
    model = VisionEncoderDecoderModel.from_pretrained(
        "naver-clova-ix/donut-base-finetuned-cord-v2"
    )

    print("torch.cuda.is_available() :", torch.cuda.is_available())

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # prepare decoder inputs
    # @see https://github.com/clovaai/cord
    task_prompt = "<s_cord-v2>"
    decoder_input_ids = processor.tokenizer(
        task_prompt, add_special_tokens=False, return_tensors="pt"
    ).input_ids

    image = Image.open(image_path)
    pixel_values = processor(image, return_tensors="pt").pixel_values

    outputs = model.generate(
        pixel_values.to(device),
        decoder_input_ids=decoder_input_ids.to(device),
        max_length=model.decoder.config.max_position_embeddings,
        pad_token_id=processor.tokenizer.pad_token_id,
        eos_token_id=processor.tokenizer.eos_token_id,
        use_cache=True,
        bad_words_ids=[[processor.tokenizer.unk_token_id]],
        return_dict_in_generate=True,
    )

    sequence = processor.batch_decode(outputs.sequences)[0]
    sequence = sequence.replace(processor.tokenizer.eos_token, "").replace(
        processor.tokenizer.pad_token, ""
    )
    sequence = re.sub(
        r"<.*?>", "", sequence, count=1
    ).strip()  # remove first task start token

    print("sequence :", sequence)

    result = processor.token2json(sequence)
    print("result :", result)

    return "oh donut"


def getResultByLayoutLM(text):
    # 잘 안 된다. 'ㅅ');;
    model_name = "microsoft/layoutlmv2-base-uncased"  # LayoutLM 모델의 이름
    model = LayoutLMForTokenClassification.from_pretrained(model_name)
    tokenizer = LayoutLMTokenizer.from_pretrained(model_name)

    # 입력 텍스트를 토크나이즈하고 모델에 입력할 수 있는 형태로 변환
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)

    # 모델을 통해 예측 수행
    with torch.no_grad():
        outputs = model(**inputs)

    # 예측된 라벨과 토큰을 추출
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)

    # 토큰과 예측된 라벨 출력
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    predicted_labels = predictions[0].tolist()

    # 모델 레이블 정보
    labels = model.config.id2label

    # 예측된 레이블과 함께 토큰 출력
    for token, label_id in zip(tokens, predicted_labels):
        label = labels[label_id]
        print(f"{token} -> {label}")

    # 토큰과 라벨을 함께 출력합니다.
    # for token, label in zip(tokens, token_labels):
    # print(f"Token: {token}, Label: {label}")

    # 각 토큰에 대해 예측된 라벨을 출력합니다.
    # for token, label_id in zip(tokens, predictions[0].tolist()):
    # print(f"{token} -> {labels[label_id]}")

    return "oh yes"


def main():
    app.run(debug=True, port=5000)


if __name__ == "__main__":
    main()
