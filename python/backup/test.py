from flask import Flask, request, jsonify
import easyocr
from transformers import BertTokenizer, BertForTokenClassification, pipeline
import torch
import requests
import certifi
import ssl
import os
from flask_cors import CORS

# ssl_context = ssl.create_default_context(cafile=certifi.where())
# ssl._create_default_https_context = ssl._create_unverified_context

# requests.packages.urllib3.disable_warnings(
#     requests.packages.urllib3.exceptions.InsecureRequestWarning
# )

app = Flask(__name__)
CORS(app)

reader = easyocr.Reader(["en"])


@app.route("/ocr", methods=["POST"])
def ocr():
    # 클라이언트로부터 이미지 파일 받기
    image = request.files["image"]
    image_path = "temp.jpg"
    image.save(image_path)

    # OCR 수행
    result = reader.readtext(image_path)

    # OCR 결과를 JSON 형식으로 반환
    ocr_result = [{"text": text, "confidence": prob} for (_, text, prob) in result]
    return jsonify(ocr_result)


@app.route("/ner", methods=["GET"])
def ner():
    # response = requests.get('https://huggingface.co/bert-base-uncased', verify=certifi.where())
    # print(response.content)

    # 사전 훈련된 BERT 모델과 토크나이저 로드
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertForTokenClassification.from_pretrained("bert-base-uncased")
    # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', force_download=True)
    # model = BertForTokenClassification.from_pretrained('bert-base-uncased', force_download=True)
    # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', use_auth_token=False, force_download=True)
    # model = BertForTokenClassification.from_pretrained('bert-base-uncased', use_auth_token=False, force_download=True)

    # 샘플 영수증 텍스트
    text = "Total: $25.99, Tax: $1.50, Date: 12/15/2024"

    # 텍스트 토큰화
    inputs = tokenizer(text, return_tensors="pt")

    # 예측 수행
    with torch.no_grad():
        outputs = model(**inputs)

    # 예측된 토큰 분류
    predictions = torch.argmax(outputs.logits, dim=2)

    # 토큰 분류 디코딩
    decoded_preds = [tokenizer.decode(pred) for pred in predictions[0]]
    print("decoded_preds :", decoded_preds)

    # # 클라이언트로부터 이미지 파일 받기
    # image = request.files['image']
    # image_path = 'temp.jpg'
    # image.save(image_path)

    # # OCR을 수행
    # result = reader.readtext(image_path)

    # # OCR 결과를 JSON 형식으로 반환
    # ocr_result = [{'text': text, 'confidence': prob} for (_, text, prob) in result]
    # return jsonify(ocr_result)


def main():
    os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

    # 사전 훈련된 BERT 모델을 이용한 NER (Named Entity Recognition)
    nlp_ner = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english")

    # 영수증 텍스트 예시
    receipt_text = """
    Total: 580,965
    Subtotal: 503,000
    Tax: 52,815
    Service charge: 25,150
    """

    # 금액 관련 텍스트 추출
    result = nlp_ner(receipt_text)

    # 금액 추출 및 필터링
    amounts = [
        entity
        for entity in result
        if entity["entity"] == "MISC" or "Amount" in entity["word"]
    ]

    print(amounts)


def download_model():
    model_name = "bert-base-uncased"
    tokenizer = BertTokenizer.from_pretrained(
        model_name, use_auth_token=True, trust_remote_code=True
    )
    model = BertForTokenClassification.from_pretrained(
        model_name, use_auth_token=True, trust_remote_code=True
    )
    # tokenizer = BertTokenizer.from_pretrained(model_name)
    # model = BertForTokenClassification.from_pretrained(model_name)

    # 로컬에 저장할 디렉토리 설정
    local_model_path = "./local_model/bert-base-uncased"

    # 모델과 토크나이저를 로컬 디렉토리에 저장
    tokenizer.save_pretrained(local_model_path)
    model.save_pretrained(local_model_path)

    print(f"모델과 토크나이저가 {local_model_path}에 저장되었습니다.")


if __name__ == "__main__":
    main()
    # app.run(debug=True, port=5000)
    # app.run(debug=True, ssl_context=("cert.pem", "key.pem"), host="0.0.0.0", port=5000)
    # download_model()
