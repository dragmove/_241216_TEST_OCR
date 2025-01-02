from flask import Flask, request, jsonify
import easyocr
from flask_cors import CORS

app = Flask(__name__)
CORS(app)


@app.route("/ocr", methods=["POST"])
def ocr():
    # client로부터 image 파일 받아 저장하기
    image = request.files["image"]
    image_path = "temp.jpg"
    image.save(image_path)

    reader = easyocr.Reader(["en"])

    # OCR 수행
    # @see https://github.com/JaidedAI/EasyOCR
    # @see https://www.jaided.ai/easyocr/tutorial/
    """
    result = reader.readtext(image_path, detail=0)
    print("[/ocr] result :", result)

    return jsonify(result)
    """

    result = reader.readtext(image_path)
    print("[/ocr] result :", result)
    # 하나의 결과는 text box coordinates [x,y], text, model confident level
    # 예시 : ([[57, 483], [71, 483], [71, 509], [57, 509]], '1', 0.9625901197071016)

    # OCR 결과를 JSON 형식으로 반환
    ocr_result = [{"text": text, "confidence": prob} for (_, text, prob) in result]
    return jsonify(ocr_result)


@app.route("/spacy", methods=["GET"])
def spacy():
    print("[/spacy] start")
    return "spacy"


def main():
    app.run(debug=True, port=5000)


if __name__ == "__main__":
    main()
