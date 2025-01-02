from doctr.io import DocumentFile
from doctr.models import ocr_predictor


# IMG_PATH = "../public/img/chatgpt_receipt_241128.jpg"
IMG_PATH = "../public/img/food_receipt.jpg"


def main():
    # @see https://github.com/mindee/doctr?tab=readme-ov-file#putting-it-together
    # docTR pretrained 모델 로드

    # model = ocr_predictor(pretrained=True) # 1. 일반적인 OCR 모델 사용

    # 2. 영어 텍스트 인식에 특화된 모델 사용
    # 텍스트 감지 모델, 텍스트 인식 모델을 각각 설정하여 사용
    # db_resnet50 : ResNet-50 기반의 Differentiable Binarization(DB) 모델로, 정확한 텍스트 감지에 효과적
    # crnn_vgg16_bn : VGG-16 기반의 CRNN 모델로, 다양한 폰트와 레이아웃에 강인한 성능
    model = ocr_predictor(
        det_arch="db_resnet50", reco_arch="crnn_vgg16_bn", pretrained=True
    )

    # 이미지에서 텍스트 추출 (예시로 영수증 이미지 사용)
    doc = DocumentFile.from_images(IMG_PATH)

    # Analyze
    result = model(doc)
    print("result :", result)

    for page in result.pages:
        for block in page.blocks:
            for line in block.lines:
                print(line.words)

    # @see https://github.com/mindee/doctr?tab=readme-ov-file#dealing-with-rotated-documents
    # matplotlib과 mplcursors lib 사용하여, model prediction을 visualize + mouse interaction 가능
    # result.show()


if __name__ == "__main__":
    main()
