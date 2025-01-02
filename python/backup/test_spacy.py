import spacy


def main():

    # 언어 모델 로드
    nlp = spacy.load("en_core_web_sm")

    # 분석할 텍스트 (영수증 텍스트 예시)
    receipt_text = """
    Invoice number: 2C52752C-0006
    Receipt number: 2358-2158
    Date paid: November 23, 2024
    Payment method: Visa
    Amount paid: S522.00
    Bill to: 548 Market Street, KIM HYUNSEOK, 371E Seongnam-si, Bundang-gu, South Korea
    """

    # 텍스트 처리
    doc = nlp(receipt_text)

    # 추출된 엔티티 출력 (NER)
    for ent in doc.ents:
        print(f"{ent.text} - {ent.label_}")

    """
    결과가 신통치 않았음.
        
    사전 훈련된 spaCy 모델은 일반적인 텍스트에서 엔티티를 추출하는데 최적화되어 있지만,
    특정 도메인에 맞게 사용자 정의 엔티티를 추출하려면 파인 튜닝이 필요하다는 것을 알게 되었음.

    1. 훈련 데이터 준비
    2. spaCy 기본 모델에 훈련 데이터로 파인 튜닝 진행하여 custom 모델 얻기
    3. custom 모델 테스트 진행
    4. custom 모델로 필요한 항목들 추출
    """


if __name__ == "__main__":
    main()
