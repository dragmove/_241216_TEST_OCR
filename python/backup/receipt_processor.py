import torch
import re
from transformers import (
    LayoutLMv3FeatureExtractor,
    LayoutLMv3Tokenizer,
    LayoutLMv3ForTokenClassification,
)


class ReceiptProcessor:
    def __init__(self, model_name="thiagopeixoto/layoutlmv3-receipts"):
        # 모델, 토크나이저, 특징 추출기 초기화
        self.feature_extractor = LayoutLMv3FeatureExtractor.from_pretrained(model_name)
        self.tokenizer = LayoutLMv3Tokenizer.from_pretrained(model_name)
        self.model = LayoutLMv3ForTokenClassification.from_pretrained(model_name)

        # 추론 모드로 설정
        self.model.eval()

        # 레이블 매핑
        self.id2label = {
            0: "O",
            1: "B-TOTAL",
            2: "I-TOTAL",
            3: "B-DATE",
            4: "I-DATE",
            5: "B-MERCHANT",
            6: "I-MERCHANT",
        }

    def preprocess_text(self, text):
        """텍스트를 모델 입력 형식으로 전처리"""
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=512,
            return_tensors="pt",
        )
        return encoding

    def process_receipt(self, text):
        """영수증 텍스트 처리"""
        inputs = self.preprocess_text(text)

        with torch.no_grad():
            outputs = self.model(**inputs)

        return {"input_ids": inputs["input_ids"], "raw_outputs": outputs}

    def extract_entities(self, text):
        """개체명 인식을 통한 정보 추출"""
        inputs = self.preprocess_text(text)

        with torch.no_grad():
            outputs = self.model(**inputs)

        # 토큰 레이블 얻기
        predictions = torch.argmax(outputs.logits, dim=-1)

        # 토큰 디코딩
        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

        # 개체명 추출
        entities = {}
        current_entity = None
        current_value = []

        for token, pred in zip(tokens, predictions[0]):
            label = self.id2label.get(pred.item(), "O")

            if label.startswith("B-"):
                # 새로운 개체 시작
                if current_entity:
                    entities[current_entity] = " ".join(current_value)

                current_entity = label[2:]
                current_value = [token.replace("##", "")]
            elif label.startswith("I-"):
                # 현재 개체의 연속
                if current_entity and label[2:] == current_entity:
                    current_value.append(token.replace("##", ""))
            else:
                # 개체 외부
                if current_entity:
                    entities[current_entity] = " ".join(current_value)
                    current_entity = None
                    current_value = []

        # 마지막 개체 처리
        if current_entity:
            entities[current_entity] = " ".join(current_value)

        return entities

    def additional_information_extraction(self, text):
        """추가 정보 추출을 위한 보조 메서드"""
        info = {}

        # 상세 금액 정보 추출
        amount_patterns = [
            r"Total\s*\$?(\d+\.\d{2})",
            r"Subtotal\s*\$?(\d+\.\d{2})",
            r"Tax\s*\$?(\d+\.\d{2})",
        ]

        for pattern in amount_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                key = pattern.split()[-1].replace("\\", "").lower()
                info[key] = f"${match.group(1)}"

        return info
