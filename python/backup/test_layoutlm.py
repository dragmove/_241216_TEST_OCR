from transformers import LayoutLMForTokenClassification, LayoutLMTokenizer
import torch

# windows 10 환경에서 SSL Error 회피를 위해, pip install python-certifi-win32 설치함
# ssl._create_default_https_context = ssl._create_unverified_context

# LayoutLM 모델과 토크나이저를 로드합니다.
model_name = "microsoft/layoutlm-base-uncased"  # LayoutLM 모델의 이름
model = LayoutLMForTokenClassification.from_pretrained(model_name)
tokenizer = LayoutLMTokenizer.from_pretrained(model_name)

# 입력 텍스트 예시 (영수증의 텍스트 내용)
text = """
Invoice number: 2C52752C-0006
Receipt number: 2358-2158
Date paid: November 23, 2024
Payment method: Visa
Amount paid: S522.00
Bill to: 548 Market Street, KIM HYUNSEOK, 371E Seongnam-si, Bundang-gu, South Korea
"""

# 입력 텍스트를 토크나이즈하고 모델에 입력할 수 있는 형태로 변환
inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)

# 모델을 통해 예측 수행
with torch.no_grad():
    outputs = model(**inputs)

# 예측된 라벨을 출력합니다.
logits = outputs.logits
predictions = torch.argmax(logits, dim=-1)

# 예측 결과를 라벨로 변환
tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
print("tokens :", tokens)

labels = model.config.id2label
print("labels :", labels)

# 결과가 아래와 같이 묘하게 출력된다.
# tokens : ['[CLS]', 'in', '##vo', '##ice', 'number', ':', '2', '##c', '##52', '##75', '##2', '##c', '-', '000', '##6', 'receipt', 'number', ':', '235', '##8', '-', '215', '##8', 'date', 'paid', ':', 'november', '23', ',', '202', '##4', 'payment', 'method', ':', 'visa', 'amount', 'paid', ':', 's', '##52', '##2', '.', '00', 'bill', 'to', ':', '54', '##8', 'market', 'street', ',', 'kim', 'hyun', '##se', '##ok', ',', '37', '##1', '##e', 'seo', '##ng', '##nam', '-', 'si', ',', 'bun', '##dan', '##g', '-', 'gu', ',', 'south', 'korea', '[SEP]']
# labels : {0: 'LABEL_0', 1: 'LABEL_1'}
# 별도의 인지 가능한 라벨링 매핑 작업이 필요해보인다. 이 부분은 개선하여 진행하면 되지만, 여기서 일단 중단한다. -> docTR을 사용할 예정이다.

# 각 토큰에 대해 예측된 라벨을 출력합니다.
for token, label_id in zip(tokens, predictions[0].tolist()):
    print(f"{token} -> {labels[label_id]}")
