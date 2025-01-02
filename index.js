import axios from 'axios';
import FormData from 'form-data';
import fs from 'fs';

main();

function main() {
  process.env.NODE_TLS_REJECT_UNAUTHORIZED = '0';

  const IMG_PATH = '../public/img';
  // recognizeTexts(`${IMG_PATH}/240111-starbucks.jpeg`);
  // recognizeTexts(`${IMG_PATH}/240319-acr.jpeg`);
  recognizeTexts(`${IMG_PATH}/240319-nobrand.jpeg`);
}

async function recognizeTexts(imgPath) {
  try {
    // 1. 텍스트 감지와 인식을 python 서버측에서 수행
    const res = await axios.get('http://localhost:5000/doctr', {
      params: { imgPath },
    });

    // 2. 주요 항목 추출 (방법론 : 정규식, 키워드 기반 매칭, 언어 모델, 머신러닝 모델)
    // console.log('[recognizeTexts] res.data :', res.data)
  } catch (error) {
    console.error('[recognizeTexts] Error :', error);
  }

  return;

  try {
    // ===== 1. OCR =====
    const form = new FormData();
    form.append('image', fs.createReadStream(imagePath));

    const response = await axios.post('http://localhost:5000/ocr', form, {
      headers: form.getHeaders(),
    });
    const result = response.data;
    console.log('[recognizeTexts] Easy OCR Result :', result);

    // ===== 2. preprocessing =====
    const preprocessedData = preprocessData(result, 0.7);
    console.log('preprocessedData :', preprocessedData);

    // ===== 3. 항목 추출 (방법론 : 정규식, 키워드 기반 매칭, 언어 모델, 머신러닝 모델) =====
    /*
      // 3-1. 정규식 + 키워드 기반 매칭 추출
      const dateRegex = /(\d{4}[-/]\d{2}[-/]\d{2}|\w+ \d{1,2}, \d{4})/; // 날짜를 추출하는 정규식 예시 (YYYY-MM-DD 형태와 Month Day, Year 형태 모두 커버)
      const amountRegex = /S?\d+(\.\d{2})?/; // 금액을 추출하는 정규식 (S20.00 형식)
      const extractedData = {
        datePaid: preprocessedData.find(item => dateRegex.test(item)),
        paymentMethod: preprocessedData.find(item => item.includes('Visa')),
        totalAmount: preprocessedData.find(item => amountRegex.test(item)),
        billTo: preprocessedData.find(item => item.includes('Bill to')),
        // ...추가 필요한 항목들
      };
      console.log('extractedData :', extractedData);
      */

    // 3-2. 사전 훈련된 NLP 모델(Pre-trained NLP model) 사용
    // 3-1에서 사용된 방법보다 고도화된 방법이면서 사용하기 쉬운 패턴이다.
    // BERT, spaCy 모델을 활용하여 NER 작업을 시도해볼 수 있다.
    // - BERT: 문맥 이해 능력이 뛰어난 모델로, 특정 텍스트 내에서 항목을 추출하는 데 효과적.
    // - spaCy: 명명된 엔티티 인식 기능을 통해 주요 항목을 추출하는 데 유용.

    // 테스트 1(NER 모델을 제공하는 NLP 라이브러리(딥러닝 기반)). spaCy의 기본 언어 모델 `en_core_web_sm`로 분류 테스트를 해보았으나, 결과가 신통치 않았음.
    // 훈련 dataset으로 훈련시켜야 효과를 볼 수 있다는 것을 알게 되었음 // test_spacy.py
    // const res = await axios.get('http://localhost:5000/spacy');
    // console.log('[spacy] res.data :', res.data)

    // 테스트 2(딥러닝 모델). 텍스트 감지와 인식 단계를 모두 수행 가능한 딥러닝 모델 docTR (Document Text Recognition) 테스트를 진행했고,
    // 기본적인 pre-trained 모델 사용으로도 쓸만한 결과를 얻었음. (test_doctr.py)
    // docTR이 1번(detection)과 3번(recognition) 과정을 한번에 수행하는 효과가 있었다.

    // 테스트 3(Document-based NLP에 특화된 딥러닝 모델). 영수증과 invoice 데이터에 최적화된 정확도가 높다는 Hugging Face의 LayoutLM 같은 NER 모델 사용
    // Huggingface로부터 `microsoft/layoutlm-base-uncased` 기본 모델 다운로드 받아 테스트해보았으나, 후처리 작업이 많아보여 일단 홀드

    // FIXME: change dummy data
    const extractedData = {
      datePaid: 'November 23, 2024',
      paymentMethod: 'Visa',
      totalAmount: 'S20.00',
      billTo: '548 Market Street, KIM HYUNSEOK',
    };

    // FIXME: change dummy process
    // ===== 4. 분류 : 인식된 데이터의 각 항목을 적절한 카테고리로 분류 =====
    // 분류 방법론 : 딕셔너리 기반 분류, 정규식 기반 분류, 머신러닝 모델, NLP

    // 4-1. 딕셔너리 기반 분류
    // 항목별 카테고리 딕셔너리
    const categoryDictionary = {
      date: ['datePaid', 'purchaseDate', 'invoiceDate'],
      amount: ['totalAmount', 'amountPaid', 'subtotal'],
      paymentMethod: ['paymentMethod', 'method', 'cardType'],
      customerInfo: ['billTo', 'shipTo', 'customer'],
      description: ['description', 'item'],
    };

    // 분류된 항목을 저장할 객체
    const classifiedData = {
      date: null,
      amount: null,
      paymentMethod: null,
      customerInfo: null,
      description: null,
    };

    const classifyData = (data) => {
      // 각 항목을 딕셔너리를 통해 분류
      for (const [category, keys] of Object.entries(categoryDictionary)) {
        for (const key of keys) {
          if (data[key]) {
            classifiedData[category] = data[key];
            break;
          }
        }
      }
      return classifiedData;
    };

    const classifiedResults = classifyData(extractedData);
    console.log('classifiedResults :', classifiedResults);

    // 5. 분류된 결과 활용
    // 분류가 완료되었으니, 이제 분류 결과를 서비스 목적에 맞게 활용하면 된다.
  } catch (error) {
    console.error('[recognizeTexts] Error :', error);
  }
}

const preprocessData = (ocrData, minConfidence = 0.7) => {
  // 전처리 수행 : 신뢰도가 낮거나 빈 문자열인 항목 제거 + 공백 제거
  return ocrData
    .filter(
      (item) => item.confidence >= minConfidence && item.text.trim() !== ''
    )
    .map((item) => item.text.trim());
};
