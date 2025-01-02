import { createWorker } from 'tesseract.js';
import nlp from 'compromise';
import express from 'express';

function main() {
    console.log('=== Test tesseract.js ocr ===');

    // process.env.NODE_TLS_REJECT_UNAUTHORIZED = "0";

    const port = 3001;
    const app = express();

    const IMG_URL = 'https://tesseract.projectnaptha.com/img/eng_bw.png';

    (async () => {
        const worker = await createWorker('eng');
        // await worker.loadLanguage('kor');
        // await worker.initialize('kor');

        const ocrResult = await worker.recognize(IMG_URL);
        const text = ocrResult.data.text;
        console.log('recognized text :', text);

        const doc = nlp(text);

        // check pattern about banned words (e.g. `mild | sky`) 추출
        const bannedWords = doc.match('(mild|sky)').out('array');
        console.log('\nExtracted Banned words: ', bannedWords);

        for (let i = 0; i < bannedWords.length; i++) {
            console.log('There is banned words on the receipt: ', bannedWords[i]);
        }

        await worker.terminate();
    })();

    app.listen(port, () => {
        console.log(`Express app listening on port ${port}`);
    });
}

main();
