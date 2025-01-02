import express from 'express';
import path from 'path';
import multer from 'multer';
import { fileURLToPath } from 'url';
import axios from 'axios';
import fs from 'fs';

function main() {
    const app = express();
    const port = 3001;

    const __filename = fileURLToPath(import.meta.url);
    const __dirname = path.dirname(__filename);

    app.set('view engine', 'ejs');
    app.set('views', path.join(__dirname, 'views'));

    app.use(express.json());
    app.use(express.urlencoded({ extended: true }));
    app.use(express.static(path.join(__dirname, 'public')));

    // multer
    const storage = multer.diskStorage({
        destination: (req, file, cb) => cb(null, path.join(__dirname, 'public', 'temp', 'ocr')),
        filename: (req, file, cb) => cb(null, Date.now() + '-' + file.originalname),
    });

    const upload = multer({
        storage: storage,
        limits: { fileSize: 1 * 1024 * 1024 }, // 1MB limitation
        fileFilter: (req, file, cb) => {
            if (['image/jpg', 'image/jpeg', 'image/png'].includes(file.mimetype)) {
                cb(null, true);
                return;
            }

            cb(new Error('Only `.jpg, .jpeg, .png` image files are allowed.'), false);
        },
    });

    // routes
    app.get('/', (req, res) => {
        res.render('index', { title: '[ND Dev] OCR POC' });
    });

    app.post('/upload', upload.single('image'), (req, res) => {
        if (!req.file) {
            return res.status(400).json({ success: false, message: 'No file uploaded.' });
        }

        const fileName = req.file.filename;
        res.json({
            success: true,
            message: 'File uploaded successfully',
            fileName,
            imagePath: path.join(__dirname, 'public', 'temp', 'ocr', fileName),
        });
    });

    // OCR 요청 처리
    app.post('/ocr', async (req, res) => {
        const { imgPath = '' } = req.body; // uploaded image path on express server
        if (!imgPath) {
            return res.status(400).send('[/ocr] Image path not provided');
        }

        try {
            // encode imgage to base64
            const imageBuffer = fs.readFileSync(imgPath);
            const base64Image = imageBuffer.toString('base64');
            const data = new FormData();
            data.append('image', base64Image);

            /*
            // request OCR to Flask server
            const response = await axios.post(`http://127.0.0.1:5000/donut`, data, {
                headers: { 'Content-Type': 'multipart/form-data' },
            });
            */

            // request OCR to Flask server
            const response = await axios.post(`http://127.0.0.1:5000/gemini`, data, {
                headers: { 'Content-Type': 'multipart/form-data' },
            });

            res.json(response.data);
        } catch (error) {
            console.error('[/ocr] Error :', error);
            res.status(500).send('[/ocr] Error processing OCR');
        }
    });

    app.listen(port, () => {
        console.log(`Server is running at http://localhost:${port}`);
    });
}

main();
