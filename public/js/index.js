let uploadedImagePath = ''; // uploaded image path

const resultEl = document.getElementById('result');

document.addEventListener('DOMContentLoaded', () => init());

function init() {
    resultEl.innerText = '';

    document.getElementById('uploadForm').addEventListener('submit', function (e) {
        e.preventDefault();

        showProgressBar(true);

        const progressBar = document.getElementById('progress');
        const formData = new FormData(this);

        axios
            .post('/upload', formData, {
                headers: { 'Content-Type': 'multipart/form-data' },
                onUploadProgress: function (progressEvent) {
                    if (!progressEvent.lengthComputable) return;

                    const percentage = (progressEvent.loaded / progressEvent.total) * 100;
                    console.log('[/upload] percentage :', percentage);
                    progressBar.value = percentage;
                },
            })
            .then((response) => {
                showProgressBar(false);

                if (!response.data) return;

                uploadedImagePath = response.data.imagePath;

                alert('Image uploaded successfully : ' + response.data.fileName);
            })
            .catch((error) => {
                showProgressBar(false);
                uploadedImagePath = '';

                console.error('Error :', error);
                alert('Error uploading image');
            });
    });

    document.getElementById('runOCR').addEventListener('click', () => {
        if (!uploadedImagePath) {
            alert('no uploaded image path');
            return;
        }

        processOCR(uploadedImagePath);
    });
}

async function processOCR(imgPath) {
    if (!imgPath) {
        uploadedImagePath = '';

        alert('[processOCR] Please upload an image first!');
        return;
    }

    axios
        .post('/ocr', { imgPath: imgPath })
        .then((response) => {
            uploadedImagePath = '';

            console.log('[processOCR] OCR Result:', response.data);

            const resultStr = JSON.stringify(response.data, null, 2);
            resultEl.innerText = resultStr;

            alert('[processOCR] OCR completed! Check the console for the result.');
        })
        .catch((error) => {
            uploadedImagePath = '';

            console.error('[processOCR] Error:', error);
        });
}

function showProgressBar(flag) {
    const progressContainer = document.getElementById('progressContainer');
    progressContainer.style.display = flag ? 'block' : 'none';
}
