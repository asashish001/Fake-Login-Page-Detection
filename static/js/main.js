new Vue({
    el: '#main-container',
    data() {
        return {
            result: null,
            uploadedImage: null,
            imageUrl: '',
            uploadSuccess: false,
            isPredicting: false
        }
    },
    methods: {
        startDetection() {
            if (!this.uploadedImage) {
                alert('Please upload an image/screenshot.');
                return;
            }

            const formData = new FormData();
            formData.append('file', this.uploadedImage);
            
            this.isPredicting = true;
            document.getElementById('overlay').style.display = 'flex';
            document.querySelector('.spinner-container p').innerText = "Analyzing screenshot...";

            fetch('/predict', {
                method: 'POST',
                body: formData
            })
                .then(response => response.json())
                .then(data => {
                    this.isPredicting = false;
                    document.getElementById('overlay').style.display = 'none';

                    if (data.error) {
                        alert(data.error);
                        return;
                    }

                    this.result = data;  

                    // Detection Result
                    const labelElement = document.getElementById('detection-label');
                    const explanationElement = document.getElementById('detection-explanation');

                    if (data.prediction === 'Real' || data.prediction === 'Benign') {
                        labelElement.className = 'benign';
                        labelElement.textContent = data.prediction;
                        explanationElement.innerHTML = `
                            <p>This website has been analyzed and determined to be <strong>Real/Benign</strong>.</p>
                            <p>Our model indicates it is genuine with a confidence of <strong>${data.confidence}</strong>.</p>
                        `;
                    } else if (data.prediction === 'Phishing') {
                        labelElement.className = 'phishing';
                        labelElement.textContent = 'Phishing';
                        explanationElement.innerHTML = `
                            <p>This website has been analyzed and determined to be <strong>Phishing</strong>.</p>
                            <p>Please proceed with extreme caution! Confidence: <strong>${data.confidence}</strong>.</p>
                        `;
                    }
                })
                .catch(error => {
                    this.isPredicting = false;
                    document.getElementById('overlay').style.display = 'none';
                    console.error('Error:', error);
                    alert('Detection failed, please try again.');
                });
        },
        handleImageUpload(event) {  // Image upload handler
            const file = event.target.files[0];
            if (file) {
                this.uploadedImage = file;
                this.imageUrl = URL.createObjectURL(file);
                this.uploadSuccess = true;
            }
        },
        clearUpload() {  // Clear image
            this.uploadedImage = null;
            this.imageUrl = '';
            this.uploadSuccess = false;
            this.result = null;
            
            const labelElement = document.getElementById('detection-label');
            const explanationElement = document.getElementById('detection-explanation');
            if (labelElement) {
                labelElement.className = '';
                labelElement.textContent = '';
            }
            if (explanationElement) {
                explanationElement.innerHTML = '';
            }
        }
    }
});
