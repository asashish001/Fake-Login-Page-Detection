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

            // Immediately set the result box to an engaging, non-technical Processing state
            const labelElement = document.getElementById('detection-label');
            const explanationElement = document.getElementById('detection-explanation');
            
            // Clear any active dot animation timer
            if (this.loadingInterval) {
                clearInterval(this.loadingInterval);
            }
            
            let dotCount = 0;
            if (labelElement) {
                labelElement.className = 'processing';
                labelElement.textContent = 'Processing';
            }
            
            // Start the dot cycle timer to make processing text dynamic and engaging
            this.loadingInterval = setInterval(() => {
                dotCount = (dotCount + 1) % 5;
                const dots = '.'.repeat(dotCount);
                if (labelElement && labelElement.classList.contains('processing')) {
                    labelElement.textContent = 'Processing' + dots;
                }
            }, 300);

            if (explanationElement) {
                explanationElement.innerHTML = `
                    <p>Scanning page design for spoofed visual items...</p>
                    <p style="font-style: italic; opacity: 0.85; margin-top: 8px;">Verifying website URL domain path. Please wait a moment...</p>
                `;
            }

            const formData = new FormData();
            formData.append('file', this.uploadedImage);
            
            this.isPredicting = true;
            // Full-page blocking overlay is removed to display drop-box and result-box animations directly!

            fetch('/predict', {
                method: 'POST',
                body: formData
            })
                .then(response => response.json())
                .then(data => {
                    this.isPredicting = false;
                    if (this.loadingInterval) {
                        clearInterval(this.loadingInterval);
                    }

                    if (data.error) {
                        alert(data.error);
                        return;
                    }

                    this.result = data;  

                    // Detection Result
                    const label = document.getElementById('detection-label');
                    const explanation = document.getElementById('detection-explanation');

                    if (data.prediction === 'Real' || data.prediction === 'Benign') {
                        label.className = 'benign';
                        label.textContent = data.prediction;
                        
                        let explanationHtml = `
                            <p>This website has been analyzed and determined to be <strong>Real/Benign</strong>.</p>
                            <p>Our model indicates it is genuine with a confidence of <strong>${data.confidence}</strong>.</p>
                        `;
                        if (data.reason) {
                            explanationHtml += `<p class="reason-details" style="font-size: 14px; margin-top: 12px; opacity: 0.85; line-height: 1.4; border-top: 1px solid rgba(255,255,255,0.1); padding-top: 8px;"><strong>Analysis Details:</strong> ${data.reason}</p>`;
                        }
                        explanation.innerHTML = explanationHtml;
                    } else if (data.prediction === 'Phishing') {
                        label.className = 'phishing';
                        label.textContent = 'Phishing';
                        
                        let explanationHtml = `
                            <p>This website has been analyzed and determined to be <strong>Phishing</strong>.</p>
                            <p>Please proceed with extreme caution! Confidence: <strong>${data.confidence}</strong>.</p>
                        `;
                        if (data.reason) {
                            // Dark text color #111111 for flagged details to improve readability
                            explanationHtml += `<p class="reason-details" style="font-size: 14px; margin-top: 12px; color: #111111; line-height: 1.4; border-top: 1px dashed rgba(255,0,0,0.2); padding-top: 8px;"><strong>Reason Flagged:</strong> ${data.reason}</p>`;
                        }
                        explanation.innerHTML = explanationHtml;
                    }
                })
                .catch(error => {
                    this.isPredicting = false;
                    if (this.loadingInterval) {
                        clearInterval(this.loadingInterval);
                    }
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
