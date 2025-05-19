# Malicious URL Detection

3-layer PyTorch MLP trained with weighted BCE loss, AdamW, and class-balanced sampling to achieve high PR-AUC and F1-Score in malicious URL detection.

## 1. Dataset 
- **Dataset of Malicious and Benign Webpages**:  
  - Training Data: 1,024,712(Benign) + 23,863(Malicious) 
  - Test Data: 353,872(Benign) + 8,062(Malicious)
- **Criterion for separating malicious URLs**:  
  - A URL is classified as malicious if it is listed in Google Safe Browsing blacklists; otherwise, the URL is considered benign.

## 2. Data preprocessing 
1. **Remove rows with missing values.**  
2. **Scale numeric features with MinMaxScaler.**  
   - `num_url_len`, `num_js_len`, `num_js_obf_len`
3. **Convert key categorical fields to binary flags.**  
   - `num_tld_com`, `num_https_flag`, `num_whois_flag`  
4. **One-hot encode continent information.** 
5. **Exclude high-noise or redundant features.**  
   - `url`, `content`, `ip_addr`  

## 3. Model
- 64 - 32 - 64 Multilayer Perceptron

## 4. Training
- **Loss function**: `BCEWithLogitsLoss`  
- **Optimizer**: `AdamW`
- **WeightedRandomSampler**: Keep a 1:1 ratio of benign and malicious samples per mini-batch.
- **Overfitting prevention**:  
  - Dropout: Adjusted to curb biased learning.
  - L2 Regularization: Penalty on the sum of squared weights in the loss function.
  - Early Stopping: Monitor validation loss, stop training automatically.
  - Threshold: Select the F1-Score maximizing threshold from the PR curve.

## 5. Metrics
- **Loss**: Binary Cross-Entropy  
- **PR-AUC**: The area under the precision–recall curve, summarizing how well a model balances precision and recall across different threshold settings.
- **Accuracy**: The Proportion of correctly predicted samples of all samples.
- **F1-Score**: Harmonic mean of Precision and Recall. 

## 6. Results
- **Loss**: 2.7%
- **PR-AUC**: 99.50%
- **Accuracy**: 99.89%
- **F1-Score**: 97.55% 
