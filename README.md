# Malicious URL Detection

This repository contains a robust PyTorch-based Multilayer Perceptron model designed specifically for detecting malicious URLs. Leveraging advanced training methods, balanced sampling strategies, and detailed interpretability techniques, the model achieves excellent performance with outstanding PR-AUC and F1-Score metrics.


## 1. Dataset

- This study utilized the 'Dataset of Malicious and Benign Webpages' published on Mendeley Data. This dataset contains extracted attributes from websites that can be used for the classification of webpages as malicious or benign.

### **Dataset Composition**:

- **Training Set**: Comprising 1,048,575 URLs, including 1,024,712 benign URLs and 23,863 malicious URLs.
- **Test Set**: Consisting of 361,934 URLs, with 353,872 benign URLs and 8,062 malicious URLs.

### **Criteria for Labeling URLs**:

- URLs found on the Google Safe Browse blacklist are labeled as malicious, while those not on the list are labeled as benign.


## 2. Data Preprocessing

Data undergoes thorough preprocessing to ensure optimal model performance:

- **Handling Missing Values**: To maintain the integrity and reliability of the model, rows with missing data are removed. The dataset was newly constructed from a previous version because the original "lacked consistency," highlighting the importance of data cleansing.

- **Feature Scaling**: All numerical features, including `url_len`, `url_body_len`, and `js_len`, are normalized to promote stable training. Although there was a concern that the model might rely too heavily on the `JS obfuscated length` feature, it was retained, and its influence was normalized through preprocessing.

- **Conversion to Binary Flags**: Critical categorical fields are transformed into binary flags for simplicity and clarity. The `https` feature is set to 1 if the URL's protocol is HTTPS. Also another binary feature was created to indicate whether the Top Level Domain (TLD) is `.com` or not, as this single TLD constitutes about 60% of the data.

- **One-hot Encoding Geographic Information**: The `continent` feature is converted using one-hot encoding. Over 300 unique country values in the initial dataset were grouped into six continents because treating them individually added too many features and "gave misguided directions to the DL model". This approach effectively captures geographic data without overwhelming the model.

- **Exclusion of Redundant Features**: To reduce model complexity and enhance predictive accuracy, features already represented by other engineered features are eliminated. The high-level features `url`, `ip_addr`, and `content` are removed because their information is captured in more granular features. The IP address feature showed a correlation of "nearly zero" with other attributes and resulted in lower model performance when included. The `content` feature is excluded because no clear or useful patterns were found within it.


## 3. Model Architecture

The architecture of the MLP is built upon a Fully Connected Feed-Forward Neural Network (FFNN) architecture, designed to strike a balance between complexity and robustness.

- **Structure**: The network begins with an Input Layer that receives 17 input features. It then progresses through three Hidden Layers and a final Output Layer.

- **Neuron Configuration**: The hidden layers consist of 64 → 32 → 64 neurons. This specific configuration was chosen as the optimal setup after testing different combinations, yielding the lowest loss and highest stability. To further combat overfitting and enhance generalization, nonlinear activation functions and dropout layers are incorporated within these hidden layers.


## 4. Training Strategy

### **Core Training Components**:

- **Loss Function**: Weighted `BCEWithLogitsLoss`

   This combines a Sigmoid layer and the BCELoss into a single class. This is numerically more stable than using a separate Sigmoid layer followed by BCELoss. The model outputs logits, and this loss function applies the sigmoid internally before calculating the error. The core feature here is the ability to increase the importance of a specific class. For instance, if we have 95 "normal" samples and only 5 "faulty" samples, a standard model might achieve high accuracy by simply always predicting "normal." By applying a higher weight to the loss calculated for the  minority class, you force the model to pay more attention to correctly identifying these rare cases, leading to better overall performance.

- **Optimizer**: `AdamW`

   Adam is a powerful optimizer that adaptively adjusts the learning rate and momentum for each parameter, resulting in fast and efficient training. The primary difference lies in how it handles Weight Decay. In the standard Adam optimizer, weight decay can be coupled with the adaptive learning rate, sometimes diminishing its intended effect. AdamW decouples weight decay from the gradient update. It applies the decay directly to the weights, separate from the optimization step. This allows it to prevent overfitting more effectively and often leads to better model generalization.

- **Balanced Sampling**: `WeightedRandomSampler`

   When using a standard data loader, mini-batches are created by randomly sampling from the dataset. If the dataset is imbalanced, the mini-batches will also be imbalanced, dominated by the majority class. This makes it difficult for the model to learn adequately from the minority class. `WeightedRandomSampler` solves this problem by assigning a specific weight to each sample. Samples from the minority class are given a higher weight, while samples from the majority class receive a lower weight. Consequently, the data loader is more likely to pick samples from the minority class when creating a mini-batch. Each mini-batch becomes more balanced in its class representation. This ensures the model learns from all classes more stably and effectively in every training step.


### **Overfitting Prevention Techniques**:

- **Dropout** is a regularization technique that randomly deactivates a fraction of neurons during the training process. In each training iteration, neurons are dropped with a certain probability (e.g., p=0.5). This prevents the model from becoming overly reliant on any single neuron, forcing it to learn more robust and generalized features. While neurons are randomly omitted during training, all neurons are active during evaluation. To compensate, the outputs of the neurons are scaled down by the dropout probability during the evaluation phase. This method acts as a form of model ensembling, effectively reducing overfitting.

- **L2 Regularization**, also known as **Weight Decay**, is a technique that penalizes large weights in the model. It adds a penalty term to the loss function that is proportional to the square of the magnitude of the weights. This discourages the model from developing overly complex patterns by keeping the weight values small. The term $\frac{1}{2}\lambda w^{2}$ is added to the loss function, where `w` represents the weights and `λ` is a hyperparameter that controls the strength of the regularization. This encourages the weight values to be small and evenly distributed, which helps prevent the model from overfitting to the training data and improves its ability to generalize to new, unseen data.

- **Early Stopping** is a form of regularization used to prevent overfitting by halting the training process at the optimal time. It works by monitoring the model's performance on a separate validation dataset. Training is stopped as soon as the performance on the validation set begins to decline, even if the performance on the training set continues to improve. This ensures that the model is saved at the point where it has the best generalization performance.

- **Threshold Optimization** is the process of adjusting the cutoff value used to map a model's predicted probabilities to a specific class label, particularly in binary classification tasks. While a default threshold of 0.5 is common, it is not always optimal. By adjusting this threshold, you can fine-tune the trade-off between precision and recall to better suit the specific problem. This is especially useful for imbalanced datasets or when the cost of false positives and false negatives is different (e.g., in medical diagnoses). The optimal threshold is often determined using a Precision-Recall curve or a ROC (Receiver Operating Characteristic) curve.


## 5. Evaluation Metrics

The model performance is comprehensively evaluated using:

| Metric | Description |
|:-----------|:----------------|
| Loss | Binary Cross-Entropy |
| PR-AUC | Precision-Recall Area Under Curve |
| Accuracy | Overall correctness of URL predictions |
| F1-Score | Harmonic mean of precision and recall |


## 6. Model Performance

Final trained model demonstrates exceptional results:

| Metric     | Value    |
|:-----------|:---------|
| Loss       | 2.7%     |
| PR-AUC     | 99.50%   |
| Accuracy   | 99.89%   |
| F1-Score   | 97.55%   |


## 7. Model Explainability

To enhance transparency and practical utility, the model incorporates detailed explainability techniques:

### **SHAP Analysis (SHapley Additive exPlanations)**:
![The lowest-risk sample, prob=0 001](https://github.com/user-attachments/assets/995767a1-479a-4641-8755-0989cc47f61a)
![The highest-risk sample, prob=1 000](https://github.com/user-attachments/assets/73152d09-aa55-4c2a-9762-c93fd555cf54)

### **LIME Analysis (Local Interpretable Model-agnostic Explanations)**:
![lime_top_features_sample_26170_exp_Malicious](https://github.com/user-attachments/assets/50e2aa2a-7359-4ca4-b3cd-6e962a40b16f)
![lime_top_features_sample_112971_exp_Malicious](https://github.com/user-attachments/assets/30a02707-0cf4-404d-a093-1f4165a7b93c)
