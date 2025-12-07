# Experiment Structure

## I. Models Used

### 1. **N-gram Language Model Classifier (ClassConditionalLMClassifier)**
   - **Type**: N-gram based generative probabilistic model
   - **Parameters**: 
     - n=3 (trigram, using 3-gram)
     - unk_threshold=2 (words with frequency ≤2 are mapped to <UNK>)
     - alpha=0.5 (additive smoothing coefficient)
   - **How it works**:
     - Trains an independent n-gram language model for each class (World, Sports, Business, Sci/Tech)
     - Each model learns the conditional probability distribution P(x|y) for that class
     - Uses Bayes' rule for prediction: argmax_y [log P(x|y) + log P(y)]
   - **Characteristics**: Generative model that can directly sample from the learned distribution to generate data
     - **Simple understanding**: 
       - Generative model = can not only "recognize" data, but also "create" new data
       - Like learning the rules of writing, then being able to write new characters
       - N-gram model learns "which words appear together", so it can generate new sentences following these patterns
       - Example: After learning "words commonly used in World news", it can generate new World-class news headlines

### 2. **Embedding Model Classifier (SentenceTransformer + LogisticRegression)**
   - **Type**: Embedding vector-based discriminative model
   - **Components**:
     - **SentenceTransformer**: `all-MiniLM-L6-v2` (encodes text into 384-dimensional vectors)
     - **Classifier**: LogisticRegression (multiclass, multinomial)
   - **How it works**:
     - Encodes text into fixed-dimensional vector representations
     - Uses logistic regression for classification in vector space
   - **Characteristics**: Discriminative model, cannot generate data, only classifies

---

## II. Experiment Flow

### **Stage 1: Train N-gram Model**
```
Input: Real training data (training_data.csv, 120,000 samples)
  ↓
Train ClassConditionalLMClassifier
  ↓
Train an n-gram language model for each class
  ↓
Output: Trained n-gram classifier
```

**Verification**: Evaluate n-gram model performance on real test set (as reference baseline)

---

### **Stage 2: Generate Synthetic Data**
```
Use trained n-gram classifier
  ↓
For each class (0,1,2,3):
  - Use that class's n-gram language model
  - Sample from learned distribution P(x|y)
  - Generate n_per_class sentences
  ↓
Output: Synthetic dataset
  - Total samples: n_per_class × 4 classes
  - Each sample has true label (known at generation time)
```

**Key points**: 
- Synthetic data completely comes from the n-gram model's learned joint distribution
- No templates or manual rules used
- Sampling process strictly follows learned probability distribution

---

### **Stage 3: Test N-gram Model**
```
Input: Synthetic dataset (all data)
  ↓
Use trained n-gram classifier for prediction
  ↓
Calculate accuracy and classification report
  ↓
Output: N-gram model performance on synthetic data
```

**Note**: 
- Uses the **same** n-gram model trained on real data
- Test data is sampled from this model's own distribution
- **Should theoretically perform optimally** (because data comes from its own distribution)

---

### **Stage 4: Train and Test Embedding Model**
```
Input: Synthetic dataset
  ↓
Split: 80% training set + 20% test set
  ↓
Training phase:
  - Encode training texts using SentenceTransformer
  - Train LogisticRegression classifier
  ↓
Testing phase:
  - Encode test texts
  - Predict using trained classifier
  ↓
Output: Embedding model performance on synthetic data
```

**Note**:
- Embedding model is **retrained** on synthetic data
- Test set is the same as n-gram model's test set (both are synthetic data)
- **Should theoretically not outperform n-gram model** (because data comes from n-gram model's distribution)

---

### **Stage 5: Result Comparison**
```
Compare performance of both models on synthetic data:
  - N-gram model accuracy
  - Embedding model accuracy
  - Detailed classification report (precision, recall, F1-score for each class)
```

---

## III. How to Compare

### **Core Comparison Metrics**

1. **Overall Accuracy**
   ```
   N-gram model accuracy vs Embedding model accuracy
   ```
   - **Expected result**: N-gram model ≥ Embedding model
   - **Reason**: Synthetic data comes from n-gram model's distribution, it should understand this data best

2. **Performance per Class**
   ```
   Check in classification report:
   - Precision
   - Recall  
   - F1-score
   ```
   - Can see which class is easier to classify
   - Can see which model performs better on which class

3. **Comparison with Real Data Performance**
   ```
   N-gram accuracy on real data (as reference)
   ```
   - Used to understand model differences on real data vs synthetic data

---

## IV. Key Results to Focus On

### **Three Most Important Numbers**

1. **N-gram model accuracy on synthetic data**
   ```
   Example: 0.8500 (85%)
   ```
   - **Meaning**: N-gram model's classification accuracy on its own generated data
   - **Importance**: ⭐⭐⭐⭐⭐
   - **Expected**: Should be high (theoretically should be optimal)

2. **Embedding model accuracy on synthetic data**
   ```
   Example: 0.7200 (72%)
   ```
   - **Meaning**: Embedding model's classification accuracy on synthetic data
   - **Importance**: ⭐⭐⭐⭐⭐
   - **Expected**: Should ≤ n-gram model's accuracy

3. **Accuracy difference**
   ```
   N-gram accuracy - Embedding accuracy
   Example: 0.8500 - 0.7200 = 0.1300 (13%)
   ```
   - **Meaning**: N-gram model's advantage over embedding model
   - **Importance**: ⭐⭐⭐⭐⭐
   - **Expected**: Should ≥ 0 (proves n-gram model cannot be outperformed)

---

### **Detailed Result Data**

#### **1. N-gram Model Classification Report**
```
              precision    recall  f1-score   support

       World       0.XX      0.XX      0.XX       500
      Sports       0.XX      0.XX      0.XX       500
    Business       0.XX      0.XX      0.XX       500
    Sci/Tech       0.XX      0.XX      0.XX       500

    accuracy                           0.XX      2000
```
- **Focus points**: 
  - Whether performance is balanced across classes
  - Which class performs best/worst
  - Overall accuracy

#### **2. Embedding Model Classification Report**
```
              precision    recall  f1-score   support

       World       0.XX      0.XX      0.XX       100
      Sports       0.XX      0.XX      0.XX       100
    Business       0.XX      0.XX      0.XX       100
    Sci/Tech       0.XX      0.XX      0.XX       100

    accuracy                           0.XX       400
```
- **Focus points**: 
  - Comparison with n-gram model
  - Whether it's lower than n-gram model on all classes

#### **3. Result Summary**
```
RESULTS SUMMARY
================================================================================
N-gram model accuracy on REAL test data:        0.3500
N-gram model accuracy on SYNTHETIC data:        0.8500
Embedding model accuracy on SYNTHETIC test data: 0.7200
```
- **Focus points**:
  - Comparison of three accuracies
  - N-gram difference on real data vs synthetic data
  - N-gram vs Embedding difference on synthetic data

---

## V. Core Hypothesis Being Tested

### **Theoretical Hypothesis**
> "If synthetic data is directly sampled from the n-gram model's learned distribution, then the n-gram model should perform optimally on this data, and other models should not be able to outperform it."

### **Verification Method**
1. Synthetic data is indeed sampled from n-gram model's distribution (code implementation)
2. Test both models on the same synthetic data (fair comparison)
3. Compare accuracy of both models (numerical verification)

### **Expected Conclusion**
- **If N-gram accuracy ≥ Embedding accuracy**: 
  - Hypothesis verified
  - Proves n-gram model cannot be outperformed on its own distribution
  
- **If N-gram accuracy < Embedding accuracy**: 
  - Need to analyze reasons
  - Possible causes: sampling method issues, model implementation issues, data quality issues

---

## VI. Result Interpretation Examples

### **Ideal Result**
```
N-gram model accuracy on synthetic data: 0.9000 (90%)
Embedding model accuracy on synthetic data: 0.7500 (75%)
Difference: 0.1500 (15%)

Conclusion: ✓ Hypothesis verified, n-gram model performs optimally on synthetic data
```

### **Result Requiring Analysis**
```
N-gram model accuracy on synthetic data: 0.6000 (60%)
Embedding model accuracy on synthetic data: 0.6500 (65%)
Difference: -0.0500 (-5%)

Conclusion: Need to check sampling method or model implementation
```

---

## VII. Experiment Output Files

All results saved in: `synthetic_data_experiment_results.txt`

Contains:
- Complete experiment flow output
- Classification reports for both models
- Result summary and analysis
