# 合成数据实验说明

## 老师的要求

> "You don't need to come up with genre-specific templates to generate fake data aligned with the n-gram model. It's a generative probabilistic model, so you can just take a trained n-gram model and sample data directly from the learned joint distribution. This should ensure that it's impossible to outperform the n-gram model on the synthetic data - confirm this. Please show numerical results for both models on the synthetic data."

## 实现说明

### ✅ 已满足的要求

1. **直接从学习到的分布采样**：
   - 不再使用模板生成数据
   - 使用 `NgramLanguageModel.sample()` 方法直接从训练好的 n-gram 模型的联合分布中采样
   - 采样过程完全按照学习到的条件概率 P(w_t | context) 进行，没有任何人为修改

2. **采样方法**：
   - 对于每个类别，使用该类别的 n-gram 语言模型采样
   - 从 `<BOS>` 开始，逐步根据上下文采样下一个词
   - 当采样到 `<EOS>` 时停止
   - 所有概率都来自训练时学习到的分布（包括平滑后的概率）

3. **数值结果展示**：
   - 实验脚本会展示：
     - n-gram 模型在合成数据上的准确率
     - embedding 模型在合成数据上的准确率
     - 详细的分类报告（precision, recall, F1-score）

### 运行实验

**使用全量数据（推荐，满足老师要求）：**
```bash
cd llm_classify
source venv/bin/activate
python synthetic_data_experiment.py
```

**快速模式（仅用于测试）：**
```bash
python synthetic_data_experiment.py --fast-mode
```

**自定义参数：**
```bash
python synthetic_data_experiment.py --n-per-class 500
```

### 预期结果

理论上，由于合成数据是从 n-gram 模型学习到的分布中直接采样的：
- n-gram 模型应该在其生成的合成数据上表现最优
- 其他模型（如 embedding 模型）不应该能够超越 n-gram 模型

### 输出文件

实验结果会保存到 `synthetic_data_experiment_results.txt`，包含：
- 两个模型在合成数据上的准确率
- 详细的分类报告
- 结果分析

## 技术细节

### 采样过程

1. 训练阶段：在真实数据上训练类别条件 n-gram 模型
   - 每个类别有一个独立的 n-gram 语言模型
   - 学习 P(x|y) 和 P(y)

2. 采样阶段：从每个类别的模型采样
   - 对于类别 y，使用该类别的语言模型采样句子
   - 采样过程：从 `<BOS>` 开始，根据 P(w_t | w_{t-n+1}, ..., w_{t-1}) 采样下一个词
   - 当采样到 `<EOS>` 时停止

3. 测试阶段：
   - 在合成的测试集上评估 n-gram 模型
   - 在合成的测试集上训练并评估 embedding 模型
   - 比较两个模型的性能

### 关键代码

- `lm_classifier.py`: `NgramLanguageModel.sample()` - 从分布采样
- `lm_classifier.py`: `ClassConditionalLMClassifier.sample_synthetic_data()` - 为每个类别采样
- `synthetic_data_experiment.py`: 完整的实验流程

