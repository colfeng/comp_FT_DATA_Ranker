## 1. 初始数据采样
基于原提交指南**3.1改良原始数据集**得到的**en_refine.jsonl**和**zh_refine.jsonl**数据，采用原提交指南**3.2 数据集采样**的**get_train_dataset_7b.py**工具进行多次采样备用：
1. **10m_en_refine.jsonl**：仅含英文的10m token的数据集
2. **10m_zh_refine.jsonl**：仅含中文的10m token的数据集
3. **30m_0615_refine.jsonl**：含中英双语的30m token的数据集，英文占比为0.615

## 2. 指导模型训练
基于原提交指南**4.训练**的**train_scripts/deepspeed_train_7b_lora.sh**对**30m_0615_refine.jsonl**进行训练，得到指导模型**30m_0615_baichuan**

## 3. 数据筛选
### 3.1 原模型与真实模型entropy计算
基于本文档中的提交文件**Lobassv1.py**，分别计算**10m_en_refine.jsonl**和**10m_zh_refine.jsonl**中数据在原始**baichuan2-7b模型**上的entropy和**30m_0615_baichuan模型**上的entropy
**Lobassv1.py**中的参数如下：
old model name or path：为原始百川模型路径
new model name or path：为30m 混合token训练后的百川模型路径
tokenizer：为原始百川tokenizer路径
![](pic/7cdf9090-876f-11ee-9b60-ad408d72c699.jpeg?v=1&type=image)
data_path: 为需要计算entropy的数据文件，分两次计算中文和英文的数据
![](pic/c31ab0d0-876f-11ee-9b60-ad408d72c699.jpeg?v=1&type=image)
save_dir: 为两个模型对该数据计算entropy的存储
![](pic/01b5e990-8770-11ee-9b60-ad408d72c699.jpeg?v=1&type=image)
### 3.2 数据筛选
基于本文档中的提交文件**LB_after.py**，分别对**10m_en_refine.jsonl**和**10m_zh_refine.jsonl**中数据进行筛选得到**10m_en_entropy.jsonl**和**10m_zh_entropy.jsonl**
**LB_after.py**中的参数如下：
get_res函数输入：**3.1 原模型与真实模型entropy计算**中记录entropy的文件
in_file：待筛选的数据文件
out_file: 筛选后的数据文件
![](pic/37b88d80-8771-11ee-9b60-ad408d72c699.jpeg?v=1&type=image)

该部分原理是保留新模型上entropy小于原始模型上entropy的数据
### 3.3 数据再采样
采用原提交指南**3.2 数据集采样**的**get_train_dataset_7b.py**工具对**10m_en_entropy.jsonl**和**10m_zh_entropy.jsonl**再次进行采样得到**10m_05_entropy.jsonl**。
**get_train_dataset_7b.py**工具的参数设置为token数为10m，中英文数据比例为0.5

## 4. 最终模型训练
基于原提交指南**4.训练**的**train_scripts/deepspeed_train_7b_lora.sh**对**10m_05_entropy.jsonl**进行训练，得到最终的模型**10m_05_entropy_baichuan**

