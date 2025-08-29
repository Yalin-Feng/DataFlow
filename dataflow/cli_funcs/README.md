# DataFlow-pdf2model&LlaMA-Factory

## 功能特性

### 核心流程

1. **PDF检测** - 自动扫描目录中的PDF文件
2. **文档处理** - PDF转换为Markdown，智能分块
3. **知识清洗** - 基于LLM的内容清洗和增强
4. **QA生成** - 多跳问答对生成
5. **数据转换** - 转换为LlamaFactory训练格式
6. **模型训练** - LoRA微调Qwen2.5-7B模型
7. **模型对话** - 启动训练后模型的对话界面

### 技术栈

- **模型**: Qwen2.5-7B-Instruct
- **训练框架**: LlamaFactory
- **微调方法**: LoRA (Low-Rank Adaptation)
- **文档处理**: MinerU + DataFlow
- **推理引擎**: vLLM

## 安装环境

```
#创建环境
conda create -n dataflow python=3.10

#激活环境
conda activate dataflow

#进入根目录
cd DataFlow

#下载mineru基础环境
pip install -e .[mineru]

#下载llamafactory环境
pip install llamafactory[torch,metrics]
pip install open-dataflow[vllm]
```



## quick start

```
#环境配置
conda create -n dataflow python=3.10
conda activate dataflow
cd DataFlow
pip install -e .[mineru]
pip install llamafactory[torch,metrics]
pip install open-dataflow[vllm]
#模型下载
#第一个两者都可以选
#第二个选all
mineru-models-download

#运行程序
cd ..
mkdir test
cd test

#初始化 
dataflow pdf2model init

#训练
dataflow pdf2model train

#与训练好的模型进行对话,也可以与本地训练好的模型对话
dataflow chat
```





## 如何使用

```
#退出根目录
cd ..

mkdir test  #创建一个新的文件夹

cd test

#初始化 
#--cache 可以指定.cache目录的位置（可选）
#默认值为当前文件夹目录
dataflow pdf2model init --cache ./my_project

#训练
#--lf_yaml 可以指定训练所用的llamafactory的yaml参数文件（可选）
#默认值为.cache/train_config.yaml
#Tip! 可以直接修改train_config.yaml中的参数，训练时会直接读取
dataflow pdf2model train --lf_yaml custom_config.yaml

#与训练好的模型进行对话,也可以与本地训练好的模型对话
#--model 可以指定 对话模型的路径位置（可选）
#默认值为.cache/saves/qwen2.5_7b_sft_model
dataflow chat --model ./custom_model_path
```



## 可视化展示

`dataflow pdf2model init` 运行结果

```
项目根目录/
├── Pdf2QAPipeline.py  # pipeline执行文件
└── .cache/            # 缓存目录
    └── train_config.yaml  # llamafactory训练的默认配置文件


```



`dataflow pdf2model train`  运行结果

```
项目根目录/
├── Pdf2QAPipeline.py  # pipeline执行文件
└── .cache/            # 缓存目录
    ├── train_config.yaml  # llamafactory训练的默认配置文件
    ├── data/
    │   ├── dataset_info.json
    │   └── qa.json
    ├── gpu/
    │   ├── batch_cleaning_step_step1.json
    │   ├── batch_cleaning_step_step2.json
    │   ├── batch_cleaning_step_step3.json
    │   ├── batch_cleaning_step_step4.json
    │   └── pdf_list.jsonl
    ├── mineru/
    │   └── sample-1-7/auto/
    └── saves/
        └── qwen2.5_7b_sft_model/
```





