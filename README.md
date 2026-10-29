<div align="center">

# TechRAG Lab

**中文技术知识与 AI 工程实验室**

数据工程 · 模型训练实验 · 基线评估

[数据流水线](data/README.md) · [数据集规划](docs/dataset_plan.md) · [实验记录](docs/experiment_log.md)

</div>

---

## 项目简介

TechRAG Lab 围绕中文技术知识开展数据集与 AI 工程实验。仓库目前包含数据处理流水线、训练组件和实验、基线评估与测试；检索增强生成（RAG）和部署属于仓库描述中的方向，当前目录中没有完整的 RAG 或部署应用。

## 仓库导览

- [data/](data/)：原始样本、数据结构、清洗去重与训练数据导出。
- [training/](training/)：分词、数据加载、Transformer 组件及训练实验。
- [evaluation/](evaluation/)：基线评估脚本与样例。
- [tests/](tests/)：数据处理、训练组件和评估相关测试。
- [docs/](docs/)：数据集规划和实验记录。

## 运行数据流水线

在仓库根目录安装依赖，然后将样例数据处理到单独的输出目录：

```bash
python -m pip install -r requirements.txt
python -m data.pipeline.run_pipeline --output-dir ./work-output
```

默认输入为 `data/raw/sample.json`，结果写入指定目录的 `train.jsonl`；成功运行会替换该目录中的同名文件。各阶段的输入格式、验证和独立运行方式见 [数据说明](data/README.md)。

## 开发检查

```bash
python -m pytest
```

训练实验可能需要额外依赖，具体以对应脚本为准。实验记录描述的是当时的阶段，请结合代码和测试查看当前状态。
