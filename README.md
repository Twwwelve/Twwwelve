# Hanlin He / 何翰林

Computer Science Undergraduate @ Guangdong Pharmaceutical University  
广东药科大学 · 计算机科学与技术本科生｜AI 应用开发 / LLM Agent / 医学影像

我关注大语言模型应用、Agent 系统与医学影像深度学习，具有医学多模态自动训练应用开发经历。能够将模型能力与实际任务结合，完成工具调用、实验追踪、模型评估与结果验证。

I focus on LLM applications, agent systems, and deep learning for medical imaging, with experience developing a medical multimodal training application. I can translate model capabilities into practical workflows covering tool calling, experiment tracking, model evaluation, and result verification.

> 关注方向：AI 应用开发 · LLM Agent · RAG · 医学影像分析  
> Focus: AI Applications · LLM Agents · RAG · Medical Image Analysis

---

## Technical Focus / 技术方向

- **LLM Agent**：任务规划、Tool Calling、分层 Skill 路由、记忆与上下文管理，以及多 Agent 调度与结果传递。
- **RAG**：文档解析与切块、稀疏与向量混合召回、Rerank 重排、引用核查与低置信度二次检索。
- **Medical Imaging**：多模态 MRI、三维医学图像分割、轻量化网络与模态融合。
- **Model Training & Evaluation**：模型搭建、训练调优与推理部署，关注患者级数据划分、数据泄漏防范与交叉验证。
- **Reliable Tool Execution**：使用 JSON Schema、动作白名单和预算校验约束 Agent 执行，结合异常降级、断点恢复与实验追踪。

---

## Featured Projects / 代表项目

### GSM-UNet — 区域先验引导的多模态脑肿瘤分割

**Region-Guided Multimodal Brain Tumor Segmentation**

面向 BraTS 2020 四模态 MRI（T1 / T1ce / T2 / FLAIR）的三维脑肿瘤分割项目，覆盖数据预处理、网络设计、训练、整卷推理和可视化评估。基于 U-Net 构建轻量化骨干，结合区域模态融合与感受野选择，处理不同肿瘤区域对模态信息和空间尺度的不同需求。

A 3D brain tumor segmentation project for four-modality BraTS 2020 MRI, covering preprocessing, network design, training, full-volume inference, and visual evaluation. The U-Net-based architecture combines a lightweight backbone with region-guided modality fusion and receptive-field selection to accommodate regional differences in imaging information and spatial scale.

**Tech Stack / 技术栈**  
`Python` · `PyTorch` · `NumPy` · `SciPy` · `NiBabel` · `PyYAML` · `TensorBoard` · `Matplotlib`

**Technical Highlights / 创新点与技术亮点**

- **区域先验引导的双尺度融合**：在全分辨率与半分辨率独立编码各模态，由影像特征预测四类区域先验，再组合区域模态专家，生成体素级、归一化的模态权重。真实标签仅用于训练监督，推理时由影像自身生成先验。  
  **Region-guided dual-scale fusion:** Encode each modality at full and half resolution; combine image-derived regional priors with modality experts to produce normalized voxel-wise modality weights. Ground-truth masks provide training supervision only.
- **自适应感受野选择**：在中层特征上引入空洞率 1 / 2 的双分支分组卷积，以通道级 Softmax 权重融合不同尺度特征，并通过可学习残差系数控制其贡献。  
  **Adaptive receptive-field selection:** Fuse grouped-convolution branches with dilation rates of 1 and 2 using channel-wise softmax weights and a learnable residual scale.
- **轻量化骨干与条件解码**：组合残差分组卷积、Channel Shuffle、MiniMFRM 多尺度聚合、Attention Gate 与 ECA，并通过模态描述子产生 FiLM 参数调制解码特征。  
  **Lightweight backbone and conditioned decoding:** Combine residual grouped convolutions, channel shuffle, MiniMFRM aggregation, attention gates, and ECA, with modality-conditioned FiLM modulation in the decoder.
- **兼顾小区域的训练与验证**：采用区域 Dice 与交叉熵、两级解码深监督及先验辅助监督；将辅助输出上采样到原始标签网格，避免下采样标签丢失微小增强肿瘤。支持重叠滑窗、TTA、五折评估，以及 ET 体积与置信度后处理。  
  **Small-region-aware training and evaluation:** Use regional Dice and cross-entropy with decoder and prior supervision at the original label resolution. Support overlapping sliding-window inference, TTA, five-fold evaluation, and ET volume-and-confidence postprocessing.

---

### ClinMmTrain — Agent 驱动的医学多模态自动训练

**Agent-Driven Medical Multimodal Training**

面向医学影像与多分支数据建模的自动训练系统，将数据分析、模型设计、训练决策与评估组织为“观察—决策—执行—反馈”流程。LLM 提出候选方案，本地执行层负责校验、预算控制和验收，支持基线搜索、多模态融合、实验追踪与断点恢复。

An automated training system for medical imaging and multi-branch modeling. It organizes data analysis, model design, training decisions, and evaluation into an observation–decision–execution–feedback loop. LLMs propose candidates while local execution enforces validation, budgets, and acceptance criteria, with support for baseline search, multimodal fusion, experiment tracking, and recovery.

**Tech Stack / 技术栈**  
`Python` · `PyTorch` · `PyTorch Lightning` · `Optuna` · `LLM Agent` · `Tool Calling` · `JSON Schema`

**Technical Highlights / 创新点与技术亮点**

- **受约束的 Agent 训练闭环**：通过分析、设计、训练决策与评估 Agent 分工，将训练反馈转为后续动作；结合工具参数校验、动作白名单与本地审计，限制模型建议的执行范围。  
  **Constrained agent training loop:** Separate analysis, design, training decisions, and evaluation into agent roles; translate observations into actions governed by tool-argument validation, action allowlists, and local audit records.
- **学习信号验证与数据隔离**：结合患者级划分、真实标签重复训练与配对置换对照，验证当前协议下是否存在足够学习证据；冻结测试集，避免将测试反馈用于搜索与调参。  
  **Learning-signal checks and data isolation:** Combine patient-level splits, repeated real-label training, and paired permutation controls to assess learning evidence, keeping the test set frozen and outside model selection.
- **跨窗口停滞监管与可恢复搜索**：根据多个训练窗口的改善幅度、学习率及近期趋势约束续训，接入 Optuna ask/tell/report 管理候选配置；通过原子 JSON 检查点保存试验状态并重建搜索过程。  
  **Window-level supervision and recoverable search:** Govern continuation using improvement across training windows, learning rates, and recent trends. Integrate Optuna ask/tell/report with atomic JSON checkpoints for trial-state persistence and study reconstruction.
- **多分支融合与统一预算管理**：支持 Concat、Late、Gated 与 DAFT 融合；累计记录基线、对照、融合及重训练耗时，结合配置与数据指纹、检查点和运行清单追踪实验，支持异常回退与恢复。  
  **Multi-branch fusion and unified budget accounting:** Support concat, late, gated, and DAFT fusion; track time across baselines, controls, fusion trials, and retraining, with configuration/data fingerprints, checkpoints, and run manifests for traceability and recovery.
---

## Experience / 工作经历

### 佰茵云（广州）医学研究有限责任公司

**AI 应用开发工程师｜2026.06 – 2026.10**

参与 LLM Agent 驱动的医学多模态自动调参训练应用开发，围绕训练状态分析、工具执行与评估反馈构建自动调优流程。

- 将训练曲线、梯度、预测分布与历史调参效果转为结构化观察，支持预定义空间内的超参调整、架构切换与融合选择。
- 封装数据校验、模型构建、训练与评估工具，通过 Tool Calling、JSON Schema、动作白名单与预算校验约束执行。
- 支持异常降级与断点恢复，关联数据指纹、配置和检查点，便于追踪实验过程。
- 参与数据去重、患者级划分及真实标签与置换对照验证，关注训练与评估过程中的数据泄漏问题。

Contributed to an LLM-agent-driven medical multimodal training application, including structured training observations, constrained tool execution, experiment traceability, and leakage-aware evaluation.

---

## Education / 教育背景

**广东药科大学 · 计算机科学与技术（本科）**  
Guangdong Pharmaceutical University · Computer Science and Technology  
2023.09 – 至今 / Present

主修课程：数据结构、算法分析与设计、人工智能、医学图像处理、大数据技术与应用、C 语言程序设计、Java 核心技术。

---

## Honors & Awards / 荣誉奖项

- **国家级三等奖**｜第十九届中国大学生计算机设计大赛 · 移动应用开发。
- **省级二等奖**｜第十九届中国大学生计算机设计大赛粤港澳大湾区赛 · 移动应用开发赛道。
- **省级三等奖**｜第十九届中国大学生计算机设计大赛粤港澳大湾区赛 · 人工智能赛道。
- **省级三等奖**｜2025 年（第十一届）全国大学生统计建模大赛广东赛区。
- **一等奖**｜广东药科大学“省职防院杯”大学生数学建模竞赛。
- **2025 年度奖学金**｜广东药科大学蒲公英奖学金（学术科技类）、百合奖学金（优秀班干部）。

---

## Tech Stack / 技术栈

**Languages / 编程语言**  
`Python` · `C++`

**Machine Learning & Deep Learning / 机器学习与深度学习**  
`PyTorch` · `PyTorch Lightning` · `TensorFlow` · `scikit-learn` · `U-Net` · `3D Segmentation` · `Multimodal Learning`

**LLM & Agents / 大模型与智能体**  
`LLM Agent` · `Tool Calling` · `Multi-Agent` · `Skill Routing` · `Context Management` · `JSON Schema`

**Retrieval / 检索增强**  
`RAG` · `Document Chunking` · `Hybrid Retrieval` · `Reranking` · `Citation Verification`

**Training & Evaluation / 训练与评估**  
`Optuna` · `Cross-Validation` · `AMP` · `EMA` · `Test-Time Augmentation` · `Experiment Tracking`

---

## Areas of Interest / 持续关注

- 将 LLM Agent 与实际任务结合，形成可追踪、可验证的执行流程。
- 优化 RAG 检索与答案核查，提高信息来源的可追溯性。
- 探索医学影像中的轻量化网络、多模态融合与分割方法。
- 完善模型训练、评估与推理流程，让实验结果更容易复现。

---

## Contact / 联系方式

- **Email**：[2110452773@qq.com](mailto:2110452773@qq.com)
- **GitHub**：[@Twwwelve](https://github.com/Twwwelve)
