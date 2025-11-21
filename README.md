# Awesome Adaptation for Agentic AI
[![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)
[![Stars](https://img.shields.io/github/stars/pat-jj/Awesome-Adaptation-for-Agentic-AI?style=social)](https://img.shields.io/github/stars/pat-jj/Awesome-Adaptation-for-Agentic-AI?style=social)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PRWelcome](https://img.shields.io/badge/PRs-Welcome-red)](https://img.shields.io/badge/PRs-Welcome-red)

<p align="center">
    <img src="intro.png" width="90%" style="align:center;"/>
</p>

A curated list of papers on adaptation strategies for agentic AI systems. This repository accompanies the survey paper "Adaptation for Agentic AI: A Survey and Roadmap".

## Table of Contents
- [Agent Adaptation](#agent-adaptation)
  - [A1: Tool Execution Signaled](#a1-tool-execution-signaled)
  - [A2: Agent Output Signaled](#a2-agent-output-signaled)
- [Tool Adaptation](#tool-adaptation)
  - [T1: Agent-Agnostic Tool Adaptation](#t1-agent-agnostic-tool-adaptation)
  - [T2: Agent-Supervised Tool Adaptation](#t2-agent-supervised-tool-adaptation)

---

## Agent Adaptation

### A1: Tool Execution Signaled

Earlier methods using Supervised Fine-Tuning (SFT) and Direct Preference Optimization (DPO), along with recent on-policy reinforcement learning approaches.

#### RL-based Methods

| Time | Method | Venue | Task(s) | Tool(s) | Agent Backbone | Tuning | Links |
|------|--------|-------|---------|---------|----------------|--------|-------|
| 2025.11 | Orion | arXiv | IR | Retrievers | LFM2 | GRPO | [📄](https://arxiv.org/abs/2511.07581) |
| 2025.10 | olmOCR2 | arXiv | Document OCR | Synthetic Document Verifier | Qwen2.5-VL | SFT, GRPO | [📄](https://arxiv.org/abs/2510.19817) [💻](https://github.com/allenai/olmocr) |
| 2025.10 | ToolExpander | arXiv | Tool-Calling | Various APIs | Qwen2.5 | SFT + GRPO | [📄](https://arxiv.org/abs/2510.07737) |
| 2025.09 | WebGen-Agent | arXiv | Website Generation | VLM, GUI Agent, Code Executor | Various Models | SFT, Step-GRPO | [📄](https://arxiv.org/abs/2509.22644) [💻](https://github.com/mnluzimu/WebGen-Agent) |
| 2025.09 | Tool-R1 | arXiv | General Tool-Augmented Reasoning, Multimodal QA | Code Execution, Multimedia Tools | Qwen2.5 | GRPO | [📄](https://arxiv.org/abs/2509.12867) [💻](https://github.com/YBYBZhang/Tool-R1) |
| 2025.08 | FTRL | arXiv | Multi-Step Tool-Use | Simulated APIs | Qwen3 | GRPO | [📄](https://arxiv.org/abs/2508.08791) [💻](https://github.com/bytedance/FTRL) |
| 2025.06 | Router-R1 | NeurIPS'25 | Multi-Round Routing | LLM Routing Pool | Qwen2.5, LLaMA3.2 | PPO | [📄](https://arxiv.org/abs/2506.09033) [💻](https://github.com/ulab-uiuc/Router-R1) |
| 2025.05 | R1-Code-Interpreter | arXiv | Coding | Code Execution Sandbox | Qwen2.5 | GRPO | [📄](https://arxiv.org/abs/2505.21668) [💻](https://github.com/yongchao98/R1-Code-Interpreter) |
| 2025.05 | Tool-N1 | arXiv | Tool-Calling | Various APIs | Qwen2.5 | GRPO | [📄](https://arxiv.org/abs/2505.00024) [💻](https://github.com/NVlabs/Tool-N1) |
| 2025.04 | SQL-R1 | NeurIPS'25 | Text2SQL Search | SQL Engine | Qwen2.5, OmniSQL | SFT, GRPO | [📄](https://arxiv.org/abs/2504.08600) [💻](https://github.com/DataArcTech/SQL-R1) |
| 2025.03 | Rec-R1 | TMLR'25 | Recommendation Optimization | Recommendation System | Qwen2.5, LLaMA3.2 | GRPO | [📄](https://openreview.net/forum?id=YBRU9MV2vE) [💻](https://github.com/linjc16/Rec-R1) |
| 2025.03 | ReZero | arXiv | Web Search, IR | Web Search Engine | LLaMA3.2 | GRPO | [📄](https://arxiv.org/abs/2504.11001) [💻](https://github.com/janhq/ReZero) |
| 2025.03 | Code-R1 | --- | Coding | Code Executor | Qwen2.5 | GRPO | [💻](https://github.com/ganler/code-r1) |
| 2025.02 | DeepRetrieval | COLM'25 | Web Search, IR, Text2SQL | Search Engine, Retrievers, SQL exec. | Qwen2.5, LLaMA3.2 | PPO, GRPO | [📄](https://arxiv.org/abs/2503.00223) [💻](https://github.com/pat-jj/DeepRetrieval) |
| 2025.01 | DeepSeek-R1-Zero (Code) | Nature | Coding | Code Executor | DeepSeek-V3-Base | GRPO | [📄](https://arxiv.org/abs/2501.12948) |
| 2024.10 | RLEF | ICML'25 | Coding | Code Executor | LLaMA3.1 | PPO | [📄](https://arxiv.org/abs/2410.02089) |
| 2024.05 | LeDex | NeurIPS'24 | Coding | Code Executor | StarCoder & CodeLlaMA | SFT, PPO | [📄](https://arxiv.org/abs/2405.18649) |


#### SFT & DPO Methods

| Time | Method | Venue | Task(s) | Tool(s) | Agent Backbone | Tuning | Links |
|------|--------|-------|---------|---------|----------------|--------|-------|
| 2024.10 | LeReT | ICLR'25 | IR | Dense Retriever | LLaMA3, Gemma2 | DPO-like (IPO) | [📄](https://arxiv.org/abs/2410.23214) [💻](https://github.com/sher222/LeReT) |
| 2024.10 | ToolFlow | NAACL'25 | Tool-Calling | Various APIs | LLaMA3.1 | SFT | [📄](https://arxiv.org/abs/2410.18447) |
| 2024.06 | TP-LLaMA | NeurIPS'24 | Tool-Calling | Various APIs | LLaMA2 | SFT, DPO | [📄](https://arxiv.org/abs/2406.07115) |
| 2024.05 | AutoTools | WWW'25 | Automated Tool-Calling | Various APIs | GPT4, LLaMA3, Mistral | SFT | [📄](https://arxiv.org/abs/2405.16533) [💻](https://github.com/mangopy/AutoTools) |
| 2024.03 | CYCLE | OOPSLA'24 | Coding | Code Executor | CodeGen, StarCoder | SFT | [📄](https://arxiv.org/abs/2403.18746) |
| 2024.02 | RetPO | NAACL'25 | IR | Retriever | LLaMA2-7B | SFT, DPO | [📄](https://arxiv.org/abs/2402.11827) [💻](https://github.com/dmis-lab/RetPO) |
| 2024.02 | CodeAct | ICML'24 | Coding | Code Executor | LLaMA2, Mistral | SFT | [📄](https://arxiv.org/abs/2402.01030) [💻](https://github.com/xingyaoww/code-act) |
| 2024.01 | NExT | ICML'24 | Program Repair | Code Executor | PaLM2 | SFT | [📄](https://arxiv.org/abs/2404.14662) |
| 2023.07 | ToolLLM | ICLR'24 | Tool-Calling, API Planning, Multi-Tool Reasoning | Real-World APIs | LLaMA, Vicuna | SFT | [📄](https://arxiv.org/abs/2307.16789) [💻](https://github.com/OpenBMB/ToolBench) |
| 2023.06 | ToolAlpaca | arXiv | Multi-Turn Tool-Use | Simulated APIs | Vicuna | SFT | [📄](https://arxiv.org/abs/2306.05301) [💻](https://github.com/tangqiaoyu/ToolAlpaca) |
| 2023.05 | Gorilla | NeurIPS'24 | Tool-Calling, API Retrieval | Various APIs | LLaMA | SFT | [📄](https://arxiv.org/abs/2305.15334) [💻](https://github.com/ShishirPatil/gorilla) |
| 2023.05 | TRICE | NAACL'24 | Math Reasoning, QA, Multilingual QA, Knowledge Retrieval | Calculator, WikiSearch, Atlas QA Model, NLLB Translator | ChatGLM, Alpaca, Vicuna | SFT | [📄](https://arxiv.org/abs/2305.13068) [💻](https://github.com/zjunlp/TRICE) |
| 2023.02 | Toolformer | NeurIPS'23 | QA, Math | Calculator, QA system, Search Engine, Translation System, Calendar | GPT-J | SFT | [📄](https://arxiv.org/abs/2302.04761) [💻](https://github.com/conceptofmind/toolformer) |

---

### A2: Agent Output Signaled

Methods that adapt agents using signals from agent outputs, organized by whether they use external tools.


#### Methods with Tools

| Time | Method | Venue | Task(s) | Tool(s) | Agent Backbone | Tuning | Links |
|------|--------|-------|---------|---------|----------------|--------|-------|
| 2025.10 | TT-SI | arXiv | Tool Calling | Various APIs | Qwen2.5 | Test-Time Fine-Tuning | [📄](https://arxiv.org/abs/2510.07841) |
| 2025.10 | A²FM | arXiv | Web Navigation, Math, QA | Search Engine, Crawl, Code Executor | Qwen2.5 | APO, GRPO | [📄](https://arxiv.org/abs/2510.12838) [💻](https://github.com/OPPO-PersonalAI/Adaptive_Agent_Foundation_Models) |
| 2025.08 | MedResearcher-R1 | arXiv | Medical Multi-hop QA | Medical Retriever, Web Search API, Document Reader | MedResearcher-R1 | SFT, GRPO | [📄](https://arxiv.org/abs/2508.14880) [💻](https://github.com/AQ-MedAI/MedResearcher-R1) |
| 2025.08 | Agent Lightning | arXiv | Text-to-SQL, RAG, Math | SQL Executor, Retriever, Calculator | LLaMA3.2 | LightningRL | [📄](https://arxiv.org/abs/2508.03680) [💻](https://github.com/microsoft/agent-lightning) |
| 2025.07 | CodePRM | ACL'25 | Coding | Code Executor | Qwen2.5-Coder | SFT | [📄](https://aclanthology.org/2025.findings-acl.428/) |
| 2025.07 | DynaSearcher | arXiv | Multi-Hop QA, RAG | Document Search, KG Search | Qwen2.5, LLaMA3.1 | GRPO | [📄](https://arxiv.org/abs/2507.17365) [💻](https://modelscope.cn/home) |
| 2025.06 | MMSearch-R1 | arXiv | Web Browsing, QA, Multimodal Search | Image Search, Web Browsing, Retriever | Qwen2.5 | REINFORCE, SFT | [📄](https://arxiv.org/abs/2506.20670) [💻](https://github.com/EvolvingLMMs-Lab/multimodal-search-r1) |
| 2025.06 | Self-Challenging | arXiv | Web Browsing, Calculation, Retail, Airline | Code Interpreter, Web Browser, Database APIs | LLaMA3.1 | REINFORCE, SFT | [📄](https://arxiv.org/abs/2506.01716) |
| 2025.05 | StepSearch | EMNLP'25 | Multi-Hop QA | Search Engine, Retriever | Qwen2.5 | StePPO | [📄](https://arxiv.org/abs/2505.15107) [💻](https://github.com/Zillwang/StepSearch) |
| 2025.05 | ZeroSearch | arXiv | Multi-Hop QA, QA | Search Engine, Web Search | Qwen2.5, LLaMA3.2 | REINFORCE, GPRO, PPO, SFT | [📄](https://arxiv.org/abs/2505.04588) [💻](https://github.com/Alibaba-NLP/ZeroSearch) |
| 2025.05 | AutoRefine | NeurIPS'25 | Multi-Hop QA, QA | Retriever | Qwen2.5 | GRPO | [📄](https://arxiv.org/abs/2505.11277) [💻](https://github.com/syr-cn/AutoRefine) |
| 2025.04 | ReTool | arXiv | Math | Code Interpreter | Qwen2.5 | PPO | [📄](https://arxiv.org/abs/2504.11536) [💻](https://github.com/ReTool-RL/ReTool) |
| 2025.04 | ToolRL | arXiv | Tool Calling | Various Tools | Various Models | GRPO | [📄](https://arxiv.org/abs/2504.13958) [💻](https://github.com/qiancheng0/ToolRL) |
| 2025.04 | DeepResearcher | arXiv | QA, Multi-Hop Reasoning, Deep Research | Web Search API, Web Browser | Qwen2.5 | GRPO | [📄](https://arxiv.org/abs/2504.03160) [💻](https://github.com/GAIR-NLP/DeepResearcher) |
| 2025.03 | ReSearch | NeurIPS'25 | QA | Search Engine, Retriever | Qwen2.5 | GRPO | [📄](https://arxiv.org/abs/2503.19470) [💻](https://github.com/Agent-RL/ReCall) |
| 2025.03 | Search-R1 | COLM'25 | QA | Search Engine, Retriever | Qwen2.5 | PPO, GRPO | [📄](https://arxiv.org/abs/2503.09516) [💻](https://github.com/PeterGriffinJin/Search-R1) |
| 2025.03 | R1-Searcher | arXiv | QA | Retriever | LLaMA3.1, Qwen2.5 | REINFORCE++ | [📄](https://arxiv.org/abs/2503.05592) [💻](https://github.com/RUCAIBox/R1-Searcher) |
| 2025.02 | RAS | arXiv | QA | Retriever | LLaMA2, LLaMA3.2 | SFT | [📄](https://arxiv.org/abs/2502.10996) [💻](https://github.com/pat-jj/RAS) |
| 2025.01 | Agent-R | arXiv | Various Tasks | Monte Carlo Tree Search | Qwen2.5, LLaMA3.2 | SFT | [📄](https://arxiv.org/abs/2501.11425) [💻](https://github.com/ByteDance-Seed/Agent-R) |
| 2024.06 | Re-ReST | EMNLP'24 | Multi-Hop QA, VQA, Sequential Decision, Coding | Various APIs | Various Models | DPO | [📄](https://arxiv.org/abs/2406.01495) [💻](https://github.com/PlusLabNLP/Re-ReST) |
| 2024.06 | RPG | EMNLP'24 | RAG, QA, Multi-hop Reasoning | Search Engine, Retriever | LLaMA2, GPT3.5 | SFT | [📄](https://arxiv.org/abs/2406.14979) [💻](https://github.com/haruhi-sudo/RPG) |
| 2023.10 | Self-RAG | ICLR'24 | RAG, QA, Fact Verification | Retriever | LLaMA2 | SFT | [📄](https://arxiv.org/abs/2310.11511) [💻](https://github.com/AkariAsai/self-rag) |
| 2023.10 | FireAct | arXiv | QA | Search API | GPT3.5, LLaMA2, CodeLLaMA | SFT | [📄](https://arxiv.org/abs/2310.05915) [💻](https://fireact-agent.github.io) |

#### Methods without Tools

| Time | Method | Venue | Task(s) | Tool(s) | Agent Backbone | Tuning | Links |
|------|--------|-------|---------|---------|----------------|--------|-------|
| 2025.10 | Empower | arXiv | Coding | --- | Gemma3 | SFT | [📄](https://arxiv.org/abs/2510.13709) [💻](https://github.com/festusev/codegen_empowerment/tree/main) |
| 2025.10 | KnowRL | arXiv | Knowledge calibration | --- | LLaMA3.1, Qwen2.5 | REINFORCE++ | [📄](https://arxiv.org/abs/2510.11407) [💻](https://anonymous.4open.science/r/KnowRL-5BF0) |
| 2025.10 | GRACE | arXiv | Embedding Tasks | --- | Qwen2.5, Qwen3, LLaMA3.2 | GRPO | [📄](https://arxiv.org/abs/2510.04506) [💻](https://github.com/GasolSun36/GRACE) |
| 2025.06 | Magistral | arXiv | Math, Coding | --- | Magistral | PPO, GRPO | [📄](https://arxiv.org/abs/2506.10910) |
| 2025.05 | EHRMind | arXiv | EHR-based Reasoning | --- | LLaMA3 | SFT, GRPO | [📄](https://arxiv.org/abs/2505.24105) |
| 2025.01 | Kimi k1.5 | arXiv | Math, Coding | --- | Kimi k1.5 | GRPO | [📄](https://arxiv.org/abs/2501.12948) [💻](https://github.com/MoonshotAI/Kimi-k1.5) |
| 2025.01 | DeepSeek-R1-Zero (Math) | Nature | Math | --- | DeepSeek-V3 | GRPO | [📄](https://arxiv.org/abs/2501.12948) |
| 2024.09 | SCoRe | ICLR'25 | Math, Coding, QA | --- | Gemini1.0 Pro, Gemini1.5 Flash | REINFORCE | [📄](https://arxiv.org/abs/2409.12917) [💻](https://github.com/BY571/SCoRe) |
| 2024.07 | RISE | NeurIPS'24 | Math | --- | LLaMA2, LLaMA3, Mistral | SFT | [📄](https://arxiv.org/abs/2407.18219) [💻](https://github.com/cmu-mind/RISE) |
| 2024.06 | TextGrad | Nature | Various Tasks | --- | GPT3.5, GPT4o | Textual Gradient Descent | [📄](https://arxiv.org/abs/2406.07496) [💻](https://github.com/zou-group/textgrad) |
| 2023.03 | Self-Refine | NeurIPS'23 | Dialogue, Math, Coding | --- | GPT3.5, GPT4, CODEX | --- | [📄](https://arxiv.org/abs/2303.17651) [💻](https://github.com/madaan/self-refine) |

---

## Tool Adaptation

### T1: Agent-Agnostic Tool Adaptation

#### Foundational Systems and Architectures

| Year.Month | Method Name | Paper Name | Venue | Paper Link | Github Link |
|:-----------:|:-----------:|-----------|:-----------:|:-----------:|:-----------:|
| 2023.XX | Neural Operators | Neural Operator: Learning Maps Between Function Spaces | JMLR'23 | [paper](https://jmlr.org/papers/v24/21-1524.html) | - |
| 2023.09 | HuggingGPT | HuggingGPT: Solving AI Tasks with ChatGPT and its Friends in Hugging Face | NeurIPS'23 | [paper](https://arxiv.org/abs/2303.17580) | [code](https://github.com/microsoft/JARVIS) |
| 2023.08 | ViperGPT | ViperGPT: Visual Inference via Python Execution for Reasoning | ICCV'23 | [paper](https://arxiv.org/abs/2303.08128) | [code](https://github.com/cvlab-columbia/viper) |
| 2025.XX | SciToolAgent | SciToolAgent: A Knowledge-Graph-Driven Scientific Agent for Multitool Integration | Nature Comp. Sci.'25 | [paper](https://www.nature.com/articles/s43588-025-00748-w) | - |

#### Categories and Training Methods

| Year.Month | Method Name | Paper Name | Venue | Paper Link | Github Link |
|:-----------:|:-----------:|-----------|:-----------:|:-----------:|:-----------:|
| 2021.01 | CLIP | Learning Transferable Visual Models from Natural Language Supervision | ICML'21 | [paper](https://arxiv.org/abs/2103.00020) | [code](https://github.com/openai/CLIP) |
| 2023.04 | SAM | Segment Anything | ICCV'23 | [paper](https://arxiv.org/abs/2304.02643) | [code](https://github.com/facebookresearch/segment-anything) |
| 2024.06 | SAM-CLIP | SAM-CLIP: Merging Vision Foundation Models Towards Semantic and Spatial Understanding | CVPR'24 | [paper](https://arxiv.org/abs/2310.15308) | - |
| 2023.12 | Whisper | Robust Speech Recognition via Large-Scale Weak Supervision | ICML'23 | [paper](https://arxiv.org/abs/2212.04356) | [code](https://github.com/openai/whisper) |
| 2024.10 | CodeAct | Executable Code Actions Elicit Better LLM Agents | ICML'24 | [paper](https://arxiv.org/abs/2402.01030) | [code](https://github.com/xingyaoww/code-act) |
| 2020.04 | DPR | Dense Passage Retrieval for Open-Domain Question Answering | EMNLP'20 | [paper](https://arxiv.org/abs/2004.04906) | [code](https://github.com/facebookresearch/DPR) |
| 2020.04 | ColBERT | ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT | SIGIR'20 | [paper](https://arxiv.org/abs/2004.12832) | [code](https://github.com/stanford-futuredata/ColBERT) |
| 2021.12 | Contriever | Unsupervised Dense Information Retrieval with Contrastive Learning | TMLR'22 | [paper](https://arxiv.org/abs/2112.09118) | [code](https://github.com/facebookresearch/contriever) |
| 2022.12 | e5 | Text Embeddings by Weakly-Supervised Contrastive Pre-training | arXiv | [paper](https://arxiv.org/abs/2212.03533) | [code](https://github.com/microsoft/unilm/tree/master/e5) |
| 2021.07 | AlphaFold2 | Highly Accurate Protein Structure Prediction with AlphaFold | Nature | [paper](https://www.nature.com/articles/s41586-021-03819-2) | [code](https://github.com/deepmind/alphafold) |
| 2023.03 | ESMFold | Evolutionary-Scale Prediction of Atomic-Level Protein Structure with a Language Model | Science | [paper](https://www.science.org/doi/10.1126/science.ade2574) | - |

---

### T2: Agent-Supervised Tool Adaptation

Methods for adapting tools using agent supervision signals.

| Time | Method | Venue | Task(s) | Tool Backbone | Agent Backbone | Tuning | Links |
|------|--------|-------|---------|---------------|----------------|--------|-------|
| 2025.10 | QAgent | arXiv | QA, RAG | Qwen2.5-3B | Qwen-7B | GRPO | [📄](https://arxiv.org/abs/2510.08383) [💻](https://github.com/LivingFutureLab/QAgent) |
| 2025.10 | AgentFlow | arXiv | Web Search, Planning, Reasoning, Math | Qwen2.5-7B | Qwen2.5-7B | Flow-GRPO | [📄](https://arxiv.org/abs/2510.05592) [💻](https://github.com/lupantech/AgentFlow) |
| 2025.10 | Advisor Models | arXiv | Math, Reasoning | Qwen2.5-7B, Qwen3-8B | GPT-4o-Mini, GPT-5, Claude4-Sonnet, GPT-4.1-Mini | GRPO | [📄](https://arxiv.org/abs/2510.02453) [💻](https://github.com/az1326/advisor-models) |
| 2025.10 | AutoGraph-R1 | arXiv | KG Construction, RAG | KG Constructor (Qwen2.5-3B/7B) | Frozen RAG Generator (Qwen2.5-7B) | GRPO | [📄](https://arxiv.org/abs/2510.15339) [💻](https://github.com/HKUST-KnowComp/AutoGraph-R1) |
| 2025.10 | MAE | arXiv | Math, Coding, Commonsense Reasoning | Qwen2.5-3B | Qwen2.5-3B | REINFORCE++ | [📄](https://arxiv.org/abs/2510.23595) [💻](https://github.com/ulab-uiuc/Multi-agent-Evolve) |
| 2025.09 | Mem-α | arXiv | Retrieval, Test-Time Learning, Long-Range Understanding | Qwen3-4B | Qwen3-4B, Qwen3-32B, GPT-4.1-Mini | GRPO | [📄](https://arxiv.org/abs/2509.25911) [💻](https://github.com/wangyu-ustc/Mem-alpha) |
| 2025.08 | AI-SearchPlanner | arXiv | Web QA | Qwen3-32b | Qwen2.5-7B | PPO | [📄](https://arxiv.org/abs/2508.20368) |
| 2025.08 | Memento | arXiv | Long-Horizon Reasoning, Web Research, QA, Academic Reasoning | Q-function (two-layer MLPs) | GPT-4.1 | Soft Q-Learning | [📄](https://arxiv.org/abs/2508.16153) [💻](https://github.com/Agent-on-the-Fly/Memento) |
| 2025.08 | R-Zero | arXiv | Math, Reasoning | Qwen3-4B, Qwen3-8B, OctoThinker-3B, OctoThinker-8B | Qwen3-4B, Qwen3-8B, OctoThinker-3B, OctoThinker-8B | GRPO | [📄](https://arxiv.org/abs/2508.05004) [💻](https://github.com/Chengsong-Huang/R-Zero) |
| 2025.06 | Sysformer | arXiv | QA, RAG | Small Transformer | LLaMA-2-7B, LLaMA-3.1-8B, Mistral-7B, Phi-3.5-mini, Zephyr-7B-beta | Supervised Learning | [📄](https://arxiv.org/abs/2506.15751) |
| 2025.05 | s3 | EMNLP'25 | QA, RAG | Qwen2.5-7B | Qwen2.5-7B, Qwen2.5-14B, Claude-3-Haiku | PPO | [📄](https://arxiv.org/abs/2505.14146) [💻](https://github.com/pat-jj/s3) |
| 2024.10 | Matryoshka Pilot | NeurIPS'25 | Math, Planning, Reasoning | LLaMA3-8B, Qwen2.5-7B | GPT-4o-Mini, GPT-3.5-Turbo | DPO, IDPO | [📄](https://arxiv.org/abs/2410.20749) [💻](https://github.com/lichangh20/Matryoshka) |
| 2024.06 | CoBB | EMNLP'24 | QA, Math | Mistral-7b-inst-v2 | GPT-3.5-Turbo, Claude-3-Haiku, Phi-3-mini-4k-inst, Gemma-1.1-7B-it, Mistral-7B-inst-v2 | SFT, ORPO | [📄](https://arxiv.org/abs/2406.18695) [💻](https://github.com/bbuing9/CoBB) |
| 2024.05 | Medadapter | EMNLP'24 | Medical QA, NLI, RQE | BERT-Base-Uncased | GPT-3.5-Turbo | SFT, BPO | [📄](https://arxiv.org/abs/2405.03000) [💻](https://github.com/wshi83/MedAdapter) |
| 2024.03 | BLADE | AAAI'25 | Domain-Specific QA | BLOOMZ-1b7 | ChatGPT, ChatGLM, Baichuan, Qwen | SFT, BPO | [📄](https://arxiv.org/abs/2403.18365) [💻](https://github.com/CSHaitao/BLADE) |
| 2024.02 | ARL2 | ACL'24 | QA | LLaMA2-7B | GPT-3.5-Turbo | Contrastive Learning | [📄](https://arxiv.org/abs/2402.13542) [💻](https://github.com/zhanglingxi-cs/ARL2) |
| 2024.02 | EVOR | EMNLP'24 | RAG-based Coding | GPT-3.5-Turbo | GPT-3.5-Turbo, CodeLLaMA | Prompt Engineering | [📄](https://arxiv.org/abs/2402.12317) [💻](https://github.com/xlang-ai/EVOR) |
| 2024.02 | Bbox-Adapter | ICML'24 | QA | DeBERTa-v3-base (0.1B), DeBERTa-v3-large (0.3B) | GPT-3.5-Turbo, Mixtral-8x7B | Contrastive Learning | [📄](https://arxiv.org/abs/2402.08219) [💻](https://github.com/haotiansun14/BBox-Adapter) |
| 2024.01 | Proxy-Tuning | COLM'24 | QA, Math, Code | LLaMA2-7B | LLaMA2-70B | Proxy-Tuning | [📄](https://arxiv.org/abs/2401.08565) [💻](https://github.com/alisawuffles/proxy-tuning) |
| 2024.01 | BGM | ACL'24 | QA, Personalized Generation (NQ, HotpotQA, Email, Book) | T5-XXL-11B | PaLM2-S | SFT, PPO | [📄](https://arxiv.org/abs/2401.06954) |
| 2023.10 | RA-DIT | ICLR'24 | Knowledge-Intensive Tasks (MMLU, NQ, TQA, ELI5, HotpotQA, etc.) | DRAGON+ | LLaMA-65B | SFT, LSR | [📄](https://arxiv.org/abs/2310.01352) |
| 2023.06 | LLM-R | EACL'24 | Zero-shot NLU (Reading Comprehension, QA, NLI, Paraphrase, Sentiment, Summarization) | E5-base | GPT-Neo-2.7B, LLaMA-13B, GPT-3.5-Turbo | Contrastive Learning | [📄](https://arxiv.org/abs/2307.07164) [💻](https://github.com/microsoft/LMOps/tree/main/llm_retriever) |
| 2023.05 | AAR | ACL'23 | Zero-Shot Generalization (MMLU, PopQA) | ANCE, Contriever | Flan-T5-Small, InstructGPT | Contrastive Learning | [📄](https://arxiv.org/abs/2305.17331) [💻](https://github.com/OpenMatch/Augmentation-Adapted-Retriever) |
| 2023.05 | ToolkenGPT | NeurIPS'23 | Numerical Reasoning, QA, Plan Generation | Token Embedding | GPT-J 6B, OPT-6.7B, OPT-13B | Proxy-Tuning | [📄](https://arxiv.org/abs/2305.11554) [💻](https://github.com/Ber666/ToolkenGPT) |
| 2023.03 | UPRISE | EMNLP'23 | Zero-shot NLU (Reading Comprehension, QA, NLI, Paraphrase, Sentiment, Summarization) | GPT-Neo-2.7B | BLOOM-7.1B, OPT-66B, GPT-3-175B | Contrastive Learning | [📄](https://aclanthology.org/2023.emnlp-main.758/) [💻](https://github.com/microsoft/LMOps) |
| 2023.01 | REPLUG | NAACL'24 | QA | Contriever | GPT3-175B, PaLM, Codex, LLaMA-13B | Proxy-Tuning, LSR | [📄](https://aclanthology.org/2024.naacl-long.463.pdf) [💻](https://github.com/swj0419/REPLUG) |

---

## Citation

If you find this repository useful, please consider citing our survey:

```bibtex
@article{adaptation_agentic_ai_2025,
  title={Adaptation for Agentic AI: A Survey and Roadmap},
  author={[Authors]},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2025}
}
```

## Contributing

We welcome contributions! Please feel free to submit a Pull Request to add new papers or update existing entries.

## License

This project is licensed under the MIT License.
