# 自媒体账号内容数据分析系统

> AI 驱动的自然语言数据分析助手 · 导入数据，用大白话提问，即刻获得结论与图表

## 项目简介

本项目是一个面向自媒体账号运营者的内容数据分析系统。核心差异化创新点是 **AI 驱动的自然语言数据分析助手**：用户导入自媒体内容数据后，可通过自然语言提问获得数据分析结论和可视化图表，无需掌握 SQL 或统计知识。

### 核心能力

- **数据导入**：支持 CSV / Excel 文件，自动识别编码（UTF-8 / GBK）
- **数据预处理**：类型转换、缺失值处理、去重、描述性统计
- **统计分析**：描述性统计、相关性分析、趋势分析、内容类型占比、Top N 排行、分布分析
- **可视化**：折线图、热力图、柱状图、箱形图、横向柱状图（matplotlib + plotly）
- **AI 自然语言问答**：单轮问答 + 多轮追问，结论附带底层统计依据
- **图表联动**：AI 结论自动触发对应可视化图表渲染
- **降级模式**：未配置 API 密钥时自动降级为本地关键词匹配统计分析

## 环境配置

### 1. 创建虚拟环境

```bash
# Windows PowerShell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

依赖清单（见 `requirements.txt`）：

| 类别 | 依赖 |
|---|---|
| 数据处理 | pandas, openpyxl, xlrd |
| 可视化 | matplotlib, plotly |
| AI 接口 | requests |
| 配置管理 | python-dotenv |
| Web 界面 | streamlit |
| 测试 | pytest |

### 3. 配置 .env 文件

在项目根目录创建 `.env` 文件（参考 `.env.example`），填入大模型 API 配置：

```ini
LLM_API_KEY=your_api_key_here
LLM_API_URL=https://api.deepseek.com/v1/chat/completions
LLM_MODEL=deepseek-chat
LLM_TIMEOUT=60
```

> **注意**：若不配置 API 密钥，系统将自动进入**降级模式**，使用本地关键词匹配进行统计分析，仍可正常使用但无自然语言理解能力。

## 运行方式

### 方式一：Streamlit Web 界面（推荐）

```bash
streamlit run src/app.py
```

浏览器访问 `http://localhost:8501`，界面包含：

- 侧边栏：文件上传 / 加载示例数据 + 数据预览表格 + 数据摘要
- 主区域：AI 问答输入框 + 结论展示（带「🤖 AI 分析」标签）+ 图表展示 + 对话历史

### 方式二：命令行模块单独运行

每个模块均含 `__main__` 自测入口，可独立运行验证：

```bash
# 测试数据加载
python src/data_loader.py

# 测试数据预处理与摘要生成
python src/data_processor.py

# 测试各分析函数
python src/analyzer.py

# 测试可视化图表生成
python src/visualizer.py

# 测试 API 客户端（需配置 .env）
python src/api_client.py

# 测试 Prompt 构建与解析
python src/prompt_engine.py

# 测试对话上下文管理
python src/context_manager.py

# 测试 AI 助手编排（降级模式）
python src/ai_assistant.py
```

### 方式三：运行单元测试

```bash
python -m pytest tests/ -v
```

## 项目结构

```
SocialMedia_AccountDataAnalyzer
├── .env                       # API 密钥配置（不提交到 Git）
├── .env.example               # 密钥配置模板
├── .gitignore
├── requirements.txt           # 依赖清单
├── engineeringrules.txt       # 项目协作规则
├── README.md                  # 本文件
├── data
│   ├── raw/                   # 原始导入数据
│   └── sample/
│       ├── sample_content.csv # 示例数据
│       └── sample_content.xlsx
├── src                        # 源代码
│   ├── data_loader.py         # 数据导入与解析
│   ├── data_processor.py      # 数据预处理 + 上下文摘要生成
│   ├── analyzer.py            # 统计分析函数（封装为可调用函数）
│   ├── visualizer.py          # 可视化图表生成
│   ├── api_client.py          # 大模型 API 调用层
│   ├── prompt_engine.py       # Prompt 模板引擎
│   ├── context_manager.py     # 对话上下文管理
│   ├── ai_assistant.py        # AI 自然语言分析助手核心编排
│   └── app.py                 # 主程序入口（Streamlit 界面）
├── tests                      # 单元测试
│   ├── test_data_loader.py    # 数据加载测试
│   ├── test_data_processor.py # 预处理与摘要测试
│   ├── test_analyzer.py       # 分析函数测试
│   ├── test_prompt_engine.py  # Prompt 引擎测试（含 API mock）
│   └── test_ai_assistant.py   # AI 助手测试占位
└── workspace                  # 项目文档
    ├── Project Start Report...# 立项报告
    └── 软件综合实践第一周...    # 周报
```

### 模块依赖关系

```
叶子模块（无内部依赖）：
  data_loader · data_processor · analyzer · visualizer
  api_client · prompt_engine · context_manager

编排层：
  ai_assistant → analyzer, api_client, context_manager,
                 data_processor, prompt_engine, visualizer

入口层：
  app → data_loader, data_processor, analyzer, visualizer, ai_assistant
```

**无循环依赖**，每个叶子模块可独立测试。

## 数据格式

示例数据字段（`data/sample/sample_content.csv`）：

| 字段 | 类型 | 说明 |
|---|---|---|
| 发布日期 | datetime | 内容发布日期 |
| 内容标题 | str | 内容标题（去重依据） |
| 内容类型 | str | 短视频 / 图文 / 长视频 / 直播回放 |
| 播放量 | int | 播放数量 |
| 点赞数 | int | 点赞数量 |
| 评论数 | int | 评论数量 |
| 转发数 | int | 转发数量 |
| 收藏数 | int | 收藏数量 |
| 粉丝增量 | int | 粉丝增长数量 |

## 已知限制

1. **API 依赖**：AI 自然语言理解能力依赖外部大模型 API，未配置时降级为关键词匹配（精度有限）
2. **单文件存储**：按项目规则禁止使用数据库，数据仅以 CSV/Excel 文件形式存储，大数据量场景性能受限
3. **单工作表**：Excel 多工作表时默认只读取第一个
4. **编码识别**：CSV 仅支持 UTF-8（含 BOM）与 GBK 编码，其他编码会报错
5. **图表保存**：当前图表仅在界面展示，未提供导出保存功能
6. **并发限制**：Streamlit 单会话模型，不支持多用户并发
7. **API 重试**：网络异常时最多重试 3 次，极端网络环境下仍可能失败

## 后续优化方向

- **P2 多平台数据适配**：支持抖音、B站、小红书等不同平台数据格式自动识别
- **P2 内容策略建议**：基于数据分析结果自动生成运营策略建议
- **P2 数据异常检测**：自动识别播放量异常波动、刷量嫌疑等
- **图表导出**：支持图表保存为 PNG / SVG 文件
- **会话导出**：支持对话历史导出为 Markdown / PDF
- **数据缓存**：使用 `st.cache_data` 加载重复数据时的性能优化
- **多轮上下文增强**：结合历史问答自动推荐下一步分析方向
- **权限管理**：多用户场景下的数据隔离与权限控制

## 技术栈

- **语言**：Python 3.12
- **数据处理**：pandas 2.1.4, openpyxl 3.1.2, xlrd 2.0.1
- **可视化**：matplotlib 3.8.2, plotly 5.9.0
- **AI 接口**：requests（调用 DeepSeek 大模型 API）
- **配置管理**：python-dotenv（.env 文件管理密钥，禁止硬编码）
- **Web 界面**：Streamlit
- **测试**：pytest
- **操作系统**：Windows 11

## Git 仓库

- **远程地址**：https://github.com/immab1tch/self-media-content-analysis
- **提交规范**：`feat:` 新功能 / `fix:` 修复 / `docs:` 文档 / `refactor:` 重构
