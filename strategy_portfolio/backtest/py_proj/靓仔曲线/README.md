# 固定组合测试系统 - 模块化版本

## 项目概述

这是一个模块化的固定组合测试系统，将原有的Jupyter notebook代码重构为功能清晰的模块化结构。

## 系统架构

```
mw_quantStrategy/
├── config.py              # 配置文件模块
├── data_loader.py         # 数据获取模块
├── backtest_engine.py     # 回测引擎模块
├── report_generator.py    # 报告生成模块
├── main.py               # 主执行脚本
├── test_modular_system.ipynb  # 测试notebook
└── 固定组合测试.ipynb        # 原始notebook
```

## 模块功能说明

### 1. config.py
- 全局配置参数管理
- 资产映射配置
- 权重设置

### 2. data_loader.py
- 数据获取和下载功能
- 数据清洗和标准化
- 价格矩阵准备

### 3. backtest_engine.py
- 回测逻辑实现
- 交易执行引擎
- 绩效指标计算
- Alpha/Beta计算

### 4. report_generator.py
- 报告生成和展示
- 图表绘制
- 结果导出功能

### 5. main.py
- 主执行流程
- 模块整合
- 完整的回测流程

## 使用方法

### 方法1: 使用Jupyter notebook
```python
# 在test_modular_system.ipynb中运行
from config import *
from data_loader import *
from backtest_engine import *
from report_generator import *

# 执行完整回测流程
```

### 方法2: 使用命令行
```bash
python main.py
```

### 方法3: 单独使用模块
```python
# 单独使用数据加载模块
from data_loader import get_all_price_series, prepare_close_matrix
prices_dict = get_all_price_series(codes, start_date, end_date)

# 单独使用回测引擎
from backtest_engine import run_backtest
results = run_backtest(close_df, weights, trade_dates)

# 单独使用报告生成
from report_generator import generate_complete_report
report = generate_complete_report(results, weights, asset_map, close_df)
```

## 主要改进

1. **模块化设计**: 将功能分离到独立的模块中
2. **函数化**: 所有功能都封装为可重用的函数
3. **清晰的接口**: 每个模块都有明确的输入输出
4. **易于维护**: 代码结构清晰，便于修改和扩展
5. **可测试性**: 每个模块可以独立测试

## 配置说明

在`config.py`中可以修改以下参数：
- 回测时间范围
- 初始资金和交易费用
- 资产组合配置
- 权重分配

## 依赖库

- pandas
- numpy
- matplotlib
- akshare
- yfinance

## 注意事项

- 首次运行需要下载数据，可能需要较长时间
- 确保网络连接正常，以便下载实时数据
- 数据会缓存到本地，后续运行会更快
