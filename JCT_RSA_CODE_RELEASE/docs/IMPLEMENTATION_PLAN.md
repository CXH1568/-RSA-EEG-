# JCT-RSA Code Release Implementation Plan

> For agentic workers: REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. No experiment execution is authorized.

**Goal:** 创建独立 code-only 目录及同名 ZIP，保留真实科学代码，排除所有数据和历史产物。

**Authorized gate amendment:** 用户在 Task 4 阻塞后明确取消 scientific dependency full import 作为打包条件。不继续诊断历史超时；记录 SCIENTIFIC_DEPENDENCY_IMPORT=NOT_VERIFIED。仅要求语法、CLI help、文件白名单、无数据/结果、敏感信息/本机路径检查及原归档不变。下文历史方案中关于完整科学依赖 import 阻塞发布的要求由此条取代。EXPERIMENTS=NOT_RUN，NUMERICAL_RESULTS=NOT_VERIFIED，CODE_PACKAGE_ONLY=YES。

**Architecture:** 保留归档投稿代码的科学计算段，在新副本隔离顶层执行副作用；四个薄 CLI 调用实际流程。配置使用 INI，避免把 JSON 结果误装入发布包。绘图只读取用户提供的结果，不启动拟合或统计。

**Tech Stack:** Python、argparse、configparser、原 NumPy/SciPy/scikit-learn/MNE/PyTorch/CLIP/Matplotlib 依赖。

**Spec:** 用户最新 code-only 执行指令为准，取代旧设计中完整复现包、后续实验迁移及额外模块重组要求。旧 `docs/RELEASE_DESIGN.md` 仅供设计背景。

## Global Constraints

- 原 `ICASSP2027_CODE_DATA_ARCHIVE.zip` 及解压归档全部只读。
- 所有新增/修改文件仅在新 `ICASSP2027_JCT_RSA_CODE_RELEASE/` 内；交付 ZIP 是其同级同名文件。
- 不发布 EEG、subject、epochs、cache、results、checkpoint、logs、论文或最终实验图。
- 不发布 CSV/JSON/NPY/NPZ/MAT/H5 文件，即使某些文件是配置；改用 INI 声明必要配置。
- 保留科学实现，不造数据、不返回 mock/placeholder 结果，不运行实验。
- 验证限于 Python syntax、import、CLI help、文件/路径及安全扫描；不执行数值回归。
- 不上传 GitHub；不把代码框架可导入描述为论文结果已复现。
- 最終 GITHUB_READY 仅指本地内容及检查就绪，不代表科学复现或已公开发布。

## Review Focus

1. import 是否启动读取 EEG、模型下载、GPU 初始化或写日志：导入前静态确认顶层只有声明和安全导入。
2. 四个 --help 是否必须安装模型库/准备数据才能运行：解析帮助应早于科学模块导入。
3. 路径缺失是否自动跳过受试者或造零特征：数据加载入口必须先失败，不允许以缺数据为由返回结果。
4. generate_figures 是否暗中执行统计/拟合：必须只读用户已有结果并调用真实渲染函数。
5. 原随机基线是否误当作 fake-data 禁令而删掉：保留原科学 null-control 实现，但绝不能把它当缺 EEG/特征的替代品。

## Task 1: 来源清单与安全入口检查

**Files:** 新目录 `docs/CODE_ORIGINS.md`、`docs/check_release.py`。

- [ ] 只读记录归档 ZIP SHA256、全部归档文件哈希和投稿脚本清单，保存在当前工具会话中；发布文档仅写归档相对来源，不写私人路径。
- [ ] 逐个读取 `submission_code/scripts/` 中十个文件，识别顶层 I/O、函数间全局变量依赖、绘图块及真实输入文件布局。未读完整不能机械迁移。
- [ ] 先创建检查器，使用以下断言确认实现前四个入口不存在时检查失败，不调用实验：

```python
from pathlib import Path
release = Path(__file__).resolve().parents[1]
entries = ('run_primary', 'run_temporal', 'run_controls', 'generate_figures')
for entry in entries:
    assert (release / 'scripts' / (entry + '.py')).is_file(), entry
```

- [ ] 检查器后续用 ast.parse 检查语法；逐文件 import 与 subprocess --help 的命令仅在静态确认无顶层副作用后启用。

## Task 2: 原代码副本与配置适配

**Files:** `src/__init__.py`、`src/workflows/` 下十个原同名 `.py`、`src/runtime.py`、`configs/paths.ini`。

**Interfaces:** 每个 workflow 暴露 `run(settings)`；settings 为 `configparser.ConfigParser`。导入模块不得运行 run。runtime 暴露 `load_settings(config_path)` 和 `require_dataset(settings)`。

- [ ] 保留十份实际投稿脚本，逐份将运行语句移入显式调用域，按实际引用关系保留全局/局部变量绑定。不得简单缩进造成函数访问不到原全局参数。
- [ ] 仅替换路径、运行调度和绘图边界；保留 loader/preprocessing/ridge/projection/RDM/RSA/temporal/control 的原语义。
- [ ] 新 INI 的路径契约如下；实际子目录布局在数据文档逐项对应原 loader：

```ini
[paths]
DATA_ROOT = ./data
RESULTS_ROOT = ./results
CONFIG_ROOT = ./configs
IMAGES_ROOT = ./data/images/images_THINGS/object_images
```

- [ ] 实现缺失数据契约；不得自动建 data、返回零数据或跳过缺失 participant：

```python
if not dataset_path.is_dir():
    raise FileNotFoundError(
        'Dataset not found.\n'
        'Please download the required dataset and configure DATA_ROOT.'
    )
```

- [ ] 数据完整性检查使用原脚本所需 EEG/event/image 路径清单，不执行预处理或拟合。原代码缺图零填充/continue 分支须改为明确失败，文档列为输入错误处理变更，不宣称数值验证通过。
- [ ] 去除硬编码个人路径和旧 sys.path 注入；直接脚本调用只添加由 `__file__` 推导的新仓库根目录。不能引用旧项目作为隐藏依赖。
- [ ] 如发现必须改变科学事件对齐/统计逻辑才能运行，停止该改动并报告，不擅自修科学算法。

## Task 3: 四个真实 CLI 与绘图分离

**Files:** `scripts/run_primary.py`、`run_temporal.py`、`run_controls.py`、`generate_figures.py`、`src/visualization.py`、`src/cli.py`。

**Interfaces:** `src.cli.main(kind, argv=None)` 解析参数后调度；workflow.run(settings) 执行真实代码；visualization.render(input_path, output_path) 仅渲染原图形计算以外部分。

- [ ] 四个 CLI 均提供 `--config`，默认定位新仓库 configs/paths.ini；结果根目录由配置读取，允许显式路径参数覆盖。
- [ ] 帮助输出应由标准库 argparse 完成；只有解析结束且数据检查通过后才导入科学依赖。
- [ ] run_primary 调用 `ridge_clip_specificity_v2.run`；run_temporal 调用 `heldout_temporal_fixed_model.run`；run_controls 调用 `partial_rsa_variance_partition.run`。这些是原投稿协议，不伪称后续 10-split/C7 主结果。
- [ ] 原其余七份代码保留为可显式调用 workflow，不用无关新实现替换它们。
- [ ] 从原 numerical plotting 段提取 visualization.render，输入由用户 `--input` 指定；`--output` 默认 results/figures 下。缺输入直接报错，不先运行任何实验。
- [ ] 四个入口都用 `if __name__ == '__main__':` 保护调用，import 不解析命令行或执行流程。
- [ ] 不提供无源码概念图的假生成器，不复制最终图充当生成结果。

## Task 4: 公开说明、许可、环境与有限验证

**Files:** `README.md`、`LICENSE`、`environment/requirements.txt`、`environment/environment.yml`、`.gitignore`、`docs/CODE_ORIGINS.md`、`docs/VALIDATION.md`。

- [ ] README 按 Overview/Installation/Dataset/Usage/Citation 编写，包含用户指定四条 `python scripts/xxx.py` 用法及必要参数说明。
- [ ] Overview 使用 Jointly Constrained Temporal RSA framework；明确本包提供方法代码，不含论文数据/结果，不保证历史数值复现。
- [ ] LICENSE 从项目现有 MIT 完整保留，含 `Copyright (c) 2026 wanshuiyin`；第三方软件不擅自重新授权。
- [ ] requirements 从实际 import 提取，明确 OpenAI CLIP 包来源，不能将错误的同名包列为替代。环境版本标注观察/检查范围，不虚构锁定复现环境。
- [ ] syntax：对全部 Python 文件 ast.parse，不生成 pyc。
- [ ] import：用隔离子进程 `python -B` 导入全部新包/入口；依赖缺失则记录 FAIL，不以 mock module 掩盖。
- [ ] help：用 `python -B scripts/run_primary.py --help` 等四条命令逐个验证 exit 0。不得跑无参数科学入口作为 smoke test。
- [ ] 安全：使用 `git grep --no-index -n -I -E` 对新目录查找敏感赋值、私钥头、私人路径、内部服务器；逐项分类，公开文档不回显秘密值。版权人名不是应删除的 username。
- [ ] 文件白名单允许 .py/.ini/.yaml/.yml/.md/.txt、LICENSE、.gitignore；拒绝数据、二进制产物和历史结果文档。检查 Python 源码也不能内嵌凭据、私有路径或伪结果。
- [ ] VALIDATION.md 记录实际命令和 exit 状态；科学实验状态为 NOT_RUN，而非 PASS。

## Task 5: 打包与只读对比

**Artifact:** 新目录同级 `ICASSP2027_JCT_RSA_CODE_RELEASE.zip`。

- [ ] 再核对原归档文件哈希与 ZIP 哈希，要求无变更。
- [ ] 检查新目录没有 data/results/checkpoints/logs、禁用扩展名、最终图或 PDF。
- [ ] 只将新仓库白名单文件加入 ZIP，不包含 ZIP 本身、pycache、.git、检查运行日志。
- [ ] 只读枚举 ZIP 内容并校验 CRC、文件数量及每个条目与新目录字节一致；不运行实验。
- [ ] 全部有限检查通过后才能报告 CODE_RELEASE=COMPLETE 与 GITHUB_READY=YES；否则报告真实阻塞，不能照抄预期成功状态。

## 审阅门槛

这是实施方案，不是完成报告。当前未迁移/修改科学代码、未生成 ZIP、未验证任何实验。待用户审阅该方案后，在本任务中原生执行，不启动并行代理、不扩大实验范围。
