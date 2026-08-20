# REFRESH_CHECKLIST.md - 2025/26 测试季结束后一键刷新指南

> 论文摘要已如实声明："The test season (2025/26) is incomplete (through
> 2026-02-12): financial point estimates may shift when the season completes"。
> 本清单就是兑现这个承诺的机制。全部数字由 `refresh_all.cmd` 从
> results/*.json 自动重建，**禁止手写数字**（红线，见 EXPERIMENT_PLAN.md §9）。

## 触发时机

2025/26 赛季结束后（约 2026-05 底），football-data.co.uk 五大联赛 CSV
更新完整后。

## 步骤

1. **更新原始数据**
   - 从 football-data.co.uk 下载最新 E0/D1/F1/I1/SP1 CSV，覆盖
     `E:\论文\structured_data\`（保持同目录、同文件名可被 glob 读到）
   - 校验：`(Get-ChildItem E:\论文\structured_data\*.csv | Get-Content | Measure-Object -Line)`
     或直接跑下一步，data_pipeline 会打印 train/val/test 场次数

2. **（可选）更新 Understat xG**
   - 若需要 2025/26 后半季 xG：`python src\crawl_understat.py`
     （需要网络；若跳过，v3 特征对新增场次会以 train 中位数填充，
     匹配率下降——见 augment_xg.py 输出）

3. **一键刷新**（双击运行，约 20-40 分钟）
   ```
   refresh_all.cmd
   ```
   依次跑：data_pipeline → augment_xg → 12 个实验脚本 → make_figures → make_tables

4. **核对数字变化**
   - 重点看 `results/baselines_summary.json`（市场 acc/logloss/ECE/ROI）
     和 `results/risk_analysis.json`（UI 分层：low/medium/high acc、ROI）
   - 与 git 里的旧 results/ 对比（工作仓库有版本历史，或手动 diff）
   - 确认叙事是否成立：
     - 市场不可超越（acc ~54%）
     - UI 分层单调（low acc 最高、high ROI 最亏）
     - 错误分析结论稳定（分歧场次双方同准、异动被大热混杂）

5. **更新摘要措辞**（手动，唯一手写处）
   - `paper/sections/abstract.tex`：删除
     "The test season (2025/26) is incomplete (through 2026-02-12)..."
     一句，改为 "on the complete 2025/26 test season (N 场)"，
     N 以 `results/baselines_summary.json` 实际场次为准
   - 同步检查 intro/conclusion 里 "incomplete/through 2026-02-12"
     相关表述（grep 全局确认）

6. **（可选）重跑 LLM 实验**
   - LLM 全量（1104x3，约 $0.55）：`python src\run_llm.py`
   - 完成后重跑 `python src\make_tables.py` 刷新表 15/16
   - 若测试季场次变化，LLM 实验数量也变，成本随之变化

7. **重新编译论文**
   - Overleaf：重新上传 paper_overleaf.zip（或本地 TeX 编译）
   - 检查所有表格数字与 results/*.json 一致（make_tables 保证）

8. **同步公开仓库**
   - `E:\论文\sci_redo\publish\` 需同步 src/ 与 results/（不含 key）
   - `git add -A && git commit -m "refresh test season 2025/26" && git tag v1.1.0`
   - push 前记得开代理（http://127.0.0.1:10090），大包加
     `git config http.postBuffer 524288000`（上次推送坑，见 2026-08-19 记录）

## 红线提醒

- 任何数字必须由脚本产出；refresh_all.cmd 已强制全链路
- 结果好坏如实报告，不挑最好情况——刷新后若市场基线更强/UI 分层变弱，
  如实写进论文，不要为了保叙事改阈值
- API key 文件（llm-config.local.json / embed-config.local.json）不进
  publish/ 仓库（.gitignore 已排除）
