r"""
统一任务终端仪表盘
====================
显示所有正在运行的后台任务进度：
  1. DeepSeek 温度敏感性实验（120 场 x 5 采样 @ t=0.7）
  2. 本地 qwen GPU 版编译（WSL llama-cpp-python + RTX 5090）
  3. 本地 qwen 对照实验（编译完成后启动时自动出现）

用法：cd E:\论文\sci_redo && python src\watch_tasks.py
每 5 秒刷新，Ctrl+C 退出。只读，不干扰任务。
"""
import os
import re
import subprocess
import sys
import time

BASE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(BASE, "..", "results")

TEMP_LOG = os.path.join(RES, "llm_temp07.log")
TEMP_CKPT = os.path.join(RES, "llm_deepseek_t0.7_s5_partial.csv")
LOCAL_LOG = os.path.join(RES, "llm_local.log")

WIDTH = 36


def fmt(sec):
    sec = max(0, int(sec))
    h, m = divmod(sec // 60, 60)
    return f"{h}h{m:02d}m" if h else f"{m}m{sec % 60:02d}s"


def bar(done, total):
    pct = done / total if total else 0
    filled = int(pct * WIDTH)
    return f"[{'=' * filled}{'>' if filled < WIDTH else ''}{' ' * (WIDTH - filled - 1)}] {done:4d}/{total} ({pct*100:5.1f}%)"


def read_log_progress(log, total, label=""):
    """从日志解析 [X/total]，返回 (done, 首行时间戳)。"""
    done = 0
    t0 = None
    if os.path.exists(log):
        st = os.stat(log)
        t0 = st.st_ctime
        with open(log, "r", encoding="utf-8", errors="replace") as f:
            for m in re.finditer(r"\[(\d+)/\d+\]", f.read()):
                done = max(done, int(m.group(1)))
    # 断点文件行数补充
    ckpt = os.path.join(RES, os.path.basename(log).replace(".log", "_partial.csv"))
    if label == "temp":
        ckpt = TEMP_CKPT
    if os.path.exists(ckpt):
        with open(ckpt, "r", encoding="utf-8", errors="replace") as f:
            ckpt_rows = max(0, sum(1 for _ in f) - 1)
        done = max(done, ckpt_rows)
    return done, t0


def build_status():
    """检查 WSL 编译状态。返回 (状态文本, 是否完成)。"""
    try:
        out = subprocess.run(
            ["wsl", "bash", "-c",
             "ls ~/llama-venv/lib/python*/site-packages/ 2>/dev/null | grep -c llama_cpp; "
             "ps aux 2>/dev/null | grep -cE 'cc1plus|ninja|cmake' | head -1"],
            capture_output=True, text=True, timeout=20)
        lines = out.stdout.strip().split("\n")
        has_module = int(lines[0].strip() or 0) > 0 if lines else False
        compilers = int(lines[1].strip() or 0) if len(lines) > 1 else 0
        if has_module:
            return "编译完成（llama_cpp 已安装），可启动服务", True
        if compilers > 0:
            return f"编译中（{compilers} 个编译进程）", False
        return "编译未启动或已停止", False
    except Exception as e:
        return f"无法查询 WSL: {e}", False


def local_qwen_status():
    """检查本地 qwen 对照实验进度。"""
    if not os.path.exists(LOCAL_LOG):
        return None
    done, t0 = read_log_progress(LOCAL_LOG, 50)
    if t0 and done > 0:
        rate = done / max(time.time() - t0, 1)
        eta = (50 - done) / rate
        return f"  {bar(done, 50)}  剩余约 {fmt(eta)}"
    return "  已启动，等待首个样本..."


def main():
    print("=== 论文实验后台任务仪表盘 ===")
    print("（温度实验 / qwen GPU 编译 / 本地对照） Ctrl+C 退出\n")
    while True:
        lines = []
        lines.append(f"[1] DeepSeek 温度实验 (t=0.7, 5 采样, 120 场)")
        done, t0 = read_log_progress(TEMP_LOG, 120, "temp")
        if done == 0:
            lines.append("  等待首批样本（第 25 场才打印）...")
        else:
            rate = done / max(time.time() - t0, 1)
            eta = (120 - done) / rate
            lines.append(f"  {bar(done, 120)}  速率 {rate*60:.1f} 场/分  剩余约 {fmt(eta)}")
        if done >= 120:
            lines.append("  ✔ 已完成，结果见 results/llm_temp07.log")

        lines.append("")
        lines.append("[2] 本地 qwen GPU 编译（llama-cpp-python + RTX 5090）")
        st, is_done = build_status()
        lines.append(f"  {st}")
        if is_done:
            lines.append("  下一步：启动服务后运行 python src/run_llm.py --provider local")

        lines.append("")
        lines.append("[3] 本地 qwen 对照实验（50 场）")
        qs = local_qwen_status()
        lines.append(qs if qs else "  未启动（等编译完成）")

        sys.stdout.write("\033[2J\033[H")  # 清屏回到顶部
        sys.stdout.write("\n".join(lines) + "\n")
        sys.stdout.flush()
        time.sleep(5)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n退出。")
