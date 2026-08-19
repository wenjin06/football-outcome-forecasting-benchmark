"""
LLM 实验终端进度条（只读，不干扰运行）
用法：在 E:\论文\sci_redo 下执行  python src\watch_llm.py
每 5 秒刷新一次，Ctrl+C 退出。进度来源：results/llm_run_full.log + 断点文件。
"""
import os
import re
import sys
import time

BASE = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(BASE, "..", "results", "llm_run_full.log")
CKPT = os.path.join(BASE, "..", "results", "llm_deepseek_per_match_partial.csv")
TOTAL = 1104

PAT = re.compile(r"\[(\d+)/1104\]")


def read_progress():
    done = 0
    t0 = None
    if os.path.exists(LOG):
        st = os.stat(LOG)
        t0 = st.st_ctime  # 文件创建时间（约等于实验开始）
        with open(LOG, "r", encoding="utf-8", errors="replace") as f:
            for m in PAT.finditer(f.read()):
                done = max(done, int(m.group(1)))
    # 断点文件行数补充（每 50 场一存）
    ckpt = 0
    if os.path.exists(CKPT):
        with open(CKPT, "r", encoding="utf-8", errors="replace") as f:
            ckpt = max(0, sum(1 for _ in f) - 1)
    done = max(done, ckpt)
    return done, t0


def fmt(sec):
    sec = max(0, int(sec))
    h, m = divmod(sec // 60, 60)
    return f"{h}h{m:02d}m" if h else f"{m}m{sec % 60:02d}s"


def main():
    width = 40
    print("LLM 全量实验进度（1104 场 x 3 采样，DeepSeek）")
    print("按 Ctrl+C 退出\n")
    while True:
        done, t0 = read_progress()
        pct = done / TOTAL
        bar = "=" * int(pct * width) + ">" + " " * (width - int(pct * width) - 1)
        line = f"[{bar}] {done:4d}/{TOTAL} ({pct*100:5.1f}%)"
        if t0 and done > 50:
            rate = done / max(time.time() - t0, 1)
            eta = (TOTAL - done) / rate
            line += f"  速率 {rate:.2f} 场/分  预计剩余 {fmt(eta)}"
        elif done == 0:
            line += "  等待启动..."
        sys.stdout.write("\r" + line + " " * 10)
        sys.stdout.flush()
        if done >= TOTAL:
            print("\n\n实验已完成！最终结果见 results/llm_deepseek.json")
            break
        time.sleep(5)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n退出。")
