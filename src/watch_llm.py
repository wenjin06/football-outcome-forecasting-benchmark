"""
LLM experiment terminal progress bar (read-only, does not interfere with the run)
Usage: run  python src\watch_llm.py  from the repository root
Refreshes every 5 seconds; Ctrl+C to exit. Progress source: results/llm_run_full.log + checkpoint file.
"""
import os
import re
import sys
import time
import paths

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
        t0 = st.st_ctime  # file creation time (approximately the experiment start)
        with open(LOG, "r", encoding="utf-8", errors="replace") as f:
            for m in PAT.finditer(f.read()):
                done = max(done, int(m.group(1)))
    # Supplement with checkpoint-file rows (saved every 50 matches)
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
    print("LLM full experiment progress (1104 matches x 3 samples, DeepSeek)")
    print("press Ctrl+C to exit\n")
    while True:
        done, t0 = read_progress()
        pct = done / TOTAL
        bar = "=" * int(pct * width) + ">" + " " * (width - int(pct * width) - 1)
        line = f"[{bar}] {done:4d}/{TOTAL} ({pct*100:5.1f}%)"
        if t0 and done > 50:
            rate = done / max(time.time() - t0, 1)
            eta = (TOTAL - done) / rate
            line += f"  rate {rate:.2f} matches/min, ETA {fmt(eta)}"
        elif done == 0:
            line += "  waiting to start..."
        sys.stdout.write("\r" + line + " " * 10)
        sys.stdout.flush()
        if done >= TOTAL:
            print("\n\nexperiment complete! Final results in results/llm_deepseek.json")
            break
        time.sleep(5)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nexiting.")
