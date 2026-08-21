r"""
Unified background-task terminal dashboard
====================
Shows the progress of all running background tasks:
  1. DeepSeek temperature-sensitivity experiment (120 matches x 5 samples @ t=0.7)
  2. Local qwen GPU build (WSL llama-cpp-python + RTX 5090)
  3. Local qwen comparison experiment (appears automatically once the build is done)

Usage: cd E:\论文\sci_redo && python src\watch_tasks.py
Refreshes every 5 seconds; Ctrl+C to exit. Read-only, does not interfere with the tasks.
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
    """Parse [X/total] from the log; returns (done, first-line timestamp)."""
    done = 0
    t0 = None
    if os.path.exists(log):
        st = os.stat(log)
        t0 = st.st_ctime
        with open(log, "r", encoding="utf-8", errors="replace") as f:
            for m in re.finditer(r"\[(\d+)/\d+\]", f.read()):
                done = max(done, int(m.group(1)))
    # Supplement with checkpoint-file rows
    ckpt = os.path.join(RES, os.path.basename(log).replace(".log", "_partial.csv"))
    if label == "temp":
        ckpt = TEMP_CKPT
    if os.path.exists(ckpt):
        with open(ckpt, "r", encoding="utf-8", errors="replace") as f:
            ckpt_rows = max(0, sum(1 for _ in f) - 1)
        done = max(done, ckpt_rows)
    return done, t0


def build_status():
    """Check the WSL build status. Returns (status text, whether complete)."""
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
            return "build complete (llama_cpp installed), server can be started", True
        if compilers > 0:
            return f"building ({compilers} compiler processes)", False
        return "build not started or stopped", False
    except Exception as e:
        return f"cannot query WSL: {e}", False


def local_qwen_status():
    """Check the local qwen comparison experiment progress."""
    if not os.path.exists(LOCAL_LOG):
        return None
    done, t0 = read_log_progress(LOCAL_LOG, 50)
    if t0 and done > 0:
        rate = done / max(time.time() - t0, 1)
        eta = (50 - done) / rate
        return f"  {bar(done, 50)}  ETA ~{fmt(eta)}"
    return "  started, waiting for the first sample..."


def main():
    print("=== paper experiment background-task dashboard ===")
    print("(temperature experiment / qwen GPU build / local comparison) Ctrl+C to exit\n")
    while True:
        lines = []
        lines.append(f"[1] DeepSeek temperature experiment (t=0.7, 5 samples, 120 matches)")
        done, t0 = read_log_progress(TEMP_LOG, 120, "temp")
        if done == 0:
            lines.append("  waiting for the first samples (printed every 25 matches)...")
        else:
            rate = done / max(time.time() - t0, 1)
            eta = (120 - done) / rate
            lines.append(f"  {bar(done, 120)}  rate {rate*60:.1f} matches/min, ETA ~{fmt(eta)}")
        if done >= 120:
            lines.append("  ✔ done, results in results/llm_temp07.log")

        lines.append("")
        lines.append("[2] local qwen GPU build (llama-cpp-python + RTX 5090)")
        st, is_done = build_status()
        lines.append(f"  {st}")
        if is_done:
            lines.append("  next: start the server, then run python src/run_llm.py --provider local")

        lines.append("")
        lines.append("[3] local qwen comparison experiment (50 matches)")
        qs = local_qwen_status()
        lines.append(qs if qs else "  not started (waiting for the build)")

        sys.stdout.write("\033[2J\033[H")  # clear screen and return to top
        sys.stdout.write("\n".join(lines) + "\n")
        sys.stdout.flush()
        time.sleep(5)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nexiting.")
