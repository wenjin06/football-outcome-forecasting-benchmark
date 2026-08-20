"""合并 main.tex + sections + tables 为单文件 tex，供 pandoc 转 docx。"""
import re
import sys
import os

BASE = r"E:\论文\sci_redo\paper"


def expand(path, depth=0, seen=None):
    if seen is None:
        seen = set()
    path = os.path.normpath(path)
    if path in seen or depth > 6:
        return ""
    seen.add(path)
    with open(path, encoding="utf-8") as f:
        text = f.read()
    def repl(m):
        inc = m.group(1).strip()
        if not inc.endswith(".tex"):
            inc += ".tex"
        p = os.path.join(os.path.dirname(path), inc)
        if os.path.exists(p):
            return expand(p, depth + 1, seen)
        # 尝试 sections/ tables/ 相对 main 的路径
        p2 = os.path.join(BASE, inc)
        if os.path.exists(p2):
            return expand(p2, depth + 1, seen)
        return m.group(0)
    text = re.sub(r"\\input\{([^}]+)\}", repl, text)
    # 移除参考文献命令（pandoc citeproc 接管）
    text = re.sub(r"\\bibliographystyle\{[^}]*\}", "", text)
    text = re.sub(r"\\bibliography\{[^}]*\}", "", text)
    return text


def main():
    merged = expand(os.path.join(BASE, "main.tex"))
    # 文档类与宏包对 pandoc 无用且可能干扰：pandoc 会忽略 preamble，保留即可
    out = os.path.join(BASE, "_merged_for_docx.tex")
    with open(out, "w", encoding="utf-8") as f:
        f.write(merged)
    print("merged ->", out, len(merged), "chars")


if __name__ == "__main__":
    main()
