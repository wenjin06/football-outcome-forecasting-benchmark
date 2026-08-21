"""Merge main.tex + sections + tables into a single tex file for pandoc-to-docx conversion."""
import re
import sys
import os

import paths
BASE = paths.PAPER


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
        # Try paths under sections/ and tables/ relative to main
        p2 = os.path.join(BASE, inc)
        if os.path.exists(p2):
            return expand(p2, depth + 1, seen)
        return m.group(0)
    text = re.sub(r"\\input\{([^}]+)\}", repl, text)
    # Strip bibliography commands (pandoc citeproc takes over)
    text = re.sub(r"\\bibliographystyle\{[^}]*\}", "", text)
    text = re.sub(r"\\bibliography\{[^}]*\}", "", text)
    return text


def main():
    merged = expand(os.path.join(BASE, "main.tex"))
    # The document class and packages are useless to pandoc and may interfere; pandoc ignores the preamble, so keep them
    out = os.path.join(BASE, "_merged_for_docx.tex")
    with open(out, "w", encoding="utf-8") as f:
        f.write(merged)
    print("merged ->", out, len(merged), "chars")


if __name__ == "__main__":
    main()
