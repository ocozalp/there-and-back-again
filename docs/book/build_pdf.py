"""Compile the README, conventions, every chapter, glossary and reading list into one PDF.

Usage: uv run --group docs python docs/book/build_pdf.py

Needs `tectonic` (LaTeX engine) and `mmdc` (Mermaid CLI) on PATH; pandoc comes from the
`pypandoc-binary` package in the `docs` dependency group.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote

REPO = Path(__file__).resolve().parents[2]
DOCS = REPO / "docs"
BOOK = DOCS / "book"
CACHE = BOOK / ".cache"
OUTPUT = DOCS / "there-and-back-again.pdf"
GITHUB = "https://github.com/ocozalp/there-and-back-again"
CHAPTER_DIR = re.compile(r"^\d\d-")
URL_SCHEME = re.compile(r"^[a-z][a-z0-9+.-]*:", re.I)


@dataclass(frozen=True)
class Part:
    path: Path
    prefix: str
    title: str | None = None


def book_parts(docs: Path = DOCS) -> tuple[list[Part], list[Part], list[Part]]:
    front = [
        Part(docs.parent / "README.md", "introduction", title="Introduction"),
        Part(docs / "00-conventions.md", "conventions"),
    ]
    chapters = [
        Part(d / "README.md", f"ch-{d.name}")
        for d in sorted(docs.iterdir())
        if d.is_dir() and CHAPTER_DIR.match(d.name) and (d / "README.md").exists()
    ]
    back = [Part(docs / "glossary.md", "glossary"), Part(docs / "reading-list.md", "reading-list")]
    return front, chapters, back


def str_inlines(text: str) -> list[dict]:
    out: list[dict] = []
    for i, word in enumerate(text.split(" ")):
        if i:
            out.append({"t": "Space"})
        out.append({"t": "Str", "c": word})
    return out


class MermaidRenderer:
    def __init__(self, cache: Path):
        self.cache = cache

    def render(self, source: str) -> Path:
        self.cache.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256(source.encode()).hexdigest()[:16]
        out = self.cache / f"mermaid-{digest}.pdf"
        if out.exists():
            return out
        src = self.cache / f"mermaid-{digest}.mmd"
        src.write_text(source)
        cmd = ["mmdc", "-i", str(src), "-o", str(out), "--pdfFit", "-t", "neutral", "-b", "white"]
        result = subprocess.run(cmd, capture_output=True, text=True)
        src.unlink()
        if result.returncode != 0:
            raise RuntimeError(f"mmdc failed:\n{result.stderr}")
        return out


class Rewriter:
    """Makes one markdown file's pandoc AST safe to merge into the book."""

    def __init__(
        self,
        part: Part,
        prefixes: dict[Path, str],
        mermaid: MermaidRenderer | None = None,
        repo: Path = REPO,
    ):
        self.part = part
        self.base = part.path.parent
        self.prefixes = prefixes
        self.mermaid = mermaid
        self.repo = repo
        self.seen_title = False

    @staticmethod
    def anchor(prefix: str, fragment: str) -> str:
        return f"#{prefix}-{fragment}" if fragment else f"#{prefix}"

    def link_target(self, url: str) -> str:
        if URL_SCHEME.match(url):
            return url
        if url.startswith("#"):
            return self.anchor(self.part.prefix, url[1:])
        path_part, _, fragment = url.partition("#")
        target = (self.base / unquote(path_part)).resolve()
        if target in self.prefixes:
            return self.anchor(self.prefixes[target], fragment)
        try:
            rel = target.relative_to(self.repo.resolve())
        except ValueError:
            return url
        kind = "tree" if target.is_dir() else "blob"
        suffix = f"#{fragment}" if fragment else ""
        return f"{GITHUB}/{kind}/main/{rel.as_posix()}{suffix}"

    def image_target(self, url: str) -> str:
        if URL_SCHEME.match(url):
            return url
        return str((self.base / unquote(url)).resolve())

    def walk(self, node):
        if isinstance(node, list):
            return [self.walk(item) for item in node]
        if not isinstance(node, dict) or "t" not in node:
            return node
        kind = node["t"]
        if kind == "Header":
            return self.header(node)
        if kind == "CodeBlock" and "mermaid" in node["c"][0][1] and self.mermaid:
            pdf = self.mermaid.render(node["c"][1])
            return {"t": "Para", "c": [{"t": "Image", "c": [["", [], []], [], [str(pdf), ""]]}]}
        if kind == "Code" and len(node["c"][1]) > 15:
            return {"t": "RawInline", "c": ["latex", breakable_code(node["c"][1])]}
        if kind == "Table":
            return self.table(node)
        if kind == "Para":
            return {"t": "Para", "c": self.image_row(self.walk(node["c"]))}
        if kind in ("Link", "Image"):
            attr, inlines, (url, title) = node["c"]
            fix = self.link_target if kind == "Link" else self.image_target
            return {"t": kind, "c": [attr, self.walk(inlines), [fix(url), title]]}
        if "c" in node:
            return {"t": kind, "c": self.walk(node["c"])}
        return node

    @staticmethod
    def image_row(inlines: list) -> list:
        images = [i for i in inlines if i.get("t") == "Image"]
        others = [i for i in inlines if i.get("t") not in ("Image", "Space", "SoftBreak")]
        if len(images) < 2 or others:
            return inlines
        width = f"{96 // len(images)}%"
        for image in images:
            image["c"][0][2] = [kv for kv in image["c"][0][2] if kv[0] != "width"] + [
                ["width", width]
            ]
        return [i if i.get("t") != "SoftBreak" else {"t": "Space"} for i in inlines]

    def table(self, node: dict) -> dict:
        colspecs = node["c"][2]
        lengths = [0] * len(colspecs)
        words = [0] * len(colspecs)
        for row in table_rows(node):
            for i, cell in enumerate(row[1][: len(lengths)]):
                text = plain_text(cell[4])
                lengths[i] = max(lengths[i], len(text))
                parts = re.split(r"[\s._(,/=]+", text)
                words[i] = max([words[i]] + [len(w) for w in parts])
        if sum(lengths) > 80:
            weights = [max(n, 1.4 * w + 2) for n, w in zip(lengths, words, strict=True)]
            total = sum(weights)
            for spec, weight in zip(colspecs, weights, strict=True):
                spec[1] = {"t": "ColWidth", "c": weight / total}
        return {"t": "Table", "c": self.walk(node["c"])}

    def header(self, node: dict) -> dict:
        level, (ident, classes, kvs), inlines = node["c"]
        if level == 1 and not self.seen_title:
            self.seen_title = True
            ident = self.part.prefix
            if self.part.title:
                inlines = str_inlines(self.part.title)
        elif ident:
            ident = f"{self.part.prefix}-{ident}"
        return {"t": "Header", "c": [level, [ident, classes, kvs], self.walk(inlines)]}


LATEX_ESCAPES = {
    "\\": r"\textbackslash{}",
    "{": r"\{",
    "}": r"\}",
    "_": r"\_",
    "^": r"\^{}",
    "%": r"\%",
    "&": r"\&",
    "#": r"\#",
    "$": r"\$",
    "~": r"\textasciitilde{}",
    " ": r"\ ",
    "'": r"\textquotesingle{}",
}


def breakable_code(code: str) -> str:
    """Inline code that LaTeX may break after punctuation or between camel-case words."""
    out = []
    for prev, ch in zip(" " + code, code, strict=False):
        if prev.islower() and ch.isupper():
            out.append(r"\allowbreak{}")
        out.append(LATEX_ESCAPES.get(ch, ch))
        if ch in "._(,/=":
            out.append(r"\allowbreak{}")
    return r"\texttt{" + "".join(out) + "}"


def plain_text(node) -> str:
    if isinstance(node, list):
        return "".join(plain_text(n) for n in node)
    if not isinstance(node, dict):
        return ""
    kind = node.get("t")
    if kind in ("Str", "Code", "Math"):
        return node["c"] if kind == "Str" else node["c"][1]
    if kind in ("Space", "SoftBreak"):
        return " "
    return plain_text(node.get("c", []))


def table_rows(table: dict) -> list:
    _, _, _, head, bodies, foot = table["c"]
    rows = list(head[1]) + list(foot[1])
    for body in bodies:
        rows += body[2] + body[3]
    return rows


def to_ast(pandoc: str, path: Path) -> dict:
    result = subprocess.run(
        [pandoc, "-f", "gfm+tex_math_dollars", "-t", "json", str(path)],
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(result.stdout)


def raw_latex(text: str) -> dict:
    return {"t": "RawBlock", "c": ["latex", text]}


def git_commit() -> str:
    result = subprocess.run(
        ["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], capture_output=True, text=True
    )
    return result.stdout.strip() or "unknown"


def build(output: Path, keep_tex: bool) -> None:
    for tool in ("tectonic", "mmdc"):
        if shutil.which(tool) is None:
            sys.exit(f"{tool} not on PATH: brew install tectonic; npm i -g @mermaid-js/mermaid-cli")
    import pypandoc

    pandoc = pypandoc.get_pandoc_path()
    front, chapters, back = book_parts()
    prefixes = {p.path.resolve(): p.prefix for p in front + chapters + back}
    mermaid = MermaidRenderer(CACHE)
    api_version = None

    def blocks_for(group: list[Part]) -> list[dict]:
        nonlocal api_version
        out: list[dict] = []
        for part in group:
            ast = to_ast(pandoc, part.path)
            api_version = ast["pandoc-api-version"]
            out.extend(Rewriter(part, prefixes, mermaid).walk(ast["blocks"]))
        return out

    blocks = (
        [raw_latex("\\frontmatter")]
        + blocks_for(front)
        + [raw_latex("\\mainmatter")]
        + blocks_for(chapters)
        + [raw_latex("\\appendix")]
        + blocks_for(back)
    )
    CACHE.mkdir(parents=True, exist_ok=True)
    merged = CACHE / "book.json"
    merged.write_text(json.dumps({"pandoc-api-version": api_version, "meta": {}, "blocks": blocks}))

    common = [
        pandoc,
        "-f",
        "json",
        str(merged),
        "--standalone",
        "--toc",
        "--toc-depth=1",
        "--top-level-division=chapter",
        "--syntax-highlighting=tango",
        "--include-in-header",
        str(BOOK / "preamble.tex"),
        "--metadata-file",
        str(BOOK / "metadata.yaml"),
        "-M",
        f"date={dt.date.today().isoformat()}, commit {git_commit()}",
    ]
    if keep_tex:
        subprocess.run(common + ["-o", str(CACHE / "book.tex")], check=True)
    subprocess.run(common + ["--pdf-engine=tectonic", "-o", str(output)], check=True)
    print(f"wrote {output} ({output.stat().st_size / 1e6:.1f} MB)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("-o", "--output", type=Path, default=OUTPUT)
    parser.add_argument("--keep-tex", action="store_true", help="also write .cache/book.tex")
    args = parser.parse_args()
    build(args.output.resolve(), args.keep_tex)


if __name__ == "__main__":
    main()
