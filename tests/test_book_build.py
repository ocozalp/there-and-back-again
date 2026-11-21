import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("build_pdf", REPO / "docs/book/build_pdf.py")
build_pdf = importlib.util.module_from_spec(spec)
sys.modules["build_pdf"] = build_pdf
spec.loader.exec_module(build_pdf)


def rewriter_for(rel: str, prefix: str):
    part = build_pdf.Part(REPO / rel, prefix)
    front, chapters, back = build_pdf.book_parts()
    prefixes = {p.path.resolve(): p.prefix for p in front + chapters + back}
    return build_pdf.Rewriter(part, prefixes)


def test_book_includes_every_chapter_in_order():
    _, chapters, back = build_pdf.book_parts()
    names = [p.path.parent.name for p in chapters]
    assert names == sorted(names)
    assert {"00-skeleton", "01-autograd", "02-pytorch-mlp"} <= set(names)
    assert [p.prefix for p in back] == ["glossary", "reading-list"]


def test_links_between_docs_become_internal_anchors():
    r = rewriter_for("docs/01-autograd/README.md", "ch-01-autograd")
    assert r.link_target("../glossary.md") == "#glossary"
    assert r.link_target("../00-skeleton/README.md#fundamentals") == "#ch-00-skeleton-fundamentals"
    assert r.link_target("#exit-criteria") == "#ch-01-autograd-exit-criteria"


def test_code_links_point_at_github_and_external_links_stay():
    r = rewriter_for("docs/01-autograd/README.md", "ch-01-autograd")
    github = build_pdf.GITHUB
    assert r.link_target("../../src/dl/autograd/engine.py") == (
        f"{github}/blob/main/src/dl/autograd/engine.py"
    )
    assert r.link_target("../../src/dl/autograd") == f"{github}/tree/main/src/dl/autograd"
    assert r.link_target("https://arxiv.org/abs/2106.09685") == "https://arxiv.org/abs/2106.09685"


def test_headers_get_chapter_prefixed_ids_and_title_override():
    part = build_pdf.Part(REPO / "README.md", "introduction", title="Introduction")
    r = build_pdf.Rewriter(part, {})
    title = r.walk({"t": "Header", "c": [1, ["long-title", [], []], [{"t": "Str", "c": "x"}]]})
    section = r.walk({"t": "Header", "c": [2, ["layout", [], []], []]})
    assert title["c"][1][0] == "introduction"
    assert title["c"][2] == [{"t": "Str", "c": "Introduction"}]
    assert section["c"][1][0] == "introduction-layout"


def test_breakable_code_escapes_latex_and_allows_breaks():
    out = build_pdf.breakable_code("nn.CrossEntropy(x_1)")
    assert out.startswith(r"\texttt{nn.\allowbreak{}Cross\allowbreak{}Entropy(")
    assert r"x\_\allowbreak{}1)" in out
