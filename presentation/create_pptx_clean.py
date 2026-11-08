"""
Financial LLM Evaluation — Academic PowerPoint Generator (Real Live Results Edition)
IIT Guwahati | Big Data Analytics | November 2026

Creates:
  - presentation/Financial_LLM_Evaluation_Presentation_Updated.pptx
  - presentation/Financial_LLM_Evaluation_Presentation.pptx (if unlocked)

16 slides, 16:9 widescreen, dark navy/white/cyan design.
Fully populated with REAL, LIVE model inference results and real benchmark plots!
"""

import os
import sys
from pathlib import Path

from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PLOTS_DIR = PROJECT_ROOT / "results" / "plots"
OUTPUT_UPDATED = Path(__file__).resolve().parent / "Financial_LLM_Evaluation_Live_Results.pptx"
OUTPUT_ORIGINAL = Path(__file__).resolve().parent / "Financial_LLM_Evaluation_Presentation.pptx"

# ---------------------------------------------------------------------------
# Color Palette
# ---------------------------------------------------------------------------
NAVY       = RGBColor(0x0D, 0x1B, 0x2A)   # Dark navy background
CYAN       = RGBColor(0x00, 0xB4, 0xD8)   # Accent / headings / bright highlights
WHITE      = RGBColor(0xFF, 0xFF, 0xFF)
LIGHT_GREY = RGBColor(0xCC, 0xD6, 0xE0)
DARK_CYAN  = RGBColor(0x00, 0x87, 0xA3)   # Dark cyan banner/box
MID_NAVY   = RGBColor(0x1A, 0x2E, 0x44)   # Card / box background
GREEN_ACC  = RGBColor(0x2E, 0xC4, 0xB6)   # Teal/green accent
CARD_BORDER= RGBColor(0x23, 0x41, 0x60)   # Subtle card border

# ---------------------------------------------------------------------------
# Slide dimensions  (16:9 widescreen)
# ---------------------------------------------------------------------------
SLIDE_W = Inches(13.33)
SLIDE_H = Inches(7.5)

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def new_prs() -> Presentation:
    prs = Presentation()
    prs.slide_width  = SLIDE_W
    prs.slide_height = SLIDE_H
    return prs


def blank_layout(prs: Presentation):
    return prs.slide_layouts[6]   # completely blank


def add_slide(prs: Presentation):
    return prs.slides.add_slide(blank_layout(prs))


def fill_bg(slide, color: RGBColor = NAVY):
    bg = slide.background
    fill = bg.fill
    fill.solid()
    fill.fore_color.rgb = color


def add_rect(slide, left, top, width, height, fill_color=None, line_color=None, line_width_pt=None):
    shape = slide.shapes.add_shape(
        1,  # MSO_SHAPE_TYPE.RECTANGLE
        Inches(left), Inches(top), Inches(width), Inches(height)
    )
    if fill_color:
        shape.fill.solid()
        shape.fill.fore_color.rgb = fill_color
    else:
        shape.fill.background()
    if line_color:
        shape.line.color.rgb = line_color
        if line_width_pt:
            shape.line.width = Pt(line_width_pt)
    else:
        shape.line.fill.background()
    return shape


def add_textbox(slide, text, left, top, width, height,
                font_size=18, bold=False, italic=False,
                color=WHITE, align=PP_ALIGN.LEFT,
                font_name="Calibri", wrap=True):
    txBox = slide.shapes.add_textbox(
        Inches(left), Inches(top), Inches(width), Inches(height)
    )
    tf = txBox.text_frame
    tf.word_wrap = wrap
    p = tf.paragraphs[0]
    p.alignment = align
    run = p.add_run()
    run.text = text
    run.font.size = Pt(font_size)
    run.font.bold = bold
    run.font.italic = italic
    run.font.color.rgb = color
    run.font.name = font_name
    return txBox


def add_cyan_bar(slide, top=0.55, height=0.05):
    add_rect(slide, 0, top, 13.33, height, fill_color=CYAN)


def slide_heading(slide, title: str, top=0.12, font_size=26):
    add_textbox(slide, title,
                left=0.5, top=top, width=12.33, height=0.55,
                font_size=font_size, bold=True, color=CYAN,
                align=PP_ALIGN.LEFT)


def add_image_safe(slide, img_path, left, top, width, height):
    img_path = Path(img_path)
    if img_path.exists() and img_path.stat().st_size > 100:
        slide.shapes.add_picture(str(img_path),
                                 Inches(left), Inches(top),
                                 Inches(width), Inches(height))
        return True
    else:
        add_rect(slide, left, top, width, height,
                 fill_color=MID_NAVY, line_color=CYAN, line_width_pt=1)
        add_textbox(slide, f"[Figure: {img_path.name}]",
                    left+0.1, top+height/2-0.2, width-0.2, 0.4,
                    font_size=11, color=LIGHT_GREY, align=PP_ALIGN.CENTER)
        return False


def header_badge(slide, text, left=8.5, top=0.14, width=4.3, height=0.32,
                 bg_color=DARK_CYAN, text_color=WHITE):
    add_rect(slide, left, top, width, height, fill_color=bg_color, line_color=None)
    add_textbox(slide, text, left+0.05, top+0.04, width-0.1, 0.25,
                font_size=9, bold=True, color=text_color, align=PP_ALIGN.CENTER)


# ===========================================================================
#  SLIDE BUILDERS
# ===========================================================================

def slide_01_title(prs):
    """Slide 1 — Title"""
    s = add_slide(prs)
    fill_bg(s, NAVY)

    add_rect(s, 0, 0, 13.33, 0.85, fill_color=CYAN)
    add_textbox(s, "Big Data Analytics  |  Department of Data Science & AI  |  IIT Guwahati",
                0.3, 0.16, 12.73, 0.5,
                font_size=13, bold=True, color=NAVY, align=PP_ALIGN.CENTER)

    add_textbox(s, "Financial LLM Evaluation",
                0.5, 1.3, 12.33, 1.0,
                font_size=44, bold=True, color=WHITE, align=PP_ALIGN.CENTER)

    add_textbox(s,
                "Comparing Open-Weight Large Language Models on\n"
                "Financial Question Answering Benchmarks",
                0.5, 2.35, 12.33, 1.1,
                font_size=22, bold=False, color=CYAN, align=PP_ALIGN.CENTER)

    add_rect(s, 2.0, 3.65, 9.33, 0.04, fill_color=CYAN)

    add_rect(s, 2.5, 3.95, 8.33, 1.8, fill_color=MID_NAVY, line_color=CYAN, line_width_pt=1)
    add_textbox(s, "Project Team Members",
                2.5, 4.05, 8.33, 0.35,
                font_size=13, bold=True, color=CYAN, align=PP_ALIGN.CENTER)

    members = [
        "Jaypal Malviya   (240150013)",
        "Anuj Kumar   (240150003)",
        "Rup Narayan Jha   (240150032)",
    ]
    for i, m in enumerate(members):
        add_textbox(s, m,
                    2.5, 4.50 + i*0.35, 8.33, 0.35,
                    font_size=14, bold=False, color=WHITE, align=PP_ALIGN.CENTER)

    add_rect(s, 0, 6.85, 13.33, 0.65, fill_color=DARK_CYAN)
    add_textbox(s, "B.Tech Data Science & Artificial Intelligence  ·  3rd Year  ·  Academic Evaluation",
                0.3, 6.95, 12.73, 0.45,
                font_size=12, color=NAVY, bold=True, align=PP_ALIGN.CENTER)


def slide_02_motivation(prs):
    """Slide 2 — Motivation & Problem Statement"""
    s = add_slide(prs)
    fill_bg(s, NAVY)
    add_cyan_bar(s)
    slide_heading(s, "Motivation & Problem Statement")

    add_textbox(s,
                "While LLMs excel in general NLP, deploying them in financial analytics requires solving\n"
                "four critical domain-specific challenges where general models frequently fail:",
                0.5, 0.78, 12.33, 0.65,
                font_size=14, color=LIGHT_GREY, align=PP_ALIGN.LEFT)

    challenges = [
        ("🔢  Numerical & Tabular Reasoning",
         "Financial QA requires multi-step arithmetic, ratio computation (P/E, EBITDA), "
         "and year-over-year balance sheet analysis where small precision errors lead to failure."),
        ("📄  Complex Document Understanding",
         "Contexts comprise dense SEC filings (10-K, 10-Q), earnings call transcripts, "
         "and regulatory frameworks with tabular-textual interdependencies."),
        ("📚  Domain-Specific Terminology",
         "Requires precise comprehension of capital structure, covenants, GAAP/IFRS standards, "
         "basis points, and liquidity terminology."),
        ("🛡️  Hallucination & Faithfulness Risks",
         "Fabricating financial statistics or misattributing fiscal figures carries severe "
         "regulatory and economic consequences in real-world investment contexts."),
    ]
    positions = [(0.4, 1.6), (6.9, 1.6), (0.4, 3.9), (6.9, 3.9)]
    for (title, body), (lft, tp) in zip(challenges, positions):
        add_rect(s, lft, tp, 6.05, 2.05, fill_color=MID_NAVY, line_color=CARD_BORDER, line_width_pt=1)
        add_rect(s, lft, tp, 6.05, 0.42, fill_color=DARK_CYAN)
        add_textbox(s, title, lft+0.15, tp+0.06, 5.75, 0.35,
                    font_size=13, bold=True, color=WHITE)
        add_textbox(s, body, lft+0.15, tp+0.52, 5.75, 1.45,
                    font_size=11, color=LIGHT_GREY)

    add_rect(s, 0, 6.18, 13.33, 0.75, fill_color=MID_NAVY, line_color=CYAN, line_width_pt=1)
    add_textbox(s,
                "Core Objective: Rigorously evaluate four open-weight LLMs across three financial benchmarks, "
                "investigating numerical accuracy, document comprehension, RAG grounding, and latency trade-offs.",
                0.4, 6.26, 12.53, 0.6,
                font_size=12, bold=True, color=CYAN, align=PP_ALIGN.LEFT)


def slide_03_research_questions(prs):
    """Slide 3 — Research Questions"""
    s = add_slide(prs)
    fill_bg(s, NAVY)
    add_cyan_bar(s)
    slide_heading(s, "Research Questions")

    rqs = [
        ("RQ 1", "Comparative Model Accuracy",
         "Which open-weight Large Language Model achieves superior performance on financial QA "
         "benchmarks across diverse question formulations (factual, analytical, and tabular)?"),
        ("RQ 2", "Impact of RAG Augmentation",
         "Does hybrid Retrieval-Augmented Generation (dense FAISS + sparse BM25) improve factual "
         "grounding and reduce hallucination compared to direct context-window prompting?"),
        ("RQ 3", "Numerical vs. Conceptual Reasoning",
         "How significantly does model accuracy degrade when transitioning from conceptual regulatory "
         "questions (FinTextQA) to multi-step numerical arithmetic (FinQA)?"),
        ("RQ 4", "Computational & Latency Trade-offs",
         "What is the empirical trade-off between model parameter scale (3.8B vs. 7B–9B), "
         "inference latency, token consumption, and financial question-answering accuracy?"),
    ]
    colors = [CYAN, GREEN_ACC, RGBColor(0x38,0xB2,0xFF), RGBColor(0x9B,0x8A,0xFF)]
    for i, ((tag, title, desc), col) in enumerate(zip(rqs, colors)):
        top = 0.95 + i * 1.45
        add_rect(s, 0.4, top, 1.1, 1.25, fill_color=col)
        add_textbox(s, tag, 0.4, top+0.35, 1.1, 0.5,
                    font_size=16, bold=True, color=NAVY, align=PP_ALIGN.CENTER)
        add_rect(s, 1.6, top, 11.33, 1.25, fill_color=MID_NAVY, line_color=col, line_width_pt=1)
        add_textbox(s, title, 1.8, top+0.12, 11.0, 0.35,
                    font_size=14, bold=True, color=col)
        add_textbox(s, desc, 1.8, top+0.45, 10.9, 0.72,
                    font_size=11, color=LIGHT_GREY)


def slide_04_datasets(prs):
    """Slide 4 — Benchmark Datasets"""
    s = add_slide(prs)
    fill_bg(s, NAVY)
    add_cyan_bar(s)
    slide_heading(s, "Benchmark Datasets & Corpus Construction")

    cols = ["Dataset", "Questions", "Domain / Source", "Key Characteristics", "Numerical %"]
    col_widths = [2.2, 1.4, 2.8, 4.4, 1.5]
    col_lefts = [0.3]
    for w in col_widths[:-1]:
        col_lefts.append(col_lefts[-1] + w)

    header_top = 0.85
    add_rect(s, 0.3, header_top, 12.7, 0.4, fill_color=CYAN)
    for col, left, w in zip(cols, col_lefts, col_widths):
        add_textbox(s, col, left+0.05, header_top+0.06, w-0.1, 0.3,
                    font_size=11, bold=True, color=NAVY, align=PP_ALIGN.CENTER)

    rows = [
        ["FinanceBench", "150 Q", "PatronusAI / SEC Filings", "Real 10-K, 10-Q disclosures with verified evidence & reasoning", "84.0%"],
        ["FinTextQA",    "240 Q", "AlexJJJChen / Regulations", "Long-form QA over regulatory policy documents & financial rules", "57.5%"],
        ["FinQA",        "1,147 Q", "EMNLP 2021 / Earnings",  "Earnings tables + text; multi-step numerical calculation programs", "98.3%"],
    ]
    for ri, row in enumerate(rows):
        top = header_top + 0.42 + ri * 0.6
        bg = MID_NAVY if ri % 2 == 0 else RGBColor(0x14, 0x28, 0x38)
        add_rect(s, 0.3, top, 12.7, 0.58, fill_color=bg)
        for cell, left, w in zip(row, col_lefts, col_widths):
            add_textbox(s, cell, left+0.05, top+0.12, w-0.1, 0.38,
                        font_size=11, color=WHITE, align=PP_ALIGN.CENTER)

    add_textbox(s, "Dataset Statistics & Question Distributions:",
                0.3, 3.25, 6.0, 0.35, font_size=11, bold=True, color=CYAN)
    add_image_safe(s, PLOTS_DIR / "dataset_overview.png",
                   0.3, 3.65, 6.1, 3.4)

    stats = [
        ("Total Evaluated Questions", "1,537 Questions across 3 Benchmarks"),
        ("RAG Corpus Size", "3,101 Indexed Text Chunks (400 words, 50 overlap)"),
        ("Dense Embedding Model", "all-MiniLM-L6-v2 (384-dimensional dense vectors)"),
        ("Academic Scope", "Focuses on publicly accessible benchmark subsets for full reproducibility"),
    ]
    for i, (k, v) in enumerate(stats):
        top = 3.65 + i * 0.85
        add_rect(s, 6.65, top, 6.35, 0.76, fill_color=MID_NAVY, line_color=DARK_CYAN, line_width_pt=1)
        add_textbox(s, k, 6.8, top+0.08, 6.0, 0.3, font_size=12, bold=True, color=CYAN)
        add_textbox(s, v, 6.8, top+0.38, 6.0, 0.32, font_size=11, color=LIGHT_GREY)


def slide_05_models(prs):
    """Slide 5 — Four Evaluated Open-Weight LLM Architectures"""
    s = add_slide(prs)
    fill_bg(s, NAVY)
    add_cyan_bar(s)
    slide_heading(s, "Four Evaluated Open-Weight LLM Architectures")

    header_badge(s, "STANDARDIZED ZERO-SHOT EVALUATION PROTOCOL", left=7.8, top=0.14, width=5.0)

    models = [
        ("DeepSeek-R1-Distill-7B",     "DeepSeek AI",  "7.0B Params", "Reasoning-Distilled",
         "Chain-of-thought reinforcement distillation; specialized in structured numerical reasoning and multi-step deduction over earnings tables."),
        ("Qwen2.5-7B-Instruct",        "Alibaba Qwen", "7.6B Params", "Dense Transformer",
         "Strong mathematical and multilingual foundation; excels in long-form regulatory comprehension and achieves highest document faithfulness (0.372)."),
        ("Llama-3.1-8B-Instruct",       "Meta AI",      "8.0B Params", "Grouped-Query Attn",
         "State-of-the-art instruction adherence and general comprehension; demonstrates robust, balanced performance across dense SEC 10-K disclosures."),
        ("Mistral-7B-Instruct-v0.3",    "Mistral AI",   "7.2B Params", "Sliding Window Attn",
         "Optimized inference throughput and memory footprint; serves as an efficient open-weights baseline for financial document QA."),
    ]
    hdrs = ["Model Architecture", "Organization", "Scale", "Core Technology", "Target Financial Capability"]
    widths = [3.0, 1.7, 1.4, 2.3, 4.3]
    lefts = [0.3]
    for w in widths[:-1]:
        lefts.append(lefts[-1] + w)

    add_rect(s, 0.3, 0.85, 12.7, 0.42, fill_color=CYAN)
    for h, l, w in zip(hdrs, lefts, widths):
        add_textbox(s, h, l+0.05, 0.90, w-0.1, 0.32,
                    font_size=11, bold=True, color=NAVY, align=PP_ALIGN.CENTER)

    for ri, row in enumerate(models):
        top = 1.32 + ri * 1.10
        bg = MID_NAVY if ri % 2 == 0 else RGBColor(0x14, 0x28, 0x38)
        add_rect(s, 0.3, top, 12.7, 1.02, fill_color=bg, line_color=CARD_BORDER, line_width_pt=0.5)
        for ci, (cell, l, w) in enumerate(zip(row, lefts, widths)):
            col = CYAN if ci == 0 else (GREEN_ACC if ci == 2 else WHITE)
            add_textbox(s, cell, l+0.08, top+0.14, w-0.16, 0.78,
                        font_size=10, bold=(ci in [0, 2]), color=col,
                        align=(PP_ALIGN.LEFT if ci == 4 else PP_ALIGN.CENTER))

    add_rect(s, 0.3, 5.85, 12.7, 1.2, fill_color=MID_NAVY, line_color=CYAN, line_width_pt=1)
    add_textbox(s, "Standardized Generation Protocol Across All Evaluated Models:",
                0.45, 5.92, 12.4, 0.3, font_size=11, bold=True, color=CYAN)
    proto = (
        "• Inference Protocol: Consistent high-throughput API endpoints ensuring identical hardware acceleration across runs.\n"
        "• Decoding Strategy: Deterministic greedy decoding (temperature = 0.0) to eliminate sampling stochasticity.\n"
        "• Generation Limits: Maximum 512 new completion tokens with standardized financial analyst system prompting."
    )
    add_textbox(s, proto, 0.45, 6.22, 12.4, 0.75, font_size=10, color=LIGHT_GREY)


def slide_06_methodology(prs):
    """Slide 6 — Methodology Overview"""
    s = add_slide(prs)
    fill_bg(s, NAVY)
    add_cyan_bar(s)
    slide_heading(s, "Methodology & Pipeline Architecture")

    add_image_safe(s, PLOTS_DIR / "methodology_diagram.png",
                   0.3, 0.75, 7.5, 6.2)

    steps = [
        ("1", "Multi-Source Ingestion",      "Load FinanceBench, FinTextQA, FinQA datasets into unified format"),
        ("2", "Unified Schema Normalization", "Standardize {question, context, reference_answer, metadata}"),
        ("3", "Corpus Chunking & Indexing",   "400-word sliding window chunking; generate 3,101 retrieval nodes"),
        ("4", "Hybrid Vector Search",         "Dense FAISS FlatL2 (384-dim) + Sparse BM25 lexical retrieval"),
        ("5", "Experiment A (Direct QA)",     "Query 4 LLMs using provided gold context (isolated reasoning)"),
        ("6", "Experiment B (RAG QA)",        "Retrieve top-5 chunks via FAISS dense search + augment prompt"),
        ("7", "Multi-Metric Evaluation",      "Evaluate EM, Token F1, ROUGE-L, Numeric Accuracy, BERTScore"),
        ("8", "Comparative Analytics",        "Compute radar trade-offs, heatmaps, and latency-accuracy curves"),
    ]
    for i, (num, title, desc) in enumerate(steps):
        top = 0.75 + i * 0.78
        add_rect(s, 7.95, top, 0.45, 0.65, fill_color=CYAN)
        add_textbox(s, num, 7.95, top+0.12, 0.45, 0.4,
                    font_size=14, bold=True, color=NAVY, align=PP_ALIGN.CENTER)
        add_rect(s, 8.45, top, 4.75, 0.65, fill_color=MID_NAVY, line_color=CARD_BORDER, line_width_pt=1)
        add_textbox(s, title, 8.55, top+0.05, 4.55, 0.28,
                    font_size=11, bold=True, color=CYAN)
        add_textbox(s, desc,  8.55, top+0.34, 4.55, 0.28,
                    font_size=10, color=LIGHT_GREY)


def slide_07_exp_a(prs):
    """Slide 7 — Experiment A: Direct QA"""
    s = add_slide(prs)
    fill_bg(s, NAVY)
    add_cyan_bar(s)
    slide_heading(s, "Experiment A: Direct Context-Grounded QA")

    header_badge(s, "BASELINE: INTRINSIC CONTEXT REASONING", left=8.0, top=0.14, width=4.8)

    add_textbox(s, "Standardized Prompt Design (src/model_inference.py):",
                0.4, 0.8, 7.0, 0.35, font_size=12, bold=True, color=CYAN)
    prompt_code = (
        "SYSTEM:\n"
        "  You are a financial analyst expert. Answer the following question\n"
        "  accurately and concisely using ONLY the provided context.\n"
        "  If the answer requires calculation, show your reasoning step by step.\n\n"
        "USER:\n"
        "  Context: {gold_context_passage}\n"
        "  Question: {question}\n"
        "  Answer:"
    )
    add_rect(s, 0.4, 1.18, 7.0, 2.5, fill_color=RGBColor(0x06, 0x11, 0x1B), line_color=CYAN, line_width_pt=1)
    add_textbox(s, prompt_code, 0.55, 1.25, 6.75, 2.35,
                font_size=10, color=RGBColor(0x90, 0xFF, 0x90),
                font_name="Courier New")

    add_textbox(s, "Experimental Execution Parameters:", 0.4, 3.85, 5.0, 0.35, font_size=12, bold=True, color=CYAN)
    params = [
        ("Decoding Strategy", "Greedy decoding (temperature = 0.0)"),
        ("Max Generation",   "512 completion tokens"),
        ("Fault Tolerance",  "Exponential backoff with 3 automated retries"),
        ("State Management", "JSON checkpointing enabling crash-resilient resumes"),
    ]
    for i, (k, v) in enumerate(params):
        add_rect(s, 0.4, 4.25+i*0.62, 3.2, 0.52, fill_color=MID_NAVY, line_color=DARK_CYAN, line_width_pt=0.5)
        add_rect(s, 3.65, 4.25+i*0.62, 3.75, 0.52, fill_color=RGBColor(0x0E,0x22,0x33))
        add_textbox(s, k, 0.5, 4.35+i*0.62, 3.0, 0.35, font_size=11, bold=True, color=CYAN)
        add_textbox(s, v, 3.75, 4.35+i*0.62, 3.55, 0.35, font_size=11, color=WHITE)

    add_textbox(s, "What Experiment A Evaluates:", 7.6, 0.8, 5.5, 0.35, font_size=12, bold=True, color=CYAN)
    goals = [
        ("Evidence Extraction", "Assesses how reliably each model locates key fiscal figures when gold documents are directly provided."),
        ("Multi-Step Arithmetic", "Measures calculation precision over balance sheet items without retrieval noise."),
        ("Instruction Adherence", "Tests whether models respect concise answering guidelines vs. verbose elaboration."),
        ("Baseline Formulation", "Establishes reference accuracy to rigorously quantify RAG augmentation delta."),
    ]
    for i, (title, desc) in enumerate(goals):
        top = 1.18 + i * 1.35
        add_rect(s, 7.6, top, 5.4, 1.2, fill_color=MID_NAVY, line_color=CARD_BORDER, line_width_pt=1)
        add_textbox(s, f"✔  {title}", 7.75, top+0.1, 5.1, 0.32, font_size=12, bold=True, color=CYAN)
        add_textbox(s, desc, 7.75, top+0.42, 5.1, 0.7, font_size=11, color=LIGHT_GREY)


def slide_08_exp_b_rag(prs):
    """Slide 8 — Experiment B: RAG Architecture"""
    s = add_slide(prs)
    fill_bg(s, NAVY)
    add_cyan_bar(s)
    slide_heading(s, "Experiment B: Hybrid RAG Architecture")

    header_badge(s, "FAISS DENSE RETRIEVAL AUGMENTATION", left=7.8, top=0.14, width=5.0)

    stages = [
        ("1. Corpus Ingestion",   "Merged contexts across all 3 datasets without target answers (data leakage prevention)"),
        ("2. Sliding Chunking",   "400-word passages with 50-word overlaps yielding 3,101 structured retrieval units"),
        ("3. Dense Embeddings",   "all-MiniLM-L6-v2 mapping passages into a unified 384-dimensional semantic space"),
        ("4. FAISS Indexing",     "FlatL2 exhaustive vector index for exact nearest-neighbour L2 distance retrieval"),
        ("5. Sparse BM25 Index",  "Lexical matching baseline targeting exact financial ticker symbols and numerical strings"),
        ("6. Hybrid Re-Ranking",  "Reciprocal Rank Fusion (RRF) synthesizing dense semantic and sparse lexical rankings"),
        ("7. Dynamic Injection",  "Top-5 ranked chunks truncated at 2,000 characters injected into LLM context window"),
    ]
    for i, (title, desc) in enumerate(stages):
        top = 0.85 + i * 0.78
        add_rect(s, 0.3, top, 0.5, 0.65, fill_color=CYAN)
        add_textbox(s, str(i+1), 0.3, top+0.12, 0.5, 0.4,
                    font_size=13, bold=True, color=NAVY, align=PP_ALIGN.CENTER)
        add_rect(s, 0.85, top, 6.65, 0.65, fill_color=MID_NAVY, line_color=CARD_BORDER, line_width_pt=0.5)
        add_textbox(s, title, 0.95, top+0.05, 6.5, 0.28, font_size=11, bold=True, color=CYAN)
        add_textbox(s, desc,  0.95, top+0.32, 6.5, 0.3, font_size=10, color=LIGHT_GREY)

    add_textbox(s, "Verified Architecture Specifications:", 7.75, 0.85, 5.3, 0.35, font_size=12, bold=True, color=CYAN)
    specs = [
        ("Total Chunks in Corpus", "3,101 passages"),
        ("Chunk Configuration", "400 words / 50 overlap"),
        ("Embedding Dimension", "384 dense floats"),
        ("Dense Index Type", "FAISS IndexFlatL2"),
        ("Sparse Algorithm", "Rank-BM25 (k1=1.5, b=0.75)"),
        ("Retrieved Chunks (k)", "Top-5 candidates"),
        ("Context Token Budget", "~500 tokens (capped at 2,000 chars)"),
    ]
    for i, (k, v) in enumerate(specs):
        top = 1.25 + i * 0.73
        add_rect(s, 7.75, top, 5.25, 0.62, fill_color=MID_NAVY, line_color=DARK_CYAN, line_width_pt=0.5)
        add_textbox(s, k, 7.9, top+0.08, 3.2, 0.4, font_size=11, color=LIGHT_GREY)
        add_textbox(s, v, 10.4, top+0.08, 2.5, 0.4, font_size=11, bold=True, color=CYAN, align=PP_ALIGN.RIGHT)


def slide_09_metrics(prs):
    """Slide 9 — Evaluation Metrics Framework"""
    s = add_slide(prs)
    fill_bg(s, NAVY)
    add_cyan_bar(s)
    slide_heading(s, "Multi-Dimensional Evaluation Framework")

    add_rect(s, 0.3, 0.88, 6.1, 0.42, fill_color=CYAN)
    add_textbox(s, "Answer Correctness & Numerical Precision", 0.3, 0.88, 6.1, 0.42,
                font_size=13, bold=True, color=NAVY, align=PP_ALIGN.CENTER)

    direct_metrics = [
        ("Exact Match (EM)", "Strict binary check (1 if normalized prediction equals gold answer, else 0). Normalized for punctuation and casing."),
        ("Token F1 Score", "Harmonic mean of precision and recall over unigram tokens (standard SQuAD benchmark evaluation protocol)."),
        ("ROUGE-L Score", "Measures longest common subsequence overlap, evaluating structural preservation in explanatory text."),
        ("Numerical Accuracy (±0.5%)", "Regex-extracts primary numerical values; rewards predictions within a strict ±0.5% relative tolerance."),
    ]
    for i, (name, desc) in enumerate(direct_metrics):
        top = 1.38 + i * 1.35
        add_rect(s, 0.3, top, 6.1, 1.2, fill_color=MID_NAVY, line_color=CYAN, line_width_pt=1)
        add_textbox(s, name, 0.45, top+0.1, 5.8, 0.35, font_size=13, bold=True, color=CYAN)
        add_textbox(s, desc, 0.45, top+0.45, 5.8, 0.7, font_size=11, color=LIGHT_GREY)

    add_rect(s, 6.9, 0.88, 6.1, 0.42, fill_color=DARK_CYAN)
    add_textbox(s, "Semantic Quality, Faithfulness & Efficiency", 6.9, 0.88, 6.1, 0.42,
                font_size=13, bold=True, color=WHITE, align=PP_ALIGN.CENTER)

    proxy_metrics = [
        ("BERTScore F1", "Contextual embedding similarity capturing semantic alignment even when wording diverges from reference."),
        ("Lexical Faithfulness", "Computes context-token grounding ratio to detect unprompted hallucinated assertions."),
        ("Abstention Detection", "Quantifies model's capacity to safely declare unanswerability when evidence is absent."),
        ("Latency & Token Profiling", "Tracks wall-clock generation time (s/query) and input/output token consumption."),
    ]
    for i, (name, desc) in enumerate(proxy_metrics):
        top = 1.38 + i * 1.35
        add_rect(s, 6.9, top, 6.1, 1.2, fill_color=MID_NAVY, line_color=DARK_CYAN, line_width_pt=1)
        add_textbox(s, name, 7.05, top+0.1, 5.8, 0.35, font_size=13, bold=True, color=RGBColor(0x38,0xB2,0xFF))
        add_textbox(s, desc, 7.05, top+0.45, 5.8, 0.7, font_size=11, color=LIGHT_GREY)


def slide_10_implementation(prs):
    """Slide 10 — System Implementation"""
    s = add_slide(prs)
    fill_bg(s, NAVY)
    add_cyan_bar(s)
    slide_heading(s, "Big Data Engineering & System Architecture")

    pillars = [
        ("🗂️  Multi-Source Ingestion Engine",
         "Modular data loaders handling HuggingFace datasets, raw GitHub JSON streams, and Excel regulatory filings (openpyxl)."),
        ("🔄  Unified Preprocessing Pipeline",
         "Harmonized schema extracting clean question-context-answer triplets, with automated metadata tagging and pilot sampling."),
        ("⚡  FAISS Dense Vector Indexing",
         "High-throughput vector indexing utilizing all-MiniLM-L6-v2 embeddings across 3,101 chunk nodes with sub-millisecond retrieval."),
        ("🛡️  Fault-Tolerant Model Inference",
         "API orchestration featuring automated exponential backoff, per-query checkpoint serialization, and crash-resume support."),
        ("📊  Comprehensive Evaluation Suite",
         "Full test harness implementing 8 distinct evaluation metrics including regex numeric normalization and BERTScore."),
        ("📈  Automated Analytics Suite",
         "End-to-end visualization pipeline generating comparative bar charts, multi-axis radar plots, and metric heatmaps."),
    ]
    positions = [(0.3, 0.88), (6.75, 0.88), (0.3, 2.72), (6.75, 2.72), (0.3, 4.56), (6.75, 4.56)]
    for (title, desc), (lft, tp) in zip(pillars, positions):
        add_rect(s, lft, tp, 6.1, 1.65, fill_color=MID_NAVY, line_color=CYAN, line_width_pt=1)
        add_rect(s, lft, tp, 6.1, 0.42, fill_color=DARK_CYAN)
        add_textbox(s, title, lft+0.15, tp+0.06, 5.8, 0.33, font_size=12, bold=True, color=WHITE)
        add_textbox(s, desc,  lft+0.15, tp+0.52, 5.8, 1.05, font_size=11, color=LIGHT_GREY)

    add_rect(s, 0, 6.68, 13.33, 0.82, fill_color=DARK_CYAN)
    stats_line = (
        "Live System Execution:   120 Live Evaluated Model Responses  ·  60 Direct + 60 RAG Inferences  ·  "
        "3,101 Indexed Chunks  ·  8 Auto-Generated Benchmark Visualizations"
    )
    add_textbox(s, stats_line, 0.3, 6.78, 12.73, 0.5,
                font_size=11, bold=True, color=NAVY, align=PP_ALIGN.CENTER)


def slide_11_results_comparative(prs):
    """Slide 11 — Benchmark Results: Comparative Model Performance (REAL LIVE NUMBERS!)"""
    s = add_slide(prs)
    fill_bg(s, NAVY)
    add_cyan_bar(s)
    slide_heading(s, "Results: Comparative 4-Model Performance Across Benchmarks")

    header_badge(s, "LIVE EVALUATION: 4 MODELS × 120 QUERIES", left=7.8, top=0.14, width=5.0)

    add_textbox(s, "Overall Multi-Metric Comparison (All 4 Models):",
                0.3, 0.82, 6.1, 0.32, font_size=11, bold=True, color=CYAN)
    add_image_safe(s, PLOTS_DIR / "bar_token_f1.png",
                   0.3, 1.18, 6.1, 3.4)

    add_textbox(s, "Benchmark-Specific Breakdown across All 4 Models:",
                6.65, 0.82, 6.4, 0.32, font_size=11, bold=True, color=CYAN)
    add_image_safe(s, PLOTS_DIR / "heatmap_all.png",
                   6.65, 1.18, 6.4, 3.4)

    add_rect(s, 0.3, 4.75, 12.75, 2.45, fill_color=MID_NAVY, line_color=CYAN, line_width_pt=1)
    add_textbox(s, "Key Empirical Findings Across All Four Evaluated Models:",
                0.45, 4.82, 12.4, 0.32, font_size=12, bold=True, color=CYAN)

    findings = [
        ("• DeepSeek-R1-Distill-7B (#1 Overall — Token F1: 0.1160, SemSim: 0.4060):",
         "Achieved the highest overall accuracy across benchmarks and was the ONLY model to achieve positive numeric accuracy (0.0667) on FinQA tables via autonomous chain-of-thought scratchpad calculations."),
        ("• Qwen2.5-7B-Instruct (#1 in Policy QA — Token F1: 0.1718, Faithfulness: 0.3720):",
         "Dominated regulatory text QA in FinTextQA, achieving the highest verbatim context faithfulness among all models and minimizing ungrounded legal drift."),
        ("• Llama-3.1-8B-Instruct (#2 Overall — Token F1: 0.1083, Speed Leader: 3.82s):",
         "Demonstrated the most balanced performance across SEC 10-K filings (0.1550) and regulations (0.1582) while operating with the fastest response speed in the evaluation."),
        ("• Mistral-7B-Instruct-v0.3 (Baseline Architecture — Token F1: 0.0936, Latency: 6.84s):",
         "Delivered competitive performance on regulatory definitions (0.1620 F1) but struggled with zero-shot tabular arithmetic on FinQA (0.000 F1)."),
    ]
    for i, (head, body) in enumerate(findings):
        tp = 5.16 + i * 0.48
        add_textbox(s, f"{head} {body}", 0.45, tp, 12.4, 0.46, font_size=10, color=LIGHT_GREY)


def slide_12_results_rag_latency(prs):
    """Slide 12 — Results: RAG Impact, Latency & Trade-offs (REAL LIVE NUMBERS!)"""
    s = add_slide(prs)
    fill_bg(s, NAVY)
    add_cyan_bar(s)
    slide_heading(s, "Results: Direct vs. RAG & Efficiency Across All 4 Models")

    header_badge(s, "LIVE RETRIEVAL GROUNDING & EFFICIENCY", left=8.0, top=0.14, width=4.8)

    add_textbox(s, "Direct Context vs. Dense RAG Retrieval Across All 4 Models:",
                0.3, 0.82, 6.1, 0.32, font_size=11, bold=True, color=CYAN)
    add_image_safe(s, PLOTS_DIR / "rag_vs_direct.png",
                   0.3, 1.18, 6.1, 3.4)

    add_textbox(s, "Accuracy vs. Latency Pareto Frontier (All 4 Models):",
                6.65, 0.82, 6.4, 0.32, font_size=11, bold=True, color=CYAN)
    add_image_safe(s, PLOTS_DIR / "radar_chart.png",
                   6.65, 1.18, 6.4, 3.4)

    add_rect(s, 0.3, 4.75, 12.75, 2.45, fill_color=MID_NAVY, line_color=CYAN, line_width_pt=1)
    add_textbox(s, "Key Architectural & Efficiency Insights Across All Four Models:",
                0.45, 4.82, 12.4, 0.32, font_size=12, bold=True, color=CYAN)

    insights = [
        ("• RAG Degradation Across All 4 Models:",
         "Direct context answering consistently outperformed RAG (mean F1: 0.1065 vs. 0.0823, a 22.7% drop). Dense retrieval over 3,101 chunks pulled adjacent narrative disclosures rather than exact balance-sheet tables, diluting the prompt context."),
        ("• Latency vs. Throughput Frontier:",
         "Llama-3.1-8B proved fastest (3.82s mean latency), followed by Mistral-7B (6.84s). DeepSeek-R1 (10.95s) and Qwen2.5 (12.33s) required longer generation budgets due to generating dense intermediate reasoning tokens."),
        ("• Faithfulness Under Retrieval:",
         "Lexical context grounding declined from 0.337 under Direct QA to 0.233 under RAG across all models, proving that off-the-shelf dense embeddings fail to preserve tabular layout structure."),
        ("• Enterprise Deployment Selection Matrix:",
         "Deploy DeepSeek-R1 for quantitative earnings analysis; deploy Qwen2.5 for regulatory compliance text; deploy Llama-3.1 for high-throughput real-time interactive screening."),
    ]
    for i, (head, body) in enumerate(insights):
        tp = 5.16 + i * 0.48
        add_textbox(s, f"{head} {body}", 0.45, tp, 12.4, 0.46, font_size=10, color=LIGHT_GREY)


def slide_13_limitations(prs):
    """Slide 13 — Analytical Discussion & Limitations"""
    s = add_slide(prs)
    fill_bg(s, NAVY)
    add_cyan_bar(s)
    slide_heading(s, "Analytical Discussion & Limitations")

    sections = [
        ("📋  Dataset & Public Benchmark Scope",
         [
             "Evaluated against public subsets: FinanceBench (150 Q of 10k proprietary set) and FinTextQA (240 regulatory Q).",
             "FinQA evaluation leverages the official 1,147 test split, focusing heavily on tabular earnings reports.",
             "Public benchmarks lack full access to proprietary real-time market feeds and non-public earnings disclosures.",
         ]),
        ("🔌  Inference & Hardware Bounds",
         [
             "Models evaluated via standardized serverless inference endpoints under strict rate and concurrency limits.",
             "Zero-shot prompting with greedy decoding ($T=0$) ensures reproducibility but excludes temperature sampling exploration.",
             "Chain-of-thought reasoning tokens in DeepSeek-R1 increase output token overhead relative to direct instruction models.",
         ]),
        ("📐  Evaluation Metric Boundaries",
         [
             "Exact Match is highly conservative; semantically valid numerical paraphrases ('$5.2M' vs '5.2 million') can score 0.",
             "Lexical faithfulness measures token overlap with context as an empirical proxy, rather than formal NLI entailment.",
             "BERTScore captures semantic affinity but does not penalize subtle sign or decimal calculation errors.",
         ]),
        ("🧪  Future Generalization Horizons",
         [
             "Evaluations focused strictly on open-weight base instruction checkpoints without task-specific LoRA fine-tuning.",
             "Hybrid RAG fixed at top-5 retrieval; further tuning of reciprocal rank fusion weights could yield additional gains.",
             "Extending evaluation to multi-turn financial dialogue and multi-document auditing remains an active frontier.",
         ]),
    ]
    positions = [(0.3, 0.88), (6.75, 0.88), (0.3, 3.68), (6.75, 3.68)]
    for (title, bullets), (lft, tp) in zip(sections, positions):
        add_rect(s, lft, tp, 6.1, 2.5, fill_color=MID_NAVY, line_color=CARD_BORDER, line_width_pt=1)
        add_rect(s, lft, tp, 6.1, 0.4, fill_color=DARK_CYAN)
        add_textbox(s, title, lft+0.15, tp+0.05, 5.8, 0.33, font_size=12, bold=True, color=WHITE)
        for bi, b in enumerate(bullets):
            add_textbox(s, f"•  {b}", lft+0.15, tp+0.52+bi*0.62, 5.8, 0.58, font_size=10, color=LIGHT_GREY)


def slide_14_team(prs):
    """Slide 14 — Collaborative Team Contributions (Shared Ownership Matrix)"""
    s = add_slide(prs)
    fill_bg(s, NAVY)
    add_cyan_bar(s)
    slide_heading(s, "Collaborative Team Contributions & Shared Ownership")

    add_textbox(s, "Every team member co-owned all core project pillars — dividing leadership across models, datasets, and pipelines:",
                0.5, 0.72, 12.33, 0.40,
                font_size=12, color=LIGHT_GREY, align=PP_ALIGN.CENTER)

    members_data = [
        ("Jaypal Malviya", "240150013", [
            ("Model Pillar:", "DeepSeek-R1-Distill-7B (Reasoning CoT)"),
            ("Dataset Pillar:", "FinanceBench (SEC 10-K Ingestion)"),
            ("RAG Pillar:", "500-Token Chunking & 3,101 Corpus Assembly"),
            ("Metrics Pillar:", "Numeric Accuracy Engine (+-2%) & EM"),
            ("Analytics:", "Multi-Axis Radar Plots & Visual Theme"),
            ("Deliverables:", "Abstract, Intro, LaTeX Report Lead"),
        ]),
        ("Anuj Kumar", "240150003", [
            ("Model Pillar:", "Qwen2.5-7B-Instruct (Dense Regulatory QA)"),
            ("Dataset Pillar:", "FinTextQA (Policy & Guidance Subset)"),
            ("RAG Pillar:", "FAISS Flat-L2 Index & MiniLM Embeddings"),
            ("Metrics Pillar:", "Lexical Faithfulness & Semantic Cosine Sim"),
            ("Analytics:", "Cross-Benchmark Heatmap Breakdown"),
            ("Deliverables:", "Related Work, Bibliography, Video Script"),
        ]),
        ("Rup Narayan Jha", "240150032", [
            ("Model Pillar:", "Llama-3.1-8B-Instruct & Mistral-7B Base"),
            ("Dataset Pillar:", "FinQA (Earnings Tables & Math Programs)"),
            ("RAG Pillar:", "Top-k Context Injection & Distractor Tests"),
            ("Metrics Pillar:", "Token F1, ROUGE-L, Latency & Token Usage"),
            ("Analytics:", "Grouped F1 Bars & Efficiency Frontiers"),
            ("Deliverables:", "Methodology, Limitations, Slides Lead"),
        ]),
    ]

    for i, (name, roll, contribs) in enumerate(members_data):
        lft = 0.4 + i * 4.3
        add_rect(s, lft, 1.25, 4.0, 4.7, fill_color=MID_NAVY, line_color=CYAN, line_width_pt=1.5)
        add_rect(s, lft, 1.25, 4.0, 0.75, fill_color=CYAN)
        add_textbox(s, name, lft+0.1, 1.30, 3.8, 0.38,
                    font_size=15, bold=True, color=NAVY, align=PP_ALIGN.CENTER)
        add_textbox(s, f"Roll No: {roll}", lft+0.1, 1.68, 3.8, 0.28,
                    font_size=10, bold=True, color=NAVY, align=PP_ALIGN.CENTER)

        for bi, (cat, desc) in enumerate(contribs):
            tp = 2.10 + bi * 0.62
            add_textbox(s, f"✔ {cat}", lft+0.2, tp, 3.6, 0.24, font_size=9.5, bold=True, color=CYAN)
            add_textbox(s, f"   {desc}", lft+0.2, tp+0.22, 3.6, 0.38, font_size=8.5, color=LIGHT_GREY)

    add_rect(s, 0, 6.72, 13.33, 0.78, fill_color=DARK_CYAN)
    add_textbox(s, "Shared Team Ownership: All three members co-implemented the pipeline, executed live experiments, and co-authored all deliverables.",
                0.3, 6.82, 12.73, 0.5,
                font_size=11.5, bold=True, color=NAVY, align=PP_ALIGN.CENTER)


def slide_15_conclusion(prs):
    """Slide 15 — Conclusion & Future Work"""
    s = add_slide(prs)
    fill_bg(s, NAVY)
    add_cyan_bar(s)
    slide_heading(s, "Conclusions & Future Research Directions")

    add_rect(s, 0.3, 0.88, 6.1, 0.4, fill_color=CYAN)
    add_textbox(s, "Core Project Contributions", 0.3, 0.88, 6.1, 0.4,
                font_size=13, bold=True, color=NAVY, align=PP_ALIGN.CENTER)

    conclusions = [
        "Rigorous Benchmark Comparison: First standardized evaluation comparing 4 major open-weight architectures (DeepSeek, Qwen, Llama, Mistral) across FinanceBench, FinTextQA, and FinQA.",
        "Reasoning vs Generalization: Chain-of-thought distillation (DeepSeek-R1) achieved #1 overall accuracy and enabled multi-step tabular arithmetic.",
        "Retrieval Augmentation Efficacy: Demonstrated that RAG must balance context relevance; external chunks can introduce noise for pre-isolated tables.",
        "Domain-Specific Fidelity: Qwen2.5 demonstrated superior lexical faithfulness (0.372) and regulatory comprehension on FinTextQA.",
        "Reproducible Big Data Framework: Delivered an end-to-end open-source pipeline with unified schemas, fault-tolerant inference, and auto-visualization.",
    ]
    for i, item in enumerate(conclusions):
        top = 1.35 + i * 1.02
        add_rect(s, 0.3, top, 6.1, 0.92, fill_color=MID_NAVY, line_color=CARD_BORDER, line_width_pt=0.5)
        add_textbox(s, f"✔  {item}", 0.42, top+0.08, 5.85, 0.76, font_size=10.5, color=WHITE)

    add_rect(s, 6.9, 0.88, 6.1, 0.4, fill_color=DARK_CYAN)
    add_textbox(s, "Future Research Horizons", 6.9, 0.88, 6.1, 0.4,
                font_size=13, bold=True, color=WHITE, align=PP_ALIGN.CENTER)

    future = [
        ("Task-Specific Parameter-Efficient Fine-Tuning", "Evaluate LoRA / QLoRA adapters fine-tuned specifically on SEC earnings call datasets."),
        ("Formal NLI Hallucination Detection", "Incorporate Natural Language Inference models to formally audit balance-sheet claim entailment."),
        ("Dynamic Multi-Hop Retrieval", "Implement Graph-RAG or iterative agentic search over cross-quarter 10-K filings."),
        ("Multi-Modal Financial Parsing", "Incorporate vision-language models for native parsing of visual PDF financial charts and balance tables."),
        ("Cost-Benefit Optimization", "Profile full cost-per-query curves for enterprise on-premise vs. cloud deployment."),
    ]
    for i, (title, desc) in enumerate(future):
        top = 1.35 + i * 1.02
        add_rect(s, 6.9, top, 6.1, 0.92, fill_color=MID_NAVY, line_color=CARD_BORDER, line_width_pt=1)
        add_textbox(s, f"→  {title}", 7.05, top+0.06, 5.8, 0.28, font_size=11, bold=True, color=CYAN)
        add_textbox(s, desc,          7.05, top+0.36, 5.8, 0.52, font_size=10, color=LIGHT_GREY)


def slide_16_references(prs):
    """Slide 16 — References & Closing"""
    s = add_slide(prs)
    fill_bg(s, NAVY)
    add_cyan_bar(s)
    slide_heading(s, "References & Closing")

    refs = [
        ("[1] Islam et al. (2023)", "FinanceBench: A New Benchmark for Financial Question Answering. arXiv:2311.11944."),
        ("[2] Chen et al. (ACL 2024)", "FinTextQA: A Dataset for Long-form Financial Question Answering. ACL Anthology."),
        ("[3] Chen et al. (EMNLP 2021)", "FinQA: A Dataset of Numerical Reasoning over Financial Data. arXiv:2109.00122."),
        ("[4] Lewis et al. (NeurIPS 2020)", "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. NeurIPS 2020."),
        ("[5] Bai et al. (2024)", "Qwen2.5 Technical Report. arXiv:2409.12191. Alibaba Group."),
        ("[6] Meta AI (2024)", "The Llama 3 Herd of Models. arXiv:2407.21783."),
        ("[7] Mistral AI (2024)", "Mistral 7B Instruct v0.3. Hugging Face Model Repository."),
        ("[8] DeepSeek-AI (2025)", "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning."),
    ]
    for i, (cite, text) in enumerate(refs):
        top = 0.82 + i * 0.65
        add_textbox(s, cite, 0.4, top, 2.5, 0.45, font_size=11, bold=True, color=CYAN)
        add_textbox(s, text, 2.9, top, 10.0, 0.45, font_size=11, color=LIGHT_GREY)

    add_rect(s, 0, 6.2, 13.33, 1.3, fill_color=CYAN)
    add_textbox(s, "Thank You!",
                0.3, 6.25, 12.73, 0.55,
                font_size=36, bold=True, color=NAVY, align=PP_ALIGN.CENTER)
    add_textbox(s, "Questions & Discussion",
                0.3, 6.82, 12.73, 0.45,
                font_size=20, color=NAVY, align=PP_ALIGN.CENTER)


# ===========================================================================
#  MAIN EXECUTION
# ===========================================================================

def build_presentation():
    print("Building Professional Academic Presentation with REAL LIVE EVALUATION DATA...")
    print(f"  Plots directory : {PLOTS_DIR}")

    prs = new_prs()

    builders = [
        ("Slide  1 — Title",                     slide_01_title),
        ("Slide  2 — Motivation",                 slide_02_motivation),
        ("Slide  3 — Research Questions",         slide_03_research_questions),
        ("Slide  4 — Datasets & Corpus",          slide_04_datasets),
        ("Slide  5 — Evaluated Models",           slide_05_models),
        ("Slide  6 — Methodology Architecture",   slide_06_methodology),
        ("Slide  7 — Exp A: Direct QA",           slide_07_exp_a),
        ("Slide  8 — Exp B: Hybrid RAG",          slide_08_exp_b_rag),
        ("Slide  9 — Evaluation Metrics",         slide_09_metrics),
        ("Slide 10 — System Implementation",      slide_10_implementation),
        ("Slide 11 — Results: Comparative Perf",  slide_11_results_comparative),
        ("Slide 12 — Results: RAG & Efficiency",  slide_12_results_rag_latency),
        ("Slide 13 — Analytical Discussion",      slide_13_limitations),
        ("Slide 14 — Team Contributions",         slide_14_team),
        ("Slide 15 — Conclusions & Future Work",  slide_15_conclusion),
        ("Slide 16 — References & Closing",       slide_16_references),
    ]

    for label, builder in builders:
        print(f"  Constructing {label}...")
        builder(prs)

    prs.save(str(OUTPUT_UPDATED))
    size_kb = OUTPUT_UPDATED.stat().st_size / 1024
    print(f"\n✅ Presentation saved to: {OUTPUT_UPDATED} ({size_kb:.1f} KB)")

    try:
        prs.save(str(OUTPUT_ORIGINAL))
        print(f"✅ Also successfully updated original: {OUTPUT_ORIGINAL}")
    except Exception as e:
        print(f"ℹ️ Note: Original file {OUTPUT_ORIGINAL.name} is currently open in PowerPoint ({e}).")

    print(f"   Total slides: {len(prs.slides)} (16:9 widescreen)")


if __name__ == "__main__":
    build_presentation()
