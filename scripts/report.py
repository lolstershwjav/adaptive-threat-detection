"""Render figures and tables from the exact checked-in study result."""

import argparse
from pathlib import Path
from statistics import mean

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Table, TableStyle

from adaptive.evidence import load_study

INK = "#183549"
MUTED = "#53697a"
ACCENTS = ("#5777a4", "#00a487", "#c46a36", "#865da8")
DISPLAY = {
    "stationary": "Stationary", "gradual": "Gradual", "sudden": "Sudden",
    "sudden_burst": "Sudden + burst", "unsw": "UNSW-NB15",
    "frozen": "Frozen", "threshold": "Threshold only", "rolling": "Rolling",
    "rolling_trim": "Rolling + trim", "gated": "Drift-gated",
    "gated_trim": "Gated + trim", "gated_fit_only": "Gated fit only",
    "iforest": "Isolation Forest", "lof": "Local Outlier Factor", "ocsvm": "One-Class SVM",
    "ae": "Autoencoder", "vae": "Variational autoencoder", "ensemble": "Ensemble",
}


def get(index, scenario, model, policy, field):
    return index[scenario, model, policy][field]


def fmt(metric, scale=100):
    if metric is None:
        return "—"
    return f"{metric['mean'] * scale:.{1 if scale == 100 else 2}f}"


def tradeoff_chart(index, cases, figure_path):
    fig, charts = plt.subplots(1, len(cases), figsize=(12, 3.5), sharey=True, squeeze=False)
    charts = charts[0]
    for case, ax in zip(cases, charts):
        for policy, color in zip(["frozen", "threshold", "rolling_trim", "gated_trim"], ACCENTS):
            data = index[case, "iforest", policy]
            xs, ys = data["fpr"], data["recall"]
            x, y = xs["mean"] * 100, ys["mean"] * 100
            ax.errorbar(x, y, xerr=[[x - xs["low"] * 100], [xs["high"] * 100 - x]],
                        yerr=[[y - ys["low"] * 100], [ys["high"] * 100 - y]],
                        color=color, fmt="o", capsize=2, label=DISPLAY[policy], markersize=6)
        ax.set_title(DISPLAY[case], color=INK, weight="semibold")
        ax.set_xlabel("False positives (%)")
        ax.grid(alpha=.22)
        ax.set_xlim(left=0)
        ax.set_ylim(0, 100)
        ax.spines[["top", "right"]].set_visible(False)
    charts[0].set_ylabel("Attack recall (%)")
    handles, labels = charts[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=4, loc="upper center", bbox_to_anchor=(.5, .99),
               frameon=False)
    fig.tight_layout(rect=(0, 0, 1, .88), w_pad=1.7)
    fig.savefig(figure_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def timeline_chart(data, figure_path):
    fig, ax = plt.subplots(figsize=(10, 3.15))
    for policy, color in zip(["frozen", "threshold", "rolling_trim", "gated_trim"], ACCENTS):
        runs = [r["timeline"] for r in data["runs"]
                if (r["scenario"], r["model"], r["policy"]) == ("sudden", "iforest", policy)]
        if not runs:
            continue
        curve = np.array([[b["fpr"] for b in row] for row in runs])
        ax.plot(np.arange(curve.shape[1]), 100 * curve.mean(axis=0), marker="o", ms=3,
                label=DISPLAY[policy], color=color)
    ax.axvline(4, color=MUTED, linestyle="--", alpha=.8, label="Drift starts")
    ax.set(xlabel="Replay batch", ylabel="False positives (%)", ylim=(0, None))
    ax.legend(ncol=3, frameon=False)
    ax.grid(alpha=.22)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(figure_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def architecture_chart(path):
    fig, ax = plt.subplots(figsize=(10, 2.6))
    ax.axis("off")
    nodes = [
        (.12, .68, "Warmup +\nnext batch"),
        (.36, .68, "Fit scaler, scorer\nand rank reference"),
        (.62, .68, "Score with prior\nstate and alert"),
        (.88, .68, "Join labels to\nevaluate outcomes"),
        (.62, .22, "Check drift and update\nfit / reference / threshold"),
    ]
    for x, y, label in nodes:
        ax.text(x, y, label, ha="center", va="center", color=INK, fontsize=11,
                bbox=dict(boxstyle="round,pad=.6", ec="#b5d2df", fc="#f0f8fa"))
    for src, dst in [((.18, .68), (.26, .68)), ((.46, .68), (.53, .68)),
                     ((.71, .68), (.79, .68)), ((.62, .53), (.62, .35))]:
        ax.annotate("", xy=dst, xytext=src,
                    arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.5))
    ax.text(.87, .22, "State applies to\nsubsequent batches", ha="center", va="center",
            color=MUTED, fontsize=10)
    fig.savefig(path, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def data_table(rows, widths):
    table = Table(rows, colWidths=widths, repeatRows=1, hAlign="LEFT")
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(INK)),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f0f5f7")]),
        ("FONTNAME", (0, 0), (-1, -1), "ReportSans"),
        ("FONTNAME", (0, 0), (-1, 0), "ReportSans-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("ALIGN", (1, 0), (-1, -1), "RIGHT"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
    ]))
    return table


def report(data, index, output):
    fonts = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
    pdfmetrics.registerFont(TTFont("ReportSans", str(fonts / "DejaVuSans.ttf")))
    pdfmetrics.registerFont(TTFont("ReportSans-Bold", str(fonts / "DejaVuSans-Bold.ttf")))
    pdfmetrics.registerFontFamily("ReportSans", normal="ReportSans", bold="ReportSans-Bold")
    base = getSampleStyleSheet()
    for name in ("Normal", "BodyText"):
        base[name].fontName = "ReportSans"
    for name in ("Title", "Heading2"):
        base[name].fontName = "ReportSans-Bold"
    base.add(ParagraphStyle(name="Deck", parent=base["Normal"], fontSize=13, leading=18,
                            textColor=colors.HexColor(MUTED), spaceAfter=14))
    base["Title"].fontSize = 25
    base["Title"].leading = 29
    base["Title"].alignment = 0
    base["Title"].textColor = colors.HexColor(INK)
    base["BodyText"].leading = 14
    base["BodyText"].spaceAfter = 8
    base["Heading2"].textColor = colors.HexColor(INK)
    items = []

    def p(value, kind="BodyText"):
        items.append(Paragraph(value, base[kind]))

    def section(title, body):
        p(title, "Heading2")
        p(body)

    def graphic(name, height):
        items.append(Image(str(output / name), width=486, height=height, kind="proportional"))

    def metric(case, policy, field, model="iforest"):
        return fmt(get(index, case, model, policy, field))

    def delta(case, policy, field):
        r = get(index, case, "iforest", policy, field + "_vs_frozen")
        return f"{100*r['mean']:+.1f} pp (95% interval {100*r['low']:+.1f} to {100*r['high']:+.1f})"

    p("Adaptive Unsupervised Threat Detection", "Title")
    p("A controlled replay study of threshold adaptation, drift gating and contaminated updates",
      "Deck")
    seeds = len(data["config"]["seeds"])
    p(f"<b>Abstract.</b> We compare six anomaly scorers over {seeds} seeded replays of four "
      "controlled stream regimes and a UNSW-NB15 cyber-range replay. We measure operational "
      "alert quality and a retrospective ranking comparison at matched recall. Under sudden "
      f"drift, drift-gated trimming changes the Isolation Forest false-positive rate by "
      f"{delta('sudden', 'gated_trim', 'fpr')}; its change in recall is "
      f"{delta('sudden', 'gated_trim', 'recall')}. The public benchmark probes transfer "
      "to experimental flows with different attack prevalences and a constructed replay order.")
    section("1. Research questions",
            "Can streaming updates offset benign feature drift while preserving detection? "
            "We hypothesized that periodic refitting would improve discrimination, gating would "
            "reduce refit frequency, and trimming would lower training contamination during a burst. "
            "Alert rate, ranking and update frequency test separate parts of those hypotheses.")
    section("2. Related work",
            "Isolation Forest isolates observations by random splits [1]. LOF uses local density [2] "
            "and the One-Class SVM estimates distribution support [3]. Bottleneck reconstruction "
            "models [4] and latent-variable autoencoding [5] provide complementary baselines. "
            "ADWIN [6] motivates monitoring distribution change. Here we use a fixed batch median "
            "signal with persistence and no statistical false alarm guarantee.")
    section("3. Methodology",
            "A model and scaler fit on benign warmup rows. Two disjoint subsets set the empirical "
            "rank reference and upper alert quantile. Each batch is scored before the policy sees "
            "it for updating. Evaluation labels are held outside the policy. Refit policies replace "
            "the oldest training rows and may trim the highest prior-score candidates. The "
            "ensemble averages calibrated ranks from Isolation Forest, LOF, and the autoencoder.")
    graphic("architecture.png", 148)
    p("<b>Figure 1.</b> A batch can change model state only after it has been scored.")

    items.append(PageBreak())
    section("4. Data and experimental design",
            "The controlled generator uses a correlated 12-dimensional process, abrupt or gradual "
            "changes, a four-batch contamination burst, and an anomaly family held out until the "
            "latter half of the replay. The public UNSW-NB15 benchmark uses the official partition "
            "sizes [7-11], 16 whitelisted flow-local features, finite-value filtering, duplicate "
            "exclusion, log1p preprocessing and within-bin sampling. The partitioned release has "
            "no timestamps. Its replay order is constructed, and its attacks are cyber-range events.")
    p("We report means across seeds, 95% percentile bootstrap intervals (4,000 resamples), and "
      "paired differences to frozen policies. These intervals describe run-to-run variation, "
      "particularly resampling variability on the one fixed UNSW-NB15 corpus. We estimate FPR "
      "at 60% recall retrospectively using test labels. The alert metrics use the deployed threshold.")
    section("5. Operating-point results", "Table 1 reports percentages for Isolation Forest. "
            "The last column uses the retrospective test-score threshold.")
    rows = [["Scenario", "Policy", "FPR", "Recall", "AP", "FPR @ 60% recall"]]
    for case in ("stationary", "gradual", "sudden", "sudden_burst", "unsw"):
        for policy in ("frozen", "gated_trim"):
            rows.append([DISPLAY[case], DISPLAY[policy],
                         *(metric(case, policy, field) for field in ("fpr", "recall", "ap",
                                                                     "fpr_at_60_recall"))])
    items.append(data_table(rows, [90, 98, 51, 55, 55, 137]))
    graphic("tradeoffs.png", 174)
    p("<b>Figure 2.</b> Observed false-positive and recall trade-offs. Bars show seed bootstrap intervals.")

    items.append(PageBreak())
    section("6. Ablations, computation and latency",
            "Table 2 isolates the sudden-shift policy changes. Time is the median batch scoring "
            "time computed within a seed and then averaged across seeds. Update count indicates "
            "how many of the 20 batches changed policy state.")
    rows = [["Policy", "FPR (%)", "Recall (%)", "FPR @ 60%", "Updates", "Score (ms)"]]
    for policy in ("frozen", "threshold", "rolling", "rolling_trim", "gated", "gated_trim",
                   "gated_fit_only"):
        rows.append([DISPLAY[policy], metric("sudden", policy, "fpr"),
                     metric("sudden", policy, "recall"), metric("sudden", policy, "fpr_at_60_recall"),
                     fmt(get(index, "sudden", "iforest", policy, "updates"), 1),
                     fmt(get(index, "sudden", "iforest", policy, "score_ms"), 1)])
    items.append(data_table(rows, [111, 67, 69, 77, 67, 95]))
    graphic("timeline.png", 155)
    p("<b>Figure 3.</b> Mean false-positive rate by batch. A sudden change starts at batch 4.")
    burst = [metric("sudden_burst", policy, "retained_contamination")
             for policy in ("rolling", "rolling_trim", "gated", "gated_trim")]
    p("<b>Training contamination.</b> Across admitted rows over the burst scenario, "
      f"the attack fraction is {burst[0]}% for rolling and {burst[1]}% with trim, "
      f"and {burst[2]}% for gated and {burst[3]}% with trim. Gating changes the number "
      "and timing of admissions. The tabulated fractions include the full replay.")
    delay = get(index, "sudden", "iforest", "gated_trim", "drift_delay_batches")
    unseen = get(index, "sudden", "iforest", "gated_trim", "unknown_delay_batches")
    p(f"<b>Latency.</b> The gated, trimmed policy signals sudden drift after {fmt(delay, 1)} "
      f"batches on average over {delay['n']} of {seeds} seeds with a signal. The mean first "
      f"alert delay for the held-out synthetic family is {fmt(unseen, 1)} batches over "
      f"{unseen['n']} of {seeds} seeds with an alert.")

    items.append(PageBreak())
    section("7. Model comparison",
            "Table 3 compares average precision and observed recall for each gated and trimmed "
            "model on a common benchmark. Average precision depends on the positive class prevalence.")
    rows = [["Model", "Sudden AP", "Sudden recall", "UNSW AP", "UNSW recall"]]
    for model in data["config"]["models"]:
        rows.append([DISPLAY[model], metric("sudden", "gated_trim", "ap", model),
                     metric("sudden", "gated_trim", "recall", model),
                     metric("unsw", "gated_trim", "ap", model),
                     metric("unsw", "gated_trim", "recall", model)])
    items.append(data_table(rows, [154, 83, 83, 83, 83]))
    updates = {True: [], False: []}
    for row in data["runs"]:
        if (row["scenario"], row["model"], row["policy"]) == ("unsw", "iforest", "gated_trim"):
            for batch in row["timeline"]:
                updates[batch["updated"]].append(batch["true_anomalies"] / data["config"]["batch_size"])
    section("8. Discussion",
            "The paired sudden-shift result shows a FPR change at matched recall of "
            f"{delta('sudden', 'gated_trim', 'fpr_at_60_recall')}. In UNSW-NB15 that change "
            f"is {delta('unsw', 'gated_trim', 'fpr_at_60_recall')}. Changes in empirical "
            "ranking and the deployed quantile can push metrics in different directions. "
            f"UNSW update batches averaged {100*mean(updates[True]):.1f}% attacks versus "
            f"{100*mean(updates[False]):.1f}% across other batches. The prior-score trim admitted "
            "substantial contamination because its fixed rejection budget was smaller than the "
            "malicious share in some batches. Label prevalence is diagnostic after replay.")
    p(
            "The fit-only ablation changes the learned ranking and retains the numerical "
            "threshold, although the empirical rank reference is rebuilt with the scorer.")
    section("9. Limitations and next experiments",
            "Clean warmup is assumed. Empirical ranks can tie and shift between refits, particularly "
            "with small or contaminated reference sets. The contamination protection only trims fit "
            "candidates. UNSW duplicates are removed aggressively, changing its prevalence and "
            "subpopulation mix. The public replay cannot measure chronological drift. "
            "Test chronological datasets next, compare alternative drift statistics at a fixed "
            "alert budget, and evaluate independently adjudicated benign calibration.")
    section("10. Conclusion",
            "Adaptive anomaly detection should be tested as separate scoring and threshold policies. "
            "Track analyst-facing false positives alongside recall, ranking and update cost. "
            "The benchmark, per-batch dashboard, and ablation definitions make these distinctions "
            "reproducible.")
    p("References", "Heading2")
    p("[1] Liu et al., ACM TKDD (2012). [2] Breunig et al., SIGMOD (2000). [3] Schölkopf "
      "et al., Neural Computation (2001). [4] Hinton and Salakhutdinov, Science (2006). "
      "[5] Kingma and Welling, ICLR (2014). [6] Bifet and Gavaldà, SIAM SDM (2007). "
      "[7-11] UNSW-NB15 publications and academic data terms. Full citations and permanent "
      "links accompany the source in docs/METHODOLOGY.md.")

    def page(canvas, doc):
        canvas.setFont("ReportSans", 9)
        canvas.setFillColor(colors.HexColor(MUTED))
        canvas.drawString(52, 27, "Adaptive Threat Detection  |  Technical report")
        canvas.drawRightString(560, 27, str(doc.page))

    target = output / "adaptive-threat-detection-report.pdf"
    doc = SimpleDocTemplate(str(target), title="Adaptive Unsupervised Threat Detection",
                            leftMargin=52, rightMargin=52, topMargin=42, bottomMargin=42)
    doc.build(items, onFirstPage=page, onLaterPages=page)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("results/study.json"))
    parser.add_argument("--output", type=Path, default=Path("results"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    data = load_study(args.source)
    index = {(s["scenario"], s["model"], s["policy"]): s["metrics"] for s in data["summary"]}
    tradeoff_chart(index, ["gradual", "sudden", "unsw"], args.output / "tradeoffs.png")
    timeline_chart(data, args.output / "timeline.png")
    architecture_chart(args.output / "architecture.png")
    report(data, index, args.output)


if __name__ == "__main__":
    main()
