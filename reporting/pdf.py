"""
Jal-Rakshak — Incident Report PDF Generator
=============================================
Compiles structured IncidentReport data into an official, publication-quality
maritime intelligence investigation PDF using ReportLab.

Adheres strictly to scientific credibility:
- Explicitly flags data classification on every section (OBSERVED / INFERRED / PREDICTED / SIMULATED)
- Uses non-adjudicative candidate vessel terminology
- Disclaims legal liability and highlights model uncertainty bounds
"""

import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    HRFlowable,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.pdfgen import canvas


class NumberedCanvas(canvas.Canvas):
    """Two-pass canvas for dynamic 'Page X of Y' footers."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_number(num_pages)
            super().showPage()
        super().save()

    def draw_page_number(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#718096"))

        # Running header (pages 2+)
        if self._pageNumber > 1:
            self.drawString(
                54, 750,
                "JAL-RAKSHAK MARITIME INTELLIGENCE // INVESTIGATION DOSSIER"
            )
            self.setStrokeColor(colors.HexColor("#E2E8F0"))
            self.setLineWidth(0.5)
            self.line(54, 744, 558, 744)

        # Running footer
        page_text = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(558, 36, page_text)
        self.drawString(
            54, 36,
            "OFFICIAL DECISION SUPPORT // NOT LEGAL ADJUDICATION // RESTRICTED DISSEMINATION"
        )
        self.setStrokeColor(colors.HexColor("#E2E8F0"))
        self.setLineWidth(0.5)
        self.line(54, 48, 558, 48)
        self.restoreState()


def generate_pdf_report(report_data: Dict[str, Any], output_path: str) -> str:
    """
    Generate an official PDF investigation report from report data dictionary.

    Args:
        report_data: Dictionary returned by IncidentReport.to_dict()
        output_path: Path where the PDF should be written.

    Returns:
        Absolute path to generated PDF.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54,
    )

    styles = getSampleStyleSheet()

    # Custom color palette
    c_primary = colors.HexColor("#1A365D")      # Deep Navy
    c_secondary = colors.HexColor("#2B6CB0")    # Maritime Blue
    c_accent = colors.HexColor("#E53E3E")       # Red alert
    c_dark = colors.HexColor("#2D3748")         # Slate text
    c_light = colors.HexColor("#F7FAFC")        # Off-white bg
    c_border = colors.HexColor("#CBD5E0")       # Border grey
    c_badge_obs = colors.HexColor("#2F855A")    # Green (Observed)
    c_badge_inf = colors.HexColor("#DD6B20")    # Orange (Inferred)
    c_badge_pred = colors.HexColor("#3182CE")   # Blue (Predicted)
    c_badge_sim = colors.HexColor("#805AD5")    # Purple (Simulated)

    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=24,
        textColor=c_primary,
    )

    subtitle_style = ParagraphStyle(
        "DocSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#4A5568"),
    )

    h1_style = ParagraphStyle(
        "SectionH1",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=13,
        leading=17,
        textColor=c_primary,
        spaceBefore=14,
        spaceAfter=6,
    )

    body_style = ParagraphStyle(
        "DocBody",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=13,
        textColor=c_dark,
    )

    bold_body_style = ParagraphStyle(
        "DocBodyBold",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=9,
        leading=13,
        textColor=c_dark,
    )

    disclaimer_style = ParagraphStyle(
        "DocDisclaimer",
        parent=styles["Normal"],
        fontName="Helvetica-Oblique",
        fontSize=8,
        leading=11,
        textColor=colors.HexColor("#718096"),
    )

    story = []

    # ── Header Banner ──
    incident_id = report_data.get("incident_id", "JR-UNKNOWN")
    generated_at = report_data.get("generated_at", datetime.now(timezone.utc).isoformat())

    header_data = [
        [
            Paragraph("<b>JAL-RAKSHAK MARITIME PLATFORM</b><br/><font size=8>AI-Assisted Oil Spill Intelligence & Early Warning</font>", body_style),
            Paragraph(f"<b>INCIDENT ID:</b> {incident_id}<br/><b>GENERATED:</b> {generated_at[:19]} UTC", ParagraphStyle("RightH", parent=body_style, alignment=2)),
        ]
    ]
    t_header = Table(header_data, colWidths=[300, 204])
    t_header.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(t_header)
    story.append(HRFlowable(width="100%", thickness=1.5, color=c_primary, spaceBefore=4, spaceAfter=10))

    story.append(Paragraph("TECHNICAL INCIDENT INVESTIGATION REPORT", title_style))
    story.append(Paragraph("FORENSIC DETECTION, SOURCE ATTRIBUTION & COASTAL THREAT ASSESSMENT", subtitle_style))
    story.append(Spacer(1, 10))

    # ── Executive Classification Banner ──
    story.append(Table(
        [[
            Paragraph(
                "<b>CLASSIFICATION NOTICE:</b> Contains analytical assessments derived from synthetic/demo SAR telemetry, "
                "Euler hindcast particle integration, and AIS track correlation. "
                "Candidate vessel ratings denote <i>statistical association</i> and do NOT constitute legal proof of culpability.",
                disclaimer_style
            )
        ]],
        colWidths=[504],
        style=[
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#FEFCBF")),
            ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#D69E2E")),
            ("PADDING", (0, 0), (-1, -1), 8),
        ]
    ))
    story.append(Spacer(1, 12))

    # ── Iterate through Sections ──
    sections = report_data.get("sections", [])
    for sec in sections:
        title = sec.get("title", "Section")
        classification = sec.get("data_classification", "INFERRED")
        content = sec.get("content", {})

        # Color-coded badge
        badge_bg = c_badge_inf
        if classification == "OBSERVED":
            badge_bg = c_badge_obs
        elif classification == "PREDICTED":
            badge_bg = c_badge_pred
        elif classification == "SIMULATED":
            badge_bg = c_badge_sim

        sec_header = [
            [
                Paragraph(f"<b>{title.upper()}</b>", h1_style),
                Paragraph(f"<font color='white'><b>{classification}</b></font>", ParagraphStyle("Badge", parent=body_style, alignment=1, fontSize=8)),
            ]
        ]
        t_sec_h = Table(sec_header, colWidths=[400, 104])
        t_sec_h.setStyle(TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("BACKGROUND", (1, 0), (1, 0), badge_bg),
            ("ALIGN", (1, 0), (1, 0), "CENTER"),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
            ("TOPPADDING", (0, 0), (-1, -1), 2),
        ]))

        sec_flowables = [t_sec_h, HRFlowable(width="100%", thickness=0.5, color=c_border, spaceBefore=2, spaceAfter=6)]

        # Render content based on section type
        if isinstance(content, dict):
            # Special table formatting for Candidate Vessels
            if "Candidate" in title or "Vessel" in title:
                candidates = content.get("top_candidates", [])
                if candidates:
                    v_data = [["Rank", "MMSI", "Vessel Name", "Min Dist", "Time Δ", "Score", "Anomalies"]]
                    for idx, c in enumerate(candidates[:5]):
                        bd = c.get("breakdown", {})
                        anomalies = []
                        if bd.get("behavioral", 0) > 40:
                            anomalies.append("Behavioral")
                        v_data.append([
                            str(idx + 1),
                            str(c.get("mmsi", "—")),
                            str(c.get("name", "—"))[:18],
                            f"{c.get('features', {}).get('min_distance_to_source_km', '—')} km" if "features" in c else "—",
                            f"{c.get('features', {}).get('time_diff_to_event_min', '—')} m" if "features" in c else "—",
                            f"{c.get('score', 0):.1f}%",
                            ", ".join(anomalies) if anomalies else "None",
                        ])
                    t_cand = Table(v_data, colWidths=[36, 68, 120, 68, 60, 56, 96])
                    t_cand.setStyle(TableStyle([
                        ("BACKGROUND", (0, 0), (-1, 0), c_primary),
                        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                        ("FONTSIZE", (0, 0), (-1, -1), 8),
                        ("GRID", (0, 0), (-1, -1), 0.5, c_border),
                        ("ALIGN", (0, 0), (0, -1), "CENTER"),
                        ("ALIGN", (3, 0), (5, -1), "CENTER"),
                        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, c_light]),
                        ("PADDING", (0, 0), (-1, -1), 4),
                    ]))
                    sec_flowables.append(t_cand)
                    sec_flowables.append(Spacer(1, 4))
                else:
                    sec_flowables.append(Paragraph("No candidate vessels correlated.", body_style))

            else:
                # Format key-value pairs
                table_rows = []
                for k, v in content.items():
                    if k in ["mask", "polygon", "trajectory", "history"]:
                        continue
                    clean_k = k.replace("_", " ").title()
                    if isinstance(v, float):
                        clean_v = f"{v:.4f}" if abs(v) < 1.0 else f"{v:.2f}"
                    elif isinstance(v, (list, dict)):
                        clean_v = str(v)[:80] + ("..." if len(str(v)) > 80 else "")
                    else:
                        clean_v = str(v)
                    table_rows.append([
                        Paragraph(f"<b>{clean_k}</b>", bold_body_style),
                        Paragraph(clean_v, body_style),
                    ])

                if table_rows:
                    t_kv = Table(table_rows, colWidths=[180, 324])
                    t_kv.setStyle(TableStyle([
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [colors.white, c_light]),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                        ("TOPPADDING", (0, 0), (-1, -1), 3),
                        ("LINEBELOW", (0, 0), (-1, -1), 0.5, colors.HexColor("#EDF2F7")),
                    ]))
                    sec_flowables.append(t_kv)

        sec_flowables.append(Spacer(1, 8))
        story.append(KeepTogether(sec_flowables))

    # ── Limitations & Data Sources ──
    limits = report_data.get("limitations", [])
    if limits:
        story.append(KeepTogether([
            Paragraph("<b>METHODOLOGICAL LIMITATIONS & UNCERTAINTIES</b>", h1_style),
            HRFlowable(width="100%", thickness=0.5, color=c_border, spaceBefore=2, spaceAfter=6),
            *[Paragraph(f"• {lim}", body_style) for lim in limits],
            Spacer(1, 8),
        ]))

    sources = report_data.get("data_sources", [])
    if sources:
        story.append(KeepTogether([
            Paragraph("<b>INTEGRATED DATA PROVIDERS & MODELS</b>", h1_style),
            HRFlowable(width="100%", thickness=0.5, color=c_border, spaceBefore=2, spaceAfter=6),
            Paragraph("<b>Primary Feeds:</b> " + ", ".join(sources), body_style),
            Spacer(1, 12),
        ]))

    # ── Final Sign-Off Block ──
    signoff_data = [
        [
            Paragraph("<b>SYSTEM ANALYST:</b><br/>Automated Algorithmic Dispatch", body_style),
            Paragraph("<b>REVIEWING OFFICER:</b><br/>___________________________", body_style),
            Paragraph("<b>COMMAND ACTION:</b><br/>[ ] DISPATCH INVESTIGATION<br/>[ ] ESCALATE ALERT<br/>[ ] ARCHIVE RECORD", body_style),
        ]
    ]
    t_sign = Table(signoff_data, colWidths=[160, 160, 184])
    t_sign.setStyle(TableStyle([
        ("BOX", (0, 0), (-1, -1), 1, c_border),
        ("BACKGROUND", (0, 0), (-1, -1), c_light),
        ("PADDING", (0, 0), (-1, -1), 8),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    story.append(KeepTogether([t_sign]))

    doc.build(story, canvasmaker=NumberedCanvas)
    return os.path.abspath(output_path)
