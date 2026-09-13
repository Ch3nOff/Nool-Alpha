"""
Script to convert COMPLETE_OVERALL_EMPIRICAL_REPORT.md into a beautifully formatted .docx document.
"""

import os
import sys
import re

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import parse_xml, OxmlElement
from docx.oxml.ns import nsdecls, qn

MD_FILE = "COMPLETE_OVERALL_EMPIRICAL_REPORT.md"
DOCX_FILE = "COMPLETE_OVERALL_EMPIRICAL_REPORT.docx"


def set_cell_shading(cell, color_hex):
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{color_hex}"/>')
    cell._tc.get_or_add_tcPr().append(shd)


def set_cell_borders(cell, color="D0D7DE", sz="4", val="single"):
    tcPr = cell._tc.get_or_add_tcPr()
    tcBorders = parse_xml(
        f'<w:tcBorders {nsdecls("w")}>'
        f'<w:top w:val="{val}" w:sz="{sz}" w:space="0" w:color="{color}"/>'
        f'<w:bottom w:val="{val}" w:sz="{sz}" w:space="0" w:color="{color}"/>'
        f'<w:left w:val="{val}" w:sz="{sz}" w:space="0" w:color="{color}"/>'
        f'<w:right w:val="{val}" w:sz="{sz}" w:space="0" w:color="{color}"/>'
        f'</w:tcBorders>'
    )
    tcPr.append(tcBorders)


def set_cell_margins(cell, top=120, bottom=120, left=150, right=150):
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = parse_xml(
        f'<w:tcMar {nsdecls("w")}>'
        f'<w:top w:w="{top}" w:type="dxa"/>'
        f'<w:bottom w:w="{bottom}" w:type="dxa"/>'
        f'<w:left w:w="{left}" w:type="dxa"/>'
        f'<w:right w:w="{right}" w:type="dxa"/>'
        f'</w:tcMar>'
    )
    tcPr.append(tcMar)


def add_formatted_text(paragraph, text, default_font="Calibri", default_size=10.5, default_color=None):
    """
    Parses inline markdown like **bold**, *italic*, `code`, and plain text.
    """
    # Regex to tokenize markdown inline spans: **bold**, *italic*, `code`
    token_pattern = re.compile(r'(\*\*.*?\*\*|\*.*?\*|`.*?`|\[.*?\]\(.*?\)|<span.*?>.*?</span>)')
    pos = 0
    for match in token_pattern.finditer(text):
        start, end = match.span()
        # Plain text before match
        if start > pos:
            run = paragraph.add_run(text[pos:start])
            run.font.name = default_font
            run.font.size = Pt(default_size)
            if default_color:
                run.font.color.rgb = default_color

        token = match.group(0)
        if token.startswith("**") and token.endswith("**"):
            run = paragraph.add_run(token[2:-2])
            run.font.name = default_font
            run.font.size = Pt(default_size)
            run.bold = True
            if default_color:
                run.font.color.rgb = default_color
        elif token.startswith("*") and token.endswith("*"):
            run = paragraph.add_run(token[1:-1])
            run.font.name = default_font
            run.font.size = Pt(default_size)
            run.italic = True
            if default_color:
                run.font.color.rgb = default_color
        elif token.startswith("`") and token.endswith("`"):
            run = paragraph.add_run(token[1:-1])
            run.font.name = "Consolas"
            run.font.size = Pt(default_size * 0.92)
            run.font.color.rgb = RGBColor(0x9A, 0x1F, 0x40) # Crimson code color
        elif token.startswith("<span") and "</span>" in token:
            # Strip html tags
            clean_t = re.sub(r'<.*?>', '', token)
            run = paragraph.add_run(clean_t)
            run.font.name = default_font
            run.font.size = Pt(default_size)
            run.bold = True
            if "color:green" in token:
                run.font.color.rgb = RGBColor(0x1B, 0x7E, 0x3E)
            elif "color:red" in token:
                run.font.color.rgb = RGBColor(0xCF, 0x22, 0x2E)
        elif token.startswith("[") and "](" in token:
            link_text = token[1:token.index("](")]
            run = paragraph.add_run(link_text)
            run.font.name = default_font
            run.font.size = Pt(default_size)
            run.font.color.rgb = RGBColor(0x09, 0x69, 0xDA)
            run.underline = True
        pos = end

    if pos < len(text):
        run = paragraph.add_run(text[pos:])
        run.font.name = default_font
        run.font.size = Pt(default_size)
        if default_color:
            run.font.color.rgb = default_color


def build_docx_from_markdown():
    if not os.path.exists(MD_FILE):
        print(f"Error: {MD_FILE} does not exist.")
        return

    with open(MD_FILE, "r", encoding="utf-8") as f:
        lines = f.readlines()

    doc = Document()

    # Configure Margins (1 inch)
    for section in doc.sections:
        section.top_margin = Inches(1.0)
        section.bottom_margin = Inches(1.0)
        section.left_margin = Inches(1.0)
        section.right_margin = Inches(1.0)

    # Base Colors
    COLOR_PRIMARY = RGBColor(0x1B, 0x36, 0x5D)   # Deep Navy Blue
    COLOR_SECONDARY = RGBColor(0x2E, 0x5B, 0x88) # Slate Blue
    COLOR_TERTIARY = RGBColor(0x4A, 0x69, 0x84)  # Dark Steel
    COLOR_TEXT = RGBColor(0x24, 0x29, 0x2F)      # Dark Charcoal Text
    COLOR_MUTED = RGBColor(0x57, 0x60, 0x6A)     # Muted Gray

    i = 0
    n = len(lines)

    while i < n:
        raw_line = lines[i]
        line = raw_line.strip()

        # 1. Empty lines
        if not line:
            i += 1
            continue

        # 2. Horizontal divider
        if line == "---":
            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(6)
            p.paragraph_format.space_after = Pt(6)
            pBdr = parse_xml(f'<w:pBdr {nsdecls("w")}><w:bottom w:val="single" w:sz="6" w:space="1" w:color="D0D7DE"/></w:pBdr>')
            p._p.get_or_add_pPr().append(pBdr)
            i += 1
            continue

        # 3. Document Title (# Title)
        if line.startswith("# ") and i == 0:
            title_text = line[2:].strip()
            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(0)
            p.paragraph_format.space_after = Pt(4)
            run = p.add_run(title_text)
            run.font.name = "Calibri"
            run.font.size = Pt(22)
            run.bold = True
            run.font.color.rgb = COLOR_PRIMARY
            i += 1
            continue

        # 4. Document Subtitle (## Subtitle right after Title)
        if line.startswith("## ") and i <= 3:
            sub_text = line[3:].strip()
            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(0)
            p.paragraph_format.space_after = Pt(14)
            run = p.add_run(sub_text)
            run.font.name = "Calibri"
            run.font.size = Pt(13)
            run.italic = True
            run.font.color.rgb = COLOR_MUTED
            i += 1
            continue

        # 5. Heading 1 (# or ##)
        if line.startswith("## "):
            h_text = line[3:].strip()
            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(16)
            p.paragraph_format.space_after = Pt(6)
            p.paragraph_format.keep_with_next = True
            run = p.add_run(h_text)
            run.font.name = "Calibri"
            run.font.size = Pt(15)
            run.bold = True
            run.font.color.rgb = COLOR_PRIMARY
            # Add bottom accent line
            pBdr = parse_xml(f'<w:pBdr {nsdecls("w")}><w:bottom w:val="single" w:sz="6" w:space="2" w:color="1B365D"/></w:pBdr>')
            p._p.get_or_add_pPr().append(pBdr)
            i += 1
            continue

        # 6. Heading 2 (###)
        if line.startswith("### "):
            h_text = line[4:].strip()
            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(12)
            p.paragraph_format.space_after = Pt(4)
            p.paragraph_format.keep_with_next = True
            run = p.add_run(h_text)
            run.font.name = "Calibri"
            run.font.size = Pt(12.5)
            run.bold = True
            run.font.color.rgb = COLOR_SECONDARY
            i += 1
            continue

        # 7. Heading 3 (####)
        if line.startswith("#### "):
            h_text = line[5:].strip()
            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(9)
            p.paragraph_format.space_after = Pt(3)
            p.paragraph_format.keep_with_next = True
            run = p.add_run(h_text)
            run.font.name = "Calibri"
            run.font.size = Pt(11)
            run.bold = True
            run.font.color.rgb = COLOR_TERTIARY
            i += 1
            continue

        # 8. Code Block (``` ... ```)
        if line.startswith("```"):
            code_lang = line[3:].strip()
            code_lines = []
            i += 1
            while i < n and not lines[i].strip().startswith("```"):
                code_lines.append(lines[i].rstrip("\n"))
                i += 1
            if i < n:
                i += 1 # skip closing ```

            code_content = "\n".join(code_lines)
            table = doc.add_table(rows=1, cols=1)
            table.alignment = WD_TABLE_ALIGNMENT.CENTER
            table.autofit = False
            cell = table.cell(0, 0)
            cell.width = Inches(6.5)
            set_cell_shading(cell, "F6F8FA")
            set_cell_borders(cell, color="D0D7DE", sz="4", val="single")
            set_cell_margins(cell, top=100, bottom=100, left=140, right=140)

            cp = cell.paragraphs[0]
            cp.paragraph_format.space_before = Pt(2)
            cp.paragraph_format.space_after = Pt(2)
            cp.paragraph_format.line_spacing = 1.05
            crun = cp.add_run(code_content)
            crun.font.name = "Consolas"
            crun.font.size = Pt(8.5)
            crun.font.color.rgb = RGBColor(0x1F, 0x23, 0x28)

            # Extra space after table
            spacer = doc.add_paragraph()
            spacer.paragraph_format.space_before = Pt(0)
            spacer.paragraph_format.space_after = Pt(4)
            continue

        # 9. Markdown Table (| col | col |)
        if line.startswith("|") and "|" in line[1:]:
            table_lines = []
            while i < n and lines[i].strip().startswith("|"):
                table_lines.append(lines[i].strip())
                i += 1

            if len(table_lines) >= 2:
                # Parse rows
                def parse_table_row(row_str):
                    parts = [p.strip() for p in row_str.strip("|").split("|")]
                    return parts

                header_cells = parse_table_row(table_lines[0])
                # Skip separator line (line 1)
                data_rows = []
                for row_idx in range(1, len(table_lines)):
                    r = parse_table_row(table_lines[row_idx])
                    if all(set(c).issubset({'-', ':', ' '}) for c in r):
                        continue # separator row
                    data_rows.append(r)

                num_cols = len(header_cells)
                num_rows = 1 + len(data_rows)
                docx_table = doc.add_table(rows=num_rows, cols=num_cols)
                docx_table.alignment = WD_TABLE_ALIGNMENT.CENTER

                # Format Header Row
                hdr_row = docx_table.rows[0]
                for c_idx, text in enumerate(header_cells):
                    cell = hdr_row.cells[c_idx]
                    set_cell_shading(cell, "1B365D") # Navy Header
                    set_cell_borders(cell, color="0D233A", sz="4", val="single")
                    set_cell_margins(cell, top=100, bottom=100, left=120, right=120)
                    cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
                    p = cell.paragraphs[0]
                    p.paragraph_format.space_before = Pt(2)
                    p.paragraph_format.space_after = Pt(2)
                    run = p.add_run(text)
                    run.font.name = "Calibri"
                    run.font.size = Pt(9.5)
                    run.bold = True
                    run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)

                # Format Data Rows with Zebra Striping
                for r_idx, row_data in enumerate(data_rows):
                    docx_row = docx_table.rows[r_idx + 1]
                    bg_color = "F9FAFB" if (r_idx % 2 == 1) else "FFFFFF"
                    for c_idx in range(num_cols):
                        cell = docx_row.cells[c_idx]
                        set_cell_shading(cell, bg_color)
                        set_cell_borders(cell, color="E1E4E8", sz="4", val="single")
                        set_cell_margins(cell, top=80, bottom=80, left=120, right=120)
                        cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
                        p = cell.paragraphs[0]
                        p.paragraph_format.space_before = Pt(2)
                        p.paragraph_format.space_after = Pt(2)
                        cell_text = row_data[c_idx] if c_idx < len(row_data) else ""
                        add_formatted_text(p, cell_text, default_font="Calibri", default_size=9.0, default_color=COLOR_TEXT)

                # Space after table
                spacer = doc.add_paragraph()
                spacer.paragraph_format.space_before = Pt(0)
                spacer.paragraph_format.space_after = Pt(6)
            continue

        # 10. Callout Block (> quote / analysis)
        if line.startswith(">"):
            quote_lines = []
            while i < n and lines[i].strip().startswith(">"):
                quote_lines.append(lines[i].strip()[1:].strip())
                i += 1

            quote_text = " ".join(quote_lines)
            tbl = doc.add_table(rows=1, cols=1)
            tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
            cell = tbl.cell(0, 0)
            cell.width = Inches(6.5)
            set_cell_shading(cell, "F0F4F8") # Soft blue-gray
            # Left thick navy border
            tcPr = cell._tc.get_or_add_tcPr()
            tcBorders = parse_xml(
                f'<w:tcBorders {nsdecls("w")}>'
                f'<w:top w:val="none"/>'
                f'<w:bottom w:val="none"/>'
                f'<w:left w:val="single" w:sz="24" w:space="0" w:color="1B365D"/>'
                f'<w:right w:val="none"/>'
                f'</w:tcBorders>'
            )
            tcPr.append(tcBorders)
            set_cell_margins(cell, top=100, bottom=100, left=150, right=150)

            qp = cell.paragraphs[0]
            qp.paragraph_format.space_before = Pt(2)
            qp.paragraph_format.space_after = Pt(2)
            add_formatted_text(qp, quote_text, default_font="Calibri", default_size=10.0, default_color=COLOR_TEXT)

            spacer = doc.add_paragraph()
            spacer.paragraph_format.space_before = Pt(0)
            spacer.paragraph_format.space_after = Pt(4)
            continue

        # 11. Bullet Points (- or *)
        if line.startswith("- ") or line.startswith("* "):
            bullet_text = line[2:].strip()
            p = doc.add_paragraph(style="List Bullet")
            p.paragraph_format.space_before = Pt(1)
            p.paragraph_format.space_after = Pt(2)
            p.paragraph_format.line_spacing = 1.15
            add_formatted_text(p, bullet_text, default_font="Calibri", default_size=10.5, default_color=COLOR_TEXT)
            i += 1
            continue

        # 12. Numbered List (1. , 2. )
        num_match = re.match(r'^(\d+)\.\s+(.*)', line)
        if num_match:
            num_val = num_match.group(1)
            item_text = num_match.group(2)
            p = doc.add_paragraph(style="List Number")
            p.paragraph_format.space_before = Pt(2)
            p.paragraph_format.space_after = Pt(2)
            p.paragraph_format.line_spacing = 1.15
            add_formatted_text(p, item_text, default_font="Calibri", default_size=10.5, default_color=COLOR_TEXT)
            i += 1
            continue

        # 13. Regular Paragraph
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(2)
        p.paragraph_format.space_after = Pt(6)
        p.paragraph_format.line_spacing = 1.15
        add_formatted_text(p, line, default_font="Calibri", default_size=10.5, default_color=COLOR_TEXT)
        i += 1

    doc.save(DOCX_FILE)
    print(f"✅ Successfully created Word document: {DOCX_FILE} ({os.path.getsize(DOCX_FILE) / 1024:.1f} KB)")


if __name__ == "__main__":
    build_docx_from_markdown()
