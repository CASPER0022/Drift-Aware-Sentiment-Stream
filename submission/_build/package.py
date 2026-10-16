"""Assemble the submission folder and the team-prefixed copies for the LMS.

submission/
  report/        Final_Report.pdf + Final_Report_Source/ (editable LaTeX) + its zip
  video/         Presentation.pptx + Video_and_Run_Guide.pdf (from build_slides / build_guide)
  code/          Source_Code.zip (+ README.md)
  18_LMS_Upload/ every file to upload, renamed with the team prefix "18_"

The code zip holds the project's git-tracked files (plus new, not yet ignored ones) except the
datasets, the virtual environment, the reference papers, the course's own template and
instruction files, and the submission folder itself.

Usage:  python submission/_build/package.py
"""
import shutil
import subprocess
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SUB = ROOT / "submission"
TEAM = "18"
ZIP_ROOT = "Drift-Aware-Sentiment-Stream"
EXCLUDE_PREFIXES = ("submission/", "Reference Materials/", "data/", ".venv/")
EXCLUDE_FILES = {"presentation template.pptx", "IOE413-Mini Project Instructions-Sep 2026.pdf"}
REPORT_SOURCE = ["main.tex", "refs.bib", "figures", "tables"]


def project_files() -> list[str]:
    out = subprocess.run(["git", "ls-files", "-co", "--exclude-standard"], cwd=ROOT,
                         capture_output=True, text=True, check=True).stdout.splitlines()
    return sorted(f for f in out if not f.startswith(EXCLUDE_PREFIXES) and f not in EXCLUDE_FILES
                  and (ROOT / f).is_file())


def zip_dir(src: Path, dest: Path, arc_root: str) -> None:
    with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(src.rglob("*")):
            if f.is_file():
                z.write(f, f"{arc_root}/{f.relative_to(src).as_posix()}")


def main() -> None:
    report_dir, code_dir, upload = SUB / "report", SUB / "code", SUB / f"{TEAM}_LMS_Upload"
    for d in (report_dir, code_dir, upload):
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True)

    # Report: PDF + editable source
    shutil.copy2(ROOT / "report" / "report.pdf", report_dir / "Final_Report.pdf")
    src = report_dir / "Final_Report_Source"
    src.mkdir()
    for name in REPORT_SOURCE:
        p = ROOT / "report" / name
        (shutil.copytree if p.is_dir() else shutil.copy2)(p, src / name)
    (src / "README.txt").write_text(
        "IEEE conference paper (IEEEtran). Compile main.tex with pdflatex + bibtex, latexmk, "
        "tectonic, or upload this folder to Overleaf.\n")
    zip_dir(src, report_dir / "Final_Report_Source.zip", "Final_Report_Source")

    # Code
    files = project_files()
    with zipfile.ZipFile(code_dir / "Source_Code.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for f in files:
            z.write(ROOT / f, f"{ZIP_ROOT}/{f}")
    shutil.copy2(ROOT / "README.md", code_dir / "README.md")

    # Renamed copies for the LMS
    copies = {
        report_dir / "Final_Report.pdf": "Final_Report.pdf",
        report_dir / "Final_Report_Source.zip": "Final_Report_Source.zip",
        SUB / "video" / "Presentation.pptx": "Presentation.pptx",
        code_dir / "Source_Code.zip": "Source_Code.zip",
        ROOT / "README.md": "README.md",
    }
    for source, name in copies.items():
        shutil.copy2(source, upload / f"{TEAM}_{name}")
    (upload / f"ADD {TEAM}_Demo_Video.mp4 HERE - read me.txt").write_text(
        f"Record the video with submission/video/Video_and_Run_Guide.pdf, export it as MP4 "
        f"(at most 6 minutes), name it {TEAM}_Demo_Video.mp4, put it in this folder and delete "
        f"this note. Only the team lead uploads the files.\n")

    print(f"code zip: {len(files)} files")
    for f in sorted(SUB.rglob("*")):
        if f.is_file() and "_build" not in f.parts:
            print(f"  {f.relative_to(SUB).as_posix():60s} {f.stat().st_size / 1024:8.0f} KB")


if __name__ == "__main__":
    main()
