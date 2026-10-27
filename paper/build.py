# SPDX-License-Identifier: MIT
"""Build in an isolated local directory and require a clean final TeX log."""
import os
for name in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"):
    os.environ[name]="1"
import re
import shutil
import subprocess
from pathlib import Path
PAPER=Path(__file__).resolve().parent
BUILD=PAPER/".build"
LOGS=PAPER.parent/"verify"/"logs"
unnumbered=re.compile(r"\\\[(?:(?!\\\]).)*?\\label\{",re.S)
for tex in [PAPER/"main.tex",*sorted((PAPER/"sections").glob("*.tex"))]:
    if unnumbered.search(tex.read_text(encoding="utf-8")):
        raise SystemExit(f"Label inside an unnumbered display in {tex.name}")
BUILD.mkdir(exist_ok=True)
LOGS.mkdir(parents=True,exist_ok=True)
for p in BUILD.iterdir():
    if p.is_file() and p.suffix in {".aux",".bbl",".blg",".log",".out",".toc",".pdf"}:
        p.unlink()
env=os.environ.copy()
env["SOURCE_DATE_EPOCH"]="1793088000"
env["FORCE_SOURCE_DATE"]="1"
env["TEXMF_OUTPUT_DIRECTORY"]=str(BUILD)
env["BIBINPUTS"]=str(PAPER)+os.pathsep+env.get("BIBINPUTS","")
commands=[
    ["pdflatex","-interaction=nonstopmode","-halt-on-error","-file-line-error","-output-directory="+str(BUILD),"main.tex"],
    ["bibtex","main"],
    ["pdflatex","-interaction=nonstopmode","-halt-on-error","-file-line-error","-output-directory="+str(BUILD),"main.tex"],
    ["pdflatex","-interaction=nonstopmode","-halt-on-error","-file-line-error","-output-directory="+str(BUILD),"main.tex"],
]
for i,cmd in enumerate(commands):
    result=subprocess.run(cmd,cwd=BUILD if cmd[0]=="bibtex" else PAPER,env=env,
                          stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding="utf-8",errors="replace")
    (LOGS/f"build-pass-{i+1}.txt").write_text(result.stdout,encoding="utf-8")
    if result.returncode:
        print(result.stdout[-15000:])
        raise SystemExit(f"Build command failed: {cmd[0]}")
final=(BUILD/"main.log").read_text(encoding="utf-8",errors="replace")
bib=(BUILD/"main.blg").read_text(encoding="utf-8",errors="replace")
bad=[x for x in final.splitlines() if re.search(r"(?:LaTeX(?: Font)?|Package .+|Class .+) Warning:|Overfull|Underfull|undefined|multiply defined",x,re.I)]
bad+= [x for x in bib.splitlines() if re.search(r"Warning--|error message",x,re.I)]
(LOGS/"latex-final.log").write_text(final,encoding="utf-8")
(LOGS/"bibtex-final.log").write_text(bib,encoding="utf-8")
if bad:
    print("\n".join(bad)); raise SystemExit("Final build has warnings or bad boxes")
shutil.copy2(BUILD/"main.pdf",PAPER/"main.pdf")
summary="PASS: final LaTeX pass has zero warnings, undefined references, duplicate labels, or bad boxes.\nPASS: BibTeX has zero warnings.\n"
(LOGS/"build-validation.txt").write_text(summary,encoding="utf-8")
print(summary,end="")
print("PDF: "+str(PAPER/"main.pdf"))
