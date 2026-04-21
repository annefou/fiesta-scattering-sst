rule all:
    input:
        "results/.gitkeep"

rule run_notebook:
    input:
        "01_sst_gap_filling.py"
    output:
        touch("results/.gitkeep")
    shell:
        "jupytext --to notebook --execute {input}"
