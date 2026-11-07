# HUST Beamer theme (vendored)

The `.sty` files and the red 16:9 assets in this folder are copied unchanged from [k1m1nonawa4869/HUST-Beamer-Theme](https://github.com/k1m1nonawa4869/HUST-Beamer-Theme) at commit `15102a2`, so the deck compiles without a separate checkout. Only the files needed for `color=red` and `aspectratio=169` are included; the theme's optional `hust-setup.tex` is not used.

The upstream repository has no license file. Credit belongs to its author; check with them before redistributing the theme outside this project.

`slide/main.tex` finds these files through `\input@path`, so no `TEXINPUTS` setup is needed. The theme requires the `pgfopts` package, and uses the Lato and TeX Gyre Adventor fonts when installed (falling back to Arial otherwise) — see `slide/requirements.txt`.
