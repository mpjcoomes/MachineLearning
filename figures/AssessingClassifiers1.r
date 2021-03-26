library(xtable)

z <- xtable(data.frame(
  .=c("PD",""),
  ..=c("D1","ND0"),
  D1=c("\\cellcolor[gray]{0.85}$TruePositive$","$FalseNegative$"),
  ND0=c("$FalsePositive$","\\cellcolor[gray]{0.85}$TrueNegative$")),
  align="llrcr")
z <- print(z,type="latex",floating=F,include.rownames=FALSE,
           booktabs=T,sanitize.text.function = function(x){x},
           add.to.row=list(pos=list(-1),
           command=c("\\toprule & & \\multicolumn{2}{c}{Real Outcome} \\\\ \\cline{3-4} \\noalign{\\smallskip}")))
z <- gsub("D1", "Default [1]",z)
z <- gsub(". & .. &", " &  &",z)
z <- gsub("ND0", "Not Default [0]",z)
z <- gsub("PD", "\\\\multirow{2}{*}{Predicted Outcome}",z)
z <- gsub("smallskip} \\\\toprule", "smallskip}", z)
write(c("\\documentclass{article}
  \\usepackage{booktabs}
  \\usepackage{colortbl}
  \\pagenumbering{gobble}
  \\usepackage{multirow}
  \\begin{document}",z,
 "\\end{document}"),file="T.tex")
system("latex T.tex")
system("dvipng -T tight -bg 'rgb 1.0 1.0 1.0' T.dvi -o B.png -D 400")
