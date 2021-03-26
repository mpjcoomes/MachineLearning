library(ggplot2)

apa_style <-
  theme_bw(base_size=12,base_family='serif') %+replace%
  theme(axis.text=element_text(size=12),
        legend.text=element_text(size=12),
        axis.ticks=element_line(colour="black"),
        axis.line=element_line(),
        axis.title.x=element_text(vjust=-.1),
        axis.title.y=element_text(vjust=.3, angle=90),
        panel.background=element_blank(),
        panel.border=element_blank(),
        panel.grid.minor=element_blank(),
        panel.grid.major=element_blank(),
        plot.background=element_blank(),
        legend.background=element_blank(),
        legend.key=element_rect(fill=NA,colour=NA),
        legend.key.width=unit(12.5,"mm"),
        axis.ticks.margin=unit(1.5,"mm"),
        legend.title=element_text(face=NULL)   )

Curve <- data.frame(
  FPR=seq(0,100,length=1000),
  TPR=(100*sin(seq(0,pi/2,length=1000))),
  TPR2=100-(100*sin(seq(0,pi/2,length=1000)))[1000:1],
  TPR3=100*(1-dexp(seq(0,10,length=1000),1)))
ggsave("C.png",
       ggplot(aes(x=FPR,y=TPR),data=Curve) + geom_line(size=.2) +
         geom_line(aes(x=TPR2),size=.3) +
         geom_line(aes(y=TPR3),size=.4)+ apa_style +
         scale_y_continuous(limits=c(0,100.25),
                            expand=c(0,0),breaks=seq(0,100,20)) +
         scale_x_continuous(limits=c(0,100.22),
                            expand=c(0,0),breaks=seq(0,100,20)) +
         geom_segment(aes(x=50,y=50,xend=5,yend=95),
                      arrow=arrow(length=unit(0.5,"cm"))) +
         labs(x=expression(atop("False Positive Rate (%)",
                                atop(italic("Type I Error rate or 1-Specificity"),""))),
              y=expression(atop("True Positive Rate (%)",
                                atop(italic("Sensitivity"), "")))) +
         theme(plot.margin=unit(c(3,4,-2,-5),"mm"),
               axis.title.y=element_text(angle=90,vjust=.65)) +
         annotate("segment",x=0,xend=100,y=0,yend=100,linetype=2) +
         annotate("text",x=c(17,71,72,73,64),y=c(79,75,42,8,20),
                  parse=T,angle=c(-46,46,0,0,0),label=c(
                    "'better  models'",
                    "'line of no discrimination'","AUC",
                    "atop(italic(T)==italic('threshold parameter'),
    italic(P[0]*(T))==italic('probability of negative    '))",
                    "italic(integral(TPR*(T)*P[0]*(T)*dT,-infinity,infinity))"
                  ),family='serif',size=c(3,3,5,3,4)) +
         geom_area(aes(y=TPR3),stat="identity",alpha=.04) +
         geom_area(aes(x=TPR2),stat="identity",alpha=.06) +
         geom_area(aes(y=TPR),stat="identity",alpha=.09)
       , dpi=1000, width=4, height=4)
