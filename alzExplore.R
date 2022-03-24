#Preliminary heatmap for ADNI sample data in R
#Written by Leslie Smith on 3-21-2022

require(readr)  # for read_csv()
require(dplyr) #for dataframe manuipulation
library(ggplot2) # for graphing


platform.df <- readr::read_tsv("/blue/kgraim/leslie.smith1/MLGenomicsProj/copy_GPL15207-17536.txt")
adni.df <- readr::read_csv("/blue/kgraim/leslie.smith1/MLGenomicsProj/ADNIdata.csv")
t2t.df <- readr::read_tsv("/blue/kgraim/leslie.smith1/MLGenomicsProj/T2TmappedGenes.tsv")
colnames(t2t.df) <- c("gene_symbol","chrom")
temp1.df <- adni.df$Probs
temp2.df <- data.frame(platform.df$ID, platform.df$`Gene Symbol`)
colnames(temp2.df) <- c("prob","gene_symbol")
#----annoted samples from adni data ----
temp.3 <- temp2.df[temp2.df$prob %in% temp1.df,] # we lost samples from adni here: 49386 -> 49363
#--- figure out why we are losing samples ^^ ----
#temp4 <- subset(temp1.df, !(temp1.df %in% temp2.df$prob))

# unique telemere genes 
tst.unique <- unique(t2t.df) #only unique genes from t2t data 

#---- merge adni annotated samples with T2T annotated samples ----
adni_t2t <- merge(x=tst.unique, y=temp.3, by = "gene_symbol")
merged.df <- merge(x =adni_t2t , y = adni.df, by.x = "prob", by.y = "Probs")
merged.df <- merged.df[,2:ncol(merged.df)]
write.table(merged.df, file = "annotated_adni.tsv", sep='\t', col.names=T, row.names=F, quote=F)

#hierarchical clustering
df <- readr::read_tsv("/home/leslie.smith1/blue_kgraim/leslie.smith1/MLGenomicsProj/annotated_adni.tsv")

#normalization
to.log <- as.matrix(df[,4:747]) 
to.log <- to.log + 1
logged <- log(to.log)
logged <- logged - rowMedians(logged)
log_scaled <- scale(logged, center = TRUE, scale = TRUE)
rownames(log_scaled) <- df$gene_symbol
#find variance
genes.var <- apply(log_scaled, 1, var)#gets sample variance of rows aka genes 
genes.var.select <- order(-genes.var)[1:50] #index of top 50 most varied genes
log_scaled.s <- log_scaled[genes.var.select, ] #get most variable genes 
transpose_log_scaled.s <- t(log_scaled.s)
#heatmap with default dendrogram
heatmap(transpose_log_scaled.s, xlab = "Gene", ylab = "ADNI sample", main="Top 50 ADNI Genes of Highest Variance")
#heatmap with pearson correlation at dendrogram
hr <- hclust(as.dist(1-cor(t(transpose_log_scaled.s), method = "pearson")), method="complete")
hc <- hclust(as.dist(1-cor(transpose_log_scaled.s, method = "pearson")), method="complete")
heatmap(transpose_log_scaled.s, Rowv = as.dendrogram(hr), Colv = as.dendrogram(hc), xlab = "Gene", 
        ylab = "ADNI sample", main="Top 50 ADNI Genes of Highest Variance")





