#Gene Set Enrichment Analysis for AD - ML in Genomics Project 

library(clusterProfiler)  #for gsea 
library(data.table)
library(enrichplot) #plotting 
library(ggplot2) #plotting 
library(DESeq2) #for differential expression analysis 
library(EnhancedVolcano) #for volcano plot 
library(magrittr) #for %>% use 
library(dplyr) #data frame use 

adni.df <- readr::read_csv("/blue/kgraim/leslie.smith1/MLGenomicsProj/ADNI_Gene_Expression_Profile.csv")
adni.normal <- as.data.frame(adni.df)

#get meta data from adni.df in form: PROBESET   LOCUSLINK   SYMBOL
#metadata <- adni.df[9:nrow(adni.df),1:3] 
patient.meta <- adni.df[1:7,]

#get patient metadata in form for DESeq 
patient.meta.trans <- t(patient.meta)
df <- as.data.frame(patient.meta.trans)
what <- rownames_to_column(df) #this is horrible but I ran out of variable names
meta.tibble <- as_tibble(what)
colnames(meta.tibble) <- c("Phase","Visit","SubjectID","260/280","260/230","RIN","Affy_Plate","YearofCollection") #add colnames to metadata 
meta.tibble <- meta.tibble[4:nrow(meta.tibble),] #cutting out NAs at beginning of frame 
#creating factors for analysis 
meta.tibble$factor <- meta.tibble$Visit
meta.tibble$factor <- str_replace(meta.tibble$factor,"bl","baseline")
meta.tibble$factor <- str_replace(meta.tibble$factor,"v03","baseline")
meta.tibble$factor <- str_replace(meta.tibble$factor,"m03","progression")
meta.tibble$factor <- str_replace(meta.tibble$factor,"m36","progression")
meta.tibble$factor <- str_replace(meta.tibble$factor,"m48","progression")
meta.tibble$factor <- str_replace(meta.tibble$factor,"v11","progression")
meta.tibble$factor <- str_replace(meta.tibble$factor,"v04","progression")
meta.tibble$factor <- str_replace(meta.tibble$factor,"m60","progression")
meta.tibble$factor <- str_replace(meta.tibble$factor,"m12","progression")
meta.tibble$factor <- str_replace(meta.tibble$factor,"v02","progression")
meta.tibble$factor <- str_replace(meta.tibble$factor,"v05","progression")
meta.tibble$factor <- str_replace(meta.tibble$factor,"na","progression")

meta.tibble <- meta.tibble %>% 
  dplyr::mutate(
    factor = factor(factor, levels = c("baseline","progression"))
  )
#meta.tibble[is.na(meta.tibble)] <- "progression"

#get expression matrix into correction form 
expression <- adni.df[9:nrow(adni.df),4:ncol(adni.df)] #gene X sample
expression.num <- apply(expression,2,as.numeric)
expression.tibble <- as_tibble(expression.num) #49386

#get row sums to cut reads less than 10 
expression.tibble %>% rowwise() #group by rows to ensrue each gene has more than 10 counts for analysis 
expression.tibble <- expression.tibble %>% mutate(s = sum(c_across(ADNIGO...4:ADNI2...747))) #sum by column 
rowsums <- expression.tibble$s
expression.tibble <- expression.tibble[,1:744]

#get list of genes 
genes <- metadata$...3
#add genes to tibble to be grouped in order to get rid of duplicates
expression.tibble$Gene <- genes
test <- expression.tibble %>% group_by(Gene) 
test.sum <- test %>% summarize_if(is.numeric,mean) #20,093 - duplicate genes are now meaned 
test.sum$Gene[20093] <- "uknown"
nrow(test)
expression.filtered <- test.sum %>% column_to_rownames("Gene") 
expression.filtered <- round(expression.filtered)
expression.filtered <- expression.filtered[,-745]
expression.filtered[is.na(expression.filtered)] <- 0

#get mettadata ready for DESeq 
meta.tibble <- meta.tibble[-745,] #column 745 was NAs 

#create DESeq object 
ddset <- DESeqDataSetFromMatrix(
  countData = expression.filtered,
  
  colData = meta.tibble,
  
  design = ~factor
)
deseq_object1 <- DESeq(ddset)
deseq_results <- results(deseq_object1)
deseq_results <- lfcShrink(
  deseq_object1,
  coef = 2,
  res = deseq_results
)
write.table(deseq_df, file = "desseq_results.tsv", sep='\t', col.names=T, row.names=F, quote=F)#wrote to tabel to check something 


deseq_df <- deseq_results %>%
  as.data.frame() %>%
  tibble::rownames_to_column("Gene") %>% 
  dplyr::arrange(dplyr::desc(log2FoldChange))

#tried to look for P38 gene 
which(rownames(deseq_results) == "MAPK11")
plotCounts(ddset,gene = "MAPK11", intgroup = "factor")

most_expression <- readr::read_csv("/blue/kgraim/leslie.smith1/MLGenomicsProj/volcano_humanbase.csv") #gene from human base and volcano plot analysis
colnames(most_expression) <- c("Gene_symbol","Group","logFC","AveExpr","t","P.Value","adj.P.Val","B")

#gsea needs log fold change in decreasing order 
gene_list_FC <- most_expression$logFC
names(gene_list_FC) <- most_expression$Gene_symbol
gene_list <- sort(gene_list_FC, decreasing = TRUE)

#create gsea object 
gse <- gseGO(geneList = gene_list_FC,
             ont = "ALL",
             keyType = "SYMBOL",
             minGSSize = 3, 
             maxGSSize = 800, 
             pvalueCutoff = 0.05, 
             verbose = TRUE, 
             OrgDb = org.Hs.eg.db, 
             pAdjustMethod = "none")

require(DOSE)
#get dot plot 
dotplot(gse,showCategory=10, split=".sign") + facet_grid(.~.sign)
#get ridge pplot 
ridgeplot(gse) + labs(x = "enrichment distribution")


#gene ontology for genes 
GO <- read.gmt("/blue/kgraim/leslie.smith1/MLGenomicsProj/genesets.v7.5.1..v7.5.1.gmt")



library(msigdbr)
enrichMap(gse, vertex.label.cex=1.2, layout=igraph::layout.kamada.kawai) #FAILED
cnetplot(gse, categorySize="pvalue", foldChange=gene_list, showCategory = 3) #FAILED 

#trying to get data for cytoscape - FAILED 
gseGoTerm <- gse@result$ID
GoSets <- gse@geneSets
description <- gse@result$Description
core_enrichment <- gse@result$core_enrichment
gmtObject <- data.frame(gseGoTerm, description, core_enrichment)
write.table(expression.filtered, file = "/blue/kgraim/leslie.smith1/MLGenomicsProj/expression.tsv", sep='\t', col.names=T, row.names=F, quote=F)#wrote to tabel to check something 
write.table(results, file = "/blue/kgraim/leslie.smith1/MLGenomicsProj/gsea_results.tsv", sep='\t', col.names=T, row.names=F, quote=F)#wrote to tabel to check something 
results$Description <- gsub(" ", "", results$Description, fixed = TRUE)


