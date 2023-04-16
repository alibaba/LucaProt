######################## clusering merge to superclade ############################


library(igraph)

setwd("E:/Data/3mata-virome/find_virus/contig/diff_method/cluster/clstr_expansion/300aa/merge/")


######## use median evalue 

edges <- read.table("m3_edges.txt", sep="\t",header=T)
vertices <- read.table("m3_vertices.txt", sep="\t",header=T)

head(edges)
head(vertices)

graph <- graph_from_data_frame(edges, directed = NULL, vertices=vertices)

ceb <- cluster_edge_betweenness(graph)
head(ceb)
dendPlot(ceb, mode = "hclust", cex=0.2)
plot(ceb, graph, 
     vertex.size=4, 
     vertex.frame.color="NA",
     edge.width=1,
     vertex.label=NA
#     vertex.label.cex=0.3,
)

class(ceb)
length(ceb)

merge_result <- membership(ceb)
merge_result <- as.matrix(merge_result)
merge_result <- as.data.frame(merge_result)
write.table(merge_result, file="merge_by_m3_result.txt", quote=F, sep="\t", row.names=T, col.names=F)

modularity(ceb)
head(merge_result)
