if (!require("BiocManager", quietly = TRUE))
  install.packages("BiocManager")
BiocManager::install("splatter")

library("splatter")

clusters <- c(3,5,7)
dropout <- c(0.08, 0.17, 0.3)
de <- c(0.05, 0.1, 0.3)
cells <- c(1000, 2000, 3000)


for (exp in 1:20){
  params <- newSplatParams()
  params <- setParam(params, "nGenes", 30000)
  params <- setParam(params, "batchCells", sample(cells, 1))
  params <- setParam(params, "dropout.mid", sample(dropout, 1))
  params <- setParam(params, "de.prob", sample(de, 1))
  n_clusters= sample(clusters, 1)
  if (n_clusters == 3){
    clusters_probs = c(0.33,0.34,0.33)
  }else if (n_clusters == 5){
    clusters_probs= c(0.2,0.2,0.2,0.2,0.2)
  } else if(n_clusters == 7){
    clusters_probs = c(0.143,0.143,0.143,0.143,0.143,0.143,0.142)
  }
  params <- setParam(params, "group.prob", clusters_probs)
  sim <- splatSimulateGroups(params, verbose = FALSE)
  path = paste("~/Desktop/simulated/", as.character(exp), sep="")
  dir.create(path)
  write.csv(counts(sim), paste(path, '/counts.csv', sep=""))
  write.csv(colData(sim), paste(path, '/cells.csv', sep=""))
  write.csv(rowData(sim), paste(path, '/genes.csv', sep=""))
}