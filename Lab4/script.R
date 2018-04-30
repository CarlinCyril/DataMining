set.seed(7)
NAm2 = read.table("NAm2.txt", header = TRUE)

# Question 1
sub <- subset(NAm2, NAm2$Pop %in% c("Chipewyan", "Pima", "Huilliche"))
sub_genetic <- sub[,-(1:8)]

# Question 2
adj_mat <- data.matrix(sub_genetic) %*% t(data.matrix(sub_genetic))
diag(adj_mat) = 0

# Question 3
threshold <- quantile(adj_mat, probs = 0.7)[["70%"]]
to_delete_index <- c()
for(i in 1:nrow(adj_mat)) {
  nb <- nrow(adj_mat)
  for (j in 1:nrow(adj_mat)) {
    if (adj_mat[i, j] < threshold) {
      adj_mat[i, j] = 0
      nb <- nb - 1
    }
  }
  if (nb < 2) {
    to_delete_index <- cbind(to_delete_index, i)
  }
}
preprocessed_adj_mat <- adj_mat[-to_delete_index,-to_delete_index]

# Question 4
## Download and install the package
## install.packages("igraph")
library(igraph)

# Question 5
graph1 <- graph.adjacency(preprocessed_adj_mat, weighted=TRUE, mode = "undirected") 

# Question 6
l <- layout.fruchterman.reingold(graph1)
plot(graph1, vertex.size=5, vertex.label=NA, layout=l)

# Question 7
gorder(graph1) # nb of vertices
gsize(graph1) # number of edges
diameter(graph1, directed = FALSE, weights = NA) # length of the longest geodesic
# average path length in a graph, 
# by calculating the shortest paths between all pairs of vertices (both ways for directed graphs)
mean_distance(graph1, directed = FALSE)

# Question 8
