make arbol 3d?

make a dummy dataset to make embeddings and play with visualization


word input -> word nebula -> 3d graph

# En 3d arbol semantic

Z-Axis: calculate_depth() places nodes by their hierarchy level (lower-level concepts = higher Z-values).

XY-Plane: calculate_similarity_matrix() uses semantic similarity for clustering, arranging clusters in 2D space.

Colors: Clusters are assigned colors via SpectralClustering.

## Similarity metrics
En el código se puede modificar para que use distintas similarities.

**Wu-Palmer similarity (wup_similarity)**: This metric measures how closely two synsets are related by their position in the WordNet taxonomy. Essentially, it looks at:

- The distance to their lowest common ancestor in the hierarchy.

- The depth of the synsets in the taxonomy.

Wu-Palmer similarity relies heavily on hierarchical depth, which might not work well for edge cases. Switching to a different metric, such as **Leacock-Chodorow similarity (lch_similarity)**, which focuses on path length, might yield better results.
