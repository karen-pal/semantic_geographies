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


# Text semantic
## Installation
import nltk
nltk.download('punkt_tab')

## Usage
python text_semantic_space.py your_text_file.txt --language english --output_dir results --clusters 5


- X-axis: "Topic Similarity" - Words with similar topics or domains appear closer along this axis
- Y-axis: "Conceptual Relatedness" - Words that are conceptually related appear closer along this axis
- Z-axis: "Abstractness ↔ Concreteness" - Words higher up are more abstract concepts, while words lower down are more concrete

- Dynamically determines the similarity threshold based on the data
- Takes the top 25% of most similar word pairs (or a minimum number)
- Ensures that each node has at least one connection if possible
- Adds "best match" connections for words that would otherwise be isolated

## Arguments
### max words
The default is 100. when choosing max_words be careful:
- 100 is large enough to capture the main semantic content
- 100 is Small enough to keep visualizations clean and interpretable
- Processing a large number of words can be computationally intensive
- WordNet similarity calculations scale quadratically with word count

A few questions and observations of your code:
1) why in the resulting graph some nodes are connected while others arent? why are there edge-less nodes? are they roots? what are them?
2) Could we find better names for the axis? Something that can be interpretted easily by a non technical human?
3) shouldnt we use a fixed seed so the clustering is reproducible? Right now it's always outputting different results given the same inputs.
4) Why are we only interested in the most common 100 words of the text?
5) how could we extend this so it supports various input text files (in the same language) and to draw their semantic graphs on the same space, while retaining semantic relevance and showcasing different source so we can use our new tool to compare semantic content of different text files?
