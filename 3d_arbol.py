import networkx as nx
import plotly.graph_objects as go
from nltk.corpus import wordnet as wn

def get_word_relations(word):
    """Get all possible synsets for a word and their hypernyms and hyponyms."""
    graph = nx.DiGraph()
    synsets = wn.synsets(word)

    for synset in synsets:
        graph.add_node(synset.name(), label=synset.name(), definition=synset.definition())
        
        # Add hypernyms
        for hypernym in synset.hypernyms():
            graph.add_node(hypernym.name(), label=hypernym.name(), definition=hypernym.definition())
            graph.add_edge(synset.name(), hypernym.name())

        # Add hyponyms
        for hyponym in synset.hyponyms():
            graph.add_node(hyponym.name(), label=hyponym.name(), definition=hyponym.definition())
            graph.add_edge(synset.name(), hyponym.name())

    return graph
def assign_positions_by_hierarchy(graph):
    positions = {}
    for node in graph.nodes:
        word= str(node.split('.')[0])
        depth = len(wn.synset(node).hypernym_paths()[0])
        positions[node] = (depth, graph.degree[node], -depth)  # X, Y, Z
    return positions


def create_3d_plot(graph):
    """Visualize a directed graph in 3D using Plotly."""
    #spring layout: optimized for non overlapping - uses spring physics
    # is actually meaning-agnostic
    #pos = nx.spring_layout(graph, dim=3, seed=42)  # 3D layout
    # Use this instead of spring_layout:
    pos = assign_positions_by_hierarchy(graph)
    
    # Extract node positions
    x_nodes = [pos[node][0] for node in graph.nodes]
    y_nodes = [pos[node][1] for node in graph.nodes]
    z_nodes = [pos[node][2] for node in graph.nodes]

    # Extract edge positions
    x_edges = []
    y_edges = []
    z_edges = []

    for edge in graph.edges:
        x_edges += [pos[edge[0]][0], pos[edge[1]][0], None]
        y_edges += [pos[edge[0]][1], pos[edge[1]][1], None]
        z_edges += [pos[edge[0]][2], pos[edge[1]][2], None]

    # Create the Plotly figure
    edge_trace = go.Scatter3d(
        x=x_edges, y=y_edges, z=z_edges,
        mode='lines',
        line=dict(color='gray', width=2),
        hoverinfo='none'
    )
    node_trace = go.Scatter3d(
        x=x_nodes, y=y_nodes, z=z_nodes,
        mode='markers+text',
        marker=dict(
            size=8,
            color=list(range(len(graph.nodes))),  # Color nodes uniquely
            colorscale='Viridis',
            colorbar=dict(title='Node Index'),
        ),
        text=[graph.nodes[node]['label'] for node in graph.nodes],
        textposition='top center',
        hovertext=[f"{graph.nodes[node]['label']}: {graph.nodes[node]['definition']}" for node in graph.nodes],
        hoverinfo='text'
    )
    #on hover
    #node_trace = go.Scatter3d(
    #    x=x_nodes, y=y_nodes, z=z_nodes,
    #    mode='markers',
    #    marker=dict(
    #        size=8,
    #        color=list(range(len(graph.nodes))),  # Color nodes uniquely
    #        colorscale='Viridis',
    #        colorbar=dict(title='Node Index'),
    #    ),
    #    text=[f"{graph.nodes[node]['label']}: {graph.nodes[node]['definition']}" for node in graph.nodes],
    #    hoverinfo='text'
    #)

    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        title="3D Semantic Graph",
        scene=dict(
            xaxis=dict(title='X-axis'),
            yaxis=dict(title='Y-axis'),
            zaxis=dict(title='Z-axis')
        ),
        showlegend=False
    )

    fig.show()

# Example usage
if __name__ == "__main__":
    word = "tree"  # Replace with your input word
    semantic_graph = get_word_relations(word)
    create_3d_plot(semantic_graph)

