import networkx as nx
import plotly.graph_objects as go
from nltk.corpus import wordnet as wn
from sklearn.cluster import SpectralClustering
import numpy as np
import csv

# Function to calculate hypernym depth for Z-axis
def calculate_depth(node):
    synset = wn.synset(node)
    return len(synset.hypernym_paths()[0])

# Function to calculate semantic similarity matrix
def calculate_similarity_matrix(nodes):
    n = len(nodes)
    sim_matrix = np.zeros((n, n))
    for i, node1 in enumerate(nodes):
        for j, node2 in enumerate(nodes):
            if i >= j:  # Avoid redundant calculations
                continue
            try:
                syn1 = wn.synset(node1.split('.')[0])
                syn2 = wn.synset(node2.split('.')[0])
                #using  Wu-Palmer similarity
                #sim = syn1.wup_similarity(syn2)
                #using Leacock-Chodorow similarity
                sim = syn1.lch_similarity(syn2) if syn1._pos == syn2._pos else None

                sim_matrix[i, j] = sim if sim is not None else 0
                sim_matrix[j, i] = sim_matrix[i, j]
            except:
                sim_matrix[i, j] = 0
                sim_matrix[j, i] = 0
    return sim_matrix

# Generate a graph of semantic relationships
def generate_graph(word):
    graph = nx.DiGraph()
    synsets = wn.synsets(word)
    for syn in synsets:
        graph.add_node(syn.name())
        for hypernym in syn.hypernyms():
            graph.add_edge(syn.name(), hypernym.name())
        for hyponym in syn.hyponyms():
            graph.add_edge(hyponym.name(), syn.name())
    return graph

# Assign 3D positions based on hierarchy and similarity
def assign_positions(graph):
    nodes = list(graph.nodes)
    depth_positions = np.array([calculate_depth(node) for node in nodes])  # Z-axis (depth)
    
    # Semantic similarity for clustering
    sim_matrix = calculate_similarity_matrix(nodes)
    clustering = SpectralClustering(n_clusters=5, affinity='precomputed', random_state=42).fit(sim_matrix)
    clusters = clustering.labels_

    # Use clustering to spread nodes in XY-plane
    cluster_centers = np.random.uniform(-10, 10, (5, 2))  # Random cluster centers in 2D
    xy_positions = np.zeros((len(nodes), 2))
    for i, cluster in enumerate(clusters):
        xy_positions[i] = cluster_centers[cluster] + np.random.uniform(-1, 1, 2)  # Add some spread

    positions = {}
    for i, node in enumerate(nodes):
        x, y = xy_positions[i]
        z = depth_positions[i]
        positions[node] = (x, y, z)
    return positions, clusters

# Plot the graph in 3D
def plot_3d_graph(graph, positions, clusters, filepath=None, save_nodes=None):
    edge_x = []
    edge_y = []
    edge_z = []
    for edge in graph.edges():
        x0, y0, z0 = positions[edge[0]]
        x1, y1, z1 = positions[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        edge_z.extend([z0, z1, None])

    # Create edge traces
    edge_trace = go.Scatter3d(
        x=edge_x, y=edge_y, z=edge_z,
        line=dict(width=1, color='rgba(100, 100, 100, 0.6)'),
        hoverinfo='none',
        mode='lines'
    )
    # Create node traces
    node_x = []
    node_y = []
    node_z = []
    node_text = []  # Text to display by default
    node_hovertext = []  # Text to display on hover
    node_color = []

    for i, node in enumerate(graph.nodes()):
        x, y, z = positions[node]
        node_x.append(x)
        node_y.append(y)
        node_z.append(z)
        label = node  # Synset name
        try:
            definition = wn.synset(node).definition()  # Synset definition
        except:
            definition = "No definition available"
        
        node_text.append(label)  # Default text: label only
        node_hovertext.append(f"{label}: {definition}")  # Hover text: label + definition
        node_color.append(clusters[i])

    node_trace = go.Scatter3d(
        x=node_x, y=node_y, z=node_z,
        mode='markers+text',
        marker=dict(
            size=8,
            color=node_color,
            colorscale='Viridis',
            showscale=True
        ),
        text=node_text,  # Display only the label by default
        hovertext=node_hovertext,  # Display label + definition on hover
        hoverinfo='text',  # Ensure hover text is displayed
        textposition='top center'
    )

    # Combine traces
    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        scene=dict(
            xaxis=dict(showgrid=False, zeroline=False),
            yaxis=dict(showgrid=False, zeroline=False),
            zaxis=dict(showgrid=False, zeroline=False),
        ),
        margin=dict(l=0, r=0, t=0, b=0)
    )
    fig.show()
    if filepath is not None:
        print(f"saving plotly viz in {filepath}")
        fig.write_html(filepath)
    if save_nodes is not None:
        # Create a list of node data
        node_data = []
        for i, node in enumerate(graph.nodes()):
            x, y, z = positions[node]
            label = node  # Synset name
            hovertext = node_hovertext[i]  # Hover text (label + definition)
            node_data.append([label, x, y, z, hovertext])

        # Define the CSV file name
        csv_filename = save_nodes

        # Write data to CSV
        with open(csv_filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            # Write the header row
            writer.writerow(['Label', 'X', 'Y', 'Z', 'HoverText'])
            # Write the node data
            writer.writerows(node_data)

        print(f"Node information saved to {csv_filename}")
        # Export edge information
        edge_data = []
        for edge in graph.edges():
            x0, y0, z0 = positions[edge[0]]
            x1, y1, z1 = positions[edge[1]]
            edge_data.append([edge[0], edge[1], x0, y0, z0, x1, y1, z1])

        edge_filename = save_nodes.split(".csv")[0]+"_edge_information.csv"
        with open(edge_filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Node1', 'Node2', 'X0', 'Y0', 'Z0', 'X1', 'Y1', 'Z1'])
            writer.writerows(edge_data)

        print(f"Edge information saved to {edge_filename}")



# Main
word = "nurture"  # Change this to any word
graph = generate_graph(word)
positions, clusters = assign_positions(graph)
filepath = word+"_graph_visualization.html"
save_nodes = word+"_nodes_info.csv"
plot_3d_graph(graph, positions, clusters, filepath, save_nodes)
# Save the figure as an HTML file

