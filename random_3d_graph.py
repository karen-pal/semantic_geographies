import plotly.graph_objects as go

# Crear un grafo pequeño con posiciones aleatorias
G = nx.random_geometric_graph(10, 0.5, dim=3)
pos = nx.get_node_attributes(G, 'pos')

# Extraer coordenadas de nodos y bordes
x = [pos[i][0] for i in G.nodes]
y = [pos[i][1] for i in G.nodes]
z = [pos[i][2] for i in G.nodes]
edges = [(pos[u], pos[v]) for u, v in G.edges]

# Crear visualización en Plotly
fig = go.Figure()

# Agregar nodos
fig.add_trace(go.Scatter3d(
    x=x, y=y, z=z,
    mode='markers',
    marker=dict(size=6, color='skyblue'),
    name='Nodos'
))

# Agregar conexiones
for edge in edges:
    fig.add_trace(go.Scatter3d(
        x=[edge[0][0], edge[1][0], None],
        y=[edge[0][1], edge[1][1], None],
        z=[edge[0][2], edge[1][2], None],
        mode='lines',
        line=dict(color='red', width=2),
        showlegend=False
    ))

fig.show()

