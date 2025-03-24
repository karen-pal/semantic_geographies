import networkx as nx
import matplotlib.pyplot as plt
import math

def generate_circular_fractal(G, center_node, center_pos, radius, depth, num_nodes
, decay):
    """Genera un grafo circular con iteraciones fractales."""
    if depth == 0:
        return

    # Generar nodos en un círculo
    theta = 2 * math.pi / num_nodes
    for i in range(num_nodes):
        angle = i * theta
        new_x = center_pos[0] + radius * math.cos(angle)
        new_y = center_pos[1] + radius * math.sin(angle)
        new_node = len(G.nodes)
        G.add_node(new_node, pos=(new_x, new_y))
        G.add_edge(center_node, new_node)

        # Llamada recursiva para sub-círculos
        generate_circular_fractal(G, new_node, (new_x, new_y), radius * decay, dep
th - 1, num_nodes, decay)

# Crear un grafo vacío
G = nx.Graph()

# Nodo raíz central
G.add_node(0, pos=(0, 0))

# Generar el grafo fractal circular
generate_circular_fractal(G, center_node=0, center_pos=(0, 0), radius=1, depth=3,
num_nodes=6, decay=0.5)

# Obtener posiciones de los nodos
pos = nx.get_node_attributes(G, 'pos')

# Dibujar el grafo
plt.figure(figsize=(8, 8))
nx.draw(G, pos, node_size=10, edge_color="gray", with_labels=False)
plt.show()

