import networkx as nx
import matplotlib.pyplot as plt
import math

def generate_fractal_graph(G, node, depth, angle, length, decay, spread):
    """Genera un grafo fractal recursivamente."""
    if depth == 0:
        return

    # Calcular las posiciones de los nuevos nodos hijos
    x, y = G.nodes[node]['pos']
    for i in range(-spread, spread + 1):  # Rama hacia la izquierda y derecha
        angle_offset = angle + i * math.pi / 6  # Ángulo de separación
        new_x = x + length * math.cos(angle_offset)
        new_y = y + length * math.sin(angle_offset)
        new_node = len(G.nodes)
        G.add_node(new_node, pos=(new_x, new_y))
        G.add_edge(node, new_node)

        # Llamada recursiva para expandir las ramas
        generate_fractal_graph(G, new_node, depth - 1, angle_offset, length * deca
y, decay, spread)

# Crear un grafo vacío y añadir el nodo raíz
G = nx.Graph()
G.add_node(0, pos=(0, 0))

# Generar el grafo fractal
generate_fractal_graph(G, 0, depth=4, angle=math.pi / 2, length=1, decay=0.7, spre
ad=2)

# Dibujar el grafo
pos = nx.get_node_attributes(G, 'pos')
plt.figure(figsize=(8, 8))
nx.draw(G, pos, node_size=10, edge_color="gray", with_labels=False)
plt.show()

