import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import networkx as nx
from mpl_toolkits.mplot3d import Axes3D
import random

class FractalGraphGenerator:
    def __init__(self):
        # Set up the figure and 3D axis
        self.fig = plt.figure(figsize=(12, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        # Initialize parameters
        self.iterations = 2
        self.branch_factor = 2
        self.angle_spread = 60
        self.length_ratio = 0.75
        self.dimension = 3  # 3D by default
        
        # Create graph
        self.G = nx.Graph()
        
        # Initialize nodes and positions
        self.pos_3d = {}
        
        # Generate initial fractal
        self.generate_fractal()
        
        # Set up sliders
        slider_color = 'lightgoldenrodyellow'
        slider_ax_iterations = plt.axes([0.2, 0.02, 0.65, 0.03], facecolor=slider_color)
        slider_ax_branch = plt.axes([0.2, 0.06, 0.65, 0.03], facecolor=slider_color)
        slider_ax_angle = plt.axes([0.2, 0.10, 0.65, 0.03], facecolor=slider_color)
        slider_ax_length = plt.axes([0.2, 0.14, 0.65, 0.03], facecolor=slider_color)
        
        self.slider_iterations = Slider(slider_ax_iterations, 'Iterations', 1, 5, valinit=self.iterations, valstep=1)
        self.slider_branch = Slider(slider_ax_branch, 'Branches', 2, 5, valinit=self.branch_factor, valstep=1)
        self.slider_angle = Slider(slider_ax_angle, 'Angle Spread', 10, 180, valinit=self.angle_spread)
        self.slider_length = Slider(slider_ax_length, 'Length Ratio', 0.2, 0.9, valinit=self.length_ratio)
        
        # Set up update function for sliders
        self.slider_iterations.on_changed(self.update)
        self.slider_branch.on_changed(self.update)
        self.slider_angle.on_changed(self.update)
        self.slider_length.on_changed(self.update)
        
        # Set up reset button
        reset_ax = plt.axes([0.8, 0.18, 0.1, 0.04])
        self.reset_button = Button(reset_ax, 'Reset', color=slider_color, hovercolor='0.975')
        self.reset_button.on_clicked(self.reset)
        
        # Add a dimension toggle button
        toggle_ax = plt.axes([0.1, 0.18, 0.15, 0.04])
        self.toggle_button = Button(toggle_ax, '2D/3D Toggle', color=slider_color, hovercolor='0.975')
        self.toggle_button.on_clicked(self.toggle_dimension)
        
        # Add key press events
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        
        # Instructions
        self.fig.text(0.02, 0.95, "Controls:\n"
                               "- Use sliders to adjust parameters\n"
                               "- Press 'r' to reset view\n"
                               "- Press '+' to increase iterations\n"
                               "- Press '-' to decrease iterations\n"
                               "- Press 'd' to toggle dimension", fontsize=10)
    
    def generate_fractal(self):
        """Generate the fractal graph based on current parameters"""
        # Clear existing graph
        self.G.clear()
        self.pos_3d.clear()
        
        # Add root node
        root = 0
        self.G.add_node(root)
        self.pos_3d[root] = np.array([0, 0, 0])
        
        # Generate the fractal structure
        self._grow_fractal(root, 1, np.array([0, 0, 1]), self.iterations)
        
        # Update the plot
        self.update_plot()
    
    def _grow_fractal(self, parent_node, depth, direction_vector, remaining_iterations):
        """Recursively grow the fractal structure"""
        if remaining_iterations <= 0:
            return
        
        parent_pos = self.pos_3d[parent_node]
        branch_length = self.length_ratio ** (depth - 1)
        
        # Generate branch directions
        directions = self._generate_directions(direction_vector, self.branch_factor, self.angle_spread)
        
        for i, new_direction in enumerate(directions):
            # Create new node
            new_node = len(self.G.nodes)
            self.G.add_node(new_node)
            
            # Calculate new position
            new_pos = parent_pos + branch_length * new_direction
            self.pos_3d[new_node] = new_pos
            
            # Add edge
            self.G.add_edge(parent_node, new_node)
            
            # Continue recursion
            self._grow_fractal(new_node, depth + 1, new_direction, remaining_iterations - 1)
    
    def _generate_directions(self, parent_direction, num_branches, angle_spread):
        """Generate new branch directions based on the parent direction"""
        # Normalize parent direction
        parent_direction = parent_direction / np.linalg.norm(parent_direction)
        
        # Convert angle from degrees to radians
        angle_rad = np.radians(angle_spread)
        
        # Generate random directions that form a cone around the parent direction
        directions = []
        
        for _ in range(num_branches):
            # Generate a random direction
            if self.dimension == 3:
                # For 3D, generate random points on a sphere
                phi = np.random.uniform(0, angle_rad)  # Angle from parent direction
                theta = np.random.uniform(0, 2*np.pi)  # Angle around parent direction
                
                # Calculate the direction relative to the z-axis
                x = np.sin(phi) * np.cos(theta)
                y = np.sin(phi) * np.sin(theta)
                z = np.cos(phi)
                new_dir = np.array([x, y, z])
                
                # Rotate to align with parent direction
                # This is a simplified rotation that works if parent_direction is not too close to [0,0,0]
                if np.allclose(parent_direction, np.array([0, 0, 1])):
                    rotated_dir = new_dir
                else:
                    # Find rotation axis and angle to align z-axis with parent_direction
                    z_axis = np.array([0, 0, 1])
                    rotation_axis = np.cross(z_axis, parent_direction)
                    rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
                    cos_angle = np.dot(z_axis, parent_direction)
                    angle = np.arccos(cos_angle)
                    
                    # Apply rotation using Rodrigues' rotation formula
                    rotated_dir = new_dir * np.cos(angle) + \
                                  np.cross(rotation_axis, new_dir) * np.sin(angle) + \
                                  rotation_axis * np.dot(rotation_axis, new_dir) * (1 - np.cos(angle))
            else:
                # For 2D, generate random angles in a circle
                random_angle = np.random.uniform(-angle_rad/2, angle_rad/2)
                cos_a, sin_a = np.cos(random_angle), np.sin(random_angle)
                
                # Rotate parent direction by random angle
                x, y = parent_direction[0], parent_direction[1]
                rotated_dir = np.array([x * cos_a - y * sin_a, 
                                       x * sin_a + y * cos_a, 
                                       0 if self.dimension == 2 else parent_direction[2]])
            
            directions.append(rotated_dir)
        
        return directions
    
    def update_plot(self):
        """Update the network plot"""
        self.ax.clear()
        
        # Extract node positions
        node_xyz = np.array([self.pos_3d[v] for v in sorted(self.G.nodes)])
        edge_xyz = np.array([(self.pos_3d[u], self.pos_3d[v]) for u, v in self.G.edges()])
        
        # Plot the nodes
        self.ax.scatter(node_xyz[:, 0], node_xyz[:, 1], node_xyz[:, 2], s=30, c='b', alpha=0.7)
        
        # Plot the edges
        for edge in edge_xyz:
            self.ax.plot([edge[0][0], edge[1][0]],
                         [edge[0][1], edge[1][1]],
                         [edge[0][2], edge[1][2]], c='k', alpha=0.5)
        
        # Set axis properties
        if self.dimension == 3:
            self.ax.set_xlabel('X')
            self.ax.set_ylabel('Y')
            self.ax.set_zlabel('Z')
            self.ax.set_title(f'3D Fractal Graph - {len(self.G.nodes)} nodes, {len(self.G.edges)} edges')
            self.ax.set_proj_type('persp')
        else:
            self.ax.set_xlabel('X')
            self.ax.set_ylabel('Y')
            self.ax.set_zlim(-0.1, 0.1)  # Flatten to 2D
            self.ax.set_title(f'2D Fractal Graph - {len(self.G.nodes)} nodes, {len(self.G.edges)} edges')
            self.ax.view_init(elev=90, azim=-90)  # Top-down view
        
        # Auto-scale to fit the graph
        max_range = np.array([node_xyz[:, 0].max() - node_xyz[:, 0].min(),
                              node_xyz[:, 1].max() - node_xyz[:, 1].min(),
                              node_xyz[:, 2].max() - node_xyz[:, 2].min()]).max() / 2.0
        
        mid_x = (node_xyz[:, 0].max() + node_xyz[:, 0].min()) * 0.5
        mid_y = (node_xyz[:, 1].max() + node_xyz[:, 1].min()) * 0.5
        mid_z = (node_xyz[:, 2].max() + node_xyz[:, 2].min()) * 0.5
        
        self.ax.set_xlim(mid_x - max_range*1.1, mid_x + max_range*1.1)
        self.ax.set_ylim(mid_y - max_range*1.1, mid_y + max_range*1.1)
        self.ax.set_zlim(mid_z - max_range*1.1, mid_z + max_range*1.1)
        
        plt.draw()
    
    def update(self, val):
        """Update the fractal based on slider values"""
        self.iterations = int(self.slider_iterations.val)
        self.branch_factor = int(self.slider_branch.val)
        self.angle_spread = self.slider_angle.val
        self.length_ratio = self.slider_length.val
        
        self.generate_fractal()
    
    def reset(self, event):
        """Reset sliders to initial values"""
        self.slider_iterations.reset()
        self.slider_branch.reset()
        self.slider_angle.reset()
        self.slider_length.reset()
        
        # This will trigger update via the slider callbacks
    
    def toggle_dimension(self, event):
        """Toggle between 2D and 3D visualization"""
        self.dimension = 3 if self.dimension == 2 else 2
        self.generate_fractal()
    
    def on_key(self, event):
        """Handle key press events"""
        if event.key == 'r':
            self.reset(event)
        elif event.key == '+':
            new_val = min(self.iterations + 1, 5)
            self.slider_iterations.set_val(new_val)
        elif event.key == '-':
            new_val = max(self.iterations - 1, 1)
            self.slider_iterations.set_val(new_val)
        elif event.key == 'd':
            self.toggle_dimension(event)
    
    def show(self):
        """Display the plot"""
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.25)  # Make room for sliders
        plt.show()

# Create and show the fractal graph generator
if __name__ == "__main__":
    generator = FractalGraphGenerator()
    generator.show()
