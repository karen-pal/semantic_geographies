
def process_multiple_text_files(text_files, language='english', output_dir=None, 
                            n_clusters=5, max_words_per_file=50, random_seed=42):
    """
    Process multiple text files and visualize them in the same semantic space.
    
    Args:
        text_files (list): List of paths to text files
        language (str): Language of the texts (default: 'english')
        output_dir (str): Directory to save outputs
        n_clusters (int): Number of semantic clusters
        max_words_per_file (int): Maximum number of most frequent words to include from each file
        random_seed (int): Random seed for reproducibility
        
    Returns:
        plotly.graph_objects.Figure: The created visualization
    """
    print(f"Processing {len(text_files)} text files in {language}")
    
    # Create output directory if needed
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Step 1: Extract and merge words from all files
    all_words_with_synsets = []
    file_sources = {}  # Track which file each word came from
    
    for file_idx, text_file in enumerate(text_files):
        file_name = os.path.basename(text_file)
        print(f"Processing file {file_idx+1}/{len(text_files)}: {file_name}")
        
        # Extract words from this file
        words = preprocess_text(text_file, language, max_words_per_file)
        print(f"  Extracted {len(words)} common words")
        
        # Filter words with WordNet entries
        file_words_with_synsets = filter_words_with_wordnet(words, language)
        print(f"  Found {len(file_words_with_synsets)} words with WordNet synsets")
        
        # Add file source information
        for i, (word, synset, freq) in enumerate(file_words_with_synsets):
            # Store which file this word came from
            if word not in file_sources:
                file_sources[word] = []
            file_sources[word].append(file_idx)
            
            # Add to overall list
            all_words_with_synsets.append((word, synset, freq, file_idx))
    
    # Remove duplicates while preserving file source info
    unique_words = {}
    for word, synset, freq, file_idx in all_words_with_synsets:
        if word not in unique_words:
            unique_words[word] = (synset, 0, [])
        
        # Update frequency (sum across files) and add file source
        current_synset, current_freq, sources = unique_words[word]
        if file_idx not in sources:
            sources.append(file_idx)
        unique_words[word] = (current_synset, current_freq + freq, sources)
    
    # Convert back to list format
    unique_words_with_synsets = [(word, data[0], data[1]) for word, data in unique_words.items()]
    
    if len(unique_words_with_synsets) < 3:
        print("Too few words with WordNet synsets found across all files. Try larger text files.")
        return None
    
    print(f"Total unique words across all files: {len(unique_words_with_synsets)}")
    
    # Step 2: Calculate similarity matrix
    sim_matrix = calculate_similarity_matrix(unique_words_with_synsets)
    
    # Step 3: Assign positions in 3D space
    positions, clusters = assign_positions(unique_words_with_synsets, sim_matrix, n_clusters, random_seed)
    
    # Step 4: Plot the 3D semantic space with file source information
    fig = plot_multiple_file_semantic_space(
        unique_words_with_synsets, 
        positions, 
        clusters, 
        file_sources, 
        len(text_files),
        text_files,
        output_dir
    )
    
    # Step 5: Create explanation file
    if output_dir:
        explanation_path = os.path.join(output_dir, "semantic_space_explanation.md")
        with open(explanation_path, 'w', encoding='utf-8') as file:
            file.write("""# Understanding the Multi-File Semantic Landscape Visualization

## What am I looking at?

This 3D visualization compares the semantic content of multiple text files by placing their words in a shared semantic space. Think of it as a "meaning map" where:

- **Words from different files** are shown in different colors
- **Words that appear in multiple files** are shown with markers divided into segments for each file
- **Words closer together** have more similar meanings
- **Word size** indicates how frequently the word appears across all texts
- **Lines** connect words with directly related meanings, with color intensity showing similarity strength
- **Height** represents how abstract (higher) or concrete (lower) a word is

## How to interpret the axes:

- **X-axis (Topic Similarity)**: Words with similar topics or domains appear closer along this axis
- **Y-axis (Conceptual Relatedness)**: Words that are conceptually related appear closer along this axis
- **Z-axis (Abstractness ↔ Concreteness)**: Words higher up are more abstract concepts, while words lower down are more concrete

## Exploring the visualization:

- **Rotate** the view by clicking and dragging
- **Zoom** with the scroll wheel
- **Hover** over words to see their full label and which files they appear in
- **Look for clusters** of words from the same file (same color)
- **Notice words that bridge** between different files

## Example insights:

- **Shared vocabulary**: Words that appear in multiple files represent shared themes or concepts
- **File-specific clusters**: Areas dominated by one color represent themes unique to that file
- **Semantic bridges**: Words that connect clusters from different files show conceptual links between texts
- **Abstractness patterns**: Different files may tend to use more abstract or concrete language
- **Thematic differences**: The distribution of words across the space reveals thematic focus differences

This visualization helps reveal similarities and differences in the semantic content of your text files.
""")
        print(f"Explanation saved to {explanation_path}")
    
    return fig

def plot_multiple_file_semantic_space(words_with_synsets, positions, clusters, file_sources, 
                                     num_files, file_paths, output_dir=None):
    """
    Plot words from multiple files in 3D semantic space with color coding by file source.
    
    Args:
        words_with_synsets (list): List of (word, synset_name, frequency) tuples
        positions (dict): Dictionary mapping words to 3D positions
        clusters (list): Cluster assignments for each word
        file_sources (dict): Dictionary mapping words to list of file indices they appear in
        num_files (int): Number of files being compared
        file_paths (list): List of file paths for labeling
        output_dir (str): Directory to save outputs
    """
    # Extract file names for legend
    file_names = [os.path.basename(path) for path in file_paths]
    
    # Create a custom color palette for files
    # Using a colorblind-friendly palette
    file_colors = [
        'rgb(230, 25, 75)',   # Red
        'rgb(60, 180, 75)',   # Green
        'rgb(255, 225, 25)',  # Yellow
        'rgb(0, 130, 200)',   # Blue
        'rgb(245, 130, 48)',  # Orange
        'rgb(145, 30, 180)',  # Purple
        'rgb(70, 240, 240)',  # Cyan
        'rgb(240, 50, 230)',  # Magenta
        'rgb(210, 245, 60)',  # Lime
        'rgb(250, 190, 212)'  # Pink
    ]
    
    # For more than 10 files, generate additional colors
    if num_files > 10:
        import colorsys
        for i in range(10, num_files):
            h = i / num_files
            r, g, b = colorsys.hsv_to_rgb(h, 0.8, 0.9)
            file_colors.append(f'rgb({int(r*255)}, {int(g*255)}, {int(b*255)})')
    
    # Create traces for words from each file
    node_traces = []
    
    words = [item[0] for item in words_with_synsets]
    frequencies = [item[2] for item in words_with_synsets]
    
    # Normalize frequencies for marker size
    max_freq = max(frequencies) if frequencies else 1
    
    # Create separate node traces for words unique to each file
    for file_idx in range(num_files):
        # Filter words unique to this file
        unique_words = [word for word in words if file_sources[word] == [file_idx]]
        
        if not unique_words:
            continue
            
        x_coords = []
        y_coords = []
        z_coords = []
        sizes = []
        texts = []
        
        for word in unique_words:
            x, y, z = positions[word]
            x_coords.append(x)
            y_coords.append(y)
            z_coords.append(z)
            
            # Find the original frequency
            idx = words.index(word)
            freq = frequencies[idx]
            sizes.append(10 + (freq / max_freq) * 20)
            
            # Text for hovering
            texts.append(f"{word} (from {file_names[file_idx]})")
        
        node_trace = go.Scatter3d(
            x=x_coords,
            y=y_coords,
            z=z_coords,
            mode='markers+text',
            marker=dict(
                size=sizes,
                color=file_colors[file_idx],
                opacity=0.8
            ),
            text=unique_words,
            hovertext=texts,
            hoverinfo='text',
            textposition='top center',
            name=f"Only in {file_names[file_idx]}"
        )
        
        node_traces.append(node_trace)
    
    # Create traces for words shared between files
    shared_words = [word for word in words if len(file_sources[word]) > 1]
    
    if shared_words:
        x_coords = []
        y_coords = []
        z_coords = []
        sizes = []
        texts = []
        symbols = []
        
        for word in shared_words:
            x, y, z = positions[word]
            x_coords.append(x)
            y_coords.append(y)
            z_coords.append(z)
            
            # Find the original frequency
            idx = words.index(word)
            freq = frequencies[idx]
            sizes.append(10 + (freq / max_freq) * 20)
            
            # Text for hovering showing all files this word appears in
            source_files = [file_names[i] for i in file_sources[word]]
            texts.append(f"{word} (in: {', '.join(source_files)})")
            
            # Use different symbols based on how many files share this word
            if len(file_sources[word]) == num_files:
                symbols.append('diamond')  # In all files
            else:
                symbols.append('circle')  # In some files
        
        shared_trace = go.Scatter3d(
            x=x_coords,
            y=y_coords,
            z=z_coords,
            mode='markers+text',
            marker=dict(
                size=sizes,
                color='rgb(100, 100, 100)',  # Gray for shared words
                symbol=symbols,
                opacity=0.8,
                line=dict(
                    color='rgb(0, 0, 0)',
                    width=1
                )
            ),
            text=shared_words,
            hovertext=texts,
            hoverinfo='text',
            textposition='top center',
            name=f"Shared between files"
        )
        
        node_traces.append(shared_trace)
    
    # Create edge traces with similarity-based coloring
    graph = create_semantic_graph(words_with_synsets)
    
    # Create separate edge traces for each edge to enable different colors
    edge_traces = []
    
    # Find min and max similarity for color scaling
    similarities = [data['weight'] for _, _, data in graph.edges(data=True)]
    if similarities:
        min_sim = min(similarities)
        max_sim = max(similarities)
        
        for word1, word2, data in graph.edges(data=True):
            x0, y0, z0 = positions[word1]
            x1, y1, z1 = positions[word2]
            
            # Normalize similarity for colorscale (0 to 1)
            if max_sim > min_sim:
                norm_sim = (data['weight'] - min_sim) / (max_sim - min_sim)
            else:
                norm_sim = 0.5
            
            # Create color from similarity value
            color = f'rgba(0, 0, 255, {0.2 + norm_sim * 0.8})'
            
            # Check if this edge connects words from different files
            sources1 = set(file_sources[word1])
            sources2 = set(file_sources[word2])
            
            # If edge connects different files, make it thicker and a different color
            if not sources1.isdisjoint(sources2) and sources1 != sources2:
                # This is a connection between files
                color = 'rgba(255, 0, 0, 0.7)'  # Red for cross-file connections
                width = 2 + norm_sim * 3
            else:
                width = 1 + norm_sim * 2
            
            edge_trace = go.Scatter3d(
                x=[x0, x1],
                y=[y0, y1],
                z=[z0, z1],
                mode='lines',
                line=dict(width=width, color=color),
                hoverinfo='text',
                hovertext=f"Similarity: {data['weight']:.3f}<br>{word1} — {word2}",
                showlegend=False
            )
            
            edge_traces.append(edge_trace)
    
    # Create figure with all traces
    fig = go.Figure(data=edge_traces + node_traces)
    
    # Update layout with more intuitive axis names
    fig.update_layout(
        title="Comparative Semantic Landscape",
        scene=dict(
            xaxis_title="Topic Similarity",
            yaxis_title="Conceptual Relatedness",
            zaxis_title="Abstractness ↔ Concreteness",
            xaxis=dict(showgrid=True, zeroline=False),
            yaxis=dict(showgrid=True, zeroline=False),
            zaxis=dict(showgrid=True, zeroline=False),
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    
    # Save outputs if directory is provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        # Save visualization
        html_path = os.path.join(output_dir, "comparative_semantic_space.html")
        fig.write_html(html_path)
        print(f"Visualization saved to {html_path}")
        
        # Save node data
        csv_path = os.path.join(output_dir, "comparative_nodes.csv")
        with open(csv_path, 'w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(['Word', 'Synset', 'Frequency', 'X', 'Y', 'Z', 'Files'])
            
            for i, (word, synset, freq) in enumerate(words_with_synsets):
                x, y, z = positions[word]
                files_str = ','.join([file_names[idx] for idx in file_sources[word]])
                writer.writerow([word, synset, freq, x, y, z, files_str])
                
        print(f"Node data saved to {csv_path}")
        
        # Save edge data
        edge_csv_path = os.path.join(output_dir, "comparative_edges.csv")
        with open(edge_csv_path, 'w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(['Word1', 'Word2', 'Similarity', 'X1', 'Y1', 'Z1', 'X2', 'Y2', 'Z2', 'Files1', 'Files2'])
            
            for word1, word2, data in graph.edges(data=True):
                x1, y1, z1 = positions[word1]
                x2, y2, z2 = positions[word2]
                files1_str = ','.join([file_names[idx] for idx in file_sources[word1]])
                files2_str = ','.join([file_names[idx] for idx in file_sources[word2]])
                writer.writerow([word1, word2, data['weight'], x1, y1, z1, x2, y2, z2, files1_str, files2_str])
                
        print(f"Edge data saved to {edge_csv_path}")
    
    # Show the plot
    fig.show()
    
    return figimport networkx as nx
