#plots.py

import matplotlib.pyplot as plt
import numpy as np
from .constants import DEFAULT_SAMPLE_SIZE
import networkx as nx



def plot_distribution(stochastic_var, num_samples=DEFAULT_SAMPLE_SIZE, bins=30, density=True, title=None):
    """
    Plots the distribution of a StochasticVariable and includes a vertical line for the mean.

    Parameters:
        stochastic_var (StochasticVariable): The stochastic variable to be plotted.
        num_samples (int): Number of samples to draw for the plot (default: 1000).
        bins (int): Number of bins in the histogram (default: 30).
        density (bool): Whether to normalize the histogram to show density (default: True).
        title (str): Title for the plot (optional).
    """
    # Generate samples
    samples = stochastic_var.sample(size=num_samples)
    mean_value = np.mean(samples)  # Calculate the mean of the samples

    # Plot histogram
    plt.figure(figsize=(8, 6))
    plt.hist(samples, bins=bins, density=density, alpha=0.6, color='blue', edgecolor='black', label='Histogram')

    # Add a density line if it's a continuous variable
    if density and stochastic_var.distribution_type in ['continuous', 'mixed']:
        kde = gaussian_kde(samples)
        x_range = np.linspace(min(samples), max(samples), 1000)
        plt.plot(x_range, kde(x_range), color='red', label='Density')

    # Add a vertical line for the mean
    plt.axvline(mean_value, color='green', linestyle='--', linewidth=2, label=f'Mean = {mean_value:.2f}')

    # Set labels and title
    plt.xlabel('Value')
    plt.ylabel('Density' if density else 'Frequency')
    if title is None:
        title = f'Distribution of {stochastic_var.name}'
    plt.title(title)
    plt.legend()

    # Show the plot
    plt.grid(True)
    plt.show()



def plot_dependency_graph(variables, title="Dependency Graph"):
    """
    Builds, visualizes, and highlights circular dependencies in a stochastic variable dependency graph.
    Ensures that all dependencies of the input variables are included in the graph.

    Parameters:
        - variables (list of StochasticVariable): The stochastic variables to include in the graph.
        - title (str): Title for the graph visualization (default: "Dependency Graph").

    Returns:
        - graph (networkx.DiGraph): The constructed dependency graph.
    """
    # Build the dependency graph
    graph = nx.DiGraph()

    def add_to_graph(variable, visited):
        if variable in visited:
            return
        visited.add(variable)
        graph.add_node(variable.name)

        for dep in variable.get_all_dependencies():
            graph.add_edge(dep.name, variable.name)
            add_to_graph(dep, visited)

    visited = set()
    for var in variables:
        add_to_graph(var, visited)

    # Detect circular dependencies
    circular_dependencies = []
    try:
        cycles = list(nx.find_cycle(graph, orientation="original"))
        circular_dependencies = [edge[0] for edge in cycles] + [cycles[-1][1]]  # Nodes involved in cycles
    except nx.NetworkXNoCycle:
        pass

    # Visualize the graph
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(graph)

    # Color nodes
    node_colors = [
        "red" if node in circular_dependencies else "lightblue" for node in graph.nodes()
    ]

    nx.draw(
        graph,
        pos,
        with_labels=True,
        node_color=node_colors,
        node_size=3000,
        font_size=12,
        font_weight="bold",
        edge_color="gray",
    )

    # Highlight circular edges
    if circular_dependencies:
        circular_edges = [(u, v) for u, v in graph.edges if u in circular_dependencies and v in circular_dependencies]
        nx.draw_networkx_edges(
            graph,
            pos,
            edgelist=circular_edges,
            edge_color="red",
            width=2.0,
        )

    plt.title(title)
    plt.show()

    return graph
