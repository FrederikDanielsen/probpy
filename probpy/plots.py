#plots.py

# IMPORTS
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from scipy.stats import gaussian_kde
from .constants import DEFAULT_PLOTTING_SAMPLE_SIZE
from .core import StochasticVariable, StochasticVector


__all__ = ["plot_distribution", "plot_dependency_graph"]

def plot_distribution(stochastic_var, num_samples=DEFAULT_PLOTTING_SAMPLE_SIZE, bins=30, density=True, title=None):
    """
    Plots the distribution of a StochasticVariable and includes a vertical line for the mean.

    Parameters:
        stochastic_var (StochasticVariable): The stochastic variable to be plotted.
        num_samples (int): Number of samples to draw for the plot (default: DEFAULT_PLOTTING_SAMPLE_SIZE).
        bins (int): Number of bins in the histogram (default: 30).
        density (bool): Whether to normalize the histogram to show density (default: True).
        title (str): Title for the plot (optional).
    """
    # Generate samples
    samples = stochastic_var.sample(size=num_samples)
    mean_value = np.mean(samples)  # Calculate the mean of the samples

    plt.figure(figsize=(8, 6))

    if stochastic_var.distribution_type == "discrete":
        # Plot histogram for discrete values
        unique_values, counts = np.unique(samples, return_counts=True)
        plt.bar(
            unique_values, counts, width=0.8, color='blue', edgecolor='black', alpha=0.7, label='Histogram'
        )
        plt.ylabel("Frequency")
    else:
        # Plot histogram for continuous or mixed distributions
        plt.hist(samples, bins=bins, density=density, alpha=0.6, color='blue', edgecolor='black', label='Histogram')

        # Add a density line if it's a continuous variable
        if density and stochastic_var.distribution_type in ['continuous', 'mixed']:
            try:
                kde = gaussian_kde(samples)
                x_range = np.linspace(np.min(samples), np.max(samples), num_samples)
                plt.plot(x_range, kde(x_range), color='red', label='Density')
            except Exception as e:
                print(f"Error computing KDE: {e}")
        plt.ylabel("Density" if density else "Frequency")

    # Add a vertical line for the mean
    plt.axvline(mean_value, color='green', linestyle='--', linewidth=2, label=f'Mean = {mean_value:.2f}')

    # Set labels and title
    plt.xlabel("Value")
    if title is None:
        title = f"Distribution of {stochastic_var.name}"
    plt.title(title)
    plt.legend()

    # Show the plot
    plt.grid(True)
    plt.show()


def plot_dependency_graph(*vars, title="Dependency Graph", depth=1):
    """
    Builds, visualizes, and highlights circular dependencies in a stochastic variable dependency graph.
    Ensures that all dependencies of the input variables are included in the graph.

    Parameters:
        - variables (list of StochasticVariable or StochasticVector): The stochastic variables or vectors to include in the graph.
        - title (str): Title for the graph visualization (default: "Dependency Graph").

    Returns:
        - graph (networkx.DiGraph): The constructed dependency graph.
    """
    
    variables = vars
    
    # Build the dependency graph
    graph = nx.DiGraph()

    def add_to_graph(variable, visited):
        if variable in visited:
            return
        visited.add(variable)
        graph.add_node(variable.name)

        if isinstance(variable, StochasticVector):
            # For StochasticVector, add edges from its components and their dependencies
            for var in variable.variables:
                deps = var.get_all_dependencies()
                for dep in deps:
                    if not dep.constant and len(dep.dependencies) < depth:
                        graph.add_edge(dep.name, variable.name)
                        add_to_graph(dep, visited)
        elif isinstance(variable, StochasticVariable):
            # Add edges for dependencies
            deps = variable.get_all_dependencies()
            for dep in deps:
                if not dep.constant and len(dep.dependencies) < depth:
                    graph.add_edge(dep.name, variable.name)
                    add_to_graph(dep, visited)
        else:
            raise ValueError(f"Unsupported variable type: {type(variable)}")

    visited = set()
    for var in variables:
        add_to_graph(var, visited)

    # Detect circular dependencies
    circular_dependencies = set()
    try:
        cycles = list(nx.simple_cycles(graph))
        if cycles:
            # Flatten the list of cycles
            circular_dependencies = set(node for cycle in cycles for node in cycle)
    except nx.NetworkXNoCycle:
        pass

    # Visualize the graph
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(graph)

    # Color nodes
    node_colors = []
    for node in graph.nodes():
        if node in circular_dependencies:
            color = "red"
        else:
            # Differentiate between variables and vectors
            # Assuming names of vectors are unique and set in the StochasticVector class
            variable = next((var for var in visited if var.name == node), None)
            if isinstance(variable, StochasticVector):
                color = "lightgreen"
            else:
                color = "lightblue"
        node_colors.append(color)

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
        circular_edges = [
            (u, v) for u, v in graph.edges if u in circular_dependencies and v in circular_dependencies
        ]
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
