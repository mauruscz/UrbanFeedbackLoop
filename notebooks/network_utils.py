import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


def create_common_layout(networks_dict, net_type):
    """
    Creates a common layout for all nodes across all epochs.
    Uses min-max normalization to scale coordinates to [0,1] range.
    """
    if net_type == "mobility":
        # Collect all unique nodes and their coordinates
        all_nodes = {}
        for network in networks_dict.values():
            for node in network.nodes():
                if node not in all_nodes:
                    # Get lat/lng from node attributes
                    lat = network.nodes[node]['lat']
                    lng = network.nodes[node]['lng']
                    all_nodes[node] = (lat, lng)
        
        # Convert to numpy arrays for easier processing
        lats = np.array([coord[0] for coord in all_nodes.values()])
        lngs = np.array([coord[1] for coord in all_nodes.values()])
        
        # Calculate min and max for normalization
        lat_min, lat_max = lats.min(), lats.max()
        lng_min, lng_max = lngs.min(), lngs.max()
        
        # Create normalized layout dictionary
        layout = {}
        for node, (lat, lng) in all_nodes.items():
            # Normalize coordinates to [0,1] range
            x = (lng - lng_min) / (lng_max - lng_min)
            y = (lat - lat_min) / (lat_max - lat_min)
            layout[node] = np.array([x, y])
        
        return layout

    elif net_type == "co-location":
        # Find maximum node ID across all networks
        max_node = max(max(int(node) for node in network.nodes())
                      for network in networks_dict.values() 
                      if network.nodes())
        
        # Calculate grid dimensions based on max_node + 1 (to include node 0)
        grid_size = int(np.ceil(np.sqrt(max_node + 1)))
        
        # Create layout for all possible nodes
        layout = {}
        for node_id in range(max_node + 1):
            # Calculate grid position
            row = node_id // grid_size
            col = node_id % grid_size
            
            # Normalize to [0,1] range and add some randomness
            scaling_factor = 2.5  # Increase to spread nodes apart

            # Centering around 0.5 instead of 0
            x = (col / (grid_size - 1 or 1) - 0.5) * scaling_factor + 0.5
            y = (row / (grid_size - 1 or 1) - 0.5) * scaling_factor + 0.5

            # Add some randomness
            rand_offset = np.random.uniform(-0.1, 0.1, 2)
            x += rand_offset[0]
            y += rand_offset[1]

            # Ensure coordinates stay within [0,1] range
            x = max(0.1, min(0.9, x))
            y = max(0.1, min(0.9, y))
                        
            # Store with string node_id to match network nodes
            layout[str(node_id)] = np.array([x, y])
        
        return layout
    
    else:
        raise ValueError(f"Unknown network type: {net_type}")
    

def create_rich_club_layout(networks_dict, rich_threshold=100, rich_radius=0.25):
    """
    Creates a fixed circular layout but moves high-degree "rich club" nodes to the center.
    
    Parameters:
    -----------
    networks_dict : dict
        Dictionary of networks.
    rich_threshold : int
        Degree threshold to classify a node as part of the "rich club".
    
    Returns:
    --------
    dict
        Dictionary of node positions.
    """
    # Collect all unique nodes across networks
    all_nodes = sorted(set().union(*[net.nodes() for net in networks_dict.values()]))
    num_nodes = len(all_nodes)
    
    if num_nodes == 0:
        return {}

    # Compute degrees across all networks to identify rich club nodes
    combined_degrees = {node: 0 for node in all_nodes}
    for net in networks_dict.values():
        for node in net.nodes():
            combined_degrees[node] += net.degree(node)

    # Identify rich club nodes (high-degree nodes)
    rich_club_nodes = [node for node, deg in combined_degrees.items() if deg >= rich_threshold]
    peripheral_nodes = [node for node in all_nodes if node not in rich_club_nodes]

    # Circular positions for peripheral nodes
    layout = {}
    angle_step = 2 * np.pi / max(1, len(peripheral_nodes))  # Prevent division by zero

    for i, node in enumerate(peripheral_nodes):
        angle = i * angle_step
        x = 0.5 + 0.4 * np.cos(angle)  # Keep within [0,1] range
        y = 0.5 + 0.4 * np.sin(angle)
        layout[node] = np.array([x, y])

    # Center positions for rich club nodes
    rich_angle_step = 2 * np.pi / max(1, len(rich_club_nodes))  # Prevent division by zero
    for i, node in enumerate(rich_club_nodes):
        angle = i * rich_angle_step
        x = 0.5 + rich_radius * np.cos(angle)  # Move rich club nodes further out
        y = 0.5 + rich_radius * np.sin(angle)
        layout[node] = np.array([x, y])

    return layout


def create_circular_layout(networks_dict):
    """
    Creates a fixed circular layout for all nodes appearing in any network.

    Parameters:
    -----------
    networks_dict : dict
        Dictionary of networks.

    Returns:
    --------
    dict
        Dictionary of node positions in a circular layout.
    """
    # Collect all unique nodes across all networks
    all_nodes = sorted(set().union(*[net.nodes() for net in networks_dict.values()]))

    # Number of nodes
    num_nodes = len(all_nodes)
    if num_nodes == 0:
        return {}

    # Create circular positions
    layout = {}
    angle_step = 2 * np.pi / num_nodes  # Equal spacing in radians

    for i, node in enumerate(all_nodes):
        angle = i * angle_step  # Position in the circle
        x = 0.5 + 0.4 * np.cos(angle)  # Scale to fit within (0.1, 0.9)
        y = 0.5 + 0.4 * np.sin(angle)
        layout[node] = np.array([x, y])

    return layout


def plot_networks(networks_dict, num_epochs, suptitle="", node_color='lightblue',net_type = "mobility"):
    """
    Plots temporal networks in a single row layout without node labels.
    
    Parameters:
    -----------
    networks_dict : dict
        Dictionary of networkx graphs keyed by epoch
    num_epochs : int
        Number of epochs to plot
    """

    common_layout = create_common_layout(networks_dict, net_type)


    # Create figure with subplots in a single row
    fig, axs = plt.subplots(1, num_epochs, 
                           figsize=(5*num_epochs, 5),
                           squeeze=False,
                           facecolor='white')  # Set figure background to white
    fig.suptitle(suptitle, fontsize=35, y=1.15)
    
    # Flatten axes for easier iteration
    axs_flat = axs.flatten()
    
    # Sort epochs and take the first num_epochs
    epochs_to_plot = sorted(networks_dict.keys())[:num_epochs]
    
    for idx, epoch in enumerate(epochs_to_plot):
        ax = axs_flat[idx]
        network = networks_dict[epoch]
        
        # Set subplot background to white
        ax.set_facecolor('white')
        
        # Create layout dictionary for this network using common layout
        pos = {node: common_layout[node] for node in network.nodes()}


        # Draw edges
        nx.draw_networkx_edges(network, pos,
                             ax=ax,
                             edge_color='gray',
                             width=2,
                             alpha=0.8)

        # Draw nodes
        nodes = nx.draw_networkx_nodes(network, pos,
                                     ax=ax,
                                     node_color=node_color,
                                     node_size=300,
                                     alpha=1,
                                     node_shape='o')
        if nodes is not None:
            nodes.set_edgecolor('black')

        

        
        # Add network metrics in the title
        density = nx.density(network)
        components = nx.number_connected_components(network)
        title = f'Epoch {epoch}\n'
        title += f'N: {network.number_of_nodes()}, '
        title += f'E: {network.number_of_edges()}\n'
        title += f'D: {density:.3f}, '
        title += f'CC: {components}'
        ax.set_title(title, fontsize=20, pad=10)
        
        # Set axis properties
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_frame_on(False)
        
    # Make sure the figure background is white
    fig.patch.set_facecolor('white')
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    

    plt.close()
    return fig


def marco_plot_networks(networks_dict, num_epochs, suptitle="", node_color='lightblue', net_type="mobility",
                        epochs_to_plot=None):
    """
    Plots temporal networks in a single row layout without node labels.

    Parameters:
    -----------
    networks_dict : dict
        Dictionary of networkx graphs keyed by epoch
    num_epochs : int
        Number of epochs to plot
    """

    common_layout = create_common_layout(networks_dict, net_type)

    # Create figure with subplots in a single row
    fig, axs = plt.subplots(1, num_epochs,
                            figsize=(5 * num_epochs, 5),
                            squeeze=False,
                            facecolor='white')  # Set figure background to white
    fig.suptitle(suptitle, fontsize=24, y=1.15)

    # Flatten axes for easier iteration
    axs_flat = axs.flatten()

    # Sort epochs and take the first num_epochs
    if epochs_to_plot is None:
        epochs_to_plot = sorted(networks_dict.keys())[:num_epochs]

    for idx, epoch in enumerate(epochs_to_plot):
        ax = axs_flat[idx]
        network = networks_dict[epoch]

        # Set subplot background to white
        ax.set_facecolor('white')

        # Create layout dictionary for this network using common layout
        pos = {node: common_layout[node] for node in network.nodes()}

        # Draw edges
        nx.draw_networkx_edges(network, pos,
                               ax=ax,
                               edge_color='gray',
                               width=2,
                               alpha=0.8)

        # Draw nodes
        nodes = nx.draw_networkx_nodes(network, pos,
                                       ax=ax,
                                       node_color=node_color,
                                       node_size=300,
                                       alpha=1,
                                       node_shape='o')
        if nodes is not None:
            nodes.set_edgecolor('black')

        # Add network metrics in the title
        density = nx.density(network)
        # components = nx.number_connected_components(network)
        modularity = nx.community.modularity(network, nx.community.louvain_communities(network))
        average_clustering = nx.average_clustering(network)
        title = f'Epoch {epoch}\n'
        title += f'N: {network.number_of_nodes()}, '
        title += f'E: {network.number_of_edges()}, '
        title += f'D: {density:.3f}\n'
        title += f'Mod: {modularity:.3f},  '
        title += f'AvgClust: {average_clustering:.3f}'
        # title += f'CC: {components}'
        ax.set_title(title, fontsize=20, pad=10)

        # Set axis properties
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_frame_on(False)

    # Make sure the figure background is white
    fig.patch.set_facecolor('white')

    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)

    plt.close()
    return fig



