import os
import sys
import asyncio


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    sys.path.append(project_root)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List

from src.graph import NodeList, EdgeList
from experiment.metrics import graphMetricsLLM, graphMetricsLLM_sync

def plotResults(
    result_path_list: List[str],
    output_dir: str
):
    """
    Plot the results of the graph extraction/generation.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize data structures to hold all results
    all_data = []
    
    # Load data from all result paths
    for result_path in result_path_list:
        result_df = pd.read_csv(result_path)
        
        # Extract method and model from file name
        filename = os.path.basename(result_path)
        parts = filename.split('_')
        method = parts[0].split('-')[0]
        model = parts[1]
        
        # Add method and model columns
        result_df['method'] = method
        result_df['model'] = model
        
        all_data.append(result_df)
    
    # Combine all dataframes
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Create consistent colors for models
    model_colors = {'qwen2.5:7b': '#A1C9F4', 'qwen2.5:72b': '#0A75AD', 'gpt-4o': '#FF9966'}  # Light blue, deep blue, and orange for GPT-4o
    scatter_color = '#555555'  # Gray for scatter points
    method_markers = {'joint': 'o', 'sequential': 's'}
    method_colors = {'joint': '#E5F5E0', 'sequential': '#F5F0E0'}  # Light green and light orange backgrounds
    
    # Define level colors and ensure correct order: background, node, graph
    level_colors = {'background': '#f0f0f0', 'node': '#ffffff', 'graph': '#f0f0f0'}
    ordered_levels = ['background', 'node', 'graph']
    
    # Filter levels to only those present in the data
    available_levels = sorted(combined_df['level'].unique())
    levels = [level for level in ordered_levels if level in available_levels]
    
    # Define the model order we want
    model_order = ['qwen2.5:7b', 'qwen2.5:72b', 'gpt-4o']
    
    # Plot n-ged separately as it's a single metric
    fig_ged, ax_ged = plt.subplots(1, 1, figsize=(7, 9))  # Increased from (10, 8)
    
    plot_box_metric(
        combined_df, 
        ax_ged, 
        levels, 
        'n_ged', 
        model_colors, 
        method_markers, 
        method_colors=method_colors,
        lower_is_better=True,
        scatter_color=scatter_color,
        use_jitter=True,
        level_colors=level_colors,
        group_by_level=True,
        show_legend=True,
        y_label="Normalized Graph Edit Distance"
    )
    
    # Save the n-ged figure
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'graph-extract_n-ged.pdf'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create node metrics figure
    # Calculate how many subplots we need (3 metrics)
    # Make each subplot square
    fig_node = plt.figure(figsize=(20, 9))  # Increased from (18, 8)
    
    # Node metrics plots
    ax1 = fig_node.add_subplot(1, 3, 1)
    ax2 = fig_node.add_subplot(1, 3, 2)
    ax3 = fig_node.add_subplot(1, 3, 3)
    
    # Plot node metrics with level grouping
    plot_box_metric(
        combined_df, 
        ax1, 
        levels, 
        'node_precision', 
        model_colors, 
        method_markers, 
        method_colors=method_colors,
        level_colors=level_colors,
        group_by_level=True,
        show_legend=True,
        scatter_color=scatter_color,
        use_jitter=True,
        y_label="Precision"
    )
    
    plot_box_metric(
        combined_df, 
        ax2, 
        levels, 
        'node_recall', 
        model_colors, 
        method_markers, 
        method_colors=method_colors,
        level_colors=level_colors,
        group_by_level=True,
        scatter_color=scatter_color,
        use_jitter=True,
        y_label="Recall"
    )
    
    plot_box_metric(
        combined_df, 
        ax3, 
        levels, 
        'node_f1', 
        model_colors, 
        method_markers, 
        method_colors=method_colors,
        level_colors=level_colors,
        group_by_level=True,
        scatter_color=scatter_color,
        use_jitter=True,
        y_label="F1 Score"
    )
    
    # Make the figure more square-like by adjusting height
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'graph-extract_node.pdf'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create edge metrics figure - same layout as node metrics
    fig_edge = plt.figure(figsize=(20, 9))  # Increased from (18, 8)
    
    # Edge metrics plots
    ax4 = fig_edge.add_subplot(1, 3, 1)
    ax5 = fig_edge.add_subplot(1, 3, 2)
    ax6 = fig_edge.add_subplot(1, 3, 3)
    
    # Plot edge metrics with level grouping
    plot_box_metric(
        combined_df, 
        ax4, 
        levels, 
        'edge_precision', 
        model_colors, 
        method_markers, 
        method_colors=method_colors,
        level_colors=level_colors,
        group_by_level=True,
        show_legend=True,
        scatter_color=scatter_color,
        use_jitter=True,
        y_label="Precision"
    )
    
    plot_box_metric(
        combined_df, 
        ax5, 
        levels, 
        'edge_recall', 
        model_colors, 
        method_markers, 
        method_colors=method_colors,
        level_colors=level_colors,
        group_by_level=True,
        scatter_color=scatter_color,
        use_jitter=True,
        y_label="Recall"
    )
    
    plot_box_metric(
        combined_df, 
        ax6, 
        levels, 
        'edge_f1', 
        model_colors, 
        method_markers, 
        method_colors=method_colors,
        level_colors=level_colors,
        group_by_level=True,
        scatter_color=scatter_color,
        use_jitter=True,
        y_label="F1 Score"
    )
    
    # Make the figure more square-like by adjusting height
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'graph-extract_edge.pdf'), dpi=300, bbox_inches='tight')
    plt.close('all')

def plot_box_metric(df, ax, levels, metric, model_colors, method_markers, 
                   lower_is_better=False, level_colors=None, group_by_level=False,
                   show_legend=False, scatter_color=None, use_jitter=False, y_label=None,
                   method_colors=None):
    """Helper function to plot a particular metric using boxplots with individual data points"""
    
    # Prepare data for plotting
    plot_data = []
    error_data = []
    
    # Group by method, model, level, and instance
    grouped = df.groupby(['method', 'model', 'level', 'instance_name'])
    
    methods = sorted(df['method'].unique())
    # Ensure models are in the right order
    models = ['qwen2.5:7b', 'qwen2.5:72b', 'gpt-4o']
    
    # Calculate mean and SEM for each instance, method, model, and level
    for level in levels:  # Group by level first to keep same levels together
        for method in methods:
            for model in models:
                # Get all instances for this combination
                level_instances = df[(df['method'] == method) & 
                                  (df['model'] == model) & 
                                  (df['level'] == level)]['instance_name'].unique()
                
                for instance_name in sorted(level_instances):
                    if (method, model, level, instance_name) in grouped.indices:
                        instance_data = grouped.get_group((method, model, level, instance_name))
                        mean_value = instance_data[metric].mean()
                        sem_value = instance_data[metric].sem()
                        
                        # Store for plotting
                        plot_data.append({
                            'method': method,
                            'model': model,
                            'level': level,
                            'value': mean_value,
                            'instance': instance_name,
                            'x_position': f"{level}-{method}-{model}" if group_by_level else f"{method}-{level}-{model}"
                        })
                        
                        error_data.append({
                            'method': method,
                            'model': model,
                            'level': level,
                            'value': mean_value,
                            'sem': sem_value,
                            'instance': instance_name,
                            'x_position': f"{level}-{method}-{model}" if group_by_level else f"{method}-{level}-{model}"
                        })
    
    # Convert to DataFrame for easier plotting
    plot_df = pd.DataFrame(plot_data)
    error_df = pd.DataFrame(error_data)
    
    # Create x-axis positions with custom spacing
    x_positions = []
    x_labels = []
    position_counter = 0
    
    # Define consistent spacing
    model_spacing = 0.3       # Space between models in the same method
    method_spacing = 1.0      # Space between different methods (increased for more room)
    level_spacing = 1.2       # Space between different levels (increased for more room)
    
    if group_by_level:
        # Group by level first, then method, then model (7b, 72b, gpt-4o)
        for level_idx, level in enumerate(levels):
            for method_idx, method in enumerate(methods):
                # Position for first model (7b)
                x_positions.append(position_counter)
                # Position for second model (72b)
                x_positions.append(position_counter + model_spacing)
                # Position for third model (gpt-4o)
                x_positions.append(position_counter + 2 * model_spacing)
                
                # Only add label for first model of each method group
                x_labels.append(f"{method}")
                x_labels.append("")
                x_labels.append("")
                
                # Add space after this method (except the last one in this level)
                position_counter += 2 * model_spacing + (method_spacing if method_idx < len(methods)-1 else 0)
            
            # Add extra space between levels (if not the last level)
            if level_idx < len(levels)-1:
                position_counter += level_spacing
    else:
        # Group by method first, then level, then model
        for method_idx, method in enumerate(methods):
            for level_idx, level in enumerate(levels):
                # Position for first model (7b)
                x_positions.append(position_counter)
                # Position for second model (72b)
                x_positions.append(position_counter + model_spacing)
                # Position for third model (gpt-4o)
                x_positions.append(position_counter + 2 * model_spacing)
                
                # Only add label for first model of each level group
                x_labels.append(f"{method}\n{level}")
                x_labels.append("")
                x_labels.append("")
                
                # Add space after this level (except the last one for this method)
                position_counter += 2 * model_spacing + (method_spacing if level_idx < len(levels)-1 else 0)
            
            # Add extra space between methods (if not the last method)
            if method_idx < len(methods)-1:
                position_counter += level_spacing
    
    # Create a mapping from position keys to actual positions
    position_map = {}
    
    if group_by_level:
        # Create mapping for level-method-model to position
        model_idx = 0
        for level in levels:
            for method in methods:
                for model in models:
                    position_map[f"{level}-{method}-{model}"] = x_positions[model_idx]
                    model_idx += 1
    else:
        # Create mapping for method-level-model to position
        model_idx = 0
        for method in methods:
            for level in levels:
                for model in models:
                    position_map[f"{method}-{level}-{model}"] = x_positions[model_idx]
                    model_idx += 1
    
    # Create level sections behind the plots if level colors provided
    if level_colors and group_by_level:
        level_boundaries = []
        level_midpoints = []
        
        current_pos = 0
        for i, level in enumerate(levels):
            # Calculate the start position for this level
            level_start = x_positions[i * len(methods) * 3] - model_spacing/2
            
            # Calculate the end position for this level
            if i < len(levels) - 1:
                level_end = x_positions[(i+1) * len(methods) * 3] - model_spacing/2 - level_spacing/2
            else:
                # For the last level, calculate based on the last position
                last_pos_idx = len(x_positions) - 1
                level_end = x_positions[last_pos_idx] + model_spacing/2
            
            level_width = level_end - level_start
            level_boundaries.append((level_start, level_width))
            level_midpoints.append(level_start + level_width/2)
            
            # Add colored background for this level
            rect = plt.Rectangle(
                (level_start, ax.get_ylim()[0]),
                level_width,
                ax.get_ylim()[1] - ax.get_ylim()[0],
                color=level_colors[level],
                alpha=0.3,
                zorder=-10
            )
            ax.add_patch(rect)
            
            # Add level label at the top (non-bold)
            ax.text(
                level_midpoints[i],
                1.02,
                level.capitalize(),
                transform=ax.get_xaxis_transform(),
                ha='center',
                va='bottom',
                fontsize=18,
                fontweight='normal'
            )
            
            # Add vertical grid lines to separate levels
            if i > 0:
                # Place grid line exactly between levels
                grid_pos = (x_positions[i * len(methods) * 3] - level_spacing/2)
                ax.axvline(
                    x=grid_pos,
                    color='gray',
                    linestyle='-',
                    linewidth=0.8,
                    alpha=0.7
                )
            
            # Highlight different methods with different background colors if provided
            if method_colors:
                for j, method in enumerate(methods):
                    # Calculate the start and end indices for this method
                    start_idx = i * len(methods) * 3 + j * 3
                    
                    # Start position for this method (slightly left of the first box)
                    method_start = x_positions[start_idx] - model_spacing/2
                    
                    # End position for this method
                    if j < len(methods) - 1:
                        # If not the last method, end halfway to the next method
                        next_method_idx = start_idx + 3
                        method_end = (x_positions[next_method_idx] + x_positions[start_idx + 1]) / 2
                    else:
                        # If last method in level, end at level boundary
                        method_end = level_end
                    
                    method_width = method_end - method_start
                    
                    # Add colored background for this method
                    method_rect = plt.Rectangle(
                        (method_start, ax.get_ylim()[0]),
                        method_width,
                        ax.get_ylim()[1] - ax.get_ylim()[0],
                        color=method_colors[method],
                        alpha=0.2,
                        zorder=-5
                    )
                    ax.add_patch(method_rect)
                    
                    # Add vertical line between methods (within a level)
                    if j > 0:
                        # Place grid line exactly between methods
                        grid_pos = (x_positions[start_idx] + x_positions[start_idx-1]) / 2
                        ax.axvline(
                            x=grid_pos,
                            color='black',
                            linestyle=':',
                            linewidth=0.6,
                            alpha=0.5
                        )
    
    # Map each x_position string to its actual coordinate
    plot_df['x_coord'] = plot_df['x_position'].map(position_map)
    error_df['x_coord'] = error_df['x_position'].map(position_map)
    
    # 1. Create boxplots
    boxplot_positions = []
    boxplot_data = []
    boxplot_colors = []
    
    # Organize data for boxplots with model color
    for x_pos in plot_df['x_position'].unique():
        subset = plot_df[plot_df['x_position'] == x_pos]
        if not subset.empty:
            # Use the mapped position
            pos = position_map[x_pos]
            boxplot_positions.append(pos)
            boxplot_data.append(subset['value'].values)
            
            # Extract the model from the position string
            pos_parts = x_pos.split('-')
            pos_parts_model_part = pos_parts[2:]
            model = '-'.join(pos_parts_model_part) if len(pos_parts_model_part) > 1 else pos_parts_model_part[0]
            boxplot_colors.append(model_colors[model])
    
    # Create the boxplot
    bp = ax.boxplot(
        boxplot_data,
        positions=boxplot_positions,
        patch_artist=True,
        widths=0.25,  # Narrower boxes
        showcaps=True,
        showfliers=False,
        medianprops={'color': 'black', 'linewidth': 1.5},
        boxprops={'linewidth': 1.0},
        whiskerprops={'linewidth': 1.0},
        capprops={'linewidth': 1.0}
    )
    
    # Set box colors
    for box, color in zip(bp['boxes'], boxplot_colors):
        box.set(facecolor=color, alpha=0.7)
    
    # 2. Add individual data points with error bars
    for idx, row in error_df.iterrows():
        x_pos = row['x_coord']
        
        # Add jitter if requested - make it smaller for tighter spacing
        jitter = np.random.uniform(-0.1, 0.1) if use_jitter else 0
        
        # Plot the point with error bar
        marker = method_markers[row['method']]
        ax.errorbar(
            x=x_pos + jitter,
            y=row['value'],
            yerr=row['sem'],
            fmt=marker,
            color=scatter_color if scatter_color else model_colors[row['model']],
            alpha=0.5,
            markersize=4,
            elinewidth=0.8,
            capsize=2
        )
    
    # Make the plot more square-like by adjusting height
    ax.set_aspect(1.0/ax.get_data_ratio()*0.8)
    
    # Configure x-axis with custom ticks
    x_ticks = []
    for i in range(0, len(x_positions), 3):
        x_ticks.append((x_positions[i] + x_positions[i+1] + x_positions[i+2]) / 3)
    
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([x_labels[i] for i in range(0, len(x_labels), 3)], 
                       fontsize=16, rotation=45, ha='right')
    
    # Set x-axis limits with some padding
    ax.set_xlim(min(x_positions) - 0.5, max(x_positions) + 0.5)
    
    # Set y-axis label
    if y_label:
        ax.set_ylabel(y_label, fontsize=18)
    elif lower_is_better:
        ax.set_ylabel('Value (lower is better)', fontsize=16)
    else:
        ax.set_ylabel('Value', fontsize=13)
    
    # Add grid for better readability
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    
    # Add custom legend
    if show_legend:
        # Create custom legend entries
        legend_elements = []
        
        # Add model entries
        legend_elements.append(plt.Rectangle((0,0), 1, 1, fc=model_colors['qwen2.5:7b'], ec="none", alpha=0.7, label="Qwen2.5-7B"))
        legend_elements.append(plt.Rectangle((0,0), 1, 1, fc=model_colors['qwen2.5:72b'], ec="none", alpha=0.7, label="Qwen2.5-72B"))
        legend_elements.append(plt.Rectangle((0,0), 1, 1, fc=model_colors['gpt-4o'], ec="none", alpha=0.7, label="GPT-4o"))
        
        # Add method entries with marker symbols only (no color legend)
        # Use grey color for markers in the legend
        # legend_elements.append(plt.Line2D([0], [0], marker=method_markers['joint'], 
        #                               linestyle='None', markersize=7,  # Increased from 6 
        #                               markerfacecolor='none', markeredgecolor='#555555',
        #                               label='Joint'))
        # legend_elements.append(plt.Line2D([0], [0], marker=method_markers['sequential'], 
        #                               linestyle='None', markersize=7,  # Increased from 6
        #                               markerfacecolor='none', markeredgecolor='#555555',
        #                               label='Sequential'))
        
        ax.legend(handles=legend_elements, fontsize=14, loc='best', framealpha=0.7)  # Increased from 9
    else:
        # Remove legend if present
        if ax.get_legend():
            ax.get_legend().remove()

    # Increase y-axis tick fontsize
    ax.tick_params(axis='y', labelsize=11)

def create_distribution_plots(df, ax, title, metrics, metric_labels, model_colors, lower_is_better=False):
    """Helper function to create violin or box plots for metric distributions"""
    ax.set_title(title, fontsize=13)  # Increased from 10
    
    # Create a more compact dataframe for plotting
    plot_data = []
    for method in sorted(df['method'].unique()):
        for model in sorted(df['model'].unique()):
            # Get all instances for this method-model combination
            subset = df[(df['method'] == method) & (df['model'] == model)]
            
            # Group by instance name and get mean of each metric across replicates
            instance_means = subset.groupby('instance_name')
            
            for metric, metric_label in zip(metrics, metric_labels):
                for instance_name, group in instance_means:
                    plot_data.append({
                        'method': method,
                        'model': model,
                        'metric': metric_label,
                        'value': group[metric].mean(),
                        'instance': instance_name
                    })
    
    plot_df = pd.DataFrame(plot_data)
    
    # For smaller subplots, use boxplots instead of violin plots
    for i, (metric, metric_label) in enumerate(zip(metrics, metric_labels)):
        metric_data = plot_df[plot_df['metric'] == metric_label]
        
        # Calculate positions for each method and metric
        positions = []
        labels = []
        offset = 0.3
        for m_idx, method in enumerate(sorted(df['method'].unique())):
            # Make sure to handle all three models now
            for model in ['qwen2.5:7b', 'qwen2.5:72b', 'gpt-4o']:  # Use fixed order
                positions.append(i + offset)
                labels.append(f"{method.split('-')[0]} ({model})")
                offset += 0.2
        
        # Use boxplot for more compact representation
        method_model_groups = metric_data.groupby(['method', 'model'])
        boxes = []
        colors = []
        positions = []
        
        pos = i * 4  # Increase spacing to accommodate additional model
        for (method, model), group in method_model_groups:
            boxes.append(group['value'].values)
            colors.append(model_colors[model])
            positions.append(pos)
            pos += 0.5
        
        # Make boxplots more compact
        boxprops = dict(linewidth=1.0)
        whiskerprops = dict(linewidth=1.0)
        capprops = dict(linewidth=1.0)
        medianprops = dict(linewidth=1.5)
        
        bp = ax.boxplot(boxes, positions=positions, patch_artist=True,
                       boxprops=boxprops, whiskerprops=whiskerprops,
                       capprops=capprops, medianprops=medianprops,
                       widths=0.35)
        
        # Color the boxes based on model
        for box, color in zip(bp['boxes'], colors):
            box.set(facecolor=color, alpha=0.6)
    
    # Set axis labels and format
    ax.set_xlabel('', fontsize=11)  # Increased from 9
    ax.set_xticks([])  # Hide x-ticks for clarity
    
    if lower_is_better:
        ax.set_ylabel('Value (lower is better)', fontsize=11)  # Increased from 9
    else:
        ax.set_ylabel('Value', fontsize=11)  # Increased from 9
    
    # Add metric labels
    if title.endswith('Node Metrics'):
        for i, label in enumerate(metric_labels):
            ax.text(i*4 + 0.25, ax.get_ylim()[0] - (ax.get_ylim()[1] - ax.get_ylim()[0])*0.08, 
                   label, fontsize=10, ha='center')  # Increased from 8
    
    # Only add legend to first subplot
    if title == 'Background - Node Metrics':
        # Create custom legend entries
        legend_elements = []
        methods = sorted(df['method'].unique())
        models = sorted(df['model'].unique())
        
        for model in models:
            legend_elements.append(plt.Line2D([0], [0], color=model_colors[model], lw=4, label=model))
        
        # Add method indicators
        for method in methods:
            style = 'solid' if 'joint' in method else 'dashed'
            legend_elements.append(plt.Line2D([0], [0], color='gray', lw=2, label=method.split('-')[0],
                                            linestyle=style))
        
        ax.legend(handles=legend_elements, fontsize=10, title_fontsize=11,  # Increased from 8 and 9
                 title='Models and Methods', loc='upper right')

def analyzeResults(
    result_path
):
    """
    Analyze the results of the graph extraction/generation.
    """
    results_df = pd.read_csv(result_path)
    print(os.path.basename(result_path).split(".")[0])
    for level in ["background", "node", "graph"]:
        level_df = results_df[results_df["level"] == level]
        
        # 1. Mean metrics acorss replications.
        # calculate the mean and std of the average precision, recall, and f1 score across all replicates.
        # Use the more verbose approach to avoid errors with non-numeric columns
        mean_node_precision = level_df.groupby("instance_name")["node_precision"].mean().mean()
        std_node_precision = level_df.groupby("instance_name")["node_precision"].mean().std()
        mean_node_recall = level_df.groupby("instance_name")["node_recall"].mean().mean()
        std_node_recall = level_df.groupby("instance_name")["node_recall"].mean().std()
        mean_node_f1 = level_df.groupby("instance_name")["node_f1"].mean().mean()
        std_node_f1 = level_df.groupby("instance_name")["node_f1"].mean().std()
        mean_edge_precision = level_df.groupby("instance_name")["edge_precision"].mean().mean()
        std_edge_precision = level_df.groupby("instance_name")["edge_precision"].mean().std()
        mean_edge_recall = level_df.groupby("instance_name")["edge_recall"].mean().mean()
        std_edge_recall = level_df.groupby("instance_name")["edge_recall"].mean().std()
        mean_edge_f1 = level_df.groupby("instance_name")["edge_f1"].mean().mean()
        std_edge_f1 = level_df.groupby("instance_name")["edge_f1"].mean().std()
        mean_n_ged = level_df.groupby("instance_name")["n_ged"].mean().mean()
        std_n_ged = level_df.groupby("instance_name")["n_ged"].mean().std()
        print(f"\nResults for level: {level}")
        print(f"  Node metrics:")
        print(f"    Precision: ${mean_node_precision:.2f} \pm {std_node_precision:.2f}$")
        print(f"    Recall:    ${mean_node_recall:.2f} \pm {std_node_recall:.2f}$")
        print(f"    F1 Score:  ${mean_node_f1:.2f} \pm {std_node_f1:.2f}$")
        print(f"  Edge metrics:")
        print(f"    Precision: ${mean_edge_precision:.2f} \pm {std_edge_precision:.2f}$")
        print(f"    Recall:    ${mean_edge_recall:.2f} \pm {std_edge_recall:.2f}$")
        print(f"    F1 Score:  ${mean_edge_f1:.2f} \pm {std_edge_f1:.2f}$")
        print(f"  Graph Edit Distance:")
        print(f"    Normalized: ${mean_n_ged:.2f} \pm {std_n_ged:.2f}$")

        # 2. Coefficient of variation.
        # For the coefficient of variation, we need to calculate std/mean for each instance first
        instance_cv_node_precision = level_df.groupby("instance_name")["node_precision"].std() / level_df.groupby("instance_name")["node_precision"].mean()
        instance_cv_node_recall = level_df.groupby("instance_name")["node_recall"].std() / level_df.groupby("instance_name")["node_recall"].mean() 
        instance_cv_node_f1 = level_df.groupby("instance_name")["node_f1"].std() / level_df.groupby("instance_name")["node_f1"].mean()
        instance_cv_edge_precision = level_df.groupby("instance_name")["edge_precision"].std() / level_df.groupby("instance_name")["edge_precision"].mean()
        instance_cv_edge_recall = level_df.groupby("instance_name")["edge_recall"].std() / level_df.groupby("instance_name")["edge_recall"].mean()
        instance_cv_edge_f1 = level_df.groupby("instance_name")["edge_f1"].std() / level_df.groupby("instance_name")["edge_f1"].mean()
        instance_cv_n_ged = level_df.groupby("instance_name")["n_ged"].std() / level_df.groupby("instance_name")["n_ged"].mean()
        
        # Then calculate mean and std of these CVs
        mean_cv_node_precision = instance_cv_node_precision.mean()
        std_cv_node_precision = instance_cv_node_precision.std()
        mean_cv_node_recall = instance_cv_node_recall.mean()
        std_cv_node_recall = instance_cv_node_recall.std()
        mean_cv_node_f1 = instance_cv_node_f1.mean()
        std_cv_node_f1 = instance_cv_node_f1.std()
        mean_cv_edge_precision = instance_cv_edge_precision.mean()
        std_cv_edge_precision = instance_cv_edge_precision.std()
        mean_cv_edge_recall = instance_cv_edge_recall.mean()
        std_cv_edge_recall = instance_cv_edge_recall.std()
        mean_cv_edge_f1 = instance_cv_edge_f1.mean()
        std_cv_edge_f1 = instance_cv_edge_f1.std()
        mean_cv_n_ged = instance_cv_n_ged.mean()
        std_cv_n_ged = instance_cv_n_ged.std()
        
        print(f"\nCoefficient of variation for level: {level}")
        print(f"  Node precision: ${mean_cv_node_precision:.2f} \pm {std_cv_node_precision:.2f}$")
        print(f"  Node recall:    ${mean_cv_node_recall:.2f} \pm {std_cv_node_recall:.2f}$")
        print(f"  Node F1 score:  ${mean_cv_node_f1:.2f} \pm {std_cv_node_f1:.2f}$")
        print(f"  Edge precision: ${mean_cv_edge_precision:.2f} \pm {std_cv_edge_precision:.2f}$")
        print(f"  Edge recall:    ${mean_cv_edge_recall:.2f} \pm {std_cv_edge_recall:.2f}$")
        print(f"  Edge F1 score:  ${mean_cv_edge_f1:.2f} \pm {std_cv_edge_f1:.2f}$")
        print(f"  Graph Edit Distance: ${mean_cv_n_ged:.2f} \pm {std_cv_n_ged:.2f}$")
        

async def analyzeGraph(
    exp_path: str,
):
    """
    Analyze the graph extraction/generation.
    """
    data_dir = os.path.join(project_root, "data")
    instance_names = [f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))]
    instance_paths = [os.path.join(data_dir, instance_name) for instance_name in instance_names]

    n_replicate = int(exp_path.split("_")[-1].split("replicate")[-1])

    results = []
    for level in ["background", "node", "graph"]:
        for i, instance_path in enumerate(instance_paths):
            instance_name = instance_names[i]
            print(f"Analyzing {instance_name} {level}")
            # Create tasks for all replicates concurrently
            tasks = []
            for replicate_id in range(n_replicate):
                task = analyzeGraphTrial(
                    instance_path,
                    exp_path,
                    level,
                    replicate_id,
                    graphMetricsLLM
                )
                tasks.append(task)
            
            # Run all tasks concurrently and wait for results
            replicate_results = await asyncio.gather(*tasks)
            
            # Process the results
            for replicate_id, (node_precision, node_recall, node_f1, edge_precision, edge_recall, edge_f1, normalized_graph_edit_distance) in enumerate(replicate_results):
                results.append({
                    "instance_name": instance_name,
                    "level": level,
                    "replicate_id": replicate_id,
                    "node_precision": node_precision,
                    "node_recall": node_recall,
                    "node_f1": node_f1,
                    "edge_precision": edge_precision,
                    "edge_recall": edge_recall,
                    "edge_f1": edge_f1,
                    "n_ged": normalized_graph_edit_distance
                })

    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(project_root, "analysis", "results", "exp-graph", exp_path+".csv"), index=False, encoding="utf-8")
    return results_df

# Wrapper to run the async function
def analyzeGraph_sync(exp_path: str):
    """Synchronous wrapper for analyzeGraph"""
    return asyncio.run(analyzeGraph(exp_path))

async def analyzeGraphTrial(
    instance_path: str,
    exp_path: str,
    level: str,
    replicate_id: int,
    metric: object
):
    """
    Analyze the graph extraction/generation.

    Parameters
    ----------
    instance_path: str
        The path to the instance.
    exp_path: str
        The path to the experiment.
        {method_name}_{model_name}_{time_stamp}
    level: str
        The level of the graph.
        {background, node, graph}
    replicate_id: int
        The replicate id.
    metric: object
        The metric to use.
        metric(target_graph, generated_graph) -> float
    """
    instance_name = os.path.basename(instance_path)
    method_name, model_name, time_stamp, n_replicate = exp_path.split("_")

    # Load the target graph.
    target_graph_dir = os.path.join(
        project_root,
        instance_path,
        instance_name.split("-")[-1] + "-base"
    )
    node_list = NodeList([])
    node_list.load(target_graph_dir)
    edge_list = EdgeList([])
    edge_list.load(target_graph_dir)


    # load the generated graph.
    generated_graph_dir = os.path.join(
        project_root,
        "output",
        "exp-graph",
        exp_path,
        instance_name+"_"+level+"_"+str(replicate_id)
    )
    try:
        generated_node_list = NodeList([])
        generated_node_list.load(generated_graph_dir)
        generated_edge_list = EdgeList([])
        generated_edge_list.load(generated_graph_dir)

        return await metric(node_list, edge_list, generated_node_list, generated_edge_list, verbose=False)
    except Exception as e:
        return 0, 0, 0, 0, 0, 0, 1

# Function to run synchronous analysis for backward compatibility
def analyzeGraphTrial_sync(
    instance_path: str,
    exp_path: str,
    level: str,
    replicate_id: int
):
    """Synchronous version of analyzeGraphTrial"""
    instance_name = os.path.basename(instance_path)
    method_name, model_name, time_stamp, n_replicate = exp_path.split("_")

    # Load the target graph.
    target_graph_dir = os.path.join(
        project_root,
        instance_path,
        instance_name.split("-")[-1] + "-base"
    )
    node_list = NodeList([])
    node_list.load(target_graph_dir)
    edge_list = EdgeList([])
    edge_list.load(target_graph_dir)

    # load the generated graph.
    generated_graph_dir = os.path.join(
        project_root,
        "output",
        "exp-graph",
        exp_path,
        instance_name+"_"+level+"_"+str(replicate_id)
    )
    generated_node_list = NodeList([])
    generated_node_list.load(generated_graph_dir)
    generated_edge_list = EdgeList([])
    generated_edge_list.load(generated_graph_dir)

    return graphMetricsLLM_sync(node_list, edge_list, generated_node_list, generated_edge_list, verbose=False)


if __name__ == "__main__":
    exp_path_list = [
        "joint-extraction_qwen2.5:7b_20250514-1129_replicate5",
        "sequential-extraction_qwen2.5:7b_20250514-1206_replicate5",
        "joint-extraction_qwen2.5:72b_20250514-1037_replicate5",
        "sequential-extraction_qwen2.5:72b_20250514-0937_replicate5",
        "joint-extraction_gpt-4o_20250514-1258_replicate5",
        "sequential-extraction_gpt-4o_20250514-1246_replicate5"
    ]

    for exp_path in exp_path_list:
    #     print(f"Analyzing {exp_path}")
    #     analyzeGraph_sync(
    #         exp_path
    #     )
        analyzeResults(
            os.path.join(
                "analysis",
                "results",
                "exp-graph",
                exp_path+".csv"
            )
        )
    # plotResults(
    #     [os.path.join(
    #         "analysis",
    #         "results",
    #         "exp-graph",
    #         exp_path+".csv") for exp_path in exp_path_list],
    #     "analysis/results/exp-graph"
    # )

    
