import numpy as np 
import plotly.graph_objects as go

def plot_metrics(metric, lspline, rspline):
    """
    Compare left and right somite curves in interactive 3D plot.
    
    Params
    -------
        results: dict - output from compare_somites containing curve data
        smoothing: int - smoothing parameter for spline fitting
    Returns
    -------
        None - displays plotly interactive figure
    """    
    if metric == 'kappa':
        r_string = "κ"
        title = "Curvature"
    elif metric=='tau':
        r_string = "τ"
        title = "Torsion"
    colors_l = lspline.features[f'{metric}_centroids']
    colors_r = rspline.features[f'{metric}_centroids']
    l_stages = lspline.lookup_table['somite_stage'].to_numpy()
    r_stages = rspline.lookup_table['somite_stage'].to_numpy()
    combined_fig = go.Figure()
    # Left centroids
    combined_fig.add_trace(go.Scatter3d(
        x=lspline.features['centroids'][:, 0], 
        y=lspline.features['centroids'][:, 1], 
        z=lspline.features['centroids'][:, 2],
        mode='markers',
        customdata=l_stages,  # Add your stage data here
        hovertemplate=(
        "<b>Left </b><br>"
        "X, Y, Z: %{x:.1f}, %{y:.1f}, %{z:.1f}<br>"
        f"{r_string}: %{{marker.color}}<br>"
        "Stage: %{customdata}<br>"
        "<extra></extra>"
        ),
        marker=dict(
            color=colors_l,
            colorscale='Magma',
            coloraxis="coloraxis"  # Use first coloraxis
        ),
        name='Left Points'
    ))
    
    # Right centroids
    combined_fig.add_trace(go.Scatter3d(
        x=rspline.features['centroids'][:, 0], 
        y=rspline.features['centroids'][:, 1], 
        z=rspline.features['centroids'][:, 2],
        mode='markers',
        marker=dict(
            color=colors_r,
            coloraxis="coloraxis2"  # Use second coloraxis
        ),
        customdata=r_stages,
        hovertemplate=(
        "<b>Right </b><br>"
        "X, Y, Z: %{x:.1f}, %{y:.1f}, %{z:.1f}<br>"
        f"{r_string}: %{{marker.color}}<br>"
        "Stage: %{customdata}<br>"
        "<extra></extra>"
        ),
        name='Right Points'
    ))
    
    # Left spline 
    combined_fig.add_trace(go.Scatter3d(
        x=lspline.features['r(t)'][:, 0], 
        y=lspline.features['r(t)'][:, 1],
        z=lspline.features['r(t)'][:, 2],
        mode='lines',
        line=dict(color='black', width=4),
        name='Left Spline',
        hoverinfo='none',
    ))
    
    # Right spline curve
    combined_fig.add_trace(go.Scatter3d(
        x=rspline.features['r(t)'][:, 0], 
        y=rspline.features['r(t)'][:, 1], 
        z=rspline.features['r(t)'][:, 2],
        mode='lines',
        line=dict(color='black', width=4),
        name='Right Spline',
        hoverinfo='none'
    ))

    combined_fig.update_layout(
        coloraxis=dict(
            colorscale='Viridis',
            colorbar=dict(
                title="Left " + r_string,
                x=1.02,           # First position
                y=0.5,            # Centered vertically
                len=0.8           
            )
        ),
        coloraxis2=dict(
            colorscale='Viridis',
            colorbar=dict(
                title="Right " + r_string,
                x=1.2,           # Further right
                y=0.5,            # Centered vertically
                len=0.8         
            )
        ),
        scene=dict(
            xaxis_title='X μm', 
            yaxis_title='Y μm', 
            zaxis_title='Z μm', 
            aspectmode='data'
        ),
        title=f"L/R {title} ({r_string})",
        margin=dict(r=250, l=200, t=100, b=100),  # Increased right margin
        showlegend=False,
        width=1000  # Increased width to accommodate both colorbars
    )
    combined_fig.show()

def plot_curves(lspline, rspline):
    combined_fig = go.Figure()
    # Left centroids
    l_stages = lspline.lookup_table['somite_stage'].to_numpy()
    r_stages = rspline.lookup_table['somite_stage'].to_numpy()
    combined_fig = go.Figure()
    # Left centroids
    combined_fig.add_trace(go.Scatter3d(
        x=lspline.features['centroids'][:, 0], 
        y=lspline.features['centroids'][:, 1], 
        z=lspline.features['centroids'][:, 2],
        mode='markers',
        customdata=l_stages,  # Add your stage data here
        hovertemplate=(
        "<b>Left </b><br>"
        "X, Y, Z: %{x:.1f}, %{y:.1f}, %{z:.1f}<br>"
        "Stage: %{customdata}<br>"
        "<extra></extra>"
        ),
        marker=dict(
            color='blue'
        ),
        name='Left Points'
    ))
    
    # Right centroids
    combined_fig.add_trace(go.Scatter3d(
        x=rspline.features['centroids'][:, 0], 
        y=rspline.features['centroids'][:, 1], 
        z=rspline.features['centroids'][:, 2],
        mode='markers',
        customdata=r_stages, 
        hovertemplate=(
        "<b>Left </b><br>"
        "X, Y, Z: %{x:.1f}, %{y:.1f}, %{z:.1f}<br>"
        "Stage: %{customdata}<br>"
        "<extra></extra>"
        ),
        marker=dict(
            color='red'
        ),
        name='Left Points'
    ))
    
    # Left spline 
    combined_fig.add_trace(go.Scatter3d(
        x=lspline.features['r(t)'][:, 0], 
        y=lspline.features['r(t)'][:, 1],
        z=lspline.features['r(t)'][:, 2],
        mode='lines',
        line=dict(color='black', width=4),
        name='Left Spline',
        hoverinfo='none',
    ))
    
    # Right spline curve
    combined_fig.add_trace(go.Scatter3d(
        x=rspline.features['r(t)'][:, 0], 
        y=rspline.features['r(t)'][:, 1], 
        z=rspline.features['r(t)'][:, 2],
        mode='lines',
        line=dict(color='black', width=4),
        name='Right Spline',
        hoverinfo='none'
    ))

    combined_fig.update_layout(
        coloraxis=dict(
            colorscale='Viridis',
            colorbar=dict(
                title="Left",
                x=1.02,           # First position
                y=0.5,            # Centered vertically
                len=0.8           
            )
        ),
        coloraxis2=dict(
            colorscale='Viridis',
            colorbar=dict(
                title="Right",
                x=1.2,           # Further right
                y=0.5,            # Centered vertically
                len=0.8         
            )
        ),
        scene=dict(
            xaxis_title='X μm', 
            yaxis_title='Y μm', 
            zaxis_title='Z μm', 
            aspectmode='data'
        ),
        title=f"L/R Somite Splines",
        margin=dict(r=250, l=200, t=100, b=100),  # Increased right margin
        showlegend=False,
        width=1000  # Increased width to accommodate both colorbars
    )
    combined_fig.show()

def plot_procrustes_translation(l_proscr, r_proscr):
    stages = np.arange(1, 28)
    combined_fig = go.Figure()
    combined_fig.add_trace(go.Scatter3d(
        x=l_proscr[:, 0],
        y=l_proscr[:, 1],
        z=l_proscr[:, 2],
        mode='markers',      
        customdata=stages, 
        hovertemplate=(
        "<b>Left </b><br>"
        "X, Y, Z: %{x:.1f}, %{y:.1f}, %{z:.1f}<br>"
        "Stage: %{customdata}<br>"
        "<extra></extra>"
        ),
        marker=dict(size=2, color='blue'),
        name='Left Somites'
    ))
    combined_fig.add_trace(go.Scatter3d(
        x=r_proscr[:, 0],
        y=r_proscr[:, 1],
        z=r_proscr[:, 2],
        mode='markers',
        customdata=stages,
        hovertemplate=(
        "<b>Right </b><br>"
        "X, Y, Z: %{x:.1f}, %{y:.1f}, %{z:.1f}<br>"
        "Stage: %{customdata}<br>"
        "<extra></extra>"
        ),
        marker=dict(size=2, color='red'),
        name='Right Somites'
    ))
    combined_fig.update_layout(
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z'
        )
    )
    combined_fig.show()

def plot_ortho(mat):
    stages = np.arange(1, 28)
    combined_fig = go.Figure()
    combined_fig.add_trace(go.Scatter3d(
        x=mat[:, 0],
        y=mat[:, 1],
        z=mat[:, 2],
        mode='markers',      
        customdata=stages, 
        hovertemplate=(
        "<b>Left </b><br>"
        "X, Y, Z: %{x:.1f}, %{y:.1f}, %{z:.1f}<br>"
        "Stage: %{customdata}<br>"
        "<extra></extra>"
        ),
        marker=dict(size=2, color='blue'),
        name='Left Somites'
    ))
    combined_fig.show()