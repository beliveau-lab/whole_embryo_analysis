import paraview.simple as pv
import os, sys
import csv
import argparse
from datetime import date
import time
import math
import hashlib
import gc
import colorcet as cc
import random
import colorsys
from paraview import servermanager
import logging

# ffmpeg -framerate 5 -i frame_%04d.png  output2.gif
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def all_hex_to_rgb(hex_colors):
    def hex_to_rgb(hex):
        hex = hex.lstrip('#')
        return tuple(int(hex[i:i+2], 16) for i in (0, 2, 4))
    return [hex_to_rgb(hex) for hex in hex_colors]

def filter_colors(rgb_colors, min_saturation=0.3, min_lightness=0.2, max_lightness=0.7):
    """Filter colors by saturation and lightness.
    
    Keeps colors that are:
    - Saturated enough (not grayish)
    - Not too dark (visible on black background)
    - Not too light (not washed out/pale)
    """
    filtered = []
    for color in rgb_colors:
        r, g, b = color[0] / 255.0, color[1] / 255.0, color[2] / 255.0
        _, l, s = colorsys.rgb_to_hls(r, g, b)
        # if s >= min_saturation and min_lightness <= l <= max_lightness:
        #     filtered.append(color)
        if s >= min_saturation:
            filtered.append(color)
    return filtered

def hash_to_color(label_value, colors):
    """Assign a deterministic color to each label using hash-based indexing."""
    if label_value == 0:
        return 0.0, 0.0, 0.0
    # Use hash to get deterministic color per label (consistent across all views)
    hash_object = hashlib.sha256(str(label_value).encode('utf-8'))
    seed = int(hash_object.hexdigest(), 16)
    index = seed % len(colors)
    r, g, b = colors[index]
    # Normalize to 0-1 range for ParaView
    return r / 255.0, g / 255.0, b / 255.0

def generate_colormap(min_label, max_label, total_labels, colors):
    rgb_points = []
    num_samples = 100000
    step = max(1, total_labels // num_samples)  # Ensure step is at least 1
    for i in range(min_label, max_label+1, step):
        r, g, b = hash_to_color(i, colors)
        rgb_points.extend([i, r, g, b])
    return rgb_points

def build_colormap(min_label, max_label, colors):
    total_labels = max_label - min_label + 1
    rgb_points = generate_colormap(min_label, max_label, total_labels, colors)
    return rgb_points

def setup_mpi():
    controller = servermanager.vtkProcessModule.GetProcessModule().GetGlobalController()
    rank = controller.GetLocalProcessId()
    size = controller.GetNumberOfProcesses()
    
    if rank == 0:
        logger.info(f"Script started successfully - Rank {rank} of {size} processes")
        sys.stdout.flush()

    return rank, size

def load_data(collection_file, rank, decimated_path):
    if rank == 0:
        logger.info('='*40)
        logger.info("Loading and processing data")
        logger.info('='*40)
        sys.stdout.flush()
    
    src = pv.PVDReader(FileName=collection_file)
    src.UpdatePipeline()
    
    c2p = pv.CellDatatoPointData(Input=src)
    c2p.UpdatePipeline()
    
    dec = pv.Decimate(Input=c2p)
    dec.TargetReduction = 0.8
    dec.PreserveTopology = 1
    dec.UpdatePipeline()

    if rank == 0:
        orig_points = c2p.GetDataInformation().GetNumberOfPoints()
        new_points = dec.GetDataInformation().GetNumberOfPoints()
        logger.info(f"Decimation: {orig_points} -> {new_points} points ({(1-new_points/orig_points)*100:.1f}% reduction)")
    return dec, dec.GetDataInformation()

def find_labels(c2p, rank):
    data_info = c2p.GetDataInformation()
    point_data_info = data_info.GetPointDataInformation()

    segmentation_array_info = None
    array_names = []
    
    for i in range(point_data_info.GetNumberOfArrays()):
        array_info = point_data_info.GetArrayInformation(i)
        array_name = array_info.GetName()
        array_names.append(array_name)
        if array_name == 'SegmentationLabel':
            segmentation_array_info = array_info

    if segmentation_array_info is None:
        raise ValueError("'SegmentationLabel' array not found in point data after conversion.")
    elif rank == 0:
        logger.info(f"Available point arrays: {array_names}")
        sys.stdout.flush()
    
    return segmentation_array_info

def setup_display(c2p, rank):
    display = pv.Show(c2p)
    display.Representation = 'Surface'
    
    # Ensure all surfaces are visible using properties available in your ParaView version
    # try:
    #     display.BackfaceRepresentation = 'Surface'  # Show backfaces as surfaces
    #     display.BackfaceOpacity = 1.0               # Full opacity for backfaces
        
    #     display.Opacity = 1.0                       # Full opacity
    #     display.Visibility = 1                      # Ensure visibility
        
    #     display.UseShaderReplacements = 0
        
    #     display.DisableLighting = 0                 # Keep lighting enabled
        
    #     if rank == 0:
    #         logger.info("Display properties successfully configured for full surface visibility")
    #         sys.stdout.flush()
            
    # except Exception as e:
    #     if rank == 0:
    #         logger.info(f"Error setting display properties: {e}")
    #         sys.stdout.flush()
    return display

def setup_view(rank):
    """Configure the view settings for rendering.
    
    Returns
    -------
    view : paraview.simple.View
        The paraview view object
    """
    view = pv.GetActiveViewOrCreate('RenderView')
    
    # Disable LOD (Level of Detail) to prevent progressive rendering
    view.LODThreshold = 0.0  # Never use LOD (0 = disabled)
    view.RemoteRenderThreshold = 0  # Always do remote rendering
    view.ViewSize = [3000, 3000]
    
    # Set pure black background
    view.Background = [0.0, 0.0, 0.0]
    view.UseColorPaletteForBackground = 0  # Disable color palette override
    
    # Disable XYZ orientation axes in corner
    view.OrientationAxesVisibility = 0
    
    # Disable gradient background if available
    try:
        view.BackgroundColorMode = 'Single Color'
        view.Background2 = [0.0, 0.0, 0.0]  # Secondary color for gradient (set to black too)
        view.UseGradientBackground = 0
    except Exception:
        pass  # Older ParaView versions may not have these
    
    try:
        # Force full resolution LOD (1.0 = full resolution)
        view.LODResolution = 1.0
        
        # Additional settings to ensure complete rendering
        view.UseOutlineForLODRendering = 0  # Don't use outline placeholder
        view.NonInteractiveRenderDelay = 0  # No delay before full render
        view.StillRenderImageReductionFactor = 1  # No image reduction for still renders
        
    except Exception as e:
        if rank == 0:
            logger.info(f"Note: Some LOD settings not available: {e}")
            sys.stdout.flush()
    
    if rank == 0:
        logger.info("View configured with LOD disabled for complete rendering")
        sys.stdout.flush()
    
    return view

def apply_color_mapping(display, min_label, max_label, rank, colors)-> None:
    if rank == 0:
        logger.info("Applying colormap...")
        sys.stdout.flush()
        
    pv.ColorBy(display, ('POINTS', 'SegmentationLabel'))
    lut = pv.GetColorTransferFunction('SegmentationLabel')

    rgb_points = build_colormap(min_label, max_label, colors)
    
    if rank == 0:
        logger.info(f"Using {len(rgb_points)//4} color points for transfer function")
        sys.stdout.flush()
    
    lut.RescaleTransferFunction(min_label, max_label)
    lut.RGBPoints = rgb_points
    
    # Try to set discrete mode properties
    try:
        lut.UseLogScale = 0
        lut.Discretize = 1
        lut.NumberOfTableValues = min(65536, len(rgb_points) // 4)
    except Exception as e:
        if rank == 0:
            logger.info(f"Error setting color mapping properties: {e}")
            sys.stdout.flush()
    return

def save_state_file(output_dir, tag_str, rank):
    if rank == 0:
        os.makedirs(output_dir, exist_ok=True)
        state_file = os.path.join(output_dir, f'{tag_str}_state.pvsm')
        try:
            pv.SaveState(state_file)
            if os.path.exists(state_file):
                logger.info(f'ParaView state saved to: {state_file}')
            else:
                logger.info(f'Failed to save ParaView state to: {state_file}')
        except Exception as e:
            logger.info(f'Error saving ParaView state: {e}')
        sys.stdout.flush()

def save_colormap(output_dir, tag_str, min_label, max_label, rank, colors):
    if rank == 0:
        lut = pv.GetColorTransferFunction('SegmentationLabel')
        try:
            colormap_file = os.path.join(output_dir, f'{tag_str}_colormap.xml')
            pv.ExportLookupTable(FileName=colormap_file, LookupTable=lut)
            logger.info(f'Color map exported to: {colormap_file}')
        except:
            import xml.etree.ElementTree as ET
            import xml.dom.minidom
            os.makedirs(output_dir, exist_ok=True)
            rgb_points = build_colormap(min_label, max_label, colors)
            
            # Create XML structure following ParaView's documented format
            root = ET.Element('ColorMaps')
            colormap = ET.SubElement(root, 'ColorMap')
            colormap.set('name', f'Segmentation_{tag_str}')
            colormap.set('space', 'RGB')
            
            # Add color points (already in 0-1 range from hash_to_color)
            for i in range(0, len(rgb_points), 4):
                scalar_value = rgb_points[i]
                r = rgb_points[i+1]
                g = rgb_points[i+2]
                b = rgb_points[i+3]
                
                point = ET.SubElement(colormap, 'Point')
                point.set('x', f'{scalar_value:.3f}')
                point.set('o', '1')  # Full opacity
                point.set('r', f'{r:.6f}')
                point.set('g', f'{g:.6f}')
                point.set('b', f'{b:.6f}')
            
            # Pretty print the XML
            rough_string = ET.tostring(root, 'unicode')
            reparsed = xml.dom.minidom.parseString(rough_string)
            pretty_xml = reparsed.toprettyxml(indent='  ')
            
            # Remove empty lines and fix XML declaration
            lines = [line for line in pretty_xml.split('\n') if line.strip()]
            pretty_xml = '\n'.join(lines)
            
            xml_file = os.path.join(output_dir, f'{tag_str}_colormap.xml')
            with open(xml_file, 'w') as f:
                f.write(pretty_xml)
            logger.info(f'ParaView XML colormap saved to: {xml_file}')
        sys.stdout.flush()

def setup_camera(view, data_info, rank)-> None:
    if rank == 0:
        logger.info("Setting up camera to show full dataset...")
        sys.stdout.flush()
    
    view.ResetCamera()
    
    data_bounds = data_info.GetBounds()
    center_x = (data_bounds[0] + data_bounds[1]) / 2
    center_y = (data_bounds[2] + data_bounds[3]) / 2
    center_z = (data_bounds[4] + data_bounds[5]) / 2
    
    max_extent = max(data_bounds[1] - data_bounds[0], 
                     data_bounds[3] - data_bounds[2], 
                     data_bounds[5] - data_bounds[4])
    camera_distance = max_extent * 3
    
    view.Set(
        CameraPosition=[center_x, center_y, center_z + camera_distance],
        CameraFocalPoint=[center_x, center_y, center_z],
        CameraParallelScale=max_extent * 0.6,
    )
    
    if rank == 0:
        logger.info(f"Dataset bounds: {data_bounds}")
        logger.info(f"Camera distance: {camera_distance}")
        sys.stdout.flush()
    return

def force_complete_render(view, rank):
    """Force a complete render with all data loaded.
    
    This ensures that all data chunks are fully loaded and rendered
    before taking a screenshot, preventing partial/chunked renders
    during camera rotations.
    """
    # Force still render (full resolution, not progressive/LOD)
    view.StillRender()
    
    # Additional render pass to ensure everything is flushed
    pv.Render()
    
    # Synchronize across MPI processes
    controller = servermanager.vtkProcessModule.GetProcessModule().GetGlobalController()
    if controller and controller.GetNumberOfProcesses() > 1:
        controller.Barrier()
    
    return

def save_screenshot(output_file, rank)-> None:
    view = pv.GetActiveView()
    
    # Force complete render before screenshot
    force_complete_render(view, rank)
    
    if rank == 0:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        pv.SaveScreenshot(output_file)
        logger.info(f'Screenshot saved to: {output_file}')
        sys.stdout.flush()
    return 

def _rotate_camera(view, focal_point, radius, initial_y, start_angle, num_frames, frame_offset, animation_dir, rank):
    """Helper to perform a rotation and save frames."""
    focal_x, focal_y, focal_z = focal_point
    angle_step = 2 * math.pi / num_frames
    
    for i in range(num_frames):
        angle = start_angle + i * angle_step
        new_x = focal_x + radius * math.sin(angle)
        new_z = focal_z + radius * math.cos(angle)
        view.CameraPosition = [new_x, initial_y, new_z]
        view.CameraFocalPoint = [focal_x, focal_y, focal_z]
        
        if i % 5 == 0:
            gc.collect()
        
        # Force complete render to ensure all data chunks are loaded
        force_complete_render(view, rank)
        
        if rank == 0:
            frame_num = frame_offset + i
            frame_filename = os.path.join(animation_dir, f'frame_{frame_num:04d}.png')
            pv.SaveScreenshot(frame_filename)
            logger.info(f"Saved frame {frame_num + 1}")
            sys.stdout.flush()
    
    return frame_offset + num_frames

def _smooth_transition(view, start_focal, end_focal, start_scale, end_scale, 
                       start_pos, end_pos, num_frames, frame_offset, animation_dir, rank):
    """Smooth transition between two camera states."""
    for i in range(num_frames):
        t = i / max(1, num_frames - 1)  # 0 to 1
        # Ease in-out interpolation
        t = t * t * (3 - 2 * t)
        
        # Interpolate focal point
        focal = [start_focal[j] + t * (end_focal[j] - start_focal[j]) for j in range(3)]
        # Interpolate position
        pos = [start_pos[j] + t * (end_pos[j] - start_pos[j]) for j in range(3)]
        # Interpolate scale
        scale = start_scale + t * (end_scale - start_scale)
        
        view.CameraFocalPoint = focal
        view.CameraPosition = pos
        view.CameraParallelScale = scale
        
        # Force complete render to ensure all data chunks are loaded
        force_complete_render(view, rank)
        
        if rank == 0:
            frame_num = frame_offset + i
            frame_filename = os.path.join(animation_dir, f'frame_{frame_num:04d}.png')
            pv.SaveScreenshot(frame_filename)
            logger.info(f"Saved frame {frame_num + 1} (transition)")
            sys.stdout.flush()
    
    return frame_offset + num_frames

def generate_animation(view, animation_dir, num_frames, rank, data_info=None)-> None:
    """Generate multi-phase animation: rotate -> zoom to head -> rotate -> zoom out -> rotate.
    
    Parameters
    ----------
    view : paraview view
    animation_dir : str
    num_frames : int
        Frames per rotation phase
    rank : int
        MPI rank
    data_info : DataInformation, optional
        Data bounds info for calculating zoom target
    """
    if rank == 0:
        logger.info("--- Starting Multi-Phase Animation ---")
        os.makedirs(animation_dir, exist_ok=True)
        sys.stdout.flush()
    
    # Store initial camera state
    initial_focal = list(view.CameraFocalPoint)
    initial_position = list(view.CameraPosition)
    initial_scale = view.CameraParallelScale
    
    radius_vector = [initial_position[i] - initial_focal[i] for i in range(3)]
    radius = math.sqrt(sum([c*c for c in radius_vector]))
    initial_y = initial_position[1]
    
    # Calculate zoom target (top-left = head region)
    # Based on image: head is at higher Y (up) and lower X (left)
    if data_info:
        bounds = data_info.GetBounds()  # (xmin, xmax, ymin, ymax, zmin, zmax)
        # Target the upper-left region (head)
        zoom_focal = [
            bounds[0] + (bounds[1] - bounds[0]) * 0.3,  # 30% from left
            bounds[2] + (bounds[3] - bounds[2]) * 0.75, # 75% up (head region)
            (bounds[4] + bounds[5]) / 2                  # center Z
        ]
    else:
        # Fallback: shift focal point up and left
        zoom_focal = [
            initial_focal[0] - radius * 0.3,
            initial_focal[1] + radius * 0.3,
            initial_focal[2]
        ]
    
    # Zoom tuning: smaller zoom_factor = more zoomed in
    # Both scale and radius must decrease together to maintain proper view
    zoom_factor = 0.1  # Adjust this single value to control zoom level (0.1 = 10x zoom, 0.05 = 20x zoom)
    zoom_scale = initial_scale * zoom_factor
    zoom_radius = radius * zoom_factor  # Camera distance scales with zoom to prevent clipping
    
    frame_count = 0
    transition_frames = max(10, num_frames // 3)  # Smooth transitions
    
    # ===== PHASE 1: Initial full rotation =====
    if rank == 0:
        logger.info("Phase 1: Initial rotation")
    frame_count = _rotate_camera(
        view, initial_focal, radius, initial_y, 
        0, num_frames, frame_count, animation_dir, rank
    )
    
    # ===== PHASE 2: Zoom in to head =====
    if rank == 0:
        logger.info("Phase 2: Zooming to head region")
    
    # Calculate end position for zoom
    zoom_position = [
        zoom_focal[0],
        initial_y,
        zoom_focal[2] + zoom_radius
    ]
    
    frame_count = _smooth_transition(
        view, initial_focal, zoom_focal, initial_scale, zoom_scale,
        initial_position, zoom_position, transition_frames, frame_count, animation_dir, rank
    )
    
    # # ===== PHASE 3: Rotate while zoomed in =====
    # if rank == 0:
    #     logger.info("Phase 3: Zoomed rotation")
    # frame_count = _rotate_camera(
    #     view, zoom_focal, zoom_radius, initial_y,
    #     0, num_frames, frame_count, animation_dir, rank
    # )
    # ===== PHASE 3: Hold in place while zoomed in =====
    if rank == 0:
        logger.info("Phase 3: Holding zoomed view")
    hold_frames = 5
    for i in range(hold_frames):
        # Force complete render to ensure all data chunks are loaded
        force_complete_render(view, rank)
        if rank == 0:
            frame_num = frame_count + i
            frame_filename = os.path.join(animation_dir, f'frame_{frame_num:04d}.png')
            pv.SaveScreenshot(frame_filename)
            logger.info(f"Saved frame {frame_num + 1} (hold)")
            sys.stdout.flush()
    frame_count += hold_frames
    
    # ===== PHASE 4: Zoom out =====
    if rank == 0:
        logger.info("Phase 4: Zooming out")
    
    # Current position after zoomed rotation
    current_position = list(view.CameraPosition)
    
    frame_count = _smooth_transition(
        view, zoom_focal, initial_focal, zoom_scale, initial_scale,
        current_position, initial_position, transition_frames, frame_count, animation_dir, rank
    )
    
    # # ===== PHASE 5: Final full rotation =====
    # if rank == 0:
    #     logger.info("Phase 5: Final rotation")
    # frame_count = _rotate_camera(
    #     view, initial_focal, radius, initial_y,
    #     0, num_frames, frame_count, animation_dir, rank
    # )
    
    if rank == 0:
        logger.info(f"\nAnimation complete! Saved {frame_count} frames to {animation_dir}")
        sys.stdout.flush()
    return

def main():
    parser = argparse.ArgumentParser(description="Visualize compressed surface data with optimized color mapping.")
    parser.add_argument('--embryonic_number', type=int, required=True, help="Embryonic number (e.g., 317, 290).")
    parser.add_argument('--output_dir', type=str, required=True, help="Directory to save the output PNG file.")
    parser.add_argument('--surface_dir', type=str, required=True, help="Directory to load the surface data from.")
    parser.add_argument('--num_frames', type=int, default=25, help="Number of frames to render.")
    parser.add_argument('--output_tag', type=str, required=False, help="Tag to add to the output file name.")
    args = parser.parse_args()

    rank, _ = setup_mpi()
    embryonic_number = args.embryonic_number
    surface_dir = args.surface_dir
    output_tag = args.output_tag
    output_dir = args.output_dir
    collection_file = os.path.join(surface_dir, 'combined_surface.pvd')
    if output_tag is not None:
        tag_str = f'vtp_{embryonic_number}_{date.today()}{output_tag}'
    else:
        tag_str = f'vtp_{embryonic_number}_{date.today()}'
    output_file = os.path.join(args.output_dir, f'{tag_str}.png')
    animation_dir = os.path.join(args.output_dir, f'animation_frames_{tag_str}')

    GLASBEY_DARK_COLORS = filter_colors(all_hex_to_rgb(cc.glasbey_dark), min_saturation=0.5, min_lightness=0.2, max_lightness=0.8)
    GENERATE_ANIMATION = True

    # --- VALIDATE INPUT FILES ---
    if not os.path.exists(collection_file):
        if rank == 0:
            logger.info(f'Collection file not found: {collection_file}')
        sys.exit(1)

    # --- BEGIN VISUALIZATION PROCESS ---
    if rank == 0:
        logger.info(f"---------- Beginning Visualization of: E9.5_{args.embryonic_number} ----------")
        sys.stdout.flush()

    # --- DATA LOADING ---
    c2p, data_info = load_data(collection_file, rank, output_dir)
        
    segmentation_array_info = find_labels(c2p, rank)
    
    data_range = segmentation_array_info.GetComponentRange(0)
    min_label = int(data_range[0])
    max_label = int(data_range[1])
    
    if rank == 0:
        logger.info(f"SegmentationLabel range: {min_label} to {max_label}")
        logger.info(f"Total unique labels: {max_label - min_label + 1}")
        sys.stdout.flush()
    
    # --- VISUALIZATION SETUP ---
    display = setup_display(c2p, rank)
    view = setup_view(rank)
    apply_color_mapping(display, min_label, max_label, rank, GLASBEY_DARK_COLORS)
    setup_camera(view, data_info, rank)
    
    # --- SAVE VISUALIZATION STATE ---
    save_state_file(output_dir, tag_str, rank)
    save_colormap(output_dir, tag_str, min_label, max_label, rank, GLASBEY_DARK_COLORS)
    
    # --- SAVE DATA ---
    active_source = pv.GetActiveSource()
    pv.SaveData(f'{output_dir}/decimated_data.vtm',
                proxy=active_source,
                PointDataArrays=['Normals', 'SegmentationLabel'],
                Assembly='Hierarchy')
    
    # --- GENERATE MAIN SCREENSHOT ---
    save_screenshot(output_file, rank)

    # --- GENERATE ANIMATION ---
    gc.collect()
    if GENERATE_ANIMATION:
        generate_animation(view, animation_dir, args.num_frames, rank, data_info=data_info)

    return output_file

if __name__ == "__main__":
    try:
        start_time = time.time()
        result = main()
        end_time = time.time()
        time_taken = end_time - start_time
        num_minutes = time_taken / 60
        
        controller = servermanager.vtkProcessModule.GetProcessModule().GetGlobalController()
        rank = controller.GetLocalProcessId()
        
        if rank == 0:
            logger.info(f"Time taken: {num_minutes} minutes")
            if result:
                print("="*40)
                logger.info("Optimized Visualization Ended Successfully")
                print("="*40)
            else:
                print("="*40)
                logger.info("Optimized Visualization Failed")
                print("="*40)
            sys.stdout.flush()
            
        if not result:
            sys.exit(1)
            
    except Exception as e:
        controller = servermanager.vtkProcessModule.GetProcessModule().GetGlobalController()
        rank = controller.GetLocalProcessId()
        
        if rank == 0:
            logger.info(f"Error during optimized VTM visualization: {e}")
            import traceback
            traceback.print_exc()
            sys.stdout.flush()
        sys.exit(1)