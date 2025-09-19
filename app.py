#!/usr/bin/env python3
"""
Generic CA Identification Script - Steps 1-4 Only
=================================================

This script implements ONLY the first 4 steps of the Process Guide GENERICALLY:
1. Input Data Preprocessing
2. Define Piece Dimensions and Orientation  
3. Map Pieces in 3D Space
4. Identify and Structure Connection Areas and Choose Drilling Template

WORKS FOR ANY FURNITURE MODEL: chair, table, cabinet, etc.
NO hardcoded assumptions - pure geometric analysis.
"""

import json
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Set

# Constants
MINIMUM_OVERLAP_MM = 10.0
DRILLING_TEMPLATES = [17.0, 20.0, 25.0, 30.0]  # Available template thicknesses

@dataclass
class Bounds3D:
    """3D bounding box for a piece"""
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    z_min: float
    z_max: float

@dataclass  
class Piece:
    """Represents a furniture piece with dimensions and faces"""
    name: str
    bounds: Bounds3D
    thickness: float
    height: float
    length: float
    faces: List[Dict]

def round_measurement(value: float) -> float:
    """
    Step 1: Round measurements according to Process Guide rules.
    - Round to first decimal place
    - If within ±0.1 of whole number, round to integer
    """
    rounded = round(value, 1)
    
    # Check if within ±0.1 of a whole number
    if abs(rounded - round(rounded)) <= 0.1:
        return float(round(rounded))
    
    return rounded

def detect_input_format(data: dict) -> str:
    """
    Step 1: Automatically detect input format.
    Returns: 'illustrator', 'direct_pieces', or 'unknown'
    """
    if 'layers' in data and isinstance(data['layers'], list):
        return 'illustrator'
    elif 'pieces' in data and isinstance(data['pieces'], list):
        return 'direct_pieces'
    else:
        return 'unknown'

def parse_illustrator_format_generic(data: dict) -> dict:
    """
    Step 1: Parse Illustrator format GENERICALLY - works with any layer/field names.
    """
    print("Step 1: Input Data Preprocessing...")
    print("  Parsing Illustrator format generically...")
    
    # Auto-detect layer names and field names
    layers = data.get('layers', [])
    if not layers:
        print("  Error: No layers found in input data")
        return {}
    
    # Detect layer names (could be in any language)
    layer_names = [layer['name'].lower() for layer in layers]
    print(f"  Detected layers: {layer_names}")
    
    # Map layers to view types based on common patterns
    view_mapping = {}
    for layer_name in layer_names:
        if any(keyword in layer_name for keyword in ['top', 'cima', 'above', 'superior']):
            view_mapping[layer_name] = 'top'
        elif any(keyword in layer_name for keyword in ['front', 'frontal', 'frente', 'frontale']):
            view_mapping[layer_name] = 'front'
        elif any(keyword in layer_name for keyword in ['side', 'lateral', 'lado', 'seite']):
            view_mapping[layer_name] = 'side'
        else:
            # Default mapping if we can't identify
            if len(view_mapping) == 0:
                view_mapping[layer_name] = 'top'
            elif len(view_mapping) == 1:
                view_mapping[layer_name] = 'front'
            else:
                view_mapping[layer_name] = 'side'
    
    print(f"  View mapping: {view_mapping}")
    
    # Parse pieces from all layers
    pieces_dict = {}
    for layer in layers:
        layer_name = layer['name'].lower()
        view_type = view_mapping.get(layer_name, 'unknown')
        
        for item in layer.get('items', []):
            # Auto-detect field names
            piece_name = item.get('nome', item.get('name', item.get('piece_name', 'unknown')))
            position = item.get('posicao', item.get('position', item.get('pos', {})))
            dimensions = item.get('dimensoes', item.get('dimensions', item.get('dims', {})))
            
            if piece_name not in pieces_dict:
                pieces_dict[piece_name] = {'top': {}, 'front': {}, 'side': {}}
            
            # Convert from cm to mm (multiply by 10) and apply rounding
            pieces_dict[piece_name][view_type] = {
                'x': round_measurement(position.get('x', 0) * 10),
                'y': round_measurement(position.get('y', 0) * 10),
                'width': round_measurement(dimensions.get('largura', dimensions.get('width', 0)) * 10),
                'height': round_measurement(dimensions.get('altura', dimensions.get('height', 0)) * 10)
            }
    
    print(f"  Parsed {len(pieces_dict)} pieces from Illustrator format")
    return pieces_dict

def convert_to_millimeters_generic(data: dict) -> dict:
    """
    Step 1: Convert all measurements to millimeters GENERICALLY.
    Handle any input format automatically.
    """
    format_type = detect_input_format(data)
    print(f"  Detected input format: {format_type}")
    
    if format_type == 'illustrator':
        pieces_dict = parse_illustrator_format_generic(data)
        return {"pieces_dict": pieces_dict}
    
    elif format_type == 'direct_pieces':
        # Direct pieces format (already processed)
        processed_data = {"pieces": []}
        
        for piece_data in data.get("pieces", []):
            processed_piece = {
                "name": piece_data["name"],
                "bounds": {}
            }
            
            # Convert and round all coordinate values
            bounds = piece_data.get("bounds", {})
            for key, value in bounds.items():
                # Convert cm to mm if needed (assuming input is already in mm for this example)
                mm_value = float(value)
                processed_piece["bounds"][key] = round_measurement(mm_value)
            
            processed_data["pieces"].append(processed_piece)
        
        print(f"  Processed {len(processed_data['pieces'])} pieces")
        return processed_data
    
    else:
        print(f"  Error: Unknown input format: {format_type}")
        return {}

def calculate_piece_dimensions(bounds: Bounds3D) -> Tuple[float, float, float]:
    """
    Step 2: Calculate fundamental dimensions (height, length, thickness).
    Returns: (height, length, thickness) where height >= length >= thickness
    """
    dim_x = bounds.x_max - bounds.x_min
    dim_y = bounds.y_max - bounds.y_min  
    dim_z = bounds.z_max - bounds.z_min
    
    dimensions = [dim_x, dim_y, dim_z]
    dimensions.sort(reverse=True)  # Sort descending
    
    height = dimensions[0]     # Largest dimension
    length = dimensions[1]     # Intermediate dimension  
    thickness = dimensions[2]  # Smallest dimension
    
    return height, length, thickness

def create_person_metaphor_faces(piece: Piece) -> List[Dict]:
    """
    Step 2: Create face coordinate systems using Person Metaphor.
    Each face has its own 2D coordinate system with origin at bottom-left.
    """
    faces = []
    
    # Main face: length × height (like person's front)
    faces.append({
        "faceSide": "main",
        "connectionAreas": []
    })
    
    # Other main face: length × height (like person's back)  
    faces.append({
        "faceSide": "other_main",
        "connectionAreas": []
    })
    
    # Top face: length × thickness (like person's head)
    faces.append({
        "faceSide": "top", 
        "connectionAreas": []
    })
    
    # Bottom face: length × thickness (like person's feet)
    faces.append({
        "faceSide": "bottom",
        "connectionAreas": []
    })
    
    # Left face: thickness × height (like person's left arm)
    faces.append({
        "faceSide": "left",
        "connectionAreas": []
    })
    
    # Right face: thickness × height (like person's right arm)
    faces.append({
        "faceSide": "right",
        "connectionAreas": []
    })
    
    return faces

def create_pieces_with_dimensions_generic(data: dict) -> List[Piece]:
    """
    Step 2: Create piece objects with proper dimensions and face systems GENERICALLY.
    """
    print("Step 2: Define Piece Dimensions and Orientation...")
    
    pieces = []
    
    # Handle Illustrator format
    if "pieces_dict" in data:
        pieces_dict = data["pieces_dict"]
        
        for piece_name, views in pieces_dict.items():
            top = views.get('top', {})
            front = views.get('front', {})
            side = views.get('side', {})
            
            if not (top and front and side):
                print(f"  Warning: {piece_name} missing views, skipping")
                continue
            
            # Calculate 3D dimensions from 3 views GENERICALLY
            # Each view shows 2 of the 3 dimensions:
            # - top view: z/x dimensions (width=x, height=z)
            # - front view: x/y dimensions (width=x, height=y)  
            # - side view: z/y dimensions (width=z, height=y)
            
            # Extract all available measurements
            top_width = top.get('width', 0)    # x dimension from top view
            top_height = top.get('height', 0)  # z dimension from top view
            front_width = front.get('width', 0)   # x dimension from front view
            front_height = front.get('height', 0) # y dimension from front view
            side_width = side.get('width', 0)     # z dimension from side view
            side_height = side.get('height', 0)   # y dimension from side view
            
            # Calculate each 3D dimension by taking the maximum from views that show it
            dim_x = max(top_width, front_width)   # x: from top view or front view
            dim_y = max(front_height, side_height) # y: from front view or side view
            dim_z = max(top_height, side_width)    # z: from top view or side view
            
            # Validation: ensure we have meaningful dimensions
            if dim_x <= 0 or dim_y <= 0 or dim_z <= 0:
                print(f"  Warning: {piece_name} has invalid dimensions (x={dim_x}, y={dim_y}, z={dim_z}), skipping")
                continue
            
            print(f"  {piece_name} dimensions: x={dim_x}mm, y={dim_y}mm, z={dim_z}mm")
            
            # Calculate fundamental dimensions (sorted)
            dimensions = sorted([dim_x, dim_y, dim_z], reverse=True)
            height, length, thickness = dimensions
            
            # Calculate 3D bounds GENERICALLY
            x_min = top.get('x', 0)
            y_max = front.get('y', 0)
            z_max = top.get('y', 0)
            
            bounds = Bounds3D(
                x_min=round_measurement(x_min),
                x_max=round_measurement(x_min + dim_x),
                y_min=round_measurement(y_max - dim_y),
                y_max=round_measurement(y_max),
                z_min=round_measurement(z_max - dim_z),
                z_max=round_measurement(z_max)
            )
            
            # Create piece object
            piece = Piece(
                name=piece_name,
                bounds=bounds,
                height=round_measurement(height),
                length=round_measurement(length), 
                thickness=round_measurement(thickness),
                faces=[]
            )
            
            # Create Person Metaphor face system
            piece.faces = create_person_metaphor_faces(piece)
            
            print(f"  {piece.name}: {height}x{length}x{thickness}mm (HxLxT)")
            pieces.append(piece)
    
    # Handle direct pieces format
    else:
        for piece_data in data.get("pieces", []):
            bounds_data = piece_data["bounds"]
            bounds = Bounds3D(
                x_min=bounds_data["x_min"],
                x_max=bounds_data["x_max"], 
                y_min=bounds_data["y_min"],
                y_max=bounds_data["y_max"],
                z_min=bounds_data["z_min"],
                z_max=bounds_data["z_max"]
            )
            
            # Calculate fundamental dimensions
            height, length, thickness = calculate_piece_dimensions(bounds)
            
            # Create piece object
            piece = Piece(
                name=piece_data["name"],
                bounds=bounds,
                height=height,
                length=length, 
                thickness=thickness,
                faces=[]
            )
            
            # Create Person Metaphor face system
            piece.faces = create_person_metaphor_faces(piece)
            
            print(f"  {piece.name}: {height}x{length}x{thickness}mm (HxLxT)")
            pieces.append(piece)
    
    return pieces

def check_3d_overlap_generic(piece1: Piece, piece2: Piece) -> Optional[Dict]:
    """
    Step 3: Check if two pieces overlap in 3D space GENERICALLY.
    Returns overlap information if found, None otherwise.
    """
    # Axis overlaps
    x_overlap = max(0, min(piece1.bounds.x_max, piece2.bounds.x_max) -
                       max(piece1.bounds.x_min, piece2.bounds.x_min))
    y_overlap = max(0, min(piece1.bounds.y_max, piece2.bounds.y_max) -
                       max(piece1.bounds.y_min, piece2.bounds.y_min))
    z_overlap = max(0, min(piece1.bounds.z_max, piece2.bounds.z_max) -
                       max(piece1.bounds.z_min, piece2.bounds.z_min))

    tolerance = 0.5  # floating point tolerance for adjacency

    # Axis adjacencies
    x_adjacent = (abs(piece1.bounds.x_max - piece2.bounds.x_min) <= tolerance or
                  abs(piece2.bounds.x_max - piece1.bounds.x_min) <= tolerance)
    y_adjacent = (abs(piece1.bounds.y_max - piece2.bounds.y_min) <= tolerance or
                  abs(piece2.bounds.y_max - piece1.bounds.y_min) <= tolerance)
    z_adjacent = (abs(piece1.bounds.z_max - piece2.bounds.z_min) <= tolerance or
                  abs(piece2.bounds.z_max - piece1.bounds.z_min) <= tolerance)

    # Additional validation: check if pieces are too far apart to be meaningful connections
    piece1_center = {
        'x': (piece1.bounds.x_min + piece1.bounds.x_max) / 2,
        'y': (piece1.bounds.y_min + piece1.bounds.y_max) / 2,
        'z': (piece1.bounds.z_min + piece1.bounds.z_max) / 2
    }
    piece2_center = {
        'x': (piece2.bounds.x_min + piece2.bounds.x_max) / 2,
        'y': (piece2.bounds.y_min + piece2.bounds.y_max) / 2,
        'z': (piece2.bounds.z_min + piece2.bounds.z_max) / 2
    }
    
    # Calculate distance between piece centers
    distance = math.sqrt(
        (piece1_center['x'] - piece2_center['x'])**2 +
        (piece1_center['y'] - piece2_center['y'])**2 +
        (piece1_center['z'] - piece2_center['z'])**2
    )
    
    # If pieces are too far apart, they shouldn't connect
    # Use a reasonable threshold based on piece sizes
    max_reasonable_distance = max(
        piece1.height + piece2.height,
        piece1.length + piece2.length,
        piece1.thickness + piece2.thickness
    ) * 1.5  # 1.5x the maximum possible connection distance
    
    if distance > max_reasonable_distance:
        return None

    connections = []

    # --- X axis ---
    if x_overlap >= MINIMUM_OVERLAP_MM:
        connections.append(('x', x_overlap))
    elif x_adjacent:
        if y_overlap >= MINIMUM_OVERLAP_MM and z_overlap >= MINIMUM_OVERLAP_MM:
            connections.append(('x', MINIMUM_OVERLAP_MM))

    # --- Y axis ---
    if y_overlap >= MINIMUM_OVERLAP_MM:
        connections.append(('y', y_overlap))
    elif y_adjacent:
        if x_overlap >= MINIMUM_OVERLAP_MM and z_overlap >= MINIMUM_OVERLAP_MM:
            connections.append(('y', MINIMUM_OVERLAP_MM))

    # --- Z axis ---
    if z_overlap >= MINIMUM_OVERLAP_MM:
        connections.append(('z', z_overlap))
    elif z_adjacent:
        if x_overlap >= MINIMUM_OVERLAP_MM and y_overlap >= MINIMUM_OVERLAP_MM:
            connections.append(('z', MINIMUM_OVERLAP_MM))

    if not connections:
        return None

    # Choose axis with minimal overlap → most likely connection axis
    connections.sort(key=lambda x: x[1])
    connection_axis, connection_overlap = connections[0]

    # Intersection bounds
    overlap_x_min = max(piece1.bounds.x_min, piece2.bounds.x_min)
    overlap_x_max = min(piece1.bounds.x_max, piece2.bounds.x_max)
    overlap_y_min = max(piece1.bounds.y_min, piece2.bounds.y_min)
    overlap_y_max = min(piece1.bounds.y_max, piece2.bounds.y_max)
    overlap_z_min = max(piece1.bounds.z_min, piece2.bounds.z_min)
    overlap_z_max = min(piece1.bounds.z_max, piece2.bounds.z_max)

    # Extend if adjacency was used (to create a small non-zero area)
    if connection_axis == 'x' and x_overlap < MINIMUM_OVERLAP_MM:
        mid_x = (overlap_x_min + overlap_x_max) / 2
        overlap_x_min = mid_x - MINIMUM_OVERLAP_MM / 2
        overlap_x_max = mid_x + MINIMUM_OVERLAP_MM / 2
    elif connection_axis == 'y' and y_overlap < MINIMUM_OVERLAP_MM:
        mid_y = (overlap_y_min + overlap_y_max) / 2
        overlap_y_min = mid_y - MINIMUM_OVERLAP_MM / 2
        overlap_y_max = mid_y + MINIMUM_OVERLAP_MM / 2
    elif connection_axis == 'z' and z_overlap < MINIMUM_OVERLAP_MM:
        mid_z = (overlap_z_min + overlap_z_max) / 2
        overlap_z_min = mid_z - MINIMUM_OVERLAP_MM / 2
        overlap_z_max = mid_z + MINIMUM_OVERLAP_MM / 2

    return {
        'piece1': piece1,
        'piece2': piece2,
        'connection_axis': connection_axis,
        'overlap_bounds': {
            'x_min': overlap_x_min, 'x_max': overlap_x_max,
            'y_min': overlap_y_min, 'y_max': overlap_y_max,
            'z_min': overlap_z_min, 'z_max': overlap_z_max
        },
        'overlap_amount': connection_overlap
    }

def is_piece_between(piece1: Piece, piece2: Piece, middle_piece: Piece, axis: str) -> bool:
    """
    Check if a middle piece is positioned between two other pieces on a given axis.
    """
    if axis == 'x':
        p1_min, p1_max = piece1.bounds.x_min, piece1.bounds.x_max
        p2_min, p2_max = piece2.bounds.x_min, piece2.bounds.x_max
        m_min, m_max = middle_piece.bounds.x_min, middle_piece.bounds.x_max
    elif axis == 'y':
        p1_min, p1_max = piece1.bounds.y_min, piece1.bounds.y_max
        p2_min, p2_max = piece2.bounds.y_min, piece2.bounds.y_max
        m_min, m_max = middle_piece.bounds.y_min, middle_piece.bounds.y_max
    elif axis == 'z':
        p1_min, p1_max = piece1.bounds.z_min, piece1.bounds.z_max
        p2_min, p2_max = piece2.bounds.z_min, piece2.bounds.z_max
        m_min, m_max = middle_piece.bounds.z_min, middle_piece.bounds.z_max
    else:
        return False
    
    # Check if middle piece is between the other two
    return ((p1_max <= m_min and m_max <= p2_min) or 
            (p2_max <= m_min and m_max <= p1_min))

def map_pieces_in_3d_space_generic(pieces: List[Piece]) -> List[Dict]:
    """
    Step 3: Map all pieces in 3D space and identify connections GENERICALLY.
    """
    print("Step 3: Map Pieces in 3D Space...")
    
    connections = []
    
    # Check all piece pairs for overlaps
    for i in range(len(pieces)):
        for j in range(i + 1, len(pieces)):
            piece1, piece2 = pieces[i], pieces[j]
            
            overlap = check_3d_overlap_generic(piece1, piece2)
            if overlap:
                # Check if any other piece is between these two pieces
                connection_axis = overlap['connection_axis']
                has_piece_between = False
                
                for k in range(len(pieces)):
                    if k != i and k != j:
                        middle_piece = pieces[k]
                        if is_piece_between(piece1, piece2, middle_piece, connection_axis):
                            has_piece_between = True
                            break
                
                if not has_piece_between:
                    connections.append(overlap)
                    print(f"  Found connection: {piece1.name} <-> {piece2.name} on {overlap['connection_axis']}-axis")
                else:
                    print(f"  Skipping {piece1.name} <-> {piece2.name}: piece between them on {connection_axis}-axis")
    
    print(f"  Total connections found: {len(connections)}")
    return connections

def determine_face_for_connection_axis_generic(piece: Piece, connection_axis: str, overlap_bounds: Dict, all_pieces: List[Piece] = None) -> str:
    """
    Step 4: Geometry-driven mapping from connection axis -> face name GENERICALLY.
    - Determines which axis is height/length/thickness for this piece.
    - Chooses the face plane (pair) that corresponds to the two axes orthogonal to connection_axis.
    - Uses pure geometric analysis, no hardcoded piece names.
    """
    # compute per-axis sizes
    dim_x = piece.bounds.x_max - piece.bounds.x_min
    dim_y = piece.bounds.y_max - piece.bounds.y_min
    dim_z = piece.bounds.z_max - piece.bounds.z_min
    sizes = {'x': dim_x, 'y': dim_y, 'z': dim_z}

    # rank axes: largest -> height, mid -> length, smallest -> thickness
    sorted_axes = sorted(sizes.items(), key=lambda kv: kv[1], reverse=True)
    height_axis = sorted_axes[0][0]
    length_axis = sorted_axes[1][0]
    thickness_axis = sorted_axes[2][0]

    # centers for comparison
    piece_centers = {
        'x': (piece.bounds.x_min + piece.bounds.x_max) / 2,
        'y': (piece.bounds.y_min + piece.bounds.y_max) / 2,
        'z': (piece.bounds.z_min + piece.bounds.z_max) / 2
    }
    overlap_centers = {
        'x': (overlap_bounds['x_min'] + overlap_bounds['x_max']) / 2,
        'y': (overlap_bounds['y_min'] + overlap_bounds['y_max']) / 2,
        'z': (overlap_bounds['z_min'] + overlap_bounds['z_max']) / 2
    }

    # the two axes that lie on the face (plane) are the ones != connection_axis
    plane_axes = [a for a in ('x', 'y', 'z') if a != connection_axis]

    # Determine which named face pair corresponds to that plane.
    # We choose the pair whose (axis role) set matches the plane axes:
    # - (thickness, height) -> left/right
    # - (length, height)    -> main (always prefer main over other_main)
    # - (length, thickness) -> top/bottom
    plane_set = set(plane_axes)
    if plane_set == set([thickness_axis, height_axis]):
        positive_face, negative_face = 'right', 'left'
    elif plane_set == set([length_axis, height_axis]):
        # For main faces, always use 'main' to consolidate CAs on same face
        return 'main'
    elif plane_set == set([length_axis, thickness_axis]):
        positive_face, negative_face = 'top', 'bottom'
    else:
        # fallback (very unusual geometry) — prefer main
        return 'main'

    # For thickness faces (left/right, top/bottom), determine which side based on geometry
    # Compare overlap center vs piece center along connection axis
    center_piece = piece_centers[connection_axis]
    center_overlap = overlap_centers[connection_axis]

    return positive_face if center_overlap >= center_piece else negative_face

def calculate_face_coordinates_for_ca_generic(piece: Piece, face_side: str, overlap_bounds: Dict) -> Tuple[float, float, float, float]:
    """
    Step 4: Project 3D overlap_bounds into a 2D coordinate box GENERICALLY.
    Uses the piece-specific axis roles (height/length/thickness) so mappings are always correct.
    Returns: (x_min, x_max, y_min, y_max) in the face's 2D coordinate system.
    """
    # compute per-axis sizes and identify roles
    dim_x = piece.bounds.x_max - piece.bounds.x_min
    dim_y = piece.bounds.y_max - piece.bounds.y_min
    dim_z = piece.bounds.z_max - piece.bounds.z_min
    sizes = {'x': dim_x, 'y': dim_y, 'z': dim_z}
    sorted_axes = sorted(sizes.items(), key=lambda kv: kv[1], reverse=True)
    height_axis = sorted_axes[0][0]
    length_axis = sorted_axes[1][0]
    thickness_axis = sorted_axes[2][0]

    # helper to get min/max for an axis from overlap_bounds and piece.bounds
    def o_min(a): return overlap_bounds[f"{a}_min"]
    def o_max(a): return overlap_bounds[f"{a}_max"]
    def b_min(a): return getattr(piece.bounds, f"{a}_min")
    
    # Map face_side -> (u_axis, v_axis), where u = horizontal on that face, v = vertical on that face
    if face_side in ['main', 'other_main']:
        # main faces = length × height
        u_axis, v_axis = length_axis, height_axis
    elif face_side in ['top', 'bottom']:
        # top/bottom = length × thickness
        u_axis, v_axis = length_axis, thickness_axis
    elif face_side in ['left', 'right']:
        # left/right = thickness × height
        u_axis, v_axis = thickness_axis, height_axis
    else:
        # fallback empty box
        return 0.0, 0.0, 0.0, 0.0

    # Project overlap bounds into the face's 2D coordinates
    u_min = o_min(u_axis) - b_min(u_axis)
    u_max = o_max(u_axis) - b_min(u_axis)
    v_min = o_min(v_axis) - b_min(v_axis)
    v_max = o_max(v_axis) - b_min(v_axis)
    
    # Get face dimensions for coordinate validation
    if face_side in ['main', 'other_main']:
        # main faces = length × height
        face_width = piece.length
        face_height = piece.height
    elif face_side in ['top', 'bottom']:
        # top/bottom = length × thickness
        face_width = piece.length
        face_height = piece.thickness
    elif face_side in ['left', 'right']:
        # left/right = thickness × height
        face_width = piece.thickness
        face_height = piece.height
    else:
        # fallback
        face_width = face_height = 0.0

    # Safety: ensure min <= max (swap if needed) and clamp extremely small negatives to 0
    if u_min > u_max:
        u_min, u_max = u_max, u_min
    if v_min > v_max:
        v_min, v_max = v_max, v_min

    # Clamp coordinates to valid face bounds (0 to face_dimension)
    u_min = max(0.0, min(u_min, face_width))
    u_max = max(0.0, min(u_max, face_width))
    v_min = max(0.0, min(v_min, face_height))
    v_max = max(0.0, min(v_max, face_height))

    # Ensure min <= max after clamping
    if u_min > u_max:
        u_min, u_max = u_max, u_min
    if v_min > v_max:
        v_min, v_max = v_max, v_min

    # Check for zero-width or zero-height connection areas
    ca_width = u_max - u_min
    ca_height = v_max - v_min
    
    # Filter out connection areas that are too small to be meaningful
    if ca_width < 0.1 or ca_height < 0.1:
        return 0.0, 0.0, 0.0, 0.0

    return u_min, u_max, v_min, v_max

def calculate_thickness_face_areas_generic(connections: List[Dict]) -> Dict[float, float]:
    """
    Step 4: Calculate total connected areas on thickness faces for each template GENERICALLY.
    """
    template_areas = {template: 0.0 for template in DRILLING_TEMPLATES}
    
    for connection in connections:
        for piece in [connection['piece1'], connection['piece2']]:
            face = determine_face_for_connection_axis_generic(
                piece, 
                connection['connection_axis'], 
                connection['overlap_bounds']
            )
            
            # Check if this is a thickness face connection
            if face in ['top', 'bottom', 'left', 'right']:
                # Calculate area of this connection
                bounds = connection['overlap_bounds']
                if connection['connection_axis'] == 'x':
                    area = (bounds['y_max'] - bounds['y_min']) * (bounds['z_max'] - bounds['z_min'])
                elif connection['connection_axis'] == 'y':
                    area = (bounds['x_max'] - bounds['x_min']) * (bounds['z_max'] - bounds['z_min'])
                else:  # z-axis
                    area = (bounds['x_max'] - bounds['x_min']) * (bounds['y_max'] - bounds['y_min'])
                
                # Add to all compatible templates (templates >= piece thickness)
                for template in DRILLING_TEMPLATES:
                    if template >= piece.thickness:
                        template_areas[template] += area
    
    return template_areas

def select_drilling_template(template_areas: Dict[float, float]) -> float:
    """
    Step 4: Select the best drilling template based on thickness face areas.
    """
    # Find template with highest total area
    max_area = max(template_areas.values())
    
    # In case of tie, choose smallest template thickness
    for template in sorted(DRILLING_TEMPLATES):
        if template_areas[template] == max_area:
            return template
    
    return DRILLING_TEMPLATES[0]  # Default fallback

def add_connection_area_to_piece_generic(piece: Piece, face_side: str, x_min: float, x_max: float, 
                                       y_min: float, y_max: float, connection_id: int):
    """
    Step 4: Add connection area to piece face in JSON structure GENERICALLY.
    Skips degenerate CAs (coordinates that are all zeros indicate filtered out CA).
    Adds 0.1mm margin to ensure CAs render properly.
    """
    # Check for degenerate CA (filtered out by calculate_face_coordinates_for_ca_generic)
    if x_min == 0.0 and x_max == 0.0 and y_min == 0.0 and y_max == 0.0:
        print(f"  Skipping degenerate CA on {piece.name}.{face_side} (too small to be meaningful)")
        return
    
    # Add 0.1mm margin to ensure CAs render properly
    margin = 0.1
    
    # Apply margin if coordinates are 0.0 or very close to 0.0 (within 0.001mm)
    tolerance = 0.001
    x_min_with_margin = x_min + margin if abs(x_min) <= tolerance else x_min
    y_min_with_margin = y_min + margin if abs(y_min) <= tolerance else y_min
    
    # Ensure we don't exceed the maximum bounds
    x_min_with_margin = min(x_min_with_margin, x_max - margin)
    y_min_with_margin = min(y_min_with_margin, y_max - margin)
        
    # Round the coordinates
    x_min_rounded = round_measurement(x_min_with_margin)
    y_min_rounded = round_measurement(y_min_with_margin)
    
    # Final safety check: if rounded coordinates are still 0.0, apply margin
    if x_min_rounded == 0.0:
        x_min_rounded = margin
    if y_min_rounded == 0.0:
        y_min_rounded = margin
    
    connection_area = {
        'x_min': x_min_rounded,
        'x_max': round_measurement(x_max),
        'y_min': y_min_rounded, 
        'y_max': round_measurement(y_max),
        'fill': 'black',
        'opacity': 0.05,
        'connectionId': connection_id
    }
    
    # Find the face and add the connection area
    for face in piece.faces:
        if face['faceSide'] == face_side:
            face['connectionAreas'].append(connection_area)
            return
    
    # Face not found - this shouldn't happen with proper Person Metaphor setup
    print(f"Warning: Face {face_side} not found on piece {piece.name}")

def identify_and_structure_connection_areas_generic(pieces: List[Piece], connections: List[Dict]) -> Tuple[float, int]:
    """
    Step 4: Identify connection areas, choose drilling template, and structure in JSON GENERICALLY.
    """
    print("Step 4: Identify and Structure Connection Areas and Choose Drilling Template...")
    
    # Calculate thickness face areas for template selection
    template_areas = calculate_thickness_face_areas_generic(connections)
    selected_template = select_drilling_template(template_areas)
    
    print(f"  Template analysis:")
    for template, area in template_areas.items():
        marker = " <- SELECTED" if template == selected_template else ""
        print(f"    {template}mm: {area:.1f}mm^2 total thickness face area{marker}")
    
    # Process each connection and create connection areas
    for connection_id, connection in enumerate(connections, 1):
        piece1 = connection['piece1']
        piece2 = connection['piece2']
        overlap_bounds = connection['overlap_bounds']
        connection_axis = connection['connection_axis']
        
        # Determine faces for each piece GENERICALLY
        piece1_face = determine_face_for_connection_axis_generic(piece1, connection_axis, overlap_bounds, pieces)
        piece2_face = determine_face_for_connection_axis_generic(piece2, connection_axis, overlap_bounds, pieces)
        
        # Calculate face coordinates for connection areas
        piece1_coords = calculate_face_coordinates_for_ca_generic(piece1, piece1_face, overlap_bounds)
        piece2_coords = calculate_face_coordinates_for_ca_generic(piece2, piece2_face, overlap_bounds)
        
        # Add connection areas to pieces
        add_connection_area_to_piece_generic(piece1, piece1_face, *piece1_coords, connection_id)
        add_connection_area_to_piece_generic(piece2, piece2_face, *piece2_coords, connection_id)
        
        print(f"  Connection {connection_id}: {piece1.name}.{piece1_face} <-> {piece2.name}.{piece2_face}")
    
    print(f"  Selected drilling template: {selected_template}mm")
    return selected_template, len(connections)

def create_final_json_output_generic(pieces: List[Piece], selected_template: float, num_connections: int) -> dict:
    """
    Create final JSON output with only connection areas (no holes) GENERICALLY.
    """
    output = {
        "pieces": []
    }
    
    for piece in pieces:
        piece_data = {
            "name": piece.name,
            "bounds": {
                "x_min": piece.bounds.x_min,
                "x_max": piece.bounds.x_max,
                "y_min": piece.bounds.y_min,
                "y_max": piece.bounds.y_max,
                "z_min": piece.bounds.z_min,
                "z_max": piece.bounds.z_max
            },
            "length": piece.length,
            "height": piece.height, 
            "thickness": piece.thickness,
            "quantity": 1,
            "faces": piece.faces
        }
        output["pieces"].append(piece_data)
    
    return output

def process_ca_identification_generic(input_file: str = "illustrator_positions box small.json", 
                                    output_file: str = "output box small.json"):
    """
    Main function - processes only Steps 1-4 GENERICALLY for any furniture model.
    """
    print("=== Generic CA Identification - Steps 1-4 ===")
    print("Works for ANY furniture model: chair, table, cabinet, etc.")
    print("Pure geometric analysis - no hardcoded assumptions")
    print()
    
    # Load input data
    try:
        # Try different encodings to handle potential encoding issues
        encodings = ['utf-8', 'cp1252', 'latin1', 'utf-8-sig']
        input_data = None
        
        for encoding in encodings:
            try:
                with open(input_file, 'r', encoding=encoding) as f:
                    input_data = json.load(f)
                print(f"Loaded input file: {input_file} (encoding: {encoding})")
                break
            except UnicodeDecodeError:
                continue
        
        if input_data is None:
            print(f"Error: Could not decode {input_file} with any encoding")
            return
            
    except FileNotFoundError:
        print(f"Error: Input file {input_file} not found")
        return
    
    # Step 1: Input Data Preprocessing
    processed_data = convert_to_millimeters_generic(input_data)
    
    if not processed_data:
        print("Error: Failed to process input data")
        return
    
    # Step 2: Define Piece Dimensions and Orientation
    pieces = create_pieces_with_dimensions_generic(processed_data)
    
    if not pieces:
        print("Error: No pieces could be created from input data")
        return
    
    # Step 3: Map Pieces in 3D Space  
    connections = map_pieces_in_3d_space_generic(pieces)
    
    # Step 4: Identify and Structure Connection Areas and Choose Drilling Template
    selected_template, num_connections = identify_and_structure_connection_areas_generic(pieces, connections)
    
    # Create final JSON output
    output_data = create_final_json_output_generic(pieces, selected_template, num_connections)
    
    # Save output
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2)
    
    print()
    print("=== Generic CA Identification Completed ===")
    print(f"Generated {len(pieces)} pieces with {num_connections} connections")
    print(f"Selected template: {selected_template}mm")
    print(f"Output saved to: {output_file}")

if __name__ == "__main__":
    process_ca_identification_generic()
