#!/usr/bin/env python3
"""
Steps 5-9: Holes and Hardware Processor (Process Guide Implementation)
=====================================================================

This script implements Steps 5-9 of the Process Guide:
5. Position Objective Holes in Connection Areas
6. Classify Holes Based on Proximity to Edges  
7. Add Reinforcement Holes (Singer)
8. Define Hardware for Each Hole
9. Structure Final JSON

Takes the output from ca_identification_only.py as input.
Follows the exact specifications from the Process Guide.
"""

import json
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

# Constants from Process Guide
DRILLING_TEMPLATES = [17.0, 20.0, 25.0, 30.0]  # Available template thicknesses
MAX_HOLE_DISTANCE = 200.0  # Maximum distance between holes before adding intermediate
SINGER_MIN_DISTANCE = 50.0  # Minimum distance from edges for singer_central holes

# Hardware types from Process Guide Step 8
HARDWARE_TYPES = {
    'dowel_P_with_glue': {'diameter': 8.0, 'depth_face': 10.0, 'depth_thickness': 20.0},
    'dowel_M_with_glue': {'diameter': 8.0, 'depth_face': 10.0, 'depth_thickness': 20.0},
    'dowel_G_with_glue': {'diameter': 8.0, 'depth_face': 10.0, 'depth_thickness': 20.0, 'depth_singer': 40.0},
    'glue': {'type': 'adhesive'}
}

@dataclass
class ConnectionArea:
    """Represents a connection area from Steps 1-4 output"""
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    connection_id: int
    fill: str = "black"
    opacity: float = 0.05

@dataclass
class Hole:
    """Represents a drilled hole according to Process Guide Step 9"""
    x: float
    y: float
    type: str  # flap_corner, flap_central, face_central, singer_flap, singer_central, singer_channel
    targetType: str  # "17", "20", "25", or "30" (drilling template)
    ferragemSymbols: List[str]  # Hardware symbols
    connectionId: Optional[int] = None  # None for singer holes

@dataclass
class Face:
    """Represents a piece face with connection areas and holes"""
    face_side: str
    connection_areas: List[ConnectionArea]
    holes: List[Hole]

@dataclass
class Piece:
    """Represents a furniture piece with all drilling information"""
    name: str
    bounds: Dict
    length: float
    height: float
    thickness: float
    quantity: int
    faces: List[Face]

def load_ca_identification_output(input_file: str) -> Tuple[Dict, float]:
    """
    Step 5: Load the output from Steps 1-4 (CA identification) and determine drilling template
    """
    print("Step 5: Loading Connection Area identification results...")
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"  Loaded CA data from: {input_file}")
        print(f"  Found {len(data['pieces'])} pieces with connection areas")
        
        # Determine drilling template from thickness face areas (from Steps 1-4)
        # We'll use a heuristic to select the template based on piece thicknesses
        drilling_template = determine_drilling_template_from_pieces(data['pieces'])
        print(f"  Determined drilling template: {drilling_template}mm")
        
        return data, drilling_template
    except FileNotFoundError:
        print(f"Error: Input file {input_file} not found")
        return None, None
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in {input_file}: {e}")
        return None, None

def determine_drilling_template_from_pieces(pieces: List[Dict]) -> float:
    """
    Determine the best drilling template based on Step 4 logic from Process Guide.
    
    From Process Guide Step 4:
    - Calculate total connected areas on thickness faces for each template
    - Choose template with highest total sum of connected areas on thickness faces  
    - In case of tie, use smallest value (17, 20, 25, 30 mm)
    """
    # For now, use a simplified approach based on most common thickness
    # This should ideally come from the Step 4 template analysis
    
    thicknesses = [piece['thickness'] for piece in pieces]
    
    # Count thickness frequencies to find most common
    thickness_count = {}
    for thickness in thicknesses:
        thickness_count[thickness] = thickness_count.get(thickness, 0) + 1
    
    # Find most common thickness
    most_common_thickness = max(thickness_count.items(), key=lambda x: x[1])[0]
    
    # Select template closest to most common thickness
    # Process Guide uses 17, 20, 25, 30mm templates
    for template in DRILLING_TEMPLATES:
        if template >= most_common_thickness:
            return template
    
    return DRILLING_TEMPLATES[-1]  # Default to largest template

def parse_pieces_from_ca_data(ca_data: Dict) -> List[Piece]:
    """
    Step 5: Parse the CA data into our internal Piece objects
    """
    print("  Parsing pieces and connection areas...")
    
    pieces = []
    for piece_data in ca_data['pieces']:
        # Parse connection areas for each face
        faces = []
        total_cas = 0
        
        for face_data in piece_data['faces']:
            connection_areas = []
            for ca_data in face_data['connectionAreas']:
                ca = ConnectionArea(
                    x_min=ca_data['x_min'],
                    x_max=ca_data['x_max'],
                    y_min=ca_data['y_min'],
                    y_max=ca_data['y_max'],
                    connection_id=ca_data['connectionId'],
                    fill=ca_data.get('fill', 'black'),
                    opacity=ca_data.get('opacity', 0.05)
                )
                connection_areas.append(ca)
                total_cas += 1
            
            face = Face(
                face_side=face_data['faceSide'],
                connection_areas=connection_areas,
                holes=[]  # Will be populated in Step 6
            )
            faces.append(face)
        
        piece = Piece(
            name=piece_data['name'],
            bounds=piece_data['bounds'],
            length=piece_data['length'],
            height=piece_data['height'],
            thickness=piece_data['thickness'],
            quantity=piece_data['quantity'],
            faces=faces
        )
        pieces.append(piece)
        
        if total_cas > 0:
            print(f"    {piece.name}: {total_cas} connection areas")
    
    return pieces

def calculate_hole_positions_in_ca(ca: ConnectionArea, piece_half_thickness: float, face_side: str, drilling_template: float) -> List[Tuple[float, float]]:
    """
    Step 5: Position objective holes in connection areas according to Process Guide
    
    From Process Guide Step 5:
    - Place initial holes at corners of connection areas: half_thickness from edges
    - ADJUSTED: Be more flexible with small connection areas
    - SPECIAL CASE: For full-piece CAs, use enhanced margin to avoid border appearance
    """
    ca_width = ca.x_max - ca.x_min
    ca_height = ca.y_max - ca.y_min
    
    # Check if this is a full-piece connection area (starts at 0.0)
    is_full_piece_ca = (ca.x_min == 0.0 and ca.y_min == 0.0)
    
    # Use piece's half_thickness as primary margin, but adapt for very small CAs
    # Add tolerance to avoid 0.1mm differences causing different logic
    tolerance = 0.5  # 0.5mm tolerance for manufacturing consistency
    
    if (ca_width >= (piece_half_thickness * 2 - tolerance) and 
        ca_height >= (piece_half_thickness * 2 - tolerance)):
        # Standard case: use piece's half_thickness margin
        margin = piece_half_thickness
        
        # SPECIAL: For full-piece CAs, use piece's actual half_thickness (Process Guide compliant)
        if is_full_piece_ca:
            margin = piece_half_thickness  # Use piece's actual half_thickness for full-piece CAs
            margin_reason = "full-piece-piece-thickness"
        else:
            margin_reason = "standard"
    else:
        # Small CA: use adaptive margin but not smaller than 5mm
        margin = min(piece_half_thickness, min(ca_width, ca_height) * 0.4)
        margin = max(margin, 5.0)
        margin_reason = "adaptive"
    
    
    holes = []
    
    # Calculate potential corner positions with proper margins
    corner_positions = [
        (ca.x_min + margin, ca.y_min + margin),  # Bottom-left
        (ca.x_max - margin, ca.y_min + margin),  # Bottom-right  
        (ca.x_min + margin, ca.y_max - margin),  # Top-left
        (ca.x_max - margin, ca.y_max - margin)   # Top-right
    ]
    
    # Filter corner positions that fit within CA bounds
    valid_corners = []
    for x, y in corner_positions:
        if (ca.x_min < x < ca.x_max and ca.y_min < y < ca.y_max):
            valid_corners.append((x, y))
    
    # Strategy based on CA size
    # Check if BOTH dimensions are very small
    if ca_width < margin * 2.5 and ca_height < margin * 2.5:
        # Very small CA: place 1 hole in center
        center_x = (ca.x_min + ca.x_max) / 2
        center_y = (ca.y_min + ca.y_max) / 2
        holes = [(center_x, center_y)]
        
    elif ca_width < margin * 4 and ca_height < margin * 4:
        # Small CA: place 2 holes
        if valid_corners and len(valid_corners) >= 2:
            # Place 2 holes diagonally opposite or along the longer dimension
            if ca_width > ca_height:
                # Place holes along width
                holes = [valid_corners[0], valid_corners[1]]  # Bottom corners
            else:
                # Place holes along height  
                holes = [valid_corners[0], valid_corners[2]]  # Left corners
        elif valid_corners:
            holes = [valid_corners[0]]
        else:
            # Fallback to center
            center_x = (ca.x_min + ca.x_max) / 2
            center_y = (ca.y_min + ca.y_max) / 2
            holes = [(center_x, center_y)]
        
        # Even small CAs can have intermediate holes if one dimension > 200mm
        should_add_intermediate = ca_width > 200 or ca_height > 200
        if should_add_intermediate:
            holes = add_intermediate_holes(holes, ca, margin)
            
    else:
        # Normal/Large CA: place 4 corner holes
        holes = valid_corners
        
        # Add intermediate holes only if distance > 200mm  
        should_add_intermediate = ca_width > 200 or ca_height > 200
        if should_add_intermediate:
            holes = add_intermediate_holes(holes, ca, margin)
    
    return holes

def add_intermediate_holes(initial_holes: List[Tuple[float, float]], 
                          ca: ConnectionArea, piece_half_thickness: float) -> List[Tuple[float, float]]:
    """
    Add intermediate holes when distance between holes > 200mm (Process Guide Step 5)
    More aggressive approach for better hole distribution.
    """
    if len(initial_holes) < 2:
        return initial_holes
    
    all_holes = initial_holes.copy()
    ca_width = ca.x_max - ca.x_min
    ca_height = ca.y_max - ca.y_min
    
    
    # For large connection areas, add center holes and edge midpoints
    
    # Process Guide Step 5: Add intermediate holes when distance > 200mm
    # First check if we need intermediate holes at all
    needs_intermediate = False
    
    # Check horizontal distances
    if len(initial_holes) >= 2:
        x_positions = sorted(set(hole[0] for hole in initial_holes))
        for i in range(len(x_positions) - 1):
            if x_positions[i+1] - x_positions[i] > MAX_HOLE_DISTANCE:
                needs_intermediate = True
                break
    
    # Check vertical distances  
    if len(initial_holes) >= 2:
        y_positions = sorted(set(hole[1] for hole in initial_holes))
        for i in range(len(y_positions) - 1):
            if y_positions[i+1] - y_positions[i] > MAX_HOLE_DISTANCE:
                needs_intermediate = True
                break
    
    # If no distance exceeds 200mm, return original holes
    if not needs_intermediate:
        return all_holes
    
    # Add intermediate holes systematically
    # Strategy 1: For wide areas (width > 200mm), add vertical lines of holes
    if ca_width > MAX_HOLE_DISTANCE:
        center_x = (ca.x_min + ca.x_max) / 2
        # Add center line of holes
        y_positions = sorted(set(hole[1] for hole in initial_holes))
        for y in y_positions:
            all_holes.append((center_x, y))
        
        # Add intermediate Y positions if needed
        for i in range(len(y_positions) - 1):
            if y_positions[i+1] - y_positions[i] > MAX_HOLE_DISTANCE:
                intermediate_y = (y_positions[i] + y_positions[i+1]) / 2
                all_holes.append((center_x, intermediate_y))
    
    # Strategy 2: For tall areas (height > 200mm), add horizontal lines of holes  
    if ca_height > MAX_HOLE_DISTANCE:
        center_y = (ca.y_min + ca.y_max) / 2
        # Add center line of holes
        x_positions = sorted(set(hole[0] for hole in initial_holes))
        for x in x_positions:
            all_holes.append((x, center_y))
        
        # Add intermediate X positions if needed
        for i in range(len(x_positions) - 1):
            if x_positions[i+1] - x_positions[i] > MAX_HOLE_DISTANCE:
                intermediate_x = (x_positions[i] + x_positions[i+1]) / 2
                all_holes.append((intermediate_x, center_y))
    
    # Strategy 3: For very large areas, add center hole
    if ca_width > MAX_HOLE_DISTANCE and ca_height > MAX_HOLE_DISTANCE:
        center_x = (ca.x_min + ca.x_max) / 2
        center_y = (ca.y_min + ca.y_max) / 2
        all_holes.append((center_x, center_y))
    
    # Remove duplicate holes (within 5mm tolerance)
    unique_holes = []
    tolerance = 5.0
    for hole in all_holes:
        is_duplicate = False
        for existing in unique_holes:
            if (abs(hole[0] - existing[0]) < tolerance and 
                abs(hole[1] - existing[1]) < tolerance):
                is_duplicate = True
                break
        if not is_duplicate:
            unique_holes.append(hole)
    
    
    return unique_holes

def classify_hole_by_proximity(x: float, y: float, face_side: str, piece: Piece, 
                              piece_half_thickness: float) -> str:
    """
    Step 6: Classify holes based on proximity to edges (Process Guide)
    
    Classifications:
    - flap_corner: At half_thickness from both edges of the face
    - flap_central: At half_thickness from one edge of the face  
    - face_central: In the middle of the face, not near edges
    - top_corner: In thickness (top, bottom, left or right), at corners
    - top_central: In thickness (top, bottom, left or right), centered between other holes
    """
    main_faces = {'main', 'other_main'}
    thickness_faces = {'top', 'bottom', 'left', 'right'}
    
    # Get face dimensions
    if face_side in main_faces:
        face_width = piece.length
        face_height = piece.height
    elif face_side in thickness_faces:
        if face_side in {'top', 'bottom'}:
            face_width = piece.length
            face_height = piece.thickness
        else:  # left, right
            face_width = piece.thickness  
            face_height = piece.height
    else:
        return "face_central"  # Fallback
    
    # Use tolerance for half_thickness comparison (Process Guide compliant)
    tolerance = 2.0  # 2mm tolerance for manufacturing consistency
    
    # Check proximity to edges using piece's half_thickness
    near_left = abs(x - piece_half_thickness) <= tolerance
    near_right = abs(x - (face_width - piece_half_thickness)) <= tolerance
    near_bottom = abs(y - piece_half_thickness) <= tolerance
    near_top = abs(y - (face_height - piece_half_thickness)) <= tolerance
    
    # For thickness faces (top, bottom, left, right), use different classification
    if face_side in thickness_faces:
        # Check if hole is at corners of thickness face
        at_corner = (near_left and near_bottom) or (near_left and near_top) or \
                   (near_right and near_bottom) or (near_right and near_top)
        
        if at_corner:
            return "top_corner"  # At corners of thickness face
        elif near_left or near_right or near_bottom or near_top:
            return "top_central"  # At edges of thickness face
        else:
            return "face_central"  # In middle of thickness face
    
    # For main faces (main, other_main), use standard classification
    else:
        # Classify based on proximity
        near_edges_count = sum([near_left, near_right, near_bottom, near_top])
        
        if near_edges_count >= 2:
            return "flap_corner"  # Near at least 2 edges (corner)
        elif near_edges_count == 1:
            return "flap_central"  # Near exactly 1 edge
        else:
            return "face_central"  # Not near any edges

def determine_connection_type(face_side1: str, face_side2: str) -> str:
    """
    Step 8: Determine connection type based on face types (Process Guide)
    """
    main_faces = {'main', 'other_main'}
    thickness_faces = {'top', 'bottom', 'left', 'right'}
    
    face1_is_main = face_side1 in main_faces
    face2_is_main = face_side2 in main_faces
    face1_is_thickness = face_side1 in thickness_faces
    face2_is_thickness = face_side2 in thickness_faces
    
    if face1_is_main and face2_is_main:
        return "face-face"  # Both main faces
    elif (face1_is_main and face2_is_thickness) or (face1_is_thickness and face2_is_main):
        return "face-top"  # One main, one thickness (face-top in guide)
    elif face1_is_thickness and face2_is_thickness:
        return "top-top"  # Both thickness faces (top-top in guide)
    else:
        return "mixed"      # Fallback

def assign_hardware_for_connection_type(connection_type: str, is_primary_piece: bool) -> List[str]:
    """
    Step 8: Assign hardware based on connection type (Process Guide)
    
    Hardware assignment:
    - dowel_P_with_glue: Face-face (one piece gets dowel, other gets glue)
    - dowel_M_with_glue: Face-top (face gets dowel, top gets glue)  
    - dowel_G_with_glue: Top-top (one piece gets dowel, other gets glue)
    - dowel_G_with_glue: Singer holes (dowel, no_limiter)
    """
    if connection_type == "face-face":
        return ["dowel_P_with_glue"] if is_primary_piece else ["glue"]
    elif connection_type == "face-top":
        return ["dowel_M_with_glue"] if is_primary_piece else ["glue"]
    elif connection_type == "top-top":
        return ["dowel_G_with_glue"] if is_primary_piece else ["glue"]
    else:
        return ["glue"]  # Fallback

def classify_singer_hole_type(x: float, y: float, face_side: str, piece: Piece, piece_half_thickness: float) -> str:
    """
    Classify singer hole type according to Process Guide specifications:
    
    - singer_flap: Half thickness from borders.
    - singer_central: In the middle of the face, at least 50 mm from edges.
    - singer_channel: At specific junction positions, at half_thickness from two parallel edges. Used more in slats.
    """
    main_faces = {'main', 'other_main'}
    thickness_faces = {'top', 'bottom', 'left', 'right'}
    
    # Get face dimensions
    if face_side in main_faces:
        face_width = piece.length
        face_height = piece.height
    elif face_side in thickness_faces:
        if face_side in {'top', 'bottom'}:
            face_width = piece.length
            face_height = piece.thickness
        else:  # left, right
            face_width = piece.thickness  
            face_height = piece.height
    else:
        return "singer_central"  # Fallback
    
    # Check distances from edges
    dist_from_left = x
    dist_from_right = face_width - x
    dist_from_bottom = y
    dist_from_top = face_height - y
    
    min_dist_from_edge = min(dist_from_left, dist_from_right, dist_from_bottom, dist_from_top)
    
    # Check if exactly half_thickness from borders (singer_flap)
    tolerance = 1.0  # 1mm tolerance for half_thickness positioning
    is_at_half_thickness = abs(min_dist_from_edge - piece_half_thickness) <= tolerance
    
    # Check if at least 50mm from all edges (singer_central)
    is_central = min_dist_from_edge >= SINGER_MIN_DISTANCE
    
    # Check if at half_thickness from two parallel edges (singer_channel)
    # This means either (left/right both at half_thickness) OR (top/bottom both at half_thickness)
    left_right_at_half = (abs(dist_from_left - piece_half_thickness) <= tolerance and 
                         abs(dist_from_right - piece_half_thickness) <= tolerance)
    top_bottom_at_half = (abs(dist_from_top - piece_half_thickness) <= tolerance and 
                         abs(dist_from_bottom - piece_half_thickness) <= tolerance)
    is_channel = left_right_at_half or top_bottom_at_half
    
    # Apply classification logic
    if is_channel:
        return "singer_channel"  # At half_thickness from two parallel edges
    elif is_central:
        return "singer_central"  # At least 50mm from all edges
    elif is_at_half_thickness:
        return "singer_flap"     # Half thickness from borders
    else:
        return "singer_flap"     # Default fallback

def add_singer_holes(piece: Piece, face: Face, connection_areas: List[ConnectionArea], 
                    drilling_template: str) -> None:
    """
    Step 7: Add reinforcement holes (Singer) according to Process Guide
    
    For connections involving main faces:
    - Add singer holes on the face opposite to the connected face
    - Mirror coordinates to the opposite face
    - Do not assign connectionId to singer holes
    """
    main_faces = {'main', 'other_main'}
    
    if face.face_side not in main_faces:
        return  # Only add singer holes for main faces
    
    # Determine opposite face
    opposite_face_side = 'other_main' if face.face_side == 'main' else 'main'
    
    # Find the opposite face
    opposite_face = None
    for f in piece.faces:
        if f.face_side == opposite_face_side:
            opposite_face = f
            break
    
    if not opposite_face:
        return
    
    # Add singer holes mirroring the connection area holes
    for ca in connection_areas:
        # Calculate singer hole positions using the same adaptive logic
        piece_half_thickness = piece.thickness / 2
        drilling_template_float = float(drilling_template)
        singer_positions = calculate_hole_positions_in_ca(ca, piece_half_thickness, opposite_face_side, drilling_template_float)
        
        for x, y in singer_positions:
            # Classify singer hole according to Process Guide definitions
            singer_type = classify_singer_hole_type(x, y, opposite_face_side, piece, piece_half_thickness)
            
            # Create singer hole
            singer_hole = Hole(
                x=x,
                y=y,
                type=singer_type,
                targetType=drilling_template,
                ferragemSymbols=["dowel_G_with_glue"],  # Singer holes use dowel_G
                connectionId=None  # No connectionId for singer holes
            )
            
            # Check for duplicate singer holes on this face
            is_duplicate = False
            tolerance = 5.0
            for existing_hole in opposite_face.holes:
                if (abs(singer_hole.x - existing_hole.x) < tolerance and 
                    abs(singer_hole.y - existing_hole.y) < tolerance):
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                opposite_face.holes.append(singer_hole)
            
        print(f"    Added {len(singer_positions)} singer holes to {piece.name}.{opposite_face_side}")

def calculate_hole_depth(piece_thickness: float, hardware_type: str) -> float:
    """
    Step 6: Calculate appropriate hole depth based on piece thickness and hardware
    """
    if hardware_type == "glue":
        return 0.0  # No holes needed for glue-only
    
    hardware_info = HARDWARE_TYPES.get(hardware_type, {})
    hardware_length = hardware_info.get('length', 20.0)
    
    # Hole depth should be roughly half the hardware length, but not exceed piece thickness
    target_depth = hardware_length / 2
    max_depth = piece_thickness - 2.0  # Leave 2mm minimum material
    
    return min(target_depth, max_depth) if max_depth > 0 else 0.0

def find_matching_connection_areas(pieces: List[Piece]) -> Dict[int, List[Tuple[Piece, Face, ConnectionArea]]]:
    """
    Step 7: Group connection areas by connectionId to find matching pairs
    """
    connection_groups = {}
    
    for piece in pieces:
        for face in piece.faces:
            for ca in face.connection_areas:
                conn_id = ca.connection_id
                if conn_id not in connection_groups:
                    connection_groups[conn_id] = []
                connection_groups[conn_id].append((piece, face, ca))
    
    return connection_groups

def process_holes_and_hardware(pieces: List[Piece], drilling_template: float) -> List[Piece]:
    """
    Steps 5-8: Process holes and hardware according to Process Guide
    """
    print("Step 5: Positioning objective holes in connection areas...")
    print("Step 6: Classifying holes based on proximity to edges...")
    print("Step 7: Adding reinforcement holes (Singer)...")
    print("Step 8: Defining hardware for each hole...")
    
    # Don't use global drilling template for half_thickness calculations
    # Each piece will use its own thickness / 2
    drilling_template_str = str(int(drilling_template))
    
    # Group connection areas by connectionId
    connection_groups = find_matching_connection_areas(pieces)
    
    # Step 5: Position objective holes in connection areas
    for conn_id, connections in connection_groups.items():
        if len(connections) != 2:
            print(f"  Warning: Connection {conn_id} has {len(connections)} pieces (expected 2)")
            continue
        
        piece1, face1, ca1 = connections[0]
        piece2, face2, ca2 = connections[1]
        
        # Determine connection type
        connection_type = determine_connection_type(face1.face_side, face2.face_side)
        
        print(f"  Connection {conn_id}: {piece1.name}.{face1.face_side} <-> {piece2.name}.{face2.face_side}")
        print(f"    Type: {connection_type}")
        print(f"    DEBUG: Primary piece selection - connection_type:{connection_type}")
        
        # Process each piece in the connection  
        # For face-face connections, prefer main face for dowels
        if connection_type == "face-face":
            # Check which piece has main face in this connection
            main_face_piece_idx = None
            for i, (piece, face, ca) in enumerate(connections):
                if face.face_side == "main":
                    main_face_piece_idx = i
                    break
            
            # If one piece has main face, it gets dowels; otherwise use order
            if main_face_piece_idx is not None:
                primary_piece_idx = main_face_piece_idx
                print(f"    DEBUG: Face-face connection - main face found at index {main_face_piece_idx}")
            else:
                primary_piece_idx = 0  # Fallback to first piece
                print(f"    DEBUG: Face-face connection - no main face found, using first piece")
        elif connection_type == "face-top":
            # For face-top connections: face gets dowel, top gets glue (Process Guide)
            face_piece_idx = None
            for i, (piece, face, ca) in enumerate(connections):
                if face.face_side in {'main', 'other_main'}:  # Face sides
                    face_piece_idx = i
                    break
            
            if face_piece_idx is not None:
                primary_piece_idx = face_piece_idx
                print(f"    DEBUG: Face-top connection - face found at index {face_piece_idx}, gets dowel")
            else:
                primary_piece_idx = 0  # Fallback
                print(f"    DEBUG: Face-top connection - no face found, using first piece")
        else:
            # For other connection types, use standard logic
            primary_piece_idx = 0
            print(f"    DEBUG: Other connection - using first piece as primary")
            
        for i, (piece, face, ca) in enumerate(connections):
            is_primary_piece = (i == primary_piece_idx)
            
            # Step 5: Calculate hole positions within connection area
            piece_half_thickness = piece.thickness / 2
            
            hole_positions = calculate_hole_positions_in_ca(ca, piece_half_thickness, face.face_side, drilling_template)
            
            # Step 6: Create holes with proper classification and hardware
            for x, y in hole_positions:
                # Classify hole based on proximity to edges
                hole_type = classify_hole_by_proximity(x, y, face.face_side, piece, piece_half_thickness)
                
                # Step 8: Assign hardware
                hardware_symbols = assign_hardware_for_connection_type(connection_type, is_primary_piece)
                print(f"      DEBUG HW: {piece.name}.{face.face_side} (idx:{i}, primary:{is_primary_piece}) -> {hardware_symbols}")
                
                # Create hole with Process Guide structure
                hole = Hole(
                    x=x,
                    y=y,
                    type=hole_type,
                    targetType=drilling_template_str,
                    ferragemSymbols=hardware_symbols,
                    connectionId=conn_id
                )
                # Check for duplicate holes on this face
                is_duplicate = False
                tolerance = 5.0
                for existing_hole in face.holes:
                    if (abs(hole.x - existing_hole.x) < tolerance and 
                        abs(hole.y - existing_hole.y) < tolerance):
                        is_duplicate = True
                        break
                
                if not is_duplicate:
                    face.holes.append(hole)
            
            print(f"    {piece.name}.{face.face_side}: {len(hole_positions)} holes")
        
        # Step 7: Add singer holes for main face connections (avoid duplicates)
        singer_added = set()  # Track pieces that already have singer holes
        for piece, face, ca in connections:
            if face.face_side in {'main', 'other_main'}:
                piece_face_key = f"{piece.name}_{face.face_side}"
                if piece_face_key not in singer_added:
                    add_singer_holes(piece, face, [ca], drilling_template_str)
                    singer_added.add(piece_face_key)
    
    return pieces

def generate_hardware_list(pieces: List[Piece]) -> Dict[str, int]:
    """
    Step 8: Generate comprehensive hardware list from ferragemSymbols
    """
    print("Step 8: Generating hardware lists...")
    
    hardware_count = {}
    
    for piece in pieces:
        for face in piece.faces:
            for hole in face.holes:
                for hardware_symbol in hole.ferragemSymbols:
                    if hardware_symbol != "glue":  # Don't count glue-only
                        hardware_count[hardware_symbol] = hardware_count.get(hardware_symbol, 0) + 1
    
    print("  Hardware inventory:")
    for hardware, count in hardware_count.items():
        hardware_info = HARDWARE_TYPES.get(hardware, {})
        print(f"    {hardware}: {count} pieces")
    
    return hardware_count

def create_final_output(pieces: List[Piece], hardware_list: Dict[str, int]) -> Dict:
    """
    Step 9: Structure Final JSON according to Process Guide
    """
    print("Step 9: Structuring final JSON...")
    
    # Main structure from Process Guide
    output = {
        "pieces": []
    }
    
    total_holes = 0
    total_cas = 0
    
    for piece in pieces:
        piece_holes = 0
        piece_cas = 0
        
        faces_data = []
        for face in piece.faces:
            # Convert holes to Process Guide format
            holes = []
            for hole in face.holes:
                holes.append({
                    "x": hole.x,
                    "y": hole.y,
                    "type": hole.type,
                    "targetType": hole.targetType,
                    "ferragemSymbols": hole.ferragemSymbols,
                    "connectionId": hole.connectionId
                })
                piece_holes += 1
                total_holes += 1
            
            # Convert connection areas to dict format  
            connection_areas = []
            for ca in face.connection_areas:
                connection_areas.append({
                    "x_min": ca.x_min,
                    "x_max": ca.x_max,
                    "y_min": ca.y_min,
                    "y_max": ca.y_max,
                    "fill": ca.fill,
                    "opacity": ca.opacity,
                    "connectionId": ca.connection_id
                })
                piece_cas += 1
                total_cas += 1
            
            faces_data.append({
                "faceSide": face.face_side,
                "holes": holes,
                "connectionAreas": connection_areas
            })
        
        # Process Guide piece structure
        piece_data = {
            "name": piece.name,
            "length": piece.length,
            "height": piece.height,
            "thickness": piece.thickness,
            "quantity": piece.quantity,
            "faces": faces_data
        }
        output["pieces"].append(piece_data)
        
        print(f"  {piece.name}: {piece_cas} CAs, {piece_holes} holes")
    
    print(f"  Summary: {total_cas} connection areas, {total_holes} holes")
    print(f"  Hardware list: {hardware_list}")
    return output

def process_steps_5_9(input_file: str = "output_new.json", 
                     output_file: str = "output_complete_chair_test.json"):
    """
    Main function - processes Steps 5-9 using CA identification output as input
    Follows the exact Process Guide specifications
    """
    print("=== Steps 5-9: Holes and Hardware Processing (Process Guide) ===")
    print("Adding holes and hardware to identified connection areas")
    print()
    
    # Step 5: Load CA identification results and determine drilling template
    ca_data, drilling_template = load_ca_identification_output(input_file)
    if ca_data is None or drilling_template is None:
        return
    
    # Parse pieces and connection areas
    pieces = parse_pieces_from_ca_data(ca_data)
    if not pieces:
        print("Error: No pieces found in input data")
        return
    
    # Steps 5-8: Process holes and hardware with drilling template
    pieces = process_holes_and_hardware(pieces, drilling_template)
    
    # Step 8: Generate hardware list
    hardware_list = generate_hardware_list(pieces)
    
    # Step 9: Create final output
    final_output = create_final_output(pieces, hardware_list)
    
    # Save output
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(final_output, f, indent=2)
    
    print()
    print("=== Steps 5-9 Completed (Process Guide) ===")
    print(f"Complete assembly data saved to: {output_file}")
    print(f"Drilling template used: {drilling_template}mm")
    print("Ready for manufacturing!")

if __name__ == "__main__":
    process_steps_5_9()
