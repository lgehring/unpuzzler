import copy
import glob as gl
import os

import cv2
import numpy as np
from PIL import Image
from colormath import color_conversions, color_objects, color_diff

import edge_matching as em


def get_pieces(puzzle_dir, pieces_images=None, block_size=5, ksize=5, k=0.04, area_mod=0.018, new_are_mod=0.85,
               inward_offset=0):
    """
    Read and evaluate all puzzle pieces from the puzzle_dir
    :param pieces_images: An array of images to use as puzzle pieces instead of reading them from the directory
    :param inward_offset: The amount of pixels to offset the puzzle color vector inward
    :param new_are_mod: [0, 1] regulates the threshold for the proposed area relative the current best area in corner detection
    :param area_mod: [0, 1] regulates the areas relative influence on the score in corner detection
    :param k: Parameter for the corner detection algorithm
    :param ksize: Parameter for the corner detection algorithm
    :param block_size: Parameter for the corner detection algorithm
    :param puzzle_dir: a directory containing all puzzle pieces of a single puzzle
    :return: a list of lists of parameters of the puzzle pieces:
    [id, contours, corns, closest_corners, segments, col_segments, center, edge_types, piece_type]
    """
    raw_pieces = sorted(gl.glob(os.path.join(puzzle_dir, '*.png')), key=len)
    all_pieces = []
    corner_pieces = []
    edge_pieces = []
    inner_pieces = []

    pieces_images = pieces_images if pieces_images is not None else []
    if len(pieces_images) == 0:
        for i in range(len(raw_pieces)):
            piece_img = em.read_img(raw_pieces[i])
            pieces_images.append(piece_img)

    for i, piece_img in enumerate(pieces_images):
        contours, corns, closest_corners, segments, col_segments, \
        center, edge_types, piece_type = em.puzzle_characterization(piece_img, block_size=block_size, ksize=ksize, k=k,
                                                                    area_mod=area_mod, new_are_mod=new_are_mod,
                                                                    inward_offset=inward_offset)
        all_pieces.append([i, contours, corns, closest_corners, segments, col_segments, center, edge_types, piece_type])
        print('Piece {}: {}, {}'.format(i, edge_types, piece_type))
        if "r" in piece_type:
            corner_pieces.append(
                [i, contours, corns, closest_corners, segments, col_segments, center, edge_types, piece_type])
        elif "e" in piece_type:
            edge_pieces.append(
                [i, contours, corns, closest_corners, segments, col_segments, center, edge_types, piece_type])
        else:
            inner_pieces.append(
                [i, contours, corns, closest_corners, segments, col_segments, center, edge_types, piece_type])

    return all_pieces, corner_pieces, edge_pieces, inner_pieces


def compare_segments_contour(piece1, segment_id1, piece2, segment_id2):
    """
    Compare two segments based on contour (Hu invariants) and return the similarity distance
    :param piece1: a puzzle piece
    :param piece2: a puzzle piece
    :param segment_id1: a segment id
    :param segment_id2: a segment id
    :return: a similarity distance [0, inf) where smaller is better
    """
    segment_ind = 4
    contours1 = piece1[segment_ind][segment_id1]
    contours2 = piece2[segment_ind][segment_id2]
    return cv2.matchShapes(contours1, contours2, 2, 0.0)


# noinspection PyTypeChecker
def compare_segments_color(piece1, segment_id1, piece2, segment_id2):
    """
    Compare two segments based on color (Delta E (CIE2000)) and return the similarity distance
    :param piece1: a puzzle piece
    :param segment_id1: a segment id
    :param piece2: a puzzle piece
    :param segment_id2: a segment id
    :return: a similarity distance [0, 1] where smaller is better
    """

    def convert_to_lab(color_vec):
        return np.array([rgba_to_lab(xi) for xi in color_vec])

    def rgba_to_lab(rgba_arr):
        srgb = color_objects.sRGBColor(rgba_arr[0] / 255, rgba_arr[1] / 255, rgba_arr[2] / 255)
        return color_conversions.convert_color(srgb, color_objects.LabColor)

    # Get segment colors
    col_segment_ind = 5
    color_vec1 = piece1[col_segment_ind][segment_id1]
    color_vec2 = piece2[col_segment_ind][segment_id2]

    # Omit pixel coords and convert to image, then normalize to the longer vector
    max_vec_length = max(len(color_vec1), len(color_vec2))
    color_vec1 = [np.array([color_vec1[i][0] for i in range(len(color_vec1))])]
    color_vec2 = [np.array([color_vec2[i][0] for i in range(len(color_vec2))])]
    pil_img1 = Image.fromarray(np.array(color_vec1), 'RGBA')
    pil_img2 = Image.fromarray(np.array(color_vec2), 'RGBA')
    pil_img1 = pil_img1.resize((max_vec_length, 1))
    pil_img2 = pil_img2.resize((max_vec_length, 1))
    color_vec1 = convert_to_lab(np.array(pil_img1)[0])
    color_vec2 = convert_to_lab(np.array(pil_img2)[0])

    # Calculate the absolute color difference in both directions and return the minimum
    diff = 0
    for j in range(len(color_vec1)):
        diff += color_diff.delta_e_cie2000(color_vec1[j], color_vec2[-j])
    return diff / 100


def score_edge_against_edge(piece1, segment_id1, piece2, segment_id2, color_mod=3):
    """
    Score two segments based on the similarity of their shape and colour.
    Only pieces that can logically be matched are compared. Only sides of matching type are compared.
    :param piece1: a puzzle piece
    :param segment_id1: the segment id of the first piece to find a match for
    :param piece2: a puzzle piece
    :param segment_id2: the segment id of the second piece to find a match for
    :param color_mod: regulates the colors relative influence on the score [0, inf)
    :return: a vector of similarity distances [0, inf) where smaller is better
    """
    piece_type_ind = 8
    piece1_type = piece1[piece_type_ind]
    piece2_type = piece2[piece_type_ind]
    segment_type_ind = 7
    segment1 = piece1[segment_type_ind][segment_id1]
    segment2 = piece2[segment_type_ind][segment_id2]

    inner = ["i0", "i1", "i2", "i3", "i4", "i5"]
    corner = ["r0", "r1", "r2", "r3"]
    if not ((piece1_type in corner and piece2_type in inner) or (piece1_type in inner and piece2_type in corner) or (
            piece1_type in corner and piece2_type in corner)):
        # Pieces are compatible
        if (segment1 == "i" and segment2 == "o") or (segment1 == "o" and segment2 == "i"):
            # Segments are compatible
            color_score = compare_segments_color(piece1, segment_id1, piece2, segment_id2) * color_mod
            shape_score = compare_segments_contour(piece1, segment_id1, piece2, segment_id2)
            coefficient = color_score + shape_score
            print("Color: {}, Shape: {}, Coefficient: {}, P1:{}, S1:{}, P2:{}, S2:{}".format(
                color_score, shape_score, coefficient, piece1[0], segment_id1, piece2[0], segment_id2))
            return coefficient
    return np.inf


def score_edge_against_piece(piece1, segment_id1, piece2, assembling_frame, color_mod=3):
    """
    Score all combinations of a pair of puzzle pieces segments based on the similarity of their shape and colour.
    Only pieces that can logically be matched are compared. Only sides of matching type are compared.
    :param color_mod: regulates the colors relative influence on the score [0, inf)
    :param assembling_frame:
    :param piece1: a puzzle piece
    :param segment_id1: the segment id of the first piece to find a match for
    :param piece2: a puzzle piece
    :return: a vector of similarity distances [0, inf) where smaller is better
    """
    segment_type_ind = 7
    piece_type_ind = 8
    piece2_segment_types = piece2[segment_type_ind]
    similarity_vec = np.full((len(piece2_segment_types)), np.inf)

    for segment_id2 in range(len(piece2_segment_types)):
        similarity_vec[segment_id2] = score_edge_against_edge(piece1, segment_id1, piece2, segment_id2,
                                                              color_mod=color_mod)
        # Overwrite other edge pieces inner segments with infinity when assembling frame
        if assembling_frame and "e" in piece2[piece_type_ind]:
            # Check if segment is an inner segment
            edge_ind = np.where(np.chararray.find(piece2_segment_types, "e") == 0)[0]
            if ((edge_ind == 0 and segment_id2 == 2) or
                    (edge_ind == 1 and segment_id2 == 3) or
                    (edge_ind == 2 and segment_id2 == 0) or
                    (edge_ind == 3 and segment_id2 == 1)):
                # Is an inner segment
                similarity_vec[segment_id2] = np.inf
    return similarity_vec


def score_piece_against_neighbours(piece, loc, matrix, pieces, color_mod=3):
    """
    Score puzzle piece against all neighbours in a matrix.
    Only pieces that can logically be matched are compared. Only sides of matching type are compared.
    :param color_mod: regulates the colors relative influence on the score [0, inf)
    :param piece: a puzzle piece
    :param loc: the location of the piece in the matrix [row, col]
    :param matrix: holds the partly completed assembly
    :param pieces: a list of pieces to find a match from
    :return: a similarity distance [0, inf) where smaller is better,
    a list of pairs of the compared piece and segment id's, and a list of the own compared segment id's
    """
    score = 0
    number_of_scores = 0
    neighbors_and_segments = []
    own_segments = []
    for direction in range(4):
        # Get the neighbour piece
        neighbour_id = get_by_location(loc, direction, matrix)
        if neighbour_id != -1:  # Piece present in this direction
            neighbour = get_by_id(neighbour_id, pieces)
            # Identify the neighbor segment id to match against
            neighbour_edge_id = 2  # left
            if direction == 1:
                neighbour_edge_id = 3  # bottom
            elif direction == 2:
                neighbour_edge_id = 0  # right
            elif direction == 3:
                neighbour_edge_id = 1  # top
            # Add the score
            local_score = score_edge_against_edge(piece, direction, neighbour, neighbour_edge_id, color_mod=color_mod)
            if local_score != np.inf:
                # Mark neighbors and own segment as matched
                neighbors_and_segments.append([neighbour, neighbour_edge_id])
                own_segments.append(direction)
            score += local_score
            number_of_scores += 1
    return score / number_of_scores, neighbors_and_segments, own_segments


def find_match_for_segment(piece, segment, pieces, assembling_frame=False, color_mod=3):
    """
    Find the best match for a segment of a given piece from a list of puzzle pieces.
    :param color_mod: regulates the colors relative influence on the score [0, inf)
    :param assembling_frame:
    :param piece: a piece to find a match for
    :param segment: th segment index of the piece to find a match for
    :param pieces: a list of puzzle pieces
    :return: the index of the best piece and its segment and the similarity distance
    """
    id_ind = 0
    segment_type_ind = 7

    best_piece_id = -1
    best_piece_ind = -1
    best_segment = -1
    best_score = np.inf
    if piece[segment_type_ind][segment] == "i" or piece[segment_type_ind][segment] == "o":
        for ind, other_piece in enumerate(pieces):  # For all pieces
            if other_piece[0] != piece[0]:  # Don't compare to self
                similarity_vec = score_edge_against_piece(piece, segment, other_piece, assembling_frame,
                                                          color_mod=color_mod)
                local_best_score = np.min(similarity_vec)
                local_best_segment = np.where(similarity_vec == np.min(similarity_vec))[0][0]
                if local_best_score < best_score:
                    best_piece_id = other_piece[id_ind]
                    best_piece_ind = ind
                    best_segment = local_best_segment
                    best_score = local_best_score
    return best_piece_id, best_piece_ind, best_segment, best_score


def find_match_for_position(loc, matrix, placed_pieced, unplaced_pieces, color_mod=3):
    """
    Find the best match for a position of a given piece from a list of puzzle pieces.
    :param color_mod: [0, inf) regulates the colors relative influence on the score
    :param loc: the position in the matrix to find a match for
    :param matrix: the assembly matrix
    :param placed_pieced: a list of pieces that are already in the assembly
    :param unplaced_pieces: a list of pieces to find a match from
    :return: the index of the best piece and its position and the similarity distance
    """
    best_piece_id = -1
    best_score = np.inf
    best_neighbors_and_segments = []
    best_own_segments = []

    for piece in unplaced_pieces:
        score, neighbors_and_segments, own_segments = score_piece_against_neighbours(piece, loc, matrix, placed_pieced,
                                                                                     color_mod=color_mod)
        if score < best_score:
            best_piece_id = piece[0]
            best_score = score
            best_neighbors_and_segments = neighbors_and_segments
            best_own_segments = own_segments
    # Mark the matched segments as matched
    for neighbor, segment in best_neighbors_and_segments:
        neighbor[7][segment] = "c"
    for segment in best_own_segments:
        get_by_id(best_piece_id, unplaced_pieces)[7][segment] = "c"

    return best_piece_id


def convert_to_matrix(pieces, matches):
    """
    Convert a list of puzzle pieces and their matches to a matrix
    :param pieces: a list of puzzle pieces
    :param matches: a vector of puzzle piece segment vectors that each contain the matched piece id, and it's segment id
    :return: a matrix of the puzzle piece id's in the correct assembly layout
    """

    def get_new_pos(curr_pos, alignment):
        pos = curr_pos.copy()
        if alignment == 2:
            # Left
            pos[0] += 0  # y
            pos[1] += 1  # x
        if alignment == 3:
            # Bottom
            pos[0] += -1
            pos[1] += 0
        if alignment == 0:
            # Right
            pos[0] += 0
            pos[1] += -1
        if alignment == 1:
            # Top
            pos[0] += 1
            pos[1] += 0
        return pos

    def get_new_rot(segment_id_1, segment_id2):
        roll = 0
        if segment_id_1 == 0:
            if segment_id2 == 0:
                roll = 2  # rotate 180 degrees
            elif segment_id2 == 1:
                roll = 3  # rotate 270 degrees
            elif segment_id2 == 2:
                roll = 0
            elif segment_id2 == 3:
                roll = 1
        elif segment_id_1 == 1:
            if segment_id2 == 0:
                roll = 1
            elif segment_id2 == 1:
                roll = 2
            elif segment_id2 == 2:
                roll = 3
            elif segment_id2 == 3:
                roll = 0
        elif segment_id_1 == 2:
            if segment_id2 == 0:
                roll = 0
            elif segment_id2 == 1:
                roll = 1
            elif segment_id2 == 2:
                roll = 2
            elif segment_id2 == 3:
                roll = 3
        elif segment_id_1 == 3:
            if segment_id2 == 0:
                roll = 3
            elif segment_id2 == 1:
                roll = 0
            elif segment_id2 == 2:
                roll = 1
            elif segment_id2 == 3:
                roll = 2
        return roll

    rotations = []
    matrix = np.full((len(pieces) * 2, len(pieces) * 2), -1)
    old_state = []
    idle_counter = 0
    while pieces:
        # prevents infinite loop when no match is found for a piece
        current_state = pieces.copy()
        if current_state == old_state:
            idle_counter += 1
            if idle_counter > len(pieces):
                break
        else:
            idle_counter = 0

        for idp, piece in enumerate(pieces):
            # Place the first piece in the center for reference
            if piece[0] == 0 and np.where(matrix == piece[0])[0].size == 0:
                matrix[len(pieces), len(pieces)] = piece[0]
            # Get piece position if already assigned
            assembly_pos = np.where(matrix == piece[0])
            if assembly_pos[0].size > 0:
                assembly_pos = [assembly_pos[0][0], assembly_pos[1][0]]
            else:
                assembly_pos = None
            # Place all aligned pieces relative to the current piece if not already placed
            curr_matches = matches[piece[0]]
            if not (curr_matches == [[-1, -1], [-1, -1], [-1, -1], [-1, -1]]).all():
                for ids, segment in enumerate(curr_matches):
                    if not (segment == [-1, -1]).all():
                        # Actual match
                        other_piece_id = segment[0]
                        segment_alignment = segment[1]
                        # Check if the other piece has already been placed
                        other_piece_pos = np.where(matrix == other_piece_id)
                        if other_piece_pos[0].size == 0:
                            # Has not been placed yet, place it, if current piece is already placed
                            if assembly_pos is not None:
                                new_pos = get_new_pos(assembly_pos, ids)
                                new_rot = get_new_rot(ids, segment_alignment)
                                rotations.append([other_piece_id, new_rot])
                                rot_lookup = [0, 1, 2, 3, 0, 1, 2]
                                matches[np.where((matches[:, :, 0] == other_piece_id))] = [other_piece_id, rot_lookup[
                                    segment[1] + new_rot]]
                                matrix[new_pos[0], new_pos[1]] = other_piece_id
                                # Set match to -1 to prevent it from being placed again
                                matches[piece[0]][ids] = [-1, -1]
            if assembly_pos is not None:
                pieces.pop(idp)
        old_state = pieces.copy()
    # Remove all columns and rows that only contain -1
    cropped_assembly = matrix[:, np.where(~np.all(matrix == -1, axis=0))[0]]
    cropped_assembly = cropped_assembly[np.where(~np.all(cropped_assembly == -1, axis=1))[0], :]
    return cropped_assembly, rotations


def assemble_frame(corners, edges, matches, color_mod=3):
    """
    Assemble the puzzle frame from a list of puzzle pieces
    :param color_mod: regulates the colors relative influence on the score [0, inf)
    :param corners: a list of corner pieces
    :param edges: a list of edge pieces
    :param matches: an empty list of matches
    :return: a matrix containing the id's of pieces at their position in the assembly
    """
    id_ind = 0
    segment_type_ind = 7

    # Extend the corner pieces with edges
    print("CORNERS------------------------")
    while len(corners) > 0:
        curr_corner = corners.pop()
        print(curr_corner[id_ind])
        curr_segments = curr_corner[segment_type_ind]
        for segment_id, segment in enumerate(curr_segments):
            if segment == "i" or segment == "o":
                best_piece_id, best_piece_ind, best_segment, best_score = find_match_for_segment(curr_corner,
                                                                                                 segment_id, edges,
                                                                                                 True,
                                                                                                 color_mod=color_mod)
                # print("----------------------------------------------------" + str(best_piece_id) + " " + str(
                #     best_segment))
                # Mark both segments as connected if match was found
                if best_piece_id != -1:
                    curr_corner[segment_type_ind][segment_id] = "c"
                    edges[best_piece_ind][segment_type_ind][best_segment] = "c"
                    # Update the match in the matches list in both pieces
                    matches[curr_corner[id_ind]][segment_id] = [best_piece_id, best_segment]
                    matches[best_piece_id][best_segment] = [curr_corner[id_ind], segment_id]
    print("EDGES----------------------------------------------------")
    # Extend the edge pieces with edge pieces
    while len(edges) > 0:
        curr_edge = edges.pop(0)
        curr_segments = curr_edge[segment_type_ind]
        # Determine "inward" edges
        edge_ind = np.where(np.chararray.find(curr_segments, "e") == 0)[0]
        # Try to extend the edge with another edge
        for segment_id, segment in enumerate(curr_segments):
            if segment == "i" or segment == "o":
                # Ignore "inward" edges
                if not ((edge_ind == 0 and segment_id == 2) or
                        (edge_ind == 1 and segment_id == 3) or
                        (edge_ind == 2 and segment_id == 0) or
                        (edge_ind == 3 and segment_id == 1)):
                    try:
                        best_piece_id, best_piece_ind, best_segment, best_score = find_match_for_segment(curr_edge,
                                                                                                         segment_id,
                                                                                                         edges,
                                                                                                         True,
                                                                                                         color_mod=color_mod)
                    except Exception:
                        # print("Frame assembly error occured. Working with partial assembly now.")
                        return matches

                    # print("----------------------------------------------------" + str(best_piece_id) + " " + str(
                    #     best_segment))
                    # Mark both segments as connected if match was found
                    if best_piece_id != -1:
                        curr_edge[segment_type_ind][segment_id] = "c"
                        edges[best_piece_ind][segment_type_ind][best_segment] = "c"
                        # Update the match in the matches list in both pieces
                        matches[curr_edge[id_ind]][segment_id] = [best_piece_id, best_segment]
                        matches[best_piece_id][best_segment] = [curr_edge[id_ind], segment_id]
                    if "i" in curr_edge[segment_type_ind] or "o" in curr_edge[segment_type_ind]:
                        # Edge is not fully connected -> add back to list of edges
                        edges.append(curr_edge)
    return matches


def assemble_inner(frame_matrix, inners, edges, color_mod=3):
    """
    Assemble the puzzle inner from a list of puzzle pieces
    :param color_mod: regulates the colors relative influence on the score [0, inf)
    :param frame_matrix: a matrix of the puzzle frame
    :param inners: a list of inner pieces
    :param edges: a list of edge pieces
    :return: a matrix containing the id's of pieces at their position in the assembly
    """
    frame_dimensions = np.shape(frame_matrix)
    matrix = frame_matrix.copy()
    all_pieces = copy.deepcopy(edges) + copy.deepcopy(inners)
    # First find inner pieces that are connected to the frame, top left to bottom right
    for row in range(1, frame_dimensions[0] - 1):
        for col in range(1, frame_dimensions[1] - 1):
            if row == 1 or row == frame_dimensions[0] - 2 or col == 1 or col == frame_dimensions[1] - 2:
                # Find a match and mark the connections
                try:
                    match_id = find_match_for_position([row, col], matrix, all_pieces, inners, color_mod=color_mod)
                except Exception:
                    print("Inner assembly error occured. Working with partial assembly now.")
                    return matrix
                if match_id != -1:
                    matrix[row][col] = match_id
                    # Remove the match from the list of inners
                    inners.remove(get_by_id(match_id, inners))
    # Find inner pieces that are not connected to the frame, top left to bottom right
    for row in range(2, frame_dimensions[0] - 2):
        for col in range(2, frame_dimensions[1] - 2):
            # Find a match and mark the connections
            try:
                match_id = find_match_for_position([row, col], matrix, all_pieces, inners, color_mod=color_mod)
            except Exception:
                print("Inner assembly error occured. Working with partial assembly now.")
                return matrix
            if match_id != -1:
                matrix[row][col] = match_id
                # Remove the match from the list of inners
                inners.remove(get_by_id(match_id, inners))
    return matrix


def assemble_puzzle(puzzle_dir, pieces_images=None, only_frame=False, block_size=5, ksize=5, k=0.04, area_mod=0.018,
                    new_are_mod=0.85, inward_offset=0, color_mod=3):
    """
    Solve a puzzle by assembling the pieces
    :param color_mod: [0, inf) regulates the colors relative influence on the score
    :param pieces_images: An array of images to use as puzzle pieces instead of reading them from the directory
    :param inward_offset: The amount of pixels to offset the puzzle color vector inward
    :param new_are_mod: [0, 1] regulates the threshold for the proposed area relative the current best area in corner detection
    :param area_mod: [0, 1] regulates the areas relative influence on the score in corner detection
    :param k: Parameter for the corner detection algorithm
    :param ksize: Parameter for the corner detection algorithm
    :param block_size: Parameter for the corner detection algorithm
    :param puzzle_dir: a directory containing all puzzle pieces of a single puzzle
    :return: a matrix of the puzzle piece id's in the correct assembly layout
    :param only_frame: if True, only the frame is assembled, no inner pieces
    """
    print("Characterizing puzzle pieces...")
    all_pieces, corner_pieces, edge_pieces, inner_pieces = get_pieces(puzzle_dir, pieces_images=pieces_images,
                                                                      block_size=block_size, ksize=ksize, k=k,
                                                                      area_mod=area_mod, new_are_mod=new_are_mod,
                                                                      inward_offset=inward_offset)
    print("Assembling frame...")
    matches = np.full((len(all_pieces), 4, 2), -1)
    frame_matches = assemble_frame(corner_pieces, edge_pieces.copy(), matches, color_mod=color_mod)
    frame_matrix, rotations = convert_to_matrix(all_pieces, frame_matches)
    if not only_frame:
        print("Assembling inners...")
        solution_matrix = assemble_inner(frame_matrix, inner_pieces, edge_pieces, color_mod=color_mod)
        return solution_matrix, rotations
    return frame_matrix, rotations


def generate_solution_img(puzzle_dir, solution_matrix, rotations, pieces_images=None):
    """
    Visualize a puzzle solution by showing the pieces at the correct positions
    :param pieces_images: An array of images to use as puzzle pieces instead of reading them from the directory
    :param rotations: The puzzle pieces calculated rotations
    :param puzzle_dir: a directory containing all puzzle pieces of a single puzzle
    :param solution_matrix: a matrix of the puzzle piece id's in the correct assembly layout
    :return: a visualization img of the puzzle solution
    """
    raw_pieces = sorted(gl.glob(os.path.join(puzzle_dir, '*.png')), key=len)
    piece_images = pieces_images if pieces_images is not None else []
    piece_images_formatted = []

    if len(piece_images) == 0:
        for i in range(len(raw_pieces)):
            piece_img = em.read_img(raw_pieces[i])
            piece_images.append(piece_img)

    # Rotate the pieces to match the solution matrix
    for ind, rot in rotations:
        piece_images[ind] = np.rot90(piece_images[ind], rot)

    for piece_img in piece_images:
        # Resize image to 100x100
        piece_img = cv2.resize(piece_img, (100, 100))
        # Turn transparent background into white
        piece_img[piece_img[:, :, 3] == 0] = [255, 255, 255, 255]
        piece_img = piece_img[:, :, :3]
        piece_images_formatted.append(piece_img)
    # Create a blank white image
    img = np.ones((100 * np.shape(solution_matrix)[0], 100 * np.shape(solution_matrix)[1], 3), dtype=np.uint8) * 255
    # Draw the pieces
    for i in range(np.shape(solution_matrix)[0]):
        for j in range(np.shape(solution_matrix)[1]):
            if solution_matrix[i, j] != -1:
                curr_piece_img = piece_images_formatted[solution_matrix[i, j]]
                img[i * 100:(i + 1) * 100, j * 100:(j + 1) * 100] = curr_piece_img
                # Add piece id to image
                double_off = 0
                if solution_matrix[i, j] > 9:
                    double_off = 5
                cv2.putText(img, str(solution_matrix[i, j]), (j * 100 + 40 - double_off, i * 100 + 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    return img


def solve_puzzle(puzzle_dir, pieces_images=None, only_frame=False, block_size=5, ksize=5, k=0.04, area_mod=0.018,
                 new_are_mod=0.85, inward_offset=0, color_mod=3):
    """
    Solve a puzzle by assembling the pieces and showing the result
    :param color_mod: [0, inf) regulates the colors relative influence on the score
    :param pieces_images: An array of images to use as puzzle pieces instead of reading them from the directory
    :param inward_offset: The amount of pixels to offset the puzzle color vector inward
    :param new_are_mod: [0, 1] regulates the threshold for the proposed area relative the current best area in corner detection
    :param area_mod: [0, 1] regulates the areas relative influence on the score in corner detection
    :param k: Parameter for the corner detection algorithm
    :param ksize: Parameter for the corner detection algorithm
    :param block_size: Parameter for the corner detection algorithm
    :param puzzle_dir: a directory containing all puzzle pieces of a single puzzle
    :param only_frame: if True, only the frame is assembled, no inner pieces
    :return: a matrix of the puzzle piece id's in the correct assembly layout
    """
    solution_matrix, rotations = assemble_puzzle(puzzle_dir, pieces_images=pieces_images, only_frame=only_frame,
                                                 block_size=block_size, ksize=ksize, k=k, area_mod=area_mod,
                                                 new_are_mod=new_are_mod, inward_offset=inward_offset,
                                                 color_mod=color_mod)
    solution_img = generate_solution_img(puzzle_dir, solution_matrix, rotations, pieces_images=pieces_images)
    return solution_matrix, solution_img


def get_by_id(identifier, pieces):
    """
    Find a piece by its id
    :param identifier: the id of the piece to find
    :param pieces: the list of pieces to search in
    :return: the piece with the given id
    """
    return next((piece for piece in pieces if piece[0] == identifier), None)


def get_by_location(loc, direction, matrix):
    """
    Find a neighboring piece id by its relative location in the matrix
    :param loc: the location of the reference piece [row, col]
    :param direction: the direction of the neighbor-piece to find (0=left, 1= bottom, 2=right, 3=top)
    :param matrix: the matrix of pieces to search in
    :return: the piece at the given location
    """
    if direction == 0:
        return matrix[loc[0]][loc[1] - 1]
    elif direction == 1:
        return matrix[loc[0] + 1][loc[1]]
    elif direction == 2:
        return matrix[loc[0]][loc[1] + 1]
    elif direction == 3:
        return matrix[loc[0] - 1][loc[1]]


# ----------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    base_dir = "puzzle_data/digital_puzzles"
    puz_path = gl.glob(os.path.join(base_dir, '30_pieces/*'))

    for puz in puz_path:
        print("Solving: " + puz.lstrip(base_dir))
        sol_matrix, sol_img = solve_puzzle(puz, pieces_images=None, only_frame=False, block_size=5, ksize=5, k=0.04,
                                           area_mod=0.018, new_are_mod=0.85, inward_offset=0, color_mod=3)

        print(sol_matrix)
        # Show the image
        cv2.imshow('Solution', sol_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
