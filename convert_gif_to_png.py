import os

from PIL import Image

puzzle_dir = "puzzle_data/digital_puzzles/100_pieces/"
save_dir = "puzzle_data_new/digital_puzzles/100_pieces"

# Get all puzzle folder names where the puzzle pieces are
puzzles = os.listdir(puzzle_dir)

for pz in puzzles:

    for f in os.listdir(os.path.join(puzzle_dir, pz)):

        # Get all file names with .gif ending
        if f.endswith(".gif"):
            gif = Image.open(os.path.join(puzzle_dir, pz, f))
            transparency = gif.info['transparency']

            # Create directory for the save file
            if not os.path.isdir(os.path.join(save_dir, pz)):
                os.makedirs(os.path.join(save_dir, pz))

            # Save gif file as png
            gif.save(os.path.join(save_dir, pz, f + ".png"), transparency=transparency)
