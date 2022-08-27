import glob as gl
import io
import os
from base64 import b64encode

import cv2
import numpy as np
from PIL import Image
from PIL import ImageColor
from flask import Flask, render_template, request, redirect
from numpy import asarray

import edge_detection
import optimized_assembly as oa
from edge_matching import read_img, puzzle_characterization, plot_puzzle_segments

app = Flask(__name__)

original_img = None
uri = None
segmented_img = None
kmeans_result = {}
color_masking_result = {}
algorithm = None
transparent_mask = None
check_box_vals = []
mask_color = (0, 0, 0)
block_size = 5
puzzle_data_folder = "puzzle_data/digital_puzzles"
index = 0
k = 0.04
ksize = 5
puzzle_images = []
sol_img = None


def k_means_algorithm(filepath: str, k: int, attempts: int) -> str:
    """Calculates k-Means on image.

    :param filepath: Path to source image
    :param k:        k value
    :param attempts: Number of attempts
    :return:         Path to result image
    """
    # Convert image to numpy array
    src_image = Image.open(filepath)
    src_data = asarray(src_image)

    # Execute k-Means
    res_data = edge_detection.kmeans_masking(src_data, k, attempts)[0]
    res_image = Image.fromarray(res_data)

    # Store image
    splitted_filepath = filepath.split(".")
    res_filepath = f"{'.'.join(splitted_filepath[:-1])}_preview.{splitted_filepath[-1]}"
    res_image.save(res_filepath)

    return res_filepath


def color_masking_algorithm(filepath, range_):
    print(f"Calculate color masking on '{filepath}' here")
    return filepath


def np_to_uri(np_img):
    raw_bytes = io.BytesIO()
    img_rgb = cv2.cvtColor(np_img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img_rgb.astype("uint8"))
    img.save(raw_bytes, 'JPEG')
    raw_bytes.seek(0)
    uri = "data:image/JPEG;base64," + b64encode(raw_bytes.getvalue()).decode('ascii')
    return uri


def uri_to_np(uri):
    in_memory_file = io.BytesIO()
    uri.save(in_memory_file)
    np_img = cv2.imdecode(np.frombuffer(in_memory_file.getvalue(), np.uint8), cv2.IMREAD_COLOR)
    return np_img


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/upload")
def upload():
    return render_template("upload.html")


@app.route("/choose", methods=["GET", "POST"])
def choose():
    global index
    number_pieces = os.listdir(puzzle_data_folder)
    if request.method == "POST":
        index = int(request.form["pieces_form"])

    options = os.listdir("{}/{}".format(puzzle_data_folder, number_pieces[index]))
    return render_template("choose.html", number_pieces=number_pieces, options=options, index=index)


@app.route("/edge_detection", methods=["GET", "POST"])
def edge_detection_init():
    # # Upload image
    # if request.method == "POST":
    #
    #     # Set random session ID which will be used as file identifier
    #     session["id"] = str(uuid.uuid4())
    #     filepath = f"static/images/{session['id']}.png"
    #     file_ = request.files["file"]
    #     file_.save(filepath)
    #     image_url = filepath

    global original_img, uri

    if request.method == "POST":

        image = request.files['image_file']
        original_img = uri_to_np(image)
        uri = np_to_uri(original_img)

    else:
        uri = np_to_uri(original_img)

    return render_template("edge_detection.html", image=uri,
                           options={"algorithm": "k-means", "k": 4, "attempts": 10, "low": 64, "high": 192,
                                    "block_size": 9})


@app.route("/edge_detection/reset", methods=["GET", "POST"])
def edge_detection_reset():
    #
    # global original_img, uri
    #
    # uri = np_to_uri(original_img)

    return redirect("/edge_detection")


@app.route("/edge_detection/preview", methods=["GET", "POST"])
def edge_detection_preview():
    # if request.method == "POST":
    #
    #     # Request and response dictionaries
    #     req = json.loads(request.data.decode("utf-8"))
    #     res = {}
    #
    #     # Get filepath of uploaded image
    #     filepath = f"static/images/{req['id']}.png"
    #
    #     # Execute respective algorithm on image
    #     if req["algorithm"] == "k-means":
    #         res["previewUrl"] = k_means_algorithm(filepath, int(req["k"]), int(req["attempts"]))
    #
    #     elif req["algorithm"] == "color-masking":
    #         res["previewUrl"] = color_masking_algorithm(filepath, req["colorRange"])
    #
    #     else:
    #         raise Exception("Algorithm unknown")
    #
    #     # Construct response
    #     response = jsonify(res)
    #     response.headers.add("Access-Control-Allow-Origin", "*")
    #
    #     return response

    global original_img, uri, segmented_img, kmeans_result, algorithm, transparent_mask, block_size

    if request.method == "POST":
        forms = request.form

        if forms["algorithm"] == "k-means":
            algorithm = forms["algorithm"]
            k = int(forms["k"])
            attempts = int(forms["attempts"])
            segmented_img, ret, label, center = edge_detection.kmeans_masking(original_img, k, attempts)
            kmeans_result["img"] = segmented_img
            kmeans_result["label"] = label
            kmeans_result["center"] = center

            uri = np_to_uri(segmented_img)

        elif forms["algorithm"] == "color-masking":
            algorithm = forms["algorithm"]
            low = int(forms["color-low"])
            high = int(forms["color-high"])
            block_size = int(forms["block-size"])
            segmented_img, mask = edge_detection.color_masking(original_img, low, high)
            transparent_mask = edge_detection.color_transparent_masking(mask)

            uri = np_to_uri(segmented_img)

    else:
        uri = np_to_uri(original_img)

    return render_template("edge_detection.html", image=uri,
                           options={"algorithm": forms["algorithm"], "k": int(forms["k"]),
                                    "attempts": int(forms["attempts"]), "low": int(forms["color-low"]),
                                    "high": int(forms["color-high"]), "block_size": int(forms["block-size"])})


@app.route("/edge_detection/kmeans", methods=["POST", "GET"])
def edge_detection_kmeans():
    global kmeans_result, uri, mask_color, block_size

    if request.method == "POST":

        forms = request.form
        if forms["algorithm"] == "color-masking":
            return redirect("/edge_detection/results")

        temp = [False] * len(kmeans_result['center'])

    return render_template("edge_detection_kmeans.html", image=uri, options=kmeans_result["center"], check_vals=temp,
                           mask_color=mask_color[::-1], block_size=int(block_size))


@app.route("/edge_detection/kmeans/preview", methods=["POST", "GET"])
def edge_detection_kmeans_preview():
    global kmeans_result, uri, transparent_mask, check_box_vals, mask_color, block_size

    if request.method == "POST":

        checks = request.form.getlist('color-box')
        checks_int = []
        for ch in checks:
            checks_int.append(int(ch))

        k_img = kmeans_result["img"]
        label = kmeans_result["label"]
        center = kmeans_result["center"]
        mask_color = ImageColor.getcolor(request.form["maskColor"], "RGB")
        block_size = int(request.form["block-size"])

        check_box_vals = []
        for i in range(len(center)):
            check_box_vals.append(str(i) in checks)

        result, transparent_mask = edge_detection.black_masking(k_img, label, center, bg=checks_int,
                                                                mask_color=mask_color[::-1])

        uri = np_to_uri(result)

    return render_template("edge_detection_kmeans.html", image=uri, options=kmeans_result["center"],
                           check_vals=check_box_vals, mask_color='#%02x%02x%02x' % mask_color, block_size=block_size)


@app.route("/edge_detection/kmeans/reset", methods=["POST", "GET"])
def edge_detection_kmeans_reset():
    global segmented_img, uri, kmeans_result, mask_color, block_size

    if request.method == "POST":
        uri = np_to_uri(segmented_img)

        temp = [False] * len(kmeans_result['center'])

        mask_color = (0, 0, 0)

    return render_template("edge_detection_kmeans.html", image=uri, options=kmeans_result["center"], check_vals=temp,
                           mask_color='#%02x%02x%02x' % mask_color, block_size=block_size)


@app.route("/edge_detection/results", methods=["POST", "GET"])
def edge_detection_results():
    global transparent_mask, original_img, block_size, k, ksize, puzzle_images

    uri_list = []
    puzzle_images = []

    if request.method == "POST":
        if "pieces_form" in request.form:
            pieces_ind = int(request.form["pieces_form"])
            option = request.form["option_form"]

            number_pieces = os.listdir(puzzle_data_folder)

            puzzle_dir = "{}/{}/{}/".format(puzzle_data_folder, number_pieces[pieces_ind], option)
            pieces_paths = gl.glob(os.path.join(puzzle_dir, '*.png'))

            for pz in pieces_paths:
                img = read_img(pz)
                puzzle_images.append(img)

    else:
        contours, _ = cv2.findContours(transparent_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        mean = 0
        count = 0
        for contour in contours:
            count += 1
            mean += contour.shape[0]
        mean = mean / count

        for contour in contours:
            if contour.shape[0] < mean / 3:
                continue

            puzzle_img = edge_detection.extract_img_from_contour(original_img, contour, blocksize=block_size)
            puzzle_images.append(puzzle_img)

    for puzzle_img in puzzle_images:
        contours, corns, closest_corners, segments, col_segments, center, edge_types, piece_type = puzzle_characterization(
            puzzle_img, block_size)
        img = plot_puzzle_segments(puzzle_img, segments, closest_corners, edge_types, circle_size=5)
        uri_list.append(np_to_uri(img))

    return render_template("edge_detection_results.html", images=uri_list)


@app.route("/edge_matching", methods=["POST", "GET"])
def edge_matching():
    return render_template("edge_matching.html",
                           options={"block": 9, "k_size": 5, "k": 0.04, "area_mod": 0.018, "new_area_mod": 0.85,
                                    "inward_offset": 0, "color_mod": 2})


@app.route("/edge_matching/results", methods=["POST", "GET"])
def edge_matching_results():
    global puzzle_images, sol_img

    if request.method == "POST":
        print(request.form)
        only_frame = len(request.form.getlist('only-frame')) > 0
        block_size = int(request.form["block-size-input"])
        k_size = int(request.form["k-size-input"])
        k = float(request.form["k-input"])
        area_mod = float(request.form["area-mod-input"])
        new_area_mod = float(request.form["new-area-mod-input"])
        inward_offset = int(request.form["inward-offset-input"])
        color_mod = int(request.form["color-mod-input"])

        sol_matrix, sol_img = oa.solve_puzzle(puzzle_dir="", pieces_images=puzzle_images, only_frame=only_frame,
                                              block_size=block_size, ksize=k_size, k=k, area_mod=area_mod,
                                              new_are_mod=new_area_mod, inward_offset=inward_offset,
                                              color_mod=color_mod)

        sol_img_uri = np_to_uri(sol_img)

    return render_template("edge_matching_results.html", image=sol_img_uri)


if __name__ == "__main__":
    app.config["SECRET_KEY"] = "dev"
    app.run(debug=True)
