import json
import os
from shutil import copyfile
import numpy as np
from shapely.geometry import Polygon


def parse_vinseg_json(
    json_path: str, image_id: str, annotation_id: int
) -> tuple[list, int, int, int]:
    """
    Parse VINSEG JSON file and convert annotations to COCO format.

    Args:
        json_path (str): Path to the VINSEG JSON file.
        image_id (str): ID of the image.
        annotation_id (int): Starting ID for annotations.

    Returns:
        tuple: A tuple containing a list of annotations in COCO format,
               image height, image width, and updated annotation ID.
    """
    # load json
    with open(json_path, "r") as f:
        vineseg_json = json.load(f)
    image_height = vineseg_json["imageHeight"]
    image_width = vineseg_json["imageWidth"]
    # annotations
    annos = []
    for annot in vineseg_json["shapes"]:
        polygon = Polygon(annot["points"])
        area = polygon.area
        # Get the coordinates of the simplified polygon
        soma = list(polygon.exterior.coords)
        coco_coord = []
        tmp_x = []
        tmp_y = []
        for x, y in soma:
            coco_coord.append(x)
            coco_coord.append(y)
            tmp_x.append(x)
            tmp_y.append(y)
        # Find the minimum and maximum x coordinates
        min_x, max_x = min(tmp_x), max(tmp_x)
        # Find the minimum and maximum y coordinates
        min_y, max_y = min(tmp_y), max(tmp_y)
        # Construct the bounding box (x, y, width, height)
        bounding_box = [min_x, min_y, max_x - min_x, max_y - min_y]
        anno_entry = {
            "id": annotation_id,
            "image_id": image_id,
            "category_id": 1,
            "segmentation": [list(coco_coord)],
            "area": area,
            "bbox": bounding_box,
            "iscrowded": 0,
        }
        annos.append(anno_entry)
        annotation_id += 1
    return (annos, image_height, image_width, annotation_id)


def generate_coco_dict(
    input_path: str,
    output_path: str,
    description: str,
    url: str,
    version: str,
    year: int,
    contributor: str,
    date_created: str,
    annotation_id_start: int,
    image_fileendings: list[str],
):
    """
    Generate COCO format dictionary from VINSEG JSON files.

    Args:
        input_path (str): Path to the directory containing VINSEG JSON files.
        output_path (str): Output directory for the generated COCO JSON file.
        description (str): Description of the dataset.
        url (str): URL associated with the dataset.
        version (str): Version of the dataset.
        year (int): Year the dataset was created.
        contributor (str): Contributor or creator of the dataset.
        date_created (str): Date when the dataset was created.
        annotation_id_start (int): Starting ID for annotations.
        image_fileendings (list[str]): List of valid image file extensions.

    Returns:
        None
    """
    coco_dict = {
        "info": {
            "description": description,
            "url": url,
            "version": version,
            "year": year,
            "contributor": contributor,
            "date_created": date_created,
        },
        "images": [],
        "annotations": [],
        "categories": [{"id": 1, "name": "cell", "supercategory": "soma"}],
    }
    annotation_id = annotation_id_start
    for file in os.listdir(input_path):
        if not np.any(
            [file.startswith(fileending) for fileending in image_fileendings]
        ):
            continue

        image_id, _ = os.path.splitext(file)
        annotations, image_height, image_width, annotation_id = parse_vinseg_json(
            f"{image_id}.json", image_id, annotation_id
        )
        coco_image_entry = {
            "id": image_id,
            "width": image_width,
            "height": image_height,
            "file_name": file,
        }

        coco_dict["images"].append(coco_image_entry)
        coco_dict["annotations"] += annotations
    with open(os.path.join(output_path, "coco.json"), "w") as j:
        json.dump(coco_dict, j)


def transform_segmentations(segmentations, width, height):
    res = []
    label = 0
    for segmentation in segmentations:
        res.append(
            [label]
            + [
                val / width if idx % 2 == 0 else val / height
                for idx, val in enumerate(segmentation)
            ]
        )
    return res


def split_indices(length, splits):
    """
    Split indices into segments based on specified splits.

    Args:
        length (int): Total number of indices.
        splits (list[float]): List of values representing the desired splits.

    Returns:
        list: List of arrays containing the split indices.
    """
    splits_adjusted = [int(length * split) for split in splits]
    indices = np.arange(length)
    np.random.shuffle(indices)
    split_points = np.cumsum(splits_adjusted)[:-1]
    split_lists = np.split(indices, split_points)
    return split_lists


def coco_seg_to_yolov8(coco_path: str, output_path: str, splits: list[float]):
    """
    Convert COCO format segmentation annotations to YOLOv8 format.

    Args:
        coco_path (str): Path to the directory containing COCO JSON file.
        output_path (str): Output directory for YOLOv8 formatted data.
        splits (list[float]): List of three values representing train, val, and test splits.

    Returns:
        None
    """
    assert (
        len(splits) == 3
    ), "Please provide three values for splits. If you don't need a validation or test set, just provide 0 as a value. Example: [1, 0, 0]"

    coco_dict_name = [file for file in os.listdir(coco_path) if ".json" in file][0]
    full_file_path = os.path.join(coco_path, coco_dict_name)

    with open(full_file_path, "r") as file:
        coco_dict = json.load(file)

    num_images = len(coco_dict["images"])

    train, val, test = split_indices(num_images, splits)

    # loop over images
    for idx, image in enumerate(coco_dict["images"]):
        id, width, height, file_name = (
            image["id"],
            image["width"],
            image["height"],
            image["file_name"],
        )
        # get annotations
        annotations = filter(lambda x: x["image_id"] == id, coco_dict["annotations"])
        # check why this is an array  of arrays
        # extract segmentation from annotation
        segmentations = [annotation["segmentation"][0] for annotation in annotations]

        # transform from pixel to percent
        segmentations_percent = transform_segmentations(segmentations, width, height)

        # get path for file depending on split
        slug = "train" if idx in train else "val" if idx in val else "test"
        final_path = os.path.join(output_path, slug)

        # write to disk
        file_path_label = os.path.join(final_path, "labels", f"{id}.txt")
        with open(file_path_label, "w") as file:
            for segmentation in segmentations_percent:
                file.write(" ".join([str(val) for val in segmentation]))
                file.write("\n")

        # copy image over to yolo folder
        file_path_image_src = os.path.join(coco_path, file_name)
        file_path_image_dest = os.path.join(final_path, "images", file_name)
        copyfile(file_path_image_src, file_path_image_dest)
