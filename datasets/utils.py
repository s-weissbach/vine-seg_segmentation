import json
import os


def transform_segmentations(segmentations, width, height):
    res = []
    # label to prepend
    # TODO: make this dynamic for cleaner code
    label = 0

    for segmentation in segmentations:
        res.append([label] + [val / width if idx % 2 == 0 else val / height for idx, val in enumerate(segmentation)])

    return res


# TODO: add a way to split files into folders (train-val-test split)
def coco_seg_to_yolov8(coco_dict_path, output_path):
    with open(coco_dict_path, "r") as file:
        coco_dict = json.load(file)

    # loop over images
    for image in coco_dict["images"]:
        id, width, height = image["id"], image["width"], image["height"]
        # get annotations
        annotations = filter(lambda x: x["image_id"] == id, coco_dict["annotations"])
        # check why this is an array  of arrays
        # extract segmentation from annotation
        segmentations = [annotation["segmentation"][0] for annotation in annotations]

        # transform from pixel to percent
        segmentations_percent = transform_segmentations(segmentations, width, height)

        # write to disk
        file_path = os.path.join(output_path, f"{id}.txt")
        with open(file_path, "w") as file:
            for segmentation in segmentations_percent:
                file.write(" ".join([str(val) for val in segmentation]))
                file.write("\n")
