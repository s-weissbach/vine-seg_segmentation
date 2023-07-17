import os
from skimage import measure
import numpy as np

from shapely.geometry import Polygon
from .detectron2.engine import DefaultPredictor
from .detectron2.config import get_cfg
from PIL import Image
import os

def test():
    test_model = CellFinder("C:/Users/nruff/Documents/Vine-Seg_Stephan/vineseg/experiments/mask_rcnn_101/mask_rcnn_r101_config.yaml",
                            "C:/Users/nruff/Documents/Vine-Seg_Stephan/vineseg/experiments/mask_rcnn_101/mask_rcnn_R_101_FPN_3x.pth")

    test_model.return_vineseg(img=np.array(Image.open("C:/Users/nruff/Documents/Vine-Seg_Stephan/vineseg/data/AVG_190122_Spon_10-23-27_moco.png")),
                              min_size=1, max_size=1000, threshold=30)

def extract_coordinates_from_mask(mask:np.ndarray)->list:
    """Function to extract coordinates from a binary mask and calculate the area.

    Args:
        mask (np.ndarray): the mask as numpy array, where 1 represents a cell and
                           0 background. It is assumed that cells are uninterrupted.

    Returns:
        list: coordinates of the polygon that outlines the cell [[x1,y1],...].
    """
    # Assume that `roi_matrix` is the binary matrix representing the ROI
    # Find the contours of the binary matrix
    contours = measure.find_contours(mask, 0.5)
    # Assume that there is only one contour for the ROI
    roi_contour = contours[0]
    # Get the x and y coordinates of the contour
    x_coords = roi_contour[:, 1]
    y_coords = roi_contour[:, 0]
    # Create a list of (x, y) tuples for the polygon vertices
    polygon_vertices = [(x, y) for x, y in zip(x_coords, y_coords)]
    # Create a Shapely Polygon from the vertices
    polygon = Polygon(polygon_vertices)
    # Simplify the polygon using the Ramer-Douglas-Peucker algorithm
    tolerance = 0.0  
    simplified_polygon = polygon.simplify(tolerance)
    area = simplified_polygon.area
    # Get the coordinates of the simplified polygon
    simplified_polygon_coords = list(simplified_polygon.exterior.coords)
    return [simplified_polygon_coords, area]


class CellFinder:
    """
    Class to find cells using the Detectron2 model
    """
    def __init__(self,
                 config_file:str,
                 model_weights: str,
                 detection_threshold: float = 0.5,
                 cpu_mode: bool = False):
        """
        Args:
            config_file (str): path to config file for detectron2
            model_weights (str): path to the trained weights of model (.pth)
            detection_threshold (float, optional): Minimum score for predicting cells. Defaults to 0.5.
            cpu_mode (bool, optional): Whether to run on cpu (True) or gpu (False). Defaults to False.

        Raises:
            FileExistsError: Config File not found
            FileExistsError: Model weights not found
        """
        self.cfg = get_cfg()
        self.detection_threshold = detection_threshold
        if os.path.exists(config_file):
            self.cfg.merge_from_file(config_file)
        else:
            print(config_file)
            raise FileExistsError('Missing config file. Path provided {model_checkpoint_url}')
        if os.path.exists(model_weights):
            self.cfg.MODEL.WEIGHTS =  model_weights
        else:
            raise FileExistsError(f'Missing model weights. Path provided {model_weights}')
        self.cpu_mode = cpu_mode
        self.predictor = None

    def change_threshold(self, threshold: float) -> None:
        """Checks whether threshold is different from current threshold and
        changes if needed.

        Args:
            threshold (float): new threshold to use
        """
        if self.detection_threshold != threshold:
            self.detection_threshold = threshold
            self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold
            self.predictor = DefaultPredictor(self.cfg)
   
    def return_vineseg(self, img: np.ndarray, min_size: int, max_size: int, threshold: float) -> list:
        """_summary_

        Args:
            img (np.ndarray): input image
            min_size (int): minimum size of a neuron, otherwise marked as neuron_too_small
            max_size (int): maximum size of a neuron, otherwise marked as neuron_too_big
            threshold (float): threshold for the prediction score, if below score it will not be returned.

        Returns:
            list[dict]: returns a list with the predictions. Each prediction consists of a 
                        label (neuron, neuron_too_small, neuron_too_big), score that is the
                        prediction score, and points that are the coordinates of the mask [[x1,y1],...].
        """
        self.change_threshold(threshold)
        if self.predictor is None:
            self.predictor = DefaultPredictor(self.cfg)
        outputs = self.predictor(img)
        outputs = outputs["instances"].to("cpu").get_fields()
        # label: neuron, neuron_to_big, neuron_to_small
        vineseg_list = []
        for mask,score in zip(outputs['pred_masks'],outputs['scores']):
            m,a = extract_coordinates_from_mask(np.array(mask))
            if a < min_size:
                label = "neuron_too_small"
            elif a > max_size:
                label = "neuron_too_big"
            else:
                label = "neuron"
            vineseg_list.append({'label':label,
                                'score': float(score),
                                'points':m})
        return vineseg_list
