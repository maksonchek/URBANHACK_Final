from typing import List
from pathlib import Path
import yaml
import numpy as np
from PIL import Image
from ultralytics import YOLO
import cv2
from algorithms import DummyFacadeModel
import joblib
import numpy as np
import math
import os
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
import sklearn

def find_points_for_perspective(bboxes, image):
    def calculate_center(bbox):
        x_min, y_min, x_max, y_max = bbox
        center_x = (x_min + x_max) / 2
        center_y = (y_min + y_max) / 2
        return (center_x, center_y)

    window_centers = []

    bboxes = bboxes

    for bbox in bboxes:
            center = calculate_center(bbox)
            window_centers.append(center)

    image_height, image_width, _ = image.shape

    image_corners = [(0, 0), (0, image_height), (image_width, 0), (image_width, image_height)]

    def distance_from_point_to_ray(point, ray_start, ray_end):
        ray_vector = np.array(ray_end) - np.array(ray_start)
        point_vector = np.array(point) - np.array(ray_start)
        perpendicular_distance = np.linalg.norm(np.cross(ray_vector, point_vector)) / np.linalg.norm(ray_vector)
        return perpendicular_distance


    def point_position_relative_to_ray(ray_start, ray_end, point):
        ray_vector = (ray_end[0] - ray_start[0], ray_end[1] - ray_start[1])
        point_vector = (point[0] - ray_start[0], point[1] - ray_start[1])


        cross_product = np.cross(ray_vector, point_vector)

        if cross_product < 0:
            return "0"
        elif cross_product == 0:
            return "1"
        else:
            return "2"
    
    def find_corner(window_centers, ray_start, ray_end, image_width, image_height, mode = 'lower'):
        points_to_check = window_centers

        ray_start = ray_start
        ray_end = ray_end

        upper_points = set()
        lower_points = set()

        for point in points_to_check:
            position = point_position_relative_to_ray(ray_start, ray_end, point)
            if position == "0":
                lower_points.add(point)
            elif position == "1":
                upper_points.add(point)

        working_set = set()
        def fill_working_set(working_set, upper_points, lower_points):   
            if mode == 'lower':
                    working_set = lower_points.copy()
            elif mode == 'upper':
                    working_set = upper_points.copy()
            return working_set
        
        working_set = fill_working_set(working_set, upper_points, lower_points)
        nearest_point = (0,0)
        if working_set is not None:
            while working_set:
        
                if mode == 'lower':
                    nearest_point = min(lower_points, key=lambda x: distance_from_point_to_ray(x, ray_start, ray_end))
                elif mode == 'upper':
                    nearest_point = min(upper_points, key=lambda x: distance_from_point_to_ray(x, ray_start, ray_end))
                
                lower_points.clear()
                upper_points.clear()
                working_set.clear()

                ray_start = ray_start
                ray_end = nearest_point 
            
                for point in points_to_check:
                    position = point_position_relative_to_ray(ray_start, ray_end, point)
                    if point != nearest_point:
                        if position == "ниже луча":
                            lower_points.add(point)
                        elif position == "выше луча":
                            upper_points.add(point)
                    
                working_set = fill_working_set(working_set, upper_points, lower_points)
            return nearest_point
        else: 
            return "0"
        

    corn1_1 = find_corner(window_centers, (0, image_height/2), (image_width, image_height/2), image_width, image_height, mode='upper')
    corn1_2 = find_corner(window_centers, (0, image_height/2), (image_width, image_height/2), image_width, image_height, mode='lower')

    corn2_1 = find_corner(window_centers, (image_width, image_height/2), (0, image_height/2), image_width, image_height, mode='upper')
    corn2_2 = find_corner(window_centers, (image_width, image_height/2), (0, image_height/2), image_width, image_height, mode='lower')

    corn3_1 = find_corner(window_centers, (image_width/2, 0), (image_width/2, image_height/2), image_width, image_height, mode='upper')
    corn3_2 = find_corner(window_centers, (image_width/2, 0), (image_width/2, image_height/2), image_width, image_height, mode='lower')

    corn4_1 = find_corner(window_centers, (image_width/2, image_height), (image_width/2, 0), image_width, image_height, mode='upper')
    corn4_2 = find_corner(window_centers, (image_width/2, image_height), (image_width/2, 0), image_width, image_height, mode='lower')

    corners = []
    if corn1_1 != "0" and corn1_2 != "0" and corn2_1 != "0" and corn2_2 != "0" and corn3_1 != "0" and corn3_2 != "0" and corn4_1 != "0" and corn4_2 != "0":  
        corners.append(corn1_1)
        corners.append(corn1_2)
        corners.append(corn2_1)
        corners.append(corn2_2)
        corners.append(corn3_1)
        corners.append(corn3_2)
        corners.append(corn4_1)
        corners.append(corn4_2)

        element_count = {}

        for element in corners:
            if element in element_count:
                element_count[element] += 1
            else:
                element_count[element] = 1

        unique_elements = [element for element, count in element_count.items() if count == 1]


        new_corners = [i for i in corners if i not in unique_elements]

        def calculate_distance(point1, point2):
            x1, y1 = point1
            x2, y2 = point2
            return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

        middle_points = {}
        for point in unique_elements:
            closest_point = None
            closest_distance = 1000000000000000 


            for other_point in unique_elements:
                if point != other_point: 
                    distance = calculate_distance(point, other_point)
                    if distance < closest_distance:
                        closest_distance = distance
                        closest_point = other_point
            middle_point = ((closest_point[0] + point[0])/2, (closest_point[1] + point[1])/2)
            middle_points[middle_point] = closest_distance/2

        for point in middle_points:
            closest_point = None
            closest_distance = 10000000000000

            for other_point in image_corners:
                if point != other_point:
                    distance = calculate_distance(point, other_point)
                    if distance < closest_distance:
                        closest_distance = distance
                        closest_point = other_point


            displacement_vector = ((closest_point[0] - point[0]) * middle_points[point] / closest_distance,
                                (closest_point[1] - point[1]) * middle_points[point] / closest_distance)


            adjusted_point = (point[0] + displacement_vector[0], point[1] + displacement_vector[1])
            new_corners.append(adjusted_point)
        end_points = set()

        def calculate_diagonal(bbox):
            xmin, ymin, xmax, ymax = bbox
            return ((xmax - xmin) ** 2 + (ymax - ymin) ** 2) ** 0.5

        def distance_to_nearest_bbox(point, bboxes):
            closest_bbox = None
            closest_distance = float('inf')

            for bbox in bboxes:
                xmin, ymin, xmax, ymax = bbox
                bbox_center_x = (xmin + xmax) / 2
                bbox_center_y = (ymin + ymax) / 2
                distance = ((bbox_center_x - point[0]) ** 2 + (bbox_center_y - point[1]) ** 2) ** 0.5
                if distance < closest_distance:
                    closest_distance = distance
                    closest_bbox = bbox

            diagonal_length = calculate_diagonal(closest_bbox)

            return diagonal_length

        diagonals_dict = {}

        for point in new_corners:
            diagonal_length = distance_to_nearest_bbox(point, bboxes)
            diagonals_dict[point] = diagonal_length

        for point in new_corners:
            closest_point = None
            closest_distance = 100000000

            for other_point in image_corners:
                if point != other_point:
                    distance = calculate_distance(point, other_point)
                    if distance < closest_distance:
                        closest_distance = distance
                        closest_point = other_point


            displacement_vector = ((closest_point[0] - point[0]) * (diagonals_dict[point]/4) / closest_distance,
                                (closest_point[1] - point[1]) * (diagonals_dict[point]/4) / closest_distance)


            end_point = (point[0] + displacement_vector[0], point[1] + displacement_vector[1])
            end_points.add(end_point)

        def sum_of_values(pair):
            return sum(pair)

        end_points = sorted(end_points, key=sum_of_values)

        end_points = list(end_points)

        x_coords = [point[0] for point in end_points]
        y_coords = [point[1] for point in end_points]

        min_x = min(x_coords)
        max_x = max(x_coords)
        min_y = min(y_coords)
        max_y = max(y_coords)

        width = max_x - min_x
        length = max_y - min_y

        if length > width:
            temp = image_corners[1]

            image_corners[1] = image_corners[2]

            image_corners[2] = temp
            
        elif length < width:
            pass
        else:
            pass

        converted_points =  []
        for i in image_corners:
            converted_points.append(list(i))

        input_points = []
        for i in end_points:
            input_points.append(list(i))

        input_points = np.array(input_points, dtype=np.float32)
        if len(input_points) > 4:
            input_points = input_points[:4]
        elif len(input_points) < 4:
            return (None, None)
            
        converted_points = np.array(converted_points, dtype=np.float32)

        return (input_points, converted_points)
    else:
        return (None, None)


model = YOLO('./weights/last6.pt')



def _glob_images(folder: Path, exts: List[str] = ('*.jpg', '*.png',)) -> List[Path]:
    images = []
    for ext in exts:
        images += list(folder.glob(ext))
    return images


def predict_and_transform(input_folder: str, output_file: str) -> None:
    input_folder = Path(input_folder)
    output_file = Path(output_file)
    images_path = _glob_images(input_folder)
    output_dict = {}
    for img_path in images_path:
        image = cv2.imread(img_path)
        image_height, image_width, _ = image.shape
        results = model.predict(source = img_path, device='cpu')
        bboxes = []
        for result in results:
            boxes = result.boxes  
            
            for box in boxes:
                barr = np.array(box.xyxy.tolist())
                barr = barr.astype(int)
                xmin, ymin, xmax, ymax = map(int, barr[0][:])
                bb = []
                bb.append(xmin)
                bb.append(ymin)
                bb.append(xmax)
                bb.append(ymax)

                bboxes.append(bb)

        bboxes = np.array(bboxes)
        
        # try:
        input_points, converted_points = find_points_for_perspective(bboxes, image)
        if input_points is not None and input_points.any():
            loaded_model = joblib.load("./weights/best_tree_model.joblib")
            loaded_scaler = joblib.load("./weights/scaler_model.joblib")
            data_row = [f"{point[0]}" for i, point in enumerate(input_points)] + [f"{point[1]}" for i, point in enumerate(input_points)]
            data_row = [np.float32(i) for i in data_row]
            data_for_prediction = loaded_scaler.transform(np.array(data_row).reshape(1, -1))
            y_pred = loaded_model.predict(data_for_prediction)
            if int(y_pred[0]) == 1:
                matrix = cv2.getPerspectiveTransform(input_points, np.float32(converted_points))
                inverse_matrix = cv2.getPerspectiveTransform(converted_points, input_points)
                image_outp = cv2.warpPerspective(image, matrix, (image_width, (image_height)))
                warp_results = model.predict(source = image_outp, device='cpu')
                warped_bboxes = []
                for result in warp_results:
                    boxes = result.boxes  
                    
                    for box in boxes:
                        barr = np.array(box.xyxy.tolist())
                        barr = barr.astype(int)
                        xmin, ymin, xmax, ymax = map(int, barr[0][:])
                        bb = []
                        bb.append(xmin)
                        bb.append(ymin)
                        bb.append(xmax)
                        bb.append(ymax)
                        warped_bboxes.append(bb)
                warped_bboxes = np.array(warped_bboxes)
                
                if len(warped_bboxes) != 0:
                    facade = DummyFacadeModel()
                    facade.build(warped_bboxes, inverse_matrix)
                    output_dict[img_path.stem] = facade.todict()
                    
                    with output_file.open('w') as f:
                        yaml.safe_dump(output_dict, f)
                else: 
                    facade = DummyFacadeModel()
                    facade.build(bboxes)
                    output_dict[img_path.stem] = facade.todict()
                    with output_file.open('w') as f:
                        yaml.safe_dump(output_dict, f)
            else:
                facade = DummyFacadeModel()
                facade.build(bboxes)
                output_dict[img_path.stem] = facade.todict()
                with output_file.open('w') as f:
                    yaml.safe_dump(output_dict, f)
        else:
            facade = DummyFacadeModel()
            facade.build(bboxes)
            output_dict[img_path.stem] = facade.todict()
            with output_file.open('w') as f:
                yaml.safe_dump(output_dict, f)
        # except: 
        #     facade = DummyFacadeModel()
        #     facade.build(bboxes)
        #     output_dict[img_path.stem] = facade.todict()
        #     with output_file.open('w') as f:
        #         yaml.safe_dump(output_dict, f)


def main():
    input_folder = './test/images'
    output_file = './output/preds.yaml'

    predict_and_transform(input_folder, output_file)


if __name__ == '__main__':
    main()
