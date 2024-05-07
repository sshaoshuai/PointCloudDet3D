import numpy as np

def get_mAP(prec):
    sums = 0
    for i in range(0, prec.shape[-1], 4):
        sums = sums + prec[..., i]
    return sums / 11 * 100


def get_mAP_R40(prec):
    sums = 0
    for i in range(1, prec.shape[-1]):
        sums = sums + prec[..., i]
    return sums / 40 * 100

def transform_to_aabb(box):
            """ Convert an oriented bounding box to an axis-aligned bounding box. """
            x, y, z, dx, dy, dz, yaw = box
            # Compute the corner points of the box based on center, dimensions, and yaw
            corners = np.array([
                [x - dx / 2, y - dy / 2, z - dz / 2],
                [x + dx / 2, y - dy / 2, z - dz / 2],
                [x - dx / 2, y + dy / 2, z - dz / 2],
                [x + dx / 2, y + dy / 2, z - dz / 2],
                [x - dx / 2, y - dy / 2, z + dz / 2],
                [x + dx / 2, y - dy / 2, z + dz / 2],
                [x - dx / 2, y + dy / 2, z + dz / 2],
                [x + dx / 2, y + dy / 2, z + dz / 2]
            ])

            # Rotate corners
            c, s = np.cos(yaw), np.sin(yaw)
            rotation_matrix = np.array([
                [c, -s, 0],
                [s, c, 0],
                [0, 0, 1]
            ])lculate_max_iou(gt_boxes, dt_boxes)
            accuracies = [iou >= threshold for iou in max_ious]
            accuracy = np.mean(accuracies) if accuracies else 0 
            return accuracy
            corners = corners @ rotation_matrix.T

            # Calculate the min and max x, y, z from rotated corners for the AABB
            min_corner = np.min(corners, axis=0)
            max_corner = np.max(corners, axis=0)
            
            return np.concatenate([min_corner, max_corner - min_corner])

        def calculate_iou_3d(box1, box2):
            """ Calculates the approximate IoU between two 3D oriented bounding boxes. """
            # Transform both boxes to axis-aligned bounding boxes
            aabb1 = transform_to_aabb(box1)
            aabb2 = transform_to_aabb(box2)

            # Extract min and max points
            min1, max1 = aabb1[:3], aabb1[:3] + aabb1[3:6]
            min2, max2 = aabb2[:3], aabb2[:3] + aabb2[3:6]

            # Calculate the intersection bounds
            intersect_min = np.maximum(min1, min2)
            intersect_max = np.minimum(max1, max2)
            intersect_dims = np.maximum(intersect_max - intersect_min, 0)

            # Calculate intersection and union volumes
            intersect_volume = np.prod(intersect_dims)
            volume1 = np.prod(aabb1[3:6])
            volume2 = np.prod(aabb2[3:6])
            union_volume = volume1 + volume2 - intersect_volume

            # Calculate IoU
            iou = intersect_volume / union_volume if union_volume > 0 else 0
            return iou

        def calculate_max_iou(gt_boxes, dt_boxes):
            """ Calculates the mean IoU between lists of ground truth and detected boxes. """
            max_ious = []
            for dt_box in dt_boxes:
                ious = [calculate_iou_3d(gt_box, dt_box) for gt_box in gt_boxes]
                max_iou = max(ious) if ious else 0
                max_ious.append(max_iou)
            return max_ious

        def calculate_mean_iou(gt_boxes, dt_boxes):
            """ Calculates the mean IoU between lists of ground truth and detected boxes. """
            ious = [calculate_iou_3d(gt_box, dt_box) for gt_box, dt_box in zip(gt_boxes, dt_boxes)]
            mean_iou = np.mean(ious)
            return mean_iou

        def calculate_accuracy(gt_boxes, dt_boxes, threshold=0.5):
            """ Calculates the accuracy based on a threshold IoU value. """
            ious = [calculate_iou_3d(gt_box, dt_box) for gt_box, dt_box in zip(gt_boxes, dt_boxes)]
            accuracies = [iou >= threshold for iou in ious]
            accuracy = np.mean(accuracies)
            return accuracy

        def calculate_detection_accuracy(gt_boxes, dt_boxes, threshold=0.5):
            """ Calculates the accuracy based on a threshold IoU value. """
            max_ious = calculate_max_iou(gt_boxes, dt_boxes)
            accuracies = [iou >= threshold for iou in max_ious]
            accuracy = np.mean(accuracies) if accuracies else 0 
            return accuracy


        def get_official_eval_result(gt_annos, dt_annos):
            # check whether alpha is valid
            compute_aos = False
            for anno in dt_annos:
                if anno['alpha'].shape[0] != 0:
                    if anno['alpha'][0] != -10:
                        compute_aos = True
                    break

            mIOUs = []
            APs = []
            for gt, det in zip(eval_gt_annos, eval_det_annos):
                mIOUs.append(calculate_max_iou(gt['gt_boxes_lidar'], det['boxes_lidar']))
                APs.append(calculate_detection_accuracy(gt['gt_boxes_lidar'], det['boxes_lidar'], 0.05))

            result_string = ""
            print(f"\n\n\n\n######################################################################")
            for mIOU in mIOUs:
                if any(mIOU) > 0.0:
                    #print(mIOU)
                    ious = [i for i in mIOU if i > 0]
                    print(len(ious))
                        
            print("APs: ")
            for AP in APs:
                if AP > 0.0:
                    print(AP)
            print(f"\n\n\n\n######################################################################")