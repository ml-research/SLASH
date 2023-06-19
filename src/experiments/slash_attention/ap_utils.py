import numpy as np
import torch 
import time
import random

'''
All helper functions to compute the average precision for different datasets
'''


def get_obj_encodings(dataloader):
    '''
    Collects all object encodings in a single tensor for the test dataset
    @param dataloader: Dataloader object for a shapeworld/clevr like dataset that returns an object encoding 
    '''
    obj_list = []
    for _,_,obj_encoding in dataloader:
        for i in range(0, len(obj_encoding)):
            obj_list.append(obj_encoding[i])

    obj_list = torch.stack(obj_list)
    
    return obj_list



def inference_map_to_array(inference, only_color=False, only_shape=False, only_shade=False, only_size=False, only_material=False):
    '''
    Returns an encoding of the output objects having shape [bs, num_slots, properties] for the slot attention experiment. The encoding can be used then for the AP metric.
    properties depends on the number of classes + 1 for the prediction confidence  predicted e.g. red,green,blue + prediction confidence -> 3+1 = 4
    @param inference: Hashmap of the form {'slot':{'color' : [bs,num_colors], ...}, ...}
    '''
    
    all_object_encodings = None #contains the encoding all slots over the batchsize
    
    #iterate over all slots and collect the properties 
    for slot in inference:
        slot_object_encoding = torch.Tensor([]) #contains the encoding for one slot over the batchsize
        
        prediction_confidence = None
        
        
        #iterate over all properties and put them together into one vector
        for prop in inference[slot]:
            slot_object_encoding = torch.cat((slot_object_encoding, inference[slot][prop].cpu()), axis=1)
            

            #select the prediction confidence. #We want to select the highest prediction value for the selected properties e.g. col, shape, col+shape
            if (prop == 'color' and only_color) or (prop == 'shape' and only_shape) or (prop == 'shade' and only_shade) or (prop == 'size' and only_size) or (prop == 'material' and only_material) or (only_color is False and only_shape is False and only_shade is False and only_size is False): 
                #collect the argmax prediction values for each property

                if prediction_confidence is None:
                    prediction_confidence = torch.max(inference[slot][prop].cpu(), axis=1)[0][None,:]
                else:
                    prediction_confidence = torch.cat((prediction_confidence, torch.max(inference[slot][prop].cpu(), axis=1)[0][None,:]))

                    
        #take the mean over the prediction confidence if we have more than one property and then append it to the properties tensor
        if only_color is False and only_shape is False:
            prediction_confidence = prediction_confidence.mean(axis=0)
        else:
            prediction_confidence = prediction_confidence.squeeze()
        
        slot_object_encoding = torch.cat((slot_object_encoding, prediction_confidence[:,None]), axis=1)

        
        #concatenate all slot encodings
        if all_object_encodings is None:
            all_object_encodings = slot_object_encoding[None,:,:]
        else:
            all_object_encodings = torch.cat((all_object_encodings, slot_object_encoding[None,:,:]), axis=0)
    return torch.einsum("abc->bac", all_object_encodings)

        
def average_precision(pred, attributes, distance_threshold, dataset='', subset_size=None, only_color=False, only_shape=False, only_shade=False, only_size=False, only_material=False):
    """Computes the average precision for CLEVR or Shapeworld.

    This function computes the average precision of the predictions specifically
    for the CLEVR dataset. First, we sort the predictions of the model by
    confidence (highest confidence first). Then, for each prediction we check
    whether there was a corresponding object in the input image. A prediction is
    considered a true positive if the discrete features are predicted correctly
    and the predicted position is within a certain distance from the ground truth
    object.

    Args:
    pred: Tensor of shape [batch_size, num_elements, dimension] containing
      predictions. The last dimension is expected to be the confidence of the
      prediction.
    attributes: Tensor of shape [batch_size, num_elements, dimension] containing
      ground-truth object properties.
    distance_threshold: Threshold to accept match. -1 indicates no threshold.
    only_color: Only consider color as a relevant property for a match.
    only_shape: Only consider shape as a relevant property for a match.
    only_shade: Consider only shade as a relevant property for a match. SHAPEWORLD
    only_material: Consider only material as a relevant property for a match. CLEVR

    Returns:
    Average precision of the predictions.
    """

    if subset_size is not None:
        idx = random.sample(list(np.arange(0,attributes.shape[0])), subset_size)
        attributes = attributes[idx,:,:]
        pred = pred[idx,:,:]



    [batch_size, _, element_size] = attributes.shape
    [_, predicted_elements, _] = pred.shape


    def unsorted_id_to_image(detection_id, predicted_elements):
        """Find the index of the image from the unsorted detection index."""
        return int(detection_id // predicted_elements)

    flat_size = batch_size * predicted_elements
    flat_pred = np.reshape(pred, [flat_size, element_size])
    sort_idx = np.argsort(flat_pred[:, -1], axis=0)[::-1]  # Reverse order.

    sorted_predictions = np.take_along_axis(
        flat_pred, np.expand_dims(sort_idx, axis=1), axis=0)
    idx_sorted_to_unsorted = np.take_along_axis(
      np.arange(flat_size), sort_idx, axis=0)
    

    def process_targets_CLEVR(target):
        """Unpacks the target into the CLEVR properties."""
        size = np.argmax(target[:3])
        material = np.argmax(target[3:6])
        shape = np.argmax(target[6:10])
        color = np.argmax(target[10:19])
        real_obj = target[19]
        
        attr = [size, material, shape, color]
        
        is_bg = False
        if color == 8 and shape == 3 and material == 2 and size == 2:
            is_bg = True
           
        if only_color: #consider only the color properties for comparison
            attr = [color]                
        if only_shape: #consider only the shape properties for comparison
            attr = [shape]                
        if only_material: #consider only the shade properties for comparison
            attr = [material]                
        if only_size: #consider only the size properties for comparison
            attr = [size]             

        coords = np.array([0,0,0]) # We don't learn the coordinates of the objects
        # return coords, object_size, material, shape, color, real_obj
        return coords, attr, is_bg, real_obj
    
      
    
    
    def process_targets_SHAPEWORLD4(target):
        """Unpacks the target into the Shapeworld properties."""
        
        #[c,c,c,c,c,c,c,c,c , s,s,s,s , h,h,h, z,z,z, confidence]
            
        color = np.argmax(target[:9])
        shape = np.argmax(target[9:13])
        shade = np.argmax(target[13:16])
        size = np.argmax(target[16:19])
        real_obj = target[19]
        
        
        is_bg= False
        attr = [shape, color, shade, size]

        if color == 8 and shape == 3 and shade == 2 and size == 2:
            is_bg = True
                
        if only_color: #consider only the color properties for comparison
            attr = [color]                
        if only_shape: #consider only the shape properties for comparison
            attr = [shape]                
        if only_shade: #consider only the shade properties for comparison
            attr = [shade]                
        if only_size: #consider only the size properties for comparison
            attr = [size]                
        
        coords = np.array([0,0,0])
        return coords, attr, is_bg, real_obj
    
    
    def process_targets_SHAPEWORLD_OOD(target):
        """Unpacks the target into the Shapeworld ood properties."""
        
        #[c,c,c,c,c,c,c,c,c , s,s,s,s,s,s , h,h,h, z,z,z, confidence]
            
        color = np.argmax(target[:9])
        shape = np.argmax(target[9:15])
        shade = np.argmax(target[15:18])
        size = np.argmax(target[18:21])
        real_obj = target[21]
        
        
        is_bg= False
        attr = [shape, color, shade, size]

        if color == 8 and shape == 5 and shade == 2 and size == 2:
            is_bg = True
                
        if only_color: #consider only the color properties for comparison
            attr = [color]                
        if only_shape: #consider only the shape properties for comparison
            attr = [shape]                
        if only_shade: #consider only the shade properties for comparison
            attr = [shade]                
        if only_size: #consider only the size properties for comparison
            attr = [size]                
        
        coords = np.array([0,0,0])
        return coords, attr, is_bg, real_obj
        
    #switch for different dataset encodings
    def process_targets(target):
        if dataset == 'SHAPEWORLD4':
            return process_targets_SHAPEWORLD4(target)
        elif dataset == 'SHAPEWORLD_OOD':
            return process_targets_SHAPEWORLD_OOD(target)
        elif dataset == 'CLEVR':
            return process_targets_CLEVR(target)
        else:
            raise RuntimeError('AP metric not implemented for dataset '+dataset)

    true_positives = np.zeros(sorted_predictions.shape[0])
    false_positives = np.zeros(sorted_predictions.shape[0])
    true_negatives = np.zeros(sorted_predictions.shape[0]) 
    
    detection_set = set()
    match_count = 0
    
    for detection_id in range(sorted_predictions.shape[0]):
        # Extract the current prediction.
        current_pred = sorted_predictions[detection_id, :]
        # Find which image the prediction belongs to. Get the unsorted index from
        # the sorted one and then apply to unsorted_id_to_image function that undoes
        # the reshape.
        original_image_idx = unsorted_id_to_image(
            idx_sorted_to_unsorted[detection_id], predicted_elements)
        # Get the ground truth image.
        gt_image = attributes[original_image_idx, :, :]

        # Initialize the maximum distance and the id of the groud-truth object that
        # was found.
        best_distance = 10000
        best_id = None
          

        # Unpack the prediction by taking the argmax on the discrete attributes.
        #(pred_coords, pred_object_size, pred_material, pred_shape, pred_color,_) = process_targets(current_pred)
        (pred_coords, pred_attr, is_bg, _) = process_targets(current_pred)
        
        
        # Loop through all objects in the ground-truth image to check for hits.
        for target_object_id in range(gt_image.shape[0]):
            target_object = gt_image[target_object_id, :]
            
            
            # Unpack the targets taking the argmax on the discrete attributes.
            #(target_coords, target_object_size, target_material, target_shape,
            #   target_color, target_real_obj) = process_targets(target_object)
            
            (target_coords, target_attr ,_, target_real_obj) = process_targets(target_object)
                        
            # Only consider real objects as matches.
            if target_real_obj:
                # For the match to be valid all attributes need to be correctly
                # predicted.
            
                match = pred_attr == target_attr
                
                if match:
                    # If a match was found, we check if the distance is below the
                    # specified threshold. Recall that we have rescaled the coordinates
                    # in the dataset from [-3, 3] to [0, 1], both for `target_coords` and
                    # `pred_coords`. To compare in the original scale, we thus need to
                    # multiply the distance values by 6 before applying the norm.
                    distance = np.linalg.norm((target_coords - pred_coords) * 6.)

                    match_count +=1
                    # If this is the best match we've found so far in terms of distance we remember it.
                    if distance < best_distance:
                        best_distance = distance
                        best_id = target_object_id
                        
                    #If we found a match we need to check if another object with the same attributes was already assigned to it.
                    elif distance_threshold == -1 and (original_image_idx,target_object_id) not in detection_set:
                        best_id = target_object_id
        
        if best_distance < distance_threshold or distance_threshold == -1:
            # We have detected an object correctly within the distance confidence.
            # If this object was not detected before it's a true positive.
            if best_id is not None:
                if (original_image_idx, best_id) not in detection_set and is_bg:
                    true_negatives[detection_id] = 1
                    detection_set.add((original_image_idx, best_id))
                elif (original_image_idx, best_id) not in detection_set:
                    true_positives[detection_id] = 1
                    detection_set.add((original_image_idx, best_id))

                else:
                    false_positives[detection_id] = 1
            else:
                false_positives[detection_id] = 1
        else:
            false_positives[detection_id] = 1
    

    accumulated_fp = np.cumsum(false_positives)
    accumulated_tp = np.cumsum(true_positives)
    accumulated_tn = np.cumsum(true_negatives)
    
    #save tp, fp, tn
    true_positives = accumulated_tp[-1]
    false_positives = accumulated_fp[-1]
    true_negatives = accumulated_tn[-1]
    
    
    #the relevant examples is the amount of object substracted by the background detection per image(true negative)
    relevant_examples = batch_size * predicted_elements - true_negatives

    recall_array = accumulated_tp / relevant_examples
    precision_array = np.divide(accumulated_tp, (accumulated_fp + accumulated_tp))
    #print(detection_set)

    
    objects_detected = np.zeros((batch_size,predicted_elements)) 
    for detection in detection_set:
        objects_detected[detection[0], detection[1]] = 1

    
    #check if all detections are true and then count all correctly classified images
    correctly_classified = np.sum(np.all(objects_detected, axis=1))
    
    
    return compute_average_precision(
      np.array(precision_array, dtype=np.float32),
      np.array(recall_array, dtype=np.float32)), true_positives, false_positives, true_negatives, correctly_classified



def compute_average_precision(precision, recall):
    """Computation of the average precision from precision and recall arrays."""
    recall = recall.tolist()
    precision = precision.tolist()
    recall = [0] + recall + [1]
    precision = [0] + precision + [0]

    for i in range(len(precision) - 1, -0, -1):
        precision[i - 1] = max(precision[i - 1], precision[i])

    indices_recall = [
      i for i in range(len(recall) - 1) if recall[1:][i] != recall[:-1][i]
    ]

    average_precision = 0.
    for i in indices_recall:
        average_precision += precision[i + 1] * (recall[i + 1] - recall[i])
    return average_precision