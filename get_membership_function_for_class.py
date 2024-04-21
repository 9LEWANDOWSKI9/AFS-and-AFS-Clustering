import numpy as np
from get_membership_function import get_membership_function

def get_membership_function_for_class(class_description, instances, weight_matrix):
    num_classes = len(class_description)
    num_instances = instances.num_instances
    membership_function = np.zeros((num_classes, num_instances))

    for class_number, concept_descriptions in class_description.items():
        degrees = get_membership_function(concept_descriptions, instances, weight_matrix)
        membership_function[class_number] = degrees

    return membership_function


# class Instances:
#     def __init__(self, num_instances):
#         self.num_instances = num_instances
#
# class_descriptions = {
#     1: [[1, 2, 3], [4, 5, 6]],
#     2: [[7, 8, 9], [10, 11, 12]],
#     # Add more class descriptions as needed
# }
#
# instances = Instances(5)
# weight_matrix_data = np.random.rand(len(class_descriptions[1][0]), len(class_descriptions[1][0]))
#
# membership_function_for_class = get_membership_function_for_class(class_descriptions, instances, weight_matrix_data)
# print(f"Membership Function for Class: {membership_function_for_class}")
