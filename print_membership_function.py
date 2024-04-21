import numpy as np
from get_membership_function import get_membership_function

def print_membership_function(class_description, instances, weight_matrix):
    for class_number, concept_descriptions in class_description.items():
        print(f"第 {class_number} 类的隶属函数 =================================================================")
        degrees = get_membership_function(concept_descriptions, instances, weight_matrix)

        for degree in degrees:
            print(degree)



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
# print_membership_function(class_descriptions, instances, weight_matrix_data)
