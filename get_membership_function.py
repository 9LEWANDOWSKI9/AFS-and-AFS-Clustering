import numpy as np
from get_membership_degree import get_membership_degree

def get_membership_function(class_description, instances, weight_matrix):
    degrees = np.zeros(instances.numInstances())
    for i, instance in enumerate(instances):
        max_degree = float('-inf')

        for concept_description in class_description:
            degree = get_membership_degree(weight_matrix, concept_description, i)

            if degree > max_degree:
                max_degree = degree

        degrees[i] = max_degree

    return degrees


# class Instances:
#     def __init__(self, num_instances):
#         self.numInstances = num_instances
#
# class_descriptions = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
# instances = Instances(5)
# weight_matrix_data = np.random.rand(len(class_descriptions[0]), len(class_descriptions))
#
# membership_function = get_membership_function(class_descriptions, instances, weight_matrix_data)
# print(f"Membership Function: {membership_function}")
