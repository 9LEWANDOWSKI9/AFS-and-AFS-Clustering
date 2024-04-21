import numpy as np
from get_membership_degree import get_membership_degree

# Placeholder for AFS.getMembershipDegree() function
# def get_membership_degree(weight_matrix, description1, description2):
#     # Replace this with the actual implementation
#     return np.random.rand()



def get_description_for_class(descriptions, cluster_result, weight_matrix):
    class_description = {}

    for class_number, in_class in cluster_result.items():
        in_class_samples = in_class

        out_of_class_samples = []
        for other_class_number, other_class_samples in cluster_result.items():
            if other_class_number != class_number:
                out_of_class_samples.extend(other_class_samples)

        max_index_d = 0
        max_average = float('-inf')
        max_index_f = 0
        max_degree = float('-inf')

        for sample in in_class_samples:
            in_average = np.mean(
                [get_membership_degree(weight_matrix, descriptions[sample], ss) for ss in in_class_samples])

            out_average = np.mean(
                [get_membership_degree(weight_matrix, descriptions[sample], sss) for sss in out_of_class_samples])
            if (in_average - out_average) > max_average:
                max_average = in_average - out_average
                max_index_d = sample

            in_degree = max([get_membership_degree(weight_matrix, descriptions[sample], tt) for tt in in_class_samples])

            out_degree = max(
                [get_membership_degree(weight_matrix, descriptions[sample], ttt) for ttt in out_of_class_samples])
            if (in_degree - out_degree) > max_degree:
                max_degree = in_degree - out_degree
                max_index_f = sample

        class_description[class_number] = [descriptions[max_index_d], descriptions[max_index_f]]

    return class_description



