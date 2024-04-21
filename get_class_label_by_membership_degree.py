import numpy as np

def get_class_label_by_membership_degree(membership_function):
    num_instances = membership_function.shape[1]
    class_labels = []

    for j in range(num_instances):
        max_degree = float('-inf')
        max_class = 0

        for i in range(membership_function.shape[0]):
            if membership_function[i, j] > max_degree:
                max_degree = membership_function[i, j]
                max_class = i

        class_labels.append(max_class)

    return class_labels


# membership_function_data = np.array([[0.8, 0.4, 0.7],
#                                       [0.2, 0.9, 0.6]])
#
# class_labels_result = get_class_label_by_membership_degree(membership_function_data)
# print(f"Class Labels: {class_labels_result}")
