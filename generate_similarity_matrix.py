import numpy as np
from get_membership_degree import get_membership_degree

def generate_similarity_matrix(descriptions, instances, weight_matrix):
    num_instances = instances.num_instances()
    similarity = np.zeros((num_instances, num_instances))

    for i in range(len(descriptions)):
        for j in range(len(descriptions)):
            and_set = []

            if i == j:
                similarity[i, j] = 1
            else:
                and_set.extend(descriptions[i])

                for t in descriptions[j]:
                    if t not in and_set:
                        and_set.append(t)

                degree_i = sum([get_membership_degree(weight_matrix, t, i) for t in and_set])
                degree_i /= len(and_set)

                degree_j = sum([get_membership_degree(weight_matrix, t, j) for t in and_set])
                degree_j /= len(and_set)

                similarity[i, j] = min(degree_i, degree_j)

    return similarity



