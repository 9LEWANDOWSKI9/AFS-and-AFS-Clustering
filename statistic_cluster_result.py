from collections import Counter
from operator import itemgetter

def statistic_cluster_result(instances, class_label):
    cluster_result = []

    labels = [[] for _ in range(len(set(instances['class'].tolist())))]

    class_index = 0
    for i in range(len(instances) - 1):
        if instances['class'].iloc[i + 1] == instances['class'].iloc[i]:
            labels[class_index].append(class_label[i])
        else:
            labels[class_index].append(class_label[i])
            class_index += 1

    labels[class_index].append(class_label[i])

    for i in range(len(labels)):
        frequency = Counter(labels[i])

        sorted_frequency = sorted(frequency.items(), key=itemgetter(1), reverse=True)
        cluster_map = {"right": sorted_frequency[0][1], "wrong": sum(value for key, value in sorted_frequency[1:])}
        cluster_result.append(cluster_map)

    return cluster_result

# instances_data = {'class': [1, 1, 2, 2, 3, 3, 3]}
# class_label_data = [0, 0, 1, 1, 2, 2, 2]
#
# instances_df = pd.DataFrame(instances_data)
# cluster_result_result = statistic_cluster_result(instances_df, class_label_data)
# print(f"Cluster Result: {cluster_result_result}")
