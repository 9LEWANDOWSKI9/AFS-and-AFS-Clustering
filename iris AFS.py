from clustering import AFS
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris


iris_data = load_iris()

x = iris_data.data

AFS_model = AFS()
class_index = 0
parameters = AFS_model.get_parameters(x)
weight_matrix = AFS_model.generate_weight_matrix(x, parameters)
selected_attributes = [0, 1, 2, 3]

instances_object = AFS.Instances([AFS.Instance(values=x[i]) for i in range(len(x))], class_index)
descriptions = AFS_model.generate_descriptions(instances_object, weight_matrix, neighbour=3, feature=2, epsilon=0.1)
result = AFS_model.statistic_cluster_result(iris_data.data, AFS_model.get_class_label_by_membership_degree(descriptions))
accuracy = AFS_model.calculate_accuracy(result)
print("Accuracy:", accuracy)
print(type(result))
print("Structure of an element in result:", result[0])

AFS_model.print_membership_function(AFS_model.get_description_for_class(descriptions, result, weight_matrix), iris_data, weight_matrix)


class_labels = AFS_model.get_class_label_by_membership_degree(descriptions)

pca = PCA(n_components=2)
reduced_data = pca.fit_transform(x)

plt.figure(figsize=(8, 6))

for i in range(len(set(class_labels))):
    class_samples = reduced_data[class_labels == i]
    plt.scatter(class_samples[:, 0], class_samples[:, 1], label=f'Cluster {i + 1}')

plt.title('AFS Clustering Results')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.show()


