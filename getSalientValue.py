from get_distances_in_attribute import get_distances_in_attribute

def get_salient_value(instances, instance, attribute_index, neighbour):
    salient_value = 0
    separability = 0
    compactness = 0
    phi = 0.00000001

    distances = get_distances_in_attribute(instances, instance, attribute_index)

    for i in range(neighbour):
        compactness += distances[i][1]
    compactness /= neighbour

    flag = 0
    for i in range(len(distances) - 1, 0, -1):
        if flag == neighbour:
            break
        else:
            separability += distances[i][1]
            flag += 1

    separability /= neighbour

    if compactness == 0:
        salient_value = separability / phi
    else:
        salient_value = separability / compactness

    return salient_value



#
# class Instance:
#     def __init__(self, values):
#         self.values = values
#
#     def value(self, index):
#         return self.values[index]
#
# class Instances(list):
#     pass
#
# instance_x = Instance([1.0, 2.0, 3.0])
# instance_1 = Instance([2.0, 3.0, 4.0])
# instance_2 = Instance([3.0, 4.0, 5.0])
# instance_3 = Instance([4.0, 5.0, 6.0])
# instances = Instances([instance_1, instance_2, instance_3])
#
# attribute_index = 1
# neighbour = 2
# result = get_salient_value(instances, instance_x, attribute_index, neighbour)
# print("Salient Value:", result)