def get_distances_in_attribute(instances, x, index):
    distances = {}

    for i, ins in enumerate(instances):
        distances[i] = abs(x.values[index] - ins.values[index])

    sorted_distances = sorted(distances.items(), key=lambda x: x[1])

    return sorted_distances



class Instance:
    def __init__(self, values):
        self.values = values
class Instances:
    def __init__(self, instances, class_index):
        self.instances = instances
        self.class_index = class_index

    def numAttributes(self):
        if self.instances:
            return len(self.instances[0].values) + 1
        else:
            return 0

    def __iter__(self):
        return iter(self.instances)


# instance_x = Instance([1.0, 2.0, 3.0])
# instance_1 = Instance([2.0, 3.0, 4.0])
# instance_2 = Instance([3.0, 4.0, 5.0])
# instance_3 = Instance([4.0, 5.0, 6.0])
# instances = Instances([instance_1, instance_2, instance_3], class_index=1)
# all_distances = []
#
# for i, ins in enumerate(instances):
#     distances = {}
#     x=instance_x
#     index = 1
#     distances[i] = abs(x.values[index] - ins.values[index])
#     distance_value = distances[i]
#     all_distances.append(distance_value)
# print(all_distances)
# sorted_distances = sorted(all_distances)
# print(sorted_distances[:])


#
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
# distances = get_distances_in_attribute(instances, instance_x, index=1)
# print("Distances in Attribute:", distances)