import numpy as np
from collections import Counter
from operator import itemgetter

class TriangularFunction:
    def __init__(self, a, b, c):
        self.a = a
        self.b = b
        self.c = c
    def apply(self, x):
        # 确认模糊函数的类别
        if self.a != self.b and self.b != self.c and self.a != self.c:  # class 1: a! = b != c
            if x < self.a or x > self.c:
                return 0.0
            elif x == self.a or x == self.c:
                return 0.0
            elif x == self.b:
                return 1.0
            elif self.a < x < self.b:
                return (x - self.a) / (self.b - self.a)
            elif self.b < x < self.c:
                return (self.c - x) / (self.c - self.b)
        elif self.a == self.b and self.b != self.c:  # class 2: a = b != c
            if x == self.a:
                return 1.0
            elif x < self.a:
                return 1.0
            elif x == self.c:
                return 0.0
            elif x > self.c:
                return 0.0
            elif self.b < x < self.c:
                return (self.c - x) / (self.c - self.b)

        elif self.a != self.b and self.b == self.c:  # class 3: a != b = c
            if x < self.a:
                return 0.0
            elif x == self.a:
                return 0.0
            elif x == self.c:
                return 1.0
            elif x > self.b:
                return 1.0
            elif self.a < x < self.b:
                return (x - self.a) / (self.b - self.a)

        # 如果计算错误，返回double的最大值
        return float('inf')




class AFS():
    class Instance:
        def __init__(self, values):
            self.values = values

    class Instances:
        def __init__(self, instances, class_index):
            self.instances = instances
            self.class_index = class_index

        def numAttributes(self):
            # 返回实例集合的属性数量
            if self.instances:
                return len(self.instances[0].values) + 1  # 加上类属性
            else:
                return 0

        def num_instances(self):
            return len(self.instances)  # 添加方法来返回实例数量
        def __iter__(self):
            return iter(self.instances)

        def get_class_labels(self):
            return [instance.values[self.class_index] for instance in self.instances]
    def get_parameters(self, data):
        parameters = np.zeros((3, data.shape[1]))

        for j in range(data.shape[1]):
            max_val = float('-inf')
            min_val = float('inf')
            sum_val = 0

            for i in range(data.shape[0]):
                # 寻找最大值
                max_val = max(max_val, data[i, j])

                # 寻找最小值
                min_val = min(min_val, data[i, j])

                # 累加计算总和
                sum_val += data[i, j]

            # 设置统计参数矩阵的值
            # 最小值
            parameters[0, j] = min_val

            # 均值
            parameters[1, j] = sum_val / data.shape[0]

            # 最大值
            parameters[2, j] = max_val

        return parameters

    def generate_weight_matrix(self, data, parameters):
        num_rows, num_cols = data.shape
        weight_matrix = np.zeros((num_rows, 3 * num_cols))
        flag = 0

        for j in range(num_cols):
            tf = [None] * 3

            for i in range(len(tf)):
                if i == 0:
                    tf[i] = TriangularFunction(parameters[0, j], parameters[0, j], parameters[2, j])
                elif i == len(tf) - 1:
                    tf[i] = TriangularFunction(parameters[0, j], parameters[2, j], parameters[2, j])
                else:
                    tf[i] = TriangularFunction(parameters[0, j], parameters[1, j], parameters[2, j])

            for i in range(len(tf)):
                for m in range(num_rows):
                    weight_matrix[m, flag] = tf[0].apply(data[m, j])
                    weight_matrix[m, flag + 1] = tf[1].apply(data[m, j])
                    weight_matrix[m, flag + 2] = tf[2].apply(data[m, j])

            flag += 3

        return weight_matrix

    # @staticmethod
    def get_concepts(self, selected_attributes):
        result = []

        for i in selected_attributes:
            result.extend([i * 3, i * 3 + 1, i * 3 + 2])

        return result

    # @staticmethod
    def generate_descriptions(self, instances, weight_matrix, neighbour, feature, epsilon):
        descriptions_list = []
        max_length = 0  # 记录最大的描述长度

        for i, ins in enumerate(instances):
            attributes = self.select_attributes_by_fnsn(instances, ins, neighbour, feature)
            concepts = self.get_concepts(attributes)

            description = self.get_selected_concepts(weight_matrix, i, concepts, epsilon)
            print(f"Shape of description {i}: {description.shape}")
            descriptions_list.append(description)

            # 更新最大描述长度
            max_length = max(max_length, len(description))

            print(f"第{i + 1}个样本的描述生成完毕！")

        # 确保每个描述都具有相同的长度
        for i in range(len(descriptions_list)):
            current_length = len(descriptions_list[i])
            if current_length < max_length:
                # 如果当前描述长度小于最大长度，则补充0直到长度相等
                descriptions_list[i] = np.concatenate([descriptions_list[i], np.zeros(max_length - current_length)])

        # 将处理后的描述列表堆叠成数组
        descriptions = np.vstack(descriptions_list)

        return descriptions


    def generate_similarity_matrix(self, descriptions, instances, weight_matrix):
        num_instances = instances.num_instances()
        similarity = np.zeros((num_instances, num_instances))

        for i in range(len(descriptions)):
            for j in range(len(descriptions)):
                and_set = []

                if i == j:
                    similarity[i, j] = 1
                else:
                    # 求样本 x_i 和 x_j 的描述的交
                    and_set.extend(descriptions[i])

                    for t in descriptions[j]:
                        if t not in and_set:
                            and_set.append(t)

                    # 计算取交后的描述的隶属度
                    degree_i = sum([self.get_membership_degree(weight_matrix, t, i) for t in and_set])
                    degree_i /= len(and_set)

                    degree_j = sum([self.get_membership_degree(weight_matrix, t, j) for t in and_set])
                    degree_j /= len(and_set)

                    similarity[i, j] = min(degree_i, degree_j)

        return similarity

    def get_selected_concepts(self, weight_matrix, index_of_instance, concept, epsilon):
        degrees = {}

        for index_of_concept in concept:
            if 0 <= index_of_concept < weight_matrix.shape[1]:
                degrees[index_of_concept] = self.get_membership_degree(weight_matrix, index_of_concept, index_of_instance)

        new_concept = []
        flag = 0
        degree_max = float('-inf')
        index_max = 0

        for c in concept:
            if c in degrees and degrees[c] > degree_max:
                degree_max = degrees[c]
                index_max = c
            flag += 1

            if flag == 3:
                new_concept.append(index_max)
                flag = 0
                degree_max = float('-inf')
                index_max = 0

        max_degree = float('-inf')

        for c in new_concept:
            if c in degrees and degrees[c] > max_degree:
                max_degree = degrees[c]

        result = [c for c in new_concept if c in degrees and degrees[c] > (max_degree - epsilon)]

        result = np.array(result)

        return result


    def get_membership_degree(self, weight_matrix, index_of_concept, index_of_instance):

        '''
        计算一个实例对于一个Fuzzy_Term的隶属度。
        :param weight_matrix: 一个包含权重值的矩阵，(indexOfInstance,indexOfConcept)
        :param index_of_concept: 表示概念的索引，即在权重矩阵中的列索引。
        :param index_of_instance: 表示实例的索引，即在权重矩阵中的行索引。
        函数通过循环遍历权重矩阵的每一行，检查每一行对应的 weight_matrix[i, index_of_concept]
        是否小于等于给定概念的权重值 weight_matrix[index_of_instance, index_of_concept]。如果满足这个条件，就增加计数器的值。
        计算隶属度，即满足条件的行数除以总行数
        :return: (int) 隶属度值
        '''
        print(f"index_of_concept: {index_of_concept}, index_of_instance: {index_of_instance}, weight_matrix shape: {weight_matrix.shape}")
        # index_of_concept = index_of_concept.astype(int)
        count = sum(weight_matrix[i, index_of_concept] <= weight_matrix[index_of_instance, index_of_concept] for i in
                    range(weight_matrix.shape[0]))
        return count / weight_matrix.shape[0]

    def get_membership_complex_degree(self, weight_matrix, description, index_of_instance):
        '''
        这个函数计算了每个描述在给定样本索引上的隶属度，然后返回这些隶属度的平均值
        :param ：
        :return:  the same as get_membership_degree
        you should check the get_membership_degree.py
        '''
        degree = 0

        for value in description:
            degree += self.get_membership_degree(weight_matrix, value, index_of_instance)

        return degree / len(description)

    def select_attributes_by_fnsn(self, instances, instance, neighbour, feature):
        salient_map = {}

        flag = 0


        for i in range(instances.numAttributes()):
            if i == instances.class_index:
                continue
            else:
                salient_value = self.get_salient_value(instances, instance, flag, neighbour)
                salient_map[i] = salient_value
                flag += 1

        # 对 salient_map 进行按 value 降序排列
        sorted_salient_map = sorted(salient_map.items(), key=lambda x: x[1], reverse=True)

        result = [entry[0] for entry in sorted_salient_map[:feature]]
        return result

    def get_salient_value(self, instances, instance, attribute_index, neighbour):
        salient_value = 0
        separability = 0
        compactness = 0
        phi = 0.00000001

        distances = self.get_distances_in_attribute(instances, instance, attribute_index)

        # 求 compactness，即样本 x 与 k 个最近邻居的距离的平均值
        for i in range(neighbour):
            compactness += distances[i][1]
        compactness /= neighbour

        # 求 separability，即样本 x 与 l 个最远邻居的距离的平均值
        flag = 0
        for i in range(len(distances) - 1, 0, -1):
            if flag == neighbour:
                break
            else:
                separability += distances[i][1]
                flag += 1

        separability /= neighbour

        # 判断一下 compactness 的值是否为 0，为了防止分母为 0 的情况出现，加了一个值非常小的参数 phi = 0.0001
        if compactness == 0:
            salient_value = separability / phi
        else:
            salient_value = separability / compactness

        return salient_value


    def get_distances_in_attribute(self, instances, x, index):
        distances = {}

        for i, ins in enumerate(instances):
            distances[i] = abs(x.values[index] - ins.values[index])

        # 对 distances 进行按 value 排序
        sorted_distances = sorted(distances.items(), key=lambda x: x[1])

        return sorted_distances

    def get_description_for_class(self, descriptions, cluster_result, weight_matrix):
        class_description = {}
        print("weight_matrix type:", type(weight_matrix))
        print("weight_matrix shape:", weight_matrix.shape)

        for cn, in_class_samples in cluster_result.items():
            # 类内样本
            in_class_samples_values = list(cluster_result[cn].values())  # 获取样本的值
            print("in_class_samples_values:", in_class_samples_values)

            # 类外样本
            out_of_class = []
            for cnn, samples in cluster_result.items():
                if cnn != cn:
                    out_of_class.extend(samples)
            print("out_of_class:", out_of_class)

            # 找到评价指标D_c_i最大值样本的描述
            max_index_D = None
            max_average = float("-inf")
            max_index_F = None
            max_degree = float("-inf")

            for s in range(len(descriptions)):
                first_key = next(iter(in_class_samples))

                in_average = sum(weight_matrix[s, int(ss)] for ss in in_class_samples if ss.isdigit()) / len(
                    in_class_samples) if in_class_samples else 0

                out_average = sum(weight_matrix[s, int(ss)] for ss in out_of_class if ss.isdigit()) / len(
                    out_of_class) if out_of_class else 0

                if (in_average - out_average) > max_average:
                    max_average = in_average - out_average
                    max_index_D = s

                in_degrees = [weight_matrix[s, int(tt)] for tt in in_class_samples if tt.isdigit()]
                out_degrees = [weight_matrix[s, int(ttt)] for ttt in out_of_class if ttt.isdigit()]

                in_degree = max(in_degrees) if in_degrees else 0
                out_degree = max(out_degrees) if out_degrees else 0

                if (in_degree - out_degree) > max_degree:
                    max_degree = in_degree - out_degree
                    max_index_F = s

            and_description = [max_index_D, max_index_F]

            class_description[cn] = and_description

        return class_description

    def get_membership_function(self, class_description, instances, weight_matrix):
        degrees = np.zeros(instances.numAttributes())

        # 遍历所有样本
        for i, instance in enumerate(instances):
            max_degree = float('-inf')

            # 遍历所有类别描述
            for concept_description in class_description:
                degree = self.get_membership_degree(weight_matrix, concept_description, i)

                if degree > max_degree:
                    max_degree = degree

            degrees[i] = max_degree

        return degrees

    def print_membership_function(self, class_description, instances, weight_matrix):
        for class_number, concept_descriptions in class_description.items():
            print(f"第 {class_number} 类的隶属函数 =================================================================")
            degrees = self.get_membership_function(concept_descriptions, instances, weight_matrix)

            for degree in degrees:
                print(degree)



    def get_membership_function_for_class(self, class_description, instances, weight_matrix):
        num_classes = len(class_description)
        num_instances = instances.num_instances
        membership_function = np.zeros((num_classes, num_instances))

        for class_number, concept_descriptions in class_description.items():
            degrees = self.get_membership_function(concept_descriptions, instances, weight_matrix)
            membership_function[class_number] = degrees

        return membership_function

    def get_class_label_by_membership_degree(self, membership_function):
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

    def statistic_cluster_result(self, instances, class_label):
        print("Length of instances:", len(instances.data))
        print("Length of class_label:", len(class_label))
        print("Data type of 'instances':", type(instances.data))
        num_classes = len(np.unique(class_label))  # 加上类属性
        print("Number of unique classes:", num_classes)

        labels = [[] for _ in range(num_classes)]
        cluster_result = {}

        class_index = 0
        for i in range(len(class_label) - 1):
            if class_label[i + 1] == class_label[i]:
                labels[class_index].append(class_label[i])
            else:
                labels[class_index].append(class_label[i])
                class_index += 1
                print("Updated class index:", class_index)
                while len(labels) <= class_index:
                    labels.append([])

        # 处理最后一个数据点
        labels[class_index].append(class_label[-1])

        print("Final class index:", class_index)
        print("Length of labels list:", len(labels))

        for i in range(len(labels)):
            frequency = Counter(labels[i])
            sorted_frequency = sorted(frequency.items(), key=itemgetter(1), reverse=True)
            cluster_map = {"right": sorted_frequency[0][1], "wrong": sum(value for key, value in sorted_frequency[1:])}
            cluster_result[i] = cluster_map
        print("Cluster Result:", cluster_result)

        return cluster_result

    def calculate_accuracy(self, result):
        print("==========输出聚类结果==========")
        accuracy = 0
        wrong = 0
        right = 0

        for class_index, r in result.items():  # 迭代字典的键值对
            rt = r["right"]
            rw = r["wrong"]
            right += rt
            wrong += rw
            print(f"==========第 {class_index} 类共有 {rw + rt} 个==========")
            print(f"其中正确 {rt} 个, 错误 {rw} 个")

        accuracy = right / (right + wrong)
        print(f"准确率为 {accuracy}")

