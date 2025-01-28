import numpy as np
import copy
import pandas as pd
import matplotlib.pyplot as plt

# pairwise修正逻辑：对每个客户端 ，检查他和其他客户端的cossim如果<0，就向该客户端做投影，直到所有客户端都校准一次,不进行迭代。
def aggregate_pairwise_vertical(self ,sampled_client_indices, coeff, eps=0.001):
    def gradient_calibration(gradient, project_target):
        projection = np.dot(gradient, project_target) / np.linalg.norm(project_target) ** 2
        gradient = gradient - projection * project_target
        return gradient
    # get all gredients
    gradients = []
    for i in sampled_client_indices:
        gradient = self.clients[i].get_gradient()
        gradients.append(gradient)
    gradients = np.array(gradients)
    total_conflicts =[]
    for i, g in enumerate(gradients):
        # conflicts=[]
        for j ,h in enumerate(gradients):
            cos_sim = np.dot(h, g) / (np.linalg.norm(h) * np.linalg.norm(g))
            if cos_sim <0:
                # print('conflict pair %s and %s cossim %s' %(i,j,cos_sim))
                calibrated_gradient = gradient_calibration(g, h)
                gradients[i] = calibrated_gradient
                total_conflicts.append(h)
    print(' %s conficts' % (len(total_conflicts)))
    sum_of_gradient = np.sum(gradients, axis=0) / len(gradients)
    new_model = copy.copy(self.model)
    new_model.to('cpu')
    new_weights = new_model.state_dict()
    global_gradient = self.model.unflatten_model(sum_of_gradient)
    for key in new_model.state_dict().keys():
        new_weights[key] = new_weights[key] - 1 * global_gradient[key]

    new_model.load_state_dict(new_weights)
    return new_model


def aggregate_pairwise_vertical_cossim3(self, sampled_client_indices, coeff, eps=0.001):
    def gradient_calibration(gradient, project_target):
        projection = np.dot(gradient, project_target) / np.linalg.norm(project_target) ** 2
        gradient = gradient - projection * project_target
        return gradient

    # get all gredients
    gradients = []
    for i in sampled_client_indices:
        gradient = self.clients[i].get_gradient()
        gradients.append(gradient)
    gradients = np.array(gradients)
    total_conflicts = []
    for i, g in enumerate(gradients):
        # conflicts=[]
        for j, h in enumerate(gradients):
            cos_sim = np.dot(h, g) / (np.linalg.norm(h) * np.linalg.norm(g))
            if cos_sim < -0.3:
                # print('conflict pair %s and %s cossim %s' %(i,j,cos_sim))
                calibrated_gradient = gradient_calibration(g, h)
                gradients[i] = calibrated_gradient
                total_conflicts.append(h)
    print(' %s conficts' % (len(total_conflicts)))
    sum_of_gradient = np.sum(gradients, axis=0) / len(gradients)
    new_model = copy.copy(self.model)
    new_model.to('cpu')
    new_weights = new_model.state_dict()
    global_gradient = self.model.unflatten_model(sum_of_gradient)
    for key in new_model.state_dict().keys():
        new_weights[key] = new_weights[key] - 1 * global_gradient[key]

    new_model.load_state_dict(new_weights)
    return new_model


def aggregate_pairwise_vertical_cossim1(self, sampled_client_indices, coeff, eps=0.001):
    def gradient_calibration(gradient, project_target):
        projection = np.dot(gradient, project_target) / np.linalg.norm(project_target) ** 2
        gradient = gradient - projection * project_target
        return gradient

    # get all gredients
    gradients = []
    for i in sampled_client_indices:
        gradient = self.clients[i].get_gradient()
        gradients.append(gradient)
    gradients = np.array(gradients)
    total_conflicts = []
    for i, g in enumerate(gradients):
        # conflicts=[]
        for j, h in enumerate(gradients):
            cos_sim = np.dot(h, g) / (np.linalg.norm(h) * np.linalg.norm(g))
            if cos_sim < -0.1:
                # print('conflict pair %s and %s cossim %s' %(i,j,cos_sim))
                calibrated_gradient = gradient_calibration(g, h)
                gradients[i] = calibrated_gradient
                total_conflicts.append(h)
    print(' %s conficts' % (len(total_conflicts)))
    sum_of_gradient = np.sum(gradients, axis=0) / len(gradients)
    new_model = copy.copy(self.model)
    new_model.to('cpu')
    new_weights = new_model.state_dict()
    global_gradient = self.model.unflatten_model(sum_of_gradient)
    for key in new_model.state_dict().keys():
        new_weights[key] = new_weights[key] - 1 * global_gradient[key]

    new_model.load_state_dict(new_weights)
    return new_model


def aggregate_pairwise_vertical_both(self, sampled_client_indices, coeff, eps=0.001):
    def gradient_calibration(gradient, project_target):
        projection = np.dot(gradient, project_target) / np.linalg.norm(project_target) ** 2
        gradient = gradient - projection * project_target
        return gradient

    # get all gredients
    gradients = []
    for i in sampled_client_indices:
        gradient = self.clients[i].get_gradient()
        gradients.append(gradient)
    gradients = np.array(gradients)
    for i, g in enumerate(gradients):
        for j, h in enumerate(gradients):
            cos_sim = np.dot(h, g) / (np.linalg.norm(h) * np.linalg.norm(g))
            if cos_sim < 0:
                print('conflict pair %s and %s cossim %s' % (i, j, cos_sim))
                calibrated_gradient_i = gradient_calibration(g, h)
                calibrated_gradient_j = gradient_calibration(h, g)
                gradients[i] = calibrated_gradient_i
                gradients[j] = calibrated_gradient_j
    sum_of_gradient = np.sum(gradients, axis=0) / len(gradients)
    new_model = copy.copy(self.model)
    new_model.to('cpu')
    new_weights = new_model.state_dict()
    global_gradient = self.model.unflatten_model(sum_of_gradient)
    for key in new_model.state_dict().keys():
        new_weights[key] = new_weights[key] - 1 * global_gradient[key]

    new_model.load_state_dict(new_weights)
    return new_model


# def aggregate_pairwise_vertical_both(self, sampled_client_indices, coeff, eps=0.001):
#     def gradient_calibration(gradient, project_target):
#         projection = np.dot(gradient, project_target) / np.linalg.norm(project_target) ** 2
#         gradient = gradient - projection * project_target
#         return gradient
#
#     # get all gredients
#     gradients = []
#     for i in sampled_client_indices:
#         gradient = self.clients[i].get_gradient()
#         gradients.append(gradient)
#     gradients = np.array(gradients)
#     for i, g in enumerate(gradients):
#         for j, h in enumerate(gradients):
#             cos_sim = np.dot(h, g) / (np.linalg.norm(h) * np.linalg.norm(g))
#             if cos_sim < 0:
#                 print('conflict pair %s and %s cossim %s' % (i, j, cos_sim))
#                 calibrated_gradient_i = gradient_calibration(g, h)
#                 calibrated_gradient_j = gradient_calibration(h, g)
#                 gradients[i] = calibrated_gradient_i
#                 gradients[j] = calibrated_gradient_j
#     sum_of_gradient = np.sum(gradients, axis=0) / len(gradients)
#     new_model = copy.copy(self.model)
#     new_model.to('cpu')
#     new_weights = new_model.state_dict()
#     global_gradient = self.model.unflatten_model(sum_of_gradient)
#     for key in new_model.state_dict().keys():
#         new_weights[key] = new_weights[key] - 1 * global_gradient[key]
#
#     new_model.load_state_dict(new_weights)
#     return new_model

# def aggregate_pairwise_vertical_both_cossim1(self, sampled_client_indices, coeff, eps=0.001):
#     def gradient_calibration(gradient, project_target):
#         projection = np.dot(gradient, project_target) / np.linalg.norm(project_target) ** 2
#         gradient = gradient - projection * project_target
#         return gradient
#
#     # get all gredients
#     gradients = []
#     for i in sampled_client_indices:
#         gradient = self.clients[i].get_gradient()
#         gradients.append(gradient)
#     gradients = np.array(gradients)
#     for i, g in enumerate(gradients):
#         for j, h in enumerate(gradients):
#             cos_sim = np.dot(h, g) / (np.linalg.norm(h) * np.linalg.norm(g))
#             if cos_sim < 0:
#                 print('conflict pair %s and %s cossim %s' % (i, j, cos_sim))
#                 calibrated_gradient_i = gradient_calibration(g, h)
#                 calibrated_gradient_j = gradient_calibration(h, g)
#                 gradients[i] = calibrated_gradient_i
#                 gradients[j] = calibrated_gradient_j
#     sum_of_gradient = np.sum(gradients, axis=0) / len(gradients)
#     new_model = copy.copy(self.model)
#     new_model.to('cpu')
#     new_weights = new_model.state_dict()
#     global_gradient = self.model.unflatten_model(sum_of_gradient)
#     for key in new_model.state_dict().keys():
#         new_weights[key] = new_weights[key] - 1 * global_gradient[key]
#
#     new_model.load_state_dict(new_weights)
#     return new_model

def aggregate_pairwise_vertical_both_cossim1(self, sampled_client_indices, coeff, eps=0.001):
    def gradient_calibration(gradient, project_target):
        projection = np.dot(gradient, project_target) / np.linalg.norm(project_target) ** 2
        gradient = gradient - projection * project_target
        return gradient

    # get all gredients
    gradients = []
    for i in sampled_client_indices:
        gradient = self.clients[i].get_gradient()
        gradients.append(gradient)
    gradients = np.array(gradients)
    for i, g in enumerate(gradients):
        for j, h in enumerate(gradients):
            cos_sim = np.dot(h, g) / (np.linalg.norm(h) * np.linalg.norm(g))
            if cos_sim < -0.1:
                # print('conflict pair %s and %s cossim %s' % (i, j, cos_sim))
                calibrated_gradient_i = gradient_calibration(g, h)
                calibrated_gradient_j = gradient_calibration(h, g)
                gradients[i] = calibrated_gradient_i
                gradients[j] = calibrated_gradient_j
    sum_of_gradient = np.sum(gradients, axis=0) / len(gradients)
    new_model = copy.copy(self.model)
    new_model.to('cpu')
    new_weights = new_model.state_dict()
    global_gradient = self.model.unflatten_model(sum_of_gradient)
    for key in new_model.state_dict().keys():
        new_weights[key] = new_weights[key] - 1 * global_gradient[key]

    new_model.load_state_dict(new_weights)
    return new_model


def aggregate_pairwise_vertical_conflicts_avg(self, sampled_client_indices, coeff, eps=0.001):
    def gradient_calibration(gradient, project_target):
        projection = np.dot(gradient, project_target) / np.linalg.norm(project_target) ** 2
        gradient = gradient - projection * project_target
        return gradient

    # get all gredients
    gradients = []
    for i in sampled_client_indices:
        gradient = self.clients[i].get_gradient()
        gradients.append(gradient)
    gradients = np.array(gradients)
    for i, g in enumerate(gradients):
        conflicts = []
        for j, h in enumerate(gradients):
            cos_sim = np.dot(h, g) / (np.linalg.norm(h) * np.linalg.norm(g))
            if cos_sim < 0:
                conflicts.append(h)
                # print('conflict pair %s and %s cossim %s' % (i, j, cos_sim))
                # calibrated_gradient_i = gradient_calibration(g, h)
                # calibrated_gradient_j = gradient_calibration(h, g)
                # gradients[i] = calibrated_gradient_i
                # gradients[j] = calibrated_gradient_j
        if len(conflicts) > 0:
            print('gradient %s has %s conficts' % (i, len(conflicts)))
            sum_of_conflicts = np.sum(conflicts, axis=0) / len(conflicts)
            calibrated_gradient = gradient_calibration(g, sum_of_conflicts)
            gradients[i] = calibrated_gradient
    sum_of_gradient = np.sum(gradients, axis=0) / len(gradients)
    new_model = copy.copy(self.model)
    new_model.to('cpu')
    new_weights = new_model.state_dict()
    global_gradient = self.model.unflatten_model(sum_of_gradient)
    for key in new_model.state_dict().keys():
        new_weights[key] = new_weights[key] - 1 * global_gradient[key]

    new_model.load_state_dict(new_weights)
    return new_model


def aggregate_pairwise_vertical_conflicts_avg_cossim1(self, sampled_client_indices, coeff, eps=0.001):
    th = 0.95 ** int(self._round / 10)
    print('cossim TH is ', th)

    def gradient_calibration(gradient, project_target):
        projection = np.dot(gradient, project_target) / np.linalg.norm(project_target) ** 2
        gradient = gradient - projection * project_target
        return gradient

    # get all gredients
    gradients = []
    for i in sampled_client_indices:
        gradient = self.clients[i].get_gradient()
        gradients.append(gradient)
    gradients = np.array(gradients)
    for i, g in enumerate(gradients):
        conflicts = []
        for j, h in enumerate(gradients):
            cos_sim = np.dot(h, g) / (np.linalg.norm(h) * np.linalg.norm(g))
            if cos_sim < -0.3 * th:
                conflicts.append(h)
        if len(conflicts) > 0:
            #                print('gradient %s has %s conficts' % (i, len(conflicts)))
            sum_of_conflicts = np.sum(conflicts, axis=0) / len(conflicts)
            calibrated_gradient = gradient_calibration(g, sum_of_conflicts)
            gradients[i] = calibrated_gradient
    sum_of_gradient = np.sum(gradients, axis=0) / len(gradients)
    new_model = copy.copy(self.model)
    new_model.to('cpu')
    new_weights = new_model.state_dict()
    global_gradient = self.model.unflatten_model(sum_of_gradient)
    for key in new_model.state_dict().keys():
        new_weights[key] = new_weights[key] - 1 * global_gradient[key]

    new_model.load_state_dict(new_weights)
    return new_model


def vertical_conflicts_avg_cossim1_iteration(self, sampled_client_indices, coeff, eps=0.01):
    th = 0.95 ** int(self._round / 10)
    print('cossim TH is ', th)

    def gradient_calibration(gradient, project_target):
        projection = np.dot(gradient, project_target) / np.linalg.norm(project_target) ** 2
        gradient = gradient - projection * project_target
        return gradient

    # get all gredients
    gradients = []
    for i in sampled_client_indices:
        gradient = self.clients[i].get_gradient()
        gradients.append(gradient)
    gradients = np.array(gradients)
    sum_of_gradient = np.sum(gradients, axis=0) / len(gradients)
    # sop=sum_of_gradient
    cache = None
    if cache is None:
        cache = sum_of_gradient - 2 * eps
    calibration_iteration = 0
    while np.linalg.norm(sum_of_gradient - cache) > eps:
        calibration_iteration += 1
        print("calibration_iteration %s" % calibration_iteration)
        for i, g in enumerate(gradients):
            conflicts = []
            for j, h in enumerate(gradients):
                cos_sim = np.dot(h, g) / (np.linalg.norm(h) * np.linalg.norm(g))
                if cos_sim < -0.3 * th:
                    conflicts.append(h)
            if len(conflicts) > 0:
                print('gradient %s has %s conficts' % (i, len(conflicts)))
                sum_of_conflicts = np.sum(conflicts, axis=0) / len(conflicts)
                calibrated_gradient = gradient_calibration(g, sum_of_conflicts)
                gradients[i] = calibrated_gradient
        cache = sum_of_gradient
        sum_of_gradient = np.sum(gradients, axis=0) / len(gradients)
        print("delta of global gradient after calibration is %s" % np.linalg.norm(sum_of_gradient - cache))
    new_model = copy.copy(self.model)
    new_model.to('cpu')
    new_weights = new_model.state_dict()
    global_gradient = self.model.unflatten_model(sum_of_gradient)
    for key in new_model.state_dict().keys():
        new_weights[key] = new_weights[key] - 1 * global_gradient[key]

    new_model.load_state_dict(new_weights)
    return new_model


def vertical_conflicts_avg_cossim1_iteration_dynamic_projection(self, sampled_client_indices, coeff, eps=0.03):
    # print('projection weight is', 0.8**int(self._round / 50))
    th = 1.2 ** int(self._round / 50)
    print(' TH is ', th, 'cossim is', -0.3 * th)

    def gradient_calibration(gradient, project_target):
        projection = np.dot(gradient, project_target) / np.linalg.norm(project_target) ** 2
        gradient = gradient - (1 ** int(self._round / 50)) * projection * project_target
        return gradient

    # get all gredients
    gradients = []
    for i in sampled_client_indices:
        gradient = self.clients[i].get_gradient()
        gradients.append(gradient)
    gradients = np.array(gradients)
    sum_of_gradient = np.sum(gradients, axis=0) / len(gradients)
    # sop=sum_of_gradient
    cache = None
    if cache is None:
        cache = sum_of_gradient - 2 * eps
    calibration_iteration = 0
    while np.linalg.norm(sum_of_gradient - cache) > eps:
        calibration_iteration += 1
        # print("calibration_iteration %s" %calibration_iteration)
        for i, g in enumerate(gradients):
            conflicts = []
            for j, h in enumerate(gradients):
                cos_sim = np.dot(h, g) / (np.linalg.norm(h) * np.linalg.norm(g))
                if cos_sim < -0.3 * th:
                    conflicts.append(h)
            if len(conflicts) > 0:
                # print('gradient %s has %s conficts' % (i, len(conflicts)))
                sum_of_conflicts = np.sum(conflicts, axis=0) / len(conflicts)
                calibrated_gradient = gradient_calibration(g, sum_of_conflicts)
                gradients[i] = calibrated_gradient
        cache = sum_of_gradient
        sum_of_gradient = np.sum(gradients, axis=0) / len(gradients)
        # print("delta of global gradient after calibration is %s" %np.linalg.norm(sum_of_gradient - cache))
    new_model = copy.copy(self.model)
    new_model.to('cpu')
    new_weights = new_model.state_dict()
    global_gradient = self.model.unflatten_model(sum_of_gradient)
    for key in new_model.state_dict().keys():
        new_weights[key] = new_weights[key] - 1 * global_gradient[key]

    new_model.load_state_dict(new_weights)
    return new_model


def pairwise_vaccine(self, sampled_client_indices, coeff, eps=0.03):
    # 用VACCINE方法 判断梯度pair的相似度变化，如果变得差异更大了，就修正。所以是自适应的，没啥超参数了/。
    # 在server端，维护一个客户端cossim的表 cos_dict,缓存历史相似度。对应VACCINE的phiT.
    # 每一次当前cissim和缓存dict cossim比较 如果变得差异更大了（值更小了）就校准，公式按照VACCINE
    # 无论如何都更新cossim。

    # print('projection weight is', 0.8**int(self._round / 50))
    # th = 1.2 ** int(self._round / 50)
    # print(' TH is ', th,'cossim is', -0.3*th)
    # def gradient_calibration(gradient, project_target):
    #     projection = np.dot(gradient, project_target) / np.linalg.norm(project_target)
    #     gradient = gradient - projection * project_target
    #     return gradient
    def gradient_calibration_vaccine(gradient, project_target, i, j, cossim):
        # projection = np.dot(gradient, project_target) / np.linalg.norm(project_target) ** 2
        projection = (np.linalg.norm(gradient) * self.cos_dict[i, j] * (1 - cossim ** 2) ** 0.5 - cossim * (
                    1 - self.cos_dict[i, j] ** 2) ** 0.5) / np.linalg.norm(project_target) * (
                                 1 - self.cos_dict[i, j] ** 2) ** 0.5
        gradient = gradient + projection * project_target
        print('projection is %s' % projection)
        return gradient

    # get all gredients
    gradients = []
    for i in sampled_client_indices:
        gradient = self.clients[i].get_gradient()
        gradients.append(gradient)
    gradients = np.array(gradients)
    # sum_of_gradient=np.sum(gradients, axis=0) / len(gradients)
    # sop=sum_of_gradient
    # cache = None
    # if cache is None:
    #    cache = sum_of_gradient - 2 * eps
    # calibration_iteration=0
    # while np.linalg.norm(sum_of_gradient - cache) > eps:
    #     calibration_iteration+=1
    # print("calibration_iteration %s" %calibration_iteration)
    # df = pd.DataFrame(data=self.cos_dict[0:, 0:])
    for i, g in enumerate(gradients):
        conflicts = []
        conflict_count = 0
        for j, h in enumerate(gradients):
            cos_sim = np.dot(h, g) / (np.linalg.norm(h) * np.linalg.norm(g))
            if cos_sim < self.cos_dict[i, j]:
                # conflicts.append(h)
                calibrated_gradient = gradient_calibration_vaccine(g, h, i, j, cos_sim)
                gradients[i] = calibrated_gradient
            self.cos_dict[i, j] = 0.3 * cos_sim + 0.7 * self.cos_dict[i, j]
            if cos_sim < 0:
                conflict_count += 1
        # if conflict_count >0:
        #     print('conflict_count of client %s is %s,'%(i,conflict_count))
        # if len(conflicts)>0:
        #     # print('gradient %s has %s conficts' % (i, len(conflicts)))
        #     #sum_of_conflicts = np.sum(conflicts, axis=0) / len(conflicts)
        #     calibrated_gradient = gradient_calibration(g, sum_of_conflicts,)
        # gradients[i]=calibrated_gradient
    # cache = sum_of_gradient
    sum_of_gradient = np.sum(gradients, axis=0) / len(gradients)
    # print("delta of global gradient after calibration is %s" %np.linalg.norm(sum_of_gradient - cache))
    new_model = copy.copy(self.model)
    new_model.to('cpu')
    new_weights = new_model.state_dict()
    global_gradient = self.model.unflatten_model(sum_of_gradient)
    if self._round % 30 == 0:
        # print(self.cos_dict)
        df = pd.DataFrame(data=self.cos_dict[0:, 0:])
        self.df_sum = self.df_sum + df
        # print('average heatmap is', self.df_sum)
        # print(df)
        # hm = sns.heatmap(df,vmin=-1,vmax=1)
        # plt.savefig('./heatmap/pic-class{}-step{}.png'.format(config.NUM_SELECTED_CLASS, self._round))
        # plt.show()
        cossim_snapshot = df.mean()
        cossim_snapshot_avg = np.mean(cossim_snapshot)
        # print('cossim_snapshot is' ,cossim_snapshot)
        print('cossim_snapshot mean is', cossim_snapshot_avg)
        print('conflict account', conflict_count)
    for key in new_model.state_dict().keys():
        new_weights[key] = new_weights[key] - 1 * global_gradient[key]
    new_model.load_state_dict(new_weights)
    # if self._round%1==0:
    #     # print(self.cos_dict)
    #     df=pd.DataFrame(data=self.cos_dict[0:,0:])
    #     self.df_sum = self.df_sum + df
    #     # print('average heatmap is', self.df_sum)
    #     # print(df)
    #     hm = sns.heatmap(df,vmin=-1, vmax=1,cmqp='reds')
    #     # colors=sns.diverging_palette(200, 20, sep=10,n=20 ,as_cmap=True)
    #     plt.savefig('./heatmap/pic-class{}-step{}.png'.format(config.NUM_SELECTED_CLASS, self._round))
    #     plt.show()

    # columns=
    # print(df)
    if self._round == config.NUM_ROUNDS:
        # print(self._round )
        # print('average heatmap is',self.df_sum/(config.NUM_ROUNDS/1))
        hm = sns.heatmap(self.df_sum / (config.NUM_ROUNDS / 20), vmin=0, vmax=1)
        plt.savefig('./heatmap/pic-class{}-final.png'.format(config.NUM_SELECTED_CLASS))
        plt.show()

    return new_model


def pairwise_vaccine_2(self, sampled_client_indices, coeff, eps=0.03):
    # 个体逐渐远离则校准，顺序是随机的。改成从夹角大的投影到夹角小的
    def gradient_calibration_vaccine(gradient, project_target, i, j, cossim):
        # projection = np.dot(gradient, project_target) / np.linalg.norm(project_target) ** 2
        projection = (np.linalg.norm(gradient) * (self.cos_dict[i, j] * (1 - cossim ** 2) ** 0.5 - cossim * (
                    1 - self.cos_dict[i, j] ** 2) ** 0.5)) / np.linalg.norm(project_target) * (
                                 1 - self.cos_dict[i, j] ** 2) ** 0.5
        print('projection is %s' % projection)
        gradient = gradient + projection * project_target
        return gradient

    # get all gredients
    gradients = []
    for i in sampled_client_indices:
        gradient = self.clients[i].get_gradient()
        gradients.append(gradient)
    gradients = np.array(gradients)
    for i, g in enumerate(gradients):
        # 待投影
        conflicts = []
        conflict_count = 0
        for j, h in enumerate(gradients):
            # 投影目标
            cos_sim = np.dot(h, gradients[i]) / (np.linalg.norm(h) * np.linalg.norm(gradients[i]))
            if cos_sim < self.cos_dict[i, j]:
                print('cos_sim before is %s' % cos_sim)
                # conflicts.append(h)
                calibrated_gradient = gradient_calibration_vaccine(gradients[i], h, i, j, cos_sim)
                # 将h校准到g上
                gradients[i] = calibrated_gradient
                print('cos_sim after is %s' % (
                            np.dot(h, gradients[i]) / (np.linalg.norm(h) * np.linalg.norm(gradients[i]))))
            self.cos_dict[i, j] = max(0.3 * cos_sim + 0.7 * self.cos_dict[i, j], 0)
            if cos_sim < 0:
                conflict_count += 1
        # if conflict_count >0:
        #     print('conflict_count of client %s is %s,'%(i,conflict_count))
        # if len(conflicts)>0:
        #     # print('gradient %s has %s conficts' % (i, len(conflicts)))
        #     #sum_of_conflicts = np.sum(conflicts, axis=0) / len(conflicts)
        #     calibrated_gradient = gradient_calibration(g, sum_of_conflicts,)
        # gradients[i]=calibrated_gradient
    # cache = sum_of_gradient
    sum_of_gradient = np.sum(gradients, axis=0) / len(gradients)
    # print("delta of global gradient after calibration is %s" %np.linalg.norm(sum_of_gradient - cache))
    new_model = copy.copy(self.model)
    new_model.to('cpu')
    new_weights = new_model.state_dict()
    global_gradient = self.model.unflatten_model(sum_of_gradient)
    if self._round % 30 == 0:
        # print(self.cos_dict)
        df = pd.DataFrame(data=self.cos_dict[0:, 0:])
        self.df_sum = self.df_sum + df
        # print('average heatmap is', self.df_sum)
        # print(df)
        # hm = sns.heatmap(df,vmin=-1,vmax=1)
        # plt.savefig('./heatmap/pic-class{}-step{}.png'.format(config.NUM_SELECTED_CLASS, self._round))
        # plt.show()
        cossim_snapshot = df.mean()
        cossim_snapshot_avg = np.mean(cossim_snapshot)
        # print('cossim_snapshot is' ,cossim_snapshot)
        print('cossim_snapshot mean is', cossim_snapshot_avg)
        print('conflict account', conflict_count)
    for key in new_model.state_dict().keys():
        new_weights[key] = new_weights[key] - 1 * global_gradient[key]
    new_model.load_state_dict(new_weights)

    return new_model


def pairwise_vaccine_3(self, sampled_client_indices, coeff, eps=0.03):
    print('联邦服务端聚合的客户端包括', sampled_client_indices)

    # print('服务端要聚合的客户端包括 %s' % sampled_client_indices)
    # 个体逐渐远离则校准，顺序是随机的。改成从夹角大的投影到夹角小的
    def gradient_calibration_vaccine(gradient, project_target, i, j, cossim):
        # projection = np.dot(gradient, project_target) / np.linalg.norm(project_target) ** 2
        projection = (np.linalg.norm(gradient) * (self.cos_dict[i, j] * (1 - cossim ** 2) ** 0.5 - cossim * (
                    1 - self.cos_dict[i, j] ** 2) ** 0.5)) / np.linalg.norm(project_target) * (
                                 1 - self.cos_dict[i, j] ** 2) ** 0.5
        # print('projection is %s' %projection)
        gradient = gradient + projection * project_target
        return gradient

    # get all gredients
    gradients = []
    for i in sampled_client_indices:
        print('服务端遍历的i是', i)
        gradient = self.clients[i].get_gradient()
        gradients.append(gradient)
    gradients = np.array(gradients)
    for i, g in enumerate(gradients):
        weights_new = 0
        for j, h in enumerate(gradients):
            # 投影目标
            cos_sim = np.dot(h, gradients[i]) / (np.linalg.norm(h) * np.linalg.norm(gradients[i]))
            # print('客户端%s和%s的相似度是%s'%(i,j,cos_sim))
            if cos_sim < self.cos_dict[i, j] or cos_sim < 0:
                # print('cos_sim before is %s' % cos_sim)
                # conflicts.append(h)
                calibrated_gradient = gradient_calibration_vaccine(gradients[i], h, i, j, cos_sim)
                # 每一轮，冲突的客户端多，冲突程度大，则减少选择概率。如果防线渐远，则减小概率.概率用0.9的X次方，如果值大概率低。
                if cos_sim <= 0:
                    # print('update client_weights because cos<0 %s' %i)
                    weights_new = weights_new - cos_sim
                elif cos_sim < self.cos_dict[i, j]:
                    # self.client_weights[i] = self.client_weights[i] + (self.cos_dict[i, j]-cos_sim)
                    weights_new = weights_new + (self.cos_dict[i, j] - cos_sim)
                    # print('update client_weights because similarity decreasing %s' % i)
                gradients[i] = calibrated_gradient
            self.cos_dict[i, j] = max(0.3 * cos_sim + 0.7 * self.cos_dict[i, j], 0)
        print('将客户端%s的权重从%s更新到%s' % (
        i, self.client_weights[i], (0.5 * self.client_weights[i] + 0.5 * weights_new)))
        self.client_weights[i] = 0.5 * self.client_weights[i] + 0.5 * weights_new
    # cache = sum_of_gradient
    # print('clients weights is %s' %self.client_weights)
    sum_of_gradient = np.sum(gradients, axis=0) / len(gradients)
    # print("delta of global gradient after calibration is %s" %np.linalg.norm(sum_of_gradient - cache))
    new_model = copy.copy(self.model)
    new_model.to('cpu')
    new_weights = new_model.state_dict()
    global_gradient = self.model.unflatten_model(sum_of_gradient)
    for key in new_model.state_dict().keys():
        new_weights[key] = new_weights[key] - 1 * global_gradient[key]
    new_model.load_state_dict(new_weights)

    return new_model


def pairwise_vaccine_4(self, sampled_client_indices, coeff, eps=0.03):
    # 个体逐渐远离则校准，顺序是随机的。改成从夹角大的投影到夹角小的
    def gradient_calibration_vaccine(gradient, project_target, i, j, cossim):
        # projection = np.dot(gradient, project_target) / np.linalg.norm(project_target) ** 2
        projection = (np.linalg.norm(gradient) * (self.cos_dict[i, j] * (1 - cossim ** 2) ** 0.5 - cossim * (
                1 - self.cos_dict[i, j] ** 2) ** 0.5)) / np.linalg.norm(project_target) * (
                             1 - self.cos_dict[i, j] ** 2) ** 0.5
        # print('projection is %s' %projection)
        gradient = gradient + projection * project_target
        return gradient

    # get all gredients
    gradients = []
    for i in sampled_client_indices:
        gradient = self.clients[i].get_gradient()
        gradients.append(gradient)
    gradients = np.array(gradients)
    for i, g in enumerate(gradients):
        # 待投影
        conflicts = []
        conflict_count = 0
        for j, h in enumerate(gradients):
            # 投影目标
            cos_sim = np.dot(h, gradients[i]) / (np.linalg.norm(h) * np.linalg.norm(gradients[i]))

            # 如果冲突，则放在一起先排序
            if cos_sim < self.cos_dict[i, j] or cos_sim < 0:
                print('cos_sim %s, %s before calibration is %s' % (i, j, cos_sim))
                # print('j is ', j)
                conflicts.append([cos_sim, j, h])
                # print('length conflicts is', len(conflicts))
        if conflicts:
            conflicts.sort()
            print('length conflicts is', len(conflicts))
            # print('sorted conflicts is',conflicts)
            for p, q, y in conflicts:
                calibrated_gradient = gradient_calibration_vaccine(gradients[i], y, i, q, p)
                gradients[i] = calibrated_gradient
                calibrated_cos_sim = np.dot(y, gradients[i]) / (np.linalg.norm(y) * np.linalg.norm(gradients[i]))
                # print('cos_sim after calibration is', calibrated_cos_sim)
                print('cos_sim %s, %s after calibration is %s' % (i, q, calibrated_cos_sim))
                self.cos_dict[i, j] = max(0.2 * calibrated_cos_sim + 0.8 * self.cos_dict[i, j], 0)
                # print(self.cos_dict)
    sum_of_gradient = np.sum(gradients, axis=0) / len(gradients)
    new_model = copy.copy(self.model)
    new_model.to('cpu')
    new_weights = new_model.state_dict()
    global_gradient = self.model.unflatten_model(sum_of_gradient)
    for key in new_model.state_dict().keys():
        new_weights[key] = new_weights[key] - 1 * global_gradient[key]
    new_model.load_state_dict(new_weights)

    return new_model


def pairwise_vaccine_delete_conflits(self, sampled_client_indices, coeff, eps=0.03):
    # 个体逐渐远离则校准，顺序是随机的。改成从夹角大的投影到夹角小的。
    #
    def gradient_calibration_vaccine(gradient, project_target, i, j, cossim):
        # projection = np.dot(gradient, project_target) / np.linalg.norm(project_target) ** 2
        projection = (np.linalg.norm(gradient) * (self.cos_dict[i, j] * (1 - cossim ** 2) ** 0.5 - cossim * (
                    1 - self.cos_dict[i, j] ** 2) ** 0.5)) / np.linalg.norm(project_target) * (
                                 1 - self.cos_dict[i, j] ** 2) ** 0.5
        # print('projection is %s' %projection)
        gradient = gradient + projection * project_target
        return gradient

    # get all gredients
    gradients = []
    for i in sampled_client_indices:
        gradient = self.clients[i].get_gradient()
        gradients.append(gradient)
    gradients = np.array(gradients)
    for i, g in enumerate(gradients):
        # 待投影
        conflicts = []
        conflict_count = 0
        for j, h in enumerate(gradients):
            # 投影目标
            cos_sim = np.dot(h, gradients[i]) / (np.linalg.norm(h) * np.linalg.norm(gradients[i]))
            if cos_sim < self.cos_dict[i, j] or cos_sim < 0:
                # print('cos_sim before is %s' % cos_sim)
                # conflicts.append(h)
                calibrated_gradient = gradient_calibration_vaccine(gradients[i], h, i, j, cos_sim)
                # 将h校准到g上
                gradients[i] = calibrated_gradient
                # print('cos_sim after is %s' % (np.dot(h, gradients[i]) / (np.linalg.norm(h) * np.linalg.norm(gradients[i]))))
            self.cos_dict[i, j] = max(0.3 * cos_sim + 0.7 * self.cos_dict[i, j], 0)
            if cos_sim < 0:
                conflict_count += 1
        # if conflict_count >0:
        #     print('conflict_count of client %s is %s,'%(i,conflict_count))
        # if len(conflicts)>0:
        #     # print('gradient %s has %s conficts' % (i, len(conflicts)))
        #     #sum_of_conflicts = np.sum(conflicts, axis=0) / len(conflicts)
        #     calibrated_gradient = gradient_calibration(g, sum_of_conflicts,)
        # gradients[i]=calibrated_gradient
    # cache = sum_of_gradient
    sum_of_gradient = np.sum(gradients, axis=0) / len(gradients)
    # print("delta of global gradient after calibration is %s" %np.linalg.norm(sum_of_gradient - cache))
    new_model = copy.copy(self.model)
    new_model.to('cpu')
    new_weights = new_model.state_dict()
    global_gradient = self.model.unflatten_model(sum_of_gradient)
    for key in new_model.state_dict().keys():
        new_weights[key] = new_weights[key] - 1 * global_gradient[key]
    new_model.load_state_dict(new_weights)

    return new_model


def pairwise_vaccine_SOP1(self, sampled_client_indices, coeff, eps=0.03):
    # 个体逐渐远离则校准，
    # 投影到SOP,目标是其他客户端相似度的平均水平？维护每一个客户端和其他pair的similarity baseline。
    # 即客户端和其他
    def gradient_calibration_vaccine(gradient, project_target, target_cos, current_cosine):
        projection = (np.linalg.norm(gradient) * (target_cos * (1 - current_cosine ** 2) ** 0.5 -
                                                  current_cosine * (1 - target_cos ** 2) ** 0.5)) \
                     / np.linalg.norm(project_target) * (1 - target_cos ** 2) ** 0.5
        gradient = gradient + projection * project_target

        return gradient

    # get all gredients
    gradients = []

    for i in sampled_client_indices:
        gradient = self.clients[i].get_gradient()
        gradients.append(gradient)
    gradients = np.array(gradients)
    calibrated_gradients = gradients
    for i, g in enumerate(gradients):
        conflicts = []
        for j, h in enumerate(gradients):
            if j != i:
                conflicts.append(h)
        SOP = np.sum(conflicts, axis=0) / len(conflicts)
        current_cosine = np.dot(SOP, g) / (np.linalg.norm(SOP) * np.linalg.norm(g))
        if current_cosine < self.target_cosine[i]:
            # print(gradients[i].shape())
            calibrated_gradient = gradient_calibration_vaccine(g, SOP, self.target_cosine[i], current_cosine)
            calibrated_gradients[i] = calibrated_gradient
        self.target_cosine[i] = 0.3 * current_cosine + 0.7 * self.target_cosine[i]
    sum_of_gradient = np.sum(gradients, axis=0) / len(gradients)
    # print("delta of global gradient after calibration is %s" %np.linalg.norm(sum_of_gradient - cache))
    new_model = copy.copy(self.model)
    new_model.to('cpu')
    new_weights = new_model.state_dict()
    global_gradient = self.model.unflatten_model(sum_of_gradient)
    for key in new_model.state_dict().keys():
        new_weights[key] = new_weights[key] - 1 * global_gradient[key]
    new_model.load_state_dict(new_weights)
    # if self._round == config.NUM_ROUNDS:
    #     # print(self._round )
    #     # print('average heatmap is',self.df_sum/(config.NUM_ROUNDS/1))
    #     hm = sns.heatmap(self.df_sum / (config.NUM_ROUNDS / 20),vmin=0, vmax=1)
    #     plt.savefig('./heatmap/pic-class{}-final.png'.format(config.NUM_SELECTED_CLASS))
    #     plt.show()
    return new_model


def pairwise_vaccine_SOP2(self, sampled_client_indices, coeff, eps=0.03):
    # 个体逐渐远离则校准，目标为冲突的梯度的平均
    # 投影到SOP,目标是其他客户端相似度的平均水平？维护每一个客户端和其他pair的similarity baseline。
    # 即客户端和其他
    def gradient_calibration_vaccine(gradient, project_target, target_cos, current_cosine):
        projection = (np.linalg.norm(gradient) * (target_cos * (1 - current_cosine ** 2) ** 0.5 -
                                                  current_cosine * (1 - target_cos ** 2) ** 0.5)) \
                     / np.linalg.norm(project_target) * (1 - target_cos ** 2) ** 0.5
        gradient = gradient + projection * project_target
        # print('projection is' ,projection)
        return gradient

    # get all gredients
    gradients = []
    for i in sampled_client_indices:
        gradient = self.clients[i].get_gradient()
        gradients.append(gradient)
    gradients = np.array(gradients)
    calibrated_gradients = gradients
    for i, g in enumerate(gradients):
        conflicts = []
        for j, h in enumerate(gradients):
            cos_sim = np.dot(h, g) / (np.linalg.norm(h) * np.linalg.norm(g))
            if cos_sim < self.cos_dict[i, j] or cos_sim < 0:
                conflicts.append(h)
            self.cos_dict[i, j] = max(0.3 * cos_sim + 0.7 * self.cos_dict[i, j], 0)
        if len(conflicts) > 0:
            # print('length of conflicts',len(conflicts))
            SOP = np.sum(conflicts, axis=0) / len(conflicts)
            current_cosine = np.dot(SOP, g) / (np.linalg.norm(SOP) * np.linalg.norm(g))
            # current_cosine_with_sop=[]
            #     print('before %s vs benchmark %s'%(current_cosine,self.target_cosine[i]))
            if current_cosine < self.target_cosine[i] or current_cosine < 0:
                # print('client %s current_cosine is %s' %(i,current_cosine))
                # print(gradients[i].shape())
                calibrated_gradient = gradient_calibration_vaccine(g, SOP, self.target_cosine[i], current_cosine)
                gradients[i] = calibrated_gradient
                # print('similarity after calibration %s' % (np.dot(SOP, calibrated_gradient) / (np.linalg.norm(SOP) * np.linalg.norm(calibrated_gradient))))
        self.target_cosine[i] = max(0.3 * current_cosine + 0.7 * self.target_cosine[i], 0)
    # sum_of_gradient = np.sum(calibrated_gradients, axis=0) / len(calibrated_gradients)
    sum_of_gradient = np.sum(gradients, axis=0) / len(gradients)
    # print("delta of global gradient after calibration is %s" %np.linalg.norm(sum_of_gradient - cache))
    new_model = copy.copy(self.model)
    new_model.to('cpu')
    new_weights = new_model.state_dict()
    global_gradient = self.model.unflatten_model(sum_of_gradient)
    for key in new_model.state_dict().keys():
        new_weights[key] = new_weights[key] - 1 * global_gradient[key]
    new_model.load_state_dict(new_weights)

    return new_model


def pairwise_vaccine_SOP3(self, sampled_client_indices, coeff, eps=0.03):
    # 个体逐渐远离则校准，目标为冲突的梯度的平均
    # 投影到SOP,目标是其他客户端相似度的平均水平？维护每一个客户端和其他pair的similarity baseline。
    # 即客户端和其他
    def gradient_calibration_vaccine(gradient, project_target, target_cos, current_cosine):
        projection = (np.linalg.norm(gradient) * (target_cos * (1 - current_cosine ** 2) ** 0.5 -
                                                  current_cosine * (1 - target_cos ** 2) ** 0.5)) \
                     / np.linalg.norm(project_target) * (1 - target_cos ** 2) ** 0.5
        gradient = gradient + projection * project_target
        # print('projection is' ,projection)
        return gradient

    # get all gredients
    gradients = []
    for i in sampled_client_indices:
        gradient = self.clients[i].get_gradient()
        gradients.append(gradient)
    gradients = np.array(gradients)
    calibrated_gradients = gradients
    for i, g in enumerate(gradients):
        current_cosine_list = []
        conflicts = []
        for j, h in enumerate(gradients):
            cos_sim = np.dot(h, g) / (np.linalg.norm(h) * np.linalg.norm(g))
            if cos_sim < self.cos_dict[i, j] or cos_sim < 0:
                conflicts.append(h)
            self.cos_dict[i, j] = max(0.1 * cos_sim + 0.9 * self.cos_dict[i, j], 0)
        # print('cos_dict is %s' % self.cos_dict)
        if len(conflicts) > 0:
            print('length of conflicts', len(conflicts))
            SOP = np.sum(conflicts, axis=0) / len(conflicts)
            current_cosine = np.dot(SOP, g) / (np.linalg.norm(SOP) * np.linalg.norm(g))
            # current_cosine_with_sop=[]
            #     print('before %s vs benchmark %s'%(current_cosine,self.target_cosine[i]))
            if current_cosine < self.target_cosine[i] or current_cosine < 0:
                # print('client %s current_cosine is %s' %(i,current_cosine))
                # print(gradients[i].shape())
                calibrated_gradient = gradient_calibration_vaccine(g, SOP, self.target_cosine[i], current_cosine)
                gradients[i] = calibrated_gradient
                # print('similarity after calibration %s' % (np.dot(SOP, calibrated_gradient) / (np.linalg.norm(SOP) * np.linalg.norm(calibrated_gradient))))
                calibrated_cosine = np.dot(SOP, calibrated_gradient) / (
                            np.linalg.norm(SOP) * np.linalg.norm(calibrated_gradient))
                # 校准的幅度作为
                self.target_cosine[i] = max(0.1 * calibrated_cosine + 0.9 * self.target_cosine[i], 0)
        # else:
        #     self.target_cosine[i] = max(0.1 * calibrated_cosine + 0.9 * self.target_cosine[i], 0)
        #
    print('target_cosine is %s' % self.target_cosine)
    # sum_of_gradient = np.sum(calibrated_gradients, axis=0) / len(calibrated_gradients)
    sum_of_gradient = np.sum(gradients, axis=0) / len(gradients)
    # print("delta of global gradient after calibration is %s" %np.linalg.norm(sum_of_gradient - cache))
    new_model = copy.copy(self.model)
    new_model.to('cpu')
    new_weights = new_model.state_dict()
    global_gradient = self.model.unflatten_model(sum_of_gradient)
    for key in new_model.state_dict().keys():
        new_weights[key] = new_weights[key] - 1 * global_gradient[key]
    new_model.load_state_dict(new_weights)

    return new_model


def cossim_analysis(self, sampled_client_indices, coeff, eps=0.03):
    # 用VACCINE方法 判断梯度pair的相似度变化，如果变得差异更大了，就修正。所以是自适应的，没啥超参数了/。
    # 在server端，维护一个客户端cossim的表 cos_dict,缓存历史相似度。对应VACCINE的phiT.
    # 每一次当前cissim和缓存dict cossim比较 如果变得差异更大了（值更小了）就校准，公式按照VACCINE
    # 无论如何都更新cossim。

    # print('projection weight is', 0.8**int(self._round / 50))
    # th = 1.2 ** int(self._round / 50)
    # print(' TH is ', th,'cossim is', -0.3*th)
    # def gradient_calibration(gradient, project_target):
    #     projection = np.dot(gradient, project_target) / np.linalg.norm(project_target)
    #     gradient = gradient - projection * project_target
    #     return gradient
    def gradient_calibration_vaccine(gradient, project_target, i, j, cossim):
        # projection = np.dot(gradient, project_target) / np.linalg.norm(project_target) ** 2
        projection = (np.linalg.norm(gradient) * self.cos_dict[i, j] * (1 - cossim ** 2) ** 0.5 - cossim * (
                1 - self.cos_dict[i, j] ** 2) ** 0.5) / np.linalg.norm(project_target) * (
                             1 - self.cos_dict[i, j] ** 2) ** 0.5
        # projection= (np.linalg.norm(gradient)*self.cos_dict[i, j]*(1-cossim**2)**0.5-cossim*(
        #   1-self.cos_dict[i, j]**2)**0.5)/np.linalg.norm(project_target)*
        #   (1-self.cos_dict[i,j]**2)**0.5
        gradient = gradient + projection * project_target
        return gradient

    # get all gredients
    gradients = []
    for i in sampled_client_indices:
        gradient = self.clients[i].get_gradient()
        gradients.append(gradient)
    gradients = np.array(gradients)
    count_of_conflict = 0
    for i, g in enumerate(gradients):
        conflicts = []
        conflict_count = 0
        for j, h in enumerate(gradients):
            cos_sim = np.dot(h, g) / (np.linalg.norm(h) * np.linalg.norm(g))
            # if cos_sim < self.cos_dict[i, j]:
            #     # conflicts.append(h)
            #     calibrated_gradient = gradient_calibration_vaccine(g, h, i, j, cos_sim)
            #     gradients[i] = calibrated_gradient
            #     conflict_count += 1
            self.cos_dict[i, j] = cos_sim
            if cos_sim < 0:
                count_of_conflict += 1
        # if conflict_count >0:
        #     print('conflict_count of client %s is %s,'%(i,conflict_count))
        # if len(conflicts)>0:
        #     # print('gradient %s has %s conficts' % (i, len(conflicts)))
        #     #sum_of_conflicts = np.sum(conflicts, axis=0) / len(conflicts)
        #     calibrated_gradient = gradient_calibration(g, sum_of_conflicts,)
        # gradients[i]=calibrated_gradient
    # cache = sum_of_gradient
    if self._round % 30 == 0:
        print('number_of_conflict', count_of_conflict)
    sum_of_gradient = np.sum(gradients, axis=0) / len(gradients)
    # print("delta of global gradient after calibration is %s" %np.linalg.norm(sum_of_gradient - cache))
    new_model = copy.copy(self.model)
    new_model.to('cpu')
    new_weights = new_model.state_dict()
    global_gradient = self.model.unflatten_model(sum_of_gradient)
    for key in new_model.state_dict().keys():
        new_weights[key] = new_weights[key] - 1 * global_gradient[key]
    new_model.load_state_dict(new_weights)
    if self._round % 30 == 0:
        # print(self.cos_dict)
        df = pd.DataFrame(data=self.cos_dict[0:, 0:])
        self.df_sum = self.df_sum + df
        # print('average heatmap is', self.df_sum)
        # print(df)
        # hm = sns.heatmap(df,vmin=-1,vmax=1)
        # plt.savefig('./heatmap/pic-class{}-step{}.png'.format(config.NUM_SELECTED_CLASS, self._round))
        # plt.show()
        cossim_snapshot = df.mean()
        cossim_snapshot_avg = np.mean(cossim_snapshot)
        # print('cossim_snapshot is' ,cossim_snapshot)
        print('cossim_snapshot mean is', cossim_snapshot_avg)
        # columns=
        # print(df)
    # if self._round == config.NUM_ROUNDS:
    # print(self._round )
    # print('average heatmap is',self.df_sum/(config.NUM_ROUNDS/1))
    # hm = sns.heatmap(self.df_sum / (config.NUM_ROUNDS / 20),vmin=-1,vmax=1)
    # plt.savefig('./heatmap/pic-class{}-final.png'.format(config.NUM_SELECTED_CLASS))
    # plt.show()
    return new_model


def layerwise_vaccine(self, sampled_client_indices, coeff, eps=0.03):
    def get_cossim(h, g):
        cos_sim = np.dot(h, g) / (np.linalg.norm(h) * np.linalg.norm(g))
        return cos_sim

    def layerwise_calibration(gradient, project_target, i, j):
        gradient = self.model.unflatten_model(gradient)
        project_target = self.model.unflatten_model(project_target)
        for k, key in enumerate(gradient.keys()):
            # print(gradient[key])
            # g_layer = gradient[key].numpy().flatten()
            # sog_layer = project_target[key].numpy().flatten()
            g_layer = gradient[key].flatten()
            sog_layer = project_target[key].flatten()
            # print('flatten',g_layer)
            shape = gradient[key].shape
            cossim = get_cossim(g_layer, sog_layer)
            if cossim < self.cos_dict_layerwise[i, j, k]:
                projection = (np.linalg.norm(g_layer) * self.cos_dict_layerwise[i, j, k] * (
                            1 - cossim ** 2) ** 0.5 - cossim * (
                                          1 - self.cos_dict_layerwise[i, j, k] ** 2) ** 0.5) / np.linalg.norm(
                    sog_layer) * (1 - self.cos_dict_layerwise[i, j, k] ** 2) ** 0.5
                g_layer = g_layer + projection * sog_layer
            self.cos_dict_layerwise[i, j, k] = 0.5 * cossim + 0.5 * self.cos_dict_layerwise[i, j, k]
            g_layer = torch.tensor(g_layer.reshape(shape))
            gradient[key] = g_layer
        return self.model.flatten_model(gradient)

    gradients = []
    for i in sampled_client_indices:
        gradient = self.clients[i].get_gradient()
        gradients.append(gradient)
    gradients = np.array(gradients)
    for i, g in enumerate(gradients):
        for j, h in enumerate(gradients):
            calibrated_gradient = layerwise_calibration(g, h, i, j)
            gradients[i] = calibrated_gradient
    sum_of_gradient = np.sum(gradients, axis=0) / len(gradients)
    new_model = copy.copy(self.model)
    new_model.to('cpu')
    new_weights = new_model.state_dict()
    global_gradient = self.model.unflatten_model(sum_of_gradient)
    for key in new_model.state_dict().keys():
        new_weights[key] = new_weights[key] - 1 * global_gradient[key]
    new_model.load_state_dict(new_weights)
    return new_model


# 冲突检测和修正都是layerwise,即每层单独找到冲突的客户端，并投影到平均。
def vertical_conflicts_avg_cossim1_iteration_eps3_layerwise(self, sampled_client_indices, coeff, eps=0.03):
    # print('cossim TH is ', 0.95 ** int(self._round/10))
    # th=0.95 ** int(self._round/10)
    def gradient_calibration(gradient, project_target):
        projection = np.dot(gradient, project_target) / np.linalg.norm(project_target) ** 2
        gradient = gradient - projection * project_target
        return gradient

    def get_cossim(h, g):
        cos_sim = np.dot(h, g) / (np.linalg.norm(h) * np.linalg.norm(g))
        return cos_sim

    def layerwise_calibration(gradient, sum_of_gradient):
        gradient = self.model.unflatten_model(gradient)
        sum_of_gradient = self.model.unflatten_model(sum_of_gradient)
        for key in gradient.keys():
            g_layer = gradient[key].numpy().flatten()
            sog_layer = sum_of_gradient[key].numpy().flatten()
            shape = gradient[key].shape
            cos_sim = get_cossim(g_layer, sog_layer)
            # print(key, cos_sim)
            if cos_sim < 0:
                # print("cossim of layer %s is %s" %(key,cos_sim))
                projection = np.dot(g_layer, sog_layer) / np.linalg.norm(sog_layer) ** 2
                g_layer = g_layer - projection * sog_layer
            g_layer = torch.tensor(g_layer.reshape(shape))
            gradient[key] = g_layer
        return self.model.flatten_model(gradient)

    # def layerwise_detection(j,gradient, h,target):
    #     gradient = self.model.unflatten_model(gradient)
    #     target = self.model.unflatten_model(target)
    #     for key in gradient.keys():
    #         g_layer = gradient[key].numpy().flatten()
    #         sog_layer = target[key].numpy().flatten()
    #         shape = gradient[key].shape
    #         cos_sim = get_cossim(g_layer, sog_layer)
    #         # print(key, cos_sim)
    #         if cos_sim < -0.1:
    #             conflicts.append(h)
    #             projection = np.dot(g_layer, sog_layer) / np.linalg.norm(sog_layer) ** 2
    #             g_layer = g_layer -  projection * sog_layer
    #         g_layer = torch.tensor(g_layer.reshape(shape))
    #         gradient[key] = g_layer
    #     return self.model.flatten_model(gradient)

    # get all gredients
    gradients = []
    for i in sampled_client_indices:
        gradient = self.clients[i].get_gradient()
        gradients.append(gradient)
    gradients = np.array(gradients)
    sum_of_gradient = np.sum(gradients, axis=0) / len(gradients)
    # sop=sum_of_gradient
    cache = None
    if cache is None:
        cache = sum_of_gradient - 2 * eps
    calibration_iteration = 0
    while np.linalg.norm(sum_of_gradient - cache) > eps:
        calibration_iteration += 1
        # print("calibration_iteration %s" %calibration_iteration)
        for i, g in enumerate(gradients):
            conflicts = []
            for j, h in enumerate(gradients):
                cos_sim = np.dot(h, g) / (np.linalg.norm(h) * np.linalg.norm(g))
                if cos_sim < -0.3 * 0.9 ** self._round:
                    conflicts.append(h)
            if len(conflicts) > 0:
                # print('gradient %s has %s conficts' % (i, len(conflicts)))
                sum_of_conflicts = np.sum(conflicts, axis=0) / len(conflicts)
                calibrated_gradient = layerwise_calibration(g, sum_of_conflicts)
                gradients[i] = calibrated_gradient
        cache = sum_of_gradient
        sum_of_gradient = np.sum(gradients, axis=0) / len(gradients)
        # print("delta of global gradient after calibration is %s" %np.linalg.norm(sum_of_gradient - cache))
    new_model = copy.copy(self.model)
    new_model.to('cpu')
    new_weights = new_model.state_dict()
    global_gradient = self.model.unflatten_model(sum_of_gradient)
    for key in new_model.state_dict().keys():
        new_weights[key] = new_weights[key] - 1 * global_gradient[key]

    new_model.load_state_dict(new_weights)
    return new_model


#
# def layerwise_calibration_all_conflicted_required(gradient, gradients):
#     gradient = self.model.unflatten_model(gradient)
#     all_gradients = []
#     for g in gradients:
#         all_gradients.append(self.model.unflatten_model(g))
#     conflicts = [i for i in range(20)]                  # conflict clients from the last layer
#     for key in gradient.keys():
#         target_layer = gradient[key].numpy().flatten()
#         new_conflict = []                               # conflict clients in current layer
#         for i, g in enumerate(all_gradients):
#             if i not in conflicts:                      # has to find conflict in the last layer
#                 continue
#             g_layer = g[key].numpy().flatten()
#
#         if get_cossim(target_layer, g_layer) < -0.3:
#             new_conflict.append(i)
#         conflicts = new_conflict
#
#     print('Conflicts layers:' + str(conflicts))
#
#     if conflicts:
#         for key in gradient.keys():
#             avg = []
#             for i in conflicts:
#                 g_layer = all_gradients[i][key].numpy().flatten()
#                 avg.append(g_layer)
#             average_of_conflicts = np.average(avg, axis=0)             # the average of all conflicts
#
#             shape = gradient[key].shape
#             cos_sim = get_cossim(target_layer, average_of_conflicts)
#             if cos_sim < -0.3:                              # calibration only occurs when cossim < -0.3
#                 print(key, cos_sim)
#                 projection = np.dot(target_layer, average_of_conflicts) / np.linalg.norm(average_of_conflicts) ** 2
#                 target_layer = target_layer - projection * average_of_conflicts
#
#             g_layer = torch.tensor(target_layer.reshape(shape))
#             gradient[key] = target_layer
#
#     return self.model.flatten_model(gradient)
# def layerwise_calibration_no_restriction(gradient, gradients):
#     gradient = self.model.unflatten_model(gradient)
#
#     all_gradients = []
#     for g in gradients:
#         all_gradients.append(self.model.unflatten_model(g))
#
#     for key in gradient.keys():
#         target_layer = gradient[key].numpy().flatten()
#         conflicts = []
#         for g in all_gradients:
#             g_layer = g[key].numpy().flatten()
#
#             if get_cossim(target_layer, g_layer) < -0.3:
#                 conflicts.append(g_layer)
#
#         print('Conflicts layers:' + len(conflicts))
#
#         if conflicts:
#             average_of_conflicts = np.average(conflicts, axis=0)  # the average of all conflicts
#
#             shape = gradient[key].shape
#             cos_sim = get_cossim(target_layer, average_of_conflicts)
#             if cos_sim < -0.3:  # calibration only occurs when cossim < -0.3
#                 print(key, cos_sim)
#                 projection = np.dot(target_layer, average_of_conflicts) / np.linalg.norm(average_of_conflicts) ** 2
#                 target_layer = target_layer - projection * average_of_conflicts
#
#             target_layer = torch.tensor(target_layer.reshape(shape))
#             gradient[key] = target_layer
#
#     return self.model.flatten_model(gradient)
# def layerwise_calibration_all_conflicted_required(gradient, gradients):
#     gradient = self.model.unflatten_model(gradient)
#
#     all_gradients = []
#     for g in gradients:
#         all_gradients.append(self.model.unflatten_model(g))
#
#     conflicts = [i for i in range(20)]  # conflict clients from the last layer
#     for key in gradient.keys():
#         target_layer = gradient[key].numpy().flatten()
#         new_conflict = []  # conflict clients in current layer
#         for i, g in enumerate(all_gradients):
#             if i not in conflicts:  # has to find conflict in the last layer
#                 continue
#             g_layer = g[key].numpy().flatten()
#
#         if get_cossim(target_layer, g_layer) < -0.3:
#             new_conflict.append(i)
#         conflicts = new_conflict
#
#     print('Conflicts layers:' + str(conflicts))
#
#     if conflicts:
#         for key in gradient.keys():
#             avg = []
#             for i in conflicts:
#                 g_layer = all_gradients[i][key].numpy().flatten()
#                 avg.append(g_layer)
#             average_of_conflicts = np.average(avg, axis=0)  # the average of all conflicts
#
#             shape = gradient[key].shape
#             cos_sim = get_cossim(target_layer, average_of_conflicts)
#             if cos_sim < -0.3:  # calibration only occurs when cossim < -0.3
#                 print(key, cos_sim)
#                 projection = np.dot(target_layer, average_of_conflicts) / np.linalg.norm(average_of_conflicts) ** 2
#                 target_layer = target_layer - projection * average_of_conflicts
#
#             g_layer = torch.tensor(target_layer.reshape(shape))
#             gradient[key] = target_layer
#
#     return self.model.flatten_model(gradient)
# def layerwise_calibration_no_restriction(gradient, gradients):
#     gradient = self.model.unflatten_model(gradient)
#
#     all_gradients = []
#     for g in gradients:
#         all_gradients.append(self.model.unflatten_model(g))
#
#     for key in gradient.keys():
#         target_layer = gradient[key].numpy().flatten()
#         conflicts = []
#         for g in all_gradients:
#             g_layer = g[key].numpy().flatten()
#
#             if get_cossim(target_layer, g_layer) < -0.3:
#                 conflicts.append(g_layer)
#
#         print('Conflicts layers:' + len(conflicts))
#
#         if conflicts:
#             average_of_conflicts = np.average(conflicts, axis=0)  # the average of all conflicts
#
#             shape = gradient[key].shape
#             cos_sim = get_cossim(target_layer, average_of_conflicts)
#             if cos_sim < -0.3:  # calibration only occurs when cossim < -0.3
#                 print(key, cos_sim)
#                 projection = np.dot(target_layer, average_of_conflicts) / np.linalg.norm(average_of_conflicts) ** 2
#                 target_layer = target_layer - projection * average_of_conflicts
#
#             target_layer = torch.tensor(target_layer.reshape(shape))
#             gradient[key] = target_layer
#
#     return self.model.flatten_model(gradient)

def vertical_conflicts_avg_cossim1_iteration_eps4(self, sampled_client_indices, coeff, eps=0.04):
    def gradient_calibration(gradient, project_target):
        projection = np.dot(gradient, project_target) / np.linalg.norm(project_target) ** 2
        gradient = gradient - projection * project_target
        return gradient

    # get all gredients
    gradients = []
    for i in sampled_client_indices:
        gradient = self.clients[i].get_gradient()
        gradients.append(gradient)
    gradients = np.array(gradients)
    sum_of_gradient = np.sum(gradients, axis=0) / len(gradients)
    # sop=sum_of_gradient
    cache = None
    if cache is None:
        cache = sum_of_gradient - 2 * eps
    calibration_iteration = 0
    while np.linalg.norm(sum_of_gradient - cache) > eps:
        calibration_iteration += 1
        print("calibration_iteration %s" % calibration_iteration)
        for i, g in enumerate(gradients):
            conflicts = []
            for j, h in enumerate(gradients):
                cos_sim = np.dot(h, g) / (np.linalg.norm(h) * np.linalg.norm(g))
                if cos_sim < -0.1:
                    conflicts.append(h)
            if len(conflicts) > 0:
                print('gradient %s has %s conficts' % (i, len(conflicts)))
                sum_of_conflicts = np.sum(conflicts, axis=0) / len(conflicts)
                calibrated_gradient = gradient_calibration(g, sum_of_conflicts)
                gradients[i] = calibrated_gradient
        cache = sum_of_gradient
        sum_of_gradient = np.sum(gradients, axis=0) / len(gradients)
        print("delta of global gradient after calibration is %s" % np.linalg.norm(sum_of_gradient - cache))
    new_model = copy.copy(self.model)
    new_model.to('cpu')
    new_weights = new_model.state_dict()
    global_gradient = self.model.unflatten_model(sum_of_gradient)
    for key in new_model.state_dict().keys():
        new_weights[key] = new_weights[key] - 1 * global_gradient[key]

    new_model.load_state_dict(new_weights)
    return new_model


# def vertical_conflicts_avg_cossim1_iteration_eps4(self, sampled_client_indices, coeff, eps=0.04):
#     def gradient_calibration(gradient, project_target):
#         projection = np.dot(gradient, project_target) / np.linalg.norm(project_target) ** 2
#         gradient = gradient - projection * project_target
#         return gradient
#     # get all gredients
#     gradients = []
#     for i in sampled_client_indices:
#         gradient = self.clients[i].get_gradient()
#         gradients.append(gradient)
#     gradients = np.array(gradients)
#     sum_of_gradient=np.sum(gradients, axis=0) / len(gradients)
#     # sop=sum_of_gradient
#     cache = None
#     if cache is None:
#        cache = sum_of_gradient - 2 * eps
#     calibration_iteration=0
#     while np.linalg.norm(sum_of_gradient - cache) > eps:
#         calibration_iteration+=1
#         print("calibration_iteration %s" %calibration_iteration)
#         for i, g in enumerate(gradients):
#             conflicts = []
#             for j, h in enumerate(gradients):
#                 cos_sim = np.dot(h, g) / (np.linalg.norm(h) * np.linalg.norm(g))
#                 if cos_sim < -0.1:
#                     conflicts.append(h)
#             if len(conflicts)>0:
#                 print('gradient %s has %s conficts' % (i, len(conflicts)))
#                 sum_of_conflicts = np.sum(conflicts, axis=0) / len(conflicts)
#                 calibrated_gradient = gradient_calibration(g, sum_of_conflicts)
#                 gradients[i]=calibrated_gradient
#         cache = sum_of_gradient
#         sum_of_gradient = np.sum(gradients, axis=0) / len(gradients)
#         print("delta of global gradient after calibration is %s" %np.linalg.norm(sum_of_gradient - cache))
#     new_model = copy.copy(self.model)
#     new_model.to('cpu')
#     new_weights = new_model.state_dict()
#     global_gradient = self.model.unflatten_model(sum_of_gradient)
#     for key in new_model.state_dict().keys():
#         new_weights[key] = new_weights[key] - 1 * global_gradient[key]
#
#     new_model.load_state_dict(new_weights)
#     return new_model

def aggregate_pairwise_vertical_cossim3(self, sampled_client_indices, coeff, eps=0.001):
    def gradient_calibration(gradient, project_target):
        projection = np.dot(gradient, project_target) / np.linalg.norm(project_target) ** 2
        gradient = gradient - projection * project_target
        return gradient

    # get all gredients
    gradients = []
    for i in sampled_client_indices:
        gradient = self.clients[i].get_gradient()
        gradients.append(gradient)
    gradients = np.array(gradients)
    total_conflicts = []
    for i, g in enumerate(gradients):
        # conflicts=[]
        for j, h in enumerate(gradients):
            cos_sim = np.dot(h, g) / (np.linalg.norm(h) * np.linalg.norm(g))
            if cos_sim < -0.3:
                # print('conflict pair %s and %s cossim %s' %(i,j,cos_sim))
                calibrated_gradient = gradient_calibration(g, h)
                gradients[i] = calibrated_gradient
                total_conflicts.append(h)
    print(' %s conficts' % (len(total_conflicts)))
    sum_of_gradient = np.sum(gradients, axis=0) / len(gradients)
    new_model = copy.copy(self.model)
    new_model.to('cpu')
    new_weights = new_model.state_dict()
    global_gradient = self.model.unflatten_model(sum_of_gradient)
    for key in new_model.state_dict().keys():
        new_weights[key] = new_weights[key] - 1 * global_gradient[key]

    new_model.load_state_dict(new_weights)
    return new_model


def aggregate_models_with_pruning(self, sampled_client_indices, coeff, eps=0.001):
    def gradient_calibration(gradient, sum_of_gradient):
        projection = np.dot(gradient, sum_of_gradient) / np.linalg.norm(sum_of_gradient) ** 2
        if projection > 0:
            gradient = gradient + projection * sum_of_gradient
        else:
            gradient = gradient - 2 * projection * sum_of_gradient
        return gradient

    # get gradient (delta) of selected clients
    gradients = []
    for i in sampled_client_indices:
        gradient = self.clients[i].get_gradient()
        gradients.append(gradient)
    gradients = np.array(gradients)

    # get sum of gradient
    sum_of_gradient = np.sum(gradients, axis=0) / len(gradients)
    sop = sum_of_gradient
    if self.sop_cache is None:
        self.sop_cache = sum_of_gradient - 2 * 0.001

    while np.linalg.norm(sop - self.sop_cache) > eps:
        for i, g in enumerate(gradients):
            # g_layers = self.model.unflatten_model(g)
            # sop_layers = self.model.unflatten_model(sop)
            #
            # for key in g_layers.keys():
            #     g_temp = g_layers[key].numpy().flatten()
            #     sop_temp = sop_layers[key].numpy().flatten()
            #     cos_sim = np.dot(g_temp, sop_temp) / (np.linalg.norm(g_temp) * np.linalg.norm(sop_temp))
            #     print(i, cos_sim)

            cos_sim = np.dot(sop, g) / (np.linalg.norm(sop) * np.linalg.norm(g))
            if cos_sim < 0.3:
                print(i, cos_sim)
                calibrated_gradient = gradient_calibration(g, sop)
                gradients[i] = calibrated_gradient
                cos_sim_new = np.dot(sop, calibrated_gradient) / (
                            np.linalg.norm(sop) * np.linalg.norm(calibrated_gradient))
                print(i, cos_sim_new)

        self.sop_cache = sum_of_gradient
        sum_of_gradient = np.sum(gradients, axis=0) / len(gradients)
        sop = sum_of_gradient

    new_model = copy.copy(self.model)
    new_model.to('cpu')
    new_weights = new_model.state_dict()
    global_gradient = self.model.unflatten_model(sum_of_gradient)
    for key in new_model.state_dict().keys():
        new_weights[key] = new_weights[key] - 1 * global_gradient[key]

    new_model.load_state_dict(new_weights)

    # client = self.clients[0]
    # client_current = copy.deepcopy(client.client_current)
    # client_current = client_current.state_dict()
    # client_global = copy.deepcopy(client.global_current)
    #
    # averaged_weights = client.get_gradient()
    #
    # new_model = client_global.unflatten_model(averaged_weights)
    # client_global.to('cpu')
    # client_global = client_global.state_dict()
    # for key in client_global.keys():
    #     client_global[key] = client_global[key] - new_model[key]
    #     a = np.subtract(client_global[key], client_current[key])
    #     print(a)

    return new_model


def aggregate_models_with_pruning_nagetive_cossim_only(self, sampled_client_indices, coeff, eps=0.001):
    def gradient_calibration(gradient, sum_of_gradient):
        projection = np.dot(gradient, sum_of_gradient) / np.linalg.norm(sum_of_gradient) ** 2
        gradient = gradient - 2 * projection * sum_of_gradient
        return gradient

    # get gradient (delta) of selected clients
    gradients = []
    for i in sampled_client_indices:
        gradient = self.clients[i].get_gradient()
        gradients.append(gradient)
    gradients = np.array(gradients)

    # get sum of gradient
    sum_of_gradient = np.sum(gradients, axis=0) / len(gradients)
    sop = sum_of_gradient
    if self.sop_cache is None:
        self.sop_cache = sum_of_gradient - 2 * 0.001

    while np.linalg.norm(sop - self.sop_cache) > eps:
        for i, g in enumerate(gradients):
            # g_layers = self.model.unflatten_model(g)
            # sop_layers = self.model.unflatten_model(sop)
            #
            # for key in g_layers.keys():
            #     g_temp = g_layers[key].numpy().flatten()
            #     sop_temp = sop_layers[key].numpy().flatten()
            #     cos_sim = np.dot(g_temp, sop_temp) / (np.linalg.norm(g_temp) * np.linalg.norm(sop_temp))
            #     print(i, cos_sim)

            cos_sim = np.dot(sop, g) / (np.linalg.norm(sop) * np.linalg.norm(g))
            if cos_sim < 0:
                print(i, 'cossim before projection', cos_sim)
                calibrated_gradient = gradient_calibration(g, sop)
                gradients[i] = calibrated_gradient
                cos_sim_new = np.dot(sop, calibrated_gradient) / (
                            np.linalg.norm(sop) * np.linalg.norm(calibrated_gradient))
                print(i, 'cossim after projection', cos_sim_new)

        self.sop_cache = sum_of_gradient
        sum_of_gradient = np.sum(gradients, axis=0) / len(gradients)
        sop = sum_of_gradient

    new_model = copy.copy(self.model)
    new_model.to('cpu')
    new_weights = new_model.state_dict()
    global_gradient = self.model.unflatten_model(sum_of_gradient)
    for key in new_model.state_dict().keys():
        new_weights[key] = new_weights[key] - 1 * global_gradient[key]

    new_model.load_state_dict(new_weights)

    # client = self.clients[0]
    # client_current = copy.deepcopy(client.client_current)
    # client_current = client_current.state_dict()
    # client_global = copy.deepcopy(client.global_current)
    #
    # averaged_weights = client.get_gradient()
    #
    # new_model = client_global.unflatten_model(averaged_weights)
    # client_global.to('cpu')
    # client_global = client_global.state_dict()
    # for key in client_global.keys():
    #     client_global[key] = client_global[key] - new_model[key]
    #     a = np.subtract(client_global[key], client_current[key])
    #     print(a)

    return new_model


def aggregate_models_with_pruning_nagetive_cossim_vertical_cossim1(self, sampled_client_indices, coeff, eps=0.001):
    def gradient_calibration(gradient, sum_of_gradient):
        projection = np.dot(gradient, sum_of_gradient) / np.linalg.norm(sum_of_gradient) ** 2
        gradient = gradient - projection * sum_of_gradient
        return gradient

    # get gradient (delta) of selected clients
    gradients = []
    for i in sampled_client_indices:
        gradient = self.clients[i].get_gradient()
        gradients.append(gradient)
    gradients = np.array(gradients)

    # get sum of gradient
    sum_of_gradient = np.sum(gradients, axis=0) / len(gradients)
    sop = sum_of_gradient
    if self.sop_cache is None:
        self.sop_cache = sum_of_gradient - 2 * 0.001

    while np.linalg.norm(sop - self.sop_cache) > eps:
        for i, g in enumerate(gradients):
            cos_sim = np.dot(sop, g) / (np.linalg.norm(sop) * np.linalg.norm(g))
            if cos_sim < -0.1:
                print(i, 'cossim before projection', cos_sim)
                calibrated_gradient = gradient_calibration(g, sop)
                gradients[i] = calibrated_gradient
                cos_sim_new = np.dot(sop, calibrated_gradient) / (
                            np.linalg.norm(sop) * np.linalg.norm(calibrated_gradient))
                print(i, 'cossim after projection', cos_sim_new)

        self.sop_cache = sum_of_gradient
        sum_of_gradient = np.sum(gradients, axis=0) / len(gradients)
        sop = sum_of_gradient

    new_model = copy.copy(self.model)
    new_model.to('cpu')
    new_weights = new_model.state_dict()
    global_gradient = self.model.unflatten_model(sum_of_gradient)
    for key in new_model.state_dict().keys():
        new_weights[key] = new_weights[key] - 1 * global_gradient[key]

    new_model.load_state_dict(new_weights)
    return new_model


def aggregate_models_with_pruning_nagetive_cossim_only_vertical(self, sampled_client_indices, coeff, eps=0.001):
    def gradient_calibration(gradient, sum_of_gradient):
        projection = np.dot(gradient, sum_of_gradient) / np.linalg.norm(sum_of_gradient) ** 2
        gradient = gradient - projection * sum_of_gradient
        return gradient

    # get gradient (delta) of selected clients
    gradients = []
    for i in sampled_client_indices:
        gradient = self.clients[i].get_gradient()
        gradients.append(gradient)
    gradients = np.array(gradients)

    # get sum of gradient
    sum_of_gradient = np.sum(gradients, axis=0) / len(gradients)
    sop = sum_of_gradient
    if self.sop_cache is None:
        self.sop_cache = sum_of_gradient - 2 * 0.001

    while np.linalg.norm(sop - self.sop_cache) > eps:
        for i, g in enumerate(gradients):
            cos_sim = np.dot(sop, g) / (np.linalg.norm(sop) * np.linalg.norm(g))
            if cos_sim < 0:
                print(i, 'cossim before projection', cos_sim)
                calibrated_gradient = gradient_calibration(g, sop)
                gradients[i] = calibrated_gradient
                cos_sim_new = np.dot(sop, calibrated_gradient) / (
                            np.linalg.norm(sop) * np.linalg.norm(calibrated_gradient))
                print(i, 'cossim after projection', cos_sim_new)
        self.sop_cache = sum_of_gradient
        sum_of_gradient = np.sum(gradients, axis=0) / len(gradients)
        sop = sum_of_gradient

    new_model = copy.copy(self.model)
    new_model.to('cpu')
    new_weights = new_model.state_dict()
    global_gradient = self.model.unflatten_model(sum_of_gradient)
    for key in new_model.state_dict().keys():
        new_weights[key] = new_weights[key] - 1 * global_gradient[key]

    new_model.load_state_dict(new_weights)

    # client = self.clients[0]
    # client_current = copy.deepcopy(client.client_current)
    # client_current = client_current.state_dict()
    # client_global = copy.deepcopy(client.global_current)
    #
    # averaged_weights = client.get_gradient()
    #
    # new_model = client_global.unflatten_model(averaged_weights)
    # client_global.to('cpu')
    # client_global = client_global.state_dict()
    # for key in client_global.keys():
    #     client_global[key] = client_global[key] - new_model[key]
    #     a = np.subtract(client_global[key], client_current[key])
    #     print(a)

    return new_model