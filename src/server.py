import copy
import random
from collections import OrderedDict

import numpy as np
import scipy
from tqdm.auto import tqdm

from src.client import Client
from src.models import *
from utils.CommunicationController import CommunicationController
from utils.DatasetController import DatasetController
from utils.Printer import *

logger = logging.getLogger(__name__)


class Server(object):
    def __init__(self, writer):
        self._round = 0
        self.seed = 5959
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 需要初始化
        self.round = None
        self.clients = None
        self.writer = writer
        self.CommunicationController = None
        self.DatasetController = None
        self.model = None
        self.total_rounds = None
        self.run_type = None
        self.update_type = None
        self.save = None
        self.number_of_classes = None

        # records
        self.data = None
        self.dataloader = None
        self.round_upload = []
        self.round_accuracy = []
        self.round_compression_rate = []

        # scaffold
        self.c_global = None

        # DGT
        self.cos_dict = None
        self.ca4 = []
        self.client_weights_list = None
        self.client_weights = None
        self.client_conflicts_accu = None
        self.client_conflicts = None
        self.latency = None


    def log(self, message):
        message = f"[Round: {str(self._round).zfill(4)}] " + message
        print(message);
        logging.info(message)
        del message;
        gc.collect()

    @staticmethod
    def load_model(model_name, num_class=10):
        # load model architecture
        if model_name == 'CNN':
            model_config = {
                'name': 'CNN',
                'in_channels': 1,
                'hidden_channels': 32,
                'num_hiddens': 512,
                'num_classes': num_class,
            }
        elif model_name == 'CNN2':
            model_config = {
                'name': 'CNN2',
                'in_channels': 3,
                'hidden_channels': 32,
                'num_hiddens': 512,
                'num_classes': num_class,
            }
        elif model_name == 'TwoNN':
            model_config = {
                'name': 'TwoNN',
                'in_features': 784,
                'num_hiddens': 512,
                'num_classes': num_class,
            }
        else:
            raise ValueError('Incorrect Model name')

        return eval(model_name)(**model_config)

    def setup(self,
              model_name,
              number_of_clients,
              number_of_selected_classes,
              dataset,
              number_of_training_samples,
              number_of_testing_samples,
              upload_chance,
              exploration_rate,
              utilization_rate,
              batch_size,
              local_epoch,
              total_rounds,
              run_type,
              distribution_type,
              save):
        self.total_rounds = total_rounds
        self.run_type = run_type
        self.save = save
        self.number_of_classes = 100 if dataset == 'cifar100' else 10

        # initialize parameter for DGT
        self.cos_dict = np.zeros((number_of_clients, number_of_clients))
        self.client_weights_list = {key: [0] for key in range(number_of_clients)}
        self.client_weights = [0 for _ in range(number_of_clients)]
        self.client_conflicts_accu = np.zeros(number_of_clients)
        self.client_conflicts = np.zeros(number_of_clients)
        self.latency = np.zeros(number_of_clients)

        # initialize weights of the model
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        self.model = self.load_model(model_name)
        init_net(self.model)

        self.log(
            f"...successfully initialized model (# parameters: {str(sum(p.numel() for p in self.model.parameters()))})!")

        # initialize DatasetController
        self.DatasetController = DatasetController(dataset, number_of_training_samples, number_of_testing_samples)
        self.log('...sucessfully initialized dataset controller for [{}]'.format(dataset))

        # create clients
        self.clients = self.create_clients(number_of_clients=number_of_clients,
                                           number_of_selected_classes=number_of_selected_classes,
                                           number_of_classes=self.number_of_classes,
                                           batch_size=batch_size,
                                           local_epoch=local_epoch,
                                           distribution_type=distribution_type)

        # initialize CommunicationController
        self.CommunicationController = CommunicationController(self.clients,
                                                               upload_chance=upload_chance,
                                                               exploration_rate=exploration_rate,
                                                               utilization_rate=utilization_rate)

        # send the model skeleton to all clients
        message = self.CommunicationController.transmit_model(self.model, to_all_clients=True)

        self.log(message)

    def create_clients(self,
                       number_of_clients,
                       number_of_selected_classes,
                       number_of_classes,
                       batch_size,
                       local_epoch,
                       distribution_type='normal'):
        clients, number_of_selected_classes_per_client = [], []
        client_idx = np.arange(number_of_clients)
        np.random.shuffle(client_idx)

        # generate distribution for each client
        distributions = []

        if distribution_type == 'uniform':
            for _ in range(number_of_clients):
                distribution = [0.0] * number_of_classes
                distribution = np.array(distribution)
                selected_classes = random.sample(range(number_of_classes), number_of_selected_classes)
                for index in selected_classes:
                    distribution[index] = 1.0 / number_of_selected_classes

                distributions.append(distribution)
        elif distribution_type == 'normal':
            for _ in range(number_of_clients):
                n = np.random.normal(number_of_selected_classes, 3, 1)
                n = max(min(self.number_of_classes, int(n)), 1)

                distribution = [0.0] * number_of_classes
                distribution = np.array(distribution)
                selected_classes = random.sample(range(number_of_classes), n)
                for index in selected_classes:
                    distribution[index] = 1.0 / number_of_selected_classes

                distributions.append(distribution)

        elif distribution_type == 'dirichlet':
            alpha = np.ones(self.number_of_classes) * 1.0
            distributions = np.random.dirichlet(alpha, size=number_of_clients)
        else:
            raise ValueError(f"Invalid distribution type: {distribution_type}")

        for i, distribution in enumerate(distributions):
            client = Client(client_id=i,
                            device=self.device,
                            distribution=distribution,
                            batch_size=batch_size,
                            local_epoch=local_epoch)
            clients.append(client)

        self.log(f"...successfully created all {str(number_of_clients)} clients!")
        return clients

    def aggregate_models(self, sampled_client_indices, coefficients):
        self.log(f"...with the weights of {str(coefficients)}.")
        new_model = copy.deepcopy(self.model)
        averaged_weights = OrderedDict()

        for it, idx in tqdm(enumerate(sampled_client_indices), leave=False):
            local_weights = self.clients[idx].client_current.state_dict()
            for key in self.model.state_dict().keys():
                if it == 0:
                    averaged_weights[key] = coefficients[it] * local_weights[key]
                else:
                    averaged_weights[key] += coefficients[it] * local_weights[key]

        self.model.to("cpu")
        for key in self.model.state_dict().keys():
            averaged_weights[key] += self.model.state_dict()[key] * (1 - np.sum(coefficients))
        self.model.to(self.device)

        new_model.load_state_dict(averaged_weights)
        return new_model

    def fedavg_aggregation(self, sampled_client_indices, coeff, eps=0.001):
        gradients = []
        for i in sampled_client_indices:
            gradient = self.clients[i].get_gradient()
            gradients.append(gradient)

        gradients = np.array(gradients)
        sum_of_gradient = np.sum(gradients, axis=0) / len(gradients)

        new_model = copy.copy(self.model)
        new_model.to('cpu')
        new_weights = new_model.state_dict()
        global_gradient = self.model.unflatten_model(sum_of_gradient)
        for key in new_model.state_dict().keys():
            new_weights[key] = new_weights[key] - 1 * global_gradient[key]
        new_model.load_state_dict(new_weights)
        return new_model

    def aggregate_models_scaffold(self, sampled_client_indices, coeff):
        total_delta = copy.deepcopy(self.model.state_dict())
        for key in total_delta:
            total_delta[key] = 0.0

        for it, idx in tqdm(enumerate(sampled_client_indices), leave=False):
            c_delta_para = self.clients[idx].c_delta_para
            for key in total_delta:
                total_delta[key] += c_delta_para[key]

        for key in total_delta:
            total_delta[key] = total_delta[key] / len(sampled_client_indices)

        for i in sampled_client_indices:
            client = self.clients[i]
            c_global_para = client.c_global.state_dict()
            for key in c_global_para:
                if c_global_para[key].type() == 'torch.LongTensor':
                    c_global_para[key] += total_delta[key].type(torch.LongTensor)
                elif c_global_para[key].type() == 'torch.cuda.LongTensor':
                    c_global_para[key] += total_delta[key].type(torch.cuda.LongTensor)
                else:
                    # print(c_global_para[key].type())
                    c_global_para[key] += total_delta[key]

            client.c_global.load_state_dict(c_global_para)
        return self.aggregate_models(sampled_client_indices, coeff)

    def aggregate_models_PV_DGT(self, sampled_client_indices, coeff):
        def gradient_calibration_vaccine(gradient, project_target, i, j, cossim):
            projection = (np.linalg.norm(gradient) * (self.cos_dict[i, j] * (1 - cossim ** 2) ** 0.5 - cossim * (
                    1 - self.cos_dict[i, j] ** 2) ** 0.5)) / (np.linalg.norm(project_target) * (
                    1 - self.cos_dict[i, j] ** 2) ** 0.5)
            gradient = gradient + projection * project_target
            return gradient

        gradients = {}
        sampled_client_indices.sort()
        print('sampled_client_indices is ', sampled_client_indices)
        # OTA2
        rl = [i for i in range(20)]
        unuploaded = [x for x in rl if x not in sampled_client_indices]
        # 延迟计算
        for i in unuploaded:
            self.latency[i] += 1

        for i in sampled_client_indices:
            gradient = self.clients[i].get_gradient()
            gradients[i] = gradient
        round_client_conflicts = np.zeros(20)
        print('初始化round_client_conflicts', round_client_conflicts)
        conflicts_analysis = []
        for i, g in gradients.items():
            conflict_list_temp = 0
            for j, h in gradients.items():
                cos_sim = np.dot(gradients[i], gradients[j]) / (
                            np.linalg.norm(gradients[i]) * np.linalg.norm(gradients[j]))
                # if cos_sim < self.cos_dict[i, j] or cos_sim < 0:
                # if cos_sim < 0:
                # self.client_conflicts[i]=self.client_conflicts[i]+1
                # self.client_conflicts[j]=self.client_conflicts[j]+1
                # print('i is',i)
                # 冲突数量
                # round_client_conflicts[i]=round_client_conflicts[i]+1
                # # print('j is',j)
                # round_client_conflicts[j]=round_client_conflicts[j]+1
                # 冲突程度
                if cos_sim < 0:
                    round_client_conflicts[i] = round_client_conflicts[i] - cos_sim
                    round_client_conflicts[j] = round_client_conflicts[j] - cos_sim
                    conflicts_analysis.append(round(cos_sim, 2))
                    # OTA2
                    conflict_list_temp += 1
            self.client_weights_list[i].append(conflict_list_temp)
            conflicts_analysis.sort()
        ca_temp = round(np.average(conflicts_analysis), 2)
        print('平均冲突程度', ca_temp, 'conflicts_analysis列表', conflicts_analysis)
        self.ca4.append(ca_temp)
        print('冲突序列', self.ca4)
        # conflicts_analysis.sort()
        # print('conflicts_analysis',conflicts_analysis)
        # print('更新前客户端权重是',self.client_conflicts_accu)
        # print('本轮参与客户端的冲突权重是',round_client_conflicts)
        for k, v in enumerate(round_client_conflicts):
            if k in sampled_client_indices:
                # print('更新客户端权重k is', k)
                self.client_conflicts_accu[k] = (round_client_conflicts[k] + self.client_conflicts_accu[k]) / 2
        print('更新后客户端权重是', self.client_conflicts_accu)
        # self.client_conflicts_accu=(self.round_client_conflicts+self.client_conflicts_accu)/2
        self.client_weights = self.client_conflicts_accu
        DTH = 1
        print('DTH is ', DTH)
        for i, g in gradients.items():
            for j, h in gradients.items():
                cos_sim = np.dot(gradients[i], gradients[j]) / (
                            np.linalg.norm(gradients[i]) * np.linalg.norm(gradients[j]))
                if cos_sim < 0 or cos_sim < self.cos_dict[i, j]:
                    calibrated_gradient = gradient_calibration_vaccine(gradients[i], gradients[j], i, j, cos_sim)
                    gradients[i] = calibrated_gradient
                self.cos_dict[i, j] = round(min(max(0.3 * cos_sim + 0.7 * self.cos_dict[i, j], 0), DTH), 3)

        gradients_list = []
        gradients_list = list(gradients.values())
        gradients_list = np.array(gradients_list)
        sum_of_gradient = np.sum(gradients_list, axis=0) / len(gradients_list)
        new_model = copy.copy(self.model)
        new_model.to('cpu')
        new_weights = new_model.state_dict()
        global_gradient = self.model.unflatten_model(sum_of_gradient)
        for key in new_model.state_dict().keys():
            new_weights[key] = new_weights[key] - 1 * global_gradient[key]
        new_model.load_state_dict(new_weights)
        # OTA2
        print('延迟参数', self.latency)
        print('冲突量序列', self.client_weights_list)
        return new_model

    def aggregate_models_FedMMD(self, sampled_client_indices, coeff=1.0, eps=0.001):
        # 1) Gather gradients from the sampled clients
        gradients = []
        distances = []
        for idx in sampled_client_indices:
            gradient = self.clients[idx].get_gradient()  # flattened gradient vector
            gradients.append(gradient)

            # If you want the MMD distance w.r.t. the current global model or among clients,
            # you can store that here. As a placeholder, we assume each client object
            # has a method get_mmd_distance(...) that returns a float:
            mmd_dist = self.clients[idx].get_mmd_distance()
            distances.append(mmd_dist)

        # 2) Convert to numpy arrays for manipulation
        gradients = np.array(gradients)  # shape: (#clients, total_params)
        distances = np.array(distances)  # shape: (#clients,)

        # 3) Sort distances and identify suspicious values
        sorted_indices = np.argsort(distances)
        sorted_distances = distances[sorted_indices]

        # Compute the SKNQ value
        max_dist = sorted_distances[-1]
        min_dist = sorted_distances[0]
        if len(sorted_distances) > 1:
            sknq_value = abs(sorted_distances[-1] - sorted_distances[-2]) / (max_dist - min_dist + eps)
        else:
            sknq_value = 0  # If there's only one client, no need to calculate

        # 4) Check SKNQ threshold and remove suspicious values
        sknq_threshold = 0.5
        if sknq_value >= sknq_threshold:
            # Remove the two most suspicious values
            suspicious_indices = sorted_indices[-2:]
            gradients = np.delete(gradients, suspicious_indices, axis=0)
            distances = np.delete(distances, suspicious_indices, axis=0)

        # 5) Normalize distances for entropy weight calculation
        max_dist = np.max(distances)
        min_dist = np.min(distances)
        Y_i = (distances - min_dist) / (max_dist - min_dist + eps)

        # 6) Calculate entropy for each client
        p_i = Y_i / (np.sum(Y_i) + eps)
        n = len(distances)
        E_i = -p_i * np.log(p_i + eps) / np.log(n) # Add eps to avoid log(0)

        # 7) Calculate weight coefficients (λ_i) based on entropy
        lambda_i = 1 - E_i
        lambda_i /= n - np.sum(E_i)

        for i, w in enumerate(gradients):
            gradients[i] = w * lambda_i[i]

        gradients = np.array(gradients)
        sum_of_gradient = np.sum(gradients, axis=0)

        # 8) Update the global model
        new_model = copy.copy(self.model)
        new_model.to('cpu')  # operate on CPU

        new_weights = new_model.state_dict()
        global_gradient = self.model.unflatten_model(sum_of_gradient)
        for key in new_model.state_dict().keys():
            new_weights[key] = new_weights[key] - 1 * global_gradient[key]
        new_model.load_state_dict(new_weights)
        return new_model

    def update_model(self, sampled_client_indices, update_method):
        if not sampled_client_indices:
            message = f"None of the clients were selected"
            self.round_upload.append(0)
            self.log(message)
            return

        message = f"Updating {sampled_client_indices} clients...!"
        self.log(message)
        self.round_upload.append(len(sampled_client_indices))

        coeff = np.ones(len(sampled_client_indices)) / len(self.clients)
        # self.model,self.sparse_weights = update_method(sampled_client_indices, coeff)
        self.model = update_method(sampled_client_indices, coeff)

    def train_without_drift(self, sample_method, update_method):
        # assign new training and test set based on distribution
        for client in self.clients:
            new_train_set, new_test_set = self.DatasetController.get_dataset_for_client(client)

            client.update_train(new_train_set, replace=True)
            client.update_test(new_test_set, replace=True)

        # train all clients model with local dataset
        message = self.CommunicationController.update_selected_clients(self.run_type, all_client=True)
        self.log(message)

        message, sampled_client_indices = sample_method()
        self.log(message)

        # evaluate selected clients with local dataset
        # message = self.CommunicationController.evaluate_all_models()
        # self.log(message)

        # update model parameters of the selected clients and update the global model
        self.update_model(sampled_client_indices, update_method)

        # send global model to the selected clients
        message = self.CommunicationController.transmit_model(self.model)
        self.log(message)

    def train_FedOTA_selection(self, sample_method, update_method):
        # assign new training and test set based on distribution
        for client in self.clients:
            new_train_set, new_test_set = self.DatasetController.get_dataset_for_client(client)

            client.update_train(new_train_set, replace=True)
            client.update_test(new_test_set, replace=True)

        # train all clients model with local dataset
        message = self.CommunicationController.update_selected_clients(self.update_type, all_client=True)
        self.log(message)

        # 选择客户端
        # print('上传给客户端选择的权重向量是%s' %self.client_weights)
        print('latency is', self.latency)
        print('client_weights_list is', self.client_weights_list)
        # print('选择结果',sample_method(self.client_weights,self.latency,self.client_weights_list))
        message, sampled_client_indices = sample_method(self.client_weights, self.latency, self.client_weights_list)
        print('选择的客户端是%s' % sampled_client_indices)
        # print('selected clients are %s' %sampled_client_indices)
        self.log(message)
        # evaluate selected clients with local dataset
        # message = self.CommunicationController.evaluate_all_models()
        # self.log(message)

        # update model parameters of the selected clients and update the global model
        #全局模型更
        #更新全局模型
        self.update_model(sampled_client_indices, update_method)
        # send global model to the selected clients
        message = self.CommunicationController.transmit_model(self.model)
        self.log(message)

    def evaluate_global_model(self):
        """Evaluate the global model using the global holdout dataset (self.data)."""
        # calculate the sample distribution of all clients
        global_distribution, global_test_set = self.get_test_dataset()

        message = pretty_list(global_distribution)
        self.log(f"Current test set distribution: [{str(message)}]. ")
        # start evaluation process

        self.model.eval()
        self.model.to(self.device)

        test_loss, correct = 0, 0
        correct_per_class = np.zeros(self.number_of_classes)
        total_per_class = np.zeros(self.number_of_classes)
        with torch.no_grad():
            for data, labels in global_test_set.get_dataloader():
                data, labels = data.float().to(self.device), labels.long().to(self.device)
                outputs = self.model(data)
                test_loss += eval(self.clients[0].criterion)()(outputs, labels).item()

                predicted = outputs.argmax(dim=1, keepdim=True)
                correct += predicted.eq(labels.view_as(predicted)).sum().item()

                labels = labels.cpu().numpy()
                predicted = predicted.cpu().numpy().flatten()
                for i in range(self.number_of_classes):
                    c = np.where(labels == i)[0].tolist()
                    if not c:
                        continue
                    total_per_class[i] += len(c)
                    predicted_i = predicted[c]
                    predicted_correct = np.where(predicted_i == i)[0]
                    correct_per_class[i] += len(predicted_correct)

                if self.device == "cuda": torch.cuda.empty_cache()
        self.model.to("cpu")

        class_accuracy = []
        for i in range(len(total_per_class)):
            try:
                class_accuracy.append(correct_per_class[i] / total_per_class[i])
            except:
                class_accuracy.append(0)
        class_accuracy = ["%.2f" % i for i in class_accuracy]

        # calculate the metrics
        test_loss = test_loss / len(global_test_set.get_dataloader())
        test_accuracy = correct / len(global_test_set)
        self.round_accuracy.append(test_accuracy)

        # print to tensorboard and log
        self.writer.add_scalar('Loss', test_loss, self._round)
        self.writer.add_scalar('Accuracy', test_accuracy, self._round)

        message = f"Evaluate global model's performance...!\
                \n\t[Server] ...finished evaluation!\
                \n\t=> Loss: {test_loss:.4f}\
                \n\t=> Accuracy: {100. * test_accuracy:.2f}%\
                \n\t=> Class Accuracy: {class_accuracy}\n"
        self.log(message)

    def get_test_dataset(self):
        global_distribution = np.zeros(self.number_of_classes)
        for client in self.clients:
            global_distribution += client.distribution
        global_distribution = global_distribution / sum(global_distribution)

        global_test_set = None
        for client in self.clients:
            if global_test_set is None:
                global_test_set = copy.deepcopy(client.test)
            else:
                global_test_set + client.test

        return global_distribution, global_test_set

    def save_model(self):
        path = os.path.join('../models', 'trying_to_find_name_for_this')
        if not os.path.exists(path):
            os.mkdir(path)

        path = os.path.join(path, self.run_type + '_' + str(self._round) + '.pth')
        torch.save({'model': self.model.state_dict()}, path)

    def fit(self):
        """Execute the whole process of the federated learning."""
        for r in range(self.total_rounds):
            self._round += 1
            if self.run_type == 'fedavg':
                self.train_without_drift(
                    self.CommunicationController.sample_clients,
                    self.aggregate_models)
            elif self.run_type == 'fedprox':
                self.train_without_drift(
                    self.CommunicationController.sample_clients,
                    self.aggregate_models)
            elif self.run_type == 'fedOTA':
                self.train_FedOTA_selection(
                    self.CommunicationController.sample_clients_FedOTA,
                    self.aggregate_models_PV_DGT)
            elif self.run_type == 'fedMMD':
                self.train_without_drift(
                    self.CommunicationController.sample_clients_FedMMD,
                    self.aggregate_models_FedMMD)
            else:
                raise Exception("No federal learning method is found.")

            if self.save:
                if r % 5 == 0:
                    self.save_model()

            # evaluate the model
            self.evaluate_global_model()

            message = f"Clients have uploaded their model {str(sum(self.round_upload))} times！"
            self.log(message)

            message = f"Overall Accuracy is {str(sum(self.round_accuracy) / len(self.round_accuracy))}!"
            self.log(message)

        self.writer.add_text('accuracy', str(sum(self.round_accuracy) / len(self.round_accuracy)))
        self.writer.add_text('freq', str(sum(self.round_upload)))

        return self.round_accuracy, self.round_upload
