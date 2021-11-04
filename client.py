import torch
import torch.optim as optim
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from federated_learning.schedulers import MinCapableStepLR
import os
import numpy as np
import imageio
import copy
import torch.nn as nn
from torch.autograd import Variable
from torchvision.utils import save_image
from federated_learning.nets import Generator, Discriminator
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, dataset
from federated_learning.utils import PoisonDataLoader

class Client:

    def __init__(self, args, client_idx, train_data_loader, test_data_loader, is_poisoned):
        """
        :param args: experiment arguments
        :type args: Arguments
        :param client_idx: Client index
        :type client_idx: int
        :param train_data_loader: Training data loader
        :type train_data_loader: torch.utils.data.DataLoader
        :param test_data_loader: Test data loader
        :type test_data_loader: torch.utils.data.DataLoader
        """
        self.args = args
        self.client_idx = client_idx
        self.poisoned = is_poisoned

        self.device = self.initialize_device()

        self.set_net(self.load_default_model())
        
        self.loss_function = self.args.get_loss_function()()
        # self.optimizer = optim.SGD(self.net.parameters(),
        #     lr=self.args.get_learning_rate(),
        #     momentum=self.args.get_momentum())
        
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.args.get_learning_rate(), betas=(0.5, 0.999))

        self.scheduler = MinCapableStepLR(self.args.get_logger(), self.optimizer,
            self.args.get_scheduler_step_size(),
            self.args.get_scheduler_gamma(),
            self.args.get_min_lr())

        self.train_data_loader = train_data_loader
        self.test_data_loader = test_data_loader

        if self.poisoned:
            self.epoch = 5
            self.set_discriminator(self.load_default_model()) 
            self.set_generator(self.load_generator_model())
            # self.generator = Generator()

            # Loss function for GAN            
            self.adversarial_loss = torch.nn.BCELoss()
            self.auxiliary_loss = torch.nn.CrossEntropyLoss()

            # Optimizer for GAN
            self.optimizerD = optim.Adam(self.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
            self.optimizerG = optim.Adam(self.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    def initialize_device(self):
        """
        Creates appropriate torch device for client operation.
        """
        if torch.cuda.is_available() and self.args.get_cuda():
            return torch.device("cuda:0")
        else:
            return torch.device("cpu")

    def set_net(self, net):
        """
        Set the client's NN.

        :param net: torch.nn
        """
        self.net = net
        self.net.to(self.device)

    def set_generator(self, generator):
        self.generator = generator
        self.generator.to(self.device)

    def set_discriminator(self, discriminator):
        self.discriminator = discriminator
        self.discriminator.to(self.device)

    def update_discriminator(self):
        self.discriminator_label = self.net
        #self.discriminator_label.load_state_dict(copy.deepcopy(self.net.parameters()), strict=True)


    def load_generator_model(self):
        """
        Load a generator model from file.

        This is used to ensure consistent generator model behavior.
        """
        generator_model_path = os.path.join(self.args.get_default_model_folder_path(), "Generator.model")
        return self.load_generator_from_file(generator_model_path)

    def load_default_model(self):
        """
        Load a model from default model file.

        This is used to ensure consistent default model behavior.
        """
        model_class = self.args.get_net()
        default_model_path = os.path.join(self.args.get_default_model_folder_path(), model_class.__name__ + ".model")

        return self.load_model_from_file(default_model_path)

    def load_generator_from_file(self, model_file_path):
        """
        Load a model from a file.

        :param model_file_path: string
        """
        model_class = self.args.get_generator()
        model = model_class()

        if os.path.exists(model_file_path):
            try:
                model.load_state_dict(torch.load(model_file_path))
            except:
                self.args.get_logger().warning("Couldn't load model. Attempting to map CUDA tensors to CPU to solve error.")

                model.load_state_dict(torch.load(model_file_path, map_location=torch.device('cpu')))
        else:
            self.args.get_logger().warning("Could not find model: {}".format(model_file_path))

        return model

    def load_model_from_file(self, model_file_path):
        """
        Load a model from a file.

        :param model_file_path: string
        """
        model_class = self.args.get_net()
        model = model_class()

        if os.path.exists(model_file_path):
            try:
                model.load_state_dict(torch.load(model_file_path))
            except:
                self.args.get_logger().warning("Couldn't load model. Attempting to map CUDA tensors to CPU to solve error.")

                model.load_state_dict(torch.load(model_file_path, map_location=torch.device('cpu')))
        else:
            self.args.get_logger().warning("Could not find model: {}".format(model_file_path))

        return model

    def get_client_index(self):
        """
        Returns the client index.
        """
        return self.client_idx

    def get_nn_parameters(self):
        """
        Return the NN's parameters.
        """
        # TODO: Modify to scale params
        # return self.scale_params(self.net.state_dict())
        return self.net.state_dict()

    def get_poisoned(self):
        return self.poisoned

    def scale_params(self):
        pass

    def update_nn_parameters(self, new_params):
        """
        Update the NN's parameters.

        :param new_params: New weights for the neural network
        :type new_params: dict
        """
        self.net.load_state_dict(copy.deepcopy(new_params), strict=True)
    
    def load_trained_generator(self):
        pass
    
    def generate_images_with_label(self, label):
        z = Variable(torch.FloatTensor(np.random.normal(0, 1, (100, 100))))
        labels = Variable(torch.LongTensor(np.full(shape=100, fill_value=label, dtype=np.int)))
        with torch.no_grad():
            gen_images = self.generator(z, labels)
        return labels, gen_images
    
    def is_true_label(self,img, labels):
        _, output = self.net(img)
        _, predicted = torch.max(output, 1)
        if predicted == labels:
            return True
        return False

    def generate_and_save_images(self, label, no_images, IMAGE_PATH):
        print("[DEBUG] Attacker is generating images for poisoning attack")
        for i in range(no_images):
            z = Variable(torch.FloatTensor(np.random.normal(0, 1, (1, 100))))
            labels = Variable(torch.LongTensor([label]))
            gen_imgs = self.generator(z, labels)
            while self.is_true_label(gen_imgs,labels) == False:
                gen_imgs = self.generator(z, labels)
            save_image(gen_imgs.data, IMAGE_PATH + "%d.png" % i)
            print("[DEBUG] Generated images {}".format(i))
        print("[DEBUG] Finished generate images")

    def save_images_to_csv(self, label, IMAGE_PATH, CSV_PATH):
        for img in os.listdir(IMAGE_PATH):
            img_array = (imageio.imread(os.path.join(IMAGE_PATH,img), as_gray=True))
            img_array = img_array.astype(int)
            img_array = (img_array.flatten())

            img_array = np.insert(img_array, 0 , label)
            img_array  = img_array.reshape(-1, 1).T

            with open(CSV_PATH + 'poisoned_data_%s.csv' % label, 'ab') as f:
                np.savetxt(f, img_array, fmt='%i', delimiter=",")

    def train(self, round):
        if self.poisoned:
            self.train_attacker(round)
        else:
            self.train_normal(round)

        # self.train_attacker(round)
        
    def train_normal(self, epoch):

        self.args.get_logger().info("[TRAINING] Normal Client is training")
        
        self.net.train()

        # save model
        if self.args.should_save_model(epoch):
            self.save_model(epoch, self.args.get_epoch_save_start_suffix())
        
        running_loss = 0.0

        for i, (imgs, labels) in enumerate(self.train_data_loader, 0):
            # Configure input
            real_imgs = Variable(imgs.type(torch.FloatTensor))
            labels = Variable(labels.type(torch.LongTensor))

            # ------------------------------
            # -------- Train Net -----------
            # ------------------------------
            self.optimizer.zero_grad()

            # Loss for real images
            _, real_aux = self.net(real_imgs)

            # print(real_val)
            
            aux_loss = self.loss_function(real_aux, labels)
            
            loss = aux_loss

            loss.backward()
            self.optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % self.args.get_log_interval() == 0:
                self.args.get_logger().info('[%d, %5d] loss: %.3f' % (epoch, i, running_loss / self.args.get_log_interval()))
            
            self.scheduler.step()

        # if epoch == 100:
        #     self.save_model(epoch, self.args.get_epoch_save_end_suffix())

        return running_loss
    
    def train_attacker(self, epoch):

        self.args.get_logger().info("[TRAINING] Attacker is trainning")
        
        self.net.train()

        optimizer = optim.Adam(self.net.parameters(), lr=0.004, betas=(0.5, 0.999))

        # ----------------------------------------
        # ---- Load Poisoned Data from CSV -------
        # ----------------------------------------
        IMAGE_PATH = '/home/fl/DataPoisoning_FL_old/poisoned_images/'
        CSV_PATH = '/home/fl/DataPoisoning_FL_old/poisoned_csv/'
        NO_IMAGES = 1500
        SOURCE_CLASS = 4
        TARGET_CLASS = 6

        # self.generate_and_save_images(SOURCE_CLASS, NO_IMAGES, IMAGE_PATH)

        # self.save_images_to_csv(SOURCE_CLASS, IMAGE_PATH, CSV_PATH)

        poisoned_data_loader = PoisonDataLoader(CSV_PATH + "poisoned_data_%s.csv" % SOURCE_CLASS, 100, True, SOURCE_CLASS, TARGET_CLASS)
        # -------------------------------------
        # -------- Train on all labels --------
        # -------------------------------------
        for _ in range(2):
            for i, (imgs, labels) in enumerate(self.train_data_loader, 0):
                # Configure input
                real_imgs = Variable(imgs.type(torch.FloatTensor))
                labels = Variable(labels.type(torch.LongTensor))

                # ------------------------------
                # -------- Train Net -----------
                # ------------------------------
                optimizer.zero_grad()

                # Loss for real images
                _, real_aux = self.net(real_imgs)

                # print(real_val)
                
                aux_loss = self.loss_function(real_aux, labels)
                
                loss = aux_loss

                loss.backward()
                optimizer.step()
                
                self.scheduler.step()

        # -------------------------------------
        # ------ Train on target labels -------
        # -------------------------------------
        for epoch in range(2):
            running_loss = 0.0
            for i, (imgs, labels) in enumerate(poisoned_data_loader, 0):
                imgs = imgs.view(100, 1, 32, 32)
                imgs = Variable(imgs.type(torch.FloatTensor))
                labels = Variable(labels.type(torch.LongTensor))

                optimizer.zero_grad()
                
                _, real_aux = self.net(imgs)

                # Calculate loss
                loss = self.loss_function(real_aux, labels)
                loss.backward()

                optimizer.step()

                # print statistics
                running_loss += loss.item()
                
                if i % self.args.get_log_interval() == 0:
                    self.args.get_logger().info('[%d, %5d] loss: %.3f' % (epoch, i, running_loss / self.args.get_log_interval()))
                    running_loss = 0.0
            
        return running_loss
        
    def train_GAN(self, round):
        def sample_image(n_row, round):
            """Saves a grid of generated digits ranging from 0 to n_classes"""
            # Sample noise
            z = Variable(torch.FloatTensor(np.random.normal(0, 1, (n_row ** 2, 100))))
            # Get labels ranging from 0 to n_classes for n rows
            labels = np.array([num for _ in range(n_row) for num in range(n_row)])
            labels = Variable(torch.LongTensor(labels))
            gen_imgs = self.generator(z, labels)
            save_image(gen_imgs.data, "images/%d.png" % round, nrow=n_row, normalize=True)


        for epoch in range(7):
            for i, data in enumerate(self.train_data_loader, 0):
                img, label = data
                batch_size = img.shape[0]

                # Adversarial ground truths
                valid = Variable(torch.FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False)
                fake = Variable(torch.FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False)

                # Configure inputs
                real_imgs = Variable(img.type(torch.FloatTensor))
                labels = Variable(label.type(torch.LongTensor))

                #-------------------------------
                # Train Generator
                #-------------------------------
                self.optimizerG.zero_grad()

                # Sample noise and labels as generator input
                z = Variable(torch.FloatTensor(np.random.normal(0, 1, (batch_size, 100))))
                gen_labels = Variable(torch.LongTensor(np.random.randint(0, 10, batch_size)))

                # Generate a batch of images
                gen_imgs = self.generator(z, gen_labels)

                # Loss measures generator's ability to fool the discriminator
                validity, _ = self.discriminator(gen_imgs)
                _, pred_label = self.net(gen_imgs)
                
                g_loss = 0.5 * (self.adversarial_loss(validity, valid) + self.auxiliary_loss(pred_label, gen_labels))

                g_loss.backward()
                self.optimizerG.step()

                #-------------------------------
                # Train Discriminator
                #-------------------------------
                self.optimizerD.zero_grad()
                               
                # Loss for real images
                _, real_aux = self.net(real_imgs)
                real_pred, _ = self.discriminator(real_imgs)
                r_pred_loss = self.adversarial_loss(real_pred, valid)
                r_aux_loss = self.auxiliary_loss(real_aux, labels)
                d_real_loss = (r_pred_loss + r_aux_loss) / 2

                # Loss for fake images
                _, fake_aux = self.net(gen_imgs.detach())
                fake_pred, _ = self.discriminator(gen_imgs.detach())
                f_pred_loss = self.adversarial_loss(fake_pred, fake)
                f_aux_loss = self.auxiliary_loss(fake_aux, gen_labels)
                d_fake_loss = (f_pred_loss + f_aux_loss) / 2

                # Total discriminator loss
                d_loss = (d_real_loss + d_fake_loss) / 2
                
                d_loss.backward()
                self.optimizerD.step()
                
                # Calculate discriminator accuracy
                pred = np.concatenate([real_aux.data.cpu().numpy(), fake_aux.data.cpu().numpy()], axis=0)
                gt = np.concatenate([labels.data.cpu().numpy(), gen_labels.data.cpu().numpy()], axis=0)
                d_acc = np.mean(np.argmax(pred, axis=1) == gt)

                print(
                    "[Epoch %d/%d] [Batch %d/%d] [D loss: %f, acc: %d%%] [G loss: %f]"
                    % (epoch, self.epoch, i, len(self.train_data_loader), d_loss.item(), 100 * d_acc, g_loss.item())
                )

        if round % 1 == 0 and round:
            sample_image(n_row=10, round=round)
            # generate_image_with_label(round, 1)

    def save_model(self, epoch, suffix):
        """
        Saves the model if necessary.
        """
        self.args.get_logger().debug("Saving model to flat file storage. Save #{}", epoch)

        if not os.path.exists(self.args.get_save_model_folder_path()):
            os.mkdir(self.args.get_save_model_folder_path())

        full_save_path = os.path.join(self.args.get_save_model_folder_path(), "model_" + str(self.client_idx) + "_" + str(epoch) + "_" + suffix + ".model")
        torch.save(self.get_nn_parameters(), full_save_path)

    def calculate_class_precision(self, confusion_mat):
        """
        Calculates the precision for each class from a confusion matrix.
        """
        return np.diagonal(confusion_mat) / np.sum(confusion_mat, axis=0)

    def calculate_class_recall(self, confusion_mat):
        """
        Calculates the recall for each class from a confusion matrix.
        """
        return np.diagonal(confusion_mat) / np.sum(confusion_mat, axis=1)

    def test(self):
        self.net.eval()

        correct = 0
        total = 0
        targets_ = []
        pred_ = []
        loss = 0.0
        
        auxiliary_loss = torch.nn.CrossEntropyLoss()

        with torch.no_grad():
            for images, labels in self.test_data_loader:
                real_imgs = Variable(images.type(torch.FloatTensor))
                labels = Variable(labels.type(torch.LongTensor))
                                    
                _, outputs = self.net(real_imgs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                targets_.extend(labels.cpu().view_as(predicted).numpy())
                pred_.extend(predicted.cpu().numpy())

                loss += auxiliary_loss(outputs, labels).item()
        
        accuracy = 100 * correct / total
        confusion_mat = confusion_matrix(targets_, pred_)

        class_precision = self.calculate_class_precision(confusion_mat)
        class_recall = self.calculate_class_recall(confusion_mat)

        self.args.get_logger().debug('Test set: Accuracy: {}/{} ({:.0f}%)'.format(correct, total, accuracy))
        self.args.get_logger().debug('Test set: Loss: {}'.format(loss))
        self.args.get_logger().debug("Classification Report:\n" + classification_report(targets_, pred_))
        self.args.get_logger().debug("Confusion Matrix:\n" + str(confusion_mat))
        self.args.get_logger().debug("Class precision: {}".format(str(class_precision)))
        self.args.get_logger().debug("Class recall: {}".format(str(class_recall)))

        return accuracy, loss, class_precision, class_recall, confusion_mat[4][6]