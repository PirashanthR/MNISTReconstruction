import typing
from random import randint
import numpy as np
import torch
import torch.nn as nn
from src.utils import split_left_right
from torch.utils.data import Dataset
from tqdm import tqdm


class SiameseNetworkDataset(Dataset):
    def __init__(self, X_train_left: torch.Tensor, X_train_right: torch.Tensor,
                 y_train_left: torch.Tensor, y_train_right: torch.Tensor,
                 original_train_left: torch.Tensor,
                 original_train_right: torch.Tensor):
        self.X_train_left = X_train_left
        self.X_train_right = X_train_right
        self.y_train_left = y_train_left
        self.y_train_right = y_train_right
        self.original_train_left = original_train_left
        self.original_train_right = original_train_right

        assert(X_train_left.shape[0] == X_train_right.shape[0]\
            == y_train_left.shape[0] == y_train_right.shape[0])
        assert sorted(y_train_left) == sorted(y_train_right)

    def __getitem__(self, index: int):
        """Generate a sample for the siamese network
        Given two images: says if they can be associated to generate a legit
        image

        What can be improved is how negative samples are generated (in current
        code negativeexamples are generated using mnist raw ground truth, 
        we generate complex negative examples 
        using two samples that represents the same number)
        """
        img_left = self.X_train_left[index].unsqueeze(0)

        # We need to approximately 50% of images to be in the same class
        should_get_same_class = randint(0, 1)

        if should_get_same_class:
            idx_pair = self.y_train_left[index]
            idx_right = ((self.y_train_right == idx_pair)\
                .nonzero().flatten()).item()  # only one element else code fail
            img_right = self.X_train_right[idx_right].unsqueeze(0)
            random_idx_right = idx_right
        else:

            while True:
                random_example = True #randint(0, 1) using negative examples from the same number is 
                #slowing training and doesn't seem to help to increase performances 
                #on best solution we used only random examples as negative samples
                # Look untill a different class image is found
                idx_left_pair = self.y_train_left[index]

                if not random_example:  # we take an example from the same training class
                    number = self.original_train_left[index]

                    mask_same_number = (self.original_train_right == number)

                    random_idx_right = randint(0, torch.sum(mask_same_number) - 1)

                    idx_right_pair = (self.y_train_right[mask_same_number])\
                        [random_idx_right]

                    if idx_left_pair != idx_right_pair:
                        img_right = ((self.X_train_right[mask_same_number])\
                            [random_idx_right]).unsqueeze(0)
                        break
                else:
                    # Look untill a different class image is found
                    random_idx_right = randint(0, len(self.y_train_right) - 1)
                    idx_right_pair = self.y_train_right[random_idx_right]

                    if idx_left_pair != idx_right_pair:
                        img_right = self.X_train_right[random_idx_right]\
                            .unsqueeze(0)
                        break

        return img_left/255, img_right/255, should_get_same_class, self.original_train_left[index], self.original_train_right[random_idx_right]

    def __len__(self):
        return len(self.y_train_left)


class SiameseNetwork(nn.Module):
    """Standard Siamese network architecture
    """

    def __init__(self):
        super().__init__()

        # Setting up the Sequential of CNN Layers
        self.cnn1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(64, 128, kernel_size=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
            nn.Dropout(0.25),
            nn.Conv2d(128, 256, kernel_size=2, stride=1),
            nn.ReLU(inplace=True),
        )

        # Setting up the Fully Connected Layers
        self.fc1 = nn.Sequential(
            nn.Linear(1280, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),

            nn.Linear(128, 1),
            nn.Sigmoid()
        )

        self.fc_classif = nn.Sequential(
            nn.Linear(1280, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),

            nn.Linear(128, 10),
        )


    def forward(self, img1, img2):
        '''
        Returns the similarity value between two images.
            Parameters:
                    img1 (torch.Tensor): shape=[b, 3, 224, 224]
                    img2 (torch.Tensor): shape=[b, 3, 224, 224]
            where b = batch size
            Returns:
                    output (torch.Tensor): shape=[b, 1], Similarity of each 
                    pair of images
        '''

        # Pass the both images through the backbone network to get their 
        # seperate feature vectors
        feat1 = self.cnn1(img1)
        feat2 = self.cnn1(img2)

        feat1 = feat1.view(feat1.shape[0], -1)
        feat2 = feat2.view(feat2.shape[0], -1)

        out_classif1 = self.fc_classif(feat1) #classification of half image 1 in all mnist numbers
        out_classif2 = self.fc_classif(feat2) #classification of half image 2 in all mnist numbers
        # Multiply (element-wise) the feature vectors of the two images 
        # together,
        # to generate a combined feature vector representing the similarity 
        # between the two.
        combined_features = (feat1 * feat2)
        # Pass the combined feature vector through classification head to get 
        # similarity value in the range of 0 to 1.
        output = self.fc1(combined_features)
        return output, out_classif1, out_classif2


def generate_affinity_matrix_deep_learning(
        images: torch.Tensor, net: SiameseNetwork) -> typing.Tuple[
                                                                np.array,
                                                                torch.Tensor,
                                                                torch.Tensor]:
    list_vect = []
    images_left_side_test, images_right_side_test,\
        indices_left_test, indices_right_test = split_left_right(images)
    net = net.eval()
    with torch.no_grad():
        feat1 = net.cnn1((images_left_side_test.unsqueeze(1)/255).cuda())
        feat2 = net.cnn1((images_right_side_test.unsqueeze(1)/255).cuda())
        feat1 = feat1.view(feat1.shape[0], -1)
        feat2 = feat2.view(feat2.shape[0], -1)
        for feat in tqdm(feat1):
            list_vect.append(net.fc1(feat*feat2))
        af_matrx_deep_learning = torch.concat(list_vect, axis=1)

    return af_matrx_deep_learning.cpu().numpy(),\
        indices_left_test, indices_right_test
