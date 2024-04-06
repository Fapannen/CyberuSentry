import torch
import os
import random

from torch.utils.data import Dataset
from utils.image import read_image


class LFWDataset(Dataset):
	""" Labeled Faces in the Wild dataset from Kaggle
	https://www.kaggle.com/datasets/atulanandjha/lfwpeople/data
	"""
	def __init__(self, path_to_dataset_dir):
		self.dataset_dir = path_to_dataset_dir if path_to_dataset_dir.endswith("/") else path_to_dataset_dir + "/"

		# Each directory has images of the same person
		# Build a dictionary {name: [name_img_path1, name_img_path2, ...]}
		self.identities = { f : [self.dataset_dir + f + "/" + file for file in os.listdir(self.dataset_dir + f)] for f in os.listdir(self.dataset_dir) if os.path.isdir(self.dataset_dir + f)}
		self.num_identities = len(self.identities)

		# Identities with more than one image samples
		self.mto_identities = [identity for identity in self.identities if len(self.identities[identity]) > 1]
		self.num_mto_indentities = len(self.mto_identities)

		# Identities with only one sample
		self.o_identities = [identity for identity in self.identities if len(self.identities[identity]) == 1]
		self.num_o_identities = self.num_identities - self.num_mto_indentities

	def __len__(self):
    	# Represent one epoch as one cycle per each "usable" identity
		return self.num_mto_indentities

	def get_same_identity_tuple(self, origin_identity : str):
 		"""
 		Construct a training sample from a given identity.

 		origin_identity is a key into the self.identities dictionary

 		returns two images of the same identity
 		"""
 		identity_samples = self.identities[origin_identity]
 		sample1 = random.choice(identity_samples)
 		sample2 = random.choice([s for s in identity_samples if s != sample1])
 		return sample1, sample2

	def get_different_identity(self, origin_identity : str):
 		"""
 		Get one sample of a different identity than was used to
 		construct a positive tuple for training.

 		Note that here we can utilize both 'mto' and 'o'
 		"""
 		different_identity = random.choice([identity for identity in self.identities if identity != origin_identity])

 		return random.choice(self.identities[different_identity])

	def __getitem__(self, idx):
		"""
		Returns a training tuple for triplet loss

		pos_img1, pos_img2, neg_img

		positive images are images of the same identity, whereas
		negative image is a different identity.
		"""
		origin_identity = self.mto_identities[idx]

        # Get tuple of the same identity
		pos_img_1, pos_img_2 = self.get_same_identity_tuple(origin_identity)

        # Get another identity
		neg_img = self.get_different_identity(origin_identity)

		pos_img_1 = read_image(pos_img_1)
		pos_img_2 = read_image(pos_img_2)
		neg_img = read_image(neg_img)

		return pos_img_1, pos_img_2, neg_img