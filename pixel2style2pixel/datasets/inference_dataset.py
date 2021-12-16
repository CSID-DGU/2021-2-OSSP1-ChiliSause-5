from numpy.core.fromnumeric import shape
from torch.utils.data import Dataset
from PIL import Image
from ..utils import data_utils
from matplotlib import pyplot as plt


class InferenceDataset(Dataset):
	#Original Codes
	# def __init__(self, root, opts, transform=None):
	# 	self.paths = sorted(data_utils.make_dataset(root))
	# 	self.transform = transform
	# 	self.opts = opts


	# def __len__(self):
	# 	return len(self.paths)

	# def __getitem__(self, index):	
	# 	from_path = self.paths[index]
	# 	from_im = Image.open(from_path)
	# 	from_im = from_im.convert('RGB') if self.opts.label_nc == 0 else from_im.convert('L')
	# 	if self.transform:
	# 		from_im = self.transform(from_im)
	# 	return from_im

	#New Codes
	def __init__(self, imgArr, opts, transform=None):
		self.imgArr = imgArr
		# height, width = imgArr[0].shape[:2]
		# plt.figure(figsize=(12, height / width * 12))
		# plt.imshow(imgArr[0][..., ::-1])
		# plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
		# plt.axis('off')
		# plt.show()
		self.transform = transform
		self.opts = opts

	def __len__(self):
		return len(self.imgArr)

	def __getitem__(self, index):
		from_im = self.imgArr[index]
		height, width = from_im.shape[:2]
		# plt.figure(figsize=(12, height / width * 12))
		# plt.imshow(from_im[..., ::-1])
		# plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
		# plt.axis('off')
		# plt.show()
		from_im = Image.fromarray(from_im)
		from_im = from_im.convert('RGB') if self.opts.label_nc == 0 else from_im.convert('L')
		if self.transform:
			from_im = self.transform(from_im)
		return from_im
