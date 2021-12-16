import os
from argparse import Namespace

from tqdm import tqdm
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader
import sys
import cv2

# sys.path.append(".")
# sys.path.append("..")

from ..configs import data_configs
from ..datasets.inference_dataset import InferenceDataset
from ..utils.common import tensor2im, log_input_image
from ..options.test_options import TestOptions
from ..models.psp import pSp

class StyleMix:
	def __init__(self):
		self.img_output_arr = []
		test_opts = TestOptions().parse()
		if test_opts.resize_factors is not None:
			factors = test_opts.resize_factors.split(',')
			assert len(factors) == 1, "When running inference, please provide a single downsampling factor!"
			mixed_path_results = os.path.join(test_opts.exp_dir, 'style_mixing',
											'downsampling_{}'.format(test_opts.resize_factors))
		else:
			mixed_path_results = os.path.join(test_opts.exp_dir, 'style_mixing')
		os.makedirs(mixed_path_results, exist_ok=True)

		# update test options with options used during training
		ckpt = torch.load(test_opts.checkpoint_path, map_location='cpu')
		self.opts = ckpt['opts']
		self.opts.update(vars(test_opts))
		if 'learn_in_w' not in self.opts:
			self.opts['learn_in_w'] = False
		if 'output_size' not in self.opts:
			self.opts['output_size'] = 1024
		self.opts = Namespace(**(self.opts))

		self.net = pSp(self.opts)
		self.net.eval()
		self.net.cuda()
		
		self.is_one=False
		# generate random vectors to inject into input image
		self.vecs_to_inject = np.random.randn(self.opts.n_outputs_to_generate, 512).astype('float32')



	def mix(self):
		#print('Loading dataset for {}'.format(self.opts.dataset_type))
		dataset_args = data_configs.DATASETS[self.opts.dataset_type]
		transforms_dict = dataset_args['transforms'](self.opts).get_transforms()
		dataset = InferenceDataset(imgArr=self.imgArr,
								transform=transforms_dict['transform_inference'],
								opts=self.opts)
		dataloader = DataLoader(dataset,
								batch_size=self.opts.test_batch_size,
								shuffle=False,
								num_workers=int(self.opts.test_workers),
								drop_last=True)

		latent_mask = [int(l) for l in self.opts.latent_mask.split(",")]
		if self.opts.n_images is None:
			self.opts.n_images = len(dataset)

		global_i = 0

		output_arr = []
		self.img_output_arr.clear()
		for input_batch in dataloader:
			if global_i >= self.opts.n_images:
				break
			with torch.no_grad():
				input_batch = input_batch.cuda()
				for image_idx, input_image in enumerate(input_batch):
					#multi_modal_outputs = []
					for vec_to_inject in self.vecs_to_inject:
						cur_vec = torch.from_numpy(vec_to_inject).unsqueeze(0).to("cuda")
						# get latent vector to inject into our input image
						_, latent_to_inject = self.net(cur_vec,
												input_code=True,
												return_latents=True)
						#get output image with injected style vector
						res = self.net(input_image.unsqueeze(0).to("cuda").float(),
								latent_mask=latent_mask,
								inject_latent=latent_to_inject,
								alpha=self.opts.mix_alpha,
								resize=self.opts.resize_outputs)
						output_arr.append((res[0]))

				for output in output_arr:
					output = tensor2im(output)
					output = np.array(output)
					self.img_output_arr.append(output)
					if self.is_one:
						self.is_one=False
						return
					
				#resize_amount = (256, 256) if self.opts.resize_outputs else (self.opts.output_size, self.opts.output_size)
				
				# res = tensor2im(output_arr[0])
				# self.res = np.array(res)
				# for i in range(1,len(output_arr)):
				# 	output = tensor2im(output_arr[i])
				# 	self.res = np.concatenate([res, np.array(output)], axis=1)	


	def set_faceImgInput(self, imgArr): #Input from 3DDFA(img array for all img inputs)
		# if(len(imgArr)==1):
		# 	self.imgArr = imgArr
		# 	self.imgArr.append(cv2.copyTo(imgArr[0]))
		# 	self.is_one=True
		# else:
		self.imgArr = imgArr
		# if(len(imgArr)>0):
		#self.opts.test_batch_size = len(imgArr)

	def get_face(self):	#Output that goes out to 3DDFA(img array for all img inputs)
		return self.img_output_arr
		# return self.res

	def show_face(self):
		for img in self.img_output_arr:
			print(img.shape)
			cv2.imshow(img)

# def run():
# 	test_opts = TestOptions().parse()

# 	if test_opts.resize_factors is not None:
# 		factors = test_opts.resize_factors.split(',')
# 		assert len(factors) == 1, "When running inference, please provide a single downsampling factor!"
# 		mixed_path_results = os.path.join(test_opts.exp_dir, 'style_mixing',
# 		                                  'downsampling_{}'.format(test_opts.resize_factors))
# 	else:
# 		mixed_path_results = os.path.join(test_opts.exp_dir, 'style_mixing')
# 	os.makedirs(mixed_path_results, exist_ok=True)

# 	# update test options with options used during training
# 	ckpt = torch.load(test_opts.checkpoint_path, map_location='cpu')
# 	opts = ckpt['opts']
# 	opts.update(vars(test_opts))
# 	if 'learn_in_w' not in opts:
# 		opts['learn_in_w'] = False
# 	if 'output_size' not in opts:
# 		opts['output_size'] = 1024
# 	opts = Namespace(**opts)

# 	net = pSp(opts)
# 	net.eval()
# 	net.cuda()

# 	print('Loading dataset for {}'.format(opts.dataset_type))
# 	dataset_args = data_configs.DATASETS[opts.dataset_type]
# 	transforms_dict = dataset_args['transforms'](opts).get_transforms()
# 	dataset = InferenceDataset(root=opts.data_path,
# 	                           transform=transforms_dict['transform_inference'],
# 	                           opts=opts)
# 	dataloader = DataLoader(dataset,
# 	                        batch_size=opts.test_batch_size,
# 	                        shuffle=False,
# 	                        num_workers=int(opts.test_workers),
# 	                        drop_last=True)

# 	latent_mask = [int(l) for l in opts.latent_mask.split(",")]
# 	if opts.n_images is None:
# 		opts.n_images = len(dataset)

# 	global_i = 0
# 	vecs_to_inject = np.random.randn(opts.n_outputs_to_generate, 512).astype('float32')
# 	for input_batch in tqdm(dataloader):
# 		if global_i >= opts.n_images:
# 			break
# 		with torch.no_grad():
# 			input_batch = input_batch.cuda()
# 			for image_idx, input_image in enumerate(input_batch):
# 				# generate random vectors to inject into input image
# 				#vecs_to_inject = np.random.randn(opts.n_outputs_to_generate, 512).astype('float32')	#random value that inject to style
# 				multi_modal_outputs = []
# 				for vec_to_inject in vecs_to_inject:
# 					cur_vec = torch.from_numpy(vec_to_inject).unsqueeze(0).to("cuda")
# 					# get latent vector to inject into our input image
# 					_, latent_to_inject = net(cur_vec,
# 					                          input_code=True,
# 					                          return_latents=True)
# 					# get output image with injected style vector
# 					res = net(input_image.unsqueeze(0).to("cuda").float(),
# 					          latent_mask=latent_mask,
# 					          inject_latent=latent_to_inject,
# 					          alpha=opts.mix_alpha,
# 							  resize=opts.resize_outputs)
# 					multi_modal_outputs.append(res[0])

# 				# visualize multi modal outputs
# 				input_im_path = dataset.paths[global_i]
# 				image = input_batch[image_idx]
# 				input_image = log_input_image(image, opts)
# 				resize_amount = (256, 256) if opts.resize_outputs else (opts.output_size, opts.output_size)
# 				res = np.array(input_image.resize(resize_amount))
# 				for output in multi_modal_outputs:
# 					output = tensor2im(output)
# 					res = np.concatenate([res, np.array(output.resize(resize_amount))], axis=1)
# 				Image.fromarray(res).save(os.path.join(mixed_path_results, os.path.basename(input_im_path)))
# 				global_i += 1



# if __name__ == '__main__':
# 	run()
