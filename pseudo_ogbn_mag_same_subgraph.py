import dgl
from dgl.data.utils import save_graphs
import numpy as np
from statistics import mean
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from block_dataloader import generate_dataloader
import dgl.nn.pytorch as dglnn
import time
import argparse
import tqdm
import deepspeed
import random
from graphsage_model import SAGE
import dgl.function as fn
from load_graph import load_reddit, inductive_split, load_ogb, load_cora, load_karate, prepare_data, prepare_data_mag
from memory_usage import see_memory_usage
import tracemalloc
from cpu_mem_usage import get_memory
import pickle
# from utils import draw_graph_global


def set_seed(args):
	random.seed(args.seed)
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)
	if args.gpu >= 0:
		torch.cuda.manual_seed_all(args.seed)
		torch.cuda.manual_seed(args.seed)
		torch.backends.cudnn.enabled = False
		torch.backends.cudnn.deterministic = True


def ttt(tic, str1):
	toc = time.time()
	print(str1 + ' step Time(s): {:.4f}'.format(toc - tic))
	return toc


def compute_acc(pred, labels):
	"""
	Compute the accuracy of prediction given the labels.
	"""
	labels = labels.long()
	return (torch.argmax(pred, dim=1)==labels).float().sum() / len(pred)


def evaluate(model, g, nfeat, labels, val_nid, device):
	"""
	Evaluate the model on the validation set specified by ``val_nid``.
	g : The entire graph.
	inputs : The features of all the nodes.
	labels : The labels of all the nodes.
	val_nid : the node Ids for validation.
	device : The GPU device to evaluate on.
	"""
	model.eval()
	with torch.no_grad():
		pred = model.inference(g, nfeat, device, args)
	model.train()
	return compute_acc(pred[val_nid], labels[val_nid].to(pred.device))

def load_subtensor(nfeat, labels, seeds, input_nodes, device):
	"""
	Extracts features and labels for a subset of nodes
	"""
	batch_inputs = nfeat[input_nodes].to(device)
	batch_labels = labels[seeds].to(device)
	print('batch_inputs device')
	print(batch_inputs.device)
	return batch_inputs, batch_labels

# def load_blocks_subtensor(g, feats, labels, blocks, device):
# 	"""
# 	Extracts features and labels for a subset of nodes
# 	"""
# 	# print('\nblocks[0].srcdata dgl.NID list')
# 	# print(blocks[0].srcdata[dgl.NID].tolist())
# 	batch_inputs = feats[0][blocks[0].srcdata[dgl.NID].tolist()].to(device)
# 	# print('blocks[-1].dstdata')
# 	# print(blocks[-1].dstdata['_ID'])
# 	# print('---------------------------------------------------------------------------------------------------')
# 	batch_labels = labels[blocks[-1].dstdata['_ID']].to(device)
# 	# print('\nlength of blocks[-1].dstdata\t blocks[-1].srcdata[dgl.NID]')
# 	# print(str(len(blocks[-1].dstdata[dgl.NID]))+ ', '+str(len(blocks[-1].srcdata[dgl.NID])))
# 	# print(blocks[-1].srcdata[dgl.NID])
# 	# print()
   
# 	# print('batch_inputs device')
# 	# print(batch_inputs.device)
# 	return batch_inputs, batch_labels

def load_blocks_subtensor(g, labels, blocks, device):
	"""
	Extracts features and labels for a subset of nodes
	"""
	# print('\nblocks[0].srcdata dgl.NID list')
	# print(blocks[0].srcdata[dgl.NID].tolist())
	batch_inputs = g.ndata['features'][blocks[0].srcdata[dgl.NID].tolist()].to(device)
	batch_labels = blocks[-1].dstdata['labels'].to(device)
	# print('\nblocks[-1].dstdata')
	# print(blocks[-1].dstdata[dgl.NID])
	# print()
	# print('batch_inputs device')
	# print(batch_inputs.device)
	return batch_inputs, batch_labels


def save_full_batch_subgraphs(save_path, graph, graph_labels):
	save_graphs(save_path, graph, graph_labels)


#### Entry point
def run(args, device, data):
	# Unpack data
	# train_g, train_nid, in_feats, n_classes, feats, train_labels, test_g, test_nid, test_labels= data
	# train_g, feats, labels, in_feats, n_classes, train_nid, test_nid = data
	n_classes, train_g, val_g, test_g, train_nfeat, train_labels, \
	val_nfeat, val_labels, test_nfeat, test_labels = data

	in_feats = train_nfeat.shape[1]
	train_nid = torch.nonzero(train_g.ndata['train_mask'], as_tuple=True)[0]
	val_nid = torch.nonzero(val_g.ndata['val_mask'], as_tuple=True)[0]
	test_nid = torch.nonzero(~(test_g.ndata['train_mask'] | test_g.ndata['val_mask']), as_tuple=True)[0]
	dataloader_device = torch.device('cpu')

	sampler = dgl.dataloading.MultiLayerNeighborSampler(
		[int(fanout) for fanout in args.fan_out.split(',')])

	# sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
	full_batch_size = len(train_nid)
	full_batch_dataloader = dgl.dataloading.NodeDataLoader(
		train_g,
		train_nid,
		sampler,

		batch_size=full_batch_size,
		shuffle=True,
		drop_last=False,
		num_workers=args.num_workers)
	
	epoch_train_CPU_time_list = []
	# see_memory_usage("-----------------------------------------before for epoch loop ")
	iter_tput = []
	avg_step_data_trans_time_list = []
	avg_step_GPU_train_time_list = []
	avg_step_time_list = []
	


	graph_checking_input=[]
	graph_checking_output=[]
	graph_cheking_blocks=[]
	full_batch_sub_graph_data_list=[]

	for epoch in range(args.num_epochs):
		print('Epoch ' + str(epoch))
		for full_batch_step, (input_nodes, output_seeds, full_batch_blocks) in enumerate(full_batch_dataloader):
			full_batch_sub_graph_data_list.append((input_nodes, output_seeds, full_batch_blocks))
			print('full_batch_blocks')
			print(full_batch_blocks)
			print(len(full_batch_blocks[0].srcdata['_ID']))
			print(len(full_batch_blocks[0].dstdata['_ID']))
			for cur_block in full_batch_blocks:
				block_to_graph=dgl.block_to_graph(cur_block)
				from dgl.data.utils import save_graphs
				save_graphs('./DATA/'+args.dataset+'_'+str(epoch)+'_subgraph.bin',[block_to_graph])
			

# #-------------------------------------------------------------------------------------------------------------------------------------------
	
# 	# batch_size_list=[4919, 9838, 19675,  39349, 78697, 157393]
# 	# batch_size_list=[153431, 76716, 38358, 19179, 9590, 4795, 2400, 1200]
# 	batch_size_list=[1200, 2400, 4795, 9590, 19179, 38358, 76716, 153431]
# 	selection_method_=['range', 'random']

# 	for method in selection_method_:
# 		args.selection_method=method
# 		print('-'*120, method)
		

# 		for i_run in range(len(batch_size_list)):
# 			args.batch_size=batch_size_list[i_run]

# 			print('-------------------------------------------------------------------------------------------------------Run '+ str(i_run))
# 			# torch.cuda.empty_cache()
# 			model = SAGE(in_feats, args.num_hidden, n_classes, args.num_layers, F.relu, args.dropout, args.aggre)
# 			model = model.to(device)
# 			loss_fcn = nn.CrossEntropyLoss()
# 			optimizer = optim.Adam(model.parameters(), lr=args.lr)

			
# 			train_accs = []
# 			test_accs = []
# 			block_generate_time_list = []
# 			connection_checking_time_list=[]
# 			blocks_generation_time_list=[]
# 			mean_per_block_generation_time_list=[]
# 			batch_list_generate_time_list=[]
# 			# for epoch in range(args.num_epochs):
# 			# 	print('-'*80 +'Run '+str(i_run)+ ' Epoch ' + str(epoch))
# 			# 	# train_start_tic = time.time()
				
# 				# data loader sampling fan-out neighbor each new epoch
# 			for epoch, (input_nodes, output_seeds, full_batch_blocks) in enumerate(full_batch_sub_graph_data_list):
# 				print('-'*90 +'Run '+str(i_run)+ ' Epoch ' + str(epoch))
# 				# if i_run == 0:
# 				# 	input_nodes_, _ =input_nodes.sort()
# 				# 	output_seeds_, _ = output_seeds.sort()
# 				# 	graph_checking_input.append(input_nodes_)
# 				# 	graph_checking_output.append(output_seeds_)
# 				# elif i_run != 0 :
# 				# 	input_nodes_, _ =input_nodes.sort()
# 				# 	output_seeds_, _ = output_seeds.sort()

# 				# 	if torch.equal(input_nodes_, graph_checking_input[epoch]) and torch.equal(output_seeds_, graph_checking_output[epoch]):
# 				# 		print('epoch '+ str(epoch))
# 				# 		print('-'*30+'the full batch subgraph is same with ----------------------------*******************')
# 				# 	else:
# 				# 		print('epoch '+ str(epoch))
# 				# 		print('-'*30+'the full batch subgraph is different --------------------------------now')



# 				# Create DataLoader for constructing blocks
# 				# print('\n----main run function: start generate block_dataloader from full batch train graph')
# 				ssss_time = time.time()
# 				block_dataloader, weights_list,time_collection = generate_dataloader(train_g, full_batch_blocks,  args)
# 				block_generate_time = time.time() - ssss_time
# 				# print('\n----main run function: block dataloader generation total spend   ' + str(block_generate_time))
# 				# print()
				
# 				connection_time, block_gen_time, mean_block_gen_time, batch_list_generate_time = time_collection

# 				block_generate_time_list.append(block_generate_time)
# 				connection_checking_time_list.append(connection_time)
# 				blocks_generation_time_list.append(block_gen_time)
# 				mean_per_block_generation_time_list.append(mean_block_gen_time)
# 				batch_list_generate_time_list.append(batch_list_generate_time)
# 				#-------------------------------------------------------------------------------------------------
# 				# Training loop
# 				step_time_list = []
# 				step_data_trans_time_list = []
# 				step_GPU_train_time_list = []

# 				# Loop over the dataloader to sample the computation dependency graph as a list of blocks.
# 				# torch.cuda.synchronize()
# 				start = torch.cuda.Event(enable_timing=True)
# 				end = torch.cuda.Event(enable_timing=True)
				
# 				# print('length of block dataloader')
# 				# print(len(block_dataloader))

# 				pseudo_mini_loss = torch.tensor([], dtype=torch.long)
# 				loss_sum = 0
# 				train_start_tic = time.time()
# 				tic_step = time.time()
# 				src_collection =[]

# 				for step, (input_node, seeds, blocks) in enumerate(block_dataloader):
# 					# print("\n   ***************************     step   " + str(step) + " mini batch block  *************************************")
# 					src_collection.append(len(input_node))
# 					torch.cuda.synchronize()
# 					start.record()
					
# 					# Load the input features as well as output labels
					
# 					batch_inputs, batch_labels = load_blocks_subtensor(train_g, train_labels, blocks, device)

# 					blocks = [block.int().to(device) for block in blocks]

# 					end.record()
# 					torch.cuda.synchronize()  # wait for move to complete
# 					step_data_trans_time_list.append(start.elapsed_time(end))
# 					#----------------------------------------------------------------------------------------
# 					start1 = torch.cuda.Event(enable_timing=True)
# 					end1 = torch.cuda.Event(enable_timing=True)
# 					start1.record()

# 					# Compute loss and prediction
# 					# see_memory_usage("----------------------------------------before batch_pred = model(blocks, batch_inputs) ")
# 					batch_pred = model(blocks, batch_inputs)
# 					# see_memory_usage("-----------------------------------------batch_pred = model(blocks, batch_inputs) ")
# 					pseudo_mini_loss = loss_fcn(batch_pred, batch_labels)
# 					# print('----------------------------------------------------------pseudo_mini_loss ', pseudo_mini_loss)
# 					pseudo_mini_loss = pseudo_mini_loss*weights_list[step]
# 					# print('----------------------------------------------------------pseudo_mini_loss ', pseudo_mini_loss)
# 					pseudo_mini_loss.backward()
# 					loss_sum += pseudo_mini_loss

# 					end1.record()
# 					torch.cuda.synchronize()  # wait for all training steps to complete
# 					step_GPU_train_time_list.append(start1.elapsed_time(end1))

# 					step_time = time.time() - tic_step
# 					step_time_list.append(step_time)
# 					torch.cuda.empty_cache()
# 					# print(step_time)

# 					iter_tput.append(len(seeds) / (time.time() - tic_step))

# 					tic_step = time.time()

# 				# see_memory_usage("-----------------------------------------before final loss ")

# 				optimizer.step()
# 				optimizer.zero_grad()

# 				train_end_toc = time.time()

# 				epoch_train_CPU_time_list.append((train_end_toc - train_start_tic ))
							
# 				# print('------------------------------------------------------pseudo_mini_loss  ' + str(pseudo_mini_loss.tolist()))
# 				print('-----------------------------------------pseudo_mini_loss sum ' + str(loss_sum.tolist()))
# 				if epoch % args.log_every==0:
# 					acc = compute_acc(batch_pred, batch_labels)
# 					gpu_mem_alloc = torch.cuda.max_memory_allocated() / 1024 / 1024 /1024 if torch.cuda.is_available() else 0
# 					print(
# 						'Run {:05d} | Epoch {:05d} | Loss {:.4f} | Train Acc {:.4f} | Speed (samples/sec) {:.4f} | GPU {:.4f} GB'.format(
# 							i_run, epoch, pseudo_mini_loss.item(), acc.item(), np.mean(iter_tput[3:]), gpu_mem_alloc))

				
# 			train_acc = evaluate(model, train_g, train_nfeat, train_labels, train_nid, device)
# 			print('train Acc: {:.4f}'.format(train_acc))
# 			test_acc = evaluate(model, test_g, test_nfeat, test_labels, test_nid, device)
# 			print('Test Acc: {:.4f}'.format(test_acc))
# 			print('='*100)
# 			# # # val_accs.append(best_val)
# 			# train_accs.append(train_acc)
# 			# test_accs.append(test_acc)

# 			# print(f"Average train accuracy: {np.mean(train_accs):.4f} ")
# 			# print(f"Average test accuracy: {np.mean(test_accs):.4f} ")
# 			# print('='*100)
# 			# print('train accuracy'+str(train_accs))
# 			# print('test accuracy'+str(test_accs))
# 			# print('='*100)
# 			print("\navg time for data loader generation " + str(mean(block_generate_time_list)))
# 			print("\nbatch NID list generation time  " + str(mean(batch_list_generate_time_list)))
# 			print("\nconnection checking time " + str(mean(connection_checking_time_list)))
# 			print("total of block generation time " + str(mean(blocks_generation_time_list)))
# 			print("average of block generation time " + str(mean(mean_per_block_generation_time_list)))
# 		torch.cuda.empty_cache()	




def main(args):
    
	device = "cpu"
	if args.dataset=='karate':
		g, n_classes = load_karate()
		device = "cuda:0"
		data=prepare_data(g, n_classes, args, device)
	elif args.dataset=='cora':
		g, n_classes = load_cora()
		device = "cuda:0"
		data=prepare_data(g, n_classes, args, device)
	elif args.dataset=='reddit':
		g, n_classes = load_reddit()
		device = "cuda:0"
		data=prepare_data(g, n_classes, args, device)
	elif args.dataset=='ogbn-products':
		g, n_classes = load_ogb(args.dataset)
		# print('#nodes:', g.number_of_nodes())
		# print('#edges:', g.number_of_edges())
		# print('#classes:', n_classes)
		device = "cuda:0"
		data=prepare_data(g, n_classes, args, device)
	elif args.dataset=='ogbn-mag':
		data = prepare_data_mag(device, args)
		device = "cuda:0"
		
	else:
		raise Exception('unknown dataset')
	
	best_test = run(args, device, data)
	

if __name__=='__main__':
	# get_memory("-----------------------------------------main_start***************************")
	tt = time.time()
	print("main start at this time " + str(tt))
	argparser = argparse.ArgumentParser("multi-gpu training")
	argparser.add_argument('--gpu', type=int, default=0,
		help="GPU device ID. Use -1 for CPU training")
	argparser.add_argument('--seed', type=int, default=1236)

	argparser.add_argument('--dataset', type=str, default='ogbn-mag')
	# argparser.add_argument('--dataset', type=str, default='ogbn-products')
	argparser.add_argument('--aggre', type=str, default='lstm')
	# argparser.add_argument('--dataset', type=str, default='cora')
	# argparser.add_argument('--dataset', type=str, default='karate')
	# argparser.add_argument('--dataset', type=str, default='reddit')
	# argparser.add_argument('--aggre', type=str, default='mean')
	argparser.add_argument('--selection-method', type=str, default='range')
	argparser.add_argument('--num-epochs', type=int, default=6)
	argparser.add_argument('--num-runs', type=int, default=2)
	argparser.add_argument('--num-hidden', type=int, default=16)
	argparser.add_argument('--num-layers', type=int, default=1)
	# argparser.add_argument('--fan-out', type=str, default='20')
	argparser.add_argument('--fan-out', type=str, default='10')

	# argparser.add_argument('--batch-size', type=int, default=629571)
	# argparser.add_argument('--batch-size', type=int, default=314786)
	# argparser.add_argument('--batch-size', type=int, default=157393)
	# argparser.add_argument('--batch-size', type=int, default=78697)
	# argparser.add_argument('--batch-size', type=int, default=39349)
	# argparser.add_argument('--batch-size', type=int, default=19675)
	# argparser.add_argument('--batch-size', type=int, default=9838)
	# argparser.add_argument('--batch-size', type=int, default=4919)


	# argparser.add_argument('--batch-size', type=int, default=196571)
	# argparser.add_argument('--batch-size', type=int, default=98308)
	# argparser.add_argument('--batch-size', type=int, default=49154)
	# argparser.add_argument('--batch-size', type=int, default=24577)
	# argparser.add_argument('--batch-size', type=int, default=12289)
	# argparser.add_argument('--batch-size', type=int, default=6145)
	# argparser.add_argument('--batch-size', type=int, default=3000)
	# argparser.add_argument('--batch-size', type=int, default=1500)
	argparser.add_argument('--batch-size', type=int, default=8)

	argparser.add_argument("--eval-batch-size", type=int, default=100000,
                        help="evaluation batch size")
	argparser.add_argument("--R", type=int, default=5,
                        help="number of hops")
	


	argparser.add_argument('--log-every', type=int, default=5)
	argparser.add_argument('--eval-every', type=int, default=1)
	# argparser.add_argument('--lr', type=float, default=0.05)
	argparser.add_argument('--lr', type=float, default=0.003)
	argparser.add_argument('--dropout', type=float, default=0.5)
	argparser.add_argument('--num-workers', type=int, default=4,
		help="Number of sampling processes. Use 0 for no extra process.")
	argparser.add_argument('--inductive', action='store_true',
		help="Inductive learning setting")
	argparser.add_argument('--data-cpu', action='store_true',
		help="By default the script puts all node features and labels "
		     "on GPU when using it to save time for data copy. This may "
		     "be undesired if they cannot fit in GPU memory at once. "
		     "This flag disables that.")

	args = argparser.parse_args()
	
	main(args)