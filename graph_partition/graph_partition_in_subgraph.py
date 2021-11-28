import numpy 
import dgl
from numpy.core.numeric import Infinity
import torch
import time

maxLeftGainIndex=0
maxRightGainIndex=0
maxDegree=0
maxGainIndex=0

leftBuckets=[]
rightBuckets=[]

gains={}
bit_dict={}
totalSteps=0
totalSegments=0
side=0

block_to_graph=None

# locked_nodes={};

def InitializeBitList(p_list):
    global bit_dict
    nums_p=len(p_list); 
    bit_dict={}
    for i in range (nums_p-1):
        A = p_list[i]
        B = p_list[i + 1]; 
        for k in A:
            bit_dict[k] = 0
        for m in B:
            bit_dict[m] = 1
        
    return 
    
def redundancy_check(cur_part_in_size,ideal_part_in_size,target_redun):
    redundancy_flag=True
    r = cur_part_in_size/ideal_part_in_size
    if r > target_redun:
        redundancy_flag=False
    return redundancy_flag
    
def balance_check_2(A_in,B_in,alpha):
    balance_flag=True
    avg = (len(A_in)+len(B_in))/2
    if abs(len(B_in)-len(A_in)) >alpha*avg:
        balance_flag=False
    return balance_flag
    
def Cost( A_o, B_o, bit_dict):
    global block_to_graph
    cost =0
    N_O=len(bit_dict) # total output nids of two partitions
    for i in A_o:
        if bit_dict[i]==0:
            for j in B_o:
                if block_to_graph.has_edges_between(i,j) and bit_dict[j]==1:
                    cost+=1
    return cost
    
def getRedundancyCost(len_A, len_B, ideal_part_in_size):
    cost =0
    ratio_A=len_A/ideal_part_in_size
    ratio_B = len_B/ideal_part_in_size
    cost = max(ratio_A,ratio_B)
    return cost   #   minimize the max ratio of 
    
def get_mini_batch_size(full_len,num_batch):
	mini_batch=int(full_len/num_batch)
	if full_len%num_batch>0:
		mini_batch+=1
	# print('current mini batch size of output nodes ', mini_batch)
	return mini_batch
	
def gen_batch_output_list(OUTPUT_NID,indices,mini_batch):
	
	map_output_list = list(numpy.array(OUTPUT_NID)[indices])
		
	batches_nid_list = [map_output_list[i:i + mini_batch] for i in range(0, len(map_output_list), mini_batch)]
			
	output_num = len(OUTPUT_NID)
	
	# print(batches_nid_list)
	weights_list = []
	for i in batches_nid_list:
		# temp = len(i)/output_num
		weights_list.append(len(i)/output_num)
		
	return batches_nid_list, weights_list
	
	
def random_shuffle(full_len):
	indices = numpy.arange(full_len)
	numpy.random.shuffle(indices)
	return indices

def gen_batched_seeds_list(OUTPUT_NID, args):
	'''

	Parameters
	----------
	OUTPUT_NID: final layer output nodes id (tensor)
	args : all given parameters collection

	Returns
	-------

	'''
	selection_method = args.selection_method
	num_batch = args.num_batch
	# mini_batch = 0
	full_len = len(OUTPUT_NID)  # get the total number of output nodes
	mini_batch_size=get_mini_batch_size(full_len,num_batch)
	args.batch_size=mini_batch_size
	
	if selection_method == 'random_init_graph_partition' :
		indices = random_shuffle(full_len)
		
	elif selection_method == 'similarity_init_graph_partition':
		indices = torch.tensor(range(full_len)) #----------------------------TO DO
	
	batches_nid_list, weights_list=gen_batch_output_list(OUTPUT_NID,indices,mini_batch_size)
	
	return batches_nid_list, weights_list


def initializeBuckets(args):
    global maxDegree; global leftBuckets; global rightBuckets; 
    global bit_dict
    global block_to_graph
    
    leftBuckets=[]
    rightBuckets=[]
    max_neighbors=int(args.fan_out) # only for 1-layer model
    
    local_nids=list(bit_dict.keys())
    local_in_degrees = block_to_graph.in_degrees(local_nids).tolist()
    max_in_degree=max(local_in_degrees) # only consider output nodes in degrees

    print(max_in_degree)
    print(max_neighbors)
    # maxDegree= min(max_neighbors, max_in_degree)
    maxDegree= max(max_neighbors, max_in_degree)
    print('maxDegree ',maxDegree)
    
    # initializeBuckets leftBuckets and rightBuckets
    for i in range(2*maxDegree +1):
        leftBuckets.append([])
        rightBuckets.append([])
    
    # print('leftBuckets ',leftBuckets)
    
    return
    
def initializeBucketSort(nid, gain, side):
    global maxGainIndex, leftBuckets, rightBuckets 
    global maxLeftGainIndex
    global maxRightGainIndex
    global gains
    global maxDegree
    
    
    index = maxDegree+gain
    
    if side ==0:
        maxGainIndex = maxLeftGainIndex
        maxGainIndex= max(index, maxGainIndex)
        if index >= len(leftBuckets):
            print(index)
        leftBuckets[index].append(nid)
        gains[nid]=gain
        maxLeftGainIndex = maxGainIndex
    else:
        maxGainIndex= maxRightGainIndex
        # index = maxDegree+gain
        maxGainIndex= max(index, maxGainIndex)
        rightBuckets[index].append(nid)
        gains[nid]=gain
        maxRightGainIndex = maxGainIndex
    
    return gains
    
def initializeGain( bit_dict):
    global gains
    global maxDegree
    global block_to_graph
    
    gains={}
    A_o=[k for k in bit_dict.keys() if bit_dict[k] == 0]
    B_o=[k for k in bit_dict.keys() if bit_dict[k] == 1]
    for i in A_o:
        gain=0
        in_nids=block_to_graph.predecessors(i)
        common=list(set(in_nids).intersection(B_o))
        for j in common:
            if bit_dict[j]==bit_dict[i]:
                gain-=1
            else:
                gain+=1
        
        if gain > maxDegree:
            maxDegree = max(gain, maxDegree)
            print(j)
            print(gain)
        gains=initializeBucketSort(i, gain, 0) # 0: left side bucket
        
    for i in B_o:
        gain=0
        in_nids=block_to_graph.predecessors(i)
        common=list(set(in_nids).intersection(A_o))
        for j in common:
            if bit_dict[i]==bit_dict[j]:
                gain-=1
            else:
                gain+=1
        gains=initializeBucketSort(i, gain, 1) # 1: right side bucket
                
    return 


def get_weight_list(batched_seeds_list):
    
    output_num = len(sum(batched_seeds_list,[]))
    # print(output_num)
    weights_list = []
    for seeds in batched_seeds_list:
		# temp = len(i)/output_num
        weights_list.append(len(seeds)/output_num)
    return weights_list

def gen_partition_input_out( bit_dict):
   
    A_o=[k for k in bit_dict if bit_dict[k] == 0]
    B_o=[k for k in bit_dict if bit_dict[k] == 1]
    A_in = get_in_nodes(A_o)
    B_in = get_in_nodes(B_o)
    
    return A_in, B_in, A_o, B_o

def get_two_partition_seeds(bit_dict):
   
    A_o=[k for k in bit_dict if bit_dict[k] == 0]
    B_o=[k for k in bit_dict if bit_dict[k] == 1]
    # A_in = get_in_nodes(A_o)
    # B_in = get_in_nodes(B_o)
    
    return  A_o, B_o
    
def incrementGain(cur_bucket, nid, cur_side):
    
    global maxGainIndex; 
    global maxLeftGainIndex; 
    global maxRightGainIndex; 
    global maxDegree; 
    global gains ; 
    
    # maxGainIndex=-Infinity
    
    if cur_side == 0:
        maxGainIndex=maxLeftGainIndex
    else:
        maxGainIndex=maxRightGainIndex
    
    # remove node from current linked list 
    gain = gains[nid]
    bucketIndex=cur_bucket[maxDegree+gain]
    if bucketIndex:
        bucketIndex.remove(nid) # remove node nid 
    # if maxDegree+gain+2>2*maxDegree:
        # print()
    nextBucketIndex = cur_bucket[maxDegree+gain+2]    
    nextBucketIndex.insert(0,nid)
    gains[nid]=gain+2
    maxGainIndex = max(maxGainIndex, maxDegree + gain+2)
    
    if cur_side == 0:
        maxLeftGainIndex=maxGainIndex
    else:
        maxRightGainIndex=maxGainIndex
    
    return cur_bucket
    
def decrementGain(comp_bucket, nid, cur_side):
    global maxGainIndex
    global maxLeftGainIndex 
    global maxRightGainIndex 
    global maxDegree  
    global gains; 
    
    if cur_side == 0:
        maxGainIndex=maxLeftGainIndex
    else:
        maxGainIndex=maxRightGainIndex
    
    # remove node from current linked list 
    gain = gains[nid]
    bucketIndex=comp_bucket[maxDegree+gain]
    if bucketIndex:
        # if nid not in bucketIndex:
            # print(nid)
        bucketIndex.remove(nid) # remove node nid 
    
    nextBucketIndex = comp_bucket[maxDegree+gain-2]    
    nextBucketIndex.insert(0,nid)
    
    # if the current linkedlist is empty only then check to reduce the maxgainIndex
    tmp_list=comp_bucket[maxDegree+gain]
    if maxGainIndex==(maxDegree + gain) and len(tmp_list)==0:
        maxGainIndex-=1 # Reduce only if it is equal to the only highest gain vertex
    gains[nid]=gain-2
    
    if cur_side == 0:
        maxLeftGainIndex=maxGainIndex
    else:
        maxRightGainIndex=maxGainIndex
    
    
    return comp_bucket
    
    
def counting(bit_dict, side):
    seeds=[k for k in bit_dict if bit_dict[k] == int(side)]
    return len(seeds)
    
def move_nid_balance_check(bit_dict_origin, side, nid,alpha):
    
    bit_dict_origin[nid]=1-side
    A_o=[k for k in bit_dict_origin if bit_dict_origin[k] == 0]
    B_o=[k for k in bit_dict_origin if bit_dict_origin[k] == 1]
    bit_dict_origin[nid]=1-side
   
    # bit_dict_local=bit_dict_origin
    # bit_dict_local[nid]=1-side
    
    # A_o=[k for k in bit_dict_local if bit_dict_local[k] == 0]
    # B_o=[k for k in bit_dict_local if bit_dict_local[k] == 1]
    # bit_dict_local[nid]=1-side
    # bit_dict_origin=bit_dict_local
    
    t1=time.time()
    len_A_part,len_B_part = get_src_nodes_len(A_o,B_o)
    # print('get_src_nodes_len ', time.time()-t1)
    len_=len_A_part+len_B_part
    avg = len_/2
    
    if len_B_part>0 and len_A_part>0 and abs(len_A_part-len_B_part) < avg*alpha:
        return True
    return False
    
def updateGain( locked_nodes, alpha):
    global maxGainIndex, maxLeftGainIndex, maxRightGainIndex, maxDegree, leftBuckets, rightBuckets 
    global side, bit_dict
    
    if counting(bit_dict, side)<=1 or counting(bit_dict, (1-side))<=1: # if only one output node left in side, stop to move
        return
    maxGainNid=None
    currentSide=side
    cur_bucket=[]
    comp_bucket=[]
    
    tif=time.time()
    if side==0:
        cur_bucket=leftBuckets
        comp_bucket=rightBuckets
        index=0
        ifFound=False
        t_wh=time.time()
        while not ifFound:
            bucketIndex = cur_bucket[maxLeftGainIndex-index] # find the maxgain node, if it not work to find the second oder gain node.
            while bucketIndex:
                i=bucketIndex[0]
                t_iif=time.time()
                if   not locked_nodes[i] and move_nid_balance_check(bit_dict, side, i, alpha):
                    maxGainNid = i
                    ifFound = True
                    t_iif_e=time.time()
                    # print('move nid balance check: ', t_iif_e-t_iif)
                    break
                # t_iif_e=time.time()
                # print()
                bucketIndex=bucketIndex[1:]
            index+=1
            if index >= maxDegree:
                break
        
        t_wh_e=time.time()
        # print('while loop spend: ', t_wh_e-t_wh)
        if maxGainNid == None: ########
            side = 1-side 
            return
    
    else:
        cur_bucket = rightBuckets
        comp_bucket = leftBuckets
        index=0
        ifFound=False
        while not ifFound:
            bucketIndex = cur_bucket[maxRightGainIndex-index] # find the maxgain node, if it not work to find the second oder gain node.
            while bucketIndex:
                nid=bucketIndex[0]
                if   not locked_nodes[nid] and move_nid_balance_check(bit_dict, side, nid, alpha):
                    maxGainNid = nid
                    ifFound = True
                    break
                bucketIndex=bucketIndex[1:]
            index+=1
            if index >= maxDegree:
                break
        
        if maxGainNid == None: ########
            side = 1-side
            return
    
    bit_dict[maxGainNid]= 1-side 
    side = 1-side
    locked_nodes[maxGainNid]=True #Lock this node
    
    in_nodes=get_in_nodes([maxGainNid])
    for idx in in_nodes:
        if idx not in locked_nodes:
            continue
        if not locked_nodes[idx]:    # only consider the outputnodes connected with maxGainNid node.
            if bit_dict[idx]==currentSide:
                cur_bucket=incrementGain(cur_bucket, idx, currentSide)
            else:
                comp_bucket=decrementGain( comp_bucket, idx, currentSide)
    
    return 
                
                
def balance_check_all_partitions(partition_dst_src_list,alpha):
    balance_flag = True
    for i in range(len(partition_dst_src_list)-1):
        A_nids= partition_dst_src_list[i]
        B_nids= partition_dst_src_list[i+1]
        avg = (len(A_nids)+len(B_nids))/2
        if abs(len(B_nids)-len(A_nids)) >alpha*avg:
            balance_flag=False
            return balance_flag
    return balance_flag
    
def balance_check_and_exchange(bit_dict,alpha):
    # global side
    
    A_o=[k for k in bit_dict if bit_dict[k] == 0]
    B_o=[k for k in bit_dict if bit_dict[k] == 1]
    
    len_A_part,len_B_part = get_src_nodes_len(A_o,B_o)
     
    # avg = (len(A_in)+len(B_in))/2
    # side = 0
    if len_B_part>0 and len_A_part>0 and abs(len_A_part-len_B_part) > 0:
        
        # side = 1
        if len_A_part>len_B_part:
            for k in A_o:
                bit_dict[k] = 0
            for m in B_o:
                bit_dict[m] = 1
                
        if len_A_part<len_B_part:
            for k in A_o:
                bit_dict[k] = 1
            for m in B_o:
                bit_dict[m] = 0
   
    return bit_dict
    # return bit_dict, side
    

       
        
def get_output_mini_batch(bit_dict):
    res=[i for i in bit_dict if bit_dict[i]==0]
    return res
    
    
# def gen_src_nodes_dict():
#     global partitions
#     global block_to_graph
#     print(block_to_graph)
#     print(type(block_to_graph))
    
#     # block_to_graph_1 = dgl.to_homogeneous(block_to_graph)
#     # print(type(block_to_graph_1))
#     # print(block_to_graph_1)
#     # print(block_to_graph.ndata)
#     # print(dgl.node_subgraph(block_to_graph_2, [0, 1, 4]))
    
    
#     partitions={}
    
#     sub_in_nids = block_to_graph.srcdata['_ID'].tolist()
#     sub_out_nids = block_to_graph.dstdata['_ID'].tolist()

#     global_nid_2_local = {sub_in_nids[i]: i for i in range(0, len(sub_in_nids))}
#     local_nid_2_global = { i: sub_in_nids[i] for i in range(0, len(sub_in_nids))}
#     # ADJ_matrix=block_to_graph.adj_sparse('coo')
#     t1=time.time()
#     iii=0
#     for nid in sub_out_nids:
#         tt=time.time()
#         local_nid = global_nid_2_local[nid]
#         # local_in_nids = list(block_to_graph.in_edges(local_nid))[0].tolist() # slower than predecessors
#         local_in_nids = list(set(block_to_graph.predecessors(local_nid).tolist()))
#         global_in_nids = list(map(local_nid_2_global.get, local_in_nids))
#         partitions[nid]=global_in_nids
#         tt_1=time.time()
#         iii+=1
#         if iii%7000==0:
#             print('each nid predecessor: ',tt_1-tt)
#     print('total time for gen src nodes dict')
#     print(time.time()-t1)
#     return partitions
    
    
# def gen_partition_src_list(batched_seeds_list):
#     global partitions
    
#     partition_src_list=[]
#     for batch_nids in batched_seeds_list:
#         part=[]
#         for nid in batch_nids:
#             in_nids = partitions[nid]
#             part.append(in_nids)
#         part=sum(part,[])
#         partition_src_list.append(list(set(part)))
#     return partition_src_list
    
def get_partition_src_list_len(batched_seeds_list):
    global block_to_graph
    
    partition_src_list_len=[]
    for seeds_nids in batched_seeds_list:
        in_nids = get_in_nodes(seeds_nids)
        part=list(set(seeds_nids+in_nids))
        partition_src_list_len.append(len(part))
    return partition_src_list_len
    

  
def get_src_nodes_len(seeds_1,seeds_2):
   
    global block_to_graph
    # t0=time.time()
    frontier_1=dgl.in_subgraph(block_to_graph, seeds_1)
    src_len_1=len(set(list(frontier_1.edges())[0].tolist()))
    # print('frontier_1',src_len_1)
    frontier_2=dgl.in_subgraph(block_to_graph, seeds_2)
    src_len_2=len(set(list(frontier_2.edges())[0].tolist()))
    return src_len_1, src_len_2
    # print('frontier_2',src_len_2)
    # print('in_subgraph method ', time.time()-t0)
    
    # t1=time.time()
    # batched_seeds=[seeds_1, seeds_2]
    # res=[]
    # for seeds in batched_seeds:
    #     in_ids=list(block_to_graph.in_edges(seeds_1))[0].tolist()
    #     src_ids= list(set(in_ids+seeds))
    #     res.append(len(src_ids))
    # print('in_edges methods spend : ', time.time()-t1)
    # print()
    # return res[0],res[1] 

def get_in_nodes(seeds):
    global block_to_graph
    in_ids=list(block_to_graph.in_edges(seeds))[0].tolist()
    in_ids= list(set(in_ids))
    return in_ids
    
    
def update_Batched_Seeds_list(batched_seeds_list, i, j):
    global bit_dict
    print(' batched_seeds_list       -----------------before')
    print(batched_seeds_list)
    batch_i=[k for k in bit_dict if bit_dict[k]==0]
    batch_j=[k for k in bit_dict if bit_dict[k]==1]
    batched_seeds_list.remove(batched_seeds_list[i])
    batched_seeds_list.insert(i,batch_i)
    batched_seeds_list.remove(batched_seeds_list[j])
    batched_seeds_list.insert(j,batch_j)
    print('batched_seeds_list -------------------------after')
    print(batched_seeds_list)
    return batched_seeds_list
    
def update_Batched_Seeds_list_final(batched_seeds_list, i, j):
    global bit_dict
    print(' batched_seeds_list       -----------------before_final')
    print(batched_seeds_list)
    batch_i=[k for k in bit_dict if bit_dict[k]==0]
    batch_j=[k for k in bit_dict if bit_dict[k]==1]
    batched_seeds_list=batched_seeds_list[:-2]
    batched_seeds_list.append(batch_i)
    batched_seeds_list.append(batch_j)
    print('batched_seeds_list -------------------------after_final')
    print(batched_seeds_list)
    print()
    return batched_seeds_list
def gen_random_batched_seeds_list(batched_seeds_list):
    num_batch=len(batched_seeds_list)
    output_nids= sum(batched_seeds_list,[])
    full_len=len(output_nids)
    indices = random_shuffle(full_len)
    mini_batch_size=get_mini_batch_size(full_len,num_batch)
    batches_nid_list, weights_list=gen_batch_output_list(output_nids,indices,mini_batch_size)
	
    return batches_nid_list

def walk_terminate_0( block_to_graph,cost,args, ideal_part_in_size):
    global totalSegments
    global totalSteps
    global bit_dict
    global side
    
    print('walk terminate 0')
    t1=time.time()
    
    bit_dict=balance_check_and_exchange(bit_dict, args.alpha)
    
    # initialize locked nodes
    bestCost = cost   # best cost in this segment
    bottom = bit_dict   # bottom of this segment
    flag = False
    best_bit_dict=bottom  # best bit dict in this segment
    
    initializeBuckets(args)  # Invokes a method to calculate max Degree and initializes bucket with 2*maxDegree+1 size
    t2=time.time()
    print('initializeBuckets  ', t2-t1)
    initializeGain(bit_dict) # compute initial gain for the segment
    t3=time.time()
    print('initializeGain  ', t3-t2)
    
    A_o, B_o = get_two_partition_seeds(bit_dict)#  variable partitions dict {output:[input nodes]}
    t4=time.time()
    print('get_two_partition_seeds  ', t4-t3)
    subgraph_o = A_o+B_o
    locked_nodes={id:False for id in subgraph_o}
    
    t_b=time.time()
    # begin segment
    for i in range(len(subgraph_o)):
        tt=time.time()
        updateGain(locked_nodes,args.alpha)
        tt_e=time.time()
        if i % 500==0:
            print('one updateGain  ', tt_e-tt)	
        totalSteps+=1
        len_A_part,len_B_part = get_src_nodes_len(A_o,B_o)
        tmpCost=getRedundancyCost(len_A_part,len_B_part,ideal_part_in_size)
        
        if tmpCost < bestCost: 
            bestCost = tmpCost
            best_bit_dict = bit_dict
        
    t_e=time.time()
    print('updateGain of segment  ', t_e-t_b)	
    totalSegments +=1 
    if (bestCost < cost) : #is there improvement?
        bit_dict = best_bit_dict
        cost = bestCost
        return True, bestCost
	
    bit_dict = best_bit_dict
    
    return False,bestCost
 
def graph_partition( batched_seeds_list, block_2_graph, args):
    global maxGainIndex
    global maxLeftGainIndex
    global maxRightGainIndex
    global maxDegree
    global gains
    global leftBuckets 
    global rightBuckets
    global totalSteps
    global totalSegments
    global bit_dict
    global side
    global block_to_graph
    
    full_batch_graph_nids_size=len(block_2_graph.srcdata['_ID'])
    ideal_part_in_size=(full_batch_graph_nids_size/args.num_batch)
    full_batch_seeds = block_2_graph.dstdata['_ID'].tolist()
    num_batch=args.num_batch
    balance_flag = False
    
    # print(list(block_2_graph.edges()))
    src_ids=list(block_2_graph.edges())[0]
    dst_ids=list(block_2_graph.edges())[1]
    g = dgl.graph((src_ids, dst_ids))

    block_to_graph = g # set g to the global variable: block to graph
    # global locked_nodes
    print('{}-'*40)
   
    # num_random_init=0
    # while not balance_flag:
    #     batched_seeds_list = gen_random_batched_seeds_list(batched_seeds_list)
    #     partition_dst_src_list = gen_partition_dst_src_list(batched_seeds_list)
    #     # use global variable partitions to generate input nodes lists for each partition
    #     balance_flag=balance_check_all_partitions(partition_dst_src_list,args.alpha)
    #     # print(balance_flag)
    #     num_random_init+=1
    #     # print('num_random_init, '+str(num_random_init))
    # # print(batched_seeds_list)
    # # print(partition_dst_src_list)
    print('{}---'*40)
    
    
    
    i=0
    for i in range(num_batch):
        # print(partition_list[i])
        tii=time.time()
        
        totalSteps=0
        totalSegments=0
        bottomList=[[]]
                
        A_o=batched_seeds_list[i]
        B_o=batched_seeds_list[i+1]
        len_A_part,len_B_part = get_src_nodes_len(A_o,B_o)
       
        tij=time.time()
        print('preparing two sides: ' , time.time()-tii)
        InitializeBitList([A_o,B_o])
        print('InitializeBitList: ' , time.time()-tij)
        tik=time.time()
        cost=getRedundancyCost(len_A_part,len_B_part,ideal_part_in_size)
        print('getRedundancyCost: ' , time.time()-tik)
        # tih=time.time()
        print('-'*80)
        
        pass_=0
        while pass_<10:
            if args.walkterm==0:
                ti=time.time()
                improvement,cost=walk_terminate_0(g,cost, args,ideal_part_in_size)
                print('walk_terminate_0 spend ', time.time()-ti)
                print(improvement)
                print(cost)
                if not improvement:
                    print(cost)
                    tmp=get_output_mini_batch(bit_dict)
                    if i<=num_batch-2:
                        batched_seeds_list=update_Batched_Seeds_list(batched_seeds_list, i, i+1)
                        # via bit_dict to update batched_seeds_list
                    elif i==num_batch-1:
                        print(batched_seeds_list)
                        print()
                        batched_seeds_list=update_Batched_Seeds_list_final(batched_seeds_list, i, i+1)
                        print(batched_seeds_list)
                        batched_seeds_list[0]=batched_seeds_list[-1]
                        batched_seeds_list=batched_seeds_list[:-1]
                        print(batched_seeds_list)
                    print('pass '+str(pass_)+'  '+str(tmp))
                    # res.append(tmp)
                    if i==num_batch-2:
                        batched_seeds_list.append(batched_seeds_list[0])
                    break
                    
            pass_ +=1

        #--------------------- initialization checking done   ----------------   
        maxGainIndex=0
        maxLeftGainIndex=0
        maxRightGainIndex=0
        maxDegree=0
        gains={}
        leftBuckets=[] 
        rightBuckets=[]
        totalSteps=0
        totalSegments=0
        bit_dict={}
        side=0
    
    print('-'*50)
    # print(res)
   
    weight_list=get_weight_list(batched_seeds_list)
    len_list=get_partition_src_list_len(batched_seeds_list)
    return batched_seeds_list, weight_list, len_list
    
def global_2_local(block_to_graph,batched_seeds_list):
    
    sub_in_nids = block_to_graph.srcdata['_ID'].tolist()
    # sub_out_nids = block_to_graph.dstdata['_ID'].tolist()
    global_nid_2_local = {sub_in_nids[i]: i for i in range(0, len(sub_in_nids))}
    # local_nid_2_global = { i: sub_in_nids[i] for i in range(0, len(sub_in_nids))}
    # ADJ_matrix=block_to_graph.adj_sparse('coo')
    t1=time.time()
    local_batched_seeds_list=[]
    for global_in_nids in batched_seeds_list:
        tt=time.time()
        local_in_nids = list(map(global_nid_2_local.get, global_in_nids))
        local_batched_seeds_list.append(local_in_nids)
    return local_batched_seeds_list

def local_2_global(block_to_graph,local_batched_seeds_list):
    
    sub_in_nids = block_to_graph.srcdata['_ID'].tolist()
    # sub_out_nids = block_to_graph.dstdata['_ID'].tolist()
    # global_nid_2_local = {sub_in_nids[i]: i for i in range(0, len(sub_in_nids))}
    local_nid_2_global = { i: sub_in_nids[i] for i in range(0, len(sub_in_nids))}
    # ADJ_matrix=block_to_graph.adj_sparse('coo')
    t1=time.time()
    global_batched_seeds_list=[]
    for local_in_nids in local_batched_seeds_list:
        tt=time.time()
        global_in_nids = list(map(local_nid_2_global.get, local_in_nids))
        global_batched_seeds_list.append(global_in_nids)
    return global_batched_seeds_list	
	

def  random_init_graph_partition(block_to_graph, args):
    tt = time.time()
    OUTPUT_NID, _ = torch.sort(block_to_graph.ndata[dgl.NID]['_N_dst'])
    batched_seeds_list,_ = gen_batched_seeds_list(OUTPUT_NID, args)
    
    t1 = time.time()
    batched_seeds_list=global_2_local(block_to_graph, batched_seeds_list) # global to local
    print('transfer time: ', time.time()-t1)
    #The graph_parition is run in block to graph local nids,it has no relationship with raw graph
    batched_seeds_list,weights_list, p_len_list=graph_partition(batched_seeds_list, block_to_graph, args)
    
    batched_seeds_list=local_2_global(block_to_graph, batched_seeds_list) # local to global
    print('graph partition total spend ', time.time()-t1)
    t2=time.time()-tt
    return batched_seeds_list, weights_list,t2, p_len_list
