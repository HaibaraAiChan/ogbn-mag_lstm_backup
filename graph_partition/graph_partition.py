import numpy 
import dgl
import torch
import time

maxLeftGainIndex=0; 

maxRightGainIndex=0;

maxDegree=0;

maxGainIndex=0; 

leftBuckets=[] ;

rightBuckets=[];

gains={};
bit_dict={};
totalSteps=0;
totalSegments=0;
side=0;


def InitializeBitList(p_list):
    global bit_dict
    nums_p=len(p_list); 
    bit_dict={}
    for i in range (nums_p-1):
        A = p_list[i];
        B = p_list[i + 1]; 
        for k in A:
            bit_dict[k] = 0;
        for m in B:
            bit_dict[m] = 1;
        
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
    
# def isNeighbor(g, i, j):
#     g.has_edges_between(i,j)
#     neighbors_i=g.in_edges(i)[0].tolist()
#     neighbors_j=g.in_edges(j)[0].tolist()
#     print(neighbors_i)
#     print(neighbors_j)
#     print()
#     if (i in neighbors_j) and (j in neighbors_i):
#         return True
#     return False
    
def getCutCost(raw_graph, A_o, B_o, bit_dict):
    cost =0
    N_O=len(bit_dict) # total output nids of two partitions
    for i in A_o:
        if bit_dict[i]==0:
            for j in B_o:
                if raw_graph.has_edges_between(i,j) and bit_dict[j]==1:
                    cost+=1
    return cost
    
def getRedundancyCost(A_in, B_in, ideal_part_in_size):
    cost =0
    ratio_A=len(A_in)/ideal_part_in_size
    ratio_B = len(B_in)/ideal_part_in_size
    cost = max(ratio_A,ratio_B)
    return cost   #   minimize the max ratio of 
    
def get_mini_batch_size(full_len,num_batch):
	mini_batch=int(full_len/num_batch)
	if full_len%num_batch>0:
		mini_batch+=1
	print('current mini batch size of output nodes ', mini_batch)
	return mini_batch
	
def gen_batch_output_list(OUTPUT_NID,indices,mini_batch):
	
	map_output_list = list(numpy.array(OUTPUT_NID)[indices])
		
	batches_nid_list = [map_output_list[i:i + mini_batch] for i in range(0, len(map_output_list), mini_batch)]
			
	output_num = len(OUTPUT_NID.tolist())
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
	mini_batch=get_mini_batch_size(full_len,num_batch)
	
	if selection_method == 'random_init_graph_partition' :
		indices = random_shuffle(full_len)
		
	elif selection_method == 'similarity_init_graph_partition':
		indices = torch.tensor(range(full_len)) #----------------------------TO DO
	
	batches_nid_list, weights_list=gen_batch_output_list(OUTPUT_NID,indices,mini_batch)
	
	return batches_nid_list, weights_list


def initializeBuckets(raw_graph,args):
    global maxDegree; global leftBuckets; global rightBuckets;
    leftBuckets=[]
    rightBuckets=[]
    max_neighbors=int(args.fan_out) # only for 1-layer model
    max_in_degree = max((raw_graph.in_degrees().tolist()))
    print(max_in_degree)
    print(max_neighbors)
    maxDegree= min(max_neighbors, max_in_degree)
    print('maxDegree ',maxDegree)
    
    
    # initializeBuckets leftBuckets and rightBuckets
    for i in range(2*maxDegree +1):
        leftBuckets.append([])
        rightBuckets.append([])
    
    print('leftBuckets ',leftBuckets)
    print(leftBuckets[5])
    return
    
def initializeBucketSort(nid, gain, side):
    global maxGainIndex, leftBuckets, rightBuckets 
    global maxLeftGainIndex
    global maxRightGainIndex
    global gains
    
    maxLeftGainIndex=0
    maxRightGainIndex=0
    
    index = maxDegree+gain
    
    if side ==0:
        maxGainIndex = maxLeftGainIndex
        maxGainIndex= max(index, maxGainIndex)
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
    
    
    
    
def initializeGain(raw_graph, bit_dict):
    global gains
    gains={}
    A_o=[k for k in bit_dict.keys() if bit_dict[k] == 0]
    B_o=[k for k in bit_dict.keys() if bit_dict[k] == 1]
    for i in A_o:
        gain=0
        for j in B_o:
            if raw_graph.has_edges_between(i,j) :
                if bit_dict[j]==bit_dict[i]:
                    gain-=1
                else:
                    gain+=1
        gains=initializeBucketSort(i, gain, 0) # 0: left side bucket
        
    for i in B_o:
        gain=0
        for j in A_o:
            if raw_graph.has_edges_between(i,j) :
                if bit_dict[i]==bit_dict[j]:
                    gain-=1
                else:
                    gain+=1
        gains=initializeBucketSort(i, gain, 1) # 1: right side bucket
                
    return 


   
    
def genPartition(raw_graph,OUTPUT_NID,args):
    batched_output_nid_list, weights_list = gen_batched_seeds_list(OUTPUT_NID, args)
    
    partition_list=[]
    for seeds in batched_output_nid_list:
        frontier = list(raw_graph.in_edges(seeds))[0]
        # now frontier need remove duplicated nids. e.g. tensor([ 0,  2, 30, 32, 33,  0,  1,  0,  4,  5,  0,  1, 33, 32, 33,  0])
        frontier= list(set(frontier.tolist()))
        partition_list.append([frontier,seeds])
        print(seeds)
        print(frontier)
    return partition_list,weights_list
    



def  random_init_graph_partition(raw_graph, block_to_graph, args):
	tt = time.time()
	OUTPUT_NID, _ = torch.sort(block_to_graph.ndata[dgl.NID]['_N_dst'])
	partition_list,weights_list=genPartition(raw_graph, OUTPUT_NID, args)
	batched_output_list=graph_partition(raw_graph, partition_list, block_to_graph, args)
	t2=time.time()-tt
	return batched_output_list, weights_list,t2




    
    
    
    
    
def gen_partition(raw_graph, bit_dict):
    A_o=[k for k in bit_dict if bit_dict[k] == 0]
    B_o=[k for k in bit_dict.keys() if bit_dict[k] == 1]
    
    A_in = list(raw_graph.in_edges(A_o))[0]
    A_in= list(set(A_in.tolist()))
    
    B_in = list(raw_graph.in_edges(B_o))[0]
    B_in= list(set(B_in.tolist()))
    return A_in, B_in, A_o, B_o
    
def incrementGain(cur_bucket, nid, side):
    
    global maxGainIndex, maxLeftGainIndex, maxRightGainIndex, maxDegree, gains 
    
    
    if side == 1:
        maxGainIndex=maxLeftGainIndex
    else:
        maxGainIndex=maxRightGainIndex
    
    # remove node from current linked list 
    gain = gains[nid]
    bucketIndex=cur_bucket[maxDegree+gain]
    if bucketIndex:
        bucketIndex.remove(nid) # remove node nid 
    
    nextBucketIndex = cur_bucket[maxDegree+gain+2]    
    nextBucketIndex.insert(0,nid)
    gains[nid]=gain+2
    maxGainIndex = max(maxGainIndex, maxDegree + gain)
    
    if side == 1:
        maxLeftGainIndex=maxGainIndex
    else:
        maxRightGainIndex=maxGainIndex
    
    return cur_bucket
    
    
def decrementGain(comp_bucket, nid, side):
    global maxGainIndex, maxLeftGainIndex, maxRightGainIndex, maxDegree , gains
    if side == 1:
        maxGainIndex=maxLeftGainIndex
    else:
        maxGainIndex=maxRightGainIndex
    
    # remove node from current linked list 
    gain = gains[nid]
    bucketIndex=comp_bucket[maxDegree+gain]
    if bucketIndex:
        bucketIndex.remove(nid) # remove node nid 
    
    nextBucketIndex = comp_bucket[maxDegree+gain-2]    
    nextBucketIndex.insert(0,nid)
    
    # if the current linkedlist is empty only then check to reduce the maxgainIndex
    tmp_list=comp_bucket[maxDegree+gain]
    if maxGainIndex==(maxDegree + gain) and len(tmp_list)==0:
        maxGainIndex-=1 # Reduce only if it is equal to the only highest gain vertex
    gains[nid]=gain-2
    
    if side == 1:
        maxLeftGainIndex=maxGainIndex
    else:
        maxRightGainIndex=maxGainIndex
    
    
    return comp_bucket
    
    
def updateGain(raw_graph, locked_nodes):
    global maxGainIndex, maxLeftGainIndex, maxRightGainIndex, maxDegree, leftBuckets, rightBuckets 
    global side, bit_dict
    
    maxGainNid=None
    currentSide=side
    cur_bucket=[]
    comp_bucket=[]
    
    if side==0:
        cur_bucket=leftBuckets
        comp_bucket=rightBuckets
        index=0
        ifFound=False
        while not ifFound:
            bucketIndex = cur_bucket[maxLeftGainIndex-index] # find the maxgain node, if it not work to find the second oder gain node.
            while bucketIndex:
                i=bucketIndex[0]
                if   not locked_nodes[i]:
                    maxGainNid = i
                    ifFound = True
                    break
                bucketIndex=bucketIndex[1:]
            index+=1
            if index >= maxDegree:
                break
        
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
                i=bucketIndex[0]
                if   not locked_nodes[i]:
                    maxGainNid = i
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
    
    in_nodes=(list(raw_graph.in_edges(maxGainNid))[0]).tolist()
    for idx in in_nodes:
        if idx not in locked_nodes:
            continue
        if not locked_nodes[idx]:    # only consider the outputnodes connected with maxGainNid node.
            if bit_dict[idx]==currentSide:
                cur_bucket=incrementGain(cur_bucket, idx, currentSide)
            else:
                comp_bucket=decrementGain( comp_bucket, idx, currentSide)
    return 
                
def balance_check(partition_list,alpha):
    balance_flag = True
    for i in range(len(partition_list)-1):
        A_in= partition_list[i][0]
        B_in= partition_list[i+1][0]
        avg = (len(A_in)+len(B_in))/2
        if abs(len(B_in)-len(A_in)) >alpha*avg:
            balance_flag=False
            return balance_flag
    return balance_flag
    
def balance_check_and_exchange(raw_graph, bit_dict,alpha):
    global side
    A_o=[k for k in bit_dict.keys() if bit_dict[k] == 0]
    B_o=[k for k in bit_dict.keys() if bit_dict[k] == 1]
    A_in = list(raw_graph.in_edges(A_o))[0]
    A_in= list(set(A_in.tolist()))
    
    B_in = list(raw_graph.in_edges(B_o))[0]
    B_in= list(set(B_in.tolist()))
    avg = (len(A_in)+len(B_in))/2
    side = 0
    if len(A_in)>0 and (len(B_in)-len(A_in)) > 0:
        side = 1
        for k in A_o:
            bit_dict[k] = 1;
        for m in B_o:
            bit_dict[m] = 0;
   
    return bit_dict, side
    

def walk_terminate_0( raw_graph,cost,args, ideal_part_in_size):
    global totalSegments
    global totalSteps
    global bit_dict
    global side
    print('walk_terminate_0')
    print(bit_dict)
       # initialize locked nodes
    bestCost = cost   # best cost in this segment
    bottom = bit_dict   # bottom of this segment
    flag = False
    best_bit_dict=bottom  # best bit dict in this segment
    initializeBuckets(raw_graph, args)  # Invokes a method to calculate max Degree and initializes bucket with 2*maxDegree+1 size
    initializeGain(raw_graph, bit_dict) # compute initial gain for the segment
    bit_dict, side=balance_check_and_exchange(raw_graph, bit_dict, args.alpha)
    A_in, B_in, A_o, B_o = gen_partition(raw_graph, bit_dict)
    subgraph_o = A_o+B_o
    locked_nodes={id:False for id in subgraph_o}
    
    
    # begin segment
    for i in subgraph_o:
        updateGain(raw_graph, locked_nodes)
        totalSteps+=1
        A_in, B_in, A_o, B_o = gen_partition(raw_graph, bit_dict)
        print(bit_dict)
        tmpCost=getRedundancyCost(A_in, B_in, ideal_part_in_size)
        if tmpCost < bestCost: 
            bestCost = tmpCost
            best_bit_dict = bit_dict
            flag = True
		
        elif (tmpCost == bestCost) and (not flag)  :
            flag = True
            bottom =  bit_dict
            best_bit_dict = bit_dict
        # bit_dict=balance_check_and_exchange(raw_graph, bit_dict, args.alpha)
		
    totalSegments +=1 
    if (bestCost < cost) : #is there improvement?
        bit_dict = best_bit_dict
        cost = bestCost
        return True, bit_dict
	
    bit_dict = best_bit_dict
    return False
        
def get_output_mini_batch(bit_dict):
    res=[i for i in bit_dict if bit_dict[i]==0]
    return res



def graph_partition(raw_graph, partition_list, block_to_graph, args):
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
    # partition_list is res of genPartition: list of [src,dst]
    full_batch_graph_nids_size=len(block_to_graph.srcdata['_ID'])
    ideal_part_in_size=(full_batch_graph_nids_size/args.num_batch)
    
    num_batch=args.num_batch
    balance_flag = False
    res=[]
    # print(partition_list)
    print('{}-'*40)
    i=0
    while not balance_flag:
        balance_flag=balance_check(partition_list,args.alpha)
        print('balance checking ',  i)
        i+=1
    print('{}---'*40)
    
    for i in range(num_batch-1):
        # print(partition_list[i])
        totalSteps=0
        totalSegments=0
        bottomList=[[]]
        A_in= partition_list[i][0]
        B_in= partition_list[i+1][0]
                
        A_o=partition_list[i][1]
        B_o=partition_list[i+1][1]
        
        InitializeBitList([A_o,B_o])
        cost=getRedundancyCost(A_in,B_in,ideal_part_in_size)
        
        pass_=0
        while pass_<2:
            if args.walkterm==0:
                if not walk_terminate_0(raw_graph,cost, args,ideal_part_in_size):
                    tmp=get_output_mini_batch(bit_dict)
                    res.append(tmp)
                    
            
            pass_ +=1
        
        # res.append(tt)
    
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
    print('-'*50)
    print(res)
    return res
    
 
	
	

