# ogbn-mag_lstm_backup
### step 1 
pseudo_ogbn_mag_same_subgraph.py
Generate 6 full batch subgraphs and save to ‘DATA/’ folder\n
{dataset}_{epoch-num}_subgraph.bin    # del binary format file for save graphs
### step 2 
### train one (batch size + batch NID selection method) combination data with block dataloader . e.g.(15341, 'range')
pseudo_ogbn_mag.py      
block_dataloader_graph.py

### step 3 
### Run training process to collect results
./ds_pseudo_run.sh

### --------------------------------------------------------------------



### basic result for comparison
products_pseudo_final_version.py
block_dataloader.py

