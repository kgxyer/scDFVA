# scDFVA



## Requirements

Installing scDFVA with pip will attempt to install PyTorch and PyTorch Geometric, however it is recommended that the appropriate GPU/CPU versions are installed manually beforehand. For Linux:

1. Install PyTorch GPU: 

   ```conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c nvidia```

   or PyTorch CPU:  

   ```conda install pytorch torchvision torchaudio cpuonly -c pytorch```

   
   
2. Install PyTorch Geometric:  

   `conda install pyg -c pyg -c conda-forge`
   


## Usage

1.Pre-trained VGAE :

```bash
python -m scDFVA  --graph_type "PKNN" --k 20 --graph_convolution "GAT" --num_hidden_layers 2 --prevgae
```

2.Training :

```bash
python -m scDFVA  --graph_type "PKNN" --k 20 --graph_convolution "GAT" --num_hidden_layers 2 
```

## Details


```
optional arguments:
  
  --graph_type {KNN Scanpy,PKNN}
                        Type of graph.
  --k K                 K for KNN or Pearson (PKNN) graph.
  --graph_n_pcs GRAPH_N_PCS
                        Use this many Principal Components for the KNN (only Scanpy).
  --graph_metric {euclidean,manhattan,cosine}
  --graph_distance_cutoff_num_stds GRAPH_DISTANCE_CUTOFF_NUM_STDS
                        Number of standard deviations to add to the mean of distances/correlation values. Can be negative.
  --save_graph          Save the generated graph to the output path specified by --model_save_path.
  --raw_counts          Enable preprocessing recipe for raw counts.
  --prevgae           Pretrain VGAE.
  --select_genes  Number of selected genes
  --graph_file_path GRAPH_FILE_PATH
                        Graph specified as an edge list (one edge per line, nodes separated by whitespace, not comma), if not using command line options to generate it.
  --graph_convolution {GAT,GATv2,GCN}
  --num_hidden_layers {2,3}
                        Number of hidden layers (must be 2 or 3).
  --num_heads [NUM_HEADS [NUM_HEADS ...]]
                        Number of attention heads for each layer. Input is a list that must match the total number of layers = num_hidden_layers + 2 in length.
  --hidden_dims [HIDDEN_DIMS [HIDDEN_DIMS ...]]
                        Output dimension for each hidden layer. Input is a list that matches --num_hidden_layers in length.
  --dropout [DROPOUT [DROPOUT ...]]
                        Dropout for each layer. Input is a list that must match the total number of layers = num_hidden_layers + 2 in length.
  --latent_dim LATENT_DIM
                        Latent dimension (output dimension for node embeddings).
  --loss {kl,mmd}       Loss function (KL or MMD).
  --lr                Learning rate for Adam.
  --epochs EPOCHS       Number of training epochs.
  --val_split VAL_SPLIT
                        Validation split e.g. 0.1.
  --test_split TEST_SPLIT
                        Test split e.g. 0.1.
  --transpose_input     Specify if inputs should be transposed.
  --use_linear_decoder  Turn on a neural network decoder, similar to traditional VAEs.
  --decoder_nn_dim1 DECODER_NN_DIM1
                        First hidden dimenson for the neural network decoder, if specified using --use_linear_decoder.
  --name NAME           Name used for the written output files.
  --model_save_path MODEL_SAVE_PATH
                        Path to save PyTorch model and output files. Will create the entire path if necessary.
```
