### 1. Parameters in the config file     
* sequence   
    * no_token_type_embeddings      
      bool: whether to include token type embedding      
    * config.hidden_size     
      int: the embedding size and hidden size of Transformer      
    * config.intermediate_size     
      int: the intermediate layer size of Transformer
    * config.max_position_embeddings    
      int: the maximum sequence length allowed of Transformer      
    * config.num_attention_heads        
      int: the head count of Transformer      
    * config.num_hidden_layers    
      int: the block number of Transformer         
    * config.pooler_fc_size      
      int: the size of fully connected layer in Transformer-block      
    * config.pooler_num_attention_heads       
      int: the head count of fully connected layer in Transformer-block
    * config.pooler_num_fc_layers     
      int: the layer number of fully connected layer in Transformer-block
    * config.pooler_size_per_head    
      int: the pooler size per head 
    * config.vocab_size      
      int: the vocab size of sequence tokenizer  
    * args.seq_pooling_type     
      int: the pooling type of sequence encoder   
    * config.seq_fc_size      
      list[int]: the hidden size of sequence encoding pooler     
  
* structure      
  * config.struct_vocab_size   
    int: the node size of structure encoder      
  * config.struct_embed_size  
    int: the embedding size of structure encoder 
  * config.struct_hidden_size        
    list[int]: the hidden size of structure encoder
  * config.struct_output_size        
    list[int]: the output size of structure encoder
  * config.struct_nb_heads      
    int: the head count of structure encoder           
  * config.struct_alpha       
    float: the alpha value of structure encoder           
  * args.struct_pooling_type    
    int: the pooling type of structure encoder       
  * config.struct_fc_size        
    list[int]: the hidden size of fully connected layer of structure encoder      

* common       
  * config.seq_weight    
    float: the weight of sequence representaion vector      
  * config.struct_weight       
    float: the weight of structure representaion vector        
  * config.hidden_dropout_prob        
    float: the dropout rate between pooling layer and dense layer 
  * config.num_labels   
    int: the label size       
  * args.output_mode     
    str: task type, binary-class, multi-class, multi-label       
  * args.sigmoid     
    bool: sigmoid for binary-class and multi-label classification, multi-class classification not need

* loss     
  * args.loss_type      
    str: loss type, focal_loss, bce, multilabel_cce, asl, ce      
  * config.pos_weight      
    list[float]: length=1(binary_class) or num_labels, the positive samples weight for bce loss  
  * config.weight  
    list[float]: length=num_labels， the weight for multi-class classification 
  * args.asl_gamma_neg     
    float: for AsymmetricLoss      
  * args.asl_gamma_pos       
    float: for AsymmetricLoss    
  * args.focal_loss_alpha    
    float: for FocalLoss
  * args.focal_loss_gamma    
    float: for FocalLoss
  * args.focal_loss_reduce   
    float: for FocalLoss     
     