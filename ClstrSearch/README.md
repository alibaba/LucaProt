# ClstrSearch     

A conventional approach that clustered all proteins based on their sequence homology and then used BLAST or HMM models to identify any resemblance to viral RdRPs or non-virus RdRP proteins.   


## 1. Clustering     
Change the input and ref db in "clustering.sh" then run    
```shell
cd cluster
sh clustering.sh      
```   
the clstr results were then subjected to RdRP hmmscan and muanual inspection


## 2. Expand      
After RdRP clstr identified, the RdRP divsersity was expand by 10 iterations    
```shell
cd expand         
sh expand       
```   


## 3. Merge      
Clstr merge to superclades        
using pariwise blast median evalue of clstrs as input (change the dir to your work path)      
```shell
cd merge 
Rscript merge_scirpt.R   
```   
