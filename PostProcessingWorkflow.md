# Guidance of Predicted RdRPs Classification    

## 1. Prediction using LucaProt     
For the usage of **LucaProt** prediction commands, please refer to **`README.md`**.     

The first step is to perform `viral-RdRPs` prediction on your `FASTA` file (protein sequences) using `LucaProt`.   
By default, a **`threshold`** of **0.5** will be used to obtain a set of candidate viral-RdRPs, denoted as (`S1`):     
`RdRP.LucaProt.positives.fasta`

**Note:** The **threshold** is a real value between `0.0` and `1.0`.       
A higher value may reduce the rate of false positives, while a lower value may decrease the false negatives rate.         
In other words, a higher value increases the `Precision`, while a lower value increases the `Recall`.       
The value to be used depends on your actual application. For example, if you would like to have more viral-RdRPs for further analysis, you could consider lowering the threshold, such as to **0.1**.  

## 2. Classification of Prediction RdRP Sequences                   
**Align** the predicted viral-RdRPs(`S1`) with the **180 RdRP-SuperGroups** in our work to confirm the classification of the candidate RdRPs.           

Use `Diamond Blastp` to align the candidate RdRPs(`S1`) with the RdRPs dataset (**180 RdRP-SuperGroups**).       
The RdRPs dataset for the 180 SuperGroups in our paper can be downloaded at:   
https://figshare.com/articles/thesis/The_LucaProt-Related_Resources/26298802/16 (the folder `RdRP_dataset/orf/` in the file `Results.tar.gz`)     
or       
http://47.93.21.181/Results/RdRP_dataset/orf/     

## 2.1 Merge All Sequences of the 180-SuperGroups into a Database        
Download all the sequences of the 180 SuperGroups from the aforementioned links,      
and merge all the sequences into a single `FASTA` file named **`80supergroup_rdrp_orf.fas`**.    

## 2.2 Build Database Index        
```shell
diamond makedb --in 180supergroup_rdrp_orf.fas --db 180supergroup_rdrp_orf;
```  

## 2.3 Alignment
**Align** the candidate RdRPs (`S1`) to the 180 SuperGroups using a threshold of `1E-3`. The hit SuperGroup ID will serve as the classification label for the corresponding sequence.    
```shell
diamond blastp -q RdRP.LucaProt.positives.fasta  -d 180supergroup_rdrp_orf -o RdRP.LucaProt.positives.180supergroup -e 1E-3 -k 1 -p 48 -f 6 qseqid sseqid qlen length pident evalue;
``` 

## 2.4 Extract Classification Labels     
From the alignment result file, extract the sequences aligned to each SuperGroup separately, denoted as:     

RdRP.LucaProt.positives.supergroup001.fasta       
RdRP.LucaProt.positives.supergroup002.fasta     
...     
RdRP.LucaProt.positives.supergroupXXX.fasta     

# 2.5 Confirmation with conserved RdRP domains        
Merge the candidate RdRP sequences of each SuperGroup with the corresponding reference RdRP sequences from the dataset,    
and perform multiple sequence alignment(MSA) to confirm whether they contain the complete conserved RdRP domains (motif C/motif A/motif B, etc.).      
This will allow us to obtain sequences that contain the complete conserved RdRP domains.       
Furthermore, the resulting multiple sequence alignment results can also be further used to construct a phylogenetic tree to confirm their phylogenetic positions.        

```shell   
mafft --localpair --maxiterate 1000 --thread 10 --reorder RdRP.LucaProt.positives.supergroup001.fasta > RdRP.LucaProt.positives.supergroup001.aln.fasta    
...      
mafft --localpair --maxiterate 1000 --thread 10 --reorder RdRP.LucaProt.positives.supergroupXXX.fasta > RdRP.LucaProt.positives.supergroupXXX.aln.fasta    
```


# 3. Processing of Unmatched RdRPs Sequences         
These RdRPs is likely to be new findings. We need perform more validation on these RdRPs that did not align with the 180 SuperGroups RdRPs database.        

## 3.1 Unmatched RdRPs Sequences     
Extract the remaining sequences that did not hit the dataset `180supergroup_rdrp_orf` from the **Diamond BLASTp** results, denoted as:
`RdRP.LucaProt.positives.novel.fasta`        

## 3.2 Clustering           
Clustering the unmatched sequences using different identity values:   

**90% identity**   
```shell
cd-hit -i RdRP.LucaProt.positives.novel.fasta -o RdRP.LucaProt.positives.novel.fasta.90 -c 0.9 -n 5 -g 1 -G 0 -aS 0.8 -d 0 -p 24 -T 128 -M 0;  
```

**60% identity**     
```shell
cd-hit -i RdRP.LucaProt.positives.novel.fasta.90 -o RdRP.LucaProt.positives.novel.fasta.60 -c 0.6 -n 4 -g 1 -G 0 -aS 0.8 -d 0 -p 24 -T 128 -M 0;
```

**20% identity**       
```shell
psi-cd-hit.pl -i RdRP.LucaProt.positives.novel.fasta.60 -o RdRP.LucaProt.positives.novel.fasta.20 -c 0.2 -ce 1e-3 -aS 0.5 -G 1 -g 1 -exec local -para 24 -blp 16;   
```

## 3.3 Domain Confirmation of Clusters       
Perform multiple sequence alignment using **`mafft`** for each cluster,      
and manually verify whether it contains the conserved RdRP domains (such as motif C, motif A, motif B, etc.).    
```shell
mafft --localpair --maxiterate 1000 --thread 10 --reorder ${novel.clstr.fasta} > ${novel.clstr.aln.fas}  
```  

## 3.4 Further Analysis of Clusters         
For clusters where conserved RdRP domains can be detected and contain more than 10 sequences,    
further analysis can be conducted, including merging to determine new supergroups.   
Details can be found in the **`Methods`** section of the article.    