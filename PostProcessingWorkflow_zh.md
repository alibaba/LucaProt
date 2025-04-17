# RdRPs分类的Workflow        

## 1. **LucaProt预测**     
**LucaProt**预测命令的使用教程参见**`README.md`**.   

首先，基于LucaProt对你的蛋白质序列fasta文件进行viral-RdRP预测，默认使用预测的(`threshold=0.5`)得到预测为True的候选RdRP序列集合，记为(`S1`)：  
`RdRP.LucaProt.positives.fasta`           

**注意：** **`threshold`**取值为**`0~1`**之间的实数，**越大**，**假阳性**可能**越低**；**越小**，则**假阴性**可能**越低**。也就是说，**越大Precision越高**；**越小Recall越高**。     
具体看你实际需求。比如：你想多一点疑似RdRP序列进行后续分析，那么可以降低**`threshold`**，比如**`0.1`**。


## 2. 预测结果分类   
比对我们文章中的**180个RdRP SuperGroups**，确认候选RdRP的分类。           

基于**`Diamond Blastp`**比对RdRP的蛋白数据集，该数据集包含基于同源性和扩张后的RdRP的所有序列，已分类并确认为180个SuperGroup。
我们文章中得到的180个SuperGroups的RdRP蛋白数据集地址：     
https://figshare.com/articles/thesis/The_LucaProt-Related_Resources/26298802/16 (the folder `RdRP_dataset/orf/` in the file `Results.tar.gz`)    
或者       
http://47.93.21.181/Results/RdRP_dataset/orf/     

### 2.1 180-SuperGroups 合并成库      
根据上述的地址下载这180个SuperGroups的序列，   
并将上述的180个SuperGroups的所有fasta文件合并到一个文件中 **`80supergroup_rdrp_orf.fas`** 

### 2.2 构建比对索引   
```shell
diamond makedb --in 180supergroup_rdrp_orf.fas --db 180supergroup_rdrp_orf;
```  

### 2.3 Blastp比对      
比对候选viral-RdRPs(`S1`)到180个uperGroups数据库，使用的阈值为1E-3，hit到的Supergroup的id即为该序列对应的分类标签。    

```shell
diamond blastp -q RdRP.LucaProt.positives.fasta  -d 180supergroup_rdrp_orf -o RdRP.LucaProt.positives.180supergroup -e 1E-3 -k 1 -p 48 -f 6 qseqid sseqid qlen length pident evalue;
``` 

### 2.4 分类结果提取      
根据比对结果文件，分别提取每一个SuperGroup比对上的序列，分别记为:      

RdRP.LucaProt.positives.supergroup001.fasta      
RdRP.LucaProt.positives.supergroup002.fasta     
...     
RdRP.LucaProt.positives.supergroupXXX.fasta     

### 2.5 Domain确认    
将每个超群的候选RdRPs序列与数据集中对应超群的参考RdRPs序列合并，并进行多序列比对，确认其是否含有完整的RdRP保守的domain（motif C/motif A/motif B等)，      
可以拿到具有完整RdRP保守domin的序列信息。
同时得到的多序列比对文件可以进一步用于构建进化树等确认其系统发育位置。          
```shell   
mafft --localpair --maxiterate 1000 --thread 10 --reorder RdRP.LucaProt.positives.supergroup001.fasta > RdRP.LucaProt.positives.supergroup001.aln.fasta    
...      
mafft --localpair --maxiterate 1000 --thread 10 --reorder RdRP.LucaProt.positives.supergroupXXX.fasta > RdRP.LucaProt.positives.supergroupXXX.aln.fasta    
```


## 3. 未比对成功的RdRPs序列处理   
这一部分RdRPs很可能是新发现(new findings)，当然也可能是假阳（可以先去与库比对)，对未比对到180-SuperGroups-RdRPs数据库（也排除了假阳）的序列进行进一步确认。           

### 3.1 未比对上的RdRPs序列        
提取diamond blastp剩余未hit到180supergroup_rdrp_orf数据集的序列，记为:   
`RdRP.LucaProt.positives.novel.fasta`        

### 3.2 聚类成簇         
对未必对上的序列基于相似性聚类为多个clstr簇              
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

### 3.3 簇Domain确认     
分别对每个clstr簇进行多序列比对(使用**mafft**)，人工验证确认其是否含有RdRP保守的domain(motif C/motif A/motif B等)         
```shell
mafft --localpair --maxiterate 1000 --thread 10 --reorder ${novel.clstr.fasta} > ${novel.clstr.aln.fas}  
```  

### 3.4 簇进一步分析      
对于可以检测到RdRP保守domain的clstr，且序列超过10条的clstr，可以进行下一步分析，包括Merge后确定为新的超群，具体可见文章中的**`Methods`**章节。       