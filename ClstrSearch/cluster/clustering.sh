#!/usr/bin/env bash

######################## clusering job ############################

#Seqfiles=/PATH/to/all_300aa.pep
#Reffiles=/PATH/TO/RdRpdb


###################################################################
##################### compare against ref RdRP ####################  
###################################################################

diamond blastp -q all_300aa.pep -d RdRpdb -o all_300aa.pep.out -e 1E+5 -k 1 -p 30 -f 6 qseqid qlen sseqid stitle pident length evalue qstart qend sstart send;  ### using 1E+5 to identify more divergent RdRP proteins
cat all_300aa.pep.out | cut -f1 | sort -u > all_300aa.pep.out.id;
seqtk subseq all_300aa.pep all_300aa.pep.out.id > all_300aa.pep.out.fa;
rm all_300aa.pep.out.id;


####################################################################
##################### multi-step clustering  #######################
####################################################################

cd-hit -i all_300aa.pep.out.fa -o all_300aa.pep.out.fa.90 -c 0.9 -n 5 -g 1 -G 0 -aS 0.8 -d 0 -p 1 -T 128 -M 0;  ### 90% identity
cd-hit -i all_300aa.pep.out.fa.90 -o all_300aa.pep.out.fa.60 -c 0.6 -n 4 -g 1 -G 0 -aS 0.8 -d 0 -p 1 -T 128 -M 0;  ### 60% identity
psi-cd-hit.pl -i all_300aa.pep.out.fa.60 -o all_300aa.pep.out.fa.20 -c 0.2 -ce 1e-3 -aS 0.5 -G 1 -g 1 -exec local -para 8 -blp 16;  ### 20% identity

clstr_rev.pl all_300aa.pep.out.fa.90.clstr all_300aa.pep.out.fa.60.clstr > all_300aa.pep.out.fa.90-60.clstr;    ### combines a .clstr file with its parent .clstr file
clstr_rev.pl all_300aa.pep.out.fa.90-60.clstr all_300aa.pep.out.fa.20.clstr > all_300aa.pep.out.fa.90-60-20.clstr;   ### combines a .clstr file with its parent .clstr file

clstr_sort_by.pl < all_300aa.pep.out.fa.90-60-20.clstr no > all_300aa.pep.out.fa.90-60-20.clstr.sort;  ### clstr sort by size
clstr2txt all_300aa.pep.out.fa.90-60-20.clstr.sort > all_300aa.pep.out.fa.90-60-20.clstr.sort.tbl;    ### clstr to tbl

rm -r all_300aa.pep.out.fa*pl;
rm -r all_300aa.pep.out.fa*sh;
rm -r all_300aa.pep.out.fa*out;
rm -r all_300aa.pep.out.fa*log;
rm -r all_300aa.pep.out.fa*restart;
rm -r all_300aa.pep.out.fa*-bl;
rm -r all_300aa.pep.out.fa*-blm;
rm -r all_300aa.pep.out.fa*-seq;
