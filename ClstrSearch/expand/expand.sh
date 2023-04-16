#!/bin/bash
#PBS -N expand
#PBS -l ncpus=128
#PBS -l walltime=9999:00:00
#PBS -M houx5@mail2.sysu.edu.cn
#PBS -m abe
#PBS -o expand.o
#PBS -e expand.e


######################## expand job ############################

#Seqfiles=/PATH/to/all_300aa.pep
#Reffiles=/PATH/TO/Update_RdRpdb (all known rdrp clstr and novel rdrp clstr identified by clusering process)


#Go to location to process files, change for each list
cd /home/houxin/data/clstr_expansion/


files=(
1
2
3
4
5
6
7
8
9
10
)

cp ../RdRp_20230215.fas ./;  ### Update_RdRpdb 
cat ../RdRp_20230215.fas | grep ">" | sed 's/>//' > RdRp_20230215.id;

for file in "${files[@]}";
	do
		cat RdRp_20230215.fas > RdRp_I"$file".fas;
		diamond makedb --in RdRp_I"$file".fas --db RdRp_I"$file";

		diamond blastp -q all_300aa.pep -d ./RdRp_I"$file" -o all_300aa.RdRp_I"$file" -e 1E-3 -k 1 -p 120 -f 6 qseqid sseqid qlen length pident evalue;
		cat all_300aa.RdRp_I"$file" | cut -f1 | sort -u > all_300aa.RdRp_I"$file".id;
		seqtk subseq all_300aa.pep all_300aa.RdRp_I"$file".id > all_300aa.RdRp_I"$file".fas;

		diamond_v2.0.14 blastp -q all_300aa.RdRp_I"$file".fas -d /home/houxin/database/protein/refseq_protein -o all_300aa.RdRp_I"$file".refseq_protein -e 1E-5 -k 1 -p 120 -f 6 qseqid qlen sseqid stitle pident length evalue;
		cat all_300aa.RdRp_I"$file".refseq_protein | grep -i -E "virus|phage|viridae|virales|AlphaCoV|viurs|Newbury agent" | cut -f1 | sort -u >  all_300aa.RdRp_I"$file".refseq_protein.virus.id;
		cat all_300aa.RdRp_I"$file".refseq_protein | cut -f1 | sort -u >  all_300aa.RdRp_I"$file".refseq_protein.id;
		grep -v -F -f all_300aa.RdRp_I"$file".refseq_protein.id all_300aa.RdRp_I"$file".id >> all_300aa.RdRp_I"$file".refseq_protein.virus.id;
		seqtk subseq all_300aa.RdRp_I"$file".fas all_300aa.RdRp_I"$file".refseq_protein.virus.id > all_300aa.RdRp_I"$file".refseq_protein.virus.fas;
		
		diamond blastp -q all_300aa.RdRp_I"$file".refseq_protein.virus.fas -d RdRp_I"$file" -o all_300aa.RdRp_I"$file".refseq_protein.virus.fas.RdRp -e 1E-3 -k 1 -p 120 -f 6 qseqid sseqid qlen length pident evalue;
		cat all_300aa.RdRp_I"$file".refseq_protein.virus.fas.RdRp | cut -f2 | sed 's/_.*//' | sed 's/NEW-//' > all_300aa.RdRp_I"$file".refseq_protein.virus.fas.RdRp.sclstr;
		paste all_300aa.RdRp_I"$file".refseq_protein.virus.fas.RdRp all_300aa.RdRp_I"$file".refseq_protein.virus.fas.RdRp.sclstr | awk -F "\t" 'BEGIN{OFS="\t"} {print $1, $7, $2, $3, $4, $5, $6}' > all_300aa.RdRp_I"$file".refseq_protein.virus.fas.RdRp.edited;
		cat all_300aa.RdRp_I"$file".refseq_protein.virus.fas.RdRp.edited | awk '!seen[$1]++' > all_300aa.RdRp_I"$file".refseq_protein.virus.fas.RdRp.edited.tophit;
		cat all_300aa.RdRp_I"$file".refseq_protein.virus.fas.RdRp.edited.tophit | awk '$6<=80' > all_300aa.RdRp_I"$file".refseq_protein.virus.fas.RdRp.edited.tophit.filt;
	
		for clstr in `ls ./db`;
			do
				cat all_300aa.RdRp_I"$file".refseq_protein.virus.fas.RdRp.edited.tophit.filt | awk '$2~/'${clstr}'/' | cut -f1 | sort -u > "$clstr".id;
				seqtk subseq all_300aa.RdRp_I"$file".fas "$clstr".id | sed "s/>/>NEW-"$clstr"_/" >> all_300aa.RdRp_I"$file".refseq_protein.virus.fas.RdRp.edited.tophit.filt.fas;
				rm "$clstr".id;
			done;
		
		cd-hit -i all_300aa.RdRp_I"$file".refseq_protein.virus.fas.RdRp.edited.tophit.filt.fas -o all_300aa.RdRp_I"$file".refseq_protein.virus.fas.RdRp.edited.tophit.filt.fas.90 -c 0.9 -n 5 -g 1 -G 0 -aS 0.8 -d 0 -p 1 -T 120 -M 0
		cd-hit -i all_300aa.RdRp_I"$file".refseq_protein.virus.fas.RdRp.edited.tophit.filt.fas.90 -o all_300aa.RdRp_I"$file".refseq_protein.virus.fas.RdRp.edited.tophit.filt.fas.60 -c 0.6 -n 4 -g 1 -G 0 -aS 0.6 -d 0 -p 1 -T 120 -M 0;

		cat all_300aa.RdRp_I"$file".refseq_protein.virus.fas.RdRp.edited.tophit.filt.fas.60 >> RdRp_20230215.fas;
		rm *id;

	done;



######### The results of the tenth iteration expansion is used ########

cat all_300aa.RdRp_I10.refseq_protein.virus.fas | grep ">" | sed 's/>//' | sort -u > all_300aa.RdRp_I10.refseq_protein.virus.id;
grep -v -F -f RdRp_20230215.id all_300aa.RdRp_I10.refseq_protein.virus.id > all_300aa.RdRp_I10.refseq_protein.virus.new.id;
seqtk subseq all_300aaRdRp_I10.refseq_protein.virus.fas all_300aa.RdRp_I10.refseq_protein.virus.new.id > all_300aa.RdRp_I10.refseq_protein.virus.new.fas;
grep -v -F -f RdRp_20230215.id all_300aa.RdRp_I10.refseq_protein.virus.fas.RdRp.edited.tophit > all_300aa.RdRp_I10.refseq_protein.virus.fas.RdRp.edited.tophit.filt;
rm all_300aa.RdRp_I10.refseq_protein.virus.id;

mkdir classify;
cp ./db/* ./classify;

for clstr in `ls ./db`;  ### a path of all rdrp clstr
	do
		cat all_300aa.RdRp_I10.refseq_protein.virus.fas.RdRp.edited.tophit.filt | awk '$2~/'${clstr}'/' | cut -f1 | sort -u > "$clstr".id;
		seqtk subseq all_300aa.RdRp_I10.refseq_protein.virus.fas "$clstr".id | sed "s/>/>NEW-"$clstr"_/" >> ./classify/"$clstr";
		rm "$clstr".id;

		hmmscan -E 10 --domE 10 --cpu 2 --noali --acc --notextw --domtblout "$file".hmmscan /home/houxin/data/metavirome/find_virus/contig_search/diff_method/cluster/clstr_expansion/300aa/hmm_db/"$file".hmm "$file".fas;

		cat "$file".hmmscan | grep -v "#" | awk -F " " 'BEGIN{OFS="\t"} {print $4,$6,$2,$1,$3,$7,$8,$9,$12,$13,$14,$15,$16,$17,$18,$19,$20,$21,$22}' | awk '{print $0"\t"($14-$13)/$5}' > "$file".hmmscan.edited;
		cat "$file".hmmscan.edited | awk -F "\t" '$7>=40 && $20>=0.4' > "$file".hmmscan.edited.filt;
		cat "$file".hmmscan.edited.filt | awk '!seen[$1]++' > "$file".hmmscan.edited.filt.tophit;
		cat "$file".hmmscan.edited.filt.tophit | cut -f1 | sort -u > "$file".hmmscan.edited.filt.tophit.id;
		seqtk subseq "$file".fas "$file".hmmscan.edited.filt.tophit.id > "$file".hmm.fas;
		rm "$file".hmmscan.edited.filt.tophit.id;
	done;
		