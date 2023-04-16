# step1
python get_biosample_from_entrez.py -i data/00all_SRA_info.txt -o data/00all_SRA_biosample.txt --email heyongcast@163.com -idx 1

# step2
python get_biosample_from_update.py -i data/00all_SRA_info.txt -o data/00all_SRA_biosample.txt -idx 1

# step3
python get_biosample_from_manual.py -i data/00all_SRA_info.txt -o data/00all_SRA_biosample.txt -idx 1

# step4
python download_biosample_page.py -i data/00all_SRA_biosample.txt

# step5
python extract_attr_from_biosample_page.py

# step6
python parse_loc_lat_lon_from_attr.py -i data/00all_SRA_info.txt -idx 1

# step7
python standardization_lat_lon_info.py