# site_subtype = [
#     ("brain", "GBM"), ("brain", "LGG"), 
#     ("gastrointestinal", "COAD"), ("gastrointestinal", "READ"), ("gastrointestinal", "ESCA"), ("gastrointestinal", "STAD"), 
#     ("gynecologic", "CESC"), ("gynecologic", "OV"), ("gynecologic", "UCEC"), ("gynecologic", "UCS"),
#     ("hematopoietic", "DLBC"), ("hematopoietic", "THYM"),
#     ("melanocytic", "SKCM"), ("melanocytic", "UVM"), 
#     ("pulmonary", "LUAD"), ("pulmonary", "LUSC"), ("pulmonary", "MESO"), 
#     ("urinary", "BLCA"), ("urinary", "KICH"), ("urinary", "KIRC"), ("urinary", "KIRP"), 
#     ("prostate", "PRAD"), ("prostate", "TGCT"), 
#     ("endocrine", "ACC"), ("endocrine", "PCPG"), ("endocrine", "THCA"), 
#     ("liver", "CHOL"), ("liver", "LIHC"), ("liver", "PAAD"), 
# ]

# SITE2LABEL = {site: idx for idx, site in enumerate(sorted(set(site for site, _ in site_subtype)))}
# SUBTYPE2LABEL = {subtype: idx for idx, subtype in enumerate(sorted(set(subtype for _, subtype in site_subtype)))}


# print(SITE2LABEL)
# print(SUBTYPE2LABEL)


import os, sys, json
sys.path.append("your/path")

data_path = f"data/TCGA_thumbnail"
sites = os.listdir(data_path)
for site in sites:
    site_path = os.path.join(data_path, site)
    subtypes = os.listdir(site_path)
    for subtype in subtypes:
        subtype_path = os.path.join(site_path, subtype)
        print(site, subtype)