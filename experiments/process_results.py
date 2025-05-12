# import re
# from collections import defaultdict

# def build_latex_table(label2class: dict, 
#                       model_results: dict,  # e.g. {"ABMIL": {...}, "GCN": {...}, ...}
#                       class2count: dict):
#     # 排版模板，双列，分组中英名 + 下属类名
#     latex_template = [
#         (r"\textbf{Pulmonary}", ["LUAD", "LUSC", "MESO"]),
#         (r"\textbf{Liver/PB}", ["CHOL", "LIHC", "PAAD"]),
#         (r"\textbf{Urinary}", ["BLCA", "KIRC", "KICH", "KIRP"]),
#         (r"\textbf{Gynecologic}", ["UCEC", "CESC", "UCS", "OV"]),
#         (r"\textbf{Melanotic}", ["UVM", "SKCM"]),
#         (r"\textbf{Prostate/Testis}", ["TGCT", "PRAD"]),
#         (r"\textbf{Brain}", ["GBM", "LGG"]),
#         (r"\textbf{Hematopoiesis}", ["DLBC", "THYM"]),
#         (r"\textbf{Gastrointestinal}", ["COAD", "ESCA", "READ", "STAD"]),
#         (r"\textbf{Endocrine}", ["ACC", "PCPG", "THCA"]),
#     ]

#     # 每行的最大模型数决定列数
#     model_names = list(model_results.keys())

#     def format_row(cancer_type, class_name):
#         label_idx = label2class.get(class_name, -1)
#         count = class2count.get(label_idx, '-')
#         if count == '-':
#             abmil_val = "00.00"
#             others = ["00.00"] * len(model_names)
#         else:
#             others = []
#             for model in model_names:
#                 acc = model_results.get(model, {}).get(label_idx, 0.0)
#                 acc_percent = round(acc * 100, 2)
#                 others.append(f"{acc_percent:.2f}")
#         return f"{class_name:<26} & {count:<5} & " + " & ".join(others) + r"  \\"

#     header = (
#         r"\toprule" + "\n" +
#         r"\hline" + "\n" +
#         r"\multirow{2}{*}{WSI Type} & \multirow{2}{*}{Slide Num} & \multicolumn{" +
#         str(len(model_names)) + r"}{c}{mMV@5} " +
#         r"& \multirow{2}{*}{WSI Type} & \multirow{2}{*}{Slide Num} & \multicolumn{" +
#         str(len(model_names)) + r"}{c}{mMV@5}\\ " + "\n" +
#         r"\cline{3-" + str(2 + len(model_names)) + r"}" + 
#         r"\cline{" + str(5 + len(model_names)) + "-" + str(4 + 2 * len(model_names)) + r"}" + "\n" +
#         r" &       & " + " & ".join(model_names) + 
#         r" &                               &       & " + " & ".join(model_names) + r"\\ " + "\n" +
#         r"\hline"
#     )

#     lines = [header]
#     # 每行两个大类组
#     for i in range(0, len(latex_template), 2):
#         left_group = latex_template[i]
#         right_group = latex_template[i + 1] if i + 1 < len(latex_template) else None

#         # summary line for both groups
#         left_count = sum(class2count.get(label2class.get(cls), 0) for cls in left_group[1])
#         right_count = sum(class2count.get(label2class.get(cls), 0) for cls in (right_group[1] if right_group else []))

#         left_summary = f"{left_group[0]:<26} & {left_count:<5} & " + " & ".join(["00.00"] * len(model_names))
#         right_summary = (
#             f" &  {right_group[0]:<26} & {right_count:<5} & " + " & ".join(["00.00"] * len(model_names))
#             if right_group else " &  - & - & " + " & ".join(["-"] * len(model_names))
#         )
#         lines.append(left_summary + right_summary + r"  \\")

#         # each class
#         for j in range(max(len(left_group[1]), len(right_group[1]) if right_group else 0)):
#             left_cls = left_group[1][j] if j < len(left_group[1]) else None
#             right_cls = right_group[1][j] if right_group and j < len(right_group[1]) else None

#             left_line = format_row(left_group[0], left_cls) if left_cls else " &  & " + " & ".join([""] * len(model_names))
#             right_line = " & " + format_row(right_group[0], right_cls) if right_cls else " &  & " + " & ".join([""] * len(model_names))
#             lines.append(left_line + right_line)

#         lines.append(r"\hline")

#     lines.append(r"\bottomrule")
#     return "\n".join(lines)

# def parse_accuracy_log_to_dict(log: str) -> dict:
#     """
#     从验证日志字符串中提取 class_id 和准确率，返回 {class_id: acc} 的字典
#     """
#     pattern = r"Class (\d+) - Val Acc: ([0-9.]+)"
#     matches = re.findall(pattern, log)
#     return {int(cls_id): float(acc) for cls_id, acc in matches}


# import re

# def convert_latex_table(latex_str):
#     lines = latex_str.strip().split('\n')
#     result = []
#     current_site_left = ''
#     current_site_right = ''
#     left_entries = []
#     right_entries = []

#     def parse_data_block(data_lines, current_site):
#         entries = []
#         for line in data_lines:
#             if '&' not in line:
#                 continue
#             fields = [f.strip().strip('\\') for f in line.split('&')]
#             if len(fields) == 1:
#                 continue
#             if re.search(r'textbf', fields[0]):
#                 current_site = fields[0]
#             elif current_site:
#                 entries.append((current_site, fields))
#         return entries, current_site

#     data_lines = [line for line in lines if '&' in line]
#     paired_lines = [data_lines[i:i+2] for i in range(0, len(data_lines), 2)]

#     for pair in paired_lines:
#         # left block
#         left_entries_block, current_site_left = parse_data_block([pair[0]], current_site_left)
#         left_entries.extend(left_entries_block)
#         # right block
#         right_entries_block, current_site_right = parse_data_block([pair[1]], current_site_right)
#         right_entries.extend(right_entries_block)

#     # group by site
#     def group_by_site(entries):
#         grouped = {}
#         for site, fields in entries:
#             grouped.setdefault(site, []).append(fields)
#         return grouped

#     left_grouped = group_by_site(left_entries)
#     right_grouped = group_by_site(right_entries)

#     # output latex
#     def build_latex(grouped_left, grouped_right):
#         output = []
#         site_pairs = list(zip(grouped_left.items(), grouped_right.items()))
#         for (site_l, subtypes_l), (site_r, subtypes_r) in site_pairs:
#             output.append(f'\\multirow{{{len(subtypes_l)}}}{{*}}{{{site_l}}}')
#             line_l = subtypes_l[0]
#             line_r = subtypes_r[0]
#             output.append(f"& {line_l[0]} & {line_l[1]} & {line_l[2]} & {line_l[3]} & {line_l[4]} & {line_l[5]} & "
#                           f"{site_r} & {line_r[0]} & {line_r[1]} & {line_r[2]} & {line_r[3]} & {line_r[4]} & {line_r[5]} \\\\")
#             for l, r in zip(subtypes_l[1:], subtypes_r[1:]):
#                 output.append(f"& {l[0]} & {l[1]} & {l[2]} & {l[3]} & {l[4]} & {l[5]} & "
#                               f"& {r[0]} & {r[1]} & {r[2]} & {r[3]} & {r[4]} & {r[5]} \\\\")
#             output.append("\\hline")
#         return "\n".join(output)

#     return build_latex(left_grouped, right_grouped)


# if __name__ == "__main__":
#     # 类别映射
#     label2class = {'ACC': 0, 'BLCA': 1, 'CESC': 2, 'CHOL': 3, 'COAD': 4, 'DLBC': 5, 'ESCA': 6, 'GBM': 7, 'KICH': 8, 'KIRC': 9,
#                 'KIRP': 10, 'LGG': 11, 'LIHC': 12, 'LUAD': 13, 'LUSC': 14, 'MESO': 15, 'OV': 16, 'PAAD': 17, 'PCPG': 18,
#                 'PRAD': 19, 'READ': 20, 'SKCM': 21, 'STAD': 22, 'TGCT': 23, 'THCA': 24, 'THYM': 25, 'UCEC': 26, 'UCS': 27,
#                 'UVM': 28}

#     # 类别总数统计
#     class2count = {0: 323, 1: 926, 2: 604, 3: 110, 4: 1442, 5: 103, 6: 396, 7: 2052, 8: 326, 9: 2173, 10: 770,
#                 11: 1572, 12: 870, 13: 1604, 14: 1612, 15: 175, 16: 1481, 17: 466, 18: 385, 19: 1172, 20: 530,
#                 21: 950, 22: 1197, 23: 413, 24: 1157, 25: 318, 26: 1371, 27: 154, 28: 150}
    
#     mean_log = """
# === Per-Class Validation Accuracy ===
# Class 0 - Val Acc: 0.7461 (241/323)
# Class 1 - Val Acc: 0.6177 (572/926)
# Class 2 - Val Acc: 0.3030 (183/604)
# Class 3 - Val Acc: 0.4727 (52/110)
# Class 4 - Val Acc: 0.7295 (1052/1442)
# Class 5 - Val Acc: 0.2621 (27/103)
# Class 6 - Val Acc: 0.4066 (161/396)
# Class 7 - Val Acc: 0.8436 (1731/2052)
# Class 8 - Val Acc: 0.6718 (219/326)
# Class 9 - Val Acc: 0.9457 (2055/2173)
# Class 10 - Val Acc: 0.6000 (462/770)
# Class 11 - Val Acc: 0.8753 (1376/1572)
# Class 12 - Val Acc: 0.8126 (707/870)
# Class 13 - Val Acc: 0.7045 (1130/1604)
# Class 14 - Val Acc: 0.7177 (1157/1612)
# Class 15 - Val Acc: 0.3314 (58/175)
# Class 16 - Val Acc: 0.9021 (1336/1481)
# Class 17 - Val Acc: 0.7704 (359/466)
# Class 18 - Val Acc: 0.6623 (255/385)
# Class 19 - Val Acc: 0.9283 (1088/1172)
# Class 20 - Val Acc: 0.2453 (130/530)
# Class 21 - Val Acc: 0.7474 (710/950)
# Class 22 - Val Acc: 0.6241 (747/1197)
# Class 23 - Val Acc: 0.7554 (312/413)
# Class 24 - Val Acc: 0.9326 (1079/1157)
# Class 25 - Val Acc: 0.7516 (239/318)
# Class 26 - Val Acc: 0.7586 (1040/1371)
# Class 27 - Val Acc: 0.3247 (50/154)
# Class 28 - Val Acc: 0.7733 (116/150)
#     """
#     mean_results = parse_accuracy_log_to_dict(mean_log)

#     max_log = """
# === Per-Class Validation Accuracy ===
# Class 0 - Val Acc: 0.4520 (146/323)
# Class 1 - Val Acc: 0.4482 (415/926)
# Class 2 - Val Acc: 0.1788 (108/604)
# Class 3 - Val Acc: 0.1545 (17/110)
# Class 4 - Val Acc: 0.8148 (1175/1442)
# Class 5 - Val Acc: 0.1456 (15/103)
# Class 6 - Val Acc: 0.1288 (51/396)
# Class 7 - Val Acc: 0.7861 (1613/2052)
# Class 8 - Val Acc: 0.5460 (178/326)
# Class 9 - Val Acc: 0.8587 (1866/2173)
# Class 10 - Val Acc: 0.3481 (268/770)
# Class 11 - Val Acc: 0.7958 (1251/1572)
# Class 12 - Val Acc: 0.6552 (570/870)
# Class 13 - Val Acc: 0.5661 (908/1604)
# Class 14 - Val Acc: 0.5689 (917/1612)
# Class 15 - Val Acc: 0.0457 (8/175)
# Class 16 - Val Acc: 0.8839 (1309/1481)
# Class 17 - Val Acc: 0.4227 (197/466)
# Class 18 - Val Acc: 0.3766 (145/385)
# Class 19 - Val Acc: 0.8814 (1033/1172)
# Class 20 - Val Acc: 0.0415 (22/530)
# Class 21 - Val Acc: 0.7547 (717/950)
# Class 22 - Val Acc: 0.5480 (656/1197)
# Class 23 - Val Acc: 0.6126 (253/413)
# Class 24 - Val Acc: 0.8142 (942/1157)
# Class 25 - Val Acc: 0.6478 (206/318)
# Class 26 - Val Acc: 0.6732 (923/1371)
# Class 27 - Val Acc: 0.0779 (12/154)
# Class 28 - Val Acc: 0.6867 (103/150)
#     """

#     max_results = parse_accuracy_log_to_dict(max_log)

#     abmil_log = """
# === Per-Class Validation Accuracy ===
# Class 0 - Val Acc: 0.8576 (277/323)
# Class 1 - Val Acc: 0.8045 (745/926)
# Class 2 - Val Acc: 0.7467 (451/604)
# Class 3 - Val Acc: 0.7818 (86/110)
# Class 4 - Val Acc: 0.8675 (1251/1442)
# Class 5 - Val Acc: 0.7767 (80/103)
# Class 6 - Val Acc: 0.7904 (313/396)
# Class 7 - Val Acc: 0.9537 (1957/2052)
# Class 8 - Val Acc: 0.8436 (275/326)
# Class 9 - Val Acc: 0.9503 (2065/2173)
# Class 10 - Val Acc: 0.8468 (652/770)
# Class 11 - Val Acc: 0.9281 (1459/1572)
# Class 12 - Val Acc: 0.8207 (714/870)
# Class 13 - Val Acc: 0.8903 (1428/1604)
# Class 14 - Val Acc: 0.8325 (1342/1612)
# Class 15 - Val Acc: 0.7600 (133/175)
# Class 16 - Val Acc: 0.9264 (1372/1481)
# Class 17 - Val Acc: 0.8670 (404/466)
# Class 18 - Val Acc: 0.8597 (331/385)
# Class 19 - Val Acc: 0.9505 (1114/1172)
# Class 20 - Val Acc: 0.6849 (363/530)
# Class 21 - Val Acc: 0.8537 (811/950)
# Class 22 - Val Acc: 0.8780 (1051/1197)
# Class 23 - Val Acc: 0.9128 (377/413)
# Class 24 - Val Acc: 0.9481 (1097/1157)
# Class 25 - Val Acc: 0.9057 (288/318)
# Class 26 - Val Acc: 0.9052 (1241/1371)
# Class 27 - Val Acc: 0.6169 (95/154)
# Class 28 - Val Acc: 0.8800 (132/150)
#     """
#     abmil_results = parse_accuracy_log_to_dict(abmil_log)

#     gcn_log = """
# === Per-Class Validation Accuracy ===
# Class 0 - Val Acc: 0.8173 (264/323)
# Class 1 - Val Acc: 0.7549 (699/926)
# Class 2 - Val Acc: 0.6391 (386/604)
# Class 3 - Val Acc: 0.7273 (80/110)
# Class 4 - Val Acc: 0.7864 (1134/1442)
# Class 5 - Val Acc: 0.7379 (76/103)
# Class 6 - Val Acc: 0.7854 (311/396)
# Class 7 - Val Acc: 0.9571 (1964/2052)
# Class 8 - Val Acc: 0.8742 (285/326)
# Class 9 - Val Acc: 0.9646 (2096/2173)
# Class 10 - Val Acc: 0.7818 (602/770)
# Class 11 - Val Acc: 0.9447 (1485/1572)
# Class 12 - Val Acc: 0.9080 (790/870)
# Class 13 - Val Acc: 0.8310 (1333/1604)
# Class 14 - Val Acc: 0.8393 (1353/1612)
# Class 15 - Val Acc: 0.6400 (112/175)
# Class 16 - Val Acc: 0.9615 (1424/1481)
# Class 17 - Val Acc: 0.9056 (422/466)
# Class 18 - Val Acc: 0.8831 (340/385)
# Class 19 - Val Acc: 0.9625 (1128/1172)
# Class 20 - Val Acc: 0.6962 (369/530)
# Class 21 - Val Acc: 0.8842 (840/950)
# Class 22 - Val Acc: 0.8279 (991/1197)
# Class 23 - Val Acc: 0.9031 (373/413)
# Class 24 - Val Acc: 0.9672 (1119/1157)
# Class 25 - Val Acc: 0.8868 (282/318)
# Class 26 - Val Acc: 0.9271 (1271/1371)
# Class 27 - Val Acc: 0.6753 (104/154)
# Class 28 - Val Acc: 0.9067 (136/150)
#     """
#     gcn_results = parse_accuracy_log_to_dict(gcn_log)

#     gw_log = """
# === Per-Class Validation Accuracy ===
# Class 0 - Val Acc: 0.7090 (229/323)
# Class 1 - Val Acc: 0.5907 (547/926)
# Class 2 - Val Acc: 0.4056 (245/604)
# Class 3 - Val Acc: 0.4182 (46/110)
# Class 4 - Val Acc: 0.6567 (947/1442)
# Class 5 - Val Acc: 0.3107 (32/103)
# Class 6 - Val Acc: 0.3586 (142/396)
# Class 7 - Val Acc: 0.8197 (1682/2052)
# Class 8 - Val Acc: 0.6104 (199/326)
# Class 9 - Val Acc: 0.9342 (2030/2173)
# Class 10 - Val Acc: 0.5506 (424/770)
# Class 11 - Val Acc: 0.8804 (1384/1572)
# Class 12 - Val Acc: 0.8069 (702/870)
# Class 13 - Val Acc: 0.6895 (1106/1604)
# Class 14 - Val Acc: 0.7128 (1149/1612)
# Class 15 - Val Acc: 0.3143 (55/175)
# Class 16 - Val Acc: 0.8940 (1324/1481)
# Class 17 - Val Acc: 0.7489 (349/466)
# Class 18 - Val Acc: 0.6468 (249/385)
# Class 19 - Val Acc: 0.9121 (1069/1172)
# Class 20 - Val Acc: 0.3019 (160/530)
# Class 21 - Val Acc: 0.7042 (669/950)
# Class 22 - Val Acc: 0.6399 (766/1197)
# Class 23 - Val Acc: 0.7966 (329/413)
# Class 24 - Val Acc: 0.9196 (1064/1157)
# Class 25 - Val Acc: 0.7610 (242/318)
# Class 26 - Val Acc: 0.7345 (1007/1371)
# Class 27 - Val Acc: 0.3182 (49/154)
# Class 28 - Val Acc: 0.7800 (117/150)
#     """

#     gw_results = parse_accuracy_log_to_dict(gw_log)

#     # 添加多个模型（如 GCN、GraphWalk）
#     model_results = {
#         "Mean":mean_results,
#         "Max":max_results,
#         "ABMIL": abmil_results,
#         "GCN": gcn_results,
#         "GraphWalk":gw_results 
#         # "GraphWalk": {...}
#     }

#     latex_code = build_latex_table(label2class, model_results, class2count)
#     res = convert_latex_table(latex_code)
#     print(latex_code)





import re

def extract_val_accs(text):
    pattern = r'Class \d+ - Val Acc: ([\d.]+)'
    accs = re.findall(pattern, text)
    accs_formatted = ' & '.join(f'{float(a)*100:.2f}' for a in accs)
    return accs_formatted


if __name__ == "__main__":
    print("    \multirow{5}{*}{ResNet50}")
    resnet_mean_log = """
Class 0 - Val Acc: 0.9393 (3404/3624)
Class 1 - Val Acc: 0.7094 (1323/1865)
Class 2 - Val Acc: 0.8017 (2858/3565)
Class 3 - Val Acc: 0.6842 (2470/3610)
Class 4 - Val Acc: 0.5131 (216/421)
Class 5 - Val Acc: 0.5615 (812/1446)
Class 6 - Val Acc: 0.5845 (643/1100)
Class 7 - Val Acc: 0.8063 (1278/1585)
Class 8 - Val Acc: 0.6128 (2078/3391)
Class 9 - Val Acc: 0.7221 (3029/4195)

    """

    res = extract_val_accs(resnet_mean_log)
    print("        & mean pooling & " + res + " &   68.62 \\\\")

    resnet_max_log = """
Class 0 - Val Acc: 0.8805 (3191/3624)
Class 1 - Val Acc: 0.5705 (1064/1865)
Class 2 - Val Acc: 0.7930 (2827/3565)
Class 3 - Val Acc: 0.5623 (2030/3610)
Class 4 - Val Acc: 0.4822 (203/421)
Class 5 - Val Acc: 0.5235 (757/1446)
Class 6 - Val Acc: 0.5409 (595/1100)
Class 7 - Val Acc: 0.7413 (1175/1585)
Class 8 - Val Acc: 0.5547 (1881/3391)
Class 9 - Val Acc: 0.6350 (2664/4195)
    """

    res = extract_val_accs(resnet_max_log)
    print("        & max pooling  & " + res + " &   65.63 \\\\")

    resnet_abmil_log = """
Class 0 - Val Acc: 0.9627 (3489/3624)
Class 1 - Val Acc: 0.8745 (1631/1865)
Class 2 - Val Acc: 0.9072 (3234/3565)
Class 3 - Val Acc: 0.8211 (2964/3610)
Class 4 - Val Acc: 0.8242 (347/421)
Class 5 - Val Acc: 0.8423 (1218/1446)
Class 6 - Val Acc: 0.7845 (863/1100)
Class 7 - Val Acc: 0.8972 (1422/1585)
Class 8 - Val Acc: 0.8514 (2887/3391)
Class 9 - Val Acc: 0.8746 (3669/4195)
    """

    res = extract_val_accs(resnet_abmil_log)
    print("        & ABMIL        & " + res + " &   88.45 \\\\")

    resnet_gcn_log = """
Class 0 - Val Acc: 0.9570 (3468/3624)
Class 1 - Val Acc: 0.8536 (1592/1865)
Class 2 - Val Acc: 0.8785 (3132/3565)
Class 3 - Val Acc: 0.8515 (3074/3610)
Class 4 - Val Acc: 0.7530 (317/421)
Class 5 - Val Acc: 0.6936 (1003/1446)
Class 6 - Val Acc: 0.7836 (862/1100)
Class 7 - Val Acc: 0.9218 (1461/1585)
Class 8 - Val Acc: 0.7782 (2639/3391)
Class 9 - Val Acc: 0.8336 (3497/4195)
    """

    res = extract_val_accs(resnet_gcn_log)
    print("        & GCN          & " + res + " &   85.87 \\\\")

    resnet_gw_log = """
Class 0 - Val Acc: 0.9134 (3310/3624)
Class 1 - Val Acc: 0.6954 (1297/1865)
Class 2 - Val Acc: 0.8132 (2899/3565)
Class 3 - Val Acc: 0.6612 (2387/3610)
Class 4 - Val Acc: 0.4751 (200/421)
Class 5 - Val Acc: 0.4931 (713/1446)
Class 6 - Val Acc: 0.5182 (570/1100)
Class 7 - Val Acc: 0.7653 (1213/1585)
Class 8 - Val Acc: 0.5677 (1925/3391)
Class 9 - Val Acc: 0.7092 (2975/4195)
    """

    res = extract_val_accs(resnet_gw_log)
    print("        & Graph Walk   & " + res + " &   66.45 \\\\")
    print("    \\hline")






    print("    \multirow{5}{*}{ViT}")
    vit_mean_log = """
Class 0 - Val Acc: 0.9134 (3310/3624)
Class 1 - Val Acc: 0.6954 (1297/1865)
Class 2 - Val Acc: 0.8132 (2899/3565)
Class 3 - Val Acc: 0.6612 (2387/3610)
Class 4 - Val Acc: 0.4751 (200/421)
Class 5 - Val Acc: 0.4931 (713/1446)
Class 6 - Val Acc: 0.5182 (570/1100)
Class 7 - Val Acc: 0.7653 (1213/1585)
Class 8 - Val Acc: 0.5677 (1925/3391)
Class 9 - Val Acc: 0.7092 (2975/4195)
    """

    res = extract_val_accs(vit_mean_log)
    print("        & mean pooling & " + res + " &   65.23 \\\\")

    vit_max_log = """
Class 0 - Val Acc: 0.8739 (3167/3624)
Class 1 - Val Acc: 0.4869 (908/1865)
Class 2 - Val Acc: 0.7217 (2573/3565)
Class 3 - Val Acc: 0.5141 (1856/3610)
Class 4 - Val Acc: 0.1829 (77/421)
Class 5 - Val Acc: 0.3326 (481/1446)
Class 6 - Val Acc: 0.2782 (306/1100)
Class 7 - Val Acc: 0.6498 (1030/1585)
Class 8 - Val Acc: 0.4577 (1552/3391)
Class 9 - Val Acc: 0.5769 (2420/4195)

    """

    res = extract_val_accs(vit_max_log)
    print("        & max pooling  & " + res + " &   53.19 \\\\")

    vit_abmil_log = """
Class 0 - Val Acc: 0.9440 (3421/3624)
Class 1 - Val Acc: 0.7582 (1414/1865)
Class 2 - Val Acc: 0.8435 (3007/3565)
Class 3 - Val Acc: 0.8122 (2932/3610)
Class 4 - Val Acc: 0.6888 (290/421)
Class 5 - Val Acc: 0.7925 (1146/1446)
Class 6 - Val Acc: 0.7318 (805/1100)
Class 7 - Val Acc: 0.8744 (1386/1585)
Class 8 - Val Acc: 0.8169 (2770/3391)
Class 9 - Val Acc: 0.8076 (3388/4195)
    """

    res = extract_val_accs(vit_abmil_log)
    print("        & ABMIL        & " + res + " &   84.63 \\\\")

    vit_gcn_log = """
Class 0 - Val Acc: 0.9558 (3464/3624)
Class 1 - Val Acc: 0.8182 (1526/1865)
Class 2 - Val Acc: 0.8769 (3126/3565)
Class 3 - Val Acc: 0.7994 (2886/3610)
Class 4 - Val Acc: 0.7173 (302/421)
Class 5 - Val Acc: 0.7801 (1128/1446)
Class 6 - Val Acc: 0.7164 (788/1100)
Class 7 - Val Acc: 0.8940 (1417/1585)
Class 8 - Val Acc: 0.8018 (2719/3391)
Class 9 - Val Acc: 0.7933 (3328/4195)
    """

    res = extract_val_accs(vit_gcn_log)
    print("        & GCN          & " + res + " &   83.43 \\\\")

    vit_gw_log = """
Class 0 - Val Acc: 0.9241 (3349/3624)
Class 1 - Val Acc: 0.7582 (1414/1865)
Class 2 - Val Acc: 0.7338 (2616/3565)
Class 3 - Val Acc: 0.6507 (2349/3610)
Class 4 - Val Acc: 0.4774 (201/421)
Class 5 - Val Acc: 0.5194 (751/1446)
Class 6 - Val Acc: 0.5273 (580/1100)
Class 7 - Val Acc: 0.7754 (1229/1585)
Class 8 - Val Acc: 0.7166 (2430/3391)
Class 9 - Val Acc: 0.6546 (2746/4195)
    """

    res = extract_val_accs(vit_gw_log)
    print("        & Graph Walk   & " + res + " &   65.70 \\\\")
    print("    \\hline")




    print("    \multirow{5}{*}{DINOv2}")
    dino_mean_log = """
Class 0 - Val Acc: 0.9528 (3453/3624)
Class 1 - Val Acc: 0.6954 (1297/1865)
Class 2 - Val Acc: 0.8115 (2893/3565)
Class 3 - Val Acc: 0.7579 (2736/3610)
Class 4 - Val Acc: 0.5487 (231/421)
Class 5 - Val Acc: 0.5837 (844/1446)
Class 6 - Val Acc: 0.6173 (679/1100)
Class 7 - Val Acc: 0.7584 (1202/1585)
Class 8 - Val Acc: 0.7027 (2383/3391)
Class 9 - Val Acc: 0.7011 (2941/4195)
    """

    res = extract_val_accs(dino_mean_log)
    print("        & mean pooling & " + res + " &   71.59 \\\\")

    dino_max_log = """
Class 0 - Val Acc: 0.9346 (3387/3624)
Class 1 - Val Acc: 0.5737 (1070/1865)
Class 2 - Val Acc: 0.6729 (2399/3565)
Class 3 - Val Acc: 0.5332 (1925/3610)
Class 4 - Val Acc: 0.3919 (165/421)
Class 5 - Val Acc: 0.4281 (619/1446)
Class 6 - Val Acc: 0.6045 (665/1100)
Class 7 - Val Acc: 0.5363 (850/1585)
Class 8 - Val Acc: 0.7128 (2417/3391)
Class 9 - Val Acc: 0.5569 (2336/4195)
    """

    res = extract_val_accs(dino_max_log)
    print("        & max pooling  & " + res + " &   59.90 \\\\")

    dino_abmil_log = """
Class 0 - Val Acc: 0.9570 (3468/3624)
Class 1 - Val Acc: 0.8477 (1581/1865)
Class 2 - Val Acc: 0.8379 (2987/3565)
Class 3 - Val Acc: 0.8662 (3127/3610)
Class 4 - Val Acc: 0.8005 (337/421)
Class 5 - Val Acc: 0.8140 (1177/1446)
Class 6 - Val Acc: 0.7891 (868/1100)
Class 7 - Val Acc: 0.9091 (1441/1585)
Class 8 - Val Acc: 0.8726 (2959/3391)
Class 9 - Val Acc: 0.8613 (3613/4195)
    """

    res = extract_val_accs(dino_abmil_log)
    print("        & ABMIL        & " + res + " &   87.94 \\\\")

    dino_gcn_log = """
Class 0 - Val Acc: 0.9724 (3524/3624)
Class 1 - Val Acc: 0.8820 (1645/1865)
Class 2 - Val Acc: 0.8886 (3168/3565)
Class 3 - Val Acc: 0.8183 (2954/3610)
Class 4 - Val Acc: 0.7886 (332/421)
Class 5 - Val Acc: 0.6909 (999/1446)
Class 6 - Val Acc: 0.8582 (944/1100)
Class 7 - Val Acc: 0.8770 (1390/1585)
Class 8 - Val Acc: 0.8237 (2793/3391)
Class 9 - Val Acc: 0.7924 (3324/4195)
    """

    res = extract_val_accs(dino_gcn_log)
    print("        & GCN          & " + res + " &   86.44 \\\\")

    dino_gw_log = """
Class 0 - Val Acc: 0.9393 (3404/3624)
Class 1 - Val Acc: 0.6611 (1233/1865)
Class 2 - Val Acc: 0.7919 (2823/3565)
Class 3 - Val Acc: 0.7294 (2633/3610)
Class 4 - Val Acc: 0.4917 (207/421)
Class 5 - Val Acc: 0.6141 (888/1446)
Class 6 - Val Acc: 0.5882 (647/1100)
Class 7 - Val Acc: 0.7218 (1144/1585)
Class 8 - Val Acc: 0.7561 (2564/3391)
Class 9 - Val Acc: 0.6849 (2873/4195)

    """

    res = extract_val_accs(dino_gw_log)
    print("        & Graph Walk   & " + res + " &   70.40 \\\\")
    print("    \\hline")




    print("    \multirow{5}{*}{UNI}")
    uni_mean_log = """
Class 0 - Val Acc: 0.9341 (3385/3624)
Class 1 - Val Acc: 0.7984 (1489/1865)
Class 2 - Val Acc: 0.8454 (3014/3565)
Class 3 - Val Acc: 0.7427 (2681/3610)
Class 4 - Val Acc: 0.6057 (255/421)
Class 5 - Val Acc: 0.7241 (1047/1446)
Class 6 - Val Acc: 0.8000 (880/1100)
Class 7 - Val Acc: 0.8902 (1411/1585)
Class 8 - Val Acc: 0.8402 (2849/3391)
Class 9 - Val Acc: 0.7945 (3333/4195)

    """

    res = extract_val_accs(uni_mean_log)
    print("        & mean pooling & " + res + " &   79.81 \\\\")

    uni_max_log = """
Class 0 - Val Acc: 0.9393 (3404/3624)
Class 1 - Val Acc: 0.6874 (1282/1865)
Class 2 - Val Acc: 0.8370 (2984/3565)
Class 3 - Val Acc: 0.5776 (2085/3610)
Class 4 - Val Acc: 0.4751 (200/421)
Class 5 - Val Acc: 0.4578 (662/1446)
Class 6 - Val Acc: 0.6845 (753/1100)
Class 7 - Val Acc: 0.7760 (1230/1585)
Class 8 - Val Acc: 0.7694 (2609/3391)
Class 9 - Val Acc: 0.6765 (2838/4195)
    """

    res = extract_val_accs(uni_max_log)
    print("        & max pooling  & " + res + " &   69.01 \\\\")

    uni_abmil_log = """
Class 0 - Val Acc: 0.9765 (3539/3624)
Class 1 - Val Acc: 0.9292 (1733/1865)
Class 2 - Val Acc: 0.9450 (3369/3565)
Class 3 - Val Acc: 0.9122 (3293/3610)
Class 4 - Val Acc: 0.7886 (332/421)
Class 5 - Val Acc: 0.8893 (1286/1446)
Class 6 - Val Acc: 0.9155 (1007/1100)
Class 7 - Val Acc: 0.9344 (1481/1585)
Class 8 - Val Acc: 0.9248 (3136/3391)
Class 9 - Val Acc: 0.9085 (3811/4195)
    """

    res = extract_val_accs(uni_abmil_log)
    print("        & ABMIL        & " + res + " &   92.96 \\\\")

    uni_gcn_log = """
Class 0 - Val Acc: 0.9782 (3545/3624)
Class 1 - Val Acc: 0.9351 (1744/1865)
Class 2 - Val Acc: 0.9330 (3326/3565)
Class 3 - Val Acc: 0.9008 (3252/3610)
Class 4 - Val Acc: 0.8290 (349/421)
Class 5 - Val Acc: 0.9073 (1312/1446)
Class 6 - Val Acc: 0.8755 (963/1100)
Class 7 - Val Acc: 0.9476 (1502/1585)
Class 8 - Val Acc: 0.8918 (3024/3391)
Class 9 - Val Acc: 0.9285 (3895/4195)
    """

    res = extract_val_accs(uni_gcn_log)
    print("        & GCN          & " + res + " &   92.76 \\\\")

    uni_gw_log = """
Class 0 - Val Acc: 0.9274 (3361/3624)
Class 1 - Val Acc: 0.8166 (1523/1865)
Class 2 - Val Acc: 0.8516 (3036/3565)
Class 3 - Val Acc: 0.7407 (2674/3610)
Class 4 - Val Acc: 0.5772 (243/421)
Class 5 - Val Acc: 0.7144 (1033/1446)
Class 6 - Val Acc: 0.7700 (847/1100)
Class 7 - Val Acc: 0.8625 (1367/1585)
Class 8 - Val Acc: 0.8207 (2783/3391)
Class 9 - Val Acc: 0.7993 (3353/4195)
    """

    res = extract_val_accs(uni_gw_log)
    print("        & Graph Walk   & " + res + " &   79.11 \\\\")
    print("    \\hline")



