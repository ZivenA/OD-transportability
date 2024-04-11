from setup import *
from helpers import *
from collections import defaultdict
from sklearn.metrics import log_loss

# Running this file generates the csv data necessary to make figure X.1 from the paper (for any dataset).
# It only runs on the first source-target zone given, and it provides the loss on a feature-by-feature basis
# from training single-feature models on each of them and comparing CP and OD outputs. It uses laplace smoothing
# on top of a basic count-based MLE estimate.

zone = source_target_zones[0]
split_col = zone[0]
source_zone = zone[1]
target_zone = zone[2]

# --- FETCH DATASET AND SET UP BASIC VARIABLES ---

df = df_fetch_and_cleanup()
source = df.loc[df[split_col] == source_zone]
target = df.loc[df[split_col] == target_zone]

all_label_vals = list(set(list(df[label_col].values)))

columns = list(source.columns)
columns.remove(split_col)
columns.remove(label_col)

source_counter = defaultdict(lambda: 0) # Count of h ^ v; keys (h, column_name, column value)
target_counter = defaultdict(lambda: 0) # Count of h ^ v; keys (h, column_name, column value)

source_p = defaultdict(lambda: 0) # p(h | v); keys (h, column_name, column value)
target_p = defaultdict(lambda: 0) # p(h | v); keys (h, column_name, column value)

laplace_num = 1
laplace_denom = len(all_label_vals)

## --- COMPUTE ACTUAL PROBABILITIES IN SOURCE, TARGET POPS ---

print("Computing probabilities for all individual columns...")

for index, row in source.iterrows():
    h = row[label_col]
    for c in columns:
        key = (h, c, row[c]) 
        source_counter[key] += 1

for index, row in target.iterrows():
    h = row[label_col]
    for c in columns:
        key = (h, c, row[c]) 
        target_counter[key] += 1

for index, row in source.iterrows():
    h = row[label_col]
    for c in columns:
        key = (h, c, row[c])
        not_key = (1-h, c, row[c])
        source_p[key] = (source_counter[key] + laplace_num) / (source_counter[key] + source_counter[not_key] + laplace_denom)

for index, row in target.iterrows():
    h = row[label_col]
    for c in columns:
        key = (h, c, row[c])
        not_key = (1-h, c, row[c])
        target_p[key] = (target_counter[key] + laplace_num) / (target_counter[key] + target_counter[not_key] + laplace_denom)

print("Probability computation completed - beginning prediction & evaluation")

# --- COMPARE PROBABILITIES THRU CP TRANSPORT & ODDS DIAGNOSTICITY USING LOG LOSS ---

# h is set to 1 for the purposes of binary classification. this would need a rethink for broader categorical use.
h = 1 

cp_od_loss_per_feature = defaultdict(lambda: 0)

for c in columns:
    cp_predictions = []
    od_predictions = []
    y_test = []
    for index, row in target.iterrows():
        key = (h, c, row[c])
        default_key = (h, c, defaults[c])

        if source_p[default_key] == 0:
            # print("source p missing default")
            continue

        if target_p[default_key] == 0:
            # print("target p missing default")
            continue

        if source_p[key] == 0:
            # print("source p missing current p")
            continue

        cp_predictions.append(source_p[key])
        odds_prediction = (odds_from_p(source_p[key]) * odds_from_p(target_p[default_key])) / odds_from_p(source_p[default_key])
        od_predictions.append(p_from_odds(odds_prediction))
        y_test.append(row[label_col])
    
    if (len(y_test) > 0):
        cp_od_loss_per_feature[c] = [log_loss(y_test, cp_predictions), log_loss(y_test, od_predictions)]
    
    print("Column", c, "modelling & evaluation completed")

output_arr = []
for key, value in cp_od_loss_per_feature.items():
    output_arr.append([key, value[0], value[1]])

append_2d_as_rows(one_split_simple_filename, output_arr)
print("Saved single-split results to csv")