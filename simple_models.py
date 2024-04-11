from setup import *
from helpers import *
from collections import defaultdict
from sklearn.metrics import log_loss

# Running this file generates the csv data necessary to make figure X.2 from the paper (for any dataset).
# It iterates through all given source-target population zones given and provides one average loss from all 
# single-feature models for each population pair. Again, using laplace smoothing on top of a basic count-based 
# MLE estimate. For figure 3, choose a pre-defined random set of columns, for figure 4, use dataset_properties.ipynb
# to get the most important columns to split by (take all with importance within 25% of max importance)

for zone in source_target_zones:
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

    # --- COMPARE PROBABILITIES THRU CP TRANSPORT, DIAGNOSTICITY, ODDS DIAGNOSTICITY USING LOG LOSS ---
    y_test = []
    cp_predictions = []
    od_predictions = []
    d_predictions = []

    # h is set to 1 for the purposes of binary classification. this would need a rethink for broader categorical use.
    h = 1 

    for index, row in target.iterrows():
        for c in columns:
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

    cp_loss = log_loss(y_test, cp_predictions)
    od_loss = log_loss(y_test, od_predictions)

    output = [split_col, label_col, source_zone, target_zone, cp_loss, od_loss]
    append_list_as_row(simple_output_filename, output)

    cp_full_output = per_example_log_loss(y_pred=cp_predictions, y_test=y_test)
    od_full_output = per_example_log_loss(y_pred=od_predictions, y_test=y_test)

    print("Completed and saved split: " + str(split_col) + ", SOURCE: " + str(source_zone) + ", TARGET: " + str(target_zone))
