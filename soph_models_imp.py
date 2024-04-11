from setup import *
from helpers import *
from collections import defaultdict
import numpy as np
import xgboost as xgb
import random
import cupy as cp
from sklearn.metrics import log_loss

NUM_ITERATIONS = 20

combined_ratio_results = cp.zeros(NUM_ITERATIONS)
combined_median_ratio_results = cp.zeros(NUM_ITERATIONS)

combined_source_target_def_difference = cp.zeros(NUM_ITERATIONS)
combined_source_target_def_ratio = cp.zeros(NUM_ITERATIONS)
# combined_raw_odcp_ratio = cp.zeros(100)

# NUM_FEATURES = 10
LAMBDA = 1
ALPHA = 0

xgb_params = dict()
xgb_params["device"] = "cuda"
xgb_params["tree_method"] = "hist"
xgb_params["reg_alpha"] = ALPHA
xgb_params["reg_lambda"] = LAMBDA

# num_feats = iterations_num_feats[0]
num_feats = 10

# because we always use the same number of features, we should always use the SAME 20 sets of features for consistency and 1:1 comparison
# otherwise, the randomness of feature selection for the models may mess things up
imp_cols_meta_list = []
df0 = df_fetch_and_cleanup()
possible_features = list(df0.columns)
possible_features.remove(label_col)

# imp_features = ['Size>2', 'Size>1', 'Winter', 'after2000', 'cause_L', 'Surface', 'Wildfire', 'Size>3', 'Spring', 'after2010', 
#                 'Crown', 'Ground']

for i in range(NUM_ITERATIONS):
    imp_cols_meta_list.append(random.sample(possible_features, num_feats))
    # imp_cols_meta_list.append(['Winter', 'Size>2', 'Size>1', 'after2010', 'Size>5', 'Summer', 'Prescribed Burn', 'cause_H', 'cause_L', '2'])

for zone in source_target_zones:
    cp_loss_list = cp.array([])
    od_loss_list = cp.array([])
    def_prob_diff_list = cp.array([])
    def_prob_ratio_list = cp.array([])

    split_col = zone[0]
    source_zone = zone[1]
    target_zone = zone[2]
    
    for iteration in range(NUM_ITERATIONS):
        # --- FETCH DATASET AND SET UP BASIC VARIABLES ---
        df = df_fetch_and_cleanup()
        possible_features = list(df.columns)
        possible_features.remove(split_col)
        possible_features.remove(label_col)

        all_cols = list(df.columns)
        # select features to be used for iteration
        columns = imp_cols_meta_list[iteration].copy()
        print("COLUMNS PRE-MOD:")
        print(columns)
        while split_col in columns:
            columns.remove(split_col)
            # columns.append(random.choice(all_cols))

        for col in all_cols:
            if (col not in columns) and (col != label_col) and (col != split_col):
                df.drop(col, axis=1, inplace=True)

        source = df.loc[df[split_col] == source_zone]
        target = df.loc[df[split_col] == target_zone]

        y_source = cp.array(source[label_col].to_numpy())
        X_source = cp.array(source.drop(columns=[label_col, split_col]).to_numpy())
        y_target = cp.array(target[label_col].to_numpy())
        X_target = cp.array(target.drop(columns=[label_col, split_col]).to_numpy())

        all_label_vals = list(set(list(df[label_col].values))) # CRITICAL THAT THIS IS IN ALPHABETICAL ORDER FOR XGB DEFAULT PROB

        source_counter = defaultdict(lambda: 0) # Count of h ^ v; keys (h, [tuple of many pairs of (column_name, column value)])
        target_counter = defaultdict(lambda: 0) # Count of h ^ v; keys (h, [tuple of many pairs of (column_name, column value)])
        
        defaults = cp.array([target[col].mode()[0] for col in columns])
        print(num_feats, "FEATURES -> ROUND: ", iteration, " SPLIT COL: ", split_col, " SOURCE: ", source_zone, " TARGET: ", target_zone)
        print("COLUMNS:", columns)
        print("DEFAULTS:", defaults)

        # --- COMPUTE DEFAULT PROBS USING XGB ITSELF ---
        source_default_probs = defaultdict(lambda: 0)
        target_default_probs = defaultdict(lambda: 0)

        # train classifiers on the full source and target datasets
        xgb_source = xgb.train(xgb_params, xgb.QuantileDMatrix(X_source.get(), y_source.get()))
        xgb_target = xgb.train(xgb_params, xgb.QuantileDMatrix(X_target.get(), y_target.get()))

        # set up a one-row X matrix with the default values for all features
        X_def = [defaults]
        X_def = cp.array(X_def, dtype=int)

        source_default_probs[0] = 1 - xgb_source.predict(xgb.QuantileDMatrix(X_def.get()))[0]
        target_default_probs[0] = 1 - xgb_target.predict(xgb.QuantileDMatrix(X_def.get()))[0]
        source_default_probs[1] = xgb_source.predict(xgb.QuantileDMatrix(X_def.get()))[0]
        target_default_probs[1] = xgb_target.predict(xgb.QuantileDMatrix(X_def.get()))[0]

        print("SOURCE DEF PROB:", source_default_probs[1])
        print("TARGET DEF PROB:", target_default_probs[1])
        def_prob_diff_list = cp.append(def_prob_diff_list, cp.abs(source_default_probs[1] - target_default_probs[1]))

        if source_default_probs[1] <= target_default_probs[1]:
            def_prob_ratio_list = cp.append(def_prob_ratio_list, (source_default_probs[1]/target_default_probs[1]))
        else:
            def_prob_ratio_list = cp.append(def_prob_ratio_list, (target_default_probs[1]/source_default_probs[1]))

        # --- XGB MODEL ---

        xgb_cl = xgb.train(xgb_params, xgb.QuantileDMatrix(X_source.get(), y_source.get()))
        xgb_prediction = cp.array(xgb_cl.predict(xgb.QuantileDMatrix(X_target.get())))
        xgb_od_prediction = apply_od(xgb_prediction, source_default_probs, target_default_probs, all_label_vals)

        # --- COMPARE PROBABILITIES TO EXPECTED VALUES WITH LOG LOSS ---

        cp_log_loss = log_loss(y_true=y_target.get(), y_pred=xgb_prediction.get())
        od_log_loss = log_loss(y_true=y_target.get(), y_pred=xgb_od_prediction.get())
        cp_loss_list = cp.append(cp_loss_list, cp_log_loss)
        od_loss_list = cp.append(od_loss_list, od_log_loss)

        cp_entire_loss = per_example_log_loss(y_pred=xgb_prediction.get(), y_test=y_target.get())
        od_entire_loss = per_example_log_loss(y_pred=xgb_od_prediction.get(), y_test=y_target.get())

    results_mtx = cp.column_stack((cp_loss_list, od_loss_list))
    ratio = results_mtx[:,1] / results_mtx[:,0]
    combined_ratio_results = cp.column_stack((combined_ratio_results, ratio))
    append_2d_as_rows(soph_xgb_filename + "_split-on_" + split_col.replace(">","").replace("*","") + ".csv", results_mtx)
    append_entries_as_rows(xgb_ratio_filename + "_split-on_" + split_col.replace(">","").replace("*","") + ".csv", ratio)

    print("************SAVED RESULTS FOR SPLIT: ", split_col)

append_2d_as_rows(soph_xgb_filename + "_combined_ratios_" + ".csv", combined_ratio_results)