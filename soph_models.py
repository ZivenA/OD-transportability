from setup import *
from helpers import *
from collections import defaultdict
import xgboost as xgb
import random
import cupy as cp
from sklearn.metrics import log_loss

# Running this file generates the csv data necessary to make figures X.3 and X.4 from the paper (for any dataset).
# It runs several times with different numbers of features (as directed in setup.py). During each of these runs, 
# it iterates through each setup.py-defined source-target population pair 20 times, training an XGB model with
# num_feats features which are dynamically randomly sampled from the available features in the dataset. The ratio between
# OD log loss and CP log loss is the ultimate saved value, and is split based on number of features. Raw losses also saved
# for reference (though these are not explicitly presented in the paper)

NUM_ITERATIONS = 20
LAMBDA = 1
ALPHA = 0

xgb_params = dict()
xgb_params["device"] = "cuda"
xgb_params["tree_method"] = "hist"
xgb_params["reg_alpha"] = ALPHA
xgb_params["reg_lambda"] = LAMBDA

for num_feats in iterations_num_feats:
    cp_loss_list = cp.array([])
    od_loss_list = cp.array([])
    median_cp_loss_list = cp.array([])
    median_od_loss_list = cp.array([])
    entire_cp_loss_list = cp.array([])
    entire_od_loss_list = cp.array([])
    for iteration in range(NUM_ITERATIONS):
        for zone in source_target_zones:
            split_col = zone[0]
            source_zone = zone[1]
            target_zone = zone[2]

            # --- FETCH DATASET AND SET UP BASIC VARIABLES ---

            df = df_fetch_and_cleanup()
            possible_features = list(df.columns)
            possible_features.remove(split_col)
            possible_features.remove(label_col)

            all_cols = list(df.columns)

            # select features to be used for iteration
            columns = random.sample(possible_features, num_feats)

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
            
            defaults = cp.array([target[col].mode()[0] for col in columns])
            print(num_feats, "FEATURES -> ROUND: ", iteration, " SPLIT COL: ", split_col, " SOURCE: ", source_zone, " TARGET: ", target_zone)
            print("COLUMNS:", columns)

            # --- COMPUTE DEFAULT PROBS USING XGB ITSELF ---
            source_default_probs = defaultdict(lambda: 0)
            target_default_probs = defaultdict(lambda: 0)

            xgb_source = xgb.train(xgb_params, xgb.QuantileDMatrix(X_source.get(), y_source.get()))
            xgb_target = xgb.train(xgb_params, xgb.QuantileDMatrix(X_target.get(), y_target.get()))

            # set up a one-row X matrix with the default values for all features
            X_def = [defaults]
            X_def = cp.array(X_def, dtype=int)

            # use source and target classifiers to get the respective probs of the default case in both source and target

            source_default_probs[0] = 1 - xgb_source.predict(xgb.QuantileDMatrix(X_def.get()))[0]
            target_default_probs[0] = 1 - xgb_target.predict(xgb.QuantileDMatrix(X_def.get()))[0]
            source_default_probs[1] = xgb_source.predict(xgb.QuantileDMatrix(X_def.get()))[0]
            target_default_probs[1] = xgb_target.predict(xgb.QuantileDMatrix(X_def.get()))[0]

            # --- XGB MODEL ---

            xgb_cl = xgb.train(xgb_params, xgb.QuantileDMatrix(X_source.get(), y_source.get()))
            xgb_prediction = cp.array(xgb_cl.predict(xgb.QuantileDMatrix(X_target.get())))
            xgb_od_prediction = apply_od(xgb_prediction, source_default_probs, target_default_probs, all_label_vals)

            # --- COMPARE PROBABILITIES TO EXPECTED VALUES WITH LOG LOSS ---

            try:
                cp_log_loss = log_loss(y_true=y_target.get(), y_pred=xgb_prediction.get())
                od_log_loss = log_loss(y_true=y_target.get(), y_pred=xgb_od_prediction.get())
                cp_loss_list = cp.append(cp_loss_list, cp_log_loss)
                od_loss_list = cp.append(od_loss_list, od_log_loss)
            except:
                print("EXCEPTION IN LOG LOSS CALCULATION")
                continue

    
    results_mtx = cp.column_stack((cp_loss_list, od_loss_list))
    ratio = results_mtx[:,1] / results_mtx[:,0]
    append_2d_as_rows(soph_xgb_filename + str(num_feats) + ".csv", results_mtx)
    append_entries_as_rows(xgb_ratio_filename + str(num_feats) + ".csv", ratio)
    print("************SAVED RESULTS FOR NUM FEATS: ", num_feats)