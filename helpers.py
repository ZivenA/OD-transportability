from csv import writer
import cupy as cp

# This file contains miscellaneous helpers for saving output data and handling probability and odds operations.
# It's imported in most files in the project.

def append_list_as_row(file_name, list_of_elem):
    # Open file in append mode
    with open(file_name, 'a+', newline='') as write_obj:
        # Create a writer object from csv module
        csv_writer = writer(write_obj)
        # Add contents of list as last row in the csv file
        csv_writer.writerow(list_of_elem)

def append_2d_as_rows(file_name, arr_of_elem):
    # Open file in append mode
    with open(file_name, 'a+', newline='') as write_obj:
        # Create a writer object from csv module
        csv_writer = writer(write_obj)
        # Iterate through rows of array, adding each to its own line in csv
        for row in arr_of_elem:
            csv_writer.writerow(row)

def append_entries_as_rows(file_name, arr_of_elem):
    # Open file in append mode
    with open(file_name, 'a+', newline='') as write_obj:
        # Create a writer object from csv module
        csv_writer = writer(write_obj)
        # Iterate through rows of array, adding each to its own line in csv
        for row in arr_of_elem:
            csv_writer.writerow([row])            

def odds_from_p(p):
    return (p / (1-p))

def p_from_odds(odds):
    return (odds / (1+odds))

def per_example_log_loss(y_pred, y_test):
    # Ensure the input arrays are numpy arrays
    y_pred = cp.array(y_pred)
    y_test = cp.array(y_test)

    # Clip predictions to avoid log of zero
    y_pred = cp.clip(y_pred, 1e-14, 1 - 1e-14)

    # Calculate per-example log loss
    log_loss = -1 * (y_test * cp.log(y_pred) + (1 - y_test) * cp.log(1 - y_pred))

    return log_loss

def apply_od(prediction, source_default_probs, target_default_probs, all_label_vals):
    od_prediction = []

    for entry in prediction:
        source_p_default = source_default_probs[1]
        target_p_default = target_default_probs[1]
        odds_prediction = p_from_odds((odds_from_p(entry) * odds_from_p(target_p_default)) / odds_from_p(source_p_default))
        od_prediction.append(odds_prediction)

    return cp.array(od_prediction)