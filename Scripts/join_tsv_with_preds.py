import pandas as pd

def join_tsv_with_preds(tsv_path, predictions_path, output_dir):
    """
        inputs: 
            - original tsv filepath
            - path to predictions 
            - output_dir where output will be saved
        purpose: 
            - connects original tsv input data to predictions
            - saves connected information to new tsv in output folder
        returns: nothing
    """
    
    # load in original tsv data
    df = pd.read_csv(tsv_path, sep="\t")
    
    predictions = []
    probabilities_Y = []
    probabilities_N = []
    
    # load in predictions
    with open(predictions_path, 'r') as fi:
        line = fi.readline()
        
        while line:
            pred_dict = eval(line)
            
            predictions.append(pred_dict['label'])            
            probabilities_N.append(pred_dict['probs'][0])
            probabilities_Y.append(pred_dict['probs'][1])
            
            line = fi.readline()
            
    all_data_output_fpath = output_dir + "/" + tsv_path.split("/")[-1].split(".")[0] + "_with_all_predictions.tsv"
    positive_data_output_fpath = output_dir + "/" + tsv_path.split("/")[-1].split(".")[0] + "_with_positive_predictions.tsv"

    out_df = df.assign(PREDICTION_PROBABILITY_N=probabilities_N,
             PREDICTION_PROBABILITY_Y=probabilities_Y,
             PREDICTION_LABEL=predictions)
    
#     all_data_output_fpath = "/".join(output_fpath.split("/")[:-1]) + "all_outputs_" + output_fpath.split("/")[-1]
#     pos_only_output_fpath = "/".join(output_fpath.split("/")[:-1]) + "positive_only_" + output_fpath.split("/")[-1]
    
    all_data_df = out_df.sort_values(by="PREDICTION_PROBABILITY_Y", ascending=False)
    
    all_data_df.to_csv(all_data_output_fpath, sep="\t", index=False)
    
    pos_data_df = all_data_df[all_data_df["PREDICTION_LABEL"] == "Y"]
    
    pos_data_df.to_csv(positive_data_output_fpath, sep="\t", index=False)
