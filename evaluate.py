
import torch.nn as nn
import argparse
import os


from data import *
from models import *
from config import PROCESSED_DATA_FOLDER, RESULTS_FILE

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--name", type=str, default='stacking_1', help="How to name the model on the results csv")
    parser.add_argument("-if", "--input_folder", type=str, default='/data/models/stacking_1', help="Folder where all models to be stacked are saved")
    parser.add_argument("-s", "--strategy", type=str, default='average', help="Stacking strategy - one of [majority, average]")

    args = parser.parse_args()

    NAME = args.name
    INPUT_FOLDER = args.input_folder
    STRATEGY = args.strategy
    

    print("\n=========== EVALUATION SETTINGS ===========")
    print("Input Folder = {}".format(INPUT_FOLDER))
    print("Stacking Strategy = {}".format(STRATEGY))
    print()

    models_checkpoints = ["{}/{}".format(INPUT_FOLDER, i) for i in os.listdir(INPUT_FOLDER)]

    preds = None
    for index, mc in enumerate(models_checkpoints):
        my_id = mc.split("/")[-1].split("histogram")[0].split("_")[1]
        face_margin = mc.split("/")[-1].split("histogram")[0].split("_")[-3]
        equalize_histogram = mc.split("/")[-1].split("histogram")[1].split("_")[1]
        equalize_histogram = False if equalize_histogram == "not" else True
        backbone_aux = mc.split("/")[-1].split("_")
        backbone = backbone_aux[0] if backbone_aux[0] != 'se' else '_'.join(backbone_aux[:3])
        data_transformation = DataTransformation(resize=(224, 224), equalize_img=equalize_histogram)

        test_file = PROCESSED_DATA_FOLDER[int(face_margin)] + 'test_balanced.h5py'
        test_config = {
            "h5py_file": test_file,
            "split": "test",
            "data_transformation": data_transformation
        }
        testset = FFDataset(**test_config)
        test = data.DataLoader(testset, batch_size=32, shuffle=False, num_workers=8)

        model = CNNModel(backbone=backbone, pretrained_backbone=False)
        model_wrapper = ModelWrapper(
            model = model,
            load_model = mc
        )
        new_preds = model_wrapper.predict(test)
        new_preds['predictions_{}'.format(index+1)] = new_preds["predictions"]
        new_preds.drop("predictions", axis=1, inplace=True)
        if preds is None:
            preds = new_preds
        else:
            preds = preds.merge(new_preds)


    results = {
        'model': [],
        'df_precision': [],
        'f2f_precision': [],
        'fs_precision': [],
        'nt_precision': [],
        'real_precision': [],
        'fake_precision': [],
        'accuracy': []
    }

    results['model'].append(NAME)

    mms = ['df', 'f2f', 'fs', 'nt', 'real']

    preds_columns = [i for i in preds.columns if 'predictions' in i]
    if STRATEGY == 'majority':
        for i in preds_columns:
            preds[i] = preds[i].apply(lambda x: 1 if x >= 0.5 else 0)
        preds['predictions'] = preds[preds_columns].mode(axis=1)
    elif STRATEGY == 'average':
        preds['predictions'] = preds[preds_columns].mean(axis=1)
        preds['predictions'] = preds['predictions'].apply(lambda x: 1 if x >= 0.5 else 0)
        



    image_precisions = preds.groupby("manipulation_method").apply(lambda x: (x.predictions == x.labels).mean() ).reset_index()
    image_precisions.columns = ['mm', 'precision']

    results["accuracy"].append((preds.predictions == preds.labels).mean())

    for i in mms:
        img_precision = image_precisions.loc[image_precisions.mm == i].precision.values[0]
        results['{}_precision'.format(i)].append(img_precision)
    fake_preds = preds.loc[np.isin(preds.manipulation_method, ['df', 'f2f', 'fs', 'nt'])]

    results['fake_precision'] = (fake_preds.predictions == fake_preds.labels).mean()
    results = pd.DataFrame(results)

    if os.path.exists(RESULTS_FILE):
        all_results = pd.read_csv(RESULTS_FILE)
        all_results = pd.concat([all_results, results])
        all_results.to_csv(RESULTS_FILE, index=False)
    else:
        results.to_csv(RESULTS_FILE, index=False)
    



