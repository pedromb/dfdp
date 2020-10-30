import os
import json
import argparse

from data import *
from models import *

from config import FF_BENCHMARK_IMAGES_FOLDER


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output", type=str, default='faceforensics_benchmark.json', help="File to output results")
    parser.add_argument("-if", "--input_folder", type=str, default='/data/models/stacking_1', help="Folder where all models used to predict are saved")
    parser.add_argument("-s", "--strategy", type=str, default='average', help="Stacking strategy - one of [majority, average]")

    args = parser.parse_args()

    OUTPUT = args.output
    INPUT_FOLDER = args.input_folder
    STRATEGY = args.strategy


    print("=========== PREDICTION SETTINGS ===========")
    print("Output Folder = {}".format(OUTPUT))
    print("Input Folder = {}".format(INPUT_FOLDER))
    print("Stacking Strategy = {}".format(STRATEGY))
    print()

    models_checkpoints = ["{}/{}".format(INPUT_FOLDER, i) for i in os.listdir(INPUT_FOLDER)]

    preds = None
    for index, mc in enumerate(models_checkpoints):
        face_margin = int(mc.split("/")[-1].split("histogram")[0].split("_")[-3]) / 100
        equalize_histogram = mc.split("/")[-1].split("histogram")[1].split("_")[1]
        equalize_histogram = False if equalize_histogram == "not" else True
        backbone_aux = mc.split("/")[-1].split("_")
        backbone = backbone_aux[0] if backbone_aux[0] != 'se' else '_'.join(backbone_aux[:3])
        data_transformation = DataTransformation(resize=(224, 224), equalize_img=equalize_histogram)
        face_detector = FaceDetection( 
            batch_size = 1,
            thresholds = [
                (0.99, [0.9, 0.9, 0.9]),
                (0.98, [0.8, 0.8, 0.9]),
                (0.95, [0.6, 0.7, 0.7]),
                (0.9, [0.6, 0.7, 0.7]),
                (0.9, [0.5, 0.6, 0.6]),
                (0.7, [0.5, 0.6, 0.6]),
                (0.3, [0.3, 0.3, 0.3]),
                (0.3, [0.1, 0.2, 0.2])
            ],
            verbose = False,
            face_margin = face_margin,
            return_scores = False
        )
        benchmark_set = FFDatasetBenchmark(folder = FF_BENCHMARK_IMAGES_FOLDER, face_detector = face_detector, data_transformation = data_transformation)
        benchmark_dataloader = data.DataLoader(benchmark_set, batch_size=1, shuffle=False)
        model = CNNModel(backbone=backbone, pretrained_backbone=False)
        model_wrapper = ModelWrapper(
            model = model,
            load_model = mc
        )
        new_preds = model_wrapper.predict_benchmark(benchmark_dataloader)
        new_preds['predictions_{}'.format(index+1)] = new_preds["predictions"]
        new_preds.drop("predictions", axis=1, inplace=True)
        if preds is None:
            preds = new_preds
        else:
            preds = preds.merge(new_preds)

    preds_columns = [i for i in preds.columns if 'predictions' in i]
    if STRATEGY == 'majority':
        for i in preds_columns:
            preds[i] = preds[i].apply(lambda x: 1 if x >= 0.5 else 0)
        preds['predictions'] = preds[preds_columns].mode(axis=1)
    elif STRATEGY == 'average':
        preds['predictions'] = preds[preds_columns].mean(axis=1)
        preds['predictions'] = preds['predictions'].apply(lambda x: 1 if x >= 0.5 else 0)
        
    preds = preds[['filename', 'predictions']]
    preds['predictions'] = preds['predictions'].apply(lambda x: 'fake' if x == 1 else 'real')
    preds = preds.sort_values('filename')
    preds = preds.set_index('filename').to_dict()['predictions']

    with open(OUTPUT, 'w') as json_file:
        json.dump(preds, json_file)