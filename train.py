
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import argparse


from data import *
from models import *
from config import PROCESSED_DATA_FOLDER, MODELS_FOLDER


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, default='resnet34', help="The model to be trained")
    parser.add_argument("-i", "--model_id", type=int, default=1, help="The id for the model - to use when training several of the same model")
    parser.add_argument("-f", "--face_margin", type=int, default=0, help="Which face margin to use")
    parser.add_argument("-eq", "--equalize_histogram", type=bool, default=False, help="Whether or not to equalize the histogram")
    parser.add_argument("-da", "--data_augmentation", type=str, default='None', help="The level of augmentation to use")

    args = parser.parse_args()

    MODEL = args.model
    MODEL_ID = args.model_id
    FACE_MARGIN = args.face_margin
    EQUALIZE_HISTOGRAM = args.equalize_histogram
    DATA_AUGMENTATION_LEVEL = None if args.data_augmentation == 'None' else args.data_augmentation

    print("=========== TRAINING SETTINGS ===========")
    print("Model = {} - {}".format(MODEL, MODEL_ID))
    print("Face Margin = {}".format(FACE_MARGIN))
    print("Equalize Histogram = {}".format(EQUALIZE_HISTOGRAM))
    print("Data Augmentation Level = {}".format(DATA_AUGMENTATION_LEVEL))
    print()

    DATA_AUGMENTATION_LEVEL_STRING = 'None' if DATA_AUGMENTATION_LEVEL is None else DATA_AUGMENTATION_LEVEL
    EQUALIZE_HISTOGRAM_STRING = 'histogram_equalized' if EQUALIZE_HISTOGRAM else 'histogram_not_equalized'


    final_data_folder = PROCESSED_DATA_FOLDER[FACE_MARGIN]
    data_augmentation = DataAugmentation(level=DATA_AUGMENTATION_LEVEL)
    data_transformation = DataTransformation(resize=(224, 224), equalize_img=EQUALIZE_HISTOGRAM)

    TRAIN_CONFIG = {
        "h5py_file": "{}train.h5py".format(final_data_folder),
        "frame_to_mm_train": "{}frames_to_mm_train.npy".format(final_data_folder),
        "split": "train",
        "data_transformation": data_transformation,
        "data_augmentation": data_augmentation,
    }

    VAL_CONFIG = dict(TRAIN_CONFIG)
    VAL_CONFIG['h5py_file'] = "{}val_balanced.h5py".format(final_data_folder)
    VAL_CONFIG['split'] = 'val'

    train_set = FFDataset(**TRAIN_CONFIG)
    val_set = FFDataset(**VAL_CONFIG)

    trainsampler = DataSampler(train_set, 32000)
    if "efficientnet" in MODEL:
        train = data.DataLoader(train_set,  batch_size=25, sampler=trainsampler, num_workers=8)
        val = data.DataLoader(val_set, batch_size=25, shuffle=False, num_workers=8)
    else:
        train = data.DataLoader(train_set,  batch_size=32, sampler=trainsampler, num_workers=8)
        val = data.DataLoader(val_set, batch_size=32, shuffle=False, num_workers=8)

    model_folder_name = "{}{}_{}_{}_{}/".format(MODELS_FOLDER, MODEL, FACE_MARGIN, DATA_AUGMENTATION_LEVEL_STRING, EQUALIZE_HISTOGRAM_STRING)
    if not os.path.exists(model_folder_name):
        os.mkdir(model_folder_name)
    model_checkpoint = "{}{}_{}_{}_{}_{}.pth".format(model_folder_name, MODEL, MODEL_ID, FACE_MARGIN, DATA_AUGMENTATION_LEVEL_STRING, EQUALIZE_HISTOGRAM_STRING)
    model = CNNModel(backbone=MODEL, pretrained_backbone=True)

    loss_function = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0001)
    scheduler = optim.lr_scheduler.CyclicLR(
        optimizer, 
        base_lr=0.00002, 
        max_lr=0.0001, 
        cycle_momentum=False,
        step_size_up=500,
        step_size_down=500

    )
    model_wrapper = ModelWrapper(
        model = model, 
        optimizer = optimizer,
        scheduler = scheduler,
        loss_function = loss_function,
        train_generator = train, 
        val_generator = val,
        save_checkpoint = model_checkpoint,
        early_stopping_rounds = 3
    )

    model_wrapper.train(20)