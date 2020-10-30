#!/bin/sh
python create_metadata.py
python get_face_coordinates.py
python process_faces.py -f 0 -c 8
python process_faces.py -f 10 -c 8
python process_faces.py -f 30 -c 8
python process_faces.py -f 50 -c 8
python balance_val_test.py
