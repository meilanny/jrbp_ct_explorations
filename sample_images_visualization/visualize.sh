#!/bin/bash
IMAGE_DIR=/home/lanmei/jrbp_ct_explorations/sample_images_visualization/sample_images_jrbp_b1/ori/
CUDA_VISIBLE_DEVICES=1 python -m speciesnet.scripts.run_model --folders ${IMAGE_DIR} --predictions_json ${IMAGE_DIR}/speciesnet-results.json

PREVIEW_DIR=/home/lanmei/jrbp_ct_explorations/sample_images_visualization/sample_images_jrbp_b1/labeled/
pip install megadetector-utils
CUDA_VISIBLE_DEVICES=1 python -m megadetector.visualization.visualize_detector_output ${IMAGE_DIR}/speciesnet-results.json ${PREVIEW_DIR}