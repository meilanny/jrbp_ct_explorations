"""
This script cleans data labels by validating all images are good and format them in the same way.
"""

import json
import csv


def wi_taxonomy(category_file):
    """Build a mapping from lower-cased common name -> taxonomy string.

    Taxonomy string format:
    uniqueIdentifier;class;order;family;genus;species;commonnameenglish
    where all taxonomic ranks and common name are lower-cased. The
    uniqueIdentifier is preserved as-is.
    """
    mapping = {}
    with open(category_file, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Defensive access with defaults
            cls = (row.get('class') or '').strip().lower()
            order = (row.get('order') or '').strip().lower()
            family = (row.get('family') or '').strip().lower()
            genus = (row.get('genus') or '').strip().lower()
            species = (row.get('species') or '').strip().lower()
            common_name = (row.get('commonNameEnglish') or '').strip().lower()
            uid = (row.get('uniqueIdentifier') or '').strip()

            if not common_name or not uid:
                continue

            taxonomy_string = ';'.join([
                uid,
                cls,
                order,
                family,
                genus,
                species,
                common_name,
            ])
            mapping[common_name] = taxonomy_string

    return mapping


def clean_jrct_data(jrct_file):
    # new json data
    new_data = {}

    wi_data = wi_taxonomy("data_labels/WI_Global_Taxonomy.csv")

    with open(jrct_file, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        # MetaId,Species,MySQLDateTime,CameraPath,FileName,Camera,WI_Species
        for row in reader:
            filepath = row['FileName']
            category = row['WI_Species']
            old_category = row['Species']
            if old_category == 'Other':
                category = "No CV Result"
            camera_path = row['CameraPath'].split('/')
            camera_array = camera_path[0]
            filepath = '/'.join([camera_path[1], filepath])

            if camera_array not in new_data:
                new_data[camera_array] = {}
            # Map WI_Species to full taxonomy string when a lowercase exact match exists
            mapped_category = wi_data.get(
                (category or '').strip().lower(), category)
            new_data[camera_array][filepath] = mapped_category

    # save to json
    with open('data_labels/jrct_cleaned_wi.json', 'w', encoding='utf-8') as f:
        json.dump(new_data, f, indent=2)

    return 0


def clean_speciesnet_data(speciesnet_file):
    with open(speciesnet_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    preds = data['predictions']

    # new json data
    new_data = {}

    for pred in preds:
        filepath = pred['filepath']
        detections = pred.get('detections', [])
        if 'detections' not in pred.keys():
            # Skip entries without detections; keep failures untouched
            continue

        # add to new data
        splits = filepath.split('/')
        camera_array = splits[4]
        filename = '/'.join(splits[5:])
        if camera_array not in new_data:
            new_data[camera_array] = {}
        new_data[camera_array][filename] = []

        if len(detections) > 0:
            for detection in detections:
                bbox = detection.get('bbox', [])
                cat = detection.get('speciesnet_category', [])
                score = detection.get('speciesnet_scores', [])

                if len(bbox) > 0 and len(cat) == 0:
                    # take the speciesnet results
                    cls_results = pred['classifications']
                    cat = cls_results['classes']
                    score = cls_results['scores']

                new_data[camera_array][filename].append({
                    'bbox': bbox,
                    'categories': cat,
                    'scores': score
                })

    # save to json
    with open('speciesnet_labels_cleaned/speciesnet_labels_a2_cleaned.json', 'w', encoding='utf-8') as f: # data_labels/speciesnet_cleaned.json
        json.dump(new_data, f, indent=2)

    return 0


if __name__ == "__main__":
    JRCT_FILE = "data_labels/jrct_class.csv" # data_labels/output_modified.jsony
    SPECIESNET_FILE = "speciesnet_labels/speciesnet_labels_a2.json"

    clean_jrct_data(JRCT_FILE)
    # clean_speciesnet_data(SPECIESNET_FILE)
