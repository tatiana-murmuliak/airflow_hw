import logging
import os
from datetime import datetime

import dill
import json
import pandas as pd

path = os.environ.get('PROJECT_PATH', '.')

def predict():
    mod = sorted(os.listdir(f'{path}/data/models/'))[-1]
    with open(f'{path}/data/models/{mod}', 'rb') as file:
        model = dill.load(file)

    preds = pd.DataFrame(columns=['car_id', 'pred'])
    tests = os.listdir(f'{path}/data/test')

    for filename in tests:
        with open(f'{path}/data/test/{filename}', 'rb') as file_test:
            form = json.load(file_test)
        df = pd.DataFrame.from_dict([form])
        y = model.predict(df)
        x = {'car_id': df.id, 'pred': y}
        df_pred = pd.DataFrame(x)
        preds = pd.concat([preds, df_pred], axis=0)

    preds_filename = datetime.now().strftime("%Y%m%d%H%M")
    preds.to_csv(f'{path}/data/predictions/preds_{preds_filename}.csv', index=False)

    logging.info(f'Predictions are saved as {preds_filename}')

if __name__ == '__main__':
    predict()
