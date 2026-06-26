import os
import joblib

def load_model(config):
        cls_path = config['models']['classifier']
        reg_path = config['models']['regressor']

        if not os.path.exists(cls_path):
           raise FileNotFoundError(f"Classifier not foun : {cls_path}")
        if not os.path.exists(reg_path):
          raise FileNotFoundError(f"Regressor not foun : {reg_path}")


        cls = joblib.load(cls_path)
        reg = joblib.load(reg_path)
        return cls,reg