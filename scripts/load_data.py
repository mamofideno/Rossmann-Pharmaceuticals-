import pandas as pd
class LoadData:
    def __init__(self, path) -> None:
        self.path=path

    def load(self):
        try:
            data=pd.read_csv(self.path)
            return data
        except Exception as e:
            print(f'Error occured:{e}')        