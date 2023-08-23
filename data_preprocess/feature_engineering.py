import pandas
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler


class Feature_engineering_SN:
    def __init__(self, method=None):
        assert method in ['normalization', 'standardization']
        type_map = {'normalization': MinMaxScaler(), 'standardization': StandardScaler()}
        self.processor = type_map[method]

    def fit(self, data):
        self.processor.fit(data)

    def transform(self, data):
        return self.processor.transform(data)


class Feature_engineering_DMR:
    def __init__(self, method=None):
        self.method = method

    def transform(self, data: pandas.DataFrame, filled_value=None, delete_missing_row=False, resample_fq=None):
        if self.method == 'duplicate values handling':
            return data.drop_duplicates()
        elif self.method == 'missing value handling':
            if delete_missing_row:
                return data.dropna()
            else:
                return data.fillna(filled_value, inplace=False)
        elif self.method == 'resample':
            return data.resample(resample_fq).asfreq(fill_value=filled_value)
