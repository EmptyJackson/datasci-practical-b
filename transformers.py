import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder

pd.options.mode.chained_assignment = None  # default='warn'

class CopyData(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X.copy()


class SmoothTargetEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, mean_samples=1):
        self.mean_samples = mean_samples

    def fit(self, X, y=None):
        self.mean = np.mean(y)
        self.labels = set(np.unique(X))
        self.label_means = {}
        for label in self.labels:
            self.label_means[label] = (np.sum(
                y[X == label]) + self.mean * self.mean_samples) / (len(y[X == label]) + self.mean_samples)
        return self

    def transform(self, X, y=None):
        ret = []
        for label in X:
            if label in self.labels:
                ret.append(self.label_means[label])
            else:
                ret.append(self.mean)
        return ret


class ModalImputer(BaseEstimator, TransformerMixin):
    def __init__(self, feature, nan_val=""):
        self.feature = feature
        self.nan_val = nan_val

    def fit(self, X, y=None):
        self.imputer = SimpleImputer(strategy='most_frequent')
        if self.nan_val:
            self.imputer.fit(X[self.feature].replace(
                self.nan_val, np.NaN).values.reshape(-1, 1))
        else:
            self.imputer.fit(X[self.feature].values.reshape(-1, 1))
        return self

    def transform(self, X, y=None):
        if self.nan_val:
            X[self.feature] = self.imputer.transform(
                X[self.feature].replace(self.nan_val, np.NaN).values.reshape(-1, 1))
        else:
            X[self.feature] = self.imputer.transform(
                X[self.feature].values.reshape(-1, 1))
        return X


class KNNImputer(BaseEstimator, TransformerMixin):
    def __init__(self, feature, nan_val="", excluded_feats=[], k=10):
        self.feature = feature
        self.excluded_feats = excluded_feats
        self.k = k
        self.nan_val = nan_val

    def fit(self, X, y=None):
        self.knn = KNeighborsClassifier(self.k)
        self.scaler = StandardScaler()
        X_not_na = X.copy()
        if self.nan_val:
            X_not_na[self.feature] = X_not_na[self.feature].replace(
                self.nan_val, np.NaN)
        X_not_na = X_not_na.dropna(subset=[self.feature]).drop(
            self.excluded_feats, axis=1)
        feat_col = X_not_na[self.feature]
        X_not_na.drop(self.feature, axis=1, inplace=True)
        self.predictor_names = X_not_na.columns
        predictors = self.scaler.fit_transform(X_not_na)
        self.knn.fit(predictors, feat_col)
        return self

    def transform(self, X, y=None):
        if self.nan_val:
            X[self.feature].replace(self.nan_val, np.NaN, inplace=True)
        feat_col = X[self.feature]
        is_feat_null = feat_col.isna()  # np.isnan(feat_col.values)
        predictors = X.copy()[self.predictor_names].values
        predictors = predictors[is_feat_null]
        # reshape(-1, 1) may be needed
        feat_preds = self.knn.predict(predictors)
        pred_i = 0
        for i in range(len(feat_col)):
            if is_feat_null[i]:
                feat_col[i] = feat_preds[pred_i]
                pred_i += 1
        X[self.feature] = X[self.feature].fillna(feat_col)
        return X


class UnhelpfulAttributeRemover(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.unhelpful_attributes = [
            'weight', 'payer_code']

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X.drop(self.unhelpful_attributes, axis=1, inplace=True)
        return X


class NanConverter(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.nan_attributes = ['race', 'medical_specialty', 'diag_1', 'diag_2', 'diag_3']

    def fit(self,X, y=None):
        return self
    
    def transform(self, X, y=None):
        for attr in self.nan_attributes:
            X[attr] = X[attr].replace('?', np.NaN)
        return X


class AgeIntervalConverter(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.range_midpoint = {}
        self.range_midpoint['[0-10)'] = 5
        for i in range(1, 10):
            self.range_midpoint['[' +
                                str(i) + '0-' + str(i+1) + '0)'] = i * 10 + 5

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X['age_mid'] = X['age'].replace(self.range_midpoint)
        X.drop('age', axis=1, inplace=True)
        return X


class NumericalTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, truncate_inpatient=True, incl_total_visits=True, drop_visits=False):
        self.incl_total_visits = incl_total_visits
        self.truncate_inpatient = truncate_inpatient
        self.drop_visits = drop_visits
        self.visit_cats = ['number_emergency',
                           'number_inpatient', 'number_outpatient']

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        if self.incl_total_visits:
            total_visits = np.zeros(X.shape[0])
            for cat in self.visit_cats:
                total_visits += X[cat].values
            X['total_visits_lg'] = np.log(total_visits + 1)
        if self.truncate_inpatient:
            X['number_inpatient'] = X['number_inpatient'].where(
                X['number_inpatient'] < 5, 5)
        if not self.drop_visits:
            for cat in self.visit_cats:
                X[cat + "_lg"] = np.log(X[cat] + 1)
        for cat in self.visit_cats:
            X.drop(cat, axis=1, inplace=True)
        return X


class BinarizeMedicationInfo(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.attrs = ['change', 'diabetesMed']

    def fit(self, X, y=None):
        self.encoders = []
        for attr in self.attrs:
            self.encoders.append(LabelEncoder())
            self.encoders[-1].fit(X[attr])
        return self

    def transform(self, X, y=None):
        for i in range(len(self.attrs)):
            X[self.attrs[i]] = self.encoders[i].transform(X[self.attrs[i]])
        return X


class GenderEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        X = X[X['gender'] != 'Unknown/Invalid']
        self.gender_enc = LabelEncoder()
        self.gender_enc.fit(X['gender'])
        return self

    def transform(self, X, y=None):
        X = X[X['gender'] != 'Unknown/Invalid']
        X.reset_index(drop=True, inplace=True)
        X['gender_bin'] = self.gender_enc.transform(X['gender'])
        X.drop('gender', axis=1, inplace=True)
        return X


class RaceEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, include_missing=False, drop_other=False):
        self.include_missing = include_missing
        self.drop_other = drop_other

    def fit(self, X, y=None):
        rs = X['race']
        if self.include_missing:
            rs = rs.fillna('Missing_race')
        rs = rs.values
        self.races = set(np.unique(rs))  # - set(['Other']))
        if self.drop_other:
            self.races -= set(['Other'])
        self.races = list(self.races)
        self.race_enc = OneHotEncoder(
            categories=[self.races], handle_unknown='ignore')
        self.race_enc.fit(rs.reshape(-1, 1))
        return self

    def transform(self, X, y=None):
        rs = X['race']
        if self.include_missing:
            rs = rs.fillna('Missing_race')
        rs = rs.values
        race_onehot = self.race_enc.transform(rs.reshape(-1, 1))
        X = X.join(pd.DataFrame(race_onehot.toarray(), columns=self.races))
        X.drop('race', axis=1, inplace=True)
        return X


class A1CEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, drop=False, include_missing=False, drop_norm=False):
        self.drop = drop
        self.include_missing = include_missing
        self.drop_norm = drop_norm

    def fit(self, X, y=None):
        self.ac_cats = list(np.unique(X['A1Cresult'].values))
        if not self.include_missing:
            self.ac_cats = list(set(self.ac_cats) - set(['None']))
        if self.drop_norm:
            self.ac_cats = list(set(self.ac_cats) - set(['Norm']))
        self.ac_enc = OneHotEncoder(
            categories=[self.ac_cats], handle_unknown='ignore')
        self.ac_enc.fit(X['A1Cresult'].values.reshape(-1, 1))
        return self

    def transform(self, X, y=None):
        if not self.drop:
            ac = self.ac_enc.transform(
                X['A1Cresult'].values.reshape(-1, 1)).toarray()
            X = X.join(pd.DataFrame(
                ac, columns=["A1Cresult_"+cat for cat in self.ac_cats]))
        X.drop(['A1Cresult'], axis=1, inplace=True)
        return X


class GlucoseEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, drop=False, include_missing=False, drop_norm=False):
        self.drop = drop
        self.include_missing = include_missing
        self.drop_norm = drop_norm

    def fit(self, X, y=None):
        self.glu_cats = list(np.unique(X['max_glu_serum'].values))
        if not self.include_missing:
            self.glu_cats = list(set(self.glu_cats) - set(['None']))
        if self.drop_norm:
            self.glu_cats = list(set(self.glu_cats) - set(['Norm']))
        self.glu_enc = OneHotEncoder(
            categories=[self.glu_cats], handle_unknown='ignore')
        self.glu_enc.fit(X['max_glu_serum'].values.reshape(-1, 1))
        return self

    def transform(self, X, y=None):
        if not self.drop:
            glu = self.glu_enc.transform(
                X['max_glu_serum'].values.reshape(-1, 1)).toarray()
            X = X.join(pd.DataFrame(glu, columns=[
                       "max_glu_serum_"+cat for cat in self.glu_cats]))
        X.drop(['max_glu_serum'], axis=1, inplace=True)
        return X


class MedicalSpecialtyEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, drop=True, include_missing=False, threshold=100):
        self.drop = drop
        self.include_missing = include_missing
        self.threshold = threshold

    def fit(self, X, y=None):
        if not self.drop:
            if self.include_missing:
                X['medical_specialty'] = X['medical_specialty'].fillna(
                    'Missing')
            spec_counts = X['medical_specialty'].value_counts()
            self.common_specs = spec_counts[spec_counts.values > self.threshold].keys(
            )
        return self

    def transform(self, X, y=None):
        if not self.drop:
            specs = X['medical_specialty']
            if self.include_missing:
                specs.fillna('Missing', inplace=True)
            med_spec_onehot = pd.get_dummies(specs)
            med_spec_onehot = med_spec_onehot[[
                spec for spec in self.common_specs if spec in med_spec_onehot.columns]]
            for spec in self.common_specs:
                if spec not in med_spec_onehot:
                    med_spec_onehot[spec] = np.zeros(len(X))
            X = X.join(med_spec_onehot)
        return X.drop('medical_specialty', axis=1)


class AdmissionDischargeEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=100, include_11=True):
        self.threshold = threshold
        self.id_types = ['admission_type_id',
                         'admission_source_id', 'discharge_disposition_id']
        self.id_nulls = [[5, 6, 8], [9, 15, 17, 20, 21], [18, 25, 26]]
        self.include_11 = include_11

    def fit(self, X, y=None):
        self.encoders = []
        self.common_ids = []
        for i, type_ in enumerate(self.id_types):
            id_counts = X[type_].value_counts()
            common = id_counts[id_counts.values > self.threshold].keys()
            common = list(set(common) - set(self.id_nulls[i]))
            self.common_ids.append(common)
            """
            encoder = OneHotEncoder(handle_unknown='ignore')
            encoder.fit(X[type_].fillna(0).values.reshape(-1, 1))
            self.encoders.append(encoder)
            """
        return self

    def transform(self, X, y=None):
        for i, type_ in enumerate(self.id_types):
            vals = X[type_]
            """
            vals = X[type_].fillna(0)
            for id_ in self.id_nulls[i]:
                vals = vals.where(vals != id_, 0)
            """
            type_onehot = pd.get_dummies(vals, type_)
            type_onehot = type_onehot[[type_ + '_' +
                                       str(id_) for id_ in self.common_ids[i]]]
            X = X.join(type_onehot)
            X.drop(type_, axis=1, inplace=True)
            #X = X.join(pd.DataFrame(vals, columns=[type_ + val for val in self.common_vals]))
        if not self.include_11 and 'discharge_disposition_id_11' in X:
            X.drop('discharge_disposition_id_11', axis=1, inplace=True)
        return X


class MedicationEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, discard_threshold=500, up_down_threshold=2000):
        self.medication_labels_str = "metformin, repaglinide, nateglinide, chlorpropamide, glimepiride, acetohexamide, glipizide, glyburide, tolbutamide, pioglitazone, rosiglitazone, acarbose, miglitol, troglitazone, tolazamide, examide, citoglipton, insulin, glyburide-metformin, glipizide-metformin, glimepiride-pioglitazone, metformin-rosiglitazone, metformin-pioglitazone"
        self.discard_threshold = discard_threshold
        self.up_down_threshold = up_down_threshold

    def fit(self, X, y=None):
        self.medication_labels = self.medication_labels_str.split(", ")
        self.unused_meds = []
        self.freq_meds = []
        for med in self.medication_labels:
            counts = X[med].value_counts()
            if (X.shape[0] - counts[0] < self.discard_threshold):
                self.unused_meds.append(med)
            elif 'Up' in counts and 'Down' in counts and counts['Up'] + counts['Down'] > self.up_down_threshold:
                self.freq_meds.append(med)
        self.medication_labels = [
            med for med in self.medication_labels if med not in self.unused_meds]
        return self

    def transform(self, X, y=None):
        X.drop(self.unused_meds, axis=1, inplace=True)
        encoder = LabelEncoder()
        for med in set(self.medication_labels) - set(self.freq_meds):
            X[med] = X[med].where(X[med] == 'No', 'Steady')
            med_enc = encoder.fit_transform(X[med])
            X[med] = med_enc
        for med in self.freq_meds:
            X = X.join(pd.get_dummies(X[med], prefix=med))
            X.drop(med, axis=1, inplace=True)
        return X


class DiagIntEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, only_first=True):
        self.num_diags = 3
        if only_first:
            self.num_diags = 1

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        for i in range(1, self.num_diags + 1):
            attr = 'diag_' + str(i)
            diag = X[attr]
            diag_arr = diag.fillna('0').values
            for i in range(len(diag_arr)):
                if isinstance(diag_arr[i], str):
                    if len(diag_arr[i]) > 3:
                        diag_arr[i] = diag_arr[i][:3]
                    if diag_arr[i][0] == 'V':
                        diag_arr[i] = '2' + diag_arr[i][1:] + '0'
                    elif diag_arr[i][0] == 'E':
                        diag_arr[i] = '1' + diag_arr[i][1:] + '0'
            diag_arr = diag_arr.astype(int)
            X[attr] = diag_arr
        for i in range(self.num_diags + 1, 4):
            X.drop('diag_' + str(i), axis=1, inplace=True)
        return X

# BASED ON PAPER GROUPINGS


class DiagIDCBin(BaseEstimator, TransformerMixin):
    def __init__(self, only_first=True):
        self.num_diags = 3
        if only_first:
            self.num_diags = 1

        diab = 1
        circ = 2
        resp = 3
        digest = 4
        inj = 5
        muscskel = 6
        genit = 7
        neo = 8
        self.bin_labels = {1: 'diabetes', 2: 'circulatory', 3: 'respiratory', 4: 'digestive',
                           5: 'injury', 6: 'musculoskeletal', 7: 'genitourinary', 8: 'neoplasms'}

        self.max_id = 8
        self.idc_bins = {250: diab, 785: circ,
                         786: resp, 787: digest, 788: genit}
        for i in range(390, 460):
            self.idc_bins[i] = circ
        for i in range(460, 520):
            self.idc_bins[i] = resp
        for i in range(520, 580):
            self.idc_bins[i] = digest
        for i in range(800, 1000):
            self.idc_bins[i] = inj
        for i in range(710, 740):
            self.idc_bins[i] = muscskel
        for i in range(580, 630):
            self.idc_bins[i] = genit
        for i in range(140, 240):
            self.idc_bins[i] = neo

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        for i in range(1, self.num_diags+1):
            diag = X['diag_' + str(i)]
            diag = pd.Series(np.where(diag <= self.max_id, 0, diag))
            diag.replace(self.idc_bins, inplace=True)
            diag = pd.Series(np.where(diag > self.max_id, 0, diag))
            diag.replace(0, 'other', inplace=True)
            for id_ in range(1, self.max_id + 1):
                diag.replace(id_, self.bin_labels[id_], inplace=True)
            X = X.join(pd.get_dummies(diag, 'diag_' + str(i)))
            #X['diag_' + str(i)] = diag
            X.drop('diag_' + str(i), axis=1, inplace=True)
        return X


class ReadmissionBinarizer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.encoder = LabelEncoder()

    def fit(self, X, y=None):
        self.encoder.fit(X['readmitted'])
        return self

    def transform(self, X, y=None):
        X['readmit_bin'] = self.encoder.transform(X['readmitted'])
        X.drop('readmitted', axis=1, inplace=True)
        return X


pipeline = Pipeline([
    ('copy data', CopyData()),
    ('unhelpful attrs', UnhelpfulAttributeRemover()),
    ('nans', NanConverter()),
    ('readmission', ReadmissionBinarizer()),
    ('gender', GenderEncoder()),
    ('age', AgeIntervalConverter()),
    ('numerical', NumericalTransformer(truncate_inpatient=False,
                                       incl_total_visits=False, drop_visits=False)),
    ('medication info', BinarizeMedicationInfo()),
    ('admission & discharge', AdmissionDischargeEncoder(threshold=50)),
    ('medication', MedicationEncoder(discard_threshold=50, up_down_threshold=np.inf)),
    ('diag int', DiagIntEncoder(only_first=True)),
    #('diag smooth target', DiagSmoothTarget(only_first=True, smoothing=5)),
    ('diag IDC bin', DiagIDCBin(only_first=True)),
    #('race impute', KNNImputer('race', excluded_feats=['medical_specialty', 'max_glu_serum', 'A1Cresult'], k=5)),
    ('race encode', RaceEncoder(include_missing=True, drop_other=True)),
    ('speciality impute', ModalImputer('medical_specialty')),
    ('speciality encode', MedicalSpecialtyEncoder(
        drop=False, include_missing=False, threshold=50)),
    ('glucose impute', ModalImputer('max_glu_serum', nan_val='None')),
    ('glucose encode', GlucoseEncoder(
        drop=False, include_missing=False, drop_norm=True)),
    ('A1C impute', KNNImputer('A1Cresult', nan_val='None')),
    ('A1C encode', A1CEncoder(drop=False, include_missing=False, drop_norm=True)),
    #('diag/spec PCA', DiagSpecPCA(k=13)),
    #('scaler', StandardScaler())
],
    verbose=False)

def main():
    data = pd.read_csv(
        "https://raw.githubusercontent.com/ekochmar/cl-datasci-pnp/master/Final_assignment/diabetes/diabetic_data_original.csv")

    data_trans = pipeline.fit_transform(data)
    return data_trans

if __name__ == '__main__':
    x = main()
    print(x.head())
