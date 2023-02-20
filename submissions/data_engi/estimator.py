from sklearn.base import BaseEstimator
 
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import make_pipeline

from sklearn.feature_selection import RFE
 
import optuna
 
from xgboost import XGBClassifier as XGBC 
# from imblearn.pipeline import make_pipeline
# from imblearn.over_sampling import RandomOverSampler
 
import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
 
def disable_pandas_warnings():
    import warnings
    warnings.resetwarnings()  # Maybe somebody else is messing with the warnings system?
    warnings.filterwarnings('ignore')  # Ignore everything
    # ignore everything does not work: ignore specific messages, using regex
    warnings.filterwarnings('ignore', '.*A value is trying to be set on a copy of a slice from a DataFrame.*')
    warnings.filterwarnings('ignore', '.*indexing past lexsort depth may impact performance*')
 
class FeatureExtractor(BaseEstimator):
    def __init__(self):
        self.functions = ['std', 'mean', 'median', 'max', 'min', 'sum', 'shift']
        self.keep = [
            "Beta", "Vth", "B", "RmsBob", "Np"
        ]
        self.windows = [
            '1h', '2h', '3h', '6h', '12h', '24h'
        ]
        # # Features list, sorted by increasing order of importance.
        # # The order was obtained with XGB Classifier.
        # # We throw the first p of them.
        # self.ordered = ['Beta_3h_sum', 'Np_1h_sum', 'RmsBob_6h_shift', 'Beta_1h_shift', 'Vth_2h_shift', 'RmsBob_12h_shift', 'Beta_2h_shift', 'RmsBob_2h_shift', 'B_2h_shift', 'RmsBob_24h_shift', 'Vth_3h_sum', 'Beta_3h_shift', 'Np_6h_sum', 'RmsBob_1h_shift', 'RmsBob', 'Np_1h_shift', 'Np_2h_shift', 'B_1h_shift', 'Beta_1h_sum', 'Beta_6h_shift', 'Vth_1h_std', 'Vth_1h_shift', 'Np_3h_sum', 'RmsBob_2h_sum', 'Vth_3h_shift', 'B_1h_sum', 'B_1h_median', 'RmsBob_1h_std', 'RmsBob_3h_sum', 'RmsBob_1h_min', 'B_2h_median', 'B_6h_median', 'Vth_2h_std', 'Vth_6h_shift', 'Beta_2h_std', 'RmsBob_1h_sum', 'Beta_12h_shift', 'B_3h_shift', 'B_2h_mean', 'B_1h_std', 'Np_1h_std', 'Vth_12h_shift', 'Np_3h_std', 'Np_3h_mean', 'Beta_3h_mean', 'RmsBob_2h_median', 'RmsBob_3h_shift', 'B_3h_sum', 'B_2h_min', 'Np_3h_shift', 'Vth_2h_sum', 'Np_12h_sum', 'B_3h_mean', 'Vth_24h_shift', 'B_12h_shift', 'Np_1h_max', 'Beta_2h_sum', 'RmsBob_2h_min', 'Beta_6h_sum', 'Np', 'Vth_3h_median', 'Beta_24h_shift', 'Vth_12h_sum', 'Np_6h_shift', 'RmsBob_1h_median', 'Np_12h_shift', 'RmsBob_3h_max', 'Vth_6h_std', 'Vth_1h_max', 'RmsBob_2h_mean', 'Vth_3h_std', 'Beta_3h_std', 'Vth_3h_min', 'B_3h_min', 'B_1h_min', 'B_6h_mean', 'Np_12h_median', 'Np_24h_shift', 'B', 'Np_2h_min', 'Vth_6h_sum', 'Np_6h_std', 'Vth_12h_mean', 'Vth_2h_mean', 'RmsBob_2h_std', 'Np_2h_std', 'Np_2h_mean', 'Vth_1h_min', 'Vth_12h_median', 'RmsBob_6h_median', 'Vth_1h_sum', 'Beta_2h_median', 'B_12h_median', 'Vth_6h_min', 'Beta_12h_sum', 'RmsBob_12h_sum', 'Beta_1h_std', 'Vth_6h_mean', 'Beta_12h_mean', 'Np_2h_max', 'Np_1h_mean', 'B_2h_sum', 'Np_3h_max', 'B_2h_std', 'RmsBob_24h_median', 'RmsBob_3h_min', 'Np_12h_std',
        #                 'Vth_1h_mean', 'Np_6h_median', 'B_6h_sum', 'B_6h_min', 'Beta_24h_median', 'Np_1h_median', 'B_3h_median', 'B_12h_std', 'B_24h_shift', 'Beta_12h_std', 'RmsBob_1h_max', 'B_6h_std', 'RmsBob_3h_std', 'Np_1h_min', 'Vth_6h_median', 'Np_12h_max', 'Vth', 'Vth_2h_max', 'RmsBob_6h_std', 'Np_24h_mean', 'Np_3h_median', 'B_3h_std', 'B_1h_mean', 'B_6h_shift', 'Vth_12h_std', 'Beta_1h_median', 'RmsBob_24h_sum', 'Beta_24h_mean', 'Np_6h_mean', 'Vth_1h_median', 'B_24h_mean', 'B_1h_max', 'B_12h_max', 'B_6h_max', 'B_12h_mean', 'Np_6h_max', 'Beta_6h_std', 'Np_12h_mean', 'Beta_24h_std', 'RmsBob_3h_median', 'RmsBob_6h_min', 'B_24h_median', 'Beta_1h_min', 'Vth_24h_sum', 'RmsBob_12h_max', 'Beta_3h_min', 'Beta_1h_max', 'Vth_24h_std', 'Np_2h_sum', 'Vth_12h_min', 'Np_24h_std', 'Beta_6h_mean', 'B_12h_sum', 'Np_3h_min', 'Beta_24h_max', 'Beta_6h_max', 'Np_6h_min', 'RmsBob_6h_sum', 'Np_24h_median', 'Beta_3h_max', 'Vth_2h_median', 'Vth_12h_max', 'B_12h_min', 'RmsBob_6h_max', 'B_24h_min', 'Vth_6h_max', 'Vth_3h_max', 'Np_24h_sum', 'Vth_2h_min', 'RmsBob_24h_mean', 'RmsBob_12h_mean', 'Beta_12h_max', 'RmsBob_24h_std', 'RmsBob_12h_std', 'Vth_24h_median', 'Vth_24h_min', 'Vth_24h_max', 'Beta_24h_sum', 'RmsBob_12h_median', 'B_24h_std', 'Beta_2h_min', 'Vth_3h_mean', 'Beta_2h_max', 'Np_24h_max', 'RmsBob_24h_min', 'RmsBob_12h_min', 'B_3h_max', 'RmsBob_24h_max', 'Np_12h_min', 'Vth_24h_mean', 'Np_24h_min', 'RmsBob_2h_max', 'Np_2h_median', 'B_2h_max', 'Beta', 'Beta_24h_min', 'B_24h_max', 'Beta_6h_median', 'Beta_12h_median', 'RmsBob_1h_mean', 'B_24h_sum', 'Beta_12h_min', 'RmsBob_6h_mean', 'Beta_6h_min', 'RmsBob_3h_mean', 'Beta_3h_median', 'Beta_1h_mean', 'Beta_2h_mean']
        # # For constant lookup times, we favor the use of dicts.
        # self.ordered_dict = {
        #     self.ordered[i]: i for i in range(len(self.ordered))
        # }
        # self.p = (len(self.ordered)//3)
        # # Handpicked removals
        # self.throw = []
        # for feature in self.throw:
        #     if feature in self.ordered_dict:
        #         self.ordered_dict[feature] = 0
    
    def fit(self, X, y):
        self.drops = list(X.columns)
        for feature in self.keep:
            self.drops.remove(feature)
        return self
 
    def transform(self, X):
        disable_pandas_warnings()
        X_df = X.copy()
        for f_name in self.functions:
            for window in self.windows:
                for var in self.keep:
                    # name = '_'.join([var, window, f_name])
                    # if name in self.ordered_dict and self.ordered_dict[name] > self.p:
                    X_df = compute_rolling_variable(X_df, [var], window, f_name)
        X_df.drop(columns=self.drops, inplace=True)
        return X_df
 
class Classifier(BaseEstimator):
 
    def __init__(self):
        self.model = HistGradientBoostingClassifier(
            l2_regularization=50,
            max_depth=8,
            max_leaf_nodes=15,
            random_state=0
        )
        self.postprocess = PostProcessor()
 
    def fit(self, X, y):
        print(X.columns)
        self.model.fit(X, y)
        self.postprocess.fit(self.model.predict_proba(X), y)
 
    def predict(self, X):
        y_pred = self.model.predict_proba(X)
        y_pred = self.postprocess.transform(y_pred)
        return y_pred
 
 
def get_estimator():
 
    feature_extractor = FeatureExtractor()
 
    # ros = RandomOverSampler(
    #     sampling_strategy=.2,
    #     random_state=42
    # )

    selector = RFE(XGBC(
            max_depth=5,
            n_jobs=-1,
            eval_metric='logloss'
        ), n_features_to_select=50, step=5)
 
    classifier = Classifier()
 
    pipe = make_pipeline(feature_extractor, selector, classifier)
    return pipe
 
def compute_rolling_variable(X_df, features, time_window, variable, center=False):
    """
    For a given dataframe, compute the rolling variable over
    a defined period of time (time_window) of a defined feature
 
    Parameters
    ----------
    X : dataframe
    features : list[str]
        features in the dataframe we wish to compute the rolling variable (e.g. std) from
    time_window : str
        string that defines the length of the time window passed to `rolling`
    variable : {'std', 'mean', 'median', 'max', 'min', 'sum', 'shift'}
        string that defines the variable used to compute the rolling variable (e.g. std)
    center : bool
        boolean to indicate if the point of the dataframe considered is
        center or end of the window
    """
    if not variable in {'std', 'mean', 'median', 'max', 'min', 'sum', 'shift'}:
        raise ValueError(f"{variable} is not a valid choice.")
    else:
        if variable == 'shift':
            for feature in features:
                name = '_'.join([feature, time_window, 'shift'])
                X_df[name] = X_df[feature].shift(freq=time_window)
                X_df[name] = X_df[name].ffill().bfill()
                X_df[name] = X_df[name].astype(X_df[feature].dtype)
            return X_df
        else:
            for feature in features:
                name = '_'.join([feature, time_window, variable])
                if variable == 'std':
                    X_df[name] = X_df[feature].rolling(time_window, center=center).std()
                elif variable == 'mean':
                    X_df[name] = X_df[feature].rolling(time_window, center=center).mean()
                elif variable == 'median':
                    X_df[name] = X_df[feature].rolling(time_window, center=center).median()
                elif variable == 'max':
                    X_df[name] = X_df[feature].rolling(time_window, center=center).max()
                elif variable == 'min':
                    X_df[name] = X_df[feature].rolling(time_window, center=center).min()
                else:
                    X_df[name] = X_df[feature].rolling(time_window, center=center).sum()
                X_df[name] = X_df[name].ffill().bfill()
                X_df[name] = X_df[name].astype(X_df[feature].dtype)
            return X_df
 
def compute_ewm_mean(X_df, features, halflife):
    """
    For a given dataframe, compute the ewm mean over
    a defined period of time (halflife) of a defined feature
 
    Parameters
    ----------
    X : dataframe
    features : list[str]
        features in the dataframe we wish to compute the rolling ewm from
    halflife : str
        string that defines the length of the time window passed to `halflife`
    """
    for feature in features:
        name = '_'.join([feature, halflife, 'ewm_mean'])
        X_df[name] = X_df[feature].ewm(halflife=halflife, times=X_df.index).mean()
        X_df[name] = X_df[name].ffill().bfill()
        X_df[name] = X_df[name].astype(X_df[feature].dtype)
    return X_df
 
def smooth_prediction(y_pred, quantile, time_window):
    s = pd.Series(y_pred[:, 1]).rolling(time_window, min_periods=0, center=True).quantile(quantile)
    return np.column_stack((1-s, s))
 
class PostProcessor():
    # I would like to thank Alexandre Bigot for sharing his great idea of postprocessing with me.
    def __init__(self):
        # self.study = optuna.create_study(direction='minimize')
        self.postprocess_fn = smooth_prediction
    
    def fit(self, y_pred, labels):
        # score = Mixed()
        # idx = np.arange(0, len(labels)*10, 10)
        # # Labels are 1-D and there are as many labels as predictions.
        # if len(labels.shape) == 1 and labels.shape[0] == y_pred.shape[0]:
        #     y_true = np.column_stack((1-labels.values, labels.values))
        #     y_true_mixed = np.column_stack((idx, 1-labels.values, labels.values))
        # else:
        #     print(f"{labels.shape=}, {y_pred.shape=}")
        #     assert 0, "error in postProcess"
        # assert y_true.shape == y_pred.shape, f"{y_true.shape=}, {y_pred.shape=}"
 
        # def objective(trial):
        #     params = {
        #     'quantile': trial.suggest_float('quantile', 0.5, 0.7),
        #     'time_window': trial.suggest_int('time_window', 50, 80)
        #     }
        #     y_pred_t = self.postprocess_fn(y_pred, **params)
        #     y_pred_t_mixed = np.column_stack((idx, y_pred_t[:, 0], y_pred_t[:, 1]))
        #     return score(y_true_mixed, y_pred_t_mixed)
        # print("Beginning study.")
        # self.study.optimize(objective, n_trials=25, n_jobs=-1)
        # print("Study complete. Best params: \n", self.study.best_params)
        pass
    
    def transform(self, y_pred):
        # return self.postprocess_fn(y_pred, **self.study.best_params)
        # Found using optuna, but the server does not support optuna.
        return self.postprocess_fn(y_pred, **{'quantile': 0.6, 'time_window': 65})
 
 
################### Death Spaghetti code for the metrics ###################
 
import datetime
from rampwf.score_types.base import BaseScoreType
from rampwf.score_types.classifier_base import ClassifierBaseScoreType
from sklearn.metrics import log_loss, recall_score, precision_score
 
# -----------------------------------------------------------------------------
# Score types
# -----------------------------------------------------------------------------
 
 
class PointwiseLogLoss(BaseScoreType):
    # subclass BaseScoreType to use raw y_pred (proba's)
    is_lower_the_better = True
    minimum = 0.0
    maximum = np.inf
 
    def __init__(self, name='pw_ll', precision=2):
        self.name = name
        self.precision = precision
 
    def __call__(self, y_true, y_pred):
        score = log_loss(y_true[:, 1:], y_pred[:, 1:])
        return score
 
 
class PointwisePrecision(ClassifierBaseScoreType):
    is_lower_the_better = False
    minimum = 0.0
    maximum = 1.0
 
    def __init__(self, name='pw_prec', precision=2):
        self.name = name
        self.precision = precision
 
    def __call__(self, y_true_label_index, y_pred_label_index):
        score = precision_score(y_true_label_index, y_pred_label_index)
        return score
 
 
class PointwiseRecall(ClassifierBaseScoreType):
    is_lower_the_better = False
    minimum = 0.0
    maximum = 1.0
 
    def __init__(self, name='pw_rec', precision=2):
        self.name = name
        self.precision = precision
 
    def __call__(self, y_true_label_index, y_pred_label_index):
        score = recall_score(y_true_label_index, y_pred_label_index)
        return score
 
 
class EventwisePrecision(BaseScoreType):
    # subclass BaseScoreType to use raw y_pred (proba's)
    is_lower_the_better = False
    minimum = 0.0
    maximum = 1.0
 
    def __init__(self, name='ev_prec', precision=2):
        self.name = name
        self.precision = precision
 
    def __call__(self, y_true, y_pred):
        y_true = pd.Series(
            y_true[:, 2],
            index=pd.to_datetime(y_true[:, 0].astype('int64'), unit='m'))
        y_pred = pd.Series(
            y_pred[:, 2],
            index=pd.to_datetime(y_pred[:, 0].astype('int64'), unit='m'))
        event_true = turn_prediction_to_event_list(y_true)
        event_pred = turn_prediction_to_event_list(y_pred)
        FP = [x for x in event_pred
              if max(overlap_with_list(x, event_true, percent=True)) < 0.5]
        if len(event_pred):
            score = 1 - len(FP) / len(event_pred)
        else:
            # no predictions -> precision not defined, but setting to 0
            score = 0
        return score
 
 
class EventwiseRecall(BaseScoreType):
    is_lower_the_better = False
    minimum = 0.0
    maximum = 1.0
 
    def __init__(self, name='ev_rec', precision=2):
        self.name = name
        self.precision = precision
 
    def __call__(self, y_true, y_pred):
        y_true = pd.Series(
            y_true[:, 2],
            index=pd.to_datetime(y_true[:, 0].astype('int64'), unit='m'))
        y_pred = pd.Series(
            y_pred[:, 2],
            index=pd.to_datetime(y_pred[:, 0].astype('int64'), unit='m'))
        event_true = turn_prediction_to_event_list(y_true)
        event_pred = turn_prediction_to_event_list(y_pred)
        if not event_pred:
            return 0.
        FN = 0
        for event in event_true:
            corresponding = find(event, event_pred, 0.5, 'best')
            if corresponding is None:
                FN += 1
        score = 1 - FN / len(event_true)
        return score
 
 
class EventwiseF1(BaseScoreType):
    is_lower_the_better = False
    minimum = 0.0
    maximum = 1.0
 
    def __init__(self, name='ev_F1', precision=2):
        self.name = name
        self.precision = precision
        self.eventwise_recall = EventwiseRecall()
        self.eventwise_precision = EventwisePrecision()
 
    def __call__(self, y_true, y_pred):
        rec = self.eventwise_recall(y_true, y_pred)
        prec = self.eventwise_precision(y_true, y_pred)
        return 2 * (prec * rec) / (prec + rec + 10 ** -15)
 
 
class Mixed(BaseScoreType):
    is_lower_the_better = True
    minimum = 0.0
    maximum = np.inf
 
    def __init__(self, name='mixed', precision=2):
        self.name = name
        self.precision = precision
        self.event_wise_f1 = EventwiseF1()
        self.pointwise_log_loss = PointwiseLogLoss()
 
    def __call__(self, y_true, y_pred):
        f1 = self.event_wise_f1(y_true, y_pred)
        ll = self.pointwise_log_loss(y_true, y_pred)
        return ll + 0.1 * (1 - f1)
 
 
class Event:
    def __init__(self, begin, end):
        self.begin = begin
        self.end = end
        self.duration = self.end - self.begin
 
    def __str__(self):
        return "{} ---> {}".format(self.begin, self.end)
 
    def __repr__(self):
        return "Event({} ---> {})".format(self.begin, self.end)
 
 
def overlap(event1, event2):
    """Return the time overlap between two events as a timedelta"""
    delta1 = min(event1.end, event2.end)
    delta2 = max(event1.begin, event2.begin)
    return max(delta1 - delta2, datetime.timedelta(0))
 
 
def overlap_with_list(ref_event, event_list, percent=False):
    """
    Return the list of the overlaps between an event and the elements of
    an event list
    Have the possibility to have it as the percentage of fthe considered event
    in the list
    """
    if percent:
        return [overlap(ref_event, elt) / elt.duration for elt in event_list]
    else:
        return [overlap(ref_event, elt) for elt in event_list]
 
 
def is_in_list(ref_event, event_list, thres):
    """
    Return True if ref_event is overlapped thres percent of its duration by
    at least one elt in event_list
    """
    return max(overlap_with_list(
        ref_event, event_list)) > thres * ref_event.duration
 
 
def merge(event1, event2):
    return Event(event1.begin, event2.end)
 
 
def choose_event_from_list(ref_event, event_list, choice='first'):
    """
    Return an event from even_list according to the choice adopted
    first return the first of the lists
    last return the last of the lists
    best return the one with max overlap
    merge return the combination of all of them
    """
    if choice == 'first':
        return event_list[0]
    if choice == 'last':
        return event_list[-1]
    if choice == 'best':
        return event_list[np.argmax(overlap_with_list(ref_event, event_list))]
    if choice == 'merge':
        return merge(event_list[0], event_list[-1])
 
 
def find(ref_event, event_list, thres, choice='best'):
    """
    Return the event in event_list that overlap ref_event for a given threshold
    if it exists
    Choice give the preference of returned :
    first return the first of the lists
    Best return the one with max overlap
    merge return the combination of all of them
    """
    if is_in_list(ref_event, event_list, thres):
        return(choose_event_from_list(ref_event, event_list, choice))
    else:
        return None
 
 
def turn_prediction_to_event_list(y, thres=0.5):
    """
    Consider y as a pandas series, returns a list of Events corresponding to
    the requested label (int), works for both smoothed and expected series
    Delta corresponds to the series frequency (in our basic case with random
    index, we consider this value to be equal to 2)
    """
 
    listOfPosLabel = y[y > thres]
    deltaBetweenPosLabel = listOfPosLabel.index[1:] - listOfPosLabel.index[:-1]
    deltaBetweenPosLabel.insert(0, datetime.timedelta(0))
    endOfEvents = np.where(deltaBetweenPosLabel >
                           datetime.timedelta(minutes=10))[0]
    indexBegin = 0
    eventList = []
    for i in endOfEvents:
        end = i
        eventList.append(Event(listOfPosLabel.index[indexBegin],
                         listOfPosLabel.index[end]))
        indexBegin = i + 1
    if len(endOfEvents):
        eventList.append(Event(listOfPosLabel.index[indexBegin],
                               listOfPosLabel.index[-1]))
    i = 0
    eventList = [evt for evt in eventList
                 if evt.duration > datetime.timedelta(0)]
    while i < len(eventList) - 1:
        if ((eventList[i + 1].begin - eventList[i].end) <
                datetime.timedelta(hours=1)):
            eventList[i] = merge(eventList[i], eventList[i + 1])
            eventList.remove(eventList[i + 1])
        else:
            i += 1
 
    eventList = [evt for evt in eventList
                 if evt.duration >= datetime.timedelta(hours=2.5)]
 
    return eventList