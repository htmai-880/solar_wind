from sklearn.base import BaseEstimator

from sklearn.ensemble import HistGradientBoostingClassifier

from imblearn.pipeline import make_pipeline
from imblearn.over_sampling import RandomOverSampler, SMOTE


class FeatureExtractor(BaseEstimator):

    def fit(self, X, y):
        # Dropping some columns
        # self.drops = ['B', 'Bx', 'Bx_rms', 'By', 'By_rms', 'Bz', 'Bz_rms', 'Na_nl', 'Np',
        #     'Np_nl', 'Range F 0', 'Range F 1', 'Range F 10', 'Range F 11',
        #     'Range F 12', 'Range F 13', 'Range F 14', 'Range F 2', 'Range F 3',
        #     'Range F 4', 'Range F 5', 'Range F 6', 'Range F 7', 'Range F 8',
        #     'Range F 9', 'V', 'Vth', 'Vx', 'Vy', 'Vz', 'Beta', 'Pdyn', 'RmsBob']
        self.drops = list(X.columns)
        keep = [
            "Beta", "Vth", "B", "RmsBob"
        ]
        for feature in keep:
            self.drops.remove(feature)
        return self

    def transform(self, X):
        # We want to compute the rolling std for the following parameter.
        # It allows us to visualize the variation of the parameter rather than
        # the parameter itself.
        X_df = X.copy()
        X_df = compute_rolling_std(X_df, [
            'B', 'Vth', 'RmsBob'
        ], '3h')
        # Compute a local integral here. Some events
        # are characterized by a large integral curve before they happen.
        X_df = compute_rolling_sum(X_df, [
            'Vth', 'RmsBob'
        ], '24h')
        # inspecting beta variations
        X_df = compute_rolling_std(X_df, [
            'Beta'
        ], '2h')
        X_df = compute_rolling_std(X_df, [
            'Beta'
        ], '12h')
        # inspecting beta integrals
        X_df = compute_rolling_sum(X_df, [
            'Beta'
        ], '12h')
        # Compute local rolling maximum.
        X_df = compute_rolling_max(X_df, [
            'Beta'
        ], '12h')
        # Compute local rolling median.
        X_df = compute_rolling_median(X_df, [
            'Beta'
        ], '24h')
        # inspecting rmsbob mean and std on larger window
        X_df = compute_rolling_std(X_df, [
            'RmsBob'
        ], '12h')
        X_df = compute_rolling_mean(X_df, [
            'RmsBob'
        ], '12h')
        # Smoothed version
        X_df = compute_rolling_mean(X_df, [
            'B', 'Beta', 'Vth', 'RmsBob'
        ], '3h')
        X_df.drop(columns=self.drops, inplace=True)
        return X_df


class Classifier(BaseEstimator):

    def __init__(self):
        self.model = HistGradientBoostingClassifier(
            l2_regularization=100,
            random_state=0
        )

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        y_pred = self.model.predict_proba(X)
        return y_pred


def get_estimator():

    feature_extractor = FeatureExtractor()

    ros = RandomOverSampler(
        sampling_strategy=.2,
        random_state=42
    )

    classifier = Classifier()

    pipe = make_pipeline(feature_extractor, ros, classifier)
    return pipe


def compute_rolling_std(X_df, features, time_window, center=False):
    """
    For a given dataframe, compute the standard deviation over
    a defined period of time (time_window) of a defined feature

    Parameters
    ----------
    X : dataframe
    features : list[str]
        features in the dataframe we wish to compute the rolling std from
    time_window : str
        string that defines the length of the time window passed to `rolling`
    center : bool
        boolean to indicate if the point of the dataframe considered is
        center or end of the window
    """
    for feature in features:
        name = '_'.join([feature, time_window, 'std'])
        X_df[name] = X_df[feature].rolling(time_window, center=center).std()
        X_df[name] = X_df[name].ffill().bfill()
        X_df[name] = X_df[name].astype(X_df[feature].dtype)
    return X_df


def compute_rolling_mean(X_df, features, time_window, center=False):
    """
    For a given dataframe, compute the mean over
    a defined period of time (time_window) of a defined feature

    Parameters
    ----------
    X : dataframe
    features : list[str]
        features in the dataframe we wish to compute the rolling mean from
    time_window : str
        string that defines the length of the time window passed to `rolling`
    center : bool
        boolean to indicate if the point of the dataframe considered is
        center or end of the window
    """
    for feature in features:
        name = '_'.join([feature, time_window, 'mean'])
        X_df[name] = X_df[feature].rolling(time_window, center=center).mean()
        X_df[name] = X_df[name].ffill().bfill()
        X_df[name] = X_df[name].astype(X_df[feature].dtype)
    return X_df


def compute_rolling_median(X_df, features, time_window, center=False):
    """
    For a given dataframe, compute the median over
    a defined period of time (time_window) of a defined feature

    Parameters
    ----------
    X : dataframe
    features : list[str]
        features in the dataframe we wish to compute the rolling median from
    time_window : str
        string that defines the length of the time window passed to `rolling`
    center : bool
        boolean to indicate if the point of the dataframe considered is
        center or end of the window
    """
    for feature in features:
        name = '_'.join([feature, time_window, 'median'])
        X_df[name] = X_df[feature].rolling(time_window, center=center).median()
        X_df[name] = X_df[name].ffill().bfill()
        X_df[name] = X_df[name].astype(X_df[feature].dtype)
    return X_df


def compute_rolling_max(X_df, features, time_window, center=False):
    """
    For a given dataframe, compute the max over
    a defined period of time (time_window) of a defined feature

    Parameters
    ----------
    X : dataframe
    features : list[str]
        features in the dataframe we wish to compute the rolling max from
    time_window : str
        string that defines the length of the time window passed to `rolling`
    center : bool
        boolean to indicate if the point of the dataframe considered is
        center or end of the window
    """
    for feature in features:
        name = '_'.join([feature, time_window, 'max'])
        X_df[name] = X_df[feature].rolling(time_window, center=center).max()
        X_df[name] = X_df[name].ffill().bfill()
        X_df[name] = X_df[name].astype(X_df[feature].dtype)
    return X_df


def compute_rolling_sum(X_df, features, time_window, center=False):
    """
    For a given dataframe, compute the sum over
    a defined period of time (time_window) of a defined feature

    Parameters
    ----------
    X : dataframe
    features : list[str]
        features in the dataframe we wish to compute the rolling sum from
    time_window : str
        string that defines the length of the time window passed to `rolling`
    center : bool
        boolean to indicate if the point of the dataframe considered is
        center or end of the window
    """
    for feature in features:
        name = '_'.join([feature, time_window, 'sum'])
        X_df[name] = X_df[feature].rolling(time_window, center=center).sum()
        X_df[name] = X_df[name].ffill().bfill()
        X_df[name] = X_df[name].astype(X_df[feature].dtype)
    return X_df
