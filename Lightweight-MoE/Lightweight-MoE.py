import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.mixture import GaussianMixture
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import accuracy_score
import optuna


class LightweightMOE:
    def __init__(self, data, target, n_clusters=3, models=None, test_size=0.2, cv_folds=5):
        """
        Initialize the Lightweight-MOE model
        :param data: Feature data (X)
        :param target: Target labels (y)
        :param n_clusters: Number of clusters for GMM
        :param models: List of selected models to use (e.g., ['svm', 'lightgbm', 'decision_tree'])
        :param test_size: Proportion of test data (default: 0.2)
        :param cv_folds: Number of folds for cross-validation (default: 5)
        """
        self.data = data
        self.target = target
        self.n_clusters = n_clusters
        self.models_dict = {
            'svm': SVC(probability=True, random_state=42),
            'lightgbm': lgb.LGBMClassifier(random_state=42),
            'xgboost': xgb.XGBClassifier(random_state=42),
            'random_forest': RandomForestClassifier(random_state=42),
            'decision_tree': DecisionTreeClassifier(random_state=42),
            'knn': KNeighborsClassifier(),
            'naive_bayes': GaussianNB(),
            'gradient_boosting': GradientBoostingClassifier(random_state=42),
            'adaboost': AdaBoostClassifier(random_state=42)
        }
        self.models = self._select_models(models)  # Select models based on input
        self.test_size = test_size
        self.cv_folds = cv_folds

    def _select_models(self, models=None):
        """
        Select the models to use based on the input list
        :param models: List of selected model names (e.g., ['svm', 'lightgbm'])
        :return: Dictionary of models to use
        """
        if models is None:
            return self.models_dict
        else:
            selected_models = {name: self.models_dict[name] for name in models if name in self.models_dict}
            return selected_models

    @staticmethod
    def train_test_split_data(data, target, test_size=0.2):
        """
        Split the data into training and testing sets
        :param data: Feature data (X)
        :param target: Target labels (y)
        :param test_size: Proportion of test data
        :return: X_train, X_test, y_train, y_test
        """
        return train_test_split(data, target, test_size=test_size, random_state=42)

    @staticmethod
    def perform_gmm_clustering(X_train, n_clusters=3):
        """
        Perform GMM clustering on the training data
        :param X_train: Training feature data
        :param n_clusters: Number of clusters for GMM
        :return: gmm model, cluster labels
        """
        gmm = GaussianMixture(n_components=n_clusters, random_state=42)
        cluster_labels = gmm.fit_predict(X_train)
        return gmm, cluster_labels

    def select_best_model(self, X_train, y_train, cluster_labels, optimize=False):
        """
        Select the best model for each cluster using cross-validation or hyperparameter optimization
        :param X_train: Training feature data
        :param y_train: Training target labels
        :param cluster_labels: Cluster labels from GMM
        :param optimize: Whether to perform hyperparameter optimization using Optuna
        :return: List of best models for each cluster
        """
        best_models = []
        for cluster in np.unique(cluster_labels):
            X_cluster = X_train[cluster_labels == cluster]
            y_cluster = y_train[cluster_labels == cluster]

            best_model = None
            best_score = -np.inf
            for name, model in self.models.items():
                if optimize:
                    # Perform hyperparameter optimization using Optuna
                    study = optuna.create_study(direction='maximize')
                    study.optimize(lambda trial: self.objective(trial, model, X_cluster, y_cluster), n_trials=50)
                    best_model = model.set_params(**study.best_params)
                    best_score = study.best_value
                else:
                    # Use cross-validation to select the best model
                    score = cross_val_score(model, X_cluster, y_cluster, cv=self.cv_folds, scoring='accuracy').mean()
                    if score > best_score:
                        best_score = score
                        best_model = model

            best_models.append(best_model.fit(X_cluster, y_cluster))
        return best_models

    @staticmethod
    def calculate_cluster_probabilities(gmm, X_test):
        """
        Calculate the probability of each test sample belonging to each cluster
        :param gmm: Trained GMM model
        :param X_test: Test feature data
        :return: Probabilities of each sample belonging to each cluster
        """
        return gmm.predict_proba(X_test)

    def weighted_prediction(self, X_test, best_models, cluster_probabilities):
        """
        Perform weighted prediction based on cluster probabilities
        :param X_test: Test feature data
        :param best_models: List of best models for each cluster
        :param cluster_probabilities: Probability of each sample belonging to each cluster
        :return: Weighted predictions
        """
        predictions = np.zeros((X_test.shape[0],))
        for i, model in enumerate(best_models):
            prob = cluster_probabilities[:, i]
            predictions += prob * model.predict(X_test)

        return np.round(predictions)

    def objective(self, trial, model, X_train, y_train):
        """
        Objective function for Optuna hyperparameter optimization
        :param trial: Optuna trial
        :param model: Model to optimize
        :param X_train: Training feature data
        :param y_train: Training target labels
        :return: Cross-validation score
        """
        if isinstance(model, SVC):
            C = trial.suggest_loguniform('C', 1e-5, 1e5)
            kernel = trial.suggest_categorical('kernel', ['linear', 'rbf', 'poly'])
            gamma = trial.suggest_loguniform('gamma', 1e-5, 1e5)
            model.set_params(C=C, kernel=kernel, gamma=gamma)
        elif isinstance(model, lgb.LGBMClassifier):
            num_leaves = trial.suggest_int('num_leaves', 20, 150)
            learning_rate = trial.suggest_loguniform('learning_rate', 1e-4, 1e-1)
            n_estimators = trial.suggest_int('n_estimators', 50, 200)
            model.set_params(num_leaves=num_leaves, learning_rate=learning_rate, n_estimators=n_estimators)
        elif isinstance(model, xgb.XGBClassifier):
            max_depth = trial.suggest_int('max_depth', 3, 10)
            learning_rate = trial.suggest_loguniform('learning_rate', 1e-4, 1e-1)
            n_estimators = trial.suggest_int('n_estimators', 50, 200)
            model.set_params(max_depth=max_depth, learning_rate=learning_rate, n_estimators=n_estimators)
        elif isinstance(model, RandomForestClassifier):
            n_estimators = trial.suggest_int('n_estimators', 50, 200)
            max_depth = trial.suggest_int('max_depth', 3, 10)
            model.set_params(n_estimators=n_estimators, max_depth=max_depth)
        elif isinstance(model, DecisionTreeClassifier):
            max_depth = trial.suggest_int('max_depth', 3, 10)
            model.set_params(max_depth=max_depth)
        elif isinstance(model, KNeighborsClassifier):
            n_neighbors = trial.suggest_int('n_neighbors', 1, 20)
            model.set_params(n_neighbors=n_neighbors)
        elif isinstance(model, GaussianNB):
            pass  # Naive Bayes does not require hyperparameter tuning
        elif isinstance(model, GradientBoostingClassifier):
            n_estimators = trial.suggest_int('n_estimators', 50, 200)
            learning_rate = trial.suggest_loguniform('learning_rate', 1e-4, 1e-1)
            max_depth = trial.suggest_int('max_depth', 3, 10)
            model.set_params(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth)
        elif isinstance(model, AdaBoostClassifier):
            n_estimators = trial.suggest_int('n_estimators', 50, 200)
            learning_rate = trial.suggest_loguniform('learning_rate', 1e-4, 1e-1)
            model.set_params(n_estimators=n_estimators, learning_rate=learning_rate)

        return cross_val_score(model, X_train, y_train, cv=self.cv_folds, scoring='accuracy').mean()

    def run(self, optimize=False):
        """
        Run the Lightweight-MOE algorithm
        :return: Final predictions
        """
        X_train, X_test, y_train, y_test = self.train_test_split_data(self.data, self.target, self.test_size)
        gmm, cluster_labels = self.perform_gmm_clustering(X_train, self.n_clusters)
        best_models = self.select_best_model(X_train, y_train, cluster_labels, optimize=optimize)
        cluster_probabilities = self.calculate_cluster_probabilities(gmm, X_test)
        predictions = self.weighted_prediction(X_test, best_models, cluster_probabilities)

        accuracy = accuracy_score(y_test, predictions)
        print(f'Accuracy: {accuracy:.4f}')

        return predictions


# Example usage:
# X = your_feature_data
# y = your_target_data

# Create the model and run with or without hyperparameter optimization
# You can select the models you want to use by specifying the list of model names.

selected_models = ['svm', 'lightgbm', 'decision_tree', 'knn', 'naive_bayes']  # Manually select models for comparison
lightweight_moe = LightweightMOE(X, y, n_clusters=3, test_size=0.2, cv_folds=5, models=selected_models)
predictions = lightweight_moe.run(optimize=True)  # Set optimize=True to enable hyperparameter optimization
