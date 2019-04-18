import os
import datetime
import pickle

import pandas as pd
import tmp2.services.statistics_service as stat

from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble
from sklearn.metrics import confusion_matrix
from tmp2.ml_techniques import ml_technique
from tmp2.nlp_techniques.nlp_technique import NLPTechnique


def check_dir():
    if not os.path.exists('artifacts/nlp_data/'):
        os.makedirs('artifacts/nlp_data/')
    if not os.path.exists('artifacts/sel_data/'):
        os.makedirs('artifacts/sel_data/')
    if not os.path.exists('artifacts/custom_data/'):
        os.makedirs('artifacts/custom_data/')
    if not os.path.exists('pickle/'):
        os.makedirs('pickle/')
    return


def calculate_statistics(input_path, output_path, features, forced=False):
    file = pd.ExcelFile(input_path)
    stat_data = file.parse('Balance', index_col='jobid')
    file.close()

    stat_nlp = read_instance_pickle('pickle/stat_nlp.pickle')
    if stat_nlp is None:
        stat_nlp = NLPTechnique(name='Stat NLP',
                                nlp_vader=True,
                                nlp_bow=True,
                                pp_lemma=True,
                                pp_pos=True,
                                pickle_file='pickle/stat_nlp.pickle')
    stat_data = stat_nlp.process_data(stat_data, forced=forced)
    attr = features + ['category']
    stat_data = stat_data[attr]
    # stat_data[features] = stat_data[features].apply(pd.to_numeric, axis='columns')
    # stat_data = stat_data.fillna(method='ffill')
    stat.bucketing(stat_data, output_path)

    svm_tech = read_instance_pickle('pickle/svm_tech.pickle')
    dt_tech = read_instance_pickle('pickle/dt_tech.pickle')
    log_reg_tech = read_instance_pickle('pickle/log_reg_tech.pickle')
    k_tech = read_instance_pickle('pickle/k_tech.pickle')
    nb_tech = read_instance_pickle('pickle/nb_tech.pickle')
    rf_tech = read_instance_pickle('pickle/rf_tech.pickle')

    if None in (svm_tech, dt_tech, log_reg_tech, k_tech, nb_tech, rf_tech):
        # Technique instances
        print("cannot find instances")
        svm_tech = ml_technique.MLTechnique(name='Support Vector Machine (w SA)',
                                            data_file='artifacts/sel_data/master.xlsx',
                                            model=svm.SVC(kernel='linear', C=1, gamma=1),
                                            pickle_file='pickle/svm_tech.pickle',
                                            features=features)
        dt_tech = ml_technique.MLTechnique(name='Decision Tree (w SA)',
                                           data_file='artifacts/sel_data/master.xlsx',
                                           model=tree.DecisionTreeClassifier(),
                                           pickle_file='pickle/dt_tech.pickle',
                                           features=features)
        log_reg_tech = ml_technique.MLTechnique(name='Logistic Regression (w SA)',
                                                data_file='artifacts/sel_data/master.xlsx',
                                                model=linear_model.LogisticRegression(),
                                                pickle_file='pickle/log_reg_tech.pickle',
                                                features=features)
        k_tech = ml_technique.MLTechnique(name='k-Nearest Neighbour (w SA)',
                                          data_file='artifacts/sel_data/master.xlsx',
                                          model=neighbors.KNeighborsClassifier(),
                                          pickle_file='pickle/k_tech.pickle',
                                          features=features)
        nb_tech = ml_technique.MLTechnique(name='Naive Bayes (w SA)',
                                           data_file='artifacts/sel_data/master.xlsx',
                                           model=naive_bayes.GaussianNB(),
                                           pickle_file='pickle/nb_tech.pickle',
                                           features=features)
        rf_tech = ml_technique.MLTechnique(name='Random Forest (w SA)',
                                           data_file='artifacts/sel_data/master.xlsx',
                                           model=ensemble.RandomForestClassifier(),
                                           pickle_file='pickle/rf_tech.pickle',
                                           features=features)

    # DUP
    elif getattr(svm_tech, 'features') != features:
        print('instance has different features')
        svm_tech = ml_technique.MLTechnique(name='Support Vector Machine (w SA)',
                                            data_file='artifacts/sel_data/master.xlsx',
                                            model=svm.SVC(kernel='linear', C=1, gamma=1),
                                            pickle_file='pickle/svm_tech.pickle',
                                            features=features)
        dt_tech = ml_technique.MLTechnique(name='Decision Tree (w SA)',
                                           data_file='artifacts/sel_data/master.xlsx',
                                           model=tree.DecisionTreeClassifier(),
                                           pickle_file='pickle/dt_tech.pickle',
                                           features=features)
        log_reg_tech = ml_technique.MLTechnique(name='Logistic Regression (w SA)',
                                                data_file='artifacts/sel_data/master.xlsx',
                                                model=linear_model.LogisticRegression(),
                                                pickle_file='pickle/log_reg_tech.pickle',
                                                features=features)
        k_tech = ml_technique.MLTechnique(name='k-Nearest Neighbour (w SA)',
                                          data_file='artifacts/sel_data/master.xlsx',
                                          model=neighbors.KNeighborsClassifier(),
                                          pickle_file='pickle/k_tech.pickle',
                                          features=features)
        nb_tech = ml_technique.MLTechnique(name='Naive Bayes (w SA)',
                                           data_file='artifacts/sel_data/master.xlsx',
                                           model=naive_bayes.GaussianNB(),
                                           pickle_file='pickle/nb_tech.pickle',
                                           features=features)
        rf_tech = ml_technique.MLTechnique(name='Random Forest (w SA)',
                                           data_file='artifacts/sel_data/master.xlsx',
                                           model=ensemble.RandomForestClassifier(),
                                           pickle_file='pickle/rf_tech.pickle',
                                           features=features)

    # Calculate statistics
    svm_a = stat.get_accuracy(svm_tech, forced=forced)
    dt_a = stat.get_accuracy(dt_tech, forced=forced)
    log_reg_a = stat.get_accuracy(log_reg_tech, forced=forced)
    k_a = stat.get_accuracy(k_tech, forced=forced)
    nb_a = stat.get_accuracy(nb_tech, forced=forced)
    rf_a = stat.get_accuracy(rf_tech, forced=forced)

    svm_f = stat.get_f1_score(svm_tech, forced=forced)
    dt_f = stat.get_f1_score(dt_tech, forced=forced)
    log_reg_f = stat.get_f1_score(log_reg_tech, forced=forced)
    k_f = stat.get_f1_score(k_tech, forced=forced)
    nb_f = stat.get_f1_score(nb_tech, forced=forced)
    rf_f = stat.get_f1_score(rf_tech, forced=forced)

    # Add time stamp to the result
    tt = datetime.datetime.now().strftime('%H:%M %d/%m/%Y')

    print("prediction done")
    return (
        {
            'svm': svm_a,
            'dt': dt_a,
            'log_reg': log_reg_a,
            'k': k_a,
            'nb': nb_a,
            'rf': rf_a
        },
        {
            'svm': svm_f,
            'dt': dt_f,
            'log_reg': log_reg_f,
            'k': k_f,
            'nb': nb_f,
            'rf': rf_f
        },
        tt
    )


def read_instance_pickle(path):
    if os.path.isfile(path):
        with open(path, "rb") as file:
            try:
                ml = pickle.load(file)
                # print("Load saved instance.")
                return ml
            except (OSError, IOError):
                pass
    return None


# DUP
def update_or_create_pickle(pickle_file, instance):
    if pickle_file:
        with open(pickle_file, "wb") as file:
            pickle.dump(instance, file)


def ml_analysis():
    category_column = 'category'
    category_map = {
        "Detractor": 0,
        "Passive": 1,
        "Promoter": 2
    }

    # Get the saved ML techniques
    svm_tech = read_instance_pickle('pickle/svm_tech.pickle')
    dt_tech = read_instance_pickle('pickle/dt_tech.pickle')
    log_reg_tech = read_instance_pickle('pickle/log_reg_tech.pickle')
    k_tech = read_instance_pickle('pickle/k_tech.pickle')
    nb_tech = read_instance_pickle('pickle/nb_tech.pickle')
    rf_tech = read_instance_pickle('pickle/rf_tech.pickle')

    # If the ML technique instances does not exist, create new instances
    if None in (svm_tech, dt_tech, log_reg_tech, k_tech, nb_tech, rf_tech):
        # Technique instances
        print("cannot find instances")
        features = ['neu', 'neg', 'pos', 'compound', 'bag_vector', 'on_time_in_full', 'deliveryday', 'region']
        svm_tech = ml_technique.MLTechnique(name='Support Vector Machine (w SA)',
                                            data_file='artifacts/sel_data/master.xlsx',
                                            model=svm.SVC(kernel='linear', C=1, gamma=1),
                                            pickle_file='pickle/svm_tech.pickle',
                                            features=features)
        dt_tech = ml_technique.MLTechnique(name='Decision Tree (w SA)',
                                           data_file='artifacts/sel_data/master.xlsx',
                                           model=tree.DecisionTreeClassifier(),
                                           pickle_file='pickle/dt_tech.pickle',
                                           features=features)
        log_reg_tech = ml_technique.MLTechnique(name='Logistic Regression (w SA)',
                                                data_file='artifacts/sel_data/master.xlsx',
                                                model=linear_model.LogisticRegression(),
                                                pickle_file='pickle/log_reg_tech.pickle',
                                                features=features)
        k_tech = ml_technique.MLTechnique(name='k-Nearest Neighbour (w SA)',
                                          data_file='artifacts/sel_data/master.xlsx',
                                          model=neighbors.KNeighborsClassifier(),
                                          pickle_file='pickle/k_tech.pickle',
                                          features=features)
        nb_tech = ml_technique.MLTechnique(name='Naive Bayes (w SA)',
                                           data_file='artifacts/sel_data/master.xlsx',
                                           model=naive_bayes.GaussianNB(),
                                           pickle_file='pickle/nb_tech.pickle',
                                           features=features)
        rf_tech = ml_technique.MLTechnique(name='Random Forest (w SA)',
                                           data_file='artifacts/sel_data/master.xlsx',
                                           model=ensemble.RandomForestClassifier(),
                                           pickle_file='pickle/rf_tech.pickle',
                                           features=features)

    # Cross Validation
    svm_cv = svm_tech.cross_validate(cat_col=category_column, cat_map=category_map)
    dt_cv = dt_tech.cross_validate(cat_col=category_column, cat_map=category_map)
    log_reg_cv = log_reg_tech.cross_validate(cat_col=category_column, cat_map=category_map)
    k_cv = k_tech.cross_validate(cat_col=category_column, cat_map=category_map)
    nb_cv = nb_tech.cross_validate(cat_col=category_column, cat_map=category_map)
    rf_cv = rf_tech.cross_validate(cat_col=category_column, cat_map=category_map)

    # Local "packaging/parsing" of cross validation results to be rendered
    def analyze_result(name, cv_result):
        cm = confusion_matrix(cv_result['true_array'], cv_result['predict_array'])

        return {
            'name': name,
            'labels': cv_result['categories'],
            'matrix': cm,
            'analysis': {
                'input_size': cv_result['total_size']
            }
        }

    return {
        'techniques': ['svm', 'dt', 'log_reg', 'k', 'nb', 'rf'],
        'results': {
            'svm': analyze_result(svm_tech.get_name(), svm_cv),
            'dt': analyze_result(dt_tech.get_name(), dt_cv),
            'log_reg': analyze_result(log_reg_tech.get_name(), log_reg_cv),
            'k': analyze_result(k_tech.get_name(), k_cv),
            'nb': analyze_result(nb_tech.get_name(), nb_cv),
            'rf': analyze_result(rf_tech.get_name(), rf_cv)
        }
    }


def predict(comment, delivery_ok, deliveryday, region):
    # parse the input into excel file
    result_id = str(datetime.datetime.now().time()).replace(".", "").replace(":", "")
    path = 'artifacts/custom_data/' + result_id + '.xlsx'

    # excel_writer = pd.ExcelWriter(path=path, engine='openpyxl')
    data_frame = pd.DataFrame({'jobid': [result_id],
                               'comments': [comment],
                               'on_time_in_full': [delivery_ok],
                               'deliveryday': [deliveryday],
                               'region': [region]})
    print('Predict: ' + str(data_frame))
    data_frame.set_index('jobid', inplace=True)

    pred_nlp = read_instance_pickle('pickle/pred_nlp.pickle')
    if pred_nlp is None:
        # Error: No Pred_NLP pickle file
        raise FileNotFoundError

    pred_ml = read_instance_pickle('pickle/pred_ml.pickle')
    if pred_ml is None:
        # Error: No Pred_NLP pickle file
        raise FileNotFoundError

    # process the input excel
    data_frame = pred_nlp.process_data(data_frame, forced=True, predict=True)
    excel_writer = pd.ExcelWriter(path, engine='openpyxl')
    data_frame.to_excel(excel_writer, sheet_name='Balance')
    excel_writer.save()
    excel_writer.close()

    [prediction] = pred_ml.predict_single(path)
    # parse the result
    result_map = {
        0: "Detractor",
        1: "Passive",
        2: "Promoter"
    }

    print("Return prediction.")
    return result_map.get(prediction)


def predict_batch(data_frame):
    data_frame.index.name = 'jobid'
    result_id = str(datetime.datetime.now().time()).replace(".", "").replace(":", "")
    path = 'artifacts/custom_data/' + result_id + '.xlsx'

    pred = read_instance_pickle('pickle/pred_nlp.pickle')
    if pred is None:
        # Error: No Pred_NLP pickle file
        raise FileNotFoundError

    pred_ml = read_instance_pickle('pickle/pred_ml.pickle')
    if pred_ml is None:
        # Error: No Pred_NLP pickle file
        raise FileNotFoundError

    # process the input excel
    print("beginning batch processing")
    data_frame = pred.process_data(data_frame, forced=True, predict=True)
    print("processed data")
    excel_writer = pd.ExcelWriter(path, engine='openpyxl')
    data_frame.to_excel(excel_writer, sheet_name='Balance')
    print("wrote to excel")
    excel_writer.close()

    predictions = pred_ml.predict_single(path)

    data_frame['predicted_category'] = predictions
    result_map = {
        0: "Detractor",
        1: "Passive",
        2: "Promoter"
    }
    data_frame['predicted_category'] = data_frame['predicted_category'].map(result_map)
    data_frame.to_excel(excel_writer, sheet_name='Balance')
    print("wrote to excel")
    excel_writer.close()

    return path


def train_pred_ml(nlp, ml_tech):
    """

    :param nlp: {'pos','lemma', 'bow', 'vader'}
    :param ml_tech: ML type to use for prediction
    :return: Trained ML to use for prediction
    """
    pos_string = '1' if nlp['pos'] else '0'
    lemma_string = '1' if nlp['lemma'] else '0'
    bow_string = '1' if nlp['bow'] else '0'
    vader_string = '1' if nlp['vader'] else '0'
    nlp_pickle = 'pickle/NLP/pred_' + pos_string + '_' + lemma_string + '_' + bow_string + '_' + vader_string

    # To check if the NLP techniques changed from last time
    # pred_nlp = read_instance_pickle('pickle/pred_nlp.pickle')
    # forced = False
    # if pred_nlp is not None:
    #     if getattr(pred_nlp, "pp_pos") != pos or getattr(pred_nlp, "pp_lemma") != lemma or \
    #                     getattr(pred_nlp, "nlp_bow") != bow:
    #         forced = True

    # Selecting features
    features = ['region', 'on_time_in_full', 'deliveryday']
    if nlp['vader']:
        features.extend(['neu', 'neg', 'pos', 'compound'])
    if nlp['bow']:
        features.extend(['bag_vector'])

    forced = False
    # To check if the ML technique changed from last time
    pred_ml = read_instance_pickle('pickle/pred_ml.pickle')
    if pred_ml is not None and not forced:
        if getattr(pred_ml, "ml_code") != ml_tech or getattr(pred_ml, "nlp") != nlp:
            forced = True

    # init selected NLP and ML tech instances for prediction
    # create directories for calculation, if directories not exist
    check_dir()
    # manage IO
    raw_path = 'artifacts/raw/master.xlsx'
    training_file = pd.ExcelFile(raw_path)
    data = training_file.parse('Balance', index_col='jobid')
    training_file.close()
    # pred_writer = pd.ExcelWriter(nlp_path, engine='openpyxl')

    # init NLP instance
    pred_nlp = read_instance_pickle(nlp_pickle)
    if pred_nlp is None:
        pred_nlp = NLPTechnique(name='Prediction NLP',
                                nlp_vader=nlp['vader'],
                                nlp_bow=nlp['bow'],
                                pp_lemma=nlp['lemma'],
                                pp_pos=nlp['pos'],
                                pickle_file=nlp_pickle)

    data = pred_nlp.process_data(data)
    # data.to_excel(pred_writer, 'Balance')
    # pred_writer.save()
    # pred_writer.close()
    update_or_create_pickle('pickle/pred_nlp.pickle', pred_nlp)

    # init ML tech
    pred_ml = read_instance_pickle('pickle/pred_ml.pickle')
    if pred_ml is None or forced:
        pred_ml = ml_technique_builder(ml_tech, features, nlp, data, pickle_file='pickle/pred_ml.pickle')

    pred_column = 'category'
    pred_map = {
        "Detractor": 0,
        "Passive": 1,
        "Promoter": 2
    }

    pred_ml.train(cat_col=pred_column,
                  cat_map=pred_map, forced=forced)
    return pred_ml


def ml_technique_builder(ml_tech, features, nlp, data_frame, pickle_file='', data_file=''):
    def svm_tech():
        return ml_technique.MLTechnique(name='SVM Predictor', data_file=data_file,
                                        model=svm.SVC(kernel='linear', C=1, gamma=1),
                                        features=features,
                                        ml_code='svm',
                                        pickle_file=pickle_file,
                                        nlp=nlp,
                                        data_frame=data_frame)

    def d_tree():
        return ml_technique.MLTechnique(name='Decision Tree Predictor', data_file=data_file,
                                        model=tree.DecisionTreeClassifier(),
                                        features=features,
                                        ml_code='d_tree',
                                        pickle_file=pickle_file,
                                        nlp=nlp,
                                        data_frame=data_frame)

    def log_reg():
        return ml_technique.MLTechnique(name='Logistic Regression Predictor', data_file=data_file,
                                        model=linear_model.LogisticRegression(),
                                        features=features,
                                        ml_code='log_reg',
                                        pickle_file=pickle_file,
                                        nlp=nlp,
                                        data_frame=data_frame)

    def k_neighbours():
        return ml_technique.MLTechnique(name='k-Nearest Neighbours Predictor', data_file=data_file,
                                        model=neighbors.KNeighborsClassifier(),
                                        features=features,
                                        ml_code='k',
                                        pickle_file=pickle_file,
                                        nlp=nlp,
                                        data_frame=data_frame)

    def naive():
        return ml_technique.MLTechnique(name='Naive Bayes Predictor', data_file=data_file,
                                        model=naive_bayes.GaussianNB(),
                                        features=features,
                                        ml_code='naive',
                                        pickle_file=pickle_file,
                                        nlp=nlp,
                                        data_frame=data_frame)

    def random_forest():
        return ml_technique.MLTechnique(name='Random Forest Predictor', data_file=data_file,
                                        model=ensemble.RandomForestClassifier(),
                                        features=features,
                                        ml_code='random',
                                        pickle_file=pickle_file,
                                        nlp=nlp,
                                        data_frame=data_frame)

    model ={
        'svm': svm_tech,
        'd_tree': d_tree,
        'log_reg': log_reg,
        'k': k_neighbours,
        'naive': naive,
        'random': random_forest
    }
    return model[ml_tech]()


train_pred_ml(nlp={'pos': True, 'lemma': True, 'bow': True, 'vader': True}, ml_tech='random')
