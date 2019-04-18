import numpy
import pandas
import os
from pyramid.response import Response, FileResponse
from pyramid.view import view_config
from tmp2.services.pipeline_service import calculate_statistics, ml_analysis, predict, read_instance_pickle, \
    predict_batch, train_pred_ml


@view_config(route_name='home', renderer='templates/home.jinja2')
def my_view(request):
    return {'project': 'NPS analyser'}


@view_config(route_name='score', renderer='templates/score.jinja2')
def score_view(request):
    if request.method == 'POST':
        data = request.get_array(field_name='excel')
        data = numpy.array(data)
        data[data == ''] = 0
        df = pandas.DataFrame(data=data[1:, 1:],
                              index=data[1:, 0],
                              columns=data[0, 1:])

        path = os.path.abspath(predict_batch(df))
        # return {'upload_status': 'uploaded'}
        return FileResponse(path)
    return {}


@view_config(route_name='result', renderer='json')
def result_view(request):
    comment = request.params.get('comment', None)
    delivery_ok = request.params.get('delivery_ok', None)
    deliveryday = request.params.get('deliveryday', None)
    region = request.params.get('region', None)

    if None in (comment, delivery_ok, deliveryday, region):
        return {'result': "cannot parse comment, please retry"}
    else:
        return {'result': predict(comment, delivery_ok, deliveryday, region)}


@view_config(route_name='option', renderer='json')
def option_view(request):
    pos = request.params.get('pos', None) == 'true'
    lemma = request.params.get('lemma', None) == 'true'
    bow = request.params.get('bag', None) == 'true'
    vader = request.params.get('vader', None) == 'true'
    ml_tech = request.params.get('ml_option', None)

    print("Part-of-speech: " + str(pos))
    print("Lemma: " + str(lemma))
    print("Bag-of-words: " + str(bow))
    print("Sentiment Analysis: " + str(vader))
    print("ML Tech: " + str(ml_tech))

    nlp = {
        'pos': pos,
        'lemma': lemma,
        'bow': bow,
        'vader': vader
    }

    train_pred_ml(nlp, ml_tech)
    return


# begin of statistics page
@view_config(route_name='statistics', renderer='templates/statistics.jinja2')
def stat_view(request):
    features = request.params.get('features', 'neu neg pos compound bag_vector on_time_in_full deliveryday region ')
    features = features.split()
    (accuracy, f1_score, time) = calculate_statistics(input_path='artifacts/raw/master.xlsx',
                                                      output_path='artifacts/sel_data/master.xlsx',
                                                      features=features)

    return {'svm_a': str(round(accuracy['svm'] * 100, 2)) + '%',
            'dtree_a': str(round(accuracy['dt'] * 100, 2)) + '%',
            'log_reg_a': str(round(accuracy['log_reg'] * 100, 2)) + '%',
            'k_a': str(round(accuracy['k'] * 100, 2)) + '%',
            'nb_a': str(round(accuracy['nb'] * 100, 2)) + '%',
            'rf_a': str(round(accuracy['rf'] * 100, 2)) + '%',
            'svm_f': str(round(f1_score['svm'], 4)),
            'dtree_f': str(round(f1_score['dt'], 4)),
            'log_reg_f': str(round(f1_score['log_reg'], 4)),
            'k_f': str(round(f1_score['k'], 4)),
            'nb_f': str(round(f1_score['nb'], 4)),
            'rf_f': str(round(f1_score['rf'], 4)),
            'time_stamp': time
            }


@view_config(route_name='re_statistics', renderer='json')
def restat_view(request):
    return stat_view(request)


# begin of analysis page
@view_config(route_name='analysis', renderer='templates/analysis.jinja2')
def analysis_view(request):
    stat_nlp = read_instance_pickle('pickle/stat_nlp.pickle')
    if stat_nlp is None:
        features = ['neu', 'neg', 'pos', 'compound', 'bag_vector', 'on_time_in_full', 'deliveryday', 'region']
        calculate_statistics(input_path='artifacts/raw/master.xlsx', output_path='artifacts/sel_data/master.xlsx',
                             features=features)

    return ml_analysis()

# end of analysis page
