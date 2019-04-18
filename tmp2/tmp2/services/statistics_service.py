import pandas as pd
from sklearn.metrics import accuracy_score, f1_score


def bucketing(data, output_path, bucket_number=5):
    """
    bucketing - Splits a dataset into 5 buckets
    """

    categories = data.groupby('category', as_index=False)

    writer = pd.ExcelWriter(output_path, engine='openpyxl')
    for i in range(0, bucket_number):
        frames = []
        factor = float(1 / bucket_number)
        # Split by category
        for _, group in categories:
            # Random sampling without replacement
            # This is to ensure we do not re-pick the same sample in different buckets
            frames.append(group.sample(frac=factor, replace=False))

        # Write to sheets
        bucket = pd.DataFrame(pd.concat(frames))
        bucket.to_excel(writer, sheet_name='Set' + str(i + 1))

    writer.save()
    writer.close()
    print("data bucketing done")
    return


def get_accuracy(ml_tech, set_number=5, forced=False):
    # arguments for ml_technique
    category_column = 'category'
    category_map = {
        "Detractor": 0,
        "Passive": 1,
        "Promoter": 2
    }

    # Cross Validation
    ml_cv = ml_tech.cross_validate(cat_col=category_column, cat_map=category_map, buckets=set_number, forced=forced)

    return accuracy_score(ml_cv['true_array'], ml_cv['predict_array'])


def get_f1_score(ml_tech, set_number=5, forced=False):
    # arguments for ml_technique
    category_column = 'category'
    category_map = {
        "Detractor": 0,
        "Passive": 1,
        "Promoter": 2
    }

    # Cross Validation
    ml_cv = ml_tech.cross_validate(cat_col=category_column, cat_map=category_map, buckets=set_number, forced=forced)

    return f1_score(ml_cv['true_array'], ml_cv['predict_array'], average='weighted')