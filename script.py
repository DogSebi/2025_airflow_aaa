import os
from datetime import timedelta
import polars as pl
import pandas as pd
import argparse

from scipy.sparse import csr_matrix
import numpy as np
import implicit
import mlflow

from itertools import product
from catboost import CatBoostRanker, Pool

EXPERIMENT_NAME = "homework-aakhasametdinov"
EVAL_DAYS_TRESHOLD = 14
DATA_DIR = 'data/'


def get_parameters():
    parser = argparse.ArgumentParser(description="Запуск эксперимента")
    parser.add_argument("--experiment_name", type=str, default="homework-aakhasametdinov", help="Название эксперимента")
    parser.add_argument("--run_name", type=str, default="als_with_catboost", help="Название рана")
    parser.add_argument("--model_name", type=str, default="als_with_catboost", help="Название модели")
    parser.add_argument("--als_factors", type=int, default=512, help="Количество als_factors")
    parser.add_argument("--als_iterations", type=int, default=4, help="Количество als_iterations")
    parser.add_argument("--als_regularization", type=float, default=0.6, help="Параметр als_regularization")
    parser.add_argument("--als_alpha", type=int, default=2, help="Параметр als_alpha")
    parser.add_argument("--cb_loss_function", type=str, default="PairLogit", help="Функция потерь Catboost")
    parser.add_argument("--cb_iterations", type=int, default=700, help="Количество catboost_iterations")
    parser.add_argument("--cb_learning_rate", type=float, default=0.05, help="Скорость обучения catboost")
    parser.add_argument("--cb_depth", type=int, default=6, help="Глубина дерева catboost")
    parser.add_argument("--cb_l2_leaf_reg", type=int, default=10, help="Регуляризация catboost")
    parameters = parser.parse_args()
    return parameters

PARAMETERS = get_parameters()
EXPERIMENT_NAME = PARAMETERS.experiment_name


def get_data():
    df_test_users = pl.read_parquet(f'{DATA_DIR}/test_users.pq')
    df_clickstream = pl.read_parquet(f'{DATA_DIR}/clickstream.pq')
    df_event = pl.read_parquet(f'{DATA_DIR}/events.pq')
    return df_test_users, df_clickstream, df_event


def split_train_test(df_clickstream: pl.DataFrame, df_event: pl.DataFrame):
    treshhold = df_clickstream['event_date'].max() - timedelta(days=EVAL_DAYS_TRESHOLD)

    df_train = df_clickstream.filter(df_clickstream['event_date'] <= treshhold)
    df_eval = df_clickstream.filter(df_clickstream['event_date'] > treshhold)[['cookie', 'node', 'event']]

    df_eval = df_eval.join(df_train, on=['cookie', 'node'], how='anti')

    df_eval = df_eval.filter(
        pl.col('event').is_in(
            df_event.filter(pl.col('is_contact') == 1)['event'].unique()
        )
    )
    df_eval = df_eval.filter(
        pl.col('cookie').is_in(df_train['cookie'].unique())
    ).filter(
        pl.col('node').is_in(df_train['node'].unique())
    )

    df_eval = df_eval.unique(['cookie', 'node'])

    return df_train, df_eval

# split train test for catboost
def catboost_train_test_split(df_train: pl.DataFrame, df_event: pl.DataFrame):
    treshhold = df_train['event_date'].max() - timedelta(days=EVAL_DAYS_TRESHOLD)
    df_train_catboost = df_train.filter(df_train['event_date']<= treshhold)
    df_eval_catboost = df_train.filter(df_train['event_date']> treshhold)[['cookie', 'node', 'event']]
    df_eval_catboost = df_eval_catboost.join(df_train_catboost, on=['cookie', 'node'], how='anti')

    df_eval_catboost = df_eval_catboost.filter(
        pl.col('event').is_in(
            df_event.filter(pl.col('is_contact')==1)['event'].unique()
        )
    )
    df_eval_catboost = df_eval_catboost.filter(
            pl.col('cookie').is_in(df_train_catboost['cookie'].unique())
        ).filter(
            pl.col('node').is_in(df_train_catboost['node'].unique())
        )
    df_eval_catboost = df_eval_catboost.unique(['cookie', 'node'])

    return df_train_catboost, df_eval_catboost


def get_als_pred(users, nodes, user_to_pred):
    user_ids = users.unique().to_list()
    item_ids = nodes.unique().to_list()

    user_id_to_index = {user_id: idx for idx, user_id in enumerate(user_ids)}
    item_id_to_index = {item_id: idx for idx, item_id in enumerate(item_ids)}
    index_to_item_id = {v: k for k, v in item_id_to_index.items()}

    rows = users.replace_strict(user_id_to_index).to_list()
    cols = nodes.replace_strict(item_id_to_index).to_list()

    values = [1] * len(users)

    sparse_matrix = csr_matrix((values, (rows, cols)), shape=(len(user_ids), len(item_ids)))

    model = implicit.als.AlternatingLeastSquares(
        iterations=PARAMETERS.als_iterations,
        factors=PARAMETERS.als_factors,
        alpha=PARAMETERS.als_alpha,
        regularization=PARAMETERS.als_regularization,
        use_gpu=False,
    )
    model.fit(sparse_matrix)

    user4pred = np.array([user_id_to_index[i] for i in user_to_pred])

    recommendations, scores = model.recommend(
        user4pred,
        sparse_matrix[user4pred],
        N=300,
        filter_already_liked_items=True,
    )

    df_pred = pl.DataFrame(
        {
            'node': [
                [index_to_item_id[i] for i in i] for i in recommendations.tolist()
            ],
            'cookie': list(user_to_pred),
            'scores': scores.tolist()

        }
    )
    df_pred = df_pred.explode(['node', 'scores'])
    return df_pred


def train_als(df_train: pl.DataFrame, df_eval: pl.DataFrame):
    users = df_train["cookie"]
    nodes = df_train["node"]
    eval_users = df_eval['cookie'].unique().to_list()
    df_pred = get_als_pred(users, nodes, eval_users)
    return df_pred


def add_target(df_pred: pl.DataFrame, df_eval: pl.DataFrame, df_event: pl.DataFrame):
    contact_event_ids = (
        df_event.filter(pl.col("is_contact") == 1)
                .select("event")
                .to_series()
                .to_list()
    )

    df_eval_contacts = df_eval.filter(pl.col("event").is_in(contact_event_ids))
    eval_true = df_eval_contacts.select(["cookie", "node"]).unique()

    df_pred = df_pred.join(
        eval_true.with_columns(pl.lit(1).alias("target")),
        on=["cookie", "node"],
        how="left"
    ).with_columns(
        pl.col("target").fill_null(0).cast(pl.Int8)
    )

    return df_pred


def make_user_features(df_pred:pl.DataFrame, df_train: pl.DataFrame):
    user_stats = (
        df_train.group_by("cookie")
                .agg([
                    pl.count().alias("user_total_clicks"),
                    pl.col("node").n_unique().alias("user_unique_nodes")
                ])
    )
    df_pred = df_pred.join(user_stats, on="cookie", how="left")

    node_stats = (
        df_train.group_by("node")
                .agg([
                    pl.count().alias("node_total_clicks"),
                    pl.col("cookie").n_unique().alias("node_unique_users")
                ])
    )
    df_pred = df_pred.join(node_stats, on="node", how="left")

    return df_pred


def make_node_features(df_pred: pl.DataFrame, df_train: pl.DataFrame, df_event: pl.DataFrame):
    contact_event_ids = (
        df_event.filter(pl.col("is_contact") == 1)
                .select("event")
                .to_series()
                .to_list()
    )

    target_users = df_pred.select("cookie").unique()
    target_nodes = df_pred.select("node").unique()

    df_train_small = (
        df_train.join(target_users, on="cookie", how="inner")
                .join(target_nodes, on="node", how="inner")
    )

    node_contacts = (
        df_train_small
        .filter(pl.col("event").is_in(contact_event_ids))
        .group_by("node")
        .agg(pl.len().alias("node_contacts"))
    )

    df_pred = df_pred.join(node_contacts, on="node", how="left").with_columns(
        (pl.col("node_contacts") / pl.col("node_total_clicks")).alias("node_contact_rate")
    ).drop("node_contacts")

    return df_pred


def make_catboost_features(
        df_pred: pl.DataFrame,
        df_eval: pl.DataFrame,
        df_event: pl.DataFrame,
        df_train: pl.DataFrame):

    df_pred = add_target(df_pred, df_eval, df_event)
    df_pred = make_user_features(df_pred, df_train)
    df_pred = make_node_features(df_pred, df_train, df_event)

    return df_pred


def train_catboost(df_pred: pl.DataFrame, df_pred_test: pl.DataFrame):
    df = df_pred.to_pandas()
    df = df.sort_values("cookie").reset_index(drop=True)

    y = df["target"]
    X = df.drop(columns=["target"])

    group_ids = df["cookie"]
    train_pool = Pool(
        data=X,
        label=y,
        group_id=group_ids
    )

    df_val = df_pred_test.to_pandas()
    df_val = df_val.sort_values("cookie").reset_index(drop=True)
    valid_pool = Pool(data=df_val.drop(columns=["target"]), label=df_val['target'], group_id=df_val["cookie"])

    model = CatBoostRanker(
            loss_function=PARAMETERS.cb_loss_function,
            iterations=PARAMETERS.cb_iterations,
            learning_rate=PARAMETERS.cb_learning_rate,
            depth=PARAMETERS.cb_depth,
            l2_leaf_reg=PARAMETERS.cb_l2_leaf_reg,
            early_stopping_rounds=100,
            verbose=50,
            random_seed=42
        )

    model.fit(train_pool, eval_set=valid_pool)
    df_test = df_pred_test.to_pandas()
    df_test["pred"] = model.predict(df_test)

    df_topk = (
        df_test.sort_values(["cookie", "pred"], ascending=[True, False])
                .groupby("cookie")
                .head(40)
    )
    df_topk = pl.from_pandas(df_topk)

    return df_topk


def recall_at(df_true, df_pred, k=40):
    return df_true[['node', 'cookie']].join(
        df_pred.group_by('cookie').head(k).with_columns(value=1)[['node', 'cookie', 'value']],
        how='left',
        on=['cookie', 'node']
        ).select(
            [pl.col('value').fill_null(0), 'cookie']
        ).group_by(
            'cookie'
        ).agg(
            [
                pl.col('value').sum() / pl.col(
                    'value'
                ).count()
            ]
        )['value'].mean()


def main():
    mlflow.set_tracking_uri(
        os.environ.get('MLFLOW_TRACKING_URI')
    )
    print(1)

    if not mlflow.get_experiment_by_name(EXPERIMENT_NAME):
        mlflow.create_experiment(EXPERIMENT_NAME, artifact_location='mlflow-artifacts:/')

    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name=PARAMETERS.run_name):
        mlflow.log_param("model_name", PARAMETERS.model_name)
        mlflow.log_param("candidates", 300)
        mlflow.log_param("factors", PARAMETERS.als_factors)
        mlflow.log_param("iterations", PARAMETERS.als_iterations)
        mlflow.log_param("regularization", PARAMETERS.als_regularization)
        mlflow.log_param("alpha", PARAMETERS.als_alpha)
        mlflow.log_param("loss_function", PARAMETERS.cb_loss_function)
        mlflow.log_param("catboost_iterations", PARAMETERS.cb_iterations)
        mlflow.log_param("learning_rate", PARAMETERS.cb_learning_rate)
        mlflow.log_param("depth", PARAMETERS.cb_depth)
        mlflow.log_param("l2_leaf_reg", PARAMETERS.cb_l2_leaf_reg)
        print(2)
        df_test_users, df_clickstream, df_event = get_data()
        print(len(df_clickstream))
        df_train, df_eval = split_train_test(df_clickstream, df_event)
        print(4)
        df_train_catboost, df_eval_catboost = catboost_train_test_split(df_train, df_event)
        print(5)
        df_pred = train_als(df_train_catboost, df_eval_catboost)
        df_pred_test = train_als(df_train, df_eval)
        print(2)

        df_pred = make_catboost_features(
            df_pred,
            df_eval_catboost,
            df_event,
            df_train_catboost
        )

        df_pred_test = make_catboost_features(
            df_pred_test,
            df_eval,
            df_event,
            df_train
        )

        df_pred = train_catboost(df_pred, df_pred_test)

        metric = recall_at(df_eval, df_pred, k=40)

        mlflow.log_metric('Recall_40', metric)


if __name__ == '__main__':
    try:
        print('start')
        main()
    except Exception as e:
        import traceback
        print("❌ Произошла ошибка:")
        print(traceback.format_exc())
