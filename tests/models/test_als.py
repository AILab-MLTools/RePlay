# pylint: disable-all
import pytest
import numpy as np

from pyspark.sql import functions as sf

from replay.models import ALSWrap, AssociationRulesItemRec
from replay.models.base_rec import HybridRecommender, UserRecommender
from replay.utils.spark_utils import join_or_return, array_mult, horizontal_explode
from replay.utils.model_handler import save, load
from tests.utils import (
    spark,
    log,
    log_to_pred,
    long_log_with_features,
    user_features,
    sparkDataFrameEqual,
)


SEED = 123


def fit_predict_selected(model, train_log, inf_log, user_features, users):
    kwargs = {}
    if isinstance(model, (HybridRecommender, UserRecommender)):
        kwargs = {"user_features": user_features}
    model.fit(train_log, **kwargs)
    return model.predict(log=inf_log, users=users, k=1, **kwargs)


@pytest.fixture
def model():
    model = ALSWrap(2, implicit_prefs=False)
    model._seed = 42
    return model


def get_first_level_model_features(model, pairs, user_features=None, item_features=None, add_factors_mult=True, prefix=""):
    users = pairs.select("user_idx").distinct()
    items = pairs.select("item_idx").distinct()
    user_factors, user_vector_len = model._get_features_wrap(
        users, user_features
    )
    item_factors, item_vector_len = model._get_features_wrap(
        items, item_features
    )

    pairs_with_features = join_or_return(
        pairs, user_factors, how="left", on="user_idx"
    )
    pairs_with_features = join_or_return(
        pairs_with_features,
        item_factors,
        how="left",
        on="item_idx",
    )

    factors_to_explode = []
    if user_factors is not None:
        pairs_with_features = pairs_with_features.withColumn(
            "user_factors",
            sf.coalesce(
                sf.col("user_factors"),
                sf.array([sf.lit(0.0)] * user_vector_len),
            ),
        )
        factors_to_explode.append(("user_factors", "uf"))

    if item_factors is not None:
        pairs_with_features = pairs_with_features.withColumn(
            "item_factors",
            sf.coalesce(
                sf.col("item_factors"),
                sf.array([sf.lit(0.0)] * item_vector_len),
            ),
        )
        factors_to_explode.append(("item_factors", "if"))

    if model.__str__() == "LightFMWrap":
        pairs_with_features = (
            pairs_with_features.fillna({"user_bias": 0, "item_bias": 0})
            .withColumnRenamed("user_bias", f"{prefix}_user_bias")
            .withColumnRenamed("item_bias", f"{prefix}_item_bias")
        )

    if (
        add_factors_mult
        and user_factors is not None
        and item_factors is not None
    ):
        pairs_with_features = pairs_with_features.withColumn(
            "factors_mult",
            array_mult(sf.col("item_factors"), sf.col("user_factors")),
        )
        factors_to_explode.append(("factors_mult", "fm"))

    for col_name, feature_prefix in factors_to_explode:
        col_set = set(pairs_with_features.columns)
        col_set.remove(col_name)
        pairs_with_features = horizontal_explode(
            data_frame=pairs_with_features,
            column_to_explode=col_name,
            other_columns=[sf.col(column) for column in sorted(list(col_set))],
            prefix=f"{prefix}_{feature_prefix}",
        )

    return pairs_with_features


def test_works(log, model):
    try:
        pred = model.fit_predict(log, k=1)
        assert pred.count() == 4
    except:  # noqa
        pytest.fail()


def test_diff_feedback_type(log, model):
    pred_exp = model.fit_predict(log, k=1)
    model.implicit_prefs = True
    pred_imp = model.fit_predict(log, k=1)
    assert not np.allclose(
        pred_exp.toPandas().sort_values("user_idx")["relevance"].values,
        pred_imp.toPandas().sort_values("user_idx")["relevance"].values,
    )


def test_enrich_with_features(log, model):
    model.fit(log.filter(sf.col("user_idx").isin([0, 2])))
    res = get_first_level_model_features(
        model, log.filter(sf.col("user_idx").isin([0, 1]))
    )

    cold_user_and_item = res.filter(
        (sf.col("user_idx") == 1) & (sf.col("item_idx") == 3)
    )
    row_dict = cold_user_and_item.collect()[0].asDict()
    assert row_dict["_if_0"] == row_dict["_uf_0"] == row_dict["_fm_1"] == 0.0

    warm_user_and_item = res.filter(
        (sf.col("user_idx") == 0) & (sf.col("item_idx") == 0)
    )
    row_dict = warm_user_and_item.collect()[0].asDict()
    np.allclose(
        [row_dict["_fm_1"], row_dict["_if_1"] * row_dict["_uf_1"]],
        [4.093189725967505, row_dict["_fm_1"]],
    )

    cold_user_warm_item = res.filter(
        (sf.col("user_idx") == 1) & (sf.col("item_idx") == 0)
    )
    row_dict = cold_user_warm_item.collect()[0].asDict()
    np.allclose(
        [row_dict["_if_1"], row_dict["_if_1"] * row_dict["_uf_1"]],
        [-2.938199281692505, 0],
    )


def test_init_args(model):
    args = model._init_args
    
    assert args["rank"] == 2
    assert args["implicit_prefs"] is False
    assert args["seed"] == 42


def test_predict_pairs_warm_items_only(log, log_to_pred):
    model = ALSWrap(seed=SEED)
    model.fit(log)
    recs = model.predict(
        log.unionByName(log_to_pred),
        k=3,
        users=log_to_pred.select("user_idx").distinct(),
        items=log_to_pred.select("item_idx").distinct(),
        filter_seen_items=False,
    )

    pairs_pred = model.predict_pairs(
        pairs=log_to_pred.select("user_idx", "item_idx"),
        log=log.unionByName(log_to_pred),
    )

    condition = ~sf.col("item_idx").isin([4, 5])
    if not model.can_predict_cold_users:
        condition = condition & (sf.col("user_idx") != 4)

    sparkDataFrameEqual(
        pairs_pred.select("user_idx", "item_idx"),
        log_to_pred.filter(condition).select("user_idx", "item_idx"),
    )

    recs_joined = (
        pairs_pred.withColumnRenamed("relevance", "pairs_relevance")
        .join(recs, on=["user_idx", "item_idx"], how="left")
        .sort("user_idx", "item_idx")
    )

    assert np.allclose(
        recs_joined.select("relevance").toPandas().to_numpy(),
        recs_joined.select("pairs_relevance").toPandas().to_numpy(),
    )


def test_predict_pairs_k(log):
    model = ALSWrap(seed=SEED)
    model.fit(log)

    pairs_pred_k = model.predict_pairs(
        pairs=log.select("user_idx", "item_idx"),
        log=log,
        k=1,
    )

    pairs_pred = model.predict_pairs(
        pairs=log.select("user_idx", "item_idx"),
        log=log,
        k=None,
    )

    assert (
        pairs_pred_k.groupBy("user_idx")
        .count()
        .filter(sf.col("count") > 1)
        .count()
        == 0
    )

    assert (
        pairs_pred.groupBy("user_idx")
        .count()
        .filter(sf.col("count") > 1)
        .count()
        > 0
    )


def test_predict_empty_log(log):
    model = ALSWrap(seed=SEED)
    model.fit(log)
    model.predict(log.limit(0), 1)


def test_predict_pairs_raises_pairs_format(log):
    model = ALSWrap(seed=SEED)
    with pytest.raises(ValueError, match="pairs must be a dataframe with .*"):
        model.fit(log)
        model.predict_pairs(log, log)


@pytest.mark.parametrize(
    "als_model, metric",
    [
        (ALSWrap(seed=SEED), "euclidean_distance_sim"),
        (ALSWrap(seed=SEED), "dot_product"),
        (ALSWrap(seed=SEED), "cosine_similarity"),
    ],
    ids=[
        "als_euclidean",
        "als_dot",
        "als_cosine",
    ],
)
def test_get_nearest_items(log, als_model, metric):
    als_model.fit(log.filter(sf.col("item_idx") != 3))
    res = als_model.get_nearest_items(items=[0, 1], k=2, metric=metric)

    assert res.count() == 4
    assert set(res.toPandas().to_dict()["item_idx"].values()) == {
        0,
        1,
    }

    res = als_model.get_nearest_items(items=[0, 1], k=1, metric=metric)
    assert res.count() == 2

    # filter neighbours
    res = als_model.get_nearest_items(
        items=[0, 1],
        k=4,
        metric=metric,
        candidates=[0, 3],
    )
    assert res.count() == 1
    assert (
        len(
            set(res.toPandas().to_dict()["item_idx"].values()).difference(
                {0, 1}
            )
        )
        == 0
    )


@pytest.mark.parametrize("metric", ["absent", None])
def test_nearest_items_raises(log, metric):
    model = AssociationRulesItemRec()
    model.fit(log.filter(sf.col("item_idx") != 3))
    with pytest.raises(
        ValueError, match=r"Select one of the valid distance metrics.*"
    ):
        model.get_nearest_items(items=[0, 1], k=2, metric=metric)
    model = ALSWrap()
    model.fit(log)
    with pytest.raises(
        ValueError, match=r"Select one of the valid distance metrics.*"
    ):
        model.get_nearest_items(items=[0, 1], k=2, metric=metric)


def test_predict_cold_and_new_filter_out(long_log_with_features):
    model = ALSWrap(rank=2, seed=SEED)
    pred = fit_predict_selected(
        model,
        train_log=long_log_with_features.filter(sf.col("user_idx") != 0),
        inf_log=long_log_with_features,
        user_features=None,
        users=[0, 3],
    )
    # assert new/cold users are filtered out in `predict`
    if not model.can_predict_cold_users:
        assert pred.count() == 0
    else:
        assert 1 <= pred.count() <= 2


def test_predict_pairs_to_file(spark, long_log_with_features, tmp_path):
    model = ALSWrap(rank=2, seed=SEED)
    path = str((tmp_path / "pred.parquet").resolve().absolute())
    model.fit(long_log_with_features)
    model.predict_pairs(
        log=long_log_with_features,
        pairs=long_log_with_features.filter(sf.col("user_idx") == 1).select(
            "user_idx", "item_idx"
        ),
        recs_file_path=path,
    )
    pred_cached = model.predict_pairs(
        log=long_log_with_features,
        pairs=long_log_with_features.filter(sf.col("user_idx") == 1).select(
            "user_idx", "item_idx"
        ),
        recs_file_path=None,
    )
    pred_from_file = spark.read.parquet(path)
    sparkDataFrameEqual(pred_cached, pred_from_file)


def test_predict_to_file(spark, long_log_with_features, tmp_path):
    model = ALSWrap(rank=2, seed=SEED)
    path = str((tmp_path / "pred.parquet").resolve().absolute())
    model.fit_predict(long_log_with_features, k=10, recs_file_path=path)
    pred_cached = model.predict(
        long_log_with_features, k=10, recs_file_path=None
    )
    pred_from_file = spark.read.parquet(path)
    sparkDataFrameEqual(pred_cached, pred_from_file)


@pytest.mark.parametrize(
    "borders",
    [
        {"wrong_name": None},
        {"rank": None},
        {"rank": 2},
        {"rank": [1]},
        {"rank": [1, 2, 3]},
    ],
    ids=[
        "wrong name",
        "None border",
        "int border",
        "border's too short",
        "border's too long",
    ],
)
def test_bad_borders(borders):
    model = ALSWrap()
    with pytest.raises(ValueError):
        model._prepare_param_borders(borders)


@pytest.mark.parametrize("borders", [None, {"rank": [5, 9]}])
def test_correct_borders(borders):
    model = ALSWrap()
    res = model._prepare_param_borders(borders)
    assert res.keys() == model._search_space.keys()
    assert "rank" in res
    assert isinstance(res["rank"], dict)
    assert res["rank"].keys() == model._search_space["rank"].keys()


@pytest.mark.parametrize(
    "borders,answer", [(None, True), ({"rank": [-10, -1]}, False)]
)
def test_param_in_borders(borders, answer):
    model = ALSWrap()
    search_space = model._prepare_param_borders(borders)
    assert model._init_params_in_search_space(search_space) == answer


def test_it_works(log):
    model = ALSWrap()
    assert model._params_tried() is False
    res = model.optimize(log, log, k=2, budget=1)
    assert isinstance(res["rank"], int)
    assert model._params_tried() is True
    model.optimize(log, log, k=2, budget=1)
    assert len(model.study.trials) == 1
    model.optimize(log, log, k=2, budget=1, new_study=False)
    assert len(model.study.trials) == 2


def test_equal_preds(long_log_with_features, tmp_path):
    recommender = ALSWrap
    path = (tmp_path / "test").resolve()
    model = recommender()
    model.fit(long_log_with_features)
    base_pred = model.predict(long_log_with_features, 5)
    save(model, path)
    loaded_model = load(path)
    new_pred = loaded_model.predict(long_log_with_features, 5)
    sparkDataFrameEqual(base_pred, new_pred)

