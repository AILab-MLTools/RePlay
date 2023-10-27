# pylint: disable-all
from datetime import datetime

import pytest
import numpy as np
import pandas as pd

from replay.splitters import TwoStageSplitter
from tests.utils import spark


@pytest.fixture
def log(spark):
    return spark.createDataFrame(
        data=[
            [0, 3, datetime(2019, 9, 12), 1.0, 1],
            [1, 4, datetime(2019, 9, 13), 2.0, 1],
            [2, 6, datetime(2019, 9, 17), 1.0, 1],
            [3, 5, datetime(2019, 9, 17), 1.0, 1],
            [4, 5, datetime(2019, 9, 17), 1.0, 1],
            [0, 5, datetime(2019, 9, 12), 1.0, 1],
            [1, 6, datetime(2019, 9, 13), 2.0, 1],
            [2, 7, datetime(2019, 9, 17), 1.0, 1],
            [3, 8, datetime(2019, 9, 17), 1.0, 1],
            [4, 0, datetime(2019, 9, 17), 1.0, 1],
        ],
        schema=["user_idx", "item_idx", "timestamp", "relevance", "session_id"],
    )


@pytest.fixture
def log_pandas(log):
    return log.toPandas()


@pytest.mark.parametrize(
    "dataset_type",
    [
        ("log"),
        ("log_pandas"),
    ]
)
@pytest.mark.parametrize("fraction", [3, 0.6])
def test_get_test_users(dataset_type, request, fraction):
    log = request.getfixturevalue(dataset_type)
    splitter = TwoStageSplitter(
        first_divide_size=fraction,
        second_divide_size=1,
        drop_cold_items=False,
        drop_cold_users=False,
        session_id_col="session_id",
        seed=1234,
    )
    test_users = splitter._get_test_users(log)
    if isinstance(log, pd.DataFrame):
        assert test_users.shape[0] == 3
        assert np.isin([0, 1, 4], test_users.user_idx).all()

    else:
        assert test_users.count() == 3
        assert np.isin([0, 2, 3], test_users.toPandas().user_idx).all()


@pytest.mark.parametrize(
    "dataset_type",
    [
        ("log"),
        ("log_pandas"),
    ]
)
@pytest.mark.parametrize("fraction", [5, 1.0])
def test_user_test_size_exception(dataset_type, request, fraction):
    log = request.getfixturevalue(dataset_type)
    splitter = TwoStageSplitter(
        first_divide_size=fraction,
        second_divide_size=1,
        drop_cold_items=False,
        drop_cold_users=False,
        session_id_col="session_id",
    )
    with pytest.raises(ValueError):
        splitter._get_test_users(log)


@pytest.fixture
def big_log(spark):
    return spark.createDataFrame(
        data=[
            [0, 3, datetime(2019, 9, 12), 1.0, 1],
            [0, 4, datetime(2019, 9, 13), 2.0, 1],
            [0, 6, datetime(2019, 9, 17), 1.0, 1],
            [0, 5, datetime(2019, 9, 17), 1.0, 1],
            [1, 3, datetime(2019, 9, 12), 1.0, 1],
            [1, 4, datetime(2019, 9, 13), 2.0, 1],
            [1, 5, datetime(2019, 9, 14), 3.0, 1],
            [1, 1, datetime(2019, 9, 15), 4.0, 1],
            [1, 2, datetime(2019, 9, 15), 4.0, 1],
            [2, 3, datetime(2019, 9, 12), 1.0, 1],
            [2, 4, datetime(2019, 9, 13), 2.0, 1],
            [2, 5, datetime(2019, 9, 14), 3.0, 1],
            [2, 1, datetime(2019, 9, 14), 3.0, 1],
            [2, 6, datetime(2019, 9, 17), 1.0, 1],
            [3, 1, datetime(2019, 9, 15), 4.0, 1],
            [3, 0, datetime(2019, 9, 16), 4.0, 1],
            [3, 3, datetime(2019, 9, 17), 4.0, 1],
            [3, 4, datetime(2019, 9, 18), 4.0, 1],
            [3, 7, datetime(2019, 9, 19), 4.0, 1],
            [3, 3, datetime(2019, 9, 20), 4.0, 1],
            [3, 0, datetime(2019, 9, 21), 4.0, 1],
        ],
        schema=["user_idx", "item_idx", "timestamp", "relevance", "session_id"],
    )


@pytest.fixture
def big_log_pandas(big_log):
    return big_log.toPandas()


test_sizes = np.arange(0.1, 1, 0.25).tolist() + list(range(1, 5))


@pytest.mark.parametrize(
    "dataset_type",
    [
        ("big_log"),
        ("big_log_pandas"),
    ]
)
@pytest.mark.parametrize("item_test_size", test_sizes)
@pytest.mark.parametrize("shuffle", [True, False])
def test_random_split(dataset_type, request, item_test_size, shuffle):
    big_log = request.getfixturevalue(dataset_type)
    splitter = TwoStageSplitter(
        first_divide_size=0.5,
        second_divide_size=item_test_size,
        drop_cold_items=False,
        drop_cold_users=False,
        seed=1234,
        session_id_col="session_id",
        shuffle=shuffle,
    )
    train, test = splitter.split(big_log)

    if isinstance(big_log, pd.DataFrame):
        assert train.shape[0] + test.shape[0] == big_log.shape[0]
        assert len(train.merge(test, on=["user_idx", "item_idx", "timestamp", "session_id"], how="inner")) == 0

        if isinstance(item_test_size, int):
            #  it's a rough check. for it to be true, item_test_size must be bigger than log length for every user
            num_users = big_log["user_idx"].nunique() * 0.5     # only half of users go to test
            assert num_users * item_test_size == test.shape[0]
            assert big_log.shape[0] - num_users * item_test_size == train.shape[0]
    else:
        assert train.count() + test.count() == big_log.count()
        assert test.intersect(train).count() == 0

        if isinstance(item_test_size, int):
            #  it's a rough check. for it to be true, item_test_size must be bigger than log length for every user
            num_users = big_log.select("user_idx").distinct().count() * 0.5
            assert num_users * item_test_size == test.count()
            assert big_log.count() - num_users * item_test_size == train.count()


@pytest.mark.parametrize(
    "dataset_type",
    [
        ("big_log"),
        ("big_log_pandas"),
    ]
)
@pytest.mark.parametrize("item_test_size", [2.0, -1, -50, 2.1, -0.01])
def test_item_test_size_exception(dataset_type, request, item_test_size):
    big_log = request.getfixturevalue(dataset_type)
    splitter = TwoStageSplitter(
        first_divide_size=2,
        second_divide_size=item_test_size,
        drop_cold_items=False,
        drop_cold_users=False,
        seed=1234,
        session_id_col="session_id",
    )
    with pytest.raises(ValueError):
        splitter.split(big_log)


@pytest.fixture
def log2(spark):
    return spark.createDataFrame(
        data=[
            [0, 0, datetime(2019, 1, 1), 1.0, 1],
            [0, 1, datetime(2019, 1, 2), 1.0, 1],
            [0, 2, datetime(2019, 1, 3), 1.0, 1],
            [0, 3, datetime(2019, 1, 4), 1.0, 1],
            [1, 4, datetime(2020, 2, 5), 1.0, 1],
            [1, 3, datetime(2020, 2, 4), 1.0, 1],
            [1, 2, datetime(2020, 2, 3), 1.0, 1],
            [1, 1, datetime(2020, 2, 2), 1.0, 1],
            [1, 0, datetime(2020, 2, 1), 1.0, 1],
            [2, 0, datetime(1995, 1, 1), 1.0, 1],
            [2, 1, datetime(1995, 1, 2), 1.0, 1],
            [2, 2, datetime(1995, 1, 3), 1.0, 1],
        ],
        schema=["user_idx", "item_idx", "timestamp", "relevance", "session_id"],
    )


@pytest.fixture
def log2_pandas(log2):
    return log2.toPandas()


@pytest.mark.parametrize(
    "dataset_type",
    [
        ("log2"),
        ("log2_pandas"),
    ]
)
def test_split_quantity(dataset_type, request):
    log2 = request.getfixturevalue(dataset_type)
    splitter = TwoStageSplitter(
        first_divide_size=0.5,
        second_divide_size=2,
        drop_cold_items=False,
        drop_cold_users=False,
    )
    train, test = splitter.split(log2)
    if isinstance(log2, pd.DataFrame):
        num_items = test.user_idx.value_counts()
    else:
        num_items = test.toPandas().user_idx.value_counts()

    assert num_items.nunique() == 1
    assert num_items.unique()[0] == 2


@pytest.mark.parametrize(
    "dataset_type",
    [
        ("log2"),
        ("log2_pandas"),
    ]
)
def test_split_proportion(dataset_type, request):
    log2 = request.getfixturevalue(dataset_type)
    splitter = TwoStageSplitter(
        first_divide_size=1,
        second_divide_size=0.4,
        drop_cold_items=False,
        drop_cold_users=False,
        seed=13,
    )
    train, test = splitter.split(log2)
    if isinstance(log2, pd.DataFrame):
        num_items = test.user_idx.value_counts()
        assert num_items[1] == 2
    else:
        num_items = test.toPandas().user_idx.value_counts()
        assert num_items[0] == 1
