# pylint: disable-all
from datetime import datetime

import pytest
import numpy as np
import pandas as pd

from replay.data import LOG_SCHEMA
from replay.splitters import NewUsersSplitter
from tests.utils import spark


@pytest.fixture
def log(spark):
    return spark.createDataFrame(
        data=[
            [1, 3, datetime(2019, 9, 14), 3.0, 1],
            [1, 0, datetime(2019, 9, 14), 3.0, 1],
            [1, 1, datetime(2019, 9, 15), 4.0, 1],
            [0, 3, datetime(2019, 9, 12), 1.0, 1],
            [3, 0, datetime(2019, 9, 12), 1.0, 1],
            [3, 1, datetime(2019, 9, 13), 2.0, 1],
            [2, 0, datetime(2019, 9, 16), 5.0, 1],
            [2, 3, datetime(2019, 9, 16), 5.0, 1],
            [0, 2, datetime(2019, 9, 17), 1.0, 1],
        ],
        schema=["user_id", "item_id", "timestamp", "relevance", "session_id"],
    )


@pytest.fixture
def log_pandas(log):
    return log.toPandas()


@pytest.mark.parametrize(
    "dataset_type",
    [
        ("log_pandas"),
        ("log"),
    ]
)
def test_users_are_cold(dataset_type, request):
    log = request.getfixturevalue(dataset_type)
    splitter = NewUsersSplitter(test_size=0.25, drop_cold_items=False, session_id_column="session_id")
    train, test = splitter.split(log)

    if isinstance(log, pd.DataFrame):
        train_users = train.user_id
        test_users = test.user_id
    else:
        train_users = train.toPandas().user_id
        test_users = test.toPandas().user_id

    assert not np.isin(test_users, train_users).any()


def test_bad_test_size():
    with pytest.raises(ValueError):
        NewUsersSplitter(1.2)
