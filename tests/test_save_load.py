# pylint: disable=redefined-outer-name, missing-function-docstring, unused-import, wildcard-import, unused-wildcard-import
from os.path import dirname, join

import numpy as np
import pytest
import pandas as pd
from pytorch_ranger import Ranger

import replay
from replay.data_preparator import Indexer
from replay.model_handler import (
    save_indexer,
    load_indexer,
    save_splitter,
    load_splitter,
    save_optimizers,
    load_optimizers,
)
from replay.utils import convert2spark
from replay.splitters import *
from replay.models import DDPG
from replay.models.ddpg import ActorDRR, CriticDRR


@pytest.fixture
def user_features(spark):
    return spark.createDataFrame(
        [(1, 20.0, -3.0, 1), (2, 30.0, 4.0, 0), (3, 40.0, 0.0, 1)]
    ).toDF("user_idx", "age", "mood", "gender")


@pytest.fixture
def df():
    folder = dirname(replay.__file__)
    res = pd.read_csv(
        join(folder, "../experiments/data/ml1m_ratings.dat"),
        sep="\t",
        names=["user_id", "item_id", "relevance", "timestamp"],
    ).head(1000)
    res = convert2spark(res)
    indexer = Indexer()
    indexer.fit(res, res)
    res = indexer.transform(res)
    return res


def test_indexer(df, tmp_path):
    path = (tmp_path / "indexer").resolve()
    indexer = Indexer("user_idx", "item_idx")
    df = convert2spark(df)
    indexer.fit(df, df)
    save_indexer(indexer, path)
    i = load_indexer(path)
    i.inverse_transform(i.transform(df))
    assert i.user_indexer.inputCol == indexer.user_indexer.inputCol


@pytest.mark.parametrize(
    "splitter, init_args",
    [
        (DateSplitter, {"test_start": 0.8}),
        (RandomSplitter, {"test_size": 0.8, "seed": 123}),
        (NewUsersSplitter, {"test_size": 0.8}),
        (ColdUserRandomSplitter, {"test_size": 0.8, "seed": 123}),
        (
            UserSplitter,
            {"item_test_size": 1, "user_test_size": 0.2, "seed": 123},
        ),
    ],
)
def test_splitter(splitter, init_args, df, tmp_path):
    path = (tmp_path / "splitter").resolve()
    splitter = splitter(**init_args)
    save_splitter(splitter, path)
    train, test = splitter.split(df)
    restored_splitter = load_splitter(path)
    for arg_, value_ in init_args.items():
        assert getattr(restored_splitter, arg_) == value_
    new_train, new_test = restored_splitter.split(df)
    assert new_train.count() == train.count()
    assert new_test.count() == test.count()


def test_optimizers(df, tmp_path):
    path = (tmp_path / "optimizers").resolve()
    ddpg = DDPG()
    model = ActorDRR(
        ddpg.user_num,
        ddpg.item_num,
        ddpg.embedding_dim,
        ddpg.hidden_dim,
        ddpg.memory_size,
    )
    value_net = CriticDRR(
        ddpg.embedding_dim * 3, ddpg.embedding_dim, ddpg.hidden_dim
    )

    policy_optimizer = Ranger(
        model.parameters(),
        lr=ddpg.policy_lr,
        weight_decay=ddpg.policy_decay,
    )
    value_optimizer = Ranger(
        value_net.parameters(),
        lr=ddpg.value_lr,
        weight_decay=ddpg.value_decay,
    )
    save_optimizers(
        {
            "policy_optimizer": policy_optimizer,
            "value_optimizer": value_optimizer,
        }
        , path
    )
    optimizers = load_optimizers(
        path, 
        {"policy_optimizer": model, "value_optimizer": value_net}
    )
    assert np.isclose(
        policy_optimizer.state_dict()['param_groups'][0]['lr'],
        optimizers["policy_optimizer"].state_dict()['param_groups'][0]['lr'],
        atol=0.01,
    )
