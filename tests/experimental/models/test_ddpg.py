# pylint: disable-all
from datetime import datetime

import pytest
import torch
import numpy as np
from pytorch_ranger import Ranger
from pyspark.sql import functions as sf

from replay.data import LOG_SCHEMA
from replay.experimental.models import DDPG
from replay.experimental.models.ddpg import (
    ActorDRR,
    CriticDRR,
    OUNoise,
    ReplayBuffer,
    to_np,
)
from tests.utils import (
    del_files_by_pattern,
    find_file_by_pattern,
    spark,
    log,
    log_to_pred,
    long_log_with_features,
    user_features,
    sparkDataFrameEqual,
)


SEED = 123


@pytest.fixture(scope="session", autouse=True)
def fix_seeds():
    torch.manual_seed(7)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(0)


@pytest.fixture
def log(spark):
    date = datetime(2019, 1, 1)
    return spark.createDataFrame(
        data=[
            (0, 0, date, 1.0),
            (0, 1, date, 1.0),
            (0, 2, date, 1.0),
            (1, 0, date, 1.0),
            (1, 1, date, 1.0),
            (0, 0, date, 1.0),
            (0, 1, date, 1.0),
            (0, 2, date, 1.0),
            (1, 0, date, 1.0),
            (1, 1, date, 1.0),
            (0, 0, date, 1.0),
            (0, 1, date, 1.0),
            (0, 2, date, 1.0),
            (1, 0, date, 1.0),
            (1, 1, date, 1.0),
            (2, 3, date, 1.0),
        ],
        schema=LOG_SCHEMA,
    )


@pytest.fixture
def model(log):
    model = DDPG(user_num=5, item_num=5)
    model.batch_size = 1
    return model


def test_fit(log, model):
    model.fit(log)
    assert len(list(model.model.parameters())) == 10
    param_shapes = [
        (16, 24),
        (16,),
        (16,),
        (16,),
        (8, 16),
        (8,),
        (3, 8),
        (5, 8),
        (1, 5, 1),
        (1,),
    ]
    for i, parameter in enumerate(model.model.parameters()):
        assert param_shapes[i] == tuple(parameter.shape)


def test_predict(log, model):
    model.noise_type = "gauss"
    model.batch_size = 4
    model.fit(log)
    try:
        pred = model.predict(log=log, k=1)
        pred.count()
    except RuntimeError:  # noqa
        pytest.fail()


def test_save_load(log, model, user_num=5, item_num=5):
    spark_local_dir = "./logs/tmp/"
    pattern = "model_final.pt"
    del_files_by_pattern(spark_local_dir, pattern)

    model.exact_embeddings_size = False
    model.fit(log=log)
    old_params = [
        param.detach().cpu().numpy() for param in model.model.parameters()
    ]
    old_policy_optimizer_params = model.policy_optimizer.state_dict()[
        "param_groups"
    ][0]
    old_value_optimizer_params = model.value_optimizer.state_dict()[
        "param_groups"
    ][0]
    path = find_file_by_pattern(spark_local_dir, pattern)
    assert path is not None

    new_model = DDPG(user_num=user_num, item_num=item_num)
    new_model.model = ActorDRR(
        user_num=user_num,
        item_num=item_num,
        embedding_dim=8,
        hidden_dim=16,
        memory_size=5,
        env_gamma_alpha=1,
        device=torch.device("cpu"),
        min_trajectory_len=10,
    )
    new_model.value_net = CriticDRR(
        state_repr_dim=24,
        action_emb_dim=8,
        hidden_dim=8,
        heads_num=10,
        heads_q=0.15,
    )
    assert len(old_params) == len(list(new_model.model.parameters()))

    new_model.policy_optimizer = Ranger(new_model.model.parameters())
    new_model.value_optimizer = Ranger(new_model.value_net.parameters())
    new_model._load_model(path)
    for i, parameter in enumerate(new_model.model.parameters()):
        assert np.allclose(
            parameter.detach().cpu().numpy(),
            old_params[i],
            atol=1.0e-3,
        )
    for param_name, parameter in new_model.policy_optimizer.state_dict()[
        "param_groups"
    ][0].items():
        assert np.allclose(
            parameter,
            old_policy_optimizer_params[param_name],
            atol=1.0e-3,
        )
    for param_name, parameter in new_model.value_optimizer.state_dict()[
        "param_groups"
    ][0].items():
        assert np.allclose(
            parameter,
            old_value_optimizer_params[param_name],
            atol=1.0e-3,
        )


def test_env_step(log, model, user=[0, 1, 2]):
    replay_buffer = ReplayBuffer(
        torch.device("cpu"),
        1000000,
        5,
        8,
    )
    # model.replay_buffer.capacity = 4
    train_matrix, _, item_num, _ = model._preprocess_log(log)
    model.model = ActorDRR(
        model.user_num,
        model.item_num,
        model.embedding_dim,
        model.hidden_dim,
        model.memory_size,
        1,
        torch.device("cpu"),
        min_trajectory_len=10,
    )
    model.model.environment.update_env(matrix=train_matrix)
    model.ou_noise = OUNoise(
        model.embedding_dim,
        torch.device("cpu"),
        theta=model.noise_theta,
        max_sigma=model.noise_sigma,
        min_sigma=model.noise_sigma,
        noise_type=model.noise_type,
    )

    user, memory = model.model.environment.reset(user)

    action_emb = model.model(user, memory)
    model.ou_noise.noise_type = "abcd"
    with pytest.raises(ValueError):
        action_emb = model.ou_noise.get_action(action_emb[0], 0)

    model.ou_noise.noise_type = "ou"
    model.model.get_action(
        action_emb,
        model.model.environment.available_items,
        torch.ones_like(model.model.environment.available_items),
        return_scores=True,
    )

    model.model.environment.memory[user, -1] = item_num + 100

    # choose related action
    global_action = model.model.environment.related_items[user, 0]
    action = torch.where(
        model.model.environment.available_items - global_action.reshape(-1, 1)
        == 0
    )[1]

    # step
    model.model.environment.step(action, action_emb, replay_buffer)

    # chech memory update
    assert (model.model.environment.memory[user, -1] == global_action).prod()


def test_predict_pairs_to_file(spark, long_log_with_features, tmp_path):
    model = DDPG(seed=SEED, user_num=6, item_num=6)
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
    model = DDPG(seed=SEED, user_num=6, item_num=6)
    path = str((tmp_path / "pred.parquet").resolve().absolute())
    model.fit_predict(long_log_with_features, k=10, recs_file_path=path)
    pred_cached = model.predict(
        long_log_with_features, k=10, recs_file_path=None
    )
    pred_from_file = spark.read.parquet(path)
    sparkDataFrameEqual(pred_cached, pred_from_file)
