import pytest
import numpy as np
import pandas as pd

from replay.data import FeatureHint, FeatureSchema, FeatureInfo,  FeatureType, Dataset, FeatureSource
from replay.utils import TORCH_AVAILABLE

if TORCH_AVAILABLE:
    import torch
    import lightning as L
    from replay.data.nn import TensorSchema, TensorFeatureInfo, TensorFeatureSource, SequenceTokenizer
    from replay.models.nn.sequential.sasrec import SasRecTrainingDataset, SasRec


@pytest.fixture()
def feature_schema_for_sasrec():
    schema = FeatureSchema(
        [
            FeatureInfo(
                column="user_id",
                feature_hint=FeatureHint.QUERY_ID,
                feature_type=FeatureType.CATEGORICAL,
            ),
            FeatureInfo(
                column="item_id",
                feature_hint=FeatureHint.ITEM_ID,
                feature_type=FeatureType.CATEGORICAL,
            ),
        ]
    )

    return schema


@pytest.fixture()
def fitted_sasrec(feature_schema_for_sasrec):  
    data = pd.DataFrame({
        "user_id": [0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2],
        "item_id": [0, 1, 2, 0, 1, 3, 1, 2, 0, 2, 3, 1, 2],
    })

    train_dataset = Dataset(feature_schema=feature_schema_for_sasrec, interactions=data)
    tensor_schema = TensorSchema(
        TensorFeatureInfo(
            name="item_id_seq",
            is_seq=True,
            cardinality=train_dataset.item_count,
            feature_type=FeatureType.CATEGORICAL,
            feature_sources=[TensorFeatureSource(FeatureSource.INTERACTIONS, train_dataset.feature_schema.item_id_column)],
            feature_hint=FeatureHint.ITEM_ID,
        )
    )
    
    tokenizer = SequenceTokenizer(tensor_schema, allow_collect_to_master=True)
    tokenizer.fit(train_dataset)
    sequential_train_dataset = tokenizer.transform(train_dataset)

    model = SasRec(tensor_schema)
    trainer = L.Trainer(max_epochs=1)
    train_loader = torch.utils.data.DataLoader(
        SasRecTrainingDataset(sequential_train_dataset, 200)
    )

    trainer.fit(model, train_dataloaders=train_loader)

    return model, tokenizer


@pytest.fixture()
def new_items_dataset():
    data = pd.DataFrame({
        "user_id": [0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2],
        "item_id": [0, 1, 2, 0, 1, 3, 1, 2, 0, 2, 3, 1, 2, 4],
    })

    return data
