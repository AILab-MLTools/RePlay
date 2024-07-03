import json
from pathlib import Path
from typing import Any, Union

from polars import from_pandas as pl_from_pandas

from replay.splitters import (
    ColdUserRandomSplitter,
    KFolds,
    LastNSplitter,
    NewUsersSplitter,
    RandomSplitter,
    RatioSplitter,
    TimeSplitter,
    TwoStageSplitter,
)
from replay.utils import (
    TORCH_AVAILABLE,
    PandasDataFrame,
    PolarsDataFrame,
    SparkDataFrame,
)
from replay.utils.spark_utils import (
    convert2spark as pandas_to_spark,
    spark_to_pandas,
)

SavableObject = Union[
    ColdUserRandomSplitter,
    KFolds,
    LastNSplitter,
    NewUsersSplitter,
    RandomSplitter,
    RatioSplitter,
    TimeSplitter,
    TwoStageSplitter,
]

if TORCH_AVAILABLE:
    from replay.data.nn import SequenceTokenizer

    SavableObject = Union[
        ColdUserRandomSplitter,
        KFolds,
        LastNSplitter,
        NewUsersSplitter,
        RandomSplitter,
        RatioSplitter,
        TimeSplitter,
        TwoStageSplitter,
        SequenceTokenizer,
    ]


def save_to_replay(obj: SavableObject, path: Union[str, Path]) -> None:
    """
    General function to save RePlay models, splitters and tokenizer.

    :param path: Path to save the object.
    """
    obj.save(path)


def load_from_replay(path: Union[str, Path], **kwargs) -> SavableObject:
    """
    General function to load RePlay models, splitters and tokenizer.

    :param path: Path to save the object.
    """
    path = Path(path).with_suffix(".replay").resolve()
    with open(path / "init_args.json", "r") as file:
        class_name = json.loads(file.read())["_class_name"]
    obj_type = globals()[class_name]
    obj = obj_type.load(path, **kwargs)

    return obj


def _check_if_dataframe(var: Any):
    if not isinstance(var, (SparkDataFrame, PolarsDataFrame, PandasDataFrame)):
        msg = f"Object of type {type(var)} is not a dataframe of known type (can be pandas|spark|polars)"
        raise ValueError(msg)


def convert2pandas(
    df: Union[SparkDataFrame, PolarsDataFrame, PandasDataFrame], allow_collect_to_master: bool = False
) -> PandasDataFrame:
    """
    Convert the spark|polars DataFrame to a pandas.DataFrame.
    Returns unchanged dataframe if the input is already of type pandas.DataFrame. 

    :param df: The dataframe to convert. Can be polars|spark|pandas DataFrame.
    :param allow_collect_to_master: If set to False (default) raises a warning about collecting parallelized data to the master node.
    """
    _check_if_dataframe(df)
    if isinstance(df, PandasDataFrame):
        return df
    if isinstance(df, PolarsDataFrame):
        return df.to_pandas()
    if isinstance(df, SparkDataFrame):
        return spark_to_pandas(df, allow_collect_to_master, from_constructor=False)


def convert2polars(
    df: Union[SparkDataFrame, PolarsDataFrame, PandasDataFrame], allow_collect_to_master: bool = False
) -> PolarsDataFrame:
    """
    Convert the spark|pandas DataFrame to a polars.DataFrame.
    Returns unchanged dataframe if the input is already of type polars.DataFrame. 

    :param df: The dataframe to convert. Can be spark|pandas|polars DataFrame.
    :param allow_collect_to_master: If set to False (default) raises a warning about collecting parallelized data to the master node.
    """
    _check_if_dataframe(df)
    if isinstance(df, PandasDataFrame):
        return pl_from_pandas(df)
    if isinstance(df, PolarsDataFrame):
        return df
    if isinstance(df, SparkDataFrame):
        return pl_from_pandas(spark_to_pandas(df, allow_collect_to_master, from_constructor=False))


def convert2spark(df: Union[SparkDataFrame, PolarsDataFrame, PandasDataFrame]) -> SparkDataFrame:
    """
    Convert the pandas|polars DataFrame to a pysaprk.sql.DataFrame.
    Returns unchanged dataframe if the input is already of type pysaprk.sql.DataFrame. 

    :param df: The dataframe to convert. Can be pandas|polars|spark Datarame.
    """
    _check_if_dataframe(df)
    if isinstance(df, (PandasDataFrame, SparkDataFrame)):
        return pandas_to_spark(df)
    if isinstance(df, PolarsDataFrame):
        return pandas_to_spark(df.to_pandas())
