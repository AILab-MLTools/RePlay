import pytest
from pandas import DataFrame as PandasDataFrame

from replay.preprocessing import MinMaxInteractionsFilter


@pytest.mark.spark
@pytest.mark.usefixtures("interactions_ml_100k_spark")
def test_interaction_entries_filter_init(interactions_ml_100k_spark):
    _ = MinMaxInteractionsFilter(
        min_inter_per_user=10,
        max_inter_per_user=300,
        min_inter_per_item=10,
        max_inter_per_item=250,
    ).transform(interactions_ml_100k_spark)


@pytest.mark.parametrize(
    "dataset_type",
    [
        pytest.param("spark", marks=pytest.mark.spark),
        pytest.param("pandas", marks=pytest.mark.core),
    ],
)
def test_interaction_entries_filter_min_user_interact(dataset_type, request):
    columns = ["user_id", "item_id"]
    data = [(1, 1), (1, 2), (3, 3)]

    if dataset_type == "spark":
        spark_session = request.getfixturevalue("spark_session")
        test_dataframe = spark_session.createDataFrame(data, schema=columns)
    else:
        test_dataframe = PandasDataFrame(data, columns=columns)

    filtered_dataframe = MinMaxInteractionsFilter(min_inter_per_user=2).transform(test_dataframe)

    if dataset_type == "spark":
        user_list = filtered_dataframe.select("user_id").distinct().collect()
        assert len([x[0] for x in user_list]) == 1
        assert user_list[0][0] == 1
    else:
        user_list = filtered_dataframe["user_id"].unique().tolist()
        assert len(user_list) == 1
        assert user_list[0] == 1


@pytest.mark.parametrize(
    "dataset_type",
    [
        pytest.param("spark", marks=pytest.mark.spark),
        pytest.param("pandas", marks=pytest.mark.core),
    ],
)
def test_interaction_entries_filter_max_user_interact(dataset_type, request):
    columns = ["user_id", "item_id"]
    data = [(1, 1), (1, 2), (3, 3)]

    if dataset_type == "spark":
        spark_session = request.getfixturevalue("spark_session")
        test_dataframe = spark_session.createDataFrame(data, schema=columns)
    else:
        test_dataframe = PandasDataFrame(data, columns=columns)

    filtered_dataframe = MinMaxInteractionsFilter(max_inter_per_user=1).transform(test_dataframe)

    if dataset_type == "spark":
        user_list = filtered_dataframe.select("user_id").distinct().collect()
        assert len([x[0] for x in user_list]) == 1
        assert user_list[0][0] == 3
    else:
        user_list = filtered_dataframe["user_id"].unique().tolist()
        assert len(user_list) == 1
        assert user_list[0] == 3


@pytest.mark.parametrize(
    "dataset_type",
    [
        pytest.param("spark", marks=pytest.mark.spark),
        pytest.param("pandas", marks=pytest.mark.core),
    ],
)
def test_interaction_entries_filter_min_item_interact(dataset_type, request):
    columns = ["user_id", "item_id"]
    data = [(1, 1), (1, 1), (3, 3)]

    if dataset_type == "spark":
        spark_session = request.getfixturevalue("spark_session")
        test_dataframe = spark_session.createDataFrame(data, schema=columns)
    else:
        test_dataframe = PandasDataFrame(data, columns=columns)

    filtered_dataframe = MinMaxInteractionsFilter(min_inter_per_item=2).transform(test_dataframe)

    if dataset_type == "spark":
        user_list = filtered_dataframe.select("item_id").distinct().collect()
        assert len([x[0] for x in user_list]) == 1
        assert user_list[0][0] == 1
    else:
        user_list = filtered_dataframe["item_id"].unique().tolist()
        assert len(user_list) == 1
        assert user_list[0] == 1


@pytest.mark.parametrize(
    "dataset_type",
    [
        pytest.param("spark", marks=pytest.mark.spark),
        pytest.param("pandas", marks=pytest.mark.core),
    ],
)
def test_interaction_entries_filter_max_item_interact(dataset_type, request):
    columns = ["user_id", "item_id"]
    data = [(1, 1), (1, 1), (3, 4)]

    if dataset_type == "spark":
        spark_session = request.getfixturevalue("spark_session")
        test_dataframe = spark_session.createDataFrame(data, schema=columns)
    else:
        test_dataframe = PandasDataFrame(data, columns=columns)

    filtered_dataframe = MinMaxInteractionsFilter(max_inter_per_item=1).transform(test_dataframe)

    if dataset_type == "spark":
        item_list = filtered_dataframe.select("item_id").distinct().collect()
        assert len([x[0] for x in item_list]) == 1
        assert item_list[0][0] == 4
    else:
        item_list = filtered_dataframe["item_id"].unique().tolist()
        assert len(item_list) == 1
        assert item_list[0] == 4


@pytest.mark.parametrize(
    "dataset_type",
    [
        pytest.param("spark", marks=pytest.mark.spark),
        pytest.param("pandas", marks=pytest.mark.core),
    ],
)
def test_interaction_entries_filter_min_max_user_interact(dataset_type, request):
    columns = ["user_id", "item_id"]
    data = [(1, 1), (1, 1), (3, 3), (2, 1), (2, 2), (2, 3), (2, 4)]

    if dataset_type == "spark":
        spark_session = request.getfixturevalue("spark_session")
        test_dataframe = spark_session.createDataFrame(data, schema=columns)
    else:
        test_dataframe = PandasDataFrame(data, columns=columns)

    filtered_dataframe = MinMaxInteractionsFilter(min_inter_per_user=2, max_inter_per_user=3).transform(test_dataframe)

    if dataset_type == "spark":
        user_list = filtered_dataframe.select("user_id").distinct().collect()
        assert len([x[0] for x in user_list]) == 1
        assert user_list[0][0] == 1
    else:
        user_list = filtered_dataframe["user_id"].unique().tolist()
        assert len(user_list) == 1
        assert user_list[0] == 1


@pytest.mark.parametrize(
    "dataset_type",
    [
        pytest.param("spark", marks=pytest.mark.spark),
        pytest.param("pandas", marks=pytest.mark.core),
    ],
)
def test_interaction_entries_filter_min_max_item_interact(dataset_type, request):
    columns = ["item_id", "user_id"]
    data = [(1, 1), (1, 1), (3, 3), (2, 1), (2, 2), (2, 3), (2, 4)]

    if dataset_type == "spark":
        spark_session = request.getfixturevalue("spark_session")
        test_dataframe = spark_session.createDataFrame(data, schema=columns)
    else:
        test_dataframe = PandasDataFrame(data, columns=columns)

    filtered_dataframe = MinMaxInteractionsFilter(min_inter_per_item=2, max_inter_per_item=3).transform(test_dataframe)

    if dataset_type == "spark":
        item_list = filtered_dataframe.select("item_id").distinct().collect()
        assert len([x[0] for x in item_list]) == 1
        assert item_list[0][0] == 1
    else:
        item_list = filtered_dataframe["item_id"].unique().tolist()
        assert len(item_list) == 1
        assert item_list[0] == 1


@pytest.mark.parametrize(
    "dataset_type",
    [
        pytest.param("spark", marks=pytest.mark.spark),
        pytest.param("pandas", marks=pytest.mark.core),
    ],
)
def test_interaction_entries_filter_min_max_item_iterative_interact(dataset_type, request):
    columns = ["user_id", "item_id"]
    data = [(1, 1), (2, 1), (2, 2), (3, 1), (3, 3), (3, 4), (4, 1), (4, 3), (4, 4)]

    if dataset_type == "spark":
        spark_session = request.getfixturevalue("spark_session")
        test_dataframe = spark_session.createDataFrame(data, schema=columns)
    else:
        test_dataframe = PandasDataFrame(data, columns=columns)

    filtered_dataframe = MinMaxInteractionsFilter(
        min_inter_per_user=2,
        min_inter_per_item=2,
    ).transform(test_dataframe)

    if dataset_type == "spark":
        user_list = [user_id[0] for user_id in filtered_dataframe.select("user_id").distinct().collect()]
        item_list = [user_id[0] for user_id in filtered_dataframe.select("item_id").distinct().collect()]
    else:
        user_list = filtered_dataframe["user_id"].unique().tolist()
        item_list = filtered_dataframe["item_id"].unique().tolist()

    assert set(user_list) == set([3, 4])
    assert set(item_list) == set([1, 3, 4])
