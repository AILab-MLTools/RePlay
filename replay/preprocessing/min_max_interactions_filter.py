from typing import Optional, Tuple

import pyspark.sql.functions as sf
from pyspark.sql.window import Window
from pandas import DataFrame as PandasDataFrame
from pyspark.sql import DataFrame as SparkDataFrame, Window

from replay.data import AnyDataFrame


class MinMaxInteractionsFilter:
    """
    Filter interactions with minimal and maximum constraints

    >>> import pandas as pd
    >>> interactions = pd.DataFrame({
    ...    "query_id": [1, 1, 1, 2, 2, 2, 3, 3, 3, 3],
    ...    "item_id": [3, 7, 10, 5, 8, 11, 4, 9, 2, 5],
    ...    "rating": [1, 2, 3, 3, 2, 1, 3, 12, 1, 4],
    ... })
    >>> interactions
      query_id  item_id  rating
    0        1        3       1
    1        1        7       2
    2        1       10       3
    3        2        5       3
    4        2        8       2
    5        2       11       1
    6        3        4       3
    7        3        9      12
    8        3        2       1
    9        3        5       4
    >>> filtered_interactions = MinMaxInteractionsFilter(min_inter_per_user=4).transform(interactions)
    >>> filtered_interactions
      query_id  item_id  rating
    6        3        4       3
    7        3        9      12
    8        3        2       1
    9        3        5       4
    <BLANKLINE>
    """

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        query_column: str = "query_id",
        item_column: str = "item_id",
        min_inter_per_user: Optional[int] = None,
        max_inter_per_user: Optional[int] = None,
        min_inter_per_item: Optional[int] = None,
        max_inter_per_item: Optional[int] = None,
    ):
        r"""
        Args:
            query_column (str): Name of user interaction column,
                default: ``query_id``.
            item_column (str): Name of item interaction column,
                default: ``item_id``.
            min_inter_per_user (int, optional): Minimum positive value of
                interactions per user. If None, filter doesn't apply,
                default: ``None``.
            max_inter_per_user (int, optional): Maximum positive value of
                interactions per user. If None, filter doesn't apply. Must be
                less than `min_inter_per_user`, default: ``None``.
            min_inter_per_item (int, optional): Minimum positive value of
                interactions per item. If None, filter doesn't apply,
                default: ``None``.
            max_inter_per_item (int, optional): Maximum positive value of
                interactions per item. If None, filter doesn't apply. Must be
                less than `min_inter_per_item`, default: ``None``.
        """
        self.query_column = query_column
        self.item_column = item_column
        self.min_inter_per_user = min_inter_per_user
        self.max_inter_per_user = max_inter_per_user
        self.min_inter_per_item = min_inter_per_item
        self.max_inter_per_item = max_inter_per_item
        self.total_dropped_interactions = 0
        self._sanity_check()

    def _sanity_check(self) -> None:
        if self.min_inter_per_user:
            assert self.min_inter_per_user > 0
        if self.min_inter_per_item:
            assert self.min_inter_per_item > 0
        if self.min_inter_per_user and self.max_inter_per_user:
            assert self.min_inter_per_user < self.max_inter_per_user
        if self.min_inter_per_item and self.max_inter_per_item:
            assert self.min_inter_per_item < self.max_inter_per_item

    def _filter_column(
        self,
        interactions: AnyDataFrame,
        column: str,
        interaction_count: int,
    ) -> Tuple[AnyDataFrame, int, int]:
        if column == "user":
            min_inter = self.min_inter_per_user
            max_inter = self.max_inter_per_user
            agg_column = self.query_column
            non_agg_column = self.item_column
        else:
            min_inter = self.min_inter_per_item
            max_inter = self.max_inter_per_item
            agg_column = self.item_column
            non_agg_column = self.query_column

        if not min_inter and not max_inter:
            return interactions, 0, interaction_count

        if isinstance(interactions, SparkDataFrame):
            return self._filter_column_spark(
                interactions, interaction_count, min_inter, max_inter, agg_column, non_agg_column
            )

        return self._filter_column_pandas(
            interactions, interaction_count, min_inter, max_inter, agg_column, non_agg_column
        )

    def _filter_column_pandas(
        self,
        interactions: PandasDataFrame,
        interaction_count: int,
        min_inter: Optional[int],
        max_inter: Optional[int],
        agg_column: str,
        non_agg_column: str,
    ) -> Tuple[PandasDataFrame, int, int]:
        filtered_interactions = interactions.copy(deep=True)

        filtered_interactions["count"] = filtered_interactions.groupby(agg_column, sort=False)[
            non_agg_column
        ].transform(len)
        if min_inter:
            filtered_interactions = filtered_interactions[filtered_interactions["count"] >= min_inter]
        if max_inter:
            filtered_interactions = filtered_interactions[filtered_interactions["count"] <= max_inter]
        filtered_interactions.drop(columns=["count"], inplace=True)

        end_len_dataframe = len(filtered_interactions)
        different_len = interaction_count - end_len_dataframe

        return filtered_interactions, different_len, end_len_dataframe

    def _filter_column_spark(
        self,
        interactions: SparkDataFrame,
        interaction_count: int,
        min_inter: Optional[int],
        max_inter: Optional[int],
        agg_column: str,
        non_agg_column: str,
    ) -> Tuple[SparkDataFrame, int, int]:
        filtered_interactions = interactions.withColumn(
            "count", sf.count(non_agg_column).over(Window.partitionBy(agg_column))
        )
        if min_inter:
            filtered_interactions = filtered_interactions.filter(sf.col("count") >= min_inter)
        if max_inter:
            filtered_interactions = filtered_interactions.filter(sf.col("count") <= max_inter)
        filtered_interactions = filtered_interactions.drop("count")

        filtered_interactions.cache()
        end_len_dataframe = filtered_interactions.count()
        interactions.unpersist()
        different_len = interaction_count - end_len_dataframe

        return filtered_interactions, different_len, end_len_dataframe

    def transform(self, interactions: AnyDataFrame) -> AnyDataFrame:
        r"""Filter interactions.

        Args:
            interactions (Union[SparkDataFrame, PandasDataFrame]): DataFrame containing columns
                ``query_column``, ``item_column``.

        Returns:
            Union[SparkDataFrame, PandasDataFrame]: filtered DataFrame.
        """
        is_no_dropped_user_item = [False, False]
        current_index = 0
        interaction_count = interactions.count() if isinstance(interactions, SparkDataFrame) else len(interactions)
        while True:
            column = "user" if current_index == 0 else "item"
            interactions, dropped_interact, interaction_count = self._filter_column(
                interactions, column, interaction_count
            )
            is_no_dropped_user_item[current_index] = not dropped_interact
            current_index ^= 1
            if is_no_dropped_user_item[0] and is_no_dropped_user_item[1]:
                break
        return interactions
