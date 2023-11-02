"""
Select or remove data by some criteria
"""
from datetime import datetime, timedelta
from typing import Union, Optional, Tuple
from abc import ABC, abstractmethod
from replay.data import AnyDataFrame

from pyspark.sql import DataFrame as SparkDataFrame, Window, functions as sf
from pyspark.sql.functions import col
from pyspark.sql.types import TimestampType
from pandas import DataFrame as PandasDataFrame


# pylint: disable=too-few-public-methods
class BaseFilter(ABC):
    """Base filter class"""

    @abstractmethod
    def transform(self, interactions: AnyDataFrame) -> AnyDataFrame:    # pragma: no cover
        """
        Performs filter transformation

        :param interactions: input dataframe to filter
        :returns: filtered dataframe
        """


class MinMaxInteractionsFilter(BaseFilter):
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
        :param query_column: Name of user interaction column,
            default: ``query_id``.
        :param item_column: Name of item interaction column,
            default: ``item_id``.
        :param min_inter_per_user: Minimum positive value of
            interactions per user. If None, filter doesn't apply,
            default: ``None``.
        :param max_inter_per_user: Maximum positive value of
            interactions per user. If None, filter doesn't apply. Must be
            less than `min_inter_per_user`, default: ``None``.
        :param min_inter_per_item: Minimum positive value of
            interactions per item. If None, filter doesn't apply,
            default: ``None``.
        :param max_inter_per_item: Maximum positive value of
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

    # pylint: disable=no-self-use
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

    # pylint: disable=no-self-use
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

        :param interactions: DataFrame containing columns
            ``query_column``, ``item_column``.

        :returns: filtered DataFrame.
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


class MinMaxValuesFilter(BaseFilter):
    """
    Remove records with records less or greater than ``value`` in ``column``.

    >>> import pandas as pd
    >>> from replay.utils.spark_utils import convert2spark
    >>> data_frame = convert2spark(pd.DataFrame({"relevance": [1, 5, 3.5, 4]}))
    >>> MinMaxValuesFilter("relevance", min_column_value=3.5).transform(data_frame).show()
    +---------+
    |relevance|
    +---------+
    |      5.0|
    |      3.5|
    |      4.0|
    +---------+
    >>> MinMaxValuesFilter("relevance", max_column_value=3.5).transform(data_frame).show()
    +---------+
    |relevance|
    +---------+
    |      1.0|
    |      3.5|
    +---------+
    >>> MinMaxValuesFilter("relevance", min_column_value=3.5, max_column_value=4).transform(data_frame).show()
    +---------+
    |relevance|
    +---------+
    |      3.5|
    |      4.0|
    +---------+
    <BLANKLINE>
    """

    def __init__(
        self,
        column: str,
        min_column_value: Optional[Union[int, float]] = None,
        max_column_value: Optional[Union[int, float]] = None,
    ):
        """
        :param column: the column in which filtering is performed.
        :param min_column_value: minimum threshold value of column
        :param max_column_value: maximum threshold value of column
        """
        self.column = column
        self.min_column_value = min_column_value
        self.max_column_value = max_column_value

    def transform(self, interactions: SparkDataFrame) -> SparkDataFrame:
        if self.min_column_value:
            interactions = interactions.filter(interactions[self.column] >= self.min_column_value)
        if self.max_column_value:
            interactions = interactions.filter(interactions[self.column] <= self.max_column_value)

        return interactions


class FirstLastInteractionsUserFilter(BaseFilter):
    """
    Get first/last ``num_interactions`` interactions for each user.

    >>> import pandas as pd
    >>> from replay.utils.spark_utils import convert2spark
    >>> log_pd = pd.DataFrame({"query_id": ["u1", "u2", "u2", "u3", "u3", "u3"],
    ...                     "item_id": ["i1", "i2","i3", "i1", "i2","i3"],
    ...                     "rel": [1., 0.5, 3, 1, 0, 1],
    ...                     "timestamp": ["2020-01-01 23:59:59", "2020-02-01",
    ...                                   "2020-02-01", "2020-01-01 00:04:15",
    ...                                   "2020-01-02 00:04:14", "2020-01-05 23:59:59"]},
    ...             )
    >>> log_pd["timestamp"] = pd.to_datetime(log_pd["timestamp"])
    >>> log_sp = convert2spark(log_pd)
    >>> log_sp.show()
    +--------+-------+---+-------------------+
    |query_id|item_id|rel|          timestamp|
    +--------+-------+---+-------------------+
    |      u1|     i1|1.0|2020-01-01 23:59:59|
    |      u2|     i2|0.5|2020-02-01 00:00:00|
    |      u2|     i3|3.0|2020-02-01 00:00:00|
    |      u3|     i1|1.0|2020-01-01 00:04:15|
    |      u3|     i2|0.0|2020-01-02 00:04:14|
    |      u3|     i3|1.0|2020-01-05 23:59:59|
    +--------+-------+---+-------------------+
    <BLANKLINE>

    Only first interaction:

    >>> FirstLastInteractionsUserFilter(1, True).transform(log_sp).orderBy('query_id').show()
    +--------+-------+---+-------------------+
    |query_id|item_id|rel|          timestamp|
    +--------+-------+---+-------------------+
    |      u1|     i1|1.0|2020-01-01 23:59:59|
    |      u2|     i2|0.5|2020-02-01 00:00:00|
    |      u3|     i1|1.0|2020-01-01 00:04:15|
    +--------+-------+---+-------------------+
    <BLANKLINE>

    Only last interaction:

    >>> FirstLastInteractionsUserFilter(1, False, item_column=None).transform(log_sp).orderBy('query_id').show()
    +--------+-------+---+-------------------+
    |query_id|item_id|rel|          timestamp|
    +--------+-------+---+-------------------+
    |      u1|     i1|1.0|2020-01-01 23:59:59|
    |      u2|     i2|0.5|2020-02-01 00:00:00|
    |      u3|     i3|1.0|2020-01-05 23:59:59|
    +--------+-------+---+-------------------+
    <BLANKLINE>

    >>> FirstLastInteractionsUserFilter(1, False).transform(log_sp).orderBy('query_id').show()
    +--------+-------+---+-------------------+
    |query_id|item_id|rel|          timestamp|
    +--------+-------+---+-------------------+
    |      u1|     i1|1.0|2020-01-01 23:59:59|
    |      u2|     i3|3.0|2020-02-01 00:00:00|
    |      u3|     i3|1.0|2020-01-05 23:59:59|
    +--------+-------+---+-------------------+
    <BLANKLINE>
    """

    def __init__(
        self,
        num_interactions: int = 10,
        first: bool = True,
        timestamp_column: str = "timestamp",
        query_column: str = "query_id",
        item_column: Optional[str] = "item_id",
    ):
        """
        :param num_interactions: number of interactions to leave per user
        :param first: take either first ``num_interactions`` or last.
        :param timestamp_column: timestamp column
        :param query_column: query column
        :param item_column: item column to help sort simultaneous interactions.
            If None, it is ignored.
        :return: filtered DataFrame
        """
        self.num_interactions = num_interactions
        self.first = first
        self.timestamp_column = timestamp_column
        self.query_column = query_column
        self.item_column = item_column

    def transform(self, interactions: SparkDataFrame) -> SparkDataFrame:
        sorting_order = [col(self.timestamp_column)]
        if self.item_column is not None:
            sorting_order.append(col(self.item_column))

        if not self.first:
            sorting_order = [col_.desc() for col_ in sorting_order]

        window = Window().orderBy(*sorting_order).partitionBy(col(self.query_column))

        return (
            interactions.withColumn("temp_rank", sf.row_number().over(window))
            .filter(col("temp_rank") <= self.num_interactions)
            .drop("temp_rank")
        )


class FirstLastDaysUserFilter(BaseFilter):
    """
    Get first/last ``days`` of users' interactions.

    >>> import pandas as pd
    >>> from replay.utils.spark_utils import convert2spark
    >>> log_pd = pd.DataFrame({"query_id": ["u1", "u2", "u2", "u3", "u3", "u3"],
    ...                     "item_id": ["i1", "i2","i3", "i1", "i2","i3"],
    ...                     "rel": [1., 0.5, 3, 1, 0, 1],
    ...                     "timestamp": ["2020-01-01 23:59:59", "2020-02-01",
    ...                                   "2020-02-01", "2020-01-01 00:04:15",
    ...                                   "2020-01-02 00:04:14", "2020-01-05 23:59:59"]},
    ...             )
    >>> log_pd["timestamp"] = pd.to_datetime(log_pd["timestamp"])
    >>> log_sp = convert2spark(log_pd)
    >>> log_sp.show()
    +--------+-------+---+-------------------+
    |query_id|item_id|rel|          timestamp|
    +--------+-------+---+-------------------+
    |      u1|     i1|1.0|2020-01-01 23:59:59|
    |      u2|     i2|0.5|2020-02-01 00:00:00|
    |      u2|     i3|3.0|2020-02-01 00:00:00|
    |      u3|     i1|1.0|2020-01-01 00:04:15|
    |      u3|     i2|0.0|2020-01-02 00:04:14|
    |      u3|     i3|1.0|2020-01-05 23:59:59|
    +--------+-------+---+-------------------+
    <BLANKLINE>

    Get first day:

    >>> FirstLastDaysUserFilter(1, True).transform(log_sp).orderBy('query_id', 'item_id').show()
    +--------+-------+---+-------------------+
    |query_id|item_id|rel|          timestamp|
    +--------+-------+---+-------------------+
    |      u1|     i1|1.0|2020-01-01 23:59:59|
    |      u2|     i2|0.5|2020-02-01 00:00:00|
    |      u2|     i3|3.0|2020-02-01 00:00:00|
    |      u3|     i1|1.0|2020-01-01 00:04:15|
    |      u3|     i2|0.0|2020-01-02 00:04:14|
    +--------+-------+---+-------------------+
    <BLANKLINE>

    Get last day:

    >>> FirstLastDaysUserFilter(1, False).transform(log_sp).orderBy('query_id', 'item_id').show()
    +--------+-------+---+-------------------+
    |query_id|item_id|rel|          timestamp|
    +--------+-------+---+-------------------+
    |      u1|     i1|1.0|2020-01-01 23:59:59|
    |      u2|     i2|0.5|2020-02-01 00:00:00|
    |      u2|     i3|3.0|2020-02-01 00:00:00|
    |      u3|     i3|1.0|2020-01-05 23:59:59|
    +--------+-------+---+-------------------+
    <BLANKLINE>
    
    """

    def __init__(
        self,
        days: int = 10,
        first: bool = True,
        timestamp_column: str = "timestamp",
        query_column: str = "query_id",
    ):
        """
        :param days: how many days to return per user
        :param first: take either first ``days`` or last
        :param timestamp_column: timestamp column
        :param query_column: query column
        :return: filtered DataFrame
        """
        self.days = days
        self.first = first
        self.timestamp_column = timestamp_column
        self.query_column = query_column

    def transform(self, interactions: SparkDataFrame) -> SparkDataFrame:
        window = Window.partitionBy(self.query_column)
        if self.first:
            return (
                interactions.withColumn("min_date", sf.min(col(self.timestamp_column)).over(window))
                .filter(
                    col(self.timestamp_column)
                    < col("min_date") + sf.expr(f"INTERVAL {self.days} days")
                )
                .drop("min_date")
            )

        return (
            interactions.withColumn("max_date", sf.max(col(self.timestamp_column)).over(window))
            .filter(
                col(self.timestamp_column) > col("max_date") - sf.expr(f"INTERVAL {self.days} days")
            )
            .drop("max_date")
        )


def take_time_period(
    log: SparkDataFrame,
    start_date: Optional[Union[str, datetime]] = None,
    end_date: Optional[Union[str, datetime]] = None,
    date_column: str = "timestamp",
) -> SparkDataFrame:
    """
    Select a part of data between ``[start_date, end_date)``.

    >>> import pandas as pd
    >>> from replay.utils.spark_utils import convert2spark
    >>> log_pd = pd.DataFrame({"user_idx": ["u1", "u2", "u2", "u3", "u3", "u3"],
    ...                     "item_idx": ["i1", "i2","i3", "i1", "i2","i3"],
    ...                     "rel": [1., 0.5, 3, 1, 0, 1],
    ...                     "timestamp": ["2020-01-01 23:59:59", "2020-02-01",
    ...                                   "2020-02-01", "2020-01-01 00:04:15",
    ...                                   "2020-01-02 00:04:14", "2020-01-05 23:59:59"]},
    ...             )
    >>> log_pd["timestamp"] = pd.to_datetime(log_pd["timestamp"])
    >>> log_sp = convert2spark(log_pd)
    >>> log_sp.show()
    +--------+--------+---+-------------------+
    |user_idx|item_idx|rel|          timestamp|
    +--------+--------+---+-------------------+
    |      u1|      i1|1.0|2020-01-01 23:59:59|
    |      u2|      i2|0.5|2020-02-01 00:00:00|
    |      u2|      i3|3.0|2020-02-01 00:00:00|
    |      u3|      i1|1.0|2020-01-01 00:04:15|
    |      u3|      i2|0.0|2020-01-02 00:04:14|
    |      u3|      i3|1.0|2020-01-05 23:59:59|
    +--------+--------+---+-------------------+
    <BLANKLINE>

    >>> take_time_period(log_sp, start_date="2020-01-01 14:00:00", end_date=datetime(2020, 1, 3, 0, 0, 0)).show()
    +--------+--------+---+-------------------+
    |user_idx|item_idx|rel|          timestamp|
    +--------+--------+---+-------------------+
    |      u1|      i1|1.0|2020-01-01 23:59:59|
    |      u3|      i2|0.0|2020-01-02 00:04:14|
    +--------+--------+---+-------------------+
    <BLANKLINE>

    :param log: historical DataFrame
    :param start_date: datetime or str with format "yyyy-MM-dd HH:mm:ss".
    :param end_date: datetime or str with format "yyyy-MM-dd HH:mm:ss".
    :param date_column: date column
    """
    if start_date is None:
        start_date = log.agg(sf.min(date_column)).first()[0]
    if end_date is None:
        end_date = log.agg(sf.max(date_column)).first()[0] + timedelta(
            seconds=1
        )

    return log.filter(
        (col(date_column) >= sf.lit(start_date).cast(TimestampType()))
        & (col(date_column) < sf.lit(end_date).cast(TimestampType()))
    )


def take_num_days_of_global_hist(
    log: SparkDataFrame,
    duration_days: int,
    first: bool = True,
    date_column: str = "timestamp",
) -> SparkDataFrame:
    """
    Select first/last days from ``log``.

    >>> import pandas as pd
    >>> from replay.utils.spark_utils import convert2spark
    >>> log_pd = pd.DataFrame({"user_idx": ["u1", "u2", "u2", "u3", "u3", "u3"],
    ...                     "item_idx": ["i1", "i2","i3", "i1", "i2","i3"],
    ...                     "rel": [1., 0.5, 3, 1, 0, 1],
    ...                     "timestamp": ["2020-01-01 23:59:59", "2020-02-01",
    ...                                   "2020-02-01", "2020-01-01 00:04:15",
    ...                                   "2020-01-02 00:04:14", "2020-01-05 23:59:59"]},
    ...             )
    >>> log_pd["timestamp"] = pd.to_datetime(log_pd["timestamp"])
    >>> log_sp = convert2spark(log_pd)
    >>> log_sp.show()
    +--------+--------+---+-------------------+
    |user_idx|item_idx|rel|          timestamp|
    +--------+--------+---+-------------------+
    |      u1|      i1|1.0|2020-01-01 23:59:59|
    |      u2|      i2|0.5|2020-02-01 00:00:00|
    |      u2|      i3|3.0|2020-02-01 00:00:00|
    |      u3|      i1|1.0|2020-01-01 00:04:15|
    |      u3|      i2|0.0|2020-01-02 00:04:14|
    |      u3|      i3|1.0|2020-01-05 23:59:59|
    +--------+--------+---+-------------------+
    <BLANKLINE>

    >>> take_num_days_of_global_hist(log_sp, 1).show()
    +--------+--------+---+-------------------+
    |user_idx|item_idx|rel|          timestamp|
    +--------+--------+---+-------------------+
    |      u1|      i1|1.0|2020-01-01 23:59:59|
    |      u3|      i1|1.0|2020-01-01 00:04:15|
    |      u3|      i2|0.0|2020-01-02 00:04:14|
    +--------+--------+---+-------------------+
    <BLANKLINE>

    >>> take_num_days_of_global_hist(log_sp, 1, first=False).show()
    +--------+--------+---+-------------------+
    |user_idx|item_idx|rel|          timestamp|
    +--------+--------+---+-------------------+
    |      u2|      i2|0.5|2020-02-01 00:00:00|
    |      u2|      i3|3.0|2020-02-01 00:00:00|
    +--------+--------+---+-------------------+
    <BLANKLINE>

    :param log: historical DataFrame
    :param duration_days: length of selected data in days
    :param first: take either first ``duration_days`` or last
    :param date_column: date column
    """
    if first:
        start_date = log.agg(sf.min(date_column)).first()[0]
        end_date = sf.lit(start_date).cast(TimestampType()) + sf.expr(
            f"INTERVAL {duration_days} days"
        )
        return log.filter(col(date_column) < end_date)

    end_date = log.agg(sf.max(date_column)).first()[0]
    start_date = sf.lit(end_date).cast(TimestampType()) - sf.expr(
        f"INTERVAL {duration_days} days"
    )
    return log.filter(col(date_column) > start_date)
