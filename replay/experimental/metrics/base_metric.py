# pylint: disable=R0801
"""
Base classes for quality and diversity metrics.
"""
from abc import abstractmethod
from typing import Dict, List, Union, Optional

from pyspark.sql import Column
from pyspark.sql import DataFrame
from pyspark.sql import functions as sf
from pyspark.sql.column import _to_java_column, _to_seq

from replay.metrics.base_metric import Metric, NCISMetric
from replay.data import AnyDataFrame, IntOrList, NumType
from replay.utils.session_handler import State
from replay.utils.spark_utils import convert2spark


class ScalaMetric(Metric):
    """Base scala metric class"""

    _scala_udf_name: Optional[str] = None

    @property
    def scala_udf_name(self) -> str:
        """Returns UDF name from `org.apache.spark.replay.utils.ScalaPySparkUDFs`"""
        if self._scala_udf_name:
            return self._scala_udf_name
        else:
            raise NotImplementedError(f"Scala UDF not implemented for {type(self).__name__} class!")

    def _get_metric_distribution(self, recs: DataFrame, k: int) -> DataFrame:
        """
        :param recs: recommendations
        :param k: depth cut-off
        :return: metric distribution for different cut-offs and users
        """
        metric_value_col = self.get_scala_udf(
            self.scala_udf_name, [sf.lit(k).alias("k"), *recs.columns[1:]]
        ).alias("value")
        return recs.select("user_idx", metric_value_col)

    @staticmethod
    def _get_metric_value_by_user(k, pred, ground_truth) -> float:
        """
        Metric calculation for one user.

        :param k: depth cut-off
        :param pred: recommendations
        :param ground_truth: test data
        :return: metric value for current user
        """
        return None

    @staticmethod
    def get_scala_udf(udf_name: str, params: List) -> Column:
        """
        Returns expression of calling scala UDF as column

        :param udf_name: UDF name from `org.apache.spark.replay.utils.ScalaPySparkUDFs`
        :param params: list of UDF params in right order
        :return: column expression
        """
        sc = State().session.sparkContext  # pylint: disable=invalid-name
        scala_udf = getattr(
            sc._jvm.org.apache.spark.replay.utils.ScalaPySparkUDFs, udf_name
        )()
        return Column(scala_udf.apply(_to_seq(sc, params, _to_java_column)))


# pylint: disable=too-few-public-methods
class ScalaRecOnlyMetric(ScalaMetric):
    """Base class for metrics that do not need holdout data"""

    @abstractmethod
    def __init__(self, log: AnyDataFrame, *args, **kwargs):  # pylint: disable=super-init-not-called
        pass

    # pylint: disable=no-self-use
    @abstractmethod
    def _get_enriched_recommendations(
        self,
        recommendations: AnyDataFrame,
        ground_truth: Optional[AnyDataFrame],
        max_k: int,
        ground_truth_users: Optional[AnyDataFrame] = None,
    ) -> DataFrame:
        pass

    def __call__(
        self,
        recommendations: AnyDataFrame,
        k: IntOrList,
        ground_truth_users: Optional[AnyDataFrame] = None,
    ) -> Union[Dict[int, NumType], NumType]:
        """
        :param recommendations: predictions of a model,
            DataFrame  ``[user_idx, item_idx, relevance]``
        :param k: depth cut-off
        :param ground_truth_users: list of users to consider in metric calculation.
            if None, only the users from ground_truth are considered.
        :return: metric value
        """
        recs = self._get_enriched_recommendations(
            recommendations,
            None,
            max_k=k if isinstance(k, int) else max(k),
            ground_truth_users=ground_truth_users,
        )
        return self._mean(recs, k)


class ScalaNCISMetric(NCISMetric, ScalaMetric):
    """
    Normalized capped importance sampling, where each recommendation is being weighted
    by the ratio of current policy score on previous policy score.
    The weight is also capped by some threshold value.

    Source: arxiv.org/abs/1801.07030
    """

    def __init__(
        self,
        prev_policy_weights: AnyDataFrame,
        threshold: float = 10.0,
        activation: Optional[str] = None,
    ):  # pylint: disable=super-init-not-called
        """
        :param prev_policy_weights: historical item of user-item relevance (previous policy values)
        :threshold: capping threshold, applied after activation,
            relevance values are cropped to interval [1/`threshold`, `threshold`]
        :activation: activation function, applied over relevance values.
            "logit"/"sigmoid", "softmax" or None
        """
        self.prev_policy_weights = convert2spark(
            prev_policy_weights
        ).withColumnRenamed("relevance", "prev_relevance")
        self.threshold = threshold
        if activation is None or activation in ("logit", "sigmoid", "softmax"):
            self.activation = activation
            if activation == "softmax":
                self.logger.info(
                    "For accurate softmax calculation pass only one `k` value "
                    "in the NCISMetric metrics `call`"
                )
        else:
            raise ValueError(f"Unexpected `activation` - {activation}")
        if threshold <= 0:
            raise ValueError("Threshold should be positive real number")
