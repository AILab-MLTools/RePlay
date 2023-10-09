from typing import Optional

from pyspark.sql import DataFrame

from replay.data import AnyDataFrame
from replay.experimental.metrics.base_metric import ScalaRecOnlyMetric
from replay.metrics.base_metric import (
    fill_na_with_empty_array,
    filter_sort
)
from replay.utils.spark_utils import convert2spark, get_top_k_recs


# pylint: disable=too-few-public-methods
class ScalaUnexpectedness(ScalaRecOnlyMetric):
    """
    Fraction of recommended items that are not present in some baseline recommendations.
    """

    _scala_udf_name = "getUnexpectednessMetricValue"

    def __init__(
        self, pred: AnyDataFrame,
    ):  # pylint: disable=super-init-not-called
        """
        :param pred: model predictions
        """
        self.pred = convert2spark(pred)

    def _get_enriched_recommendations(
        self,
        recommendations: DataFrame,
        ground_truth: DataFrame,
        max_k: int,
        ground_truth_users: Optional[AnyDataFrame] = None,
    ) -> DataFrame:
        recommendations = convert2spark(recommendations)
        ground_truth_users = convert2spark(ground_truth_users)
        base_pred = self.pred

        # TO DO: preprocess base_recs once in __init__

        base_recs = filter_sort(base_pred).withColumnRenamed("pred", "base_pred")

        # if there are duplicates in recommendations,
        # we will leave fewer than k recommendations after sort_udf
        recommendations = get_top_k_recs(recommendations, k=max_k)
        recommendations = filter_sort(recommendations)
        recommendations = recommendations.join(base_recs, how="right", on=["user_idx"])

        if ground_truth_users is not None:
            recommendations = recommendations.join(
                ground_truth_users, on="user_idx", how="right"
            )

        return fill_na_with_empty_array(
            recommendations, "pred", base_pred.schema["item_idx"].dataType
        )
