from replay.experimental.metrics.base_metric import ScalaMetric


# pylint: disable=too-few-public-methods
class ScalaMRR(ScalaMetric):
    """
    Mean Reciprocal Rank --
    Reciprocal Rank is the inverse position of the first relevant item among top-k recommendations,
    :math:`\\frac {1}{rank_i}`. This value is averaged by all users.

    >>> import pandas as pd
    >>> pred = pd.DataFrame({"user_idx": [1, 1, 1], "item_idx": [3, 2, 1], "relevance": [5 ,5, 5]})
    >>> true = pd.DataFrame({"user_idx": [1, 1, 1], "item_idx": [2, 4, 5], "relevance": [5, 5, 5]})
    >>> MRR()(pred, true, 3)
    0.5
    >>> MRR()(pred, true, 1)
    0.0
    >>> MRR()(true, pred, 1)
    1.0
    """

    _scala_udf_name = "getMRRMetricValue"
