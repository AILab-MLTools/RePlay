import pandas as pd
import pytest

from replay.utils import PYSPARK_AVAILABLE
from replay.utils.utils import get_spark_session

if PYSPARK_AVAILABLE:
    from pyspark.sql.types import IntegerType, StructType


@pytest.fixture(scope="session")
def spark_session(request):
    spark = get_spark_session(enable_hive_support=False)
    request.addfinalizer(lambda: spark.sparkContext.stop())
    return spark
