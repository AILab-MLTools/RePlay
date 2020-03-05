import doctest
from unittest import TestSuite, TextTestRunner

import pandas as pd
from tests.pyspark_testcase import PySparkTest

from sponge_bob_magic import experiment

suite = TestSuite()
suite.addTests(doctest.DocTestSuite(experiment))


class TestConverter(PySparkTest):
    def runTest(self):
        with self.assertRaises(TypeError):
            experiment.Experiment(
                pd.DataFrame({"user_id": [1, 1, 1],
                              "item_id": [1, 2, 3],
                              "relevance": [5, 3, 4]}),
                "ᕕ( ᐛ )ᕗ"
            )


suite.addTest(TestConverter())
runner = TextTestRunner()
runner.run(suite)
