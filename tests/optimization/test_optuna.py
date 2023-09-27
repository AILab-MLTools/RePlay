# pylint: disable=redefined-outer-name, missing-function-docstring, unused-import
import pytest

from replay.models import SLIM, ItemKNN
from tests.utils import log, spark


@pytest.mark.parametrize("borders", [{"beta": [1, 2]}, {"lambda_": [1, 2]}])
def test_partial_borders(borders):
    model = SLIM()
    res = model._prepare_param_borders(borders)
    assert len(res) == len(model._search_space)


def test_ItemKNN(log):
    model = ItemKNN()
    res = model.optimize(log, log, k=2, budget=1)
    assert isinstance(res["num_neighbours"], int)
