from scipy.sparse import csr_matrix

from replay.models.extensions.ann.entities.nmslib_hnsw_param import NmslibHnswParam
from replay.models.extensions.ann.index_stores.base_index_store import IndexStore
from replay.models.extensions.ann.utils import create_nmslib_index_instance
from replay.utils import PandasDataFrame, DataFrameLike


# pylint: disable=too-few-public-methods
class NmslibIndexBuilderMixin:
    """Provides nmslib index building method for different nmslib index builders"""

    @staticmethod
    def build_and_save_index(
        pdf: DataFrameLike,
        index_params: NmslibHnswParam,
        index_store: IndexStore,
    ):
        """
        Builds nmslib index and saves it to index storage.
        This function is implemented to be able to test
        the `build_index_udf` internal functionality outside of spark,
        because pytest does not see this function call if the function is called by spark.
        :param pdf: pandas dataframe containing item similarities,
         with the following columns: `item_idx_one`, `item_idx_two`, `similarity`.
        :param index_params: index parameters as instance of NmslibHnswParam.
        :param index_store: index store
        :return:
        """
        creation_index_params = {
            "M": index_params.m,
            "efConstruction": index_params.ef_c,
            "post": index_params.post,
        }

        index = create_nmslib_index_instance(index_params)

        if isinstance(pdf, PandasDataFrame):
            data = pdf["similarity"].values
            row_ind = pdf["item_idx_two"].values
            col_ind = pdf["item_idx_one"].values
        else:
            data = [x["similarity"] for x in pdf.select("similarity").collect()]
            row_ind = [x["item_idx_two"] for x in pdf.select("item_idx_two").collect()]
            col_ind = [x["item_idx_one"] for x in pdf.select("item_idx_one").collect()]

        sim_matrix = csr_matrix(
            (data, (row_ind, col_ind)),
            shape=(
                index_params.items_count,
                index_params.items_count,
            ),
        )
        index.addDataPointBatch(data=sim_matrix)
        index.createIndex(creation_index_params)

        index_store.save_to_store(
            lambda path: index.saveIndex(path, save_data=True)
        )  # pylint: disable=unnecessary-lambda)
