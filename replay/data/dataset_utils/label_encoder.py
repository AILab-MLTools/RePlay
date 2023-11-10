# pylint: skip-file
import warnings
from typing import Dict, Iterable, Iterator, Optional, Sequence, Set, Union

from replay.data import Dataset, FeatureHint, FeatureSchema, FeatureSource
from replay.preprocessing import LabelEncoder, LabelEncodingRule
from replay.preprocessing.label_encoder import HandleUnknownStrategies


class LabelEncoderTransformWarning(Warning):
    pass


class DatasetLabelEncoder:
    """
    Categorical features encoder for the Dataset class
    """

    def __init__(
        self,
        handle_unknown_rule: HandleUnknownStrategies = "error",
        default_value_rule: Optional[Union[int, str]] = None,
    ) -> None:
        """
        Args:
            handle_unknown_rule (``error``, ``use_default_value``):
                When set to ``error`` an error will be raised in case an unknown label is present during transform.
                When set to ``use_default_value``, the encoded value of unknown label will be set
                to the value given for the parameter default_value.
                Default: ``error``.
            default_value (int, str, optional): Default value that will fill the unknown labels after transform.
                When the parameter handle_unknown is set to ``use_default_value``,
                this parameter is required and will set the encoded value of unknown labels.
                It has to be distinct from the values used to encode any of the labels in fit.
                If ``None``, then keep null.
                If ``int`` value, then fill by that value.
                If ``str`` value, should be \"last\" only, then fill by ``n_classes`` value.
                Default: ``None``.
        """
        self._handle_unknown_rule = handle_unknown_rule
        self._default_value_rule = default_value_rule
        self._encoding_rules: Dict[str, LabelEncodingRule] = {}

        self._features_columns: Dict[Union[FeatureHint, FeatureSource], Sequence[str]] = {}

    def fit(self, dataset: Dataset) -> "DatasetLabelEncoder":
        """Fits an encoder by the input Dataset for categorical features.

        Args:
            dataset (Dataset): The Dataset object.

        Returns:
            self (DatasetLabelEncoder): fitted encoder.

        Raises:
            AssertionError: if any of `dataset` categorical features contains
                invalid ``FeatureSource`` type.
        """

        self._fill_features_columns(dataset.feature_schema)
        for column, feature_info in dataset.feature_schema.categorical_features.items():
            encoding_rule = LabelEncodingRule(
                column, handle_unknown=self._handle_unknown_rule, default_value=self._default_value_rule
            )
            if feature_info.feature_hint == FeatureHint.QUERY_ID:
                if dataset.query_features is None:
                    encoding_rule.fit(dataset.interactions)
                else:
                    encoding_rule.fit(dataset.query_features)
            elif feature_info.feature_hint == FeatureHint.ITEM_ID:
                if dataset.item_features is None:
                    encoding_rule.fit(dataset.interactions)
                else:
                    encoding_rule.fit(dataset.item_features)
            elif feature_info.feature_source == FeatureSource.INTERACTIONS:
                encoding_rule.fit(dataset.interactions)
            elif feature_info.feature_source == FeatureSource.QUERY_FEATURES:
                encoding_rule.fit(dataset.query_features)
            elif feature_info.feature_source == FeatureSource.ITEM_FEATURES:
                encoding_rule.fit(dataset.item_features)
            else:
                assert False, "Unknown feature source"  # pragma: no cover

            self._encoding_rules[column] = encoding_rule

        return self

    def transform(
        self,
        dataset: Dataset,
    ) -> Dataset:
        """Transforms the input Dataset categorical features by rules.

        Args:
            dataset (Dataset): The Dataset object.

        Returns:
            Dataset: Transformed dataset.
        """
        self._check_if_initialized()

        interactions = dataset.interactions
        query_features = dataset.query_features
        item_features = dataset.item_features

        for column, feature_info in dataset.feature_schema.categorical_features.items():
            if column not in self._encoding_rules:
                warnings.warn(
                    f"Cannot transform feature '{column}' " "as it was not present at the fit stage",
                    LabelEncoderTransformWarning,
                )
                continue

            encoding_rule = self._encoding_rules[column]

            if feature_info.feature_hint == FeatureHint.QUERY_ID:
                interactions = encoding_rule.transform(interactions)
                if query_features is not None:
                    query_features = encoding_rule.transform(query_features)
            elif feature_info.feature_hint == FeatureHint.ITEM_ID:
                interactions = encoding_rule.transform(interactions)
                if item_features is not None:
                    item_features = encoding_rule.transform(item_features)
            elif feature_info.feature_source == FeatureSource.INTERACTIONS:
                interactions = encoding_rule.transform(interactions)
            elif feature_info.feature_source == FeatureSource.QUERY_FEATURES:
                query_features = encoding_rule.transform(query_features)
            else:
                item_features = encoding_rule.transform(item_features)

        dataset_copy = Dataset(
            feature_schema=dataset.feature_schema,
            interactions=interactions,
            query_features=query_features,
            item_features=item_features,
            check_consistency=False,
            categorical_encoded=True,
        )

        return dataset_copy

    def fit_transform(self, dataset: Dataset) -> Dataset:
        """Fits an encoder and transforms the input Dataset categorical features.

        Args:
            dataset (Dataset): The Dataset object.

        Returns:
            Dataset: Transformed dataset.
        """
        return self.fit(dataset).transform(dataset)

    def get_encoder(self, columns: Union[str, Iterable[str]]) -> Optional[LabelEncoder]:
        """Get the encoder of fitted Dataset for columns.

        Args:
            columns (Union[str, Iterable[str]]): Columns to filter by.

        Returns:
            Optional[LabelEncoder]: Label encoder.
        """
        self._check_if_initialized()

        columns_set: Set[str]
        if isinstance(columns, str):
            columns_set = set([columns])
        else:
            columns_set = set(columns)

        def get_encoding_rules() -> Iterator[LabelEncodingRule]:
            for column, rule in self._encoding_rules.items():
                if column in columns_set:
                    yield rule

        rules = list(get_encoding_rules())
        if len(rules) == 0:
            return None

        return LabelEncoder(rules)

    @property
    def query_id_encoder(self) -> LabelEncoder:
        """
        Returns:
            LabelEncoder: Query id label encoder.
        """
        query_id_column = self._features_columns[FeatureHint.QUERY_ID]
        encoder = self.get_encoder(query_id_column)
        assert encoder is not None
        return encoder

    @property
    def item_id_encoder(self) -> LabelEncoder:
        """
        Returns:
            LabelEncoder: Item id label encoder.
        """
        item_id_column = self._features_columns[FeatureHint.ITEM_ID]
        encoder = self.get_encoder(item_id_column)
        assert encoder is not None
        return encoder

    @property
    def query_and_item_id_encoder(self) -> LabelEncoder:
        """
        Returns:
            LabelEncoder: Query id and Item id label encoder.
        """
        query_id_column = self._features_columns[FeatureHint.QUERY_ID]
        item_id_column = self._features_columns[FeatureHint.ITEM_ID]
        encoder = self.get_encoder(query_id_column + item_id_column)  # type: ignore
        assert encoder is not None
        return encoder

    @property
    def interactions_encoder(self) -> Optional[LabelEncoder]:
        """
        Returns:
            LabelEncoder: Interactions label encoder.
        """

        interactions_columns = self._features_columns[FeatureSource.INTERACTIONS]
        return self.get_encoder(interactions_columns)

    @property
    def query_features_encoder(self) -> Optional[LabelEncoder]:
        """
        Returns:
            LabelEncoder: Query features label encoder.
        """
        query_features_columns = self._features_columns[FeatureSource.QUERY_FEATURES]
        return self.get_encoder(query_features_columns)

    @property
    def item_features_encoder(self) -> Optional[LabelEncoder]:
        """
        Returns:
            LabelEncoder: Item features label encoder.
        """
        item_features_columns = self._features_columns[FeatureSource.ITEM_FEATURES]
        return self.get_encoder(item_features_columns)

    def _check_if_initialized(self) -> None:
        if not self._encoding_rules:
            raise ValueError("Encoder is not initialized")

    def _fill_features_columns(self, feature_info: FeatureSchema) -> None:
        self._features_columns = {
            FeatureHint.QUERY_ID: [feature_info.query_id_column],
            FeatureHint.ITEM_ID: [feature_info.item_id_column],
            FeatureSource.INTERACTIONS: feature_info.interaction_features.columns,
            FeatureSource.QUERY_FEATURES: feature_info.query_features.columns,
            FeatureSource.ITEM_FEATURES: feature_info.item_features.columns,
        }
