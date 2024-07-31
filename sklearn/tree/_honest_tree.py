# Adopted from: https://github.com/neurodata/honest-forests

import copy
import numbers
import numpy as np
from math import ceil
from numpy import float32 as DTYPE
from scipy.sparse import issparse

from ..base import ClassifierMixin, MetaEstimatorMixin, _fit_context, clone, is_classifier
from ..model_selection import StratifiedShuffleSplit
from ..utils import check_random_state, compute_sample_weight
from ..utils._param_validation import HasMethods, Interval, RealNotInt, StrOptions
from ..utils.multiclass import _check_partial_fit_first_call, check_classification_targets
from ..utils.validation import check_is_fitted, check_X_y

from ._classes import (
    BaseDecisionTree, DecisionTreeClassifier,
    CRITERIA_CLF, CRITERIA_REG, DENSE_SPLITTERS, SPARSE_SPLITTERS
)
from ._criterion import BaseCriterion
from ._honesty import Honesty
from ._tree import DOUBLE


class BuildTreeArgs:
    def __init__(
        self,
        X,
        y,
        sample_weight,
        missing_values_in_feature_mask,
        min_samples_leaf,
        min_weight_leaf,
        max_leaf_nodes,
        min_samples_split,
        max_depth,
        random_state
    ):
        for name, value in locals().items():
            if name != 'self':
                setattr(self, name, value)


class HonestTree(BaseDecisionTree):
    _parameter_constraints: dict = {
        **BaseDecisionTree._parameter_constraints,
        "honest_fraction": [Interval(RealNotInt, 0.0, 1.0, closed="neither")],
        "honest_prior": [StrOptions({"empirical", "uniform", "ignore"})],
        "stratify": ["boolean"],
    }

    def __init__(
        self,
        target_tree,
        random_state=None,
        honest_fraction=0.5,
        honest_prior="empirical",
        stratify=False
    ):
        self.target_tree = target_tree
        self.random_state = random_state
        self.honest_fraction = honest_fraction
        self.honest_prior = honest_prior
        self.stratify = stratify

    # def _data_prep(
    #         self,
    #         target_tree,
    #         X,
    #         y,
    #         sample_weight=None,
    #         check_input=True,
    #         missing_values_in_feature_mask=None,
    #         classes=None
    # ):
    #     random_state = check_random_state(target_tree.random_state)

    #     if check_input:
    #         # Need to validate separately here.
    #         # We can't pass multi_output=True because that would allow y to be
    #         # csr.

    #         # _compute_missing_values_in_feature_mask will check for finite values and
    #         # compute the missing mask if the tree supports missing values
    #         check_X_params = dict(
    #             dtype=DTYPE, accept_sparse="csc", force_all_finite=False
    #         )
    #         check_y_params = dict(ensure_2d=False, dtype=None)
    #         if y is not None or target_tree._get_tags()["requires_y"]:
    #             X, y = target_tree._validate_data(
    #                 X, y, validate_separately=(check_X_params, check_y_params)
    #             )
    #         else:
    #             X = target_tree._validate_data(X, **check_X_params)

    #         missing_values_in_feature_mask = (
    #             target_tree._compute_missing_values_in_feature_mask(X)
    #         )
    #         if issparse(X):
    #             X.sort_indices()

    #             if X.indices.dtype != np.intc or X.indptr.dtype != np.intc:
    #                 raise ValueError(
    #                     "No support for np.int64 index based sparse matrices"
    #                 )

    #         if y is not None and target_tree.criterion == "poisson":
    #             if np.any(y < 0):
    #                 raise ValueError(
    #                     "Some value(s) of y are negative which is"
    #                     " not allowed for Poisson regression."
    #                 )
    #             if np.sum(y) <= 0:
    #                 raise ValueError(
    #                     "Sum of y is not positive which is "
    #                     "necessary for Poisson regression."
    #                 )

    #     # Determine output settings
    #     n_samples, self.n_features_in_ = X.shape

    #     # Do preprocessing if 'y' is passed
    #     is_classification = False
    #     if y is not None:
    #         is_classification = is_classifier(target_tree)
    #         y = np.atleast_1d(y)
    #         expanded_class_weight = None

    #         if y.ndim == 1:
    #             # reshape is necessary to preserve the data contiguity against vs
    #             # [:, np.newaxis] that does not.
    #             y = np.reshape(y, (-1, 1))

    #         self.n_outputs_ = y.shape[1]

    #         if is_classification:
    #             check_classification_targets(y)
    #             y = np.copy(y)

    #             self.classes_ = []
    #             self.n_classes_ = []

    #             if target_tree.class_weight is not None:
    #                 y_original = np.copy(y)

    #             y_encoded = np.zeros(y.shape, dtype=int)
    #             if classes is not None:
    #                 classes = np.atleast_1d(classes)
    #                 if classes.ndim == 1:
    #                     classes = np.array([classes])

    #                 for k in classes:
    #                     self.classes_.append(np.array(k))
    #                     self.n_classes_.append(np.array(k).shape[0])

    #                 for i in range(n_samples):
    #                     for j in range(self.n_outputs_):
    #                         y_encoded[i, j] = np.where(
    #                             self.classes_[j] == y[i, j]
    #                         )[0][0]
    #             else:
    #                 for k in range(self.n_outputs_):
    #                     classes_k, y_encoded[:, k] = np.unique(
    #                         y[:, k], return_inverse=True
    #                     )
    #                     self.classes_.append(classes_k)
    #                     self.n_classes_.append(classes_k.shape[0])

    #             y = y_encoded

    #             if target_tree.class_weight is not None:
    #                 expanded_class_weight = compute_sample_weight(
    #                     target_tree.class_weight, y_original
    #                 )

    #             self.n_classes_ = np.array(self.n_classes_, dtype=np.intp)
    #             self._n_classes_ = self.n_classes_
    #         if getattr(y, "dtype", None) != DOUBLE or not y.flags.contiguous:
    #             y = np.ascontiguousarray(y, dtype=DOUBLE)

    #         if len(y) != n_samples:
    #             raise ValueError(
    #                 "Number of labels=%d does not match number of samples=%d"
    #                 % (len(y), n_samples)
    #             )

    #     # set decision-tree model parameters
    #     max_depth = np.iinfo(np.int32).max if target_tree.max_depth is None else target_tree.max_depth

    #     if isinstance(target_tree.min_samples_leaf, numbers.Integral):
    #         min_samples_leaf = target_tree.min_samples_leaf
    #     else:  # float
    #         min_samples_leaf = int(ceil(target_tree.min_samples_leaf * n_samples))

    #     if isinstance(target_tree.min_samples_split, str):
    #         if target_tree.min_samples_split == "sqrt":
    #             min_samples_split = max(1, int(np.sqrt(target_tree.n_features_in_)))
    #         elif target_tree.min_samples_split == "log2":
    #             min_samples_split = max(1, int(np.log2(target_tree.n_features_in_)))
    #     elif isinstance(target_tree.min_samples_split, numbers.Integral):
    #         min_samples_split = target_tree.min_samples_split
    #     else:  # float
    #         min_samples_split = int(ceil(target_tree.min_samples_split * n_samples))
    #         min_samples_split = max(2, min_samples_split)
    #     min_samples_split = max(min_samples_split, 2 * min_samples_leaf)
    #     self.min_samples_split_ = min_samples_split
    #     self.min_samples_leaf_ = min_samples_leaf

    #     if isinstance(target_tree.max_features, str):
    #         if target_tree.max_features == "sqrt":
    #             max_features = max(1, int(np.sqrt(target_tree.n_features_in_)))
    #         elif target_tree.max_features == "log2":
    #             max_features = max(1, int(np.log2(target_tree.n_features_in_)))
    #     elif target_tree.max_features is None:
    #         max_features = target_tree.n_features_in_
    #     elif isinstance(target_tree.max_features, numbers.Integral):
    #         max_features = target_tree.max_features
    #     else:  # float
    #         if target_tree.max_features > 0.0:
    #             max_features = max(1, int(target_tree.max_features * target_tree.n_features_in_))
    #         else:
    #             max_features = 0

    #     self.max_features_ = max_features

    #     max_leaf_nodes = -1 if target_tree.max_leaf_nodes is None else target_tree.max_leaf_nodes

    #     return BuildTreeArgs(
    #         X=X,
    #         y=y,
    #         sample_weight=sample_weight,
    #         missing_values_in_feature_mask=missing_values_in_feature_mask,
    #         min_samples_leaf=min_samples_leaf,
    #         min_weight_leaf=self.min_weight_fraction_leaf,
    #         max_leaf_nodes=max_leaf_nodes,
    #         min_samples_split=min_samples_split,
    #         max_depth=max_depth,
    #         random_state=random_state
    #     )


    @_fit_context(prefer_skip_nested_validation=True)
    def fit(
        self,
        X,
        y,
        sample_weight=None,
        check_input=True,
        missing_values_in_feature_mask=None,
        classes=None,
    ):
        """Build an honest tree from the training set (X, y).

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csc_matrix``.

        y : array-like of shape (n_samples,) or (n_samples, n_outputs)

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. If None, then samples are equally weighted. Splits
            that would create child nodes with net zero or negative weight are
            ignored while searching for a split in each node. Splits are also
            ignored if they would result in any single class carrying a
            negative weight in either child node.

        check_input : bool, default=True
            Allow to bypass several input checking.
            Don't use this parameter unless you know what you do.

        classes : array-like of shape (n_classes,), default=None
            List of all the classes that can possibly appear in the y vector.

        Returns
        -------
        self : HonestTree
            Fitted tree estimator.
        """

        bta = self.target_tree._prep_data(
            X=X,
            y=y,
            sample_weight=sample_weight,
            check_input=check_input,
            missing_values_in_feature_mask=missing_values_in_feature_mask,
            classes=classes
        )

        # Determine output settings
        self._init_output_shape(bta.X, bta.y, bta.classes)

        # obtain the structure sample weights
        sample_weights_structure = self._partition_honest_indices(
            bta.y,
            bta.sample_weight
        )

        # compute the honest sample indices
        not_honest_mask = np.ones(len(bta.y), dtype=bool)
        not_honest_mask[self.honest_indices_] = False

        if bta.sample_weight is None:
            sample_weight_leaves = np.ones((len(bta.y),), dtype=np.float64)
        else:
            sample_weight_leaves = np.array(bta.sample_weight)
        sample_weight_leaves[not_honest_mask] = 0

        # determine the honest indices using the sample weight
        nonzero_indices = np.where(sample_weight_leaves > 0)[0]
        # sample the structure indices
        self.honest_indices_ = nonzero_indices

        # create honesty, set up listeners in target tree
        self.honesty = Honesty(
            bta.X,
            self.honest_indices_,
            bta.min_samples_leaf
        )

        self.target_tree.presplit_conditions = self.honesty.presplit_conditions
        self.target_tree.postsplit_conditions = self.honesty.postsplit_conditions
        self.target_tree.splitter_listeners = self.honesty.splitter_event_handlers
        # self.target_tree.tree_build_listeners = self.honesty.tree_build_event_handlers

        # Learn structure on subsample
        # XXX: this allows us to use BaseDecisionTree without partial_fit API
        try:
            self.target_tree.fit(
                bta.X,
                bta.y,
                sample_weight=sample_weights_structure,
                check_input=check_input,
                #missing_values_in_feature_mask=missing_values_in_feature_mask,
                classes=bta.classes,
            )
        except Exception:
            self.target_tree.fit(
                bta.X,
                bta.y,
                sample_weight=sample_weights_structure,
                check_input=check_input,
                #missing_values_in_feature_mask=missing_values_in_feature_mask,
            )
        # self._inherit_estimator_attributes()


        # self._fit_leaves(X, y, sample_weight=sample_weight_leaves)
        return self.target_tree

    
    def _check_input(self, X, y):
        # Need to validate separately here.
        # We can't pass multi_output=True because that would allow y to be
        # csr.

        # _compute_missing_values_in_feature_mask will check for finite values and
        # compute the missing mask if the tree supports missing values
        check_X_params = dict(
            dtype=DTYPE, accept_sparse="csc", force_all_finite=False
        )
        check_y_params = dict(ensure_2d=False, dtype=None)
        if y is not None or self._get_tags()["requires_y"]:
            X, y = self._validate_data(
                X, y, validate_separately=(check_X_params, check_y_params)
            )
        else:
            X = self._validate_data(X, **check_X_params)

        if issparse(X):
            X.sort_indices()

            if X.indices.dtype != np.intc or X.indptr.dtype != np.intc:
                raise ValueError(
                    "No support for np.int64 index based sparse matrices"
                )

        if y is not None and self.criterion == "poisson":
            if np.any(y < 0):
                raise ValueError(
                    "Some value(s) of y are negative which is"
                    " not allowed for Poisson regression."
                )
            if np.sum(y) <= 0:
                raise ValueError(
                    "Sum of y is not positive which is "
                    "necessary for Poisson regression."
                )


    def _init_output_shape(self, X, y, classes=None):
        # Determine output settings
        self.n_samples_, self.n_features_in_ = X.shape

        # Do preprocessing if 'y' is passed
        is_classification = False
        if y is not None:
            is_classification = is_classifier(self)
            y = np.atleast_1d(y)
            expanded_class_weight = None

            if y.ndim == 1:
                # reshape is necessary to preserve the data contiguity against vs
                # [:, np.newaxis] that does not.
                y = np.reshape(y, (-1, 1))

            self.n_outputs_ = y.shape[1]

            if is_classification:
                check_classification_targets(y)
                y = np.copy(y)

                self.classes_ = []
                self.n_classes_ = []

                if self.class_weight is not None:
                    y_original = np.copy(y)

                y_encoded = np.zeros(y.shape, dtype=int)
                if classes is not None:
                    classes = np.atleast_1d(classes)
                    if classes.ndim == 1:
                        classes = np.array([classes])

                    for k in classes:
                        self.classes_.append(np.array(k))
                        self.n_classes_.append(np.array(k).shape[0])

                    for i in range(self.n_samples_):
                        for j in range(self.n_outputs_):
                            y_encoded[i, j] = np.where(self.classes_[j] == y[i, j])[0][
                                0
                            ]
                else:
                    for k in range(self.n_outputs_):
                        classes_k, y_encoded[:, k] = np.unique(
                            y[:, k], return_inverse=True
                        )
                        self.classes_.append(classes_k)
                        self.n_classes_.append(classes_k.shape[0])

                y = y_encoded

                if self.class_weight is not None:
                    expanded_class_weight = compute_sample_weight(
                        self.class_weight, y_original
                    )

                self.n_classes_ = np.array(self.n_classes_, dtype=np.intp)
                self._n_classes_ = self.n_classes_
            if getattr(y, "dtype", None) != DOUBLE or not y.flags.contiguous:
                y = np.ascontiguousarray(y, dtype=DOUBLE)

            if len(y) != self.n_samples_:
                raise ValueError(
                    "Number of labels=%d does not match number of samples=%d"
                    % (len(y), self.n_samples_)
                )


    def _partition_honest_indices(self, y, sample_weight):
        rng = np.random.default_rng(self.target_tree.random_state)

        # Account for bootstrapping too
        if sample_weight is None:
            _sample_weight = np.ones((len(y),), dtype=np.float64)
        else:
            _sample_weight = np.array(sample_weight)

        nonzero_indices = np.where(_sample_weight > 0)[0]
        # sample the structure indices
        if self.stratify:
            ss = StratifiedShuffleSplit(
                n_splits=1, test_size=self.honest_fraction, random_state=self.random_state
            )
            for structure_idx, _ in ss.split(
                np.zeros((len(nonzero_indices), 1)), y[nonzero_indices]
            ):
                self.structure_indices_ = nonzero_indices[structure_idx]
        else:
            self.structure_indices_ = rng.choice(
                nonzero_indices,
                int((1 - self.honest_fraction) * len(nonzero_indices)),
                replace=False,
            )

        self.honest_indices_ = np.setdiff1d(nonzero_indices, self.structure_indices_)
        _sample_weight[self.honest_indices_] = 0

        return _sample_weight


# class HonestTreeClassifier(MetaEstimatorMixin, ClassifierMixin):
#     """
#     A decision tree classifier with honest predictions.

#     Parameters
#     ----------
#     tree_estimator : object, default=None
#         Instantiated tree of type BaseDecisionTree from treeple.
#         If None, then sklearn's DecisionTreeClassifier with default parameters will
#         be used. Note that none of the parameters in ``tree_estimator`` need
#         to be set. The parameters of the ``tree_estimator`` can be set using
#         the ``tree_estimator_params`` keyword argument.

#     criterion : {"gini", "entropy"}, default="gini"
#         The function to measure the quality of a split. Supported criteria are
#         "gini" for the Gini impurity and "entropy" for the information gain.

#     splitter : {"best", "random"}, default="best"
#         The strategy used to choose the split at each node. Supported
#         strategies are "best" to choose the best split and "random" to choose
#         the best random split.

#     max_depth : int, default=None
#         The maximum depth of the tree. If None, then nodes are expanded until
#         all leaves are pure or until all leaves contain less than
#         min_samples_split samples.

#     min_samples_split : int or float, default=2
#         The minimum number of samples required to split an internal node:

#         - If int, then consider `min_samples_split` as the minimum number.
#         - If float, then `min_samples_split` is a fraction and
#           `ceil(min_samples_split * n_samples)` are the minimum
#           number of samples for each split.

#     min_samples_leaf : int or float, default=1
#         The minimum number of samples required to be at a leaf node.
#         A split point at any depth will only be considered if it leaves at
#         least ``min_samples_leaf`` training samples in each of the left and
#         right branches.  This may have the effect of smoothing the model,
#         especially in regression.

#         - If int, then consider `min_samples_leaf` as the minimum number.
#         - If float, then `min_samples_leaf` is a fraction and
#           `ceil(min_samples_leaf * n_samples)` are the minimum
#           number of samples for each node.

#     min_weight_fraction_leaf : float, default=0.0
#         The minimum weighted fraction of the sum total of weights (of all
#         the input samples) required to be at a leaf node. Samples have
#         equal weight when sample_weight is not provided.

#     max_features : int, float or {"auto", "sqrt", "log2"}, default=None
#         The number of features to consider when looking for the best split:

#             - If int, then consider `max_features` features at each split.
#             - If float, then `max_features` is a fraction and
#               `int(max_features * n_features)` features are considered at each
#               split.
#             - If "auto", then `max_features=sqrt(n_features)`.
#             - If "sqrt", then `max_features=sqrt(n_features)`.
#             - If "log2", then `max_features=log2(n_features)`.
#             - If None, then `max_features=n_features`.

#         Note: the search for a split does not stop until at least one
#         valid partition of the node samples is found, even if it requires to
#         effectively inspect more than ``max_features`` features.

#     random_state : int, RandomState instance or None, default=None
#         Controls the randomness of the tree estimator. The features are always
#         randomly permuted at each split, even if ``splitter`` is set to
#         ``"best"``. When ``max_features < n_features``, the algorithm will
#         select ``max_features`` at random at each split before finding the best
#         split among them. But the best found split may vary across different
#         runs, even if ``max_features=n_features``. That is the case, if the
#         improvement of the criterion is identical for several splits and one
#         split has to be selected at random. To obtain a deterministic behaviour
#         during fitting, ``random_state`` has to be fixed to an integer.
#         See :term:`Glossary <random_state>` for details.

#     max_leaf_nodes : int, default=None
#         Grow a tree with ``max_leaf_nodes`` in best-first fashion.
#         Best nodes are defined as relative reduction in impurity.
#         If None then unlimited number of leaf nodes.

#     min_impurity_decrease : float, default=0.0
#         A node will be split if this split induces a decrease of the impurity
#         greater than or equal to this value.

#         The weighted impurity decrease equation is the following::

#             N_t / N * (impurity - N_t_R / N_t * right_impurity
#                                 - N_t_L / N_t * left_impurity)

#         where ``N`` is the total number of samples, ``N_t`` is the number of
#         samples at the current node, ``N_t_L`` is the number of samples in the
#         left child, and ``N_t_R`` is the number of samples in the right child.

#         ``N``, ``N_t``, ``N_t_R`` and ``N_t_L`` all refer to the weighted sum,
#         if ``sample_weight`` is passed.

#     class_weight : dict, list of dict or "balanced", default=None
#         Weights associated with classes in the form ``{class_label: weight}``.
#         If None, all classes are supposed to have weight one. For
#         multi-output problems, a list of dicts can be provided in the same
#         order as the columns of y.

#         Note that for multioutput (including multilabel) weights should be
#         defined for each class of every column in its own dict. For example,
#         for four-class multilabel classification weights should be
#         [{0: 1, 1: 1}, {0: 1, 1: 5}, {0: 1, 1: 1}, {0: 1, 1: 1}] instead of
#         [{1:1}, {2:5}, {3:1}, {4:1}].

#         The "balanced" mode uses the values of y to automatically adjust
#         weights inversely proportional to class frequencies in the input data
#         as ``n_samples / (n_classes * np.bincount(y))``

#         For multi-output, the weights of each column of y will be multiplied.

#         Note that these weights will be multiplied with sample_weight (passed
#         through the fit method) if sample_weight is specified.

#     ccp_alpha : non-negative float, default=0.0
#         Complexity parameter used for Minimal Cost-Complexity Pruning. The
#         subtree with the largest cost complexity that is smaller than
#         ``ccp_alpha`` will be chosen. By default, no pruning is performed. See
#         :ref:`minimal_cost_complexity_pruning` for details.

#     monotonic_cst : array-like of int of shape (n_features), default=None
#         Indicates the monotonicity constraint to enforce on each feature.
#           - 1: monotonic increase
#           - 0: no constraint
#           - -1: monotonic decrease

#         If monotonic_cst is None, no constraints are applied.

#         Monotonicity constraints are not supported for:
#           - multiclass classifications (i.e. when `n_classes > 2`),
#           - multioutput classifications (i.e. when `n_outputs_ > 1`),
#           - classifications trained on data with missing values.

#         The constraints hold over the probability of the positive class.

#         Read more in the :ref:`User Guide <monotonic_cst_gbdt>`.

#     honest_fraction : float, default=0.5
#         Fraction of training samples used for estimates in the leaves. The
#         remaining samples will be used to learn the tree structure. A larger
#         fraction creates shallower trees with lower variance estimates.

#     honest_prior : {"ignore", "uniform", "empirical"}, default="empirical"
#         Method for dealing with empty leaves during evaluation of a test
#         sample. If "ignore", returns numpy.nan.
#         If "uniform", the prior tree posterior is 1/(number of
#         classes). If "empirical", the prior tree posterior is the relative
#         class frequency in the voting subsample.

#     stratify : bool
#         Whether or not to stratify sample when considering structure and leaf indices.
#         By default False.

#     **tree_estimator_params : dict
#         Parameters to pass to the underlying base tree estimators.
#         These must be parameters for ``tree_estimator``.

#     Attributes
#     ----------
#     estimator_ : object
#         The child tree estimator template used to create the collection
#         of fitted sub-estimators.

#     classes_ : ndarray of shape (n_classes,) or list of ndarray
#         The classes labels (single output problem),
#         or a list of arrays of class labels (multi-output problem).

#     feature_importances_ : ndarray of shape (n_features,)
#         The impurity-based feature importances.
#         The higher, the more important the feature.
#         The importance of a feature is computed as the (normalized)
#         total reduction of the criterion brought by that feature.  It is also
#         known as the Gini importance [4]_.

#         Warning: impurity-based feature importances can be misleading for
#         high cardinality features (many unique values). See
#         :func:`sklearn.inspection.permutation_importance` as an alternative.

#     max_features_ : int
#         The inferred value of max_features.

#     n_classes_ : int or list of int
#         The number of classes (for single output problems),
#         or a list containing the number of classes for each
#         output (for multi-output problems).

#     n_features_in_ : int
#         Number of features seen during :term:`fit`.

#     feature_names_in_ : ndarray of shape (`n_features_in_`,)
#         Names of features seen during :term:`fit`. Defined only when `X`
#         has feature names that are all strings.

#     n_outputs_ : int
#         The number of outputs when ``fit`` is performed.

#     tree_ : Tree instance
#         The underlying Tree object. Please refer to
#         ``help(sklearn.tree._tree.Tree)`` for attributes of Tree object and
#         :ref:`sphx_glr_auto_examples_tree_plot_unveil_tree_structure.py`
#         for basic usage of these attributes.

#     empirical_prior_ : float
#         Proportion of each class in the training labels y

#     structure_indices_ : numpy.ndarray, shape=(n_structure,)
#         Indices of training samples used to learn the structure

#     honest_indices_ : numpy.ndarray, shape=(n_honest,)
#         Indices of training samples used to learn leaf estimates

#     Notes
#     -----
#     The default values for the parameters controlling the size of the trees
#     (e.g. ``max_depth``, ``min_samples_leaf``, etc.) lead to fully grown and
#     unpruned trees which can potentially be very large on some data sets. To
#     reduce memory consumption, the complexity and size of the trees should be
#     controlled by setting those parameter values.

#     The :meth:`predict` method operates using the :func:`numpy.argmax`
#     function on the outputs of :meth:`predict_proba`. This means that in
#     case the highest predicted probabilities are tied, the classifier will
#     predict the tied class with the lowest index in :term:`classes_`.

#     References
#     ----------

#     .. [1] https://en.wikipedia.org/wiki/Decision_tree_learning

#     .. [2] L. Breiman, J. Friedman, R. Olshen, and C. Stone, "Classification
#             and Regression Trees", Wadsworth, Belmont, CA, 1984.

#     .. [3] T. Hastie, R. Tibshirani and J. Friedman. "Elements of Statistical
#             Learning", Springer, 2009.

#     .. [4] L. Breiman, and A. Cutler, "Random Forests",
#             https://www.stat.berkeley.edu/~breiman/RandomForests/cc_home.htm

#     .. [5] S. Athey, J. Tibshirani, and S. Wager. "Generalized
#             Random Forests", Annals of Statistics, 2019.

#     Examples
#     --------
#     >>> from sklearn.datasets import load_iris
#     >>> from sklearn.model_selection import cross_val_score
#     >>> from honest_forests import HonestTreeClassifier
#     >>> clf = HonestTreeClassifier(random_state=0)
#     >>> iris = load_iris()
#     >>> cross_val_score(clf, iris.data, iris.target, cv=10)
#     ...                             # doctest: +SKIP
#     ...
#     array([0.93333333, 0.93333333, 1.        , 1.        , 0.93333333,
#            0.8       , 0.8       , 0.93333333, 1.        , 1.        ])
#     """

#     def __init__(
#         self,
#         tree_estimator=None,
#         criterion="gini",
#         splitter="best",
#         max_depth=None,
#         min_samples_split=2,
#         min_samples_leaf=1,
#         min_weight_fraction_leaf=0.0,
#         max_features=None,
#         random_state=None,
#         max_leaf_nodes=None,
#         min_impurity_decrease=0.0,
#         class_weight=None,
#         ccp_alpha=0.0,
#         monotonic_cst=None,
#         honest_fraction=0.5,
#         honest_prior="empirical",
#         stratify=False,
#         **tree_estimator_params,
#     ):
#         self.tree_estimator = tree_estimator
#         self.criterion = criterion
#         self.splitter = splitter
#         self.max_depth = max_depth
#         self.min_samples_split = min_samples_split
#         self.min_samples_leaf = min_samples_leaf
#         self.min_weight_fraction_leaf = min_weight_fraction_leaf
#         self.max_features = max_features
#         self.max_leaf_nodes = max_leaf_nodes
#         self.class_weight = class_weight
#         self.random_state = random_state
#         self.min_impurity_decrease = min_impurity_decrease
#         self.ccp_alpha = ccp_alpha
#         self.monotonic_cst = monotonic_cst

#         self.honest_fraction = honest_fraction
#         self.honest_prior = honest_prior
#         self.stratify = stratify

#         # XXX: to enable this, we need to also reset the leaf node samples during `_set_leaf_nodes`
#         self.store_leaf_values = False
#         self._tree_estimator_params = tree_estimator_params

#     @_fit_context(prefer_skip_nested_validation=True)
#     def fit(
#         self,
#         X,
#         y,
#         sample_weight=None,
#         check_input=True,
#         classes=None,
#     ):
#         """Build a decision tree classifier from the training set (X, y).

#         Parameters
#         ----------
#         X : {array-like, sparse matrix} of shape (n_samples, n_features)
#             The training input samples. Internally, it will be converted to
#             ``dtype=np.float32`` and if a sparse matrix is provided
#             to a sparse ``csc_matrix``.

#         y : array-like of shape (n_samples,) or (n_samples, n_outputs)
#             The target values (class labels) as integers or strings.

#         sample_weight : array-like of shape (n_samples,), default=None
#             Sample weights. If None, then samples are equally weighted. Splits
#             that would create child nodes with net zero or negative weight are
#             ignored while searching for a split in each node. Splits are also
#             ignored if they would result in any single class carrying a
#             negative weight in either child node.

#         check_input : bool, default=True
#             Allow to bypass several input checking.
#             Don't use this parameter unless you know what you're doing.

#         classes : array-like of shape (n_classes,), default=None
#             List of all the classes that can possibly appear in the y vector.
#             Must be provided at the first call to partial_fit, can be omitted
#             in subsequent calls.

#         Returns
#         -------
#         self : HonestTreeClassifier
#             Fitted estimator.
#         """
#         self._fit(
#             X,
#             y,
#             sample_weight=sample_weight,
#             check_input=check_input,
#             classes=classes,
#         )
#         return self

#     def partial_fit(self, X, y, sample_weight=None, check_input=True, classes=None):
#         """Update a decision tree classifier from the training set (X, y).

#         Parameters
#         ----------
#         X : {array-like, sparse matrix} of shape (n_samples, n_features)
#             The training input samples. Internally, it will be converted to
#             ``dtype=np.float32`` and if a sparse matrix is provided
#             to a sparse ``csc_matrix``.

#         y : array-like of shape (n_samples,) or (n_samples, n_outputs)
#             The target values (class labels) as integers or strings.

#         sample_weight : array-like of shape (n_samples,), default=None
#             Sample weights. If None, then samples are equally weighted. Splits
#             that would create child nodes with net zero or negative weight are
#             ignored while searching for a split in each node. Splits are also
#             ignored if they would result in any single class carrying a
#             negative weight in either child node.

#         check_input : bool, default=True
#             Allow to bypass several input checking.
#             Don't use this parameter unless you know what you do.

#         classes : array-like of shape (n_classes,), default=None
#             List of all the classes that can possibly appear in the y vector.
#             Must be provided at the first call to partial_fit, can be omitted
#             in subsequent calls.

#         Returns
#         -------
#         self : HonestTreeClassifier
#             Fitted estimator.
#         """
#         self._validate_params()

#         # validate input parameters
#         first_call = _check_partial_fit_first_call(self, classes=classes)

#         # Fit if no tree exists yet
#         if first_call:
#             self._fit(
#                 X,
#                 y,
#                 sample_weight=sample_weight,
#                 check_input=check_input,
#                 classes=classes,
#             )
#             return self

#         rng = np.random.default_rng(self.random_state)

#         if sample_weight is None:
#             _sample_weight = np.ones((X.shape[0],), dtype=np.float64)
#         else:
#             _sample_weight = np.array(sample_weight)

#         nonzero_indices = np.where(_sample_weight > 0)[0]

#         self.structure_indices_ = rng.choice(
#             nonzero_indices,
#             int((1 - self.honest_fraction) * len(nonzero_indices)),
#             replace=False,
#         )
#         self.honest_indices_ = np.setdiff1d(nonzero_indices, self.structure_indices_)
#         _sample_weight[self.honest_indices_] = 0

#         self.estimator_.partial_fit(
#             X,
#             y,
#             sample_weight=_sample_weight,
#             check_input=check_input,
#             classes=classes,
#         )
#         self._inherit_estimator_attributes()

#         # set leaf nodes
#         self._fit_leaves(X, y, sample_weight=_sample_weight)

#         return self

#     def _partition_honest_indices(self, y, sample_weight):
#         rng = np.random.default_rng(self.random_state)

#         # Account for bootstrapping too
#         if sample_weight is None:
#             _sample_weight = np.ones((len(y),), dtype=np.float64)
#         else:
#             _sample_weight = np.array(sample_weight)

#         nonzero_indices = np.where(_sample_weight > 0)[0]
#         # sample the structure indices
#         if self.stratify:
#             ss = StratifiedShuffleSplit(
#                 n_splits=1, test_size=self.honest_fraction, random_state=self.random_state
#             )
#             for structure_idx, _ in ss.split(
#                 np.zeros((len(nonzero_indices), 1)), y[nonzero_indices]
#             ):
#                 self.structure_indices_ = nonzero_indices[structure_idx]
#         else:
#             self.structure_indices_ = rng.choice(
#                 nonzero_indices,
#                 int((1 - self.honest_fraction) * len(nonzero_indices)),
#                 replace=False,
#             )

#         self.honest_indices_ = np.setdiff1d(nonzero_indices, self.structure_indices_)
#         _sample_weight[self.honest_indices_] = 0

#         return _sample_weight

#     def _get_estimator(self):
#         """Resolve which estimator to return (default is DecisionTreeClassifier)"""
#         if self.tree_estimator is None:
#             self.estimator_ = DecisionTreeClassifier(random_state=self.random_state)
#         else:
#             # XXX: maybe error out if the base tree estimator is already fitted
#             self.estimator_ = clone(self.tree_estimator)
#         return self.estimator_

#     def _fit(
#         self,
#         X,
#         y,
#         sample_weight=None,
#         check_input=True,
#         missing_values_in_feature_mask=None,
#         classes=None,
#     ):
#         """Build an honest tree classifier from the training set (X, y).

#         Parameters
#         ----------
#         X : {array-like, sparse matrix} of shape (n_samples, n_features)
#             The training input samples. Internally, it will be converted to
#             ``dtype=np.float32`` and if a sparse matrix is provided
#             to a sparse ``csc_matrix``.

#         y : array-like of shape (n_samples,) or (n_samples, n_outputs)
#             The target values (class labels) as integers or strings.

#         sample_weight : array-like of shape (n_samples,), default=None
#             Sample weights. If None, then samples are equally weighted. Splits
#             that would create child nodes with net zero or negative weight are
#             ignored while searching for a split in each node. Splits are also
#             ignored if they would result in any single class carrying a
#             negative weight in either child node.

#         check_input : bool, default=True
#             Allow to bypass several input checking.
#             Don't use this parameter unless you know what you do.

#         classes : array-like of shape (n_classes,), default=None
#             List of all the classes that can possibly appear in the y vector.

#         Returns
#         -------
#         self : HonestTreeClassifier
#             Fitted tree estimator.
#         """
#         if check_input:
#             X, y = check_X_y(X, y, multi_output=True)

#         self.estimator_ = self._get_estimator()

#         # check that all of tree_estimator_params are valid
#         init_params = self.estimator_.__init__.__code__.co_varnames[1:]  # exclude 'self'
#         honest_tree_init_params = self.__init__.__code__.co_varnames[1:]  # exclude 'self'
#         invalid_params = []
#         for param in self._tree_estimator_params.keys():
#             if param not in init_params or param in honest_tree_init_params:
#                 invalid_params.append(param)

#         if invalid_params:
#             raise ValueError(
#                 f"Invalid parameter(s) for estimator {self.estimator_.__class__.__name__}: "
#                 f'{", ".join(invalid_params)}'
#             )

#         self.estimator_.set_params(
#             **dict(
#                 criterion=self.criterion,
#                 splitter=self.splitter,
#                 max_depth=self.max_depth,
#                 min_samples_split=self.min_samples_split,
#                 min_samples_leaf=self.min_samples_leaf,
#                 min_weight_fraction_leaf=self.min_weight_fraction_leaf,
#                 max_features=self.max_features,
#                 max_leaf_nodes=self.max_leaf_nodes,
#                 class_weight=self.class_weight,
#                 min_impurity_decrease=self.min_impurity_decrease,
#                 ccp_alpha=self.ccp_alpha,
#                 random_state=self.random_state,
#             )
#         )

#         try:
#             self.estimator_.set_params(**dict(monotonic_cst=self.monotonic_cst))
#             self.estimator_.set_params(
#                 **dict(
#                     store_leaf_values=self.store_leaf_values,
#                 )
#             )
#         except Exception:
#             from warnings import warn

#             warn("Using sklearn tree so store_leaf_values cannot be set.")

#         # obtain the structure sample weights
#         sample_weights_structure = self._partition_honest_indices(y, sample_weight)

#         # Learn structure on subsample
#         # XXX: this allows us to use BaseDecisionTree without partial_fit API
#         try:
#             self.estimator_._fit(
#                 X,
#                 y,
#                 sample_weight=sample_weights_structure,
#                 check_input=check_input,
#                 missing_values_in_feature_mask=missing_values_in_feature_mask,
#                 classes=classes,
#             )
#         except Exception:
#             self.estimator_._fit(
#                 X,
#                 y,
#                 sample_weight=sample_weights_structure,
#                 check_input=check_input,
#                 missing_values_in_feature_mask=missing_values_in_feature_mask,
#             )
#         self._inherit_estimator_attributes()

#         # fit the leaves on the non-structure indices
#         not_honest_mask = np.ones(len(y), dtype=bool)
#         not_honest_mask[self.honest_indices_] = False

#         if sample_weight is None:
#             sample_weight_leaves = np.ones((len(y),), dtype=np.float64)
#         else:
#             sample_weight_leaves = np.array(sample_weight)
#         sample_weight_leaves[not_honest_mask] = 0

#         # determine the honest indices using the sample weight
#         nonzero_indices = np.where(sample_weight_leaves > 0)[0]
#         # sample the structure indices
#         self.honest_indices_ = nonzero_indices

#         self._fit_leaves(X, y, sample_weight=sample_weight_leaves)
#         return self

#     def _fit_leaves(self, X, y, sample_weight):
#         # update the number of classes, unsplit
#         if y.ndim == 1:
#             # reshape is necessary to preserve the data contiguity against vs
#             # [:, np.newaxis] that does not.
#             y = np.reshape(y, (-1, 1))
#         check_classification_targets(y)
#         y = np.copy(y)  # .astype(int)

#         # Normally called by super
#         X = self.estimator_._validate_X_predict(X, True)

#         # preserve from underlying tree
#         # https://github.com/scikit-learn/scikit-learn/blob/1.0.X/sklearn/tree/_classes.py#L202
#         self._tree_classes_ = self.classes_
#         self._tree_n_classes_ = self.n_classes_
#         self.classes_ = []
#         self.n_classes_ = []
#         self.empirical_prior_ = []

#         y_encoded = np.zeros(y.shape, dtype=int)
#         for k in range(self.n_outputs_):
#             classes_k, y_encoded[:, k] = np.unique(y[:, k], return_inverse=True)
#             self.classes_.append(classes_k)
#             self.n_classes_.append(classes_k.shape[0])
#             self.empirical_prior_.append(
#                 np.bincount(y_encoded[:, k], minlength=classes_k.shape[0]) / y.shape[0]
#             )
#         y = y_encoded
#         self.n_classes_ = np.array(self.n_classes_, dtype=np.intp)

#         # XXX: implement honest pruning
#         honest_method = "apply"
#         if honest_method == "apply":
#             # Fit leaves using other subsample
#             honest_leaves = self.tree_.apply(X[self.honest_indices_])

#             # y-encoded ensures that y values match the indices of the classes
#             self._set_leaf_nodes(honest_leaves, y, sample_weight)
#         elif honest_method == "prune":
#             raise NotImplementedError("Pruning is not yet implemented.")

#         if self.n_outputs_ == 1:
#             self.n_classes_ = self.n_classes_[0]
#             self.classes_ = self.classes_[0]
#             self.empirical_prior_ = self.empirical_prior_[0]
#             y = y[:, 0]

#     def _set_leaf_nodes(self, leaf_ids, y, sample_weight):
#         """Traverse the already built tree with X and set leaf nodes with y.

#         tree_.value has shape (n_nodes, n_outputs, max_n_classes), where
#         n_nodes are the number of nodes in the tree (each node is either a split,
#         or leaf node), n_outputs is the number of outputs (1 for classification,
#         n for regression), and max_n_classes is the maximum number of classes
#         across all outputs. For classification with n_classes classes, the
#         classes are ordered by their index in the tree_.value array.
#         """
#         self.tree_.value[:, :, :] = 0

#         # apply sample-weight to the leaf nodes
#         for leaf_id, yval, y_weight in zip(
#             leaf_ids, y[self.honest_indices_, :], sample_weight[self.honest_indices_]
#         ):
#             self.tree_.value[leaf_id][:, yval] += y_weight

#     def _inherit_estimator_attributes(self):
#         """Initialize necessary attributes from the provided tree estimator"""
#         if hasattr(self.estimator_, "_inheritable_fitted_attribute"):
#             for attr in self.estimator_._inheritable_fitted_attribute:
#                 setattr(self, attr, getattr(self.estimator_, attr))

#         self.classes_ = self.estimator_.classes_
#         self.max_features_ = self.estimator_.max_features_
#         self.n_classes_ = self.estimator_.n_classes_
#         self.n_features_in_ = self.estimator_.n_features_in_
#         self.n_outputs_ = self.estimator_.n_outputs_
#         self.tree_ = self.estimator_.tree_

#         # XXX: scikit-learn trees do not store their builder, or min_samples_split_
#         self.min_samples_split_ = getattr(self.estimator_, "min_samples_split_", None)
#         self.min_samples_leaf_ = getattr(self.estimator_, "min_samples_leaf_", None)
#         self.min_weight_leaf_ = getattr(self.estimator_, "min_weight_leaf_", None)
#         self.monotonic_cst_ = getattr(self.estimator_, "monotonic_cst_", None)

#     def _empty_leaf_correction(self, proba, pos=0):
#         """Leaves with empty posteriors are assigned values.

#         This is called only during prediction.

#         The posteriors are corrected according to the honest prior.
#         In multi-output cases, the posterior corrections only correspond
#         to the respective y dimension, indicated by the position param pos.
#         """
#         zero_mask = proba.sum(axis=1) == 0.0

#         # For multi-output cases
#         if self.n_outputs_ > 1:
#             if self.honest_prior == "empirical":
#                 proba[zero_mask] = self.empirical_prior_[pos]
#             elif self.honest_prior == "uniform":
#                 proba[zero_mask] = 1 / self.n_classes_[pos]
#             elif self.honest_prior == "ignore":
#                 proba[zero_mask] = np.nan
#         else:
#             if self.honest_prior == "empirical":
#                 proba[zero_mask] = self.empirical_prior_
#             elif self.honest_prior == "uniform":
#                 proba[zero_mask] = 1 / self.n_classes_
#             elif self.honest_prior == "ignore":
#                 proba[zero_mask] = np.nan
#         return proba

#     def predict_proba(self, X, check_input=True):
#         """Predict class probabilities of the input samples X.

#         The predicted class probability is the fraction of samples of the same
#         class in a leaf.

#         Parameters
#         ----------
#         X : {array-like, sparse matrix} of shape (n_samples, n_features)
#             The input samples. Internally, it will be converted to
#             ``dtype=np.float32`` and if a sparse matrix is provided
#             to a sparse ``csr_matrix``.

#         check_input : bool, default=True
#             Allow to bypass several input checking.
#             Don't use this parameter unless you know what you do.

#         Returns
#         -------
#         proba : ndarray of shape (n_samples, n_classes) or list of n_outputs \
#             such arrays if n_outputs > 1
#             The class probabilities of the input samples. The order of the
#             classes corresponds to that in the attribute :term:`classes_`.
#         """
#         check_is_fitted(self)
#         X = self.estimator_._validate_X_predict(X, check_input)
#         proba = self.tree_.predict(X)

#         if self.n_outputs_ == 1:
#             proba = proba[:, : self._tree_n_classes_]
#             normalizer = proba.sum(axis=1)[:, np.newaxis]
#             normalizer[normalizer == 0.0] = 1.0
#             proba /= normalizer
#             proba = self._empty_leaf_correction(proba)

#             return proba

#         else:
#             all_proba = []

#             for k in range(self.n_outputs_):
#                 proba_k = proba[:, k, : self._tree_n_classes_[k]]
#                 normalizer = proba_k.sum(axis=1)[:, np.newaxis]
#                 normalizer[normalizer == 0.0] = 1.0
#                 proba_k /= normalizer
#                 proba_k = self._empty_leaf_correction(proba_k, k)
#                 all_proba.append(proba_k)

#             return all_proba

#     def predict(self, X, check_input=True):
#         """Predict class for X.

#         For a classification model, the predicted class for each sample in X is
#         returned.

#         Parameters
#         ----------
#         X : {array-like, sparse matrix} of shape (n_samples, n_features)
#             The input samples. Internally, it will be converted to
#             ``dtype=np.float32`` and if a sparse matrix is provided
#             to a sparse ``csr_matrix``.

#         check_input : bool, default=True
#             Allow to bypass several input checking.
#             Don't use this parameter unless you know what you're doing.

#         Returns
#         -------
#         y : array-like of shape (n_samples,) or (n_samples, n_outputs)
#             The predicted classes, or the predict values.
#         """
#         check_is_fitted(self)
#         X = self._validate_X_predict(X, check_input)
#         return self.estimator_.predict(X, False)
