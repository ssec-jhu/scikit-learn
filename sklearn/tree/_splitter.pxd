# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

# See _splitter.pyx for details.
from libcpp.vector cimport vector

from ._partitioner cimport Partitioner, DensePartitioner, SparsePartitioner
from ._criterion cimport BaseCriterion, Criterion
from ._tree cimport ParentInfo
from ..utils._typedefs cimport float32_t, float64_t, intp_t, int8_t, int32_t, uint32_t


# Mitigate precision differences between 32 bit and 64 bit
cdef float32_t FEATURE_THRESHOLD = 1e-7


cdef struct SplitRecord:
    # Data to track sample split
    intp_t feature         # Which feature to split on.
    intp_t pos             # Split samples array at the given position,
    #                      # i.e. count of samples below threshold for feature.
    #                      # pos is >= end if the node is a leaf.
    float64_t threshold       # Threshold to split at.
    float64_t improvement     # Impurity improvement given parent node.
    float64_t impurity_left   # Impurity of the left split.
    float64_t impurity_right  # Impurity of the right split.
    unsigned char missing_go_to_left  # Controls if missing values go to the left node.
    intp_t n_missing            # Number of missing values for the feature being split on

cdef class BaseSplitter:
    """Abstract interface for splitter."""

    # The splitter searches in the input space for a feature and a threshold
    # to split the samples samples[start:end].
    #
    # The impurity computations are delegated to a criterion object.

    # Internal structures
    cdef public intp_t max_features         # Number of features to test
    cdef public intp_t min_samples_leaf     # Min samples in a leaf
    cdef public float64_t min_weight_leaf   # Minimum weight in a leaf

    cdef object random_state             # Random state
    cdef uint32_t rand_r_state           # sklearn_rand_r random number state

    cdef intp_t[::1] samples             # Sample indices in X, y
    cdef intp_t n_samples                # X.shape[0]
    cdef float64_t weighted_n_samples       # Weighted number of samples
    cdef intp_t[::1] features            # Feature indices in X
    cdef intp_t[::1] constant_features   # Constant features indices
    cdef intp_t n_features               # X.shape[1]
    cdef float32_t[::1] feature_values   # temp. array holding feature values

    cdef intp_t start                    # Start position for the current node
    cdef intp_t end                      # End position for the current node

    cdef const float64_t[:] sample_weight

    # The samples vector `samples` is maintained by the Splitter object such
    # that the samples contained in a node are contiguous. With this setting,
    # `node_split` reorganizes the node samples `samples[start:end]` in two
    # subsets `samples[start:pos]` and `samples[pos:end]`.

    # The 1-d  `features` array of size n_features contains the features
    # indices and allows fast sampling without replacement of features.

    # The 1-d `constant_features` array of size n_features holds in
    # `constant_features[:n_constant_features]` the feature ids with
    # constant values for all the samples that reached a specific node.
    # The value `n_constant_features` is given by the parent node to its
    # child nodes.  The content of the range `[n_constant_features:]` is left
    # undefined, but preallocated for performance reasons
    # This allows optimization with depth-based tree building.

    # Methods
    cdef int node_reset(
        self,
        intp_t start,
        intp_t end,
        float64_t* weighted_n_node_samples
    ) except -1 nogil
    cdef int node_split(
        self,
        ParentInfo* parent,
        SplitRecord* split,
    ) except -1 nogil
    cdef void node_value(self, float64_t* dest) noexcept nogil
    cdef float64_t node_impurity(self) noexcept nogil
    cdef intp_t pointer_size(self) noexcept nogil

cdef class Splitter(BaseSplitter):
    """Base class for supervised splitters."""

    cdef public Criterion criterion      # Impurity criterion
    cdef const float64_t[:, ::1] y

    # Monotonicity constraints for each feature.
    # The encoding is as follows:
    #   -1: monotonic decrease
    #    0: no constraint
    #   +1: monotonic increase
    cdef const int8_t[:] monotonic_cst
    cdef bint with_monotonic_cst

    cdef int init(
        self,
        object X,
        const float64_t[:, ::1] y,
        const float64_t[:] sample_weight,
        const unsigned char[::1] missing_values_in_feature_mask,
    ) except -1

    cdef void node_samples(self, vector[vector[float64_t]]& dest) noexcept nogil

    # Methods that allow modifications to stopping conditions
    cdef bint check_presplit_conditions(
        self,
        SplitRecord* current_split,
        intp_t n_missing,
        bint missing_go_to_left,
    ) noexcept nogil

    cdef bint check_postsplit_conditions(
        self
    ) noexcept nogil

    cdef void clip_node_value(
        self,
        float64_t* dest,
        float64_t lower_bound,
        float64_t upper_bound
    ) noexcept nogil

cdef void shift_missing_values_to_left_if_required(
    SplitRecord* best,
    intp_t[::1] samples,
    intp_t end,
) noexcept nogil


# Sort n-element arrays pointed to by feature_values and samples, simultaneously,
# by the values in feature_values. Algorithm: Introsort (Musser, SP&E, 1997).
cdef void sort(float32_t* feature_values, intp_t* samples, intp_t n) noexcept nogil

cdef void swap(float32_t* feature_values, intp_t* samples, intp_t i, intp_t j) noexcept nogil
cdef void sparse_swap(
    intp_t[::1] index_to_samples,
    intp_t[::1] samples,
    intp_t pos_1,
    intp_t pos_2
) noexcept nogil
