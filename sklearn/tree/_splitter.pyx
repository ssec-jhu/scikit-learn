# Authors: Gilles Louppe <g.louppe@gmail.com>
#          Peter Prettenhofer <peter.prettenhofer@gmail.com>
#          Brian Holt <bdholt1@gmail.com>
#          Noel Dawe <noel@dawe.me>
#          Satrajit Gosh <satrajit.ghosh@gmail.com>
#          Lars Buitinck
#          Arnaud Joly <arnaud.v.joly@gmail.com>
#          Joel Nothman <joel.nothman@gmail.com>
#          Fares Hedayati <fares.hedayati@gmail.com>
#          Jacob Schreiber <jmschreiber91@gmail.com>
#          Adam Li <adam2392@gmail.com>
#          Jong Shin <jshinm@gmail.com>
#          Samuel Carliles <scarlil1@jhu.edu>
#

# License: BSD 3 clause
# SPDX-License-Identifier: BSD-3-Clause


from libc.stdlib cimport malloc
from libc.string cimport memcpy

from ._criterion cimport Criterion
from ._sort cimport FEATURE_THRESHOLD
from ._utils cimport rand_int
from ._utils cimport rand_uniform
from ._utils cimport RAND_R_MAX
from ..utils._typedefs cimport int8_t
from ._criterion cimport Criterion
from ._partitioner cimport DensePartitioner, SparsePartitioner

from ._utils cimport RAND_R_MAX, rand_int, rand_uniform

import numpy as np


cdef float64_t INFINITY = np.inf


# we refactor the inline min sample leaf split rejection criterion
# into our injectable SplitCondition pattern
cdef bint min_sample_leaf_condition(
    Splitter splitter,
    intp_t split_feature,
    intp_t split_pos,
    float64_t split_value,
    intp_t n_missing,
    bint missing_go_to_left,
    float64_t lower_bound,
    float64_t upper_bound,
    SplitConditionEnv split_condition_env
) noexcept nogil:
    cdef intp_t min_samples_leaf = splitter.min_samples_leaf
    cdef intp_t end_non_missing = splitter.end - n_missing
    cdef intp_t n_left, n_right

    if missing_go_to_left:
        n_left = split_pos - splitter.start + n_missing
        n_right = end_non_missing - split_pos
    else:
        n_left = split_pos - splitter.start
        n_right = end_non_missing - split_pos + n_missing

    # Reject if min_samples_leaf is not guaranteed
    if n_left < min_samples_leaf or n_right < min_samples_leaf:
        return False

    return True

cdef class MinSamplesLeafCondition(SplitCondition):
    def __cinit__(self):
        self.c.f = min_sample_leaf_condition
        self.c.e = NULL # min_samples is stored in splitter, which is already passed to f


# we refactor the inline min weight leaf split rejection criterion
# into our injectable SplitCondition pattern
cdef bint min_weight_leaf_condition(
    Splitter splitter,
    intp_t split_feature,
    intp_t split_pos,
    float64_t split_value,
    intp_t n_missing,
    bint missing_go_to_left,
    float64_t lower_bound,
    float64_t upper_bound,
    SplitConditionEnv split_condition_env
) noexcept nogil:
    cdef float64_t min_weight_leaf = splitter.min_weight_leaf

    # Reject if min_weight_leaf is not satisfied
    if ((splitter.criterion.weighted_n_left < min_weight_leaf) or
            (splitter.criterion.weighted_n_right < min_weight_leaf)):
        return False

    return True

cdef class MinWeightLeafCondition(SplitCondition):
    def __cinit__(self):
        self.c.f = min_weight_leaf_condition
        self.c.e = NULL # min_weight_leaf is stored in splitter, which is already passed to f


# we refactor the inline monotonic constraint split rejection criterion
# into our injectable SplitCondition pattern
cdef bint monotonic_constraint_condition(
    Splitter splitter,
    intp_t split_feature,
    intp_t split_pos,
    float64_t split_value,
    intp_t n_missing,
    bint missing_go_to_left,
    float64_t lower_bound,
    float64_t upper_bound,
    SplitConditionEnv split_condition_env
) noexcept nogil:
    if (
        splitter.with_monotonic_cst and
        splitter.monotonic_cst[split_feature] != 0 and
        not splitter.criterion.check_monotonicity(
            splitter.monotonic_cst[split_feature],
            lower_bound,
            upper_bound,
        )
    ):
        return False
    
    return True

cdef class MonotonicConstraintCondition(SplitCondition):
    def __cinit__(self):
        self.c.f = monotonic_constraint_condition
        self.c.e = NULL


cdef inline void _init_split(SplitRecord* self, intp_t start_pos) noexcept nogil:
    self.impurity_left = INFINITY
    self.impurity_right = INFINITY
    self.pos = start_pos
    self.feature = 0
    self.threshold = 0.
    self.improvement = -INFINITY
    self.missing_go_to_left = False
    self.n_missing = 0

# the default SplitRecord factory method simply mallocs a SplitRecord
cdef SplitRecord* _base_split_record_factory(SplitRecordFactoryEnv env) except NULL nogil:
    return <SplitRecord*>malloc(sizeof(SplitRecord));

cdef class BaseSplitter:
    """This is an abstract interface for splitters.

    For example, a tree model could be either supervisedly, or unsupervisedly computing splits on samples of
    covariates, labels, or both. Although scikit-learn currently only contains
    supervised tree methods, this class enables 3rd party packages to leverage
    scikit-learn's Cython code for splitting.

    A splitter is usually used in conjunction with a criterion class, which explicitly handles
    computing the criteria, which we split on. The setting of that criterion class is handled
    by downstream classes.

    The downstream classes _must_ implement methods to compute the split in a node.
    """

    def __getstate__(self):
        return {}

    def __setstate__(self, d):
        pass

    cdef int node_reset(
        self,
        intp_t start,
        intp_t end,
        float64_t* weighted_n_node_samples
    ) except -1 nogil:
        """Reset splitter on node samples[start:end].

        Returns -1 in case of failure to allocate memory (and raise MemoryError)
        or 0 otherwise.

        Parameters
        ----------
        start : intp_t
            The index of the first sample to consider
        end : intp_t
            The index of the last sample to consider
        weighted_n_node_samples : ndarray, dtype=float64_t pointer
            The total weight of those samples
        """
        pass

    cdef int node_split(
        self,
        ParentInfo* parent,
        SplitRecord* split,
    ) except -1 nogil:
        """Find the best split on node samples[start:end].

        This is a placeholder method. The majority of computation will be done
        here.

        It should return -1 upon errors.

        Parameters
        ----------
        impurity : float64_t
            The impurity of the current node.
        split : SplitRecord pointer
            A pointer to a memory-allocated SplitRecord object which will be filled with the
            split chosen.
        lower_bound : float64_t
            The lower bound of the monotonic constraint if used.
        upper_bound : float64_t
            The upper bound of the monotonic constraint if used.
        """
        pass

    cdef void node_value(self, float64_t* dest) noexcept nogil:
        """Copy the value of node samples[start:end] into dest."""
        pass

    cdef float64_t node_impurity(self) noexcept nogil:
        """Return the impurity of the current node."""
        pass

    cdef intp_t pointer_size(self) noexcept nogil:
        """Size of the pointer for split records.

        Overriding this function allows one to use different subclasses of
        `SplitRecord`.
        """
        return sizeof(SplitRecord)
    
    cdef SplitRecord* create_split_record(self) except NULL nogil:
        return self.split_record_factory.f(self.split_record_factory.e)

cdef class Splitter(BaseSplitter):
    """Abstract interface for supervised splitters."""

    def __cinit__(
        self,
        Criterion criterion,
        intp_t max_features,
        intp_t min_samples_leaf,
        float64_t min_weight_leaf,
        object random_state,
        const int8_t[:] monotonic_cst,
        presplit_conditions : [SplitCondition] = None,
        postsplit_conditions : [SplitCondition] = None,
        listeners : [EventHandler] = None,
        *argv
    ):
        """
        Parameters
        ----------
        criterion : Criterion
            The criterion to measure the quality of a split.

        max_features : intp_t
            The maximal number of randomly selected features which can be
            considered for a split.

        min_samples_leaf : intp_t
            The minimal number of samples each leaf can have, where splits
            which would result in having less samples in a leaf are not
            considered.

        min_weight_leaf : float64_t
            The minimal weight each leaf can have, where the weight is the sum
            of the weights of each sample in it.

        random_state : object
            The user inputted random state to be used for pseudo-randomness

        monotonic_cst : const int8_t[:]
            Monotonicity constraints

        """
        self.criterion = criterion

        self.n_samples = 0
        self.n_features = 0

        self.max_features = max_features
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_leaf = min_weight_leaf
        self.random_state = random_state
        self.monotonic_cst = monotonic_cst
        self.with_monotonic_cst = monotonic_cst is not None

        self.event_broker = EventBroker(listeners, [NodeSplitEvent.SORT_FEATURE])

        self.min_samples_leaf_condition = MinSamplesLeafCondition()
        self.min_weight_leaf_condition = MinWeightLeafCondition()

        l_pre = [self.min_samples_leaf_condition]
        l_post = [self.min_weight_leaf_condition]

        if(self.with_monotonic_cst):
            self.monotonic_constraint_condition = MonotonicConstraintCondition()
            l_pre.append(self.monotonic_constraint_condition)
            l_post.append(self.monotonic_constraint_condition)
            #self.presplit_conditions[offset] = self.monotonic_constraint_condition.c
            #self.postsplit_conditions[offset] = self.monotonic_constraint_condition.c
            #offset += 1

        if presplit_conditions is not None:
            l_pre += presplit_conditions
        
        if postsplit_conditions is not None:
            l_post += postsplit_conditions
        
        self.presplit_conditions.resize(0)
        self.add_presplit_conditions(l_pre)

        self.postsplit_conditions.resize(0)
        self.add_postsplit_conditions(l_post)

        self.split_record_factory.f = _base_split_record_factory
        self.split_record_factory.e = NULL

    def add_listeners(self, listeners: [EventHandler], event_types: [EventType]):
        self.broker.add_listeners(listeners, event_types)
    
    def add_presplit_conditions(self, presplit_conditions):
        self._add_conditions(&self.presplit_conditions, presplit_conditions)
    
    def add_postsplit_conditions(self, postsplit_conditions):
        self._add_conditions(&self.postsplit_conditions, postsplit_conditions)

    cdef void _add_conditions(
        self,
        vector[SplitConditionClosure]* v,
        split_conditions : [SplitCondition]
    ):
        cdef int offset, ct, i

        offset = v.size()
        if split_conditions is not None:
            ct = len(split_conditions)
            v.resize(offset + ct)
            for i in range(ct):
                v[0][i + offset] = (<SplitCondition>split_conditions[i]).c

    
    def __reduce__(self):
        return (type(self), (self.criterion,
                             self.max_features,
                             self.min_samples_leaf,
                             self.min_weight_leaf,
                             self.random_state,
                             self.monotonic_cst.base if self.monotonic_cst is not None else None), self.__getstate__())

    cdef int init(
        self,
        object X,
        const float64_t[:, ::1] y,
        const float64_t[:] sample_weight,
        const uint8_t[::1] missing_values_in_feature_mask,
    ) except -1:
        """Initialize the splitter.

        Take in the input data X, the target Y, and optional sample weights.

        Returns -1 in case of failure to allocate memory (and raise MemoryError)
        or 0 otherwise.

        Parameters
        ----------
        X : object
            This contains the inputs. Usually it is a 2d numpy array.

        y : ndarray, dtype=float64_t
            This is the vector of targets, or true labels, for the samples represented
            as a Cython memoryview.

        sample_weight : ndarray, dtype=float64_t
            The weights of the samples, where higher weighted samples are fit
            closer than lower weight samples. If not provided, all samples
            are assumed to have uniform weight. This is represented
            as a Cython memoryview.

        has_missing : bool
            At least one missing values is in X.
        """
        self.rand_r_state = self.random_state.randint(0, RAND_R_MAX)
        cdef intp_t n_samples = X.shape[0]

        # Create a new array which will be used to store nonzero
        # samples from the feature of interest
        self.samples = np.empty(n_samples, dtype=np.intp)
        cdef intp_t[::1] samples = self.samples

        cdef intp_t i, j
        cdef float64_t weighted_n_samples = 0.0
        j = 0

        for i in range(n_samples):
            # Only work with positively weighted samples
            if sample_weight is None or sample_weight[i] != 0.0:
                samples[j] = i
                j += 1

            if sample_weight is not None:
                weighted_n_samples += sample_weight[i]
            else:
                weighted_n_samples += 1.0

        # Number of samples is number of positively weighted samples
        self.n_samples = j
        self.weighted_n_samples = weighted_n_samples

        cdef intp_t n_features = X.shape[1]
        self.features = np.arange(n_features, dtype=np.intp)
        self.n_features = n_features

        self.feature_values = np.empty(n_samples, dtype=np.float32)
        self.constant_features = np.empty(n_features, dtype=np.intp)

        self.y = y

        self.sample_weight = sample_weight

        self.criterion.init(
            self.y,
            self.sample_weight,
            self.weighted_n_samples,
            self.samples
        )

        self.criterion.set_sample_pointers(
            self.start,
            self.end
        )
        if missing_values_in_feature_mask is not None:
            self.criterion.init_sum_missing()

        return 0

    cdef int node_reset(
        self,
        intp_t start,
        intp_t end,
        float64_t* weighted_n_node_samples
    ) except -1 nogil:
        """Reset splitter on node samples[start:end].

        Returns -1 in case of failure to allocate memory (and raise MemoryError)
        or 0 otherwise.

        Parameters
        ----------
        start : intp_t
            The index of the first sample to consider
        end : intp_t
            The index of the last sample to consider
        weighted_n_node_samples : ndarray, dtype=float64_t pointer
            The total weight of those samples
        """

        self.start = start
        self.end = end

        self.criterion.set_sample_pointers(start, end)

        weighted_n_node_samples[0] = self.criterion.weighted_n_node_samples
        return 0

    cdef int node_split(
        self,
        ParentInfo* parent_record,
        SplitRecord* split,
    ) except -1 nogil:

        """Find the best split on node samples[start:end].

        This is a placeholder method. The majority of computation will be done
        here.

        It should return -1 upon errors.
        """

        pass

    cdef void node_value(self, float64_t* dest) noexcept nogil:
        """Copy the value of node samples[start:end] into dest."""

        self.criterion.node_value(dest)

    cdef inline void clip_node_value(self, float64_t* dest, float64_t lower_bound, float64_t upper_bound) noexcept nogil:
        """Clip the value in dest between lower_bound and upper_bound for monotonic constraints."""

        self.criterion.clip_node_value(dest, lower_bound, upper_bound)

    cdef void node_samples(self, vector[vector[float64_t]]& dest) noexcept nogil:
        """Copy the samples[start:end] into dest."""
        self.criterion.node_samples(dest)

    cdef float64_t node_impurity(self) noexcept nogil:
        """Return the impurity of the current node."""

        return self.criterion.node_impurity()

    cdef inline bint check_presplit_conditions(
        self,
        SplitRecord* current_split,
        intp_t n_missing,
        bint missing_go_to_left,
    ) noexcept nogil:
        """Check stopping conditions pre-split.

        This is typically a metric that is cheaply computed given the
        current proposed split, which is stored as a the `current_split`
        argument.

        Returns 1 if not a valid split, and 0 if it is.
        """
        cdef intp_t min_samples_leaf = self.min_samples_leaf
        cdef intp_t end_non_missing = self.end - n_missing
        cdef intp_t n_left, n_right

        if missing_go_to_left:
            n_left = current_split.pos - self.start + n_missing
            n_right = end_non_missing - current_split.pos
        else:
            n_left = current_split.pos - self.start
            n_right = end_non_missing - current_split.pos + n_missing

        # Reject if min_samples_leaf is not guaranteed
        if n_left < min_samples_leaf or n_right < min_samples_leaf:
            return 1

        return 0

    cdef inline bint check_postsplit_conditions(
        self
    ) noexcept nogil:
        """Check stopping conditions after evaluating the split.

        This takes some metric that is stored in the Criterion
        object and checks against internal stop metrics.

        Returns 1 if condition is not met, and 0 if it is.
        """
        cdef float64_t min_weight_leaf = self.min_weight_leaf

        # Reject if min_weight_leaf is not satisfied
        if ((self.criterion.weighted_n_left < min_weight_leaf) or
                (self.criterion.weighted_n_right < min_weight_leaf)):
            return 1

        return 0


cdef inline void shift_missing_values_to_left_if_required(
    SplitRecord* best,
    intp_t[::1] samples,
    intp_t end,
) noexcept nogil:
    """Shift missing value sample indices to the left of the split if required.

    Note: this should always be called at the very end because it will
    move samples around, thereby affecting the criterion.
    This affects the computation of the children impurity, which affects
    the computation of the next node.
    """
    cdef intp_t i, p, current_end
    # The partitioner partitions the data such that the missing values are in
    # samples[-n_missing:] for the criterion to consume. If the missing values
    # are going to the right node, then the missing values are already in the
    # correct position. If the missing values go left, then we move the missing
    # values to samples[best.pos:best.pos+n_missing] and update `best.pos`.
    if best.n_missing > 0 and best.missing_go_to_left:
        for p in range(best.n_missing):
            i = best.pos + p
            current_end = end - 1 - p
            samples[i], samples[current_end] = samples[current_end], samples[i]
        best.pos += best.n_missing


cdef inline intp_t node_split_best(
    Splitter splitter,
    Partitioner partitioner,
    Criterion criterion,
    SplitRecord* split,
    ParentInfo* parent_record,
) except -1 nogil:
    """Find the best split on node samples[start:end]

    Returns -1 in case of failure to allocate memory (and raise MemoryError)
    or 0 otherwise.
    """
    cdef const int8_t[:] monotonic_cst = splitter.monotonic_cst
    cdef bint with_monotonic_cst = splitter.with_monotonic_cst

    # Find the best split
    cdef intp_t start = splitter.start
    cdef intp_t end = splitter.end
    cdef intp_t end_non_missing
    cdef intp_t n_missing = 0
    cdef bint has_missing = 0
    cdef intp_t n_searches
    cdef intp_t n_left, n_right
    cdef bint missing_go_to_left

    cdef intp_t[::1] samples = splitter.samples
    cdef intp_t[::1] features = splitter.features
    cdef intp_t[::1] constant_features = splitter.constant_features
    cdef intp_t n_features = splitter.n_features

    cdef float32_t[::1] feature_values = splitter.feature_values
    cdef intp_t max_features = splitter.max_features
    cdef intp_t min_samples_leaf = splitter.min_samples_leaf
    cdef float64_t min_weight_leaf = splitter.min_weight_leaf
    cdef uint32_t* random_state = &splitter.rand_r_state

    cdef SplitRecord best_split, current_split
    cdef float64_t current_threshold
    cdef float64_t current_proxy_improvement = -INFINITY
    cdef float64_t best_proxy_improvement = -INFINITY

    cdef float64_t impurity = parent_record.impurity
    cdef float64_t lower_bound = parent_record.lower_bound
    cdef float64_t upper_bound = parent_record.upper_bound

    cdef intp_t f_i = n_features
    cdef intp_t f_j
    cdef intp_t p
    cdef intp_t p_prev

    cdef intp_t n_visited_features = 0
    # Number of features discovered to be constant during the split search
    cdef intp_t n_found_constants = 0
    # Number of features known to be constant and drawn without replacement
    cdef intp_t n_drawn_constants = 0
    cdef intp_t n_known_constants = parent_record.n_constant_features
    # n_total_constants = n_known_constants + n_found_constants
    cdef intp_t n_total_constants = n_known_constants

    cdef bint conditions_hold = True

    # payloads for different node events
    cdef NodeSortFeatureEventData sort_event_data
    cdef NodeSplitEventData split_event_data

    _init_split(&best_split, end)

    partitioner.init_node_split(start, end)

    # Sample up to max_features without replacement using a
    # Fisher-Yates-based algorithm (using the local variables `f_i` and
    # `f_j` to compute a permutation of the `features` array).
    #
    # Skip the CPU intensive evaluation of the impurity criterion for
    # features that were already detected as constant (hence not suitable
    # for good splitting) by ancestor nodes and save the information on
    # newly discovered constant features to spare computation on descendant
    # nodes.
    while (f_i > n_total_constants and  # Stop early if remaining features
                                        # are constant
            (n_visited_features < max_features or
             # At least one drawn features must be non constant
             n_visited_features <= n_found_constants + n_drawn_constants)):

        n_visited_features += 1

        # Loop invariant: elements of features in
        # - [:n_drawn_constant[ holds drawn and known constant features;
        # - [n_drawn_constant:n_known_constant[ holds known constant
        #   features that haven't been drawn yet;
        # - [n_known_constant:n_total_constant[ holds newly found constant
        #   features;
        # - [n_total_constant:f_i[ holds features that haven't been drawn
        #   yet and aren't constant apriori.
        # - [f_i:n_features[ holds features that have been drawn
        #   and aren't constant.

        # Draw a feature at random
        f_j = rand_int(n_drawn_constants, f_i - n_found_constants,
                       random_state)

        if f_j < n_known_constants:
            # f_j in the interval [n_drawn_constants, n_known_constants[
            features[n_drawn_constants], features[f_j] = features[f_j], features[n_drawn_constants]

            n_drawn_constants += 1
            continue

        # f_j in the interval [n_known_constants, f_i - n_found_constants[
        f_j += n_found_constants
        # f_j in the interval [n_total_constants, f_i[
        current_split.feature = features[f_j]
        partitioner.sort_samples_and_feature_values(current_split.feature)

        # notify any interested parties which feature we're investingating splits for now
        sort_event_data.feature = current_split.feature
        splitter.event_broker.fire_event(NodeSplitEvent.SORT_FEATURE, &sort_event_data)

        n_missing = partitioner.n_missing
        end_non_missing = end - n_missing

        if (
            # All values for this feature are missing, or
            end_non_missing == start or
            # This feature is considered constant (max - min <= FEATURE_THRESHOLD)
            feature_values[end_non_missing - 1] <= feature_values[start] + FEATURE_THRESHOLD
        ):
            # We consider this feature constant in this case.
            # Since finding a split among constant feature is not valuable,
            # we do not consider this feature for splitting.
            features[f_j], features[n_total_constants] = features[n_total_constants], features[f_j]

            n_found_constants += 1
            n_total_constants += 1
            continue

        f_i -= 1
        features[f_i], features[f_j] = features[f_j], features[f_i]
        has_missing = n_missing != 0
        criterion.init_missing(n_missing)  # initialize even when n_missing == 0

        # Evaluate all splits
        # If there are missing values, then we search twice for the most optimal split.
        # The first search will have all the missing values going to the right node.
        # The second search will have all the missing values going to the left node.
        # If there are no missing values, then we search only once for the most
        # optimal split.
        n_searches = 2 if has_missing else 1

        for i in range(n_searches):
            missing_go_to_left = i == 1
            criterion.missing_go_to_left = missing_go_to_left
            criterion.reset()

            p = start

            while p < end_non_missing:
                partitioner.next_p(&p_prev, &p)

                if p >= end_non_missing:
                    continue

                current_split.pos = p

                # probably want to assign this to current_split.threshold later,
                # but the code is so stateful that Write Everything Twice is the
                # safer move here for now
                current_threshold = (
                    feature_values[p_prev] / 2.0 + feature_values[p] / 2.0
                )

                # check pre split rejection criteria
                conditions_hold = True
                for condition in splitter.presplit_conditions:
                    if not condition.f(
                        splitter, current_split.feature, current_split.pos,
                        current_threshold, n_missing, missing_go_to_left,
                        lower_bound, upper_bound, condition.e
                    ):
                        conditions_hold = False
                        break

                if not conditions_hold:
                    continue

                # Reject if min_samples_leaf is not guaranteed
                # this can probably (and should) be removed as it is generalized
                # by injectable split rejection criteria
                if splitter.check_presplit_conditions(&current_split, n_missing, missing_go_to_left) == 1:
                    continue

                criterion.update(current_split.pos)

                # check post split rejection criteria
                conditions_hold = True
                for condition in splitter.postsplit_conditions:
                    if not condition.f(
                        splitter, current_split.feature, current_split.pos,
                        current_threshold, n_missing, missing_go_to_left,
                        lower_bound, upper_bound, condition.e
                    ):
                        conditions_hold = False
                        break
                
                if not conditions_hold:
                    continue
                
                current_proxy_improvement = criterion.proxy_impurity_improvement()

                if current_proxy_improvement > best_proxy_improvement:
                    best_proxy_improvement = current_proxy_improvement
                    # sum of halves is used to avoid infinite value
                    current_split.threshold = (
                        feature_values[p_prev] / 2.0 + feature_values[p] / 2.0
                    )

                    if (
                        current_split.threshold == feature_values[p] or
                        current_split.threshold == INFINITY or
                        current_split.threshold == -INFINITY
                    ):
                        current_split.threshold = feature_values[p_prev]

                    current_split.n_missing = n_missing

                    # if there are no missing values in the training data, during
                    # test time, we send missing values to the branch that contains
                    # the most samples during training time.
                    if n_missing == 0:
                        if missing_go_to_left:
                            n_left = current_split.pos - splitter.start + n_missing
                            n_right = end_non_missing - current_split.pos
                        else:
                            n_left = current_split.pos - splitter.start
                            n_right = end_non_missing - current_split.pos + n_missing

                        current_split.missing_go_to_left = n_left > n_right
                    else:
                        current_split.missing_go_to_left = missing_go_to_left

                    best_split = current_split  # copy

        # Evaluate when there are missing values and all missing values goes
        # to the right node and non-missing values goes to the left node.
        if has_missing:
            n_left, n_right = end - start - n_missing, n_missing
            p = end - n_missing
            missing_go_to_left = 0

            if not (n_left < min_samples_leaf or n_right < min_samples_leaf):
                criterion.missing_go_to_left = missing_go_to_left
                criterion.update(p)

                if not ((criterion.weighted_n_left < min_weight_leaf) or
                        (criterion.weighted_n_right < min_weight_leaf)):
                    current_proxy_improvement = criterion.proxy_impurity_improvement()

                    if current_proxy_improvement > best_proxy_improvement:
                        best_proxy_improvement = current_proxy_improvement
                        current_split.threshold = INFINITY
                        current_split.missing_go_to_left = missing_go_to_left
                        current_split.n_missing = n_missing
                        current_split.pos = p
                        best_split = current_split


    # Reorganize into samples[start:best_split.pos] + samples[best_split.pos:end]
    if best_split.pos < end:
        partitioner.partition_samples_final(
            best_split.pos,
            best_split.threshold,
            best_split.feature,
            best_split.n_missing
        )

        criterion.init_missing(best_split.n_missing)
        criterion.missing_go_to_left = best_split.missing_go_to_left

        criterion.reset()
        criterion.update(best_split.pos)
        criterion.children_impurity(
            &best_split.impurity_left, &best_split.impurity_right
        )
        best_split.improvement = criterion.impurity_improvement(
            impurity,
            best_split.impurity_left,
            best_split.impurity_right
        )

        shift_missing_values_to_left_if_required(&best_split, samples, end)


    # Respect invariant for constant features: the original order of
    # element in features[:n_known_constants] must be preserved for sibling
    # and child nodes
    memcpy(&features[0], &constant_features[0], sizeof(intp_t) * n_known_constants)

    # Copy newly found constant features
    memcpy(&constant_features[n_known_constants],
           &features[n_known_constants],
           sizeof(intp_t) * n_found_constants)

    # Return values
    parent_record.n_constant_features = n_total_constants
    split[0] = best_split

    return 0


cdef inline int node_split_random(
    Splitter splitter,
    Partitioner partitioner,
    Criterion criterion,
    SplitRecord* split,
    ParentInfo* parent_record,
) except -1 nogil:
    """Find the best random split on node samples[start:end]

    Returns -1 in case of failure to allocate memory (and raise MemoryError)
    or 0 otherwise.
    """
    cdef const int8_t[:] monotonic_cst = splitter.monotonic_cst
    cdef bint with_monotonic_cst = splitter.with_monotonic_cst

    # Draw random splits and pick the best
    cdef intp_t start = splitter.start
    cdef intp_t end = splitter.end
    cdef intp_t end_non_missing
    cdef intp_t n_missing = 0
    cdef bint has_missing = 0
    cdef intp_t n_left, n_right
    cdef bint missing_go_to_left

    cdef intp_t[::1] samples = splitter.samples
    cdef intp_t[::1] features = splitter.features
    cdef intp_t[::1] constant_features = splitter.constant_features
    cdef intp_t n_features = splitter.n_features

    cdef intp_t max_features = splitter.max_features
    cdef uint32_t* random_state = &splitter.rand_r_state

    cdef SplitRecord best_split, current_split
    cdef float64_t current_proxy_improvement = - INFINITY
    cdef float64_t best_proxy_improvement = - INFINITY

    cdef float64_t impurity = parent_record.impurity
    cdef float64_t lower_bound = parent_record.lower_bound
    cdef float64_t upper_bound = parent_record.upper_bound

    cdef intp_t f_i = n_features
    cdef intp_t f_j
    # Number of features discovered to be constant during the split search
    cdef intp_t n_found_constants = 0
    # Number of features known to be constant and drawn without replacement
    cdef intp_t n_drawn_constants = 0
    cdef intp_t n_known_constants = parent_record.n_constant_features
    # n_total_constants = n_known_constants + n_found_constants
    cdef intp_t n_total_constants = n_known_constants
    cdef intp_t n_visited_features = 0
    cdef float32_t min_feature_value
    cdef float32_t max_feature_value

    _init_split(&best_split, end)

    partitioner.init_node_split(start, end)

    # Sample up to max_features without replacement using a
    # Fisher-Yates-based algorithm (using the local variables `f_i` and
    # `f_j` to compute a permutation of the `features` array).
    #
    # Skip the CPU intensive evaluation of the impurity criterion for
    # features that were already detected as constant (hence not suitable
    # for good splitting) by ancestor nodes and save the information on
    # newly discovered constant features to spare computation on descendant
    # nodes.
    while (f_i > n_total_constants and  # Stop early if remaining features
                                        # are constant
            (n_visited_features < max_features or
             # At least one drawn features must be non constant
             n_visited_features <= n_found_constants + n_drawn_constants)):
        n_visited_features += 1

        # Loop invariant: elements of features in
        # - [:n_drawn_constant[ holds drawn and known constant features;
        # - [n_drawn_constant:n_known_constant[ holds known constant
        #   features that haven't been drawn yet;
        # - [n_known_constant:n_total_constant[ holds newly found constant
        #   features;
        # - [n_total_constant:f_i[ holds features that haven't been drawn
        #   yet and aren't constant apriori.
        # - [f_i:n_features[ holds features that have been drawn
        #   and aren't constant.

        # Draw a feature at random
        f_j = rand_int(n_drawn_constants, f_i - n_found_constants,
                       random_state)

        if f_j < n_known_constants:
            # f_j in the interval [n_drawn_constants, n_known_constants[
            features[n_drawn_constants], features[f_j] = features[f_j], features[n_drawn_constants]
            n_drawn_constants += 1
            continue

        # f_j in the interval [n_known_constants, f_i - n_found_constants[
        f_j += n_found_constants
        # f_j in the interval [n_total_constants, f_i[

        current_split.feature = features[f_j]

        # Find min, max as we will randomly select a threshold between them
        partitioner.find_min_max(
            current_split.feature, &min_feature_value, &max_feature_value
        )
        n_missing = partitioner.n_missing
        end_non_missing = end - n_missing

        if (
            # All values for this feature are missing, or
            end_non_missing == start or
            # This feature is considered constant (max - min <= FEATURE_THRESHOLD)
            max_feature_value <= min_feature_value + FEATURE_THRESHOLD
        ):
            # We consider this feature constant in this case.
            # Since finding a split with a constant feature is not valuable,
            # we do not consider this feature for splitting.
            features[f_j], features[n_total_constants] = features[n_total_constants], current_split.feature

            n_found_constants += 1
            n_total_constants += 1
            continue

        f_i -= 1
        features[f_i], features[f_j] = features[f_j], features[f_i]
        has_missing = n_missing != 0
        criterion.init_missing(n_missing)

        # Draw a random threshold
        current_split.threshold = rand_uniform(
            min_feature_value,
            max_feature_value,
            random_state,
        )

        if has_missing:
            # If there are missing values, then we randomly make all missing
            # values go to the right or left.
            #
            # Note: compared to the BestSplitter, we do not evaluate the
            # edge case where all the missing values go to the right node
            # and the non-missing values go to the left node. This is because
            # this would indicate a threshold outside of the observed range
            # of the feature. However, it is not clear how much probability weight should
            # be given to this edge case.
            missing_go_to_left = rand_int(0, 2, random_state)
        else:
            missing_go_to_left = 0
        criterion.missing_go_to_left = missing_go_to_left

        if current_split.threshold == max_feature_value:
            current_split.threshold = min_feature_value

        # Partition
        current_split.pos = partitioner.partition_samples(
            current_split.threshold
        )

        if missing_go_to_left:
            n_left = current_split.pos - start + n_missing
            n_right = end_non_missing - current_split.pos
        else:
            n_left = current_split.pos - start
            n_right = end_non_missing - current_split.pos + n_missing

        # Reject if min_samples_leaf is not guaranteed
        if splitter.check_presplit_conditions(&current_split, n_missing, missing_go_to_left) == 1:
            continue

        # Evaluate split
        # At this point, the criterion has a view into the samples that was partitioned
        # by the partitioner. The criterion will use the partition to evaluating the split.
        criterion.reset()
        criterion.update(current_split.pos)

        # Reject if monotonicity constraints are not satisfied
        if (
            with_monotonic_cst and
            monotonic_cst[current_split.feature] != 0 and
            not criterion.check_monotonicity(
                monotonic_cst[current_split.feature],
                lower_bound,
                upper_bound,
            )
        ):
            continue

        # Reject if min_weight_leaf is not satisfied
        if splitter.check_postsplit_conditions() == 1:
            continue

        current_proxy_improvement = criterion.proxy_impurity_improvement()

        if current_proxy_improvement > best_proxy_improvement:
            current_split.n_missing = n_missing

            # if there are no missing values in the training data, during
            # test time, we send missing values to the branch that contains
            # the most samples during training time.
            if has_missing:
                current_split.missing_go_to_left = missing_go_to_left
            else:
                current_split.missing_go_to_left = n_left > n_right

            best_proxy_improvement = current_proxy_improvement
            best_split = current_split  # copy

    # Reorganize into samples[start:best.pos] + samples[best.pos:end]
    if best_split.pos < end:
        if current_split.feature != best_split.feature:
            partitioner.partition_samples_final(
                best_split.pos,
                best_split.threshold,
                best_split.feature,
                best_split.n_missing
            )
        criterion.init_missing(best_split.n_missing)
        criterion.missing_go_to_left = best_split.missing_go_to_left

        criterion.reset()
        criterion.update(best_split.pos)
        criterion.children_impurity(
            &best_split.impurity_left, &best_split.impurity_right
        )
        best_split.improvement = criterion.impurity_improvement(
            impurity,
            best_split.impurity_left,
            best_split.impurity_right
        )

        shift_missing_values_to_left_if_required(&best_split, samples, end)

    # Respect invariant for constant features: the original order of
    # element in features[:n_known_constants] must be preserved for sibling
    # and child nodes
    memcpy(&features[0], &constant_features[0], sizeof(intp_t) * n_known_constants)

    # Copy newly found constant features
    memcpy(&constant_features[n_known_constants],
           &features[n_known_constants],
           sizeof(intp_t) * n_found_constants)

    # Return values
    parent_record.n_constant_features = n_total_constants
    split[0] = best_split
    return 0


cdef class BestSplitter(Splitter):
    """Splitter for finding the best split on dense data."""
    cdef DensePartitioner partitioner
    cdef int init(
        self,
        object X,
        const float64_t[:, ::1] y,
        const float64_t[:] sample_weight,
        const uint8_t[::1] missing_values_in_feature_mask,
    ) except -1:
        Splitter.init(self, X, y, sample_weight, missing_values_in_feature_mask)
        self.partitioner = DensePartitioner(
            X, self.samples, self.feature_values, missing_values_in_feature_mask
        )

    cdef int node_split(
        self,
        ParentInfo* parent_record,
        SplitRecord* split,
    ) except -1 nogil:
        return node_split_best(
            self,
            self.partitioner,
            self.criterion,
            split,
            parent_record,
        )

cdef class BestSparseSplitter(Splitter):
    """Splitter for finding the best split, using the sparse data."""
    cdef SparsePartitioner partitioner
    cdef int init(
        self,
        object X,
        const float64_t[:, ::1] y,
        const float64_t[:] sample_weight,
        const uint8_t[::1] missing_values_in_feature_mask,
    ) except -1:
        Splitter.init(self, X, y, sample_weight, missing_values_in_feature_mask)
        self.partitioner = SparsePartitioner(
            X, self.samples, self.n_samples, self.feature_values, missing_values_in_feature_mask
        )

    cdef int node_split(
        self,
        ParentInfo* parent_record,
        SplitRecord* split,
    ) except -1 nogil:
        return node_split_best(
            self,
            self.partitioner,
            self.criterion,
            split,
            parent_record,
        )

cdef class RandomSplitter(Splitter):
    """Splitter for finding the best random split on dense data."""
    cdef DensePartitioner partitioner
    cdef int init(
        self,
        object X,
        const float64_t[:, ::1] y,
        const float64_t[:] sample_weight,
        const uint8_t[::1] missing_values_in_feature_mask,
    ) except -1:
        Splitter.init(self, X, y, sample_weight, missing_values_in_feature_mask)
        self.partitioner = DensePartitioner(
            X, self.samples, self.feature_values, missing_values_in_feature_mask
        )

    cdef int node_split(
        self,
        ParentInfo* parent_record,
        SplitRecord* split,
    ) except -1 nogil:
        return node_split_random(
            self,
            self.partitioner,
            self.criterion,
            split,
            parent_record,
        )

cdef class RandomSparseSplitter(Splitter):
    """Splitter for finding the best random split, using the sparse data."""
    cdef SparsePartitioner partitioner
    cdef int init(
        self,
        object X,
        const float64_t[:, ::1] y,
        const float64_t[:] sample_weight,
        const uint8_t[::1] missing_values_in_feature_mask,
    ) except -1:
        Splitter.init(self, X, y, sample_weight, missing_values_in_feature_mask)
        self.partitioner = SparsePartitioner(
            X, self.samples, self.n_samples, self.feature_values, missing_values_in_feature_mask
        )

    cdef int node_split(
            self,
            ParentInfo* parent_record,
            SplitRecord* split,
    ) except -1 nogil:
        return node_split_random(
            self,
            self.partitioner,
            self.criterion,
            split,
            parent_record,
        )
