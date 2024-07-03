from libc.math cimport floor, log2, pow


cdef class Honesty:
    def __cinit__(
        self,
        Partitioner honest_partitioner,
        list splitter_event_handlers = None,
        list split_conditions = None,
        list tree_event_handlers = None,
        intp_t min_samples_leaf
    ):
        if splitter_event_handlers is None:
            splitter_event_handlers = []
        if split_conditions is None:
            split_conditions = []
        if tree_event_handlers is None:
            tree_event_handlers = []

        self.env.partitioner = honest_partitioner
        self.splitter_event_handlers = [NodeSortFeatureHandler(&self.env)] + splitter_event_handlers
        self.split_conditions = [HonestMinSamplesLeafCondition(min_samples_leaf, &self.env)] + split_conditions
        self.tree_event_handlers = [SetActiveParentHandler(&self.env), AddNodeHandler(&self.env)] + tree_event_handlers


cdef bint _handle_set_active_parent(
    EventType event_type,
    EventHandlerEnv handler_env,
    EventData event_data
) noexcept nogil:
    if event_type != TreeBuildEvent.SET_ACTIVE_PARENT:
        return True
    
    cdef HonestEnv* env = <HonestEnv*>handler_env
    cdef TreeBuildSetActiveParentEventData* data = <TreeBuildSetActiveParentEventData*>event_data
    cdef Interval* node = &env.active_node

    if data.parent_node_id >= env.tree.size():
        return False

    env.active_is_left = data.child_is_left

    node.feature = -1
    node.split_idx = 0
    node.split_value = NAN

    if data.parent_node_id < 0:
        env.active_parent = NULL
        node.start_idx = 0
        node.n = env.samples.shape[0]
    else:
        env.active_parent = &(env.tree[data.parent_node_id])
        if env.active_is_left:
            node.start_idx = env.active_parent.start_idx
            node.n = env.active_parent.split_idx - env.active_parent.start_idx
        else:
            node.start_idx = env.active_parent.split_idx
            node.n = env.active_parent.n - env.active_parent.split_idx

    env.partitioner.init_node_split(node.start_idx, node.start_idx + node.n)

    return True

cdef class SetActiveParentHandler(EventHandler):
    def __cinit__(self, HonestEnv* env):
        self._event_types = [TreeBuildEvent.SET_ACTIVE_PARENT]
        self.event_types = self._event_types

        self.c.f = _handle_set_active_parent
        self.c.e = env


cdef bint _handle_sort_feature(
    EventType event_type,
    EventHandlerEnv handler_env,
    EventData event_data
) noexcept nogil:
    if event_type != NodeSplitEvent.SORT_FEATURE:
        return True
    
    cdef HonestEnv* env = <HonestEnv*>handler_env
    cdef NodeSortFeatureEventData* data = <NodeSortFeatureEventData*>event_data
    cdev Interval* node = &env.active_node

    node.feature = data.feature
    node.split_idx = 0
    node.split_value = NAN
    env.partitioner.sort_samples_and_feature_values(node.feature)

    return True

cdef class NodeSortFeatureHandler(EventHandler):
    def __cinit__(self, HonestEnv* env):
        self._event_types = [NodeSplitEvent.SORT_FEATURE]
        self.event_types = self._event_types

        self.c.f = _handle_sort_feature
        self.c.e = env


cdef bint _handle_add_node(
    EventType event_type,
    EventHandlerEnv handler_env,
    EventData event_data
) noexcept nogil:
    if event_type != TreeBuildEvent.ADD_NODE:
        return True

    cdef float64_t h, feature_value
    cdef intp_t i, n_left, n_missing, size = env.tree.size()
    cdef HonestEnv* env = <HonestEnv*>handler_env
    cdef TreeBuildAddNodeEventData* data = <TreeBuildAddNodeEventData*>event_data
    cdef Interval *interval, *parent

    if data.node_id >= size:
        # as a heuristic, assume a complete tree and add a level
        h = floor(log2(size))
        env.tree.resize(size + <intp_t>pow(2, h + 1))

    interval = &(env.tree[node_id])
    interval.feature = data.feature
    interval.split_value = data.split_value

    if data.parent_node_id < 0:
        # the node being added is the tree root
        interval.start_idx = 0
        interval.n = env.samples.shape[0]
    else:
        parent = &(env.tree[data.parent_node_id])

        if data.is_left:
            interval.start_idx = parent.start_idx
            interval.n = parent.split_idx - parent.start_idx
        else:
            interval.start_idx = parent.split_idx
            interval.n = parent.n - parent.split_idx

    # *we* don't need to sort to find the split pos we'll need for partitioning,
    # but the partitioner internals are so stateful we had better just do it
    # to ensure that it's in the expected state
    env.partitioner.init_node_split(interval.start_idx, interval.start_idx + interval.n)
    env.partitioner.sort_samples_and_feature_values(interval.feature)

    # count n_left to find split pos
    n_left = 0
    i = interval.start_idx
    feature_value = env.X[env.samples[i], interval.feature]

    while !isnan(feature_value) && feature_value < interval.split_value && i < interval.start_idx + interval.n:
        n_left += 1
        i += 1
        feature_value = env.X[env.samples[i], interval.feature]

    interval.split_idx = interval.start_idx + n_left

    env.partitioner.partition_samples_final(
        interval.split_idx, interval.split_value, interval.feature, partitioner.n_missing
        )

cdef class AddNodeHandler(EventHandler):
    def __cinit__(self, HonestEnv* env):
        self._event_types = [TreeBuildEvent.ADD_NODE]
        self.event_types = self._event_types

        self.c.f = _handle_add_node
        self.c.e = env


cdef bint _honest_min_sample_leaf_condition(
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
    cdef MinSamplesLeafConditionEnv* env = <MinSamplesLeafConditionEnv*>split_condition_env
    cdef HonestEnv* honest_env = env.honest_env
    cdef Interval* node = env.active_node

    cdef intp_t min_samples_leaf = env.min_samples
    cdef intp_t end_non_missing, n_left, n_right

    # we don't care about n_missing in the structure set
    n_missing = honest_env.partitioner.n_missing
    end_non_missing = node.start_idx + node.n - n_missing

    # we don't care about split_pos in the structure set,
    # need to scan forward in the honest set based on split_value to find it
    while node.split_idx < node.start_idx + node.n && env.X[node.split_idx, node.feature] <= split_value:
        node.split_idx += 1
    
    if missing_go_to_left:
        n_left = node.split_idx - node.start_idx + n_missing
        n_right = end_non_missing - node.split_idx
    else:
        n_left = node.split_idx - node.start_idx
        n_right = end_non_missing - node.split_idx + n_missing

    # Reject if min_samples_leaf is not guaranteed
    if n_left < min_samples_leaf or n_right < min_samples_leaf:
        return False

    return True

cdef class HonestMinSamplesLeafCondition(SplitCondition):
    def __cinit__(self, intp_t min_samples, HonestEnv* env):
        self._env.min_samples = min_samples
        self._env.honest_env = env

        self.c.f = _honest_min_sample_leaf_condition
        self.c.e = &self._env
