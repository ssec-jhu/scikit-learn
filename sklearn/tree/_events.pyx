
# Authors: Samuel Carliles <scarlil1@jhu.edu>
#
# License: BSD 3 clause


cdef class EventBroker:
    def __cinit__(self, EventHandler[:] listeners, int[:] event_types):
        """
        Parameters:
        - listeners (EventHandler[:])
        - event_types (int[:]): an array of EventTypes that may be fired by this EventBroker

        Notes:
        - Don't mix event types in a single EventBroker instance,
          i.e. don't use the same EventBroker for brokering NodeSplitEvent that you use
          for brokering TreeBuildEvent, etc
        """
        self.listeners.resize(max(event_types) + 1)

        if(listeners is not None):
            self.add_listeners(listeners, event_types)
        else:
            for e in event_types:
                self.listeners[e].resize(0)

    def add_listeners(self, EventHandler[:] listeners, int[:] event_types):
        cdef int e, i, j, offset, mx, ct
        cdef list l

        # listeners is a vector of vectors which we index using EventType,
        # so if event_types contains any EventType for which we don't already have a vector,
        # its integer value will be larger than our current size + 1
        mx = max(event_types)
        offset = self.listeners.size()
        if mx > offset + 1:
            self.listeners.resize(mx + 1)

        if(listeners is not None):
            for e in event_types:
                # find indices for all listeners to event type e
                l = [j for j, _l in enumerate(listeners) if e in _l.events]
                offset = self.listeners[e].size()
                ct = len(l)
                self.listeners[e].resize(offset + ct)
                for i in range(ct):
                    j = l[i]
                    self.listeners[e][offset + i] = listeners[j].c

    cdef bint fire_event(self, EventType event_type, EventData event_data) noexcept nogil:
        cdef bint result = True

        for l in self.listeners[event_type]:
            result = result and l.f(event_type, l.e, event_data)
        
        return result
