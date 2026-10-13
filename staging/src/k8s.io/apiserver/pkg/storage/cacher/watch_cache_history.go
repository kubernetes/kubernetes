/*
Copyright The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package cacher

import (
	"fmt"
	"math"
	"sort"
	"sync"
	"time"

	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apiserver/pkg/storage/cacher/metrics"
	"k8s.io/klog/v2"
)

const (
	// defaultLowerBoundCapacity is a default value for event cache capacity's lower bound.
	// TODO: Figure out, to what value we can decreased it.
	defaultLowerBoundCapacity = 100

	// defaultUpperBoundCapacity should be able to keep the required history.
	defaultUpperBoundCapacity = 100 * 1024
)

func newWatchCacheHistory(config *ImmutableWatchCacheConfig, eventFreshDuration time.Duration) *watchCacheHistory {
	h := &watchCacheHistory{
		config:             config,
		capacity:           defaultLowerBoundCapacity,
		cache:              make([]*watchCacheEvent, defaultLowerBoundCapacity),
		lowerBoundCapacity: defaultLowerBoundCapacity,
		upperBoundCapacity: capacityUpperBound(eventFreshDuration),
		startIndex:         0,
		endIndex:           0,
		eventFreshDuration: eventFreshDuration,
	}
	metrics.WatchCacheCapacity.WithLabelValues(config.groupResource.Group, config.groupResource.Resource).Set(float64(h.capacity))
	return h
}

// capacityUpperBound denotes the maximum possible capacity of the watch cache
// to which it can resize.
func capacityUpperBound(eventFreshDuration time.Duration) int {
	if eventFreshDuration <= DefaultEventFreshDuration {
		return defaultUpperBoundCapacity
	}
	// eventFreshDuration determines how long the watch events are supposed
	// to be stored in the watch cache.
	// In very high churn situations, there is a need to store more events
	// in the watch cache, hence it would have to be upsized accordingly.
	// Because of that, for larger values of eventFreshDuration, we set the
	// upper bound of the watch cache's capacity proportionally to the ratio
	// between eventFreshDuration and DefaultEventFreshDuration.
	// Given that the watch cache size can only double, we round up that
	// proportion to the next power of two.
	exponent := int(math.Ceil((math.Log2(eventFreshDuration.Seconds() / DefaultEventFreshDuration.Seconds()))))
	if maxExponent := int(math.Floor((math.Log2(math.MaxInt32 / defaultUpperBoundCapacity)))); exponent > maxExponent {
		// Making sure that the capacity's upper bound fits in a 32-bit integer.
		exponent = maxExponent
		klog.Warningf("Capping watch cache capacity upper bound to %v", defaultUpperBoundCapacity<<exponent)
	}
	return defaultUpperBoundCapacity << exponent
}

type watchCacheHistory struct {
	config *ImmutableWatchCacheConfig

	// Maximum size of history window.
	capacity int

	// upper bound of capacity since event cache has a dynamic size.
	upperBoundCapacity int

	// lower bound of capacity since event cache has a dynamic size.
	lowerBoundCapacity int

	// cache is used a cyclic buffer - the "current" contents of it are
	// stored in [start_index%capacity, end_index%capacity) - so the
	// "current" contents have exactly end_index-start_index items.
	cache      []*watchCacheEvent
	startIndex int
	endIndex   int
	// removedEventSinceRelist holds the information whether any of the events
	// were already removed from the `cache` cyclic buffer since the last relist
	removedEventSinceRelist bool

	// eventFreshDuration defines the minimum watch history watchcache will store.
	eventFreshDuration time.Duration
}

// Assumes that lock is already held for write.
func (w *watchCacheHistory) updateCache(event *watchCacheEvent) {
	w.resizeCacheLocked(event.RecordTime)
	if w.isCacheFullLocked() {
		// Cache is full - remove the oldest element.
		w.startIndex++
		w.removedEventSinceRelist = true
	}
	w.cache[w.endIndex%w.capacity] = event
	w.endIndex++
}

// resizeCacheLocked resizes the cache if necessary:
// - increases capacity by 2x if cache is full and all cached events occurred within last eventFreshDuration.
// - decreases capacity by 2x when recent quarter of events occurred outside of eventFreshDuration(protect watchCache from flapping).
func (w *watchCacheHistory) resizeCacheLocked(eventTime time.Time) {
	if w.isCacheFullLocked() && eventTime.Sub(w.cache[w.startIndex%w.capacity].RecordTime) < w.eventFreshDuration {
		capacity := min(w.capacity*2, w.upperBoundCapacity)
		if capacity > w.capacity {
			w.doCacheResizeLocked(capacity)
		}
		return
	}
	if w.isCacheFullLocked() && eventTime.Sub(w.cache[(w.endIndex-w.capacity/4)%w.capacity].RecordTime) > w.eventFreshDuration {
		capacity := max(w.capacity/2, w.lowerBoundCapacity)
		if capacity < w.capacity {
			w.doCacheResizeLocked(capacity)
		}
		return
	}
}

// isCacheFullLocked used to judge whether watchCacheEvent is full.
// Assumes that lock is already held for write.
func (w *watchCacheHistory) isCacheFullLocked() bool {
	return w.endIndex == w.startIndex+w.capacity
}

// doCacheResizeLocked resize watchCache's event array with different capacity.
// Assumes that lock is already held for write.
func (w *watchCacheHistory) doCacheResizeLocked(capacity int) {
	newCache := make([]*watchCacheEvent, capacity)
	if capacity < w.capacity {
		// adjust startIndex if cache capacity shrink.
		w.startIndex = w.endIndex - capacity
	}
	for i := w.startIndex; i < w.endIndex; i++ {
		newCache[i%capacity] = w.cache[i%w.capacity]
	}
	w.cache = newCache
	metrics.RecordsWatchCacheCapacityChange(w.config.groupResource, w.capacity, capacity)
	w.capacity = capacity
}

// isIndexValidLocked checks if a given index is still valid.
// This assumes that the lock is held.
func (w *watchCacheHistory) isIndexValidLocked(index int) bool {
	return index >= w.startIndex
}

// ResetLocked empties the cyclic buffer, ensuring startIndex doesn't decrease.
// Assumes that lock is already held for write.
func (w *watchCacheHistory) ResetLocked() {
	w.startIndex = w.endIndex
	w.removedEventSinceRelist = false
	clear(w.cache)
}

const (
	// minWatchChanSize is the min size of channels used by the watch.
	// We keep that set to 10 for "backward compatibility" until we
	// convince ourselves based on some metrics that decreasing is safe.
	minWatchChanSize = 10
	// maxWatchChanSizeWithIndexAndTrigger is the max size of the channel
	// used by the watch using the index and trigger selector.
	maxWatchChanSizeWithIndexAndTrigger = 10
	// maxWatchChanSizeWithIndexWithoutTrigger is the max size of the channel
	// used by the watch using the index but without triggering selector.
	// We keep that set to 1000 for "backward compatibility", until we
	// convinced ourselves based on some metrics that decreasing is safe.
	maxWatchChanSizeWithIndexWithoutTrigger = 1000
	// maxWatchChanSizeWithoutIndex is the max size of the channel
	// used by the watch not using the index.
	maxWatchChanSizeWithoutIndex = 100
)

func (w *watchCacheHistory) suggestedWatchChannelSize(indexExists, triggerUsed bool) int {
	// To estimate the channel size we use a heuristic that a channel
	// should roughly be able to keep one second of history.
	// We don't have an exact data, but given we store updates from
	// the last <eventFreshDuration>, we approach it by dividing the
	// capacity by the length of the history window.
	chanSize := int(math.Ceil(float64(w.capacity) / w.eventFreshDuration.Seconds()))

	// Finally we adjust the size to avoid ending with too low or
	// to large values.
	chanSize = max(chanSize, minWatchChanSize)
	var maxChanSize int
	switch {
	case indexExists && triggerUsed:
		maxChanSize = maxWatchChanSizeWithIndexAndTrigger
	case indexExists && !triggerUsed:
		maxChanSize = maxWatchChanSizeWithIndexWithoutTrigger
	case !indexExists:
		maxChanSize = maxWatchChanSizeWithoutIndex
	}
	return min(chanSize, maxChanSize)
}

// GetIntervalLocked returns a watchCacheInterval that can be used to
// retrieve events since a certain resourceVersion. This function assumes to
// be called under the lock.
func (w *watchCacheHistory) GetIntervalLocked(resourceVersion uint64, listResourceVersion uint64, locker sync.Locker) (*watchCacheInterval, error) {
	size := w.endIndex - w.startIndex
	var oldest uint64
	switch {
	case listResourceVersion > 0 && !w.removedEventSinceRelist:
		// If no event was removed from the buffer since last relist, the oldest watch
		// event we can deliver is one greater than the resource version of the list.
		oldest = listResourceVersion + 1
	case size > 0:
		// If the previous condition is not satisfied: either some event was already
		// removed from the buffer or we've never completed a list (the latter can
		// only happen in unit tests that populate the buffer without performing
		// list/replace operations), the oldest watch event we can deliver is the first
		// one in the buffer.
		oldest = w.cache[w.startIndex%w.capacity].ResourceVersion
	default:
		return nil, fmt.Errorf("watch cache isn't correctly initialized")
	}

	if resourceVersion < oldest-1 {
		return nil, errors.NewResourceExpired(fmt.Sprintf("too old resource version: %d (%d)", resourceVersion, oldest-1))
	}

	// Binary search the smallest index at which resourceVersion is greater than the given one.
	f := func(i int) bool {
		return w.cache[(w.startIndex+i)%w.capacity].ResourceVersion > resourceVersion
	}
	first := sort.Search(size, f)
	indexerFunc := func(i int) *watchCacheEvent {
		return w.cache[i%w.capacity]
	}
	ci := newCacheInterval(w.startIndex+first, w.endIndex, indexerFunc, w.config.indexValidator, resourceVersion, locker)
	return ci, nil
}

// OldestResourceVersionLocked returns the resource version of the oldest event in the cyclic buffer.
func (w *watchCacheHistory) OldestResourceVersionLocked() uint64 {
	return w.cache[w.startIndex%w.capacity].ResourceVersion
}

// Capacity returns the current capacity of the event history cache.
func (w *watchCacheHistory) Capacity() int {
	return w.capacity
}
