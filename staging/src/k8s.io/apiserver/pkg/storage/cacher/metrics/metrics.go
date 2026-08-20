/*
Copyright 2021 The Kubernetes Authors.

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

package metrics

import (
	"sync"
	"time"

	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apiserver/pkg/features"
	storagemetrics "k8s.io/apiserver/pkg/storage/metrics"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	compbasemetrics "k8s.io/component-base/metrics"
	"k8s.io/component-base/metrics/legacyregistry"
)

const (
	namespace = "apiserver"
	subsystem = "watch_cache"
)

// DispatchStage identifies a single stage of an event's lifecycle as it moves
// through the watch cache dispatch pipeline. It is used as the "stage" label
// value on the dispatchStageDuration metric.
//
// StageTotal is the end-to-end latency of a successfully delivered event.
// The remaining stages measure individual segments of that path.
type DispatchStage int

const (
	// StageTotal: end-to-end, etcd decode -> written to the result channel.
	StageTotal DispatchStage = iota

	// StageStorageToCache: event decoded from etcd -> event received by cacher.
	// Captures the delay between when an event is decoded from the storage backend
	// and when it is first processed by the cacher's reflector loop.
	StageStorageToCache

	// StageCacheToWatcher: watch.Event built -> written to watcher's result channel.
	// Captures time spent blocked handing the event off to the client,
	// i.e. downstream (result channel) backpressure.
	StageCacheToWatcher

	numDispatchStages
)

var dispatchStageName = [numDispatchStages]string{
	StageTotal:          "total",
	StageStorageToCache: "storage_to_cache",
	StageCacheToWatcher: "cache_to_watcher",
}

/*
 * By default, all the following metrics are defined as falling under
 * ALPHA stability level https://github.com/kubernetes/enhancements/blob/master/keps/sig-instrumentation/1209-metrics-stability/kubernetes-control-plane-metrics-stability.md#stability-classes)
 *
 * Promoting the stability level of the metric is a responsibility of the component owner, since it
 * involves explicitly acknowledging support for the metric across multiple releases, in accordance with
 * the metric stability policy.
 */
var (
	listCacheCount = compbasemetrics.NewCounterVec(
		&compbasemetrics.CounterOpts{
			Namespace:         namespace,
			Name:              "cache_list_total",
			Help:              "Number of LIST requests served from watch cache",
			StabilityLevel:    compbasemetrics.ALPHA,
			DeprecatedVersion: "1.37.0",
		},
		[]string{"group", "resource", "index"},
	)
	listCacheNumFetched = compbasemetrics.NewCounterVec(
		&compbasemetrics.CounterOpts{
			Namespace:         namespace,
			Name:              "cache_list_fetched_objects_total",
			Help:              "Number of objects read from watch cache in the course of serving a LIST request",
			StabilityLevel:    compbasemetrics.ALPHA,
			DeprecatedVersion: "1.37.0",
		},
		[]string{"group", "resource", "index"},
	)
	listCacheNumReturned = compbasemetrics.NewCounterVec(
		&compbasemetrics.CounterOpts{
			Namespace:         namespace,
			Name:              "cache_list_returned_objects_total",
			Help:              "Number of objects returned for a LIST request from watch cache",
			StabilityLevel:    compbasemetrics.ALPHA,
			DeprecatedVersion: "1.37.0",
		},
		[]string{"group", "resource"},
	)
	InitCounter = compbasemetrics.NewCounterVec(
		&compbasemetrics.CounterOpts{
			Namespace:      namespace,
			Name:           "init_events_total",
			Help:           "Counter of init events processed in watch cache broken by resource type.",
			StabilityLevel: compbasemetrics.ALPHA,
		},
		[]string{"group", "resource"},
	)

	EventsReceivedCounter = compbasemetrics.NewCounterVec(
		&compbasemetrics.CounterOpts{
			Namespace:      namespace,
			Subsystem:      subsystem,
			Name:           "events_received_total",
			Help:           "Counter of events received in watch cache broken by resource type.",
			StabilityLevel: compbasemetrics.ALPHA,
		},
		[]string{"group", "resource"},
	)

	EventsCounter = compbasemetrics.NewCounterVec(
		&compbasemetrics.CounterOpts{
			Namespace:      namespace,
			Subsystem:      subsystem,
			Name:           "events_dispatched_total",
			Help:           "Counter of events dispatched in watch cache broken by resource type.",
			StabilityLevel: compbasemetrics.ALPHA,
		},
		[]string{"group", "resource"},
	)

	TerminatedWatchersCounter = compbasemetrics.NewCounterVec(
		&compbasemetrics.CounterOpts{
			Namespace:      namespace,
			Name:           "terminated_watchers_total",
			Help:           "Counter of watchers closed due to unresponsiveness broken by resource type.",
			StabilityLevel: compbasemetrics.ALPHA,
		},
		[]string{"group", "resource"},
	)

	watchCacheResourceVersion = compbasemetrics.NewGaugeVec(
		&compbasemetrics.GaugeOpts{
			Namespace:      namespace,
			Subsystem:      subsystem,
			Name:           "resource_version",
			Help:           "Current resource version of watch cache broken by resource type. This is truncated to the 15 least significant digits.",
			StabilityLevel: compbasemetrics.ALPHA,
		},
		[]string{"group", "resource"},
	)

	watchCacheCapacityIncreaseTotal = compbasemetrics.NewCounterVec(
		&compbasemetrics.CounterOpts{
			Subsystem:      subsystem,
			Name:           "capacity_increase_total",
			Help:           "Total number of watch cache capacity increase events broken by resource type.",
			StabilityLevel: compbasemetrics.ALPHA,
		},
		[]string{"group", "resource"},
	)

	watchCacheCapacityDecreaseTotal = compbasemetrics.NewCounterVec(
		&compbasemetrics.CounterOpts{
			Subsystem:      subsystem,
			Name:           "capacity_decrease_total",
			Help:           "Total number of watch cache capacity decrease events broken by resource type.",
			StabilityLevel: compbasemetrics.ALPHA,
		},
		[]string{"group", "resource"},
	)

	WatchCacheCapacity = compbasemetrics.NewGaugeVec(
		&compbasemetrics.GaugeOpts{
			Subsystem:      subsystem,
			Name:           "capacity",
			Help:           "Total capacity of watch cache broken by resource type.",
			StabilityLevel: compbasemetrics.ALPHA,
		},
		[]string{"group", "resource"},
	)

	WatchCacheInitializations = compbasemetrics.NewCounterVec(
		&compbasemetrics.CounterOpts{
			Namespace:      namespace,
			Subsystem:      subsystem,
			Name:           "initializations_total",
			Help:           "Counter of watch cache initializations broken by resource type.",
			StabilityLevel: compbasemetrics.ALPHA,
		},
		[]string{"group", "resource"},
	)

	WatchCacheInitializationErrors = compbasemetrics.NewCounterVec(
		&compbasemetrics.CounterOpts{
			Namespace:      namespace,
			Subsystem:      subsystem,
			Name:           "initialization_errors_total",
			Help:           "Counter of watch cache initialization errors broken by resource type.",
			StabilityLevel: compbasemetrics.ALPHA,
		},
		[]string{"group", "resource"},
	)

	WatchCacheInitializationDuration = compbasemetrics.NewHistogramVec(
		&compbasemetrics.HistogramOpts{
			Namespace:      namespace,
			Subsystem:      subsystem,
			Name:           "initialization_duration_seconds",
			Help:           "Histogram of watch cache initialization duration in seconds, broken by resource type.",
			StabilityLevel: compbasemetrics.ALPHA,
			Buckets:        []float64{0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 180, 600},
		},
		[]string{"group", "resource"},
	)

	WatchCacheReadWait = compbasemetrics.NewHistogramVec(
		&compbasemetrics.HistogramOpts{
			Namespace:      namespace,
			Subsystem:      subsystem,
			Name:           "read_wait_seconds",
			Help:           "Histogram of time spent waiting for a watch cache to become fresh.",
			StabilityLevel: compbasemetrics.ALPHA,
			Buckets:        []float64{0.005, 0.025, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.25, 1.5, 2, 3},
		}, []string{"group", "resource"})

	// TerminatedWatchersDetailed carries the labels that the V(1) "Forcing ...
	// watcher close due to unresponsiveness" log line already prints. Every
	// conclusion about which watchers die and why has so far come from grepping
	// gigabytes of apiserver logs. This is deliberately a second metric rather
	// than labels on TerminatedWatchersCounter, so the existing series keeps its
	// meaning.
	TerminatedWatchersDetailed = compbasemetrics.NewCounterVec(
		&compbasemetrics.CounterOpts{
			Namespace:      namespace,
			Name:           "terminated_watchers_detailed_total",
			Help:           "Counter of watchers closed due to unresponsiveness, broken by which buffer was full, the scope of the watch, and the size its input channel was fixed at when it was created.",
			StabilityLevel: compbasemetrics.ALPHA,
		},
		[]string{"group", "resource", "reason", "scope", "chan_size"},
	)

	// DispatchGraceBudget records what the shared time budget actually granted.
	// A dispatch that finds blocked watchers gets one timer for all of them, so
	// this is the total grace the whole group had to share.
	DispatchGraceBudget = compbasemetrics.NewHistogramVec(
		&compbasemetrics.HistogramOpts{
			Namespace:      namespace,
			Subsystem:      "watch_events",
			Name:           "dispatch_grace_budget_seconds",
			Help:           "Histogram of the timeout granted by the dispatch time budget for a single dispatch that had blocked watchers.",
			StabilityLevel: compbasemetrics.ALPHA,
			Buckets:        []float64{0, 0.0001, 0.001, 0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 1},
		},
		[]string{"group", "resource"},
	)

	// TerminatedWatchersPerDispatch and WatchersTerminatedWithoutGrace together
	// separate "one slow watcher took the rest down with it" from "many slow
	// watchers". A single dispatch shares one timer across every blocked
	// watcher, and once it fires each remaining watcher is closed with no grace
	// at all, so a single dispatch can produce hundreds of kills.
	TerminatedWatchersPerDispatch = compbasemetrics.NewHistogramVec(
		&compbasemetrics.HistogramOpts{
			Namespace:      namespace,
			Subsystem:      "watch_events",
			Name:           "terminated_watchers_per_dispatch",
			Help:           "Histogram of how many watchers a single dispatch terminated, observed only for dispatches that terminated at least one.",
			StabilityLevel: compbasemetrics.ALPHA,
			Buckets:        []float64{1, 2, 5, 10, 25, 50, 100, 250, 500, 1000, 2500},
		},
		[]string{"group", "resource"},
	)

	WatchersTerminatedWithoutGrace = compbasemetrics.NewCounterVec(
		&compbasemetrics.CounterOpts{
			Namespace:      namespace,
			Subsystem:      "watch_events",
			Name:           "terminated_watchers_without_grace_total",
			Help:           "Counter of watchers terminated after the shared dispatch timer had already fired, so they were given no grace period of their own.",
			StabilityLevel: compbasemetrics.ALPHA,
		},
		[]string{"group", "resource"},
	)

	// The three sampled signals below are observed on one in every
	// watchBacklogSampleRate handoffs. At hundreds of millions of deliveries per
	// run an unsampled histogram would cost more than it is worth, and the
	// question these answer is distributional.
	WatchInputBacklog = compbasemetrics.NewHistogramVec(
		&compbasemetrics.HistogramOpts{
			Namespace:      namespace,
			Subsystem:      subsystem,
			Name:           "input_backlog_events",
			Help:           "Sampled depth of a watcher's input channel at the moment an event is handed to the serve loop.",
			StabilityLevel: compbasemetrics.ALPHA,
			Buckets:        []float64{0, 1, 2, 5, 10, 25, 50, 100, 250, 500, 1000},
		},
		[]string{"group", "resource"},
	)

	WatchResultBacklog = compbasemetrics.NewHistogramVec(
		&compbasemetrics.HistogramOpts{
			Namespace:      namespace,
			Subsystem:      subsystem,
			Name:           "result_backlog_events",
			Help:           "Sampled depth of a watcher's result channel at the moment an event is handed to the serve loop.",
			StabilityLevel: compbasemetrics.ALPHA,
			Buckets:        []float64{0, 1, 2, 5, 10, 25, 50, 100, 250, 500, 1000},
		},
		[]string{"group", "resource"},
	)

	// WatchEventConversionDuration covers convertToWatchEvent, which filters and
	// deep-copies. It is the one place processInterval can be parked while its
	// result channel sits empty.
	WatchEventConversionDuration = compbasemetrics.NewHistogramVec(
		&compbasemetrics.HistogramOpts{
			Namespace:      namespace,
			Subsystem:      subsystem,
			Name:           "event_conversion_duration_seconds",
			Help:           "Sampled time spent converting a watch cache event into a watch event, including filtering and deep copy.",
			StabilityLevel: compbasemetrics.ALPHA,
			Buckets:        []float64{0.00001, 0.00005, 0.0001, 0.00025, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 1},
		},
		[]string{"group", "resource"},
	)

	// SerializationCacheTotal validates the fanout-amortization model directly:
	// a cachingObject serializes once and every further watcher receiving the
	// same event reuses it, so the per-delivery serialization cost is inversely
	// proportional to the hit rate.
	SerializationCacheTotal = compbasemetrics.NewCounterVec(
		&compbasemetrics.CounterOpts{
			Namespace:      namespace,
			Subsystem:      subsystem,
			Name:           "serialization_cache_total",
			Help:           "Counter of watch event serialization attempts against the per-event serialization cache, broken by whether the serialization was reused.",
			StabilityLevel: compbasemetrics.ALPHA,
		},
		[]string{"group", "resource", "result"},
	)

	// WatchersBlockedOnResult complements the StageCacheToWatcher histogram,
	// which can only report a wait once it has ended. A watcher wedged
	// indefinitely is visible here and in no histogram.
	WatchersBlockedOnResult = compbasemetrics.NewGaugeVec(
		&compbasemetrics.GaugeOpts{
			Namespace:      namespace,
			Subsystem:      subsystem,
			Name:           "watchers_blocked_on_result",
			Help:           "Number of watchers currently blocked handing an event to the watch serve loop, i.e. whose serve loop is not consuming right now.",
			StabilityLevel: compbasemetrics.ALPHA,
		},
		[]string{"group", "resource"},
	)

	ConsistentReadTotal = compbasemetrics.NewCounterVec(
		&compbasemetrics.CounterOpts{
			Namespace:      namespace,
			Subsystem:      subsystem,
			Name:           "consistent_read_total",
			Help:           "Counter for consistent reads from cache.",
			StabilityLevel: compbasemetrics.ALPHA,
		}, []string{"group", "resource", "success", "fallback"})

	StorageConsistencyCheckTotal = compbasemetrics.NewCounterVec(
		&compbasemetrics.CounterOpts{
			Namespace:      namespace,
			Name:           "storage_consistency_checks_total",
			Help:           "Counter for status of consistency checks between etcd and watch cache",
			StabilityLevel: compbasemetrics.ALPHA,
		}, []string{"group", "resource", "status"})

	WatchShardsTotal = compbasemetrics.NewGaugeVec(
		&compbasemetrics.GaugeOpts{
			Namespace:      namespace,
			Name:           "watch_shards_total",
			Help:           "Number of active sharded watch connections broken by resource type.",
			StabilityLevel: compbasemetrics.ALPHA,
		},
		[]string{"group", "resource"},
	)

	WatchFilteredEventsTotal = compbasemetrics.NewCounterVec(
		&compbasemetrics.CounterOpts{
			Namespace:      namespace,
			Name:           "watch_filtered_events_total",
			Help:           "Counter of events filtered out by shard selector during watch dispatch, broken by resource type.",
			StabilityLevel: compbasemetrics.ALPHA,
		},
		[]string{"group", "resource"},
	)

	DispatchStageDuration = compbasemetrics.NewHistogramVec(
		&compbasemetrics.HistogramOpts{
			Namespace:      namespace,
			Subsystem:      "watch_events",
			Name:           "dispatch_duration_seconds",
			Help:           "Histogram of watch event dispatch latency broken by resource type and pipeline stage. The 'total' stage is the end-to-end latency of a delivered event.",
			StabilityLevel: compbasemetrics.ALPHA,
			Buckets:        []float64{0.0001, 0.0005, 0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5},
		}, []string{"group", "resource", "stage"})
)

var registerMetrics sync.Once

// Register all metrics.
func Register() {
	// Register the metrics.
	registerMetrics.Do(func() {
		legacyregistry.MustRegister(listCacheCount)
		legacyregistry.MustRegister(listCacheNumFetched)
		legacyregistry.MustRegister(listCacheNumReturned)
		legacyregistry.MustRegister(InitCounter)
		legacyregistry.MustRegister(EventsReceivedCounter)
		legacyregistry.MustRegister(EventsCounter)
		legacyregistry.MustRegister(TerminatedWatchersCounter)
		legacyregistry.MustRegister(watchCacheResourceVersion)
		legacyregistry.MustRegister(watchCacheCapacityIncreaseTotal)
		legacyregistry.MustRegister(watchCacheCapacityDecreaseTotal)
		legacyregistry.MustRegister(WatchCacheCapacity)
		legacyregistry.MustRegister(WatchCacheInitializations)
		legacyregistry.MustRegister(WatchCacheInitializationErrors)
		legacyregistry.MustRegister(WatchCacheInitializationDuration)
		legacyregistry.MustRegister(WatchCacheReadWait)
		legacyregistry.MustRegister(WatchersBlockedOnResult)
		legacyregistry.MustRegister(TerminatedWatchersDetailed)
		legacyregistry.MustRegister(DispatchGraceBudget)
		legacyregistry.MustRegister(TerminatedWatchersPerDispatch)
		legacyregistry.MustRegister(WatchersTerminatedWithoutGrace)
		legacyregistry.MustRegister(WatchInputBacklog)
		legacyregistry.MustRegister(WatchResultBacklog)
		legacyregistry.MustRegister(WatchEventConversionDuration)
		legacyregistry.MustRegister(SerializationCacheTotal)
		legacyregistry.MustRegister(ConsistentReadTotal)
		legacyregistry.MustRegister(StorageConsistencyCheckTotal)
		if utilfeature.DefaultFeatureGate.Enabled(features.ShardedListAndWatch) {
			legacyregistry.MustRegister(WatchShardsTotal)
			legacyregistry.MustRegister(WatchFilteredEventsTotal)
		}
		legacyregistry.MustRegister(DispatchStageDuration)
	})
}

// SerializationCacheObservers holds the pre-resolved hit and miss counters for
// one resource. CacheEncode runs once per delivery, hundreds of millions of
// times per scale run, so the label lookup is done once per cacher instead.
type SerializationCacheObservers struct {
	hit  compbasemetrics.CounterMetric
	miss compbasemetrics.CounterMetric
}

// NewSerializationCacheObservers pre-resolves the counters for a resource.
func NewSerializationCacheObservers(groupResource schema.GroupResource) *SerializationCacheObservers {
	return &SerializationCacheObservers{
		hit:  SerializationCacheTotal.WithLabelValues(groupResource.Group, groupResource.Resource, "hit"),
		miss: SerializationCacheTotal.WithLabelValues(groupResource.Group, groupResource.Resource, "miss"),
	}
}

// RecordHit notes a delivery that reused an existing serialization. It is
// nil-safe so that callers constructing a cachingObject outside a cacher do not
// have to supply observers.
func (o *SerializationCacheObservers) RecordHit() {
	if o != nil {
		o.hit.Inc()
	}
}

// RecordMiss notes a delivery that had to perform the serialization itself.
func (o *SerializationCacheObservers) RecordMiss() {
	if o != nil {
		o.miss.Inc()
	}
}

// TerminationReason classifies which buffer was backed up when a watcher was
// closed. result_full means the serve loop was not consuming; result_empty
// means it was, and the watcher fell behind before the events reached it.
func TerminationReason(inputLen, resultLen, resultCap int) string {
	switch {
	case resultCap > 0 && resultLen >= resultCap:
		return "result_full"
	case resultLen == 0:
		return "result_empty"
	default:
		return "result_partial"
	}
}

// ChanSizeBucket buckets the watcher channel size, which is fixed when the
// watcher is created from the then-current watch cache capacity and never grows
// afterwards.
func ChanSizeBucket(chanSize int) string {
	switch {
	case chanSize <= 10:
		return "10"
	case chanSize <= 50:
		return "11-50"
	case chanSize <= 200:
		return "51-200"
	default:
		return "201-1000"
	}
}

// RecordListCacheMetrics notes various metrics of the cost to serve a LIST request
func RecordListCacheMetrics(groupResource schema.GroupResource, indexName string, numFetched, numReturned int) {
	listCacheCount.WithLabelValues(groupResource.Group, groupResource.Resource, indexName).Inc()
	listCacheNumFetched.WithLabelValues(groupResource.Group, groupResource.Resource, indexName).Add(float64(numFetched))
	listCacheNumReturned.WithLabelValues(groupResource.Group, groupResource.Resource).Add(float64(numReturned))
	storagemetrics.RecordStorageListMetrics(groupResource, storagemetrics.StorageBackendWatchCache, indexName, numFetched, 0, numReturned)
}

// RecordResourceVersion sets the current resource version for a given resource type.
// The resource version is truncated to the 15 least significant digits to prevent
// the metric from growing indefinitely and losing precision when it exceeds 2^53-1.
func RecordResourceVersion(groupResource schema.GroupResource, resourceVersion uint64) {
	watchCacheResourceVersion.WithLabelValues(groupResource.Group, groupResource.Resource).Set(float64(resourceVersion % 1000000000000000))
}

// RecordShardedWatchStarted increments the active sharded watch gauge for the given resource.
func RecordShardedWatchStarted(groupResource schema.GroupResource) {
	WatchShardsTotal.WithLabelValues(groupResource.Group, groupResource.Resource).Inc()
}

// RecordShardedWatchStopped decrements the active sharded watch gauge for the given resource.
func RecordShardedWatchStopped(groupResource schema.GroupResource) {
	WatchShardsTotal.WithLabelValues(groupResource.Group, groupResource.Resource).Dec()
}

// RecordWatchFilteredEvent increments the counter for events filtered by shard selector.
func RecordWatchFilteredEvent(groupResource schema.GroupResource) {
	WatchFilteredEventsTotal.WithLabelValues(groupResource.Group, groupResource.Resource).Inc()
}

// RecordsWatchCacheCapacityChange records watchCache capacity resize(increase or decrease) operations.
func RecordsWatchCacheCapacityChange(groupResource schema.GroupResource, old, new int) {
	WatchCacheCapacity.WithLabelValues(groupResource.Group, groupResource.Resource).Set(float64(new))
	if old < new {
		watchCacheCapacityIncreaseTotal.WithLabelValues(groupResource.Group, groupResource.Resource).Inc()
		return
	}
	watchCacheCapacityDecreaseTotal.WithLabelValues(groupResource.Group, groupResource.Resource).Inc()
}

// WatcherMetricsObservers holds pre-resolved (group, resource) observers for
// every dispatch stage, so the hot path never touches the label map.
type WatcherMetricsObservers struct {
	stageDurations [numDispatchStages]compbasemetrics.ObserverMetric
}

// NewWatcherMetricsObservers creates a pre-resolved metrics observer for watch connections.
func NewWatcherMetricsObservers(groupResource schema.GroupResource) *WatcherMetricsObservers {
	o := &WatcherMetricsObservers{}
	for s := range numDispatchStages {
		o.stageDurations[s] = DispatchStageDuration.WithLabelValues(groupResource.Group, groupResource.Resource, dispatchStageName[s])
	}
	return o
}

// ObserveStage records the duration spent in the given dispatch stage.
func (d *WatcherMetricsObservers) ObserveStage(stage DispatchStage, duration time.Duration) {
	if stage < 0 || stage >= numDispatchStages {
		return
	}
	observe(d.stageDurations[stage], duration)
}

func observe(m compbasemetrics.ObserverMetric, duration time.Duration) {
	if duration < 0 {
		duration = 0
	}
	m.Observe(duration.Seconds())
}

type noopObserver struct{}

func (noopObserver) Observe(float64) {}

var noopObs noopObserver

// NewNoopWatcherMetricsObservers returns a metrics observers struct that does nothing.
func NewNoopWatcherMetricsObservers() *WatcherMetricsObservers {
	o := &WatcherMetricsObservers{}
	for s := range o.stageDurations {
		o.stageDurations[s] = noopObs
	}
	return o
}
