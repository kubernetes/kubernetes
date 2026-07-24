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
// The pipeline stages (propagation..handoff) are additive per delivery.
// StageTotal is a cumulative, whole-lifecycle observation
// folded into the same metric: StageTotal is the end-to-end latency of a
// successfully delivered event.
type DispatchStage int

const (
	// StagePropagation: etcd decode -> reflector handed event to the watch cache.
	StagePropagation DispatchStage = iota
	// StageCacheIngest: watch cache received -> event appended to ring buffer.
	StageCacheIngest
	// StageIncomingQueue: enqueued to the cacher incoming channel -> dispatched.
	StageIncomingQueue
	// StageFanout: dispatched -> enqueued on this watcher's input channel.
	StageFanout
	// StageWatcherQueue: enqueued on input channel -> dequeued by the watcher.
	StageWatcherQueue
	// StageEncode: dequeued -> outgoing watch.Event built (filter + convert).
	StageEncode
	// StageHandoff: watch.Event built -> written to the result channel.
	StageHandoff
	// StageTotal: end-to-end, etcd decode -> written to the result channel.
	StageTotal

	// The remaining stages are diagnostic and NOT part of the additive
	// partition above.
	//
	// StageWatcherQueueParked and StageWatcherQueueBacklog decompose
	// StageWatcherQueue (which is still recorded as before): exactly one of
	// the pair is recorded per delivery, so their sums equal watcher_queue.
	// A delivery is "parked" when the processing goroutine had to park in the
	// blocking receive waiting for the event (the interval is dominated by
	// goroutine wake/scheduling latency), and "backlog" when the event was
	// already waiting in the input channel (head-of-line drain backlog).
	StageWatcherQueueParked
	StageWatcherQueueBacklog
	// StageHandoffAborted: watch.Event built -> delivery aborted because the
	// watcher was done before the result channel accepted the event. Aborted
	// deliveries record no other stage.
	StageHandoffAborted

	numDispatchStages
)

var dispatchStageName = [numDispatchStages]string{
	StagePropagation:         "propagation",
	StageCacheIngest:         "cache_ingest",
	StageIncomingQueue:       "incoming_queue",
	StageFanout:              "fanout",
	StageWatcherQueue:        "watcher_queue",
	StageEncode:              "encode",
	StageHandoff:             "handoff",
	StageTotal:               "total",
	StageWatcherQueueParked:  "watcher_queue_parked",
	StageWatcherQueueBacklog: "watcher_queue_backlog",
	StageHandoffAborted:      "handoff_aborted",
}

// TerminationReason identifies why a watcher was terminated through the
// dispatch-blocked path. It is used as the "reason" label value on the
// terminatedWatchersDuration metric.
type TerminationReason int

const (
	// TerminationReasonBudgetExpired: the shared dispatch timeout budget
	// expired while blocked on this watcher's input channel.
	TerminationReasonBudgetExpired TerminationReason = iota
	// TerminationReasonCascade: the watcher was killed immediately without
	// waiting because the budget was already exhausted by another watcher
	// during the same dispatch.
	TerminationReasonCascade

	numTerminationReasons
)

var terminationReasonName = [numTerminationReasons]string{
	TerminationReasonBudgetExpired: "budget_expired",
	TerminationReasonCascade:       "cascade",
}

// Termination states for the "state" label: whether the watcher's result
// channel was full at the moment of termination. Full means the processing
// goroutine is alive but blocked on downstream handoff (client backpressure);
// not full means the goroutine never woke to drain the input channel
// (scheduling latency).
const (
	terminationStateResultFull = iota
	terminationStateResultFree
	numTerminationStates
)

var terminationStateName = [numTerminationStates]string{
	terminationStateResultFull: "result_full",
	terminationStateResultFree: "result_free",
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

	dispatchStageDuration = compbasemetrics.NewHistogramVec(
		&compbasemetrics.HistogramOpts{
			Namespace:      namespace,
			Subsystem:      "watch_events",
			Name:           "delivery_duration_seconds",
			Help:           "Histogram of watch event dispatch latency broken by resource type and pipeline stage. The additive stages (propagation, cache_ingest, incoming_queue, fanout, watcher_queue, encode, handoff) partition the delivery path; the 'total' stage is the end-to-end latency of a delivered event. The diagnostic stages are not part of the additive partition: 'watcher_queue_parked' and 'watcher_queue_backlog' split 'watcher_queue' into goroutine wake latency vs input-channel drain backlog and sum to it, and 'handoff_aborted' is the time an aborted delivery spent blocked on the result channel before the watcher was done.",
			StabilityLevel: compbasemetrics.ALPHA,
			Buckets:        []float64{0.0001, 0.0005, 0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5},
		}, []string{"group", "resource", "stage"})

	terminatedWatchersDuration = compbasemetrics.NewHistogramVec(
		&compbasemetrics.HistogramOpts{
			Namespace:      namespace,
			Name:           "terminated_watchers_duration_seconds",
			Help:           "Histogram of how long a watcher terminated for unresponsiveness had been stalled (time since it last dequeued from its input channel), broken by resource type, termination reason ('budget_expired': the shared dispatch timeout budget expired while blocked on this watcher; 'cascade': killed immediately after the budget was exhausted by another watcher), and result-channel state sampled at termination ('result_full': the delivery goroutine is alive but blocked on client handoff; 'result_free': the goroutine never woke to drain).",
			StabilityLevel: compbasemetrics.ALPHA,
			Buckets:        []float64{0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10},
		}, []string{"group", "resource", "reason", "state"})

	watcherInputDepth = compbasemetrics.NewHistogramVec(
		&compbasemetrics.HistogramOpts{
			Namespace:      namespace,
			Subsystem:      "watch",
			Name:           "watcher_input_depth",
			Help:           "Histogram of the per-watcher input channel depth observed at each successful event enqueue, broken by resource type.",
			StabilityLevel: compbasemetrics.ALPHA,
			Buckets:        []float64{0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024},
		}, []string{"group", "resource"})

	watcherInputFull = compbasemetrics.NewCounterVec(
		&compbasemetrics.CounterOpts{
			Namespace:      namespace,
			Subsystem:      "watch",
			Name:           "watcher_input_full_total",
			Help:           "Counter of failed non-blocking enqueues onto a watcher's full input channel, broken by resource type.",
			StabilityLevel: compbasemetrics.ALPHA,
		}, []string{"group", "resource"})
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
		legacyregistry.MustRegister(ConsistentReadTotal)
		legacyregistry.MustRegister(StorageConsistencyCheckTotal)
		if utilfeature.DefaultFeatureGate.Enabled(features.ShardedListAndWatch) {
			legacyregistry.MustRegister(WatchShardsTotal)
			legacyregistry.MustRegister(WatchFilteredEventsTotal)
		}
		legacyregistry.MustRegister(dispatchStageDuration)
		legacyregistry.MustRegister(terminatedWatchersDuration)
		legacyregistry.MustRegister(watcherInputDepth)
		legacyregistry.MustRegister(watcherInputFull)
		legacyregistry.CustomMustRegister(newSchedLatenciesCollector())
	})
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
	stageDurations    [numDispatchStages]compbasemetrics.ObserverMetric
	inputDepth        compbasemetrics.ObserverMetric
	inputFull         compbasemetrics.CounterMetric
	terminationStalls [numTerminationReasons][numTerminationStates]compbasemetrics.ObserverMetric
}

// NewWatcherMetricsObservers creates a pre-resolved metrics observer for watch connections.
func NewWatcherMetricsObservers(groupResource schema.GroupResource) *WatcherMetricsObservers {
	o := &WatcherMetricsObservers{}
	for s := DispatchStage(0); s < numDispatchStages; s++ {
		o.stageDurations[s] = dispatchStageDuration.WithLabelValues(groupResource.Group, groupResource.Resource, dispatchStageName[s])
	}
	o.inputDepth = watcherInputDepth.WithLabelValues(groupResource.Group, groupResource.Resource)
	o.inputFull = watcherInputFull.WithLabelValues(groupResource.Group, groupResource.Resource)
	for r := TerminationReason(0); r < numTerminationReasons; r++ {
		for s := 0; s < numTerminationStates; s++ {
			o.terminationStalls[r][s] = terminatedWatchersDuration.WithLabelValues(groupResource.Group, groupResource.Resource, terminationReasonName[r], terminationStateName[s])
		}
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

// ObserveInputDepth records the input channel depth seen at a successful enqueue.
func (d *WatcherMetricsObservers) ObserveInputDepth(depth int) {
	d.inputDepth.Observe(float64(depth))
}

// IncInputFull counts a failed non-blocking enqueue onto a full input channel.
func (d *WatcherMetricsObservers) IncInputFull() {
	d.inputFull.Inc()
}

// ObserveTerminationStall records how long a watcher terminated through the
// dispatch-blocked path had been stalled (time since its last input dequeue),
// with the result-channel state sampled at termination.
func (d *WatcherMetricsObservers) ObserveTerminationStall(reason TerminationReason, resultFull bool, duration time.Duration) {
	if reason < 0 || reason >= numTerminationReasons {
		return
	}
	state := terminationStateResultFree
	if resultFull {
		state = terminationStateResultFull
	}
	observe(d.terminationStalls[reason][state], duration)
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

type noopCounter struct{}

func (noopCounter) Inc()        {}
func (noopCounter) Add(float64) {}

// NewNoopWatcherMetricsObservers returns a metrics observers struct that does nothing.
func NewNoopWatcherMetricsObservers() *WatcherMetricsObservers {
	o := &WatcherMetricsObservers{}
	for s := range o.stageDurations {
		o.stageDurations[s] = noopObs
	}
	o.inputDepth = noopObs
	o.inputFull = noopCounter{}
	for r := range o.terminationStalls {
		for s := range o.terminationStalls[r] {
			o.terminationStalls[r][s] = noopObs
		}
	}
	return o
}
