/*
Copyright 2015 The Kubernetes Authors.

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
	"context"
	"fmt"
	"sync"
	"time"

	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apiserver/pkg/features"
	"k8s.io/apiserver/pkg/storage"
	storagemetrics "k8s.io/apiserver/pkg/storage/metrics"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	compbasemetrics "k8s.io/component-base/metrics"
	"k8s.io/component-base/metrics/legacyregistry"
	"k8s.io/klog/v2"
)

/*
 * By default, all the following metrics are defined as falling under
 * ALPHA stability level https://github.com/kubernetes/enhancements/blob/master/keps/sig-instrumentation/1209-metrics-stability/kubernetes-control-plane-metrics-stability.md#stability-classes)
 *
 * Promoting the stability level of the metric is a responsibility of the component owner, since it
 * involves explicitly acknowledging support for the metric across multiple releases, in accordance with
 * the metric stability policy.
 */
var (
	etcdRequestLatency = compbasemetrics.NewHistogramVec(
		&compbasemetrics.HistogramOpts{
			Name: "etcd_request_duration_seconds",
			Help: "Etcd request latency in seconds for each operation and object type.",
			// Etcd request latency in seconds for each operation and object type.
			// This metric is used for verifying etcd api call latencies SLO
			// keep consistent with apiserver metric 'requestLatencies' in
			// staging/src/k8s.io/apiserver/pkg/endpoints/metrics/metrics.go
			Buckets: []float64{0.005, 0.025, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.25, 1.5, 2, 3,
				4, 5, 6, 8, 10, 15, 20, 30, 45, 60},
			StabilityLevel: compbasemetrics.ALPHA,
		},
		[]string{"operation", "group", "resource"},
	)
	etcdRequestCounts = compbasemetrics.NewCounterVec(
		&compbasemetrics.CounterOpts{
			Name:           "etcd_requests_total",
			Help:           "Etcd request counts for each operation and object type.",
			StabilityLevel: compbasemetrics.ALPHA,
		},
		[]string{"operation", "group", "resource"},
	)
	etcdRequestErrorCounts = compbasemetrics.NewCounterVec(
		&compbasemetrics.CounterOpts{
			Name:           "etcd_request_errors_total",
			Help:           "Etcd failed request counts for each operation and object type.",
			StabilityLevel: compbasemetrics.ALPHA,
		},
		[]string{"operation", "group", "resource"},
	)
	objectCounts = compbasemetrics.NewGaugeVec(
		&compbasemetrics.GaugeOpts{
			Name:              "apiserver_storage_objects",
			Help:              "[DEPRECATED, consider using apiserver_resource_objects instead] Number of stored objects at the time of last check split by kind. In case of a fetching error, the value will be -1.",
			StabilityLevel:    compbasemetrics.STABLE,
			DeprecatedVersion: "1.34.0",
		},
		[]string{"resource"},
	)
	resourceSizeEstimate = compbasemetrics.NewGaugeVec(
		&compbasemetrics.GaugeOpts{
			Name:           "apiserver_resource_size_estimate_bytes",
			Help:           "Estimated size of stored objects in database. Estimate is based on sum of last observed sizes of serialized objects. In case of a fetching error, the value will be -1.",
			StabilityLevel: compbasemetrics.ALPHA,
		},
		[]string{"group", "resource"},
	)
	newObjectCounts = compbasemetrics.NewGaugeVec(
		&compbasemetrics.GaugeOpts{
			Name:           "apiserver_resource_objects",
			Help:           "Number of stored objects at the time of last check split by kind. In case of a fetching error, the value will be -1.",
			StabilityLevel: compbasemetrics.ALPHA,
		},
		[]string{"group", "resource"},
	)
	storageSizeDescription   = compbasemetrics.NewDesc("apiserver_storage_size_bytes", "Size of the storage database file physically allocated in bytes.", []string{"storage_cluster_id"}, nil, compbasemetrics.STABLE, "")
	storageMonitor           = &monitorCollector{monitorGetter: func() ([]Monitor, error) { return nil, nil }}
	etcdEventsReceivedCounts = compbasemetrics.NewCounterVec(
		&compbasemetrics.CounterOpts{
			Subsystem:      "apiserver",
			Name:           "storage_events_received_total",
			Help:           "Number of etcd events received split by kind.",
			StabilityLevel: compbasemetrics.BETA,
		},
		[]string{"group", "resource"},
	)
	etcdBookmarkCounts = compbasemetrics.NewGaugeVec(
		&compbasemetrics.GaugeOpts{
			Name:              "etcd_bookmark_counts",
			Help:              "Number of etcd bookmarks (progress notify events) split by kind.",
			StabilityLevel:    compbasemetrics.ALPHA,
			DeprecatedVersion: "1.36.0",
		},
		[]string{"group", "resource"},
	)
	etcdBookmarkTotal = compbasemetrics.NewCounterVec(
		&compbasemetrics.CounterOpts{
			Name:           "etcd_bookmark_total",
			Help:           "Number of etcd bookmarks (progress notify events) split by kind.",
			StabilityLevel: compbasemetrics.ALPHA,
		},
		[]string{"group", "resource"},
	)
	etcdLeaseObjectCounts = compbasemetrics.NewHistogramVec(
		&compbasemetrics.HistogramOpts{
			Name:           "etcd_lease_object_counts",
			Help:           "Number of objects attached to a single etcd lease.",
			Buckets:        []float64{10, 50, 100, 500, 1000, 2500, 5000},
			StabilityLevel: compbasemetrics.ALPHA,
		},
		[]string{},
	)
	decodeErrorCounts = compbasemetrics.NewCounterVec(
		&compbasemetrics.CounterOpts{
			Namespace:      "apiserver",
			Name:           "storage_decode_errors_total",
			Help:           "Number of stored object decode errors split by object type",
			StabilityLevel: compbasemetrics.ALPHA,
		},
		[]string{"group", "resource"},
	)
	listLatency = compbasemetrics.NewHistogramVec(
		&compbasemetrics.HistogramOpts{
			Namespace: "apiserver",
			Name:      "storage_list_duration_seconds",
			Help:      "Latency of the storage layer GetList call in seconds, including object decode, split by whether etcd RangeStream was used.",
			Buckets: []float64{0.005, 0.025, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.25, 1.5, 2, 3,
				4, 5, 6, 8, 10, 15, 20, 30, 45, 60},
			StabilityLevel: compbasemetrics.ALPHA,
		},
		[]string{"streamed", "group", "resource"},
	)
	locklessTraceWriteDuration = compbasemetrics.NewCounter(
		&compbasemetrics.CounterOpts{
			Namespace:      "apiserver",
			Subsystem:      "storage",
			Name:           "lockless_trace_write_duration_seconds_total",
			Help:           "Total time spent writing lockless traces to klog in seconds.",
			StabilityLevel: compbasemetrics.ALPHA,
		},
	)
	locklessTraceWrites = compbasemetrics.NewCounter(
		&compbasemetrics.CounterOpts{
			Namespace:      "apiserver",
			Subsystem:      "storage",
			Name:           "lockless_trace_writes_total",
			Help:           "Total number of lockless trace writes.",
			StabilityLevel: compbasemetrics.ALPHA,
		},
	)
)

var registerMetrics sync.Once

// Register all metrics.
func Register() {
	// Register the metrics.
	registerMetrics.Do(func() {
		legacyregistry.MustRegister(etcdRequestLatency)
		legacyregistry.MustRegister(etcdRequestCounts)
		legacyregistry.MustRegister(etcdRequestErrorCounts)
		legacyregistry.MustRegister(objectCounts)
		legacyregistry.MustRegister(resourceSizeEstimate)
		legacyregistry.MustRegister(newObjectCounts)
		legacyregistry.CustomMustRegister(storageMonitor)
		legacyregistry.MustRegister(etcdEventsReceivedCounts)
		legacyregistry.MustRegister(etcdBookmarkCounts)
		legacyregistry.MustRegister(etcdBookmarkTotal)
		legacyregistry.MustRegister(etcdLeaseObjectCounts)
		legacyregistry.MustRegister(decodeErrorCounts)
		legacyregistry.MustRegister(listLatency)
		legacyregistry.MustRegister(locklessTraceWriteDuration)
		legacyregistry.MustRegister(locklessTraceWrites)
	})
}

// UpdateStoreStats sets the stats metrics.
func UpdateStoreStats(groupResource schema.GroupResource, stats storage.Stats, err error) {
	if err != nil {
		objectCounts.WithLabelValues(groupResource.String()).Set(-1)
		newObjectCounts.WithLabelValues(groupResource.Group, groupResource.Resource).Set(-1)
		if utilfeature.DefaultFeatureGate.Enabled(features.SizeBasedListCostEstimate) {
			resourceSizeEstimate.WithLabelValues(groupResource.Group, groupResource.Resource).Set(-1)
		}
		return
	}
	objectCounts.WithLabelValues(groupResource.String()).Set(float64(stats.ObjectCount))
	newObjectCounts.WithLabelValues(groupResource.Group, groupResource.Resource).Set(float64(stats.ObjectCount))
	if utilfeature.DefaultFeatureGate.Enabled(features.SizeBasedListCostEstimate) {
		if stats.ObjectCount > 0 && stats.EstimatedAverageObjectSizeBytes == 0 {
			resourceSizeEstimate.WithLabelValues(groupResource.Group, groupResource.Resource).Set(-1)
		} else {
			resourceSizeEstimate.WithLabelValues(groupResource.Group, groupResource.Resource).Set(float64(stats.EstimatedAverageObjectSizeBytes * stats.ObjectCount))
		}
	}
}

// DeleteStoreStats delete the stats metrics.
func DeleteStoreStats(groupResource schema.GroupResource) {
	objectCounts.Delete(map[string]string{"resource": groupResource.String()})
	newObjectCounts.Delete(map[string]string{"group": groupResource.Group, "resource": groupResource.Resource})
	if utilfeature.DefaultFeatureGate.Enabled(features.SizeBasedListCostEstimate) {
		resourceSizeEstimate.DeleteLabelValues(groupResource.Group, groupResource.Resource)
	}
}

// Reset resets the etcd_request_duration_seconds metric.
func Reset() {
	etcdRequestLatency.Reset()
}

// sinceInSeconds gets the time since the specified start in seconds.
//
// This is a variable to facilitate testing.
var sinceInSeconds = func(start time.Time) float64 {
	return time.Since(start).Seconds()
}

// SetStorageMonitorGetter sets monitor getter to allow monitoring etcd stats.
func SetStorageMonitorGetter(getter func() ([]Monitor, error)) {
	storageMonitor.setGetter(getter)
}

// UpdateLeaseObjectCount sets the etcd_lease_object_counts metric.
func UpdateLeaseObjectCount(count int64) {
	// Currently we only store one previous lease, since all the events have the same ttl.
	// See pkg/storage/etcd3/lease_manager.go
	etcdLeaseObjectCounts.WithLabelValues().Observe(float64(count))
}

type Monitor interface {
	Monitor(ctx context.Context) (StorageMetrics, error)
	Close() error
}

type StorageMetrics struct {
	Size int64
}

type monitorCollector struct {
	compbasemetrics.BaseStableCollector

	mutex         sync.Mutex
	monitorGetter func() ([]Monitor, error)
}

func (m *monitorCollector) setGetter(monitorGetter func() ([]Monitor, error)) {
	m.mutex.Lock()
	defer m.mutex.Unlock()
	m.monitorGetter = monitorGetter
}

func (m *monitorCollector) getGetter() func() ([]Monitor, error) {
	m.mutex.Lock()
	defer m.mutex.Unlock()
	return m.monitorGetter
}

// DescribeWithStability implements compbasemetrics.StableColletor
func (c *monitorCollector) DescribeWithStability(ch chan<- *compbasemetrics.Desc) {
	ch <- storageSizeDescription
}

// CollectWithStability implements compbasemetrics.StableColletor
func (c *monitorCollector) CollectWithStability(ch chan<- compbasemetrics.Metric) {
	monitors, err := c.getGetter()()
	if err != nil {
		return
	}

	for i, m := range monitors {
		storageClusterID := fmt.Sprintf("etcd-%d", i)

		klog.V(4).InfoS("Start collecting storage metrics", "storage_cluster_id", storageClusterID)
		ctx, cancel := context.WithTimeout(context.Background(), time.Second)
		metrics, err := m.Monitor(ctx)
		cancel()
		if err != nil {
			klog.InfoS("Failed to get storage metrics", "storage_cluster_id", storageClusterID, "err", err)
			continue
		}

		metric, err := compbasemetrics.NewConstMetric(storageSizeDescription, compbasemetrics.GaugeValue, float64(metrics.Size), storageClusterID)
		if err != nil {
			klog.ErrorS(err, "Failed to create metric", "storage_cluster_id", storageClusterID)
		}
		ch <- metric
	}
}

// RequestLatencyTracker is a helper to record etcd call metrics
// using pre-materialized Prometheus counters and histograms to avoid WithLabelValues dynamic lookup locks.
type RequestLatencyTracker struct {
	etcdRequestLatency     compbasemetrics.ObserverMetric
	etcdRequestCounts      compbasemetrics.CounterMetric
	etcdRequestErrorCounts compbasemetrics.CounterMetric
}

// NewRequestLatencyTracker returns a pre-materialized tracker for a specific verb and resource.
func NewRequestLatencyTracker(verb string, groupResource schema.GroupResource) *RequestLatencyTracker {
	return &RequestLatencyTracker{
		etcdRequestLatency:     etcdRequestLatency.WithLabelValues(verb, groupResource.Group, groupResource.Resource),
		etcdRequestCounts:      etcdRequestCounts.WithLabelValues(verb, groupResource.Group, groupResource.Resource),
		etcdRequestErrorCounts: etcdRequestErrorCounts.WithLabelValues(verb, groupResource.Group, groupResource.Resource),
	}
}

// Record observes request duration and updates metrics.
func (t *RequestLatencyTracker) Record(err error, startTime time.Time) {
	t.etcdRequestLatency.Observe(sinceInSeconds(startTime))
	t.etcdRequestCounts.Inc()
	if err != nil {
		t.etcdRequestErrorCounts.Inc()
	}
}

// RequestLatencyTrackers holds pre-materialized trackers for all common storage operations.
type RequestLatencyTrackers struct {
	Get                       *RequestLatencyTracker
	Create                    *RequestLatencyTracker
	Delete                    *RequestLatencyTracker
	Update                    *RequestLatencyTracker
	List                      *RequestLatencyTracker
	ListWithCount             *RequestLatencyTracker
	ListOnlyKeys              *RequestLatencyTracker
	GetCurrentResourceVersion *RequestLatencyTracker
	ListStream                *RequestLatencyTracker

	StorageList *ListMetricsTracker
	DecodeError *DecodeErrorTracker
}

// NewRequestLatencyTrackers returns RequestLatencyTrackers initialized with pre-materialized metrics.
func NewRequestLatencyTrackers(groupResource schema.GroupResource) RequestLatencyTrackers {
	return RequestLatencyTrackers{
		Get:                       NewRequestLatencyTracker("get", groupResource),
		Create:                    NewRequestLatencyTracker("create", groupResource),
		Delete:                    NewRequestLatencyTracker("delete", groupResource),
		Update:                    NewRequestLatencyTracker("update", groupResource),
		List:                      NewRequestLatencyTracker("list", groupResource),
		ListWithCount:             NewRequestLatencyTracker("listWithCount", groupResource),
		ListOnlyKeys:              NewRequestLatencyTracker("listOnlyKeys", groupResource),
		GetCurrentResourceVersion: NewRequestLatencyTracker("getCurrentResourceVersion", groupResource),
		ListStream:                NewRequestLatencyTracker("listStream", groupResource),

		StorageList: NewListMetricsTracker(groupResource),
		DecodeError: NewDecodeErrorTracker(groupResource),
	}
}

// ListMetricsTracker is a helper to record LIST operation performance metrics.
type ListMetricsTracker struct {
	storageTracker         *storagemetrics.ListMetricsTracker
	listLatencyStreamed    compbasemetrics.ObserverMetric
	listLatencyNotStreamed compbasemetrics.ObserverMetric
}

// NewListMetricsTracker returns a pre-materialized tracker for LIST performance metrics.
func NewListMetricsTracker(groupResource schema.GroupResource) *ListMetricsTracker {
	return &ListMetricsTracker{
		storageTracker:         storagemetrics.NewListMetricsTracker(groupResource, storagemetrics.StorageBackendEtcd, ""),
		listLatencyStreamed:    listLatency.WithLabelValues("true", groupResource.Group, groupResource.Resource),
		listLatencyNotStreamed: listLatency.WithLabelValues("false", groupResource.Group, groupResource.Resource),
	}
}

// Record updates the list performance metrics.
func (t *ListMetricsTracker) Record(streamed bool, numFetched, numEvald, numReturned int, startTime time.Time) {
	t.storageTracker.Record(numFetched, numEvald, numReturned)
	if streamed {
		t.listLatencyStreamed.Observe(sinceInSeconds(startTime))
	} else {
		t.listLatencyNotStreamed.Observe(sinceInSeconds(startTime))
	}
}

// DecodeErrorTracker is a helper to record decode errors.
type DecodeErrorTracker struct {
	decodeErrorCounts compbasemetrics.CounterMetric
}

// NewDecodeErrorTracker returns a pre-materialized tracker for decode errors.
func NewDecodeErrorTracker(groupResource schema.GroupResource) *DecodeErrorTracker {
	return &DecodeErrorTracker{
		decodeErrorCounts: decodeErrorCounts.WithLabelValues(groupResource.Group, groupResource.Resource),
	}
}

// Record increments the decode errors metric.
func (t *DecodeErrorTracker) Record() {
	t.decodeErrorCounts.Inc()
}

// LeaseMetricsTracker is a helper to record lease metrics using pre-materialized vectors.
type LeaseMetricsTracker struct {
	etcdLeaseObjectCounts compbasemetrics.ObserverMetric
}

// NewLeaseMetricsTracker returns a pre-materialized tracker for lease metrics.
func NewLeaseMetricsTracker() *LeaseMetricsTracker {
	return &LeaseMetricsTracker{
		etcdLeaseObjectCounts: etcdLeaseObjectCounts.WithLabelValues(),
	}
}

// Record updates the lease object count metric.
func (t *LeaseMetricsTracker) Record(count int64) {
	t.etcdLeaseObjectCounts.Observe(float64(count))
}

// WatcherMetricsTracker is a helper to record watcher performance metrics.
type WatcherMetricsTracker struct {
	Get                      *RequestLatencyTracker
	List                     *RequestLatencyTracker
	ListStream               *RequestLatencyTracker
	etcdEventsReceivedCounts compbasemetrics.CounterMetric
	etcdBookmarkCounts       compbasemetrics.GaugeMetric
	etcdBookmarkTotal        compbasemetrics.CounterMetric
}

// NewWatcherMetricsTracker returns a pre-materialized tracker for watcher metrics.
func NewWatcherMetricsTracker(groupResource schema.GroupResource) *WatcherMetricsTracker {
	return &WatcherMetricsTracker{
		Get:                      NewRequestLatencyTracker("get", groupResource),
		List:                     NewRequestLatencyTracker("list", groupResource),
		ListStream:               NewRequestLatencyTracker("listStream", groupResource),
		etcdEventsReceivedCounts: etcdEventsReceivedCounts.WithLabelValues(groupResource.Group, groupResource.Resource),
		etcdBookmarkCounts:       etcdBookmarkCounts.WithLabelValues(groupResource.Group, groupResource.Resource),
		etcdBookmarkTotal:        etcdBookmarkTotal.WithLabelValues(groupResource.Group, groupResource.Resource),
	}
}

// RecordEvent increments the events received count metric.
func (t *WatcherMetricsTracker) RecordEvent() {
	t.etcdEventsReceivedCounts.Inc()
}

// RecordBookmark increments the bookmark count metric.
func (t *WatcherMetricsTracker) RecordBookmark() {
	t.etcdBookmarkCounts.Inc()
	t.etcdBookmarkTotal.Inc()
}

// RecordLocklessTraceWriteDuration records the duration of writing lockless trace to klog.
func RecordLocklessTraceWriteDuration(duration time.Duration) {
	locklessTraceWriteDuration.Add(duration.Seconds())
	locklessTraceWrites.Inc()
}
