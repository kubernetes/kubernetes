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
			Name:           "apiserver_storage_objects",
			Help:           "[DEPRECATED, consider using apiserver_resource_objects instead] Number of stored objects at the time of last check split by kind. In case of a fetching error, the value will be -1.",
			StabilityLevel: compbasemetrics.STABLE,
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
	dbTotalSize = compbasemetrics.NewGaugeVec(
		&compbasemetrics.GaugeOpts{
			Subsystem:         "apiserver",
			Name:              "storage_db_total_size_in_bytes",
			Help:              "Total size of the storage database file physically allocated in bytes.",
			StabilityLevel:    compbasemetrics.ALPHA,
			DeprecatedVersion: "1.28.0",
		},
		[]string{"endpoint"},
	)
	storageSizeDescription   = compbasemetrics.NewDesc("apiserver_storage_size_bytes", "Size of the storage database file physically allocated in bytes.", []string{"storage_cluster_id"}, nil, compbasemetrics.STABLE, "")
	storageMonitor           = &monitorCollector{monitorGetter: func() ([]Monitor, error) { return nil, nil }}
	etcdEventsReceivedCounts = compbasemetrics.NewCounterVec(
		&compbasemetrics.CounterOpts{
			Subsystem:      "apiserver",
			Name:           "storage_events_received_total",
			Help:           "Number of etcd events received split by kind.",
			StabilityLevel: compbasemetrics.ALPHA,
		},
		[]string{"group", "resource"},
	)
	etcdBookmarkCounts = compbasemetrics.NewGaugeVec(
		&compbasemetrics.GaugeOpts{
			Name:           "etcd_bookmark_counts",
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
	listStorageCount = compbasemetrics.NewCounterVec(
		&compbasemetrics.CounterOpts{
			Name:           "apiserver_storage_list_total",
			Help:           "Number of LIST requests served from storage",
			StabilityLevel: compbasemetrics.ALPHA,
		},
		[]string{"group", "resource"},
	)
	listStorageNumFetched = compbasemetrics.NewCounterVec(
		&compbasemetrics.CounterOpts{
			Name:           "apiserver_storage_list_fetched_objects_total",
			Help:           "Number of objects read from storage in the course of serving a LIST request",
			StabilityLevel: compbasemetrics.ALPHA,
		},
		[]string{"group", "resource"},
	)
	listStorageNumSelectorEvals = compbasemetrics.NewCounterVec(
		&compbasemetrics.CounterOpts{
			Name:           "apiserver_storage_list_evaluated_objects_total",
			Help:           "Number of objects tested in the course of serving a LIST request from storage",
			StabilityLevel: compbasemetrics.ALPHA,
		},
		[]string{"group", "resource"},
	)
	listStorageNumReturned = compbasemetrics.NewCounterVec(
		&compbasemetrics.CounterOpts{
			Name:           "apiserver_storage_list_returned_objects_total",
			Help:           "Number of objects returned for a LIST request from storage",
			StabilityLevel: compbasemetrics.ALPHA,
		},
		[]string{"group", "resource"},
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
		legacyregistry.MustRegister(dbTotalSize)
		legacyregistry.CustomMustRegister(storageMonitor)
		legacyregistry.MustRegister(etcdEventsReceivedCounts)
		legacyregistry.MustRegister(etcdBookmarkCounts)
		legacyregistry.MustRegister(etcdLeaseObjectCounts)
		legacyregistry.MustRegister(listStorageCount)
		legacyregistry.MustRegister(listStorageNumFetched)
		legacyregistry.MustRegister(listStorageNumSelectorEvals)
		legacyregistry.MustRegister(listStorageNumReturned)
		legacyregistry.MustRegister(decodeErrorCounts)
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

// RecordEtcdRequest updates and sets the etcd_request_duration_seconds,
// etcd_request_total, etcd_request_errors_total metrics.
func RecordEtcdRequest(verb string, groupResource schema.GroupResource, err error, startTime time.Time) {
	etcdRequestLatency.WithLabelValues(verb, groupResource.Group, groupResource.Resource).Observe(sinceInSeconds(startTime))
	etcdRequestCounts.WithLabelValues(verb, groupResource.Group, groupResource.Resource).Inc()
	if err != nil {
		etcdRequestErrorCounts.WithLabelValues(verb, groupResource.Group, groupResource.Resource).Inc()
	}
}

// RecordEtcdEvent updated the etcd_events_received_total metric.
func RecordEtcdEvent(groupResource schema.GroupResource) {
	etcdEventsReceivedCounts.WithLabelValues(groupResource.Group, groupResource.Resource).Inc()
}

// RecordEtcdBookmark updates the etcd_bookmark_counts metric.
func RecordEtcdBookmark(groupResource schema.GroupResource) {
	etcdBookmarkCounts.WithLabelValues(groupResource.Group, groupResource.Resource).Inc()
}

// RecordDecodeError sets the storage_decode_errors metrics.
func RecordDecodeError(groupResource schema.GroupResource) {
	decodeErrorCounts.WithLabelValues(groupResource.Group, groupResource.Resource).Inc()
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

// UpdateEtcdDbSize sets the etcd_db_total_size_in_bytes metric.
// Deprecated: Metric etcd_db_total_size_in_bytes will be replaced with apiserver_storage_size_bytes
func UpdateEtcdDbSize(ep string, size int64) {
	dbTotalSize.WithLabelValues(ep).Set(float64(size))
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

// RecordListEtcd3Metrics notes various metrics of the cost to serve a LIST request
func RecordStorageListMetrics(groupResource schema.GroupResource, numFetched, numEvald, numReturned int) {
	listStorageCount.WithLabelValues(groupResource.Group, groupResource.Resource).Inc()
	listStorageNumFetched.WithLabelValues(groupResource.Group, groupResource.Resource).Add(float64(numFetched))
	listStorageNumSelectorEvals.WithLabelValues(groupResource.Group, groupResource.Resource).Add(float64(numEvald))
	listStorageNumReturned.WithLabelValues(groupResource.Group, groupResource.Resource).Add(float64(numReturned))
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
		m.Close()
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
