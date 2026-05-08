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

package metrics

import (
	"sync"

	"k8s.io/apimachinery/pkg/runtime/schema"
	compbasemetrics "k8s.io/component-base/metrics"
	"k8s.io/component-base/metrics/legacyregistry"
)

const (
	subsystem                = "apiserver"
	StorageBackendEtcd       = "etcd"
	StorageBackendWatchCache = "watchcache"
)

var (
	listTotal = compbasemetrics.NewCounterVec(
		&compbasemetrics.CounterOpts{
			Subsystem:      subsystem,
			Name:           "storage_list_requests_total",
			Help:           "Number of LIST requests served from storage partitioned by backend and index.",
			StabilityLevel: compbasemetrics.ALPHA,
		},
		[]string{"group", "resource", "storage", "index"},
	)
	listFetchedObjectsTotal = compbasemetrics.NewCounterVec(
		&compbasemetrics.CounterOpts{
			Subsystem:      subsystem,
			Name:           "storage_list_fetched_objects_total",
			Help:           "Number of objects read from storage in the course of serving a LIST request partitioned by backend and index.",
			StabilityLevel: compbasemetrics.ALPHA,
		},
		[]string{"group", "resource", "storage", "index"},
	)
	listEvaluatedObjectsTotal = compbasemetrics.NewCounterVec(
		&compbasemetrics.CounterOpts{
			Subsystem:      subsystem,
			Name:           "storage_list_evaluated_objects_total",
			Help:           "Number of objects tested in the course of serving a LIST request from storage partitioned by backend.",
			StabilityLevel: compbasemetrics.ALPHA,
		},
		[]string{"group", "resource", "storage"},
	)
	listReturnedObjectsTotal = compbasemetrics.NewCounterVec(
		&compbasemetrics.CounterOpts{
			Subsystem:      subsystem,
			Name:           "storage_list_returned_objects_total",
			Help:           "Number of objects returned for a LIST request from storage partitioned by backend.",
			StabilityLevel: compbasemetrics.ALPHA,
		},
		[]string{"group", "resource", "storage"},
	)
	registerMetrics sync.Once
)

// Register registers shared storage LIST metrics.
func Register() {
	registerMetrics.Do(func() {
		legacyregistry.MustRegister(listTotal)
		legacyregistry.MustRegister(listFetchedObjectsTotal)
		legacyregistry.MustRegister(listEvaluatedObjectsTotal)
		legacyregistry.MustRegister(listReturnedObjectsTotal)
	})
}

// RegisterMetricsForTest registers shared storage LIST metrics on a test registry.
func RegisterMetricsForTest(registry compbasemetrics.KubeRegistry) {
	registry.MustRegister(listTotal)
	registry.MustRegister(listFetchedObjectsTotal)
	registry.MustRegister(listEvaluatedObjectsTotal)
	registry.MustRegister(listReturnedObjectsTotal)
}

// ResetMetricsForTest resets shared storage LIST metrics between tests.
func ResetMetricsForTest() {
	listTotal.Reset()
	listFetchedObjectsTotal.Reset()
	listEvaluatedObjectsTotal.Reset()
	listReturnedObjectsTotal.Reset()
}

// RecordListMetrics records LIST request count, fetched objects, and returned objects.
func RecordListMetrics(groupResource schema.GroupResource, storageBackend, index string, numFetched, numReturned int) {
	listTotal.WithLabelValues(groupResource.Group, groupResource.Resource, storageBackend, index).Inc()
	listFetchedObjectsTotal.WithLabelValues(groupResource.Group, groupResource.Resource, storageBackend, index).Add(float64(numFetched))
	listReturnedObjectsTotal.WithLabelValues(groupResource.Group, groupResource.Resource, storageBackend).Add(float64(numReturned))
}

// RecordListEvaluatedObjects records the number of objects evaluated while serving a LIST request.
func RecordListEvaluatedObjects(groupResource schema.GroupResource, storageBackend string, numEvaluated int) {
	listEvaluatedObjectsTotal.WithLabelValues(groupResource.Group, groupResource.Resource, storageBackend).Add(float64(numEvaluated))
}
