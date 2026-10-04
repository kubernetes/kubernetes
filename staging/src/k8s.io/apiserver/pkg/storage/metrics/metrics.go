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
	StorageBackendEtcd       = "etcd"
	StorageBackendWatchCache = "watchcache"
)

var (
	listStorageCount = compbasemetrics.NewCounterVec(
		&compbasemetrics.CounterOpts{
			Name:           "apiserver_storage_list_total",
			Help:           "Number of LIST requests served from storage",
			StabilityLevel: compbasemetrics.ALPHA,
		},
		[]string{"group", "resource", "storage", "index"},
	)
	listStorageNumFetched = compbasemetrics.NewCounterVec(
		&compbasemetrics.CounterOpts{
			Name:           "apiserver_storage_list_fetched_objects_total",
			Help:           "Number of objects read from storage in the course of serving a LIST request",
			StabilityLevel: compbasemetrics.ALPHA,
		},
		[]string{"group", "resource", "storage", "index"},
	)
	listStorageNumSelectorEvals = compbasemetrics.NewCounterVec(
		&compbasemetrics.CounterOpts{
			Name:           "apiserver_storage_list_evaluated_objects_total",
			Help:           "Number of objects tested in the course of serving a LIST request from storage",
			StabilityLevel: compbasemetrics.ALPHA,
		},
		[]string{"group", "resource", "storage"},
	)
	listStorageNumReturned = compbasemetrics.NewCounterVec(
		&compbasemetrics.CounterOpts{
			Name:           "apiserver_storage_list_returned_objects_total",
			Help:           "Number of objects returned for a LIST request from storage",
			StabilityLevel: compbasemetrics.ALPHA,
		},
		[]string{"group", "resource", "storage"},
	)
	registerMetrics sync.Once
)

// Register registers storage LIST metrics.
func Register() {
	registerMetrics.Do(func() {
		legacyregistry.MustRegister(listStorageCount)
		legacyregistry.MustRegister(listStorageNumFetched)
		legacyregistry.MustRegister(listStorageNumSelectorEvals)
		legacyregistry.MustRegister(listStorageNumReturned)
	})
}

// RecordStorageListMetrics notes various metrics of the cost to serve a LIST request.
func RecordStorageListMetrics(groupResource schema.GroupResource, storageBackend, index string, numFetched, numEvald, numReturned int) {
	listStorageCount.WithLabelValues(groupResource.Group, groupResource.Resource, storageBackend, index).Inc()
	listStorageNumFetched.WithLabelValues(groupResource.Group, groupResource.Resource, storageBackend, index).Add(float64(numFetched))
	listStorageNumSelectorEvals.WithLabelValues(groupResource.Group, groupResource.Resource, storageBackend).Add(float64(numEvald))
	listStorageNumReturned.WithLabelValues(groupResource.Group, groupResource.Resource, storageBackend).Add(float64(numReturned))
}
