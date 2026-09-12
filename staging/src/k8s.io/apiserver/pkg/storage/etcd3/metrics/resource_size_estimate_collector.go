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
	"time"

	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apiserver/pkg/storage"
	compbasemetrics "k8s.io/component-base/metrics"
)

var (
	resourceSizeEstimateDesc = compbasemetrics.NewDesc(
		"apiserver_resource_size_estimate_bytes",
		"Estimated size of stored objects in database. Estimate is based on sum of last observed sizes of serialized objects.",
		[]string{"group", "resource"},
		nil,
		compbasemetrics.ALPHA,
		"",
	)
)

type resourceSizeEstimateCollector struct {
	compbasemetrics.BaseStableCollector

	lock      sync.RWMutex
	estimates map[schema.GroupResource]resourceEstimate
}

type resourceEstimate struct {
	size      int64
	timestamp time.Time
}

func newResourceSizeEstimateCollector() *resourceSizeEstimateCollector {
	return &resourceSizeEstimateCollector{
		estimates: make(map[schema.GroupResource]resourceEstimate),
	}
}

// Check if resourceSizeEstimateCollector implements necessary interface
var _ compbasemetrics.StableCollector = &resourceSizeEstimateCollector{}

// DescribeWithStability implements compbasemetrics.StableCollector
func (c *resourceSizeEstimateCollector) DescribeWithStability(ch chan<- *compbasemetrics.Desc) {
	ch <- resourceSizeEstimateDesc
}

// CollectWithStability implements compbasemetrics.StableCollector
func (c *resourceSizeEstimateCollector) CollectWithStability(ch chan<- compbasemetrics.Metric) {
	c.lock.RLock()
	defer c.lock.RUnlock()

	for gr, estimate := range c.estimates {
		ch <- compbasemetrics.NewLazyMetricWithTimestamp(
			estimate.timestamp,
			compbasemetrics.NewLazyConstMetric(
				resourceSizeEstimateDesc,
				compbasemetrics.GaugeValue,
				float64(estimate.size),
				gr.Group,
				gr.Resource,
			),
		)
	}
}

func (c *resourceSizeEstimateCollector) updateStoreStats(gr schema.GroupResource, stats storage.Stats, err error) {
	c.lock.Lock()
	defer c.lock.Unlock()

	if err != nil || (stats.ObjectCount > 0 && stats.EstimatedAverageObjectSizeBytes == 0) {
		delete(c.estimates, gr)
		return
	}

	c.estimates[gr] = resourceEstimate{
		size:      stats.EstimatedAverageObjectSizeBytes * stats.ObjectCount,
		timestamp: now(),
	}
}

func (c *resourceSizeEstimateCollector) deleteStoreStats(gr schema.GroupResource) {
	c.lock.Lock()
	defer c.lock.Unlock()
	delete(c.estimates, gr)
}

func (c *resourceSizeEstimateCollector) Reset() {
	c.lock.Lock()
	defer c.lock.Unlock()
	c.estimates = make(map[schema.GroupResource]resourceEstimate)
}
