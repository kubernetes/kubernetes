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
	compbasemetrics "k8s.io/component-base/metrics"
	"k8s.io/utils/clock"
)

type timestampedResourceGaugeCollector struct {
	compbasemetrics.BaseStableCollector

	desc  *compbasemetrics.Desc
	clock clock.PassiveClock

	lock    sync.RWMutex
	samples map[schema.GroupResource]timestampedSample
}

type timestampedSample struct {
	value     int64
	timestamp time.Time
}

func newTimestampedResourceGaugeCollector(desc *compbasemetrics.Desc, c clock.PassiveClock) *timestampedResourceGaugeCollector {
	if c == nil {
		c = clock.RealClock{}
	}
	return &timestampedResourceGaugeCollector{
		desc:    desc,
		clock:   c,
		samples: make(map[schema.GroupResource]timestampedSample),
	}
}

func (c *timestampedResourceGaugeCollector) setClock(clk clock.PassiveClock) {
	c.lock.Lock()
	defer c.lock.Unlock()
	if clk == nil {
		clk = clock.RealClock{}
	}
	c.clock = clk
}

// Check if timestampedResourceGaugeCollector implements necessary interface
var _ compbasemetrics.StableCollector = &timestampedResourceGaugeCollector{}

// DescribeWithStability implements compbasemetrics.StableCollector
func (c *timestampedResourceGaugeCollector) DescribeWithStability(ch chan<- *compbasemetrics.Desc) {
	ch <- c.desc
}

// CollectWithStability implements compbasemetrics.StableCollector
func (c *timestampedResourceGaugeCollector) CollectWithStability(ch chan<- compbasemetrics.Metric) {
	c.lock.RLock()
	defer c.lock.RUnlock()

	for gr, sample := range c.samples {
		ch <- compbasemetrics.NewLazyMetricWithTimestamp(
			sample.timestamp,
			compbasemetrics.NewLazyConstMetric(
				c.desc,
				compbasemetrics.GaugeValue,
				float64(sample.value),
				gr.Group,
				gr.Resource,
			),
		)
	}
}

func (c *timestampedResourceGaugeCollector) set(gr schema.GroupResource, value int64) {
	c.lock.Lock()
	defer c.lock.Unlock()

	c.samples[gr] = timestampedSample{
		value:     value,
		timestamp: c.clock.Now(),
	}
}

func (c *timestampedResourceGaugeCollector) delete(gr schema.GroupResource) {
	c.lock.Lock()
	defer c.lock.Unlock()
	delete(c.samples, gr)
}

func (c *timestampedResourceGaugeCollector) Reset() {
	c.lock.Lock()
	defer c.lock.Unlock()
	c.samples = make(map[schema.GroupResource]timestampedSample)
}
