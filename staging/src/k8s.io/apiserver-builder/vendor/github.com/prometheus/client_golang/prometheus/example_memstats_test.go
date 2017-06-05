// Copyright 2014 The Prometheus Authors
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package prometheus_test

import (
	"runtime"

	"github.com/prometheus/client_golang/prometheus"
)

var (
	allocDesc = prometheus.NewDesc(
		prometheus.BuildFQName("", "memstats", "alloc_bytes"),
		"bytes allocated and still in use",
		nil, nil,
	)
	totalAllocDesc = prometheus.NewDesc(
		prometheus.BuildFQName("", "memstats", "total_alloc_bytes"),
		"bytes allocated (even if freed)",
		nil, nil,
	)
	numGCDesc = prometheus.NewDesc(
		prometheus.BuildFQName("", "memstats", "num_gc_total"),
		"number of GCs run",
		nil, nil,
	)
)

// MemStatsCollector is an example for a custom Collector that solves the
// problem of feeding into multiple metrics at the same time. The
// runtime.ReadMemStats should happen only once, and then the results need to be
// fed into a number of separate Metrics. In this example, only a few of the
// values reported by ReadMemStats are used. For each, there is a Desc provided
// as a var, so the MemStatsCollector itself needs nothing else in the
// struct. Only the methods need to be implemented.
type MemStatsCollector struct{}

// Describe just sends the three Desc objects for the Metrics we intend to
// collect.
func (_ MemStatsCollector) Describe(ch chan<- *prometheus.Desc) {
	ch <- allocDesc
	ch <- totalAllocDesc
	ch <- numGCDesc
}

// Collect does the trick by calling ReadMemStats once and then constructing
// three different Metrics on the fly.
func (_ MemStatsCollector) Collect(ch chan<- prometheus.Metric) {
	var ms runtime.MemStats
	runtime.ReadMemStats(&ms)
	ch <- prometheus.MustNewConstMetric(
		allocDesc,
		prometheus.GaugeValue,
		float64(ms.Alloc),
	)
	ch <- prometheus.MustNewConstMetric(
		totalAllocDesc,
		prometheus.GaugeValue,
		float64(ms.TotalAlloc),
	)
	ch <- prometheus.MustNewConstMetric(
		numGCDesc,
		prometheus.CounterValue,
		float64(ms.NumGC),
	)
	// To avoid new allocations on each collection, you could also keep
	// metric objects around and return the same objects each time, just
	// with new values set.
}

func ExampleCollector_memstats() {
	prometheus.MustRegister(&MemStatsCollector{})
	// Since we are dealing with custom Collector implementations, it might
	// be a good idea to enable the collect checks in the registry.
	prometheus.EnableCollectChecks(true)
}
