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
	"math"
	runtimemetrics "runtime/metrics"
	"sort"

	"github.com/prometheus/client_golang/prometheus"
	compbasemetrics "k8s.io/component-base/metrics"
)

const (
	schedLatenciesName = "apiserver_go_sched_latencies_seconds"
	schedLatenciesHelp = "Distribution of Go runtime goroutine scheduling latencies (/sched/latencies:seconds) exported at (near-)full resolution, unlike the standard Go collector whose buckets stop around 100ms. The runtime's irregular buckets are downsampled onto a fixed set of bounds up to 10s: each runtime bucket's count is credited to the smallest export bound not below that bucket's upper edge, and the sum is approximated with runtime bucket midpoints."
)

// schedLatencyBuckets is the fixed export bucket set the runtime histogram is
// downsampled onto. Bounds cover the runtime's full range up to 10 seconds;
// anything above lands in +Inf.
var schedLatencyBuckets = []float64{
	5e-6, 1e-5, 2.5e-5, 5e-5, 1e-4, 2.5e-4, 5e-4,
	1e-3, 2.5e-3, 5e-3, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5,
	1, 2.5, 5, 10,
}

type schedLatenciesCollector struct {
	compbasemetrics.BaseStableCollector

	desc     *compbasemetrics.Desc
	promDesc *prometheus.Desc
}

func newSchedLatenciesCollector() *schedLatenciesCollector {
	return &schedLatenciesCollector{
		desc: compbasemetrics.NewDesc(schedLatenciesName, schedLatenciesHelp, nil, nil, compbasemetrics.ALPHA, ""),
		// The const histogram is built with a raw prometheus desc; mirror the
		// stability prefix the framework would add.
		promDesc: prometheus.NewDesc(schedLatenciesName, "[ALPHA] "+schedLatenciesHelp, nil, nil),
	}
}

func (c *schedLatenciesCollector) DescribeWithStability(ch chan<- *compbasemetrics.Desc) {
	ch <- c.desc
}

func (c *schedLatenciesCollector) CollectWithStability(ch chan<- compbasemetrics.Metric) {
	if c.desc.IsHidden() {
		return
	}
	samples := []runtimemetrics.Sample{{Name: "/sched/latencies:seconds"}}
	runtimemetrics.Read(samples)
	if samples[0].Value.Kind() != runtimemetrics.KindFloat64Histogram {
		return
	}
	h := samples[0].Value.Float64Histogram()

	perBucket := make([]uint64, len(schedLatencyBuckets))
	var count uint64
	var sum float64
	for i, n := range h.Counts {
		if n == 0 {
			continue
		}
		count += n
		lo, hi := h.Buckets[i], h.Buckets[i+1]
		if math.IsInf(lo, -1) {
			lo = 0
		}
		if math.IsInf(hi, 1) {
			sum += lo * float64(n)
		} else {
			sum += (lo + hi) / 2 * float64(n)
		}
		if j := sort.SearchFloat64s(schedLatencyBuckets, hi); j < len(perBucket) {
			perBucket[j] += n
		}
	}
	buckets := make(map[float64]uint64, len(schedLatencyBuckets))
	var cum uint64
	for j, ub := range schedLatencyBuckets {
		cum += perBucket[j]
		buckets[ub] = cum
	}
	ch <- prometheus.MustNewConstHistogram(c.promDesc, count, sum, buckets)
}
