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

package correctness

import (
	"fmt"
	"sort"
	"strings"

	"k8s.io/apimachinery/pkg/watch"
)

// Percentiles holds statistical percentiles for a distribution.
type Percentiles struct {
	Min  int
	P25  int
	P50  int // Median
	P75  int
	P90  int
	P99  int
	Max  int
	Mean float64
}

// WatchCoverage captures aggregate metrics and distribution of recorded watch sessions.
type WatchCoverage struct {
	TotalWatches     int
	ZeroEventWatches int
	ZeroEventRatio   float64

	TotalEvents  int
	EventsByType map[watch.EventType]int

	EventsPerWatch Percentiles

	ByScope   map[string]int
	ByStartRV map[string]int
}

// ComputeWatchCoverage computes quality and distribution metrics across recorded watch sessions.
func ComputeWatchCoverage(watches []RecordedWatch) WatchCoverage {
	cov := WatchCoverage{
		TotalWatches: len(watches),
		EventsByType: make(map[watch.EventType]int),
		ByScope:      make(map[string]int),
		ByStartRV:    make(map[string]int),
	}
	if len(watches) == 0 {
		return cov
	}

	eventCounts := make([]int, len(watches))
	for i, rw := range watches {
		cnt := len(rw.Response.Events)
		eventCounts[i] = cnt
		cov.TotalEvents += cnt

		if cnt == 0 {
			cov.ZeroEventWatches++
		}

		for _, ev := range rw.Response.Events {
			cov.EventsByType[ev.Type]++
		}

		rv := rw.Request.ResourceVersion
		switch rv {
		case "", "0":
			cov.ByStartRV["zero"]++
		case "1":
			cov.ByStartRV["beginning"]++
		default:
			cov.ByStartRV["concrete"]++
		}

		key := rw.Request.Key
		switch {
		case key == "/pods" || key == "/pods/":
			cov.ByScope["cluster"]++
		case strings.Count(strings.Trim(key, "/"), "/") == 1:
			cov.ByScope["namespace"]++
		default:
			cov.ByScope["object"]++
		}
	}

	cov.ZeroEventRatio = float64(cov.ZeroEventWatches) / float64(cov.TotalWatches)
	cov.EventsPerWatch = computePercentiles(eventCounts)
	return cov
}

func computePercentiles(values []int) Percentiles {
	if len(values) == 0 {
		return Percentiles{}
	}
	sorted := make([]int, len(values))
	copy(sorted, values)
	sort.Ints(sorted)

	sum := 0
	for _, v := range sorted {
		sum += v
	}

	p := func(pct float64) int {
		idx := int(float64(len(sorted)-1) * pct)
		return sorted[idx]
	}

	return Percentiles{
		Min:  sorted[0],
		P25:  p(0.25),
		P50:  p(0.50),
		P75:  p(0.75),
		P90:  p(0.90),
		P99:  p(0.99),
		Max:  sorted[len(sorted)-1],
		Mean: float64(sum) / float64(len(sorted)),
	}
}

// Summary returns a formatted human-readable summary of watch coverage metrics.
func (c WatchCoverage) Summary() string {
	var sb strings.Builder
	sb.WriteString(fmt.Sprintf("Total Watches: %d (Zero-event: %d [%.2f%%])\n",
		c.TotalWatches, c.ZeroEventWatches, c.ZeroEventRatio*100))
	sb.WriteString(fmt.Sprintf("Total Events: %d (Added: %d, Modified: %d, Deleted: %d, Bookmarks: %d, Errors: %d)\n",
		c.TotalEvents, c.EventsByType[watch.Added], c.EventsByType[watch.Modified],
		c.EventsByType[watch.Deleted], c.EventsByType[watch.Bookmark], c.EventsByType[watch.Error]))
	sb.WriteString(fmt.Sprintf("Events/Watch: Min=%d, P25=%d, Median(P50)=%d, P75=%d, P90=%d, P99=%d, Max=%d, Mean=%.2f\n",
		c.EventsPerWatch.Min, c.EventsPerWatch.P25, c.EventsPerWatch.P50,
		c.EventsPerWatch.P75, c.EventsPerWatch.P90, c.EventsPerWatch.P99, c.EventsPerWatch.Max, c.EventsPerWatch.Mean))
	sb.WriteString(fmt.Sprintf("Start RVs: Zero/Latest=%d, Beginning(1)=%d, Concrete=%d\n",
		c.ByStartRV["zero"], c.ByStartRV["beginning"], c.ByStartRV["concrete"]))
	sb.WriteString(fmt.Sprintf("Scopes: Cluster=%d, Namespace=%d, Object=%d",
		c.ByScope["cluster"], c.ByScope["namespace"], c.ByScope["object"]))
	return sb.String()
}
