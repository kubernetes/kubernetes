package benchmark

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"testing"
	"time"

	"k8s.io/klog"
	testutils "k8s.io/kubernetes/test/utils"
)

// The scheduler generates metrics which can be used to get statistical
// data about various aspects of scheduling:
// - histogram of times it takes to schedule a pod from the first time it's seen
//   until it's successfully bounded
// - counter of pods that failed to be scheduled
// - counter of how many pods were evicted
// Though, the metrics does not capture dependencies between various
// Though, given the goal is to measure scheduler performance, other information
// are more important. Such as
// How

// To get time elapse of pods getting scheduled in time
// the scheduler logs have to be parsed separately.

func TestBenchmarkScheduling(t *testing.T) {
	tests := []struct{ nodes, existingPods, minPods int }{
		// {nodes: 100, existingPods: 0, minPods: 100},
		// {nodes: 100, existingPods: 100, minPods: 100},
		// {nodes: 1000, existingPods: 0, minPods: 1000},
		// {nodes: 1000, existingPods: 1000, minPods: 1000},
		{nodes: 1000, existingPods: 0, minPods: 5000},
		// {nodes: 1000, existingPods: 5000, minPods: 5000},
	}
	setupStrategy := testutils.NewSimpleWithControllerCreatePodStrategy("rc1")
	testStrategy := testutils.NewSimpleWithControllerCreatePodStrategy("rc2")

	histograms := make(histogramSet, 0)
	dataItems := DataItems{
		Version: "v1",
	}

	for _, test := range tests {
		name := fmt.Sprintf("%vNodes_%vExistingPods_%vPods", test.nodes, test.existingPods, test.minPods)
		klog.Infof("Running %q\n", name)
		benchmarkScheduling(test.nodes, test.existingPods, test.minPods, defaultNodeStrategy, setupStrategy, testStrategy, &testing.B{})
		metrics := []string{
			"scheduler_scheduling_algorithm_predicate_evaluation_seconds",
			"scheduler_scheduling_algorithm_priority_evaluation_seconds",
			"scheduler_scheduling_algorithm_preemption_evaluation_seconds",
			"scheduler_binding_duration_seconds",
		}
		histograms[name] = collectRelativeMetrics(metrics)

		for _, metricName := range metrics {
			// Set scheduler_scheduling_algorithm_preemption_evaluation_seconds only when enabled.
			// Otherwise, the values are -inf.
			q50 := bucketQuantile(0.50, histograms[name][metricName].buckets)
			q90 := bucketQuantile(0.90, histograms[name][metricName].buckets)
			q99 := bucketQuantile(0.95, histograms[name][metricName].buckets)

			latency := LatencyMetric{
				Perc50: time.Duration(int64(q50 * float64(time.Second))),
				Perc90: time.Duration(int64(q90 * float64(time.Second))),
				Perc99: time.Duration(int64(q99 * float64(time.Second))),
			}

			dataItem := latency.ToPerfData(metricName)
			dataItem.Labels["Name"] = name
			dataItems.DataItems = append(dataItems.DataItems, dataItem)
		}
	}

	b, _ := json.Marshal(dataItems)
	fmt.Printf("dataItems: %v\n", string(b))
	ioutil.WriteFile(fmt.Sprintf("SchedulingMetrics_density_%v.json", time.Now().Format("2006-01-02T15:04:05Z")), b, 0644)

	// fmt.Printf("histograms: %#v\n", histograms)

	// ioutil.WriteFile("scheduler_pod_scheduling_duration_seconds.dat", []byte(histograms.string("scheduler_pod_scheduling_duration_seconds")), 0644)
	// ioutil.WriteFile("scheduler_e2e_scheduling_duration_seconds.dat", []byte(histograms.string("scheduler_e2e_scheduling_duration_seconds")), 0644)
	// ioutil.WriteFile("scheduler_scheduling_algorithm_duration_seconds.dat", []byte(histograms.string("scheduler_scheduling_algorithm_duration_seconds")), 0644)
	// ioutil.WriteFile("scheduler_binding_duration_seconds.dat", []byte(histograms.string("scheduler_binding_duration_seconds")), 0644)
}
