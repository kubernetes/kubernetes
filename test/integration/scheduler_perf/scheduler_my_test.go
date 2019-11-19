package benchmark

import (
	"fmt"
	"io/ioutil"
	"testing"

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
		{nodes: 100, existingPods: 0, minPods: 100},
		{nodes: 100, existingPods: 100, minPods: 100},
		{nodes: 1000, existingPods: 0, minPods: 1000},
		{nodes: 1000, existingPods: 1000, minPods: 1000},
		{nodes: 1000, existingPods: 0, minPods: 5000},
		{nodes: 1000, existingPods: 5000, minPods: 5000},
	}
	setupStrategy := testutils.NewSimpleWithControllerCreatePodStrategy("rc1")
	testStrategy := testutils.NewSimpleWithControllerCreatePodStrategy("rc2")

	histograms := make(histogramSet, 0)

	for _, test := range tests {
		name := fmt.Sprintf("%vNodes_%vExistingPods_%vPods", test.nodes, test.existingPods, test.minPods)
		klog.Infof("Running %q\n", name)
		benchmarkScheduling(test.nodes, test.existingPods, test.minPods, defaultNodeStrategy, setupStrategy, testStrategy, &testing.B{})
		histograms[name] = collectRelativeMetrics([]string{"scheduler_pod_scheduling_duration_seconds", "scheduler_e2e_scheduling_duration_seconds", "scheduler_scheduling_algorithm_duration_seconds", "scheduler_binding_duration_seconds"})
		// fmt.Printf("Histogram: %v", histograms[name].string())
	}

	ioutil.WriteFile("scheduler_pod_scheduling_duration_seconds.dat", []byte(histograms.string("scheduler_pod_scheduling_duration_seconds")), 0644)
	ioutil.WriteFile("scheduler_e2e_scheduling_duration_seconds.dat", []byte(histograms.string("scheduler_e2e_scheduling_duration_seconds")), 0644)
	ioutil.WriteFile("scheduler_scheduling_algorithm_duration_seconds.dat", []byte(histograms.string("scheduler_scheduling_algorithm_duration_seconds")), 0644)
	ioutil.WriteFile("scheduler_binding_duration_seconds.dat", []byte(histograms.string("scheduler_binding_duration_seconds")), 0644)
}
