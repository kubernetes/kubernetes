//go:build linux
// +build linux

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

package e2enode

import (
	"context"
	"fmt"
	"os"
	"path"
	"sort"
	"strconv"
	"time"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/kubernetes/test/e2e/framework"
	e2emetrics "k8s.io/kubernetes/test/e2e/framework/metrics"
	e2eperf "k8s.io/kubernetes/test/e2e/framework/perf"
	"k8s.io/kubernetes/test/e2e/perftype"
	nodeperftype "k8s.io/kubernetes/test/e2e_node/perftype"
)

const (
	// TimeSeriesTag is the tag for time series.
	TimeSeriesTag = "[Result:TimeSeries]"
	// TimeSeriesEnd is the end tag for time series.
	TimeSeriesEnd = "[Finish:TimeSeries]"
)

// dumpDataToFile inserts the current timestamp into the labels and writes the
// data for the test into the file with the specified prefix.
func dumpDataToFile(data interface{}, labels map[string]string, prefix string) {
	testName := labels["test"]
	fileName := path.Join(framework.TestContext.ReportDir, fmt.Sprintf("%s-%s-%s.json", prefix, framework.TestContext.ReportPrefix, testName))
	labels["timestamp"] = strconv.FormatInt(time.Now().UTC().Unix(), 10)
	framework.Logf("Dumping perf data for test %q to %q.", testName, fileName)
	if err := os.WriteFile(fileName, []byte(framework.PrettyPrintJSON(data)), 0644); err != nil {
		framework.Logf("Failed to write perf data for test %q to %q: %v", testName, fileName, err)
	}
}

// logPerfData writes the perf data to a standalone json file if the
// framework.TestContext.ReportDir is non-empty, or to the general build log
// otherwise. The perfType identifies which type of the perf data it is, such
// as "cpu" and "memory". If an error occurs, no perf data will be logged.
func logPerfData(p *perftype.PerfData, perfType string) {
	if framework.TestContext.ReportDir == "" {
		printPerfData(p)
		return
	}
	dumpDataToFile(p, p.Labels, "performance-"+perfType)
}

// logDensityTimeSeries writes the time series data of operation and resource
// usage to a standalone json file if the framework.TestContext.ReportDir is
// non-empty, or to the general build log otherwise. If an error occurs,
// no perf data will be logged.
func logDensityTimeSeries(rc *ResourceCollector, create, watch map[string]metav1.Time, testInfo map[string]string) {
	timeSeries := &nodeperftype.NodeTimeSeries{
		Labels:  testInfo,
		Version: e2eperf.CurrentKubeletPerfMetricsVersion,
	}
	// Attach operation time series.
	timeSeries.OperationData = map[string][]int64{
		"create":  getCumulatedPodTimeSeries(create),
		"running": getCumulatedPodTimeSeries(watch),
	}
	// Attach resource time series.
	timeSeries.ResourceData = rc.GetResourceTimeSeries()

	if framework.TestContext.ReportDir == "" {
		framework.Logf("%s %s\n%s", TimeSeriesTag, framework.PrettyPrintJSON(timeSeries), TimeSeriesEnd)
		return
	}
	dumpDataToFile(timeSeries, timeSeries.Labels, "time_series")
}

type int64arr []int64

func (a int64arr) Len() int           { return len(a) }
func (a int64arr) Swap(i, j int)      { a[i], a[j] = a[j], a[i] }
func (a int64arr) Less(i, j int) bool { return a[i] < a[j] }

// getCumulatedPodTimeSeries gets the cumulative pod number time series.
func getCumulatedPodTimeSeries(timePerPod map[string]metav1.Time) []int64 {
	timeSeries := make(int64arr, 0)
	for _, ts := range timePerPod {
		timeSeries = append(timeSeries, ts.Time.UnixNano())
	}
	// Sort all timestamps.
	sort.Sort(timeSeries)
	return timeSeries
}

// getLatencyPerfData returns perf data of pod startup latency.
func getLatencyPerfData(latency e2emetrics.LatencyMetric, testInfo map[string]string) *perftype.PerfData {
	return &perftype.PerfData{
		Version: e2eperf.CurrentKubeletPerfMetricsVersion,
		DataItems: []perftype.DataItem{
			{
				Data: map[string]float64{
					"Perc50":  float64(latency.Perc50) / 1000000,
					"Perc90":  float64(latency.Perc90) / 1000000,
					"Perc99":  float64(latency.Perc99) / 1000000,
					"Perc100": float64(latency.Perc100) / 1000000,
				},
				Unit: "ms",
				Labels: map[string]string{
					"datatype":    "latency",
					"latencytype": "create-pod",
				},
			},
		},
		Labels: testInfo,
	}
}

// getThroughputPerfData returns perf data of pod creation startup throughput.
func getThroughputPerfData(batchLag time.Duration, e2eLags []e2emetrics.PodLatencyData, podsNr int, testInfo map[string]string) *perftype.PerfData {
	return &perftype.PerfData{
		Version: e2eperf.CurrentKubeletPerfMetricsVersion,
		DataItems: []perftype.DataItem{
			{
				Data: map[string]float64{
					"batch":        float64(podsNr) / batchLag.Minutes(),
					"single-worst": 1.0 / e2eLags[len(e2eLags)-1].Latency.Minutes(),
				},
				Unit: "pods/min",
				Labels: map[string]string{
					"datatype":    "throughput",
					"latencytype": "create-pod",
				},
			},
		},
		Labels: testInfo,
	}
}

// getTestNodeInfo returns a label map containing the test name and
// description, the name of the node on which the test will be run, the image
// name of the node, and the node capacities.
func getTestNodeInfo(f *framework.Framework, testName, testDesc string) map[string]string {
	nodeName := framework.TestContext.NodeName
	node, err := f.ClientSet.CoreV1().Nodes().Get(context.TODO(), nodeName, metav1.GetOptions{})
	framework.ExpectNoError(err)

	cpu, ok := node.Status.Capacity[v1.ResourceCPU]
	if !ok {
		framework.Failf("Fail to fetch CPU capacity value of test node.")
	}

	memory, ok := node.Status.Capacity[v1.ResourceMemory]
	if !ok {
		framework.Failf("Fail to fetch Memory capacity value of test node.")
	}

	cpuValue, ok := cpu.AsInt64()
	if !ok {
		framework.Failf("Fail to fetch CPU capacity value as Int64.")
	}

	memoryValue, ok := memory.AsInt64()
	if !ok {
		framework.Failf("Fail to fetch Memory capacity value as Int64.")
	}

	image := node.Status.NodeInfo.OSImage
	if framework.TestContext.ImageDescription != "" {
		image = fmt.Sprintf("%s (%s)", image, framework.TestContext.ImageDescription)
	}
	return map[string]string{
		"node":    nodeName,
		"test":    testName,
		"image":   image,
		"machine": fmt.Sprintf("cpu:%dcore,memory:%.1fGB", cpuValue, float32(memoryValue)/(1024*1024*1024)),
		"desc":    testDesc,
	}
}

// printPerfData prints the perfdata in json format with PerfResultTag prefix.
// If an error occurs, nothing will be printed.
func printPerfData(p *perftype.PerfData) {
	// Notice that we must make sure the perftype.PerfResultEnd is in a new line.
	if str := framework.PrettyPrintJSON(p); str != "" {
		framework.Logf("%s %s\n%s", perftype.PerfResultTag, str, perftype.PerfResultEnd)
	}
}
