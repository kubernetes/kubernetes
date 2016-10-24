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

package e2e_node

import (
	"fmt"
	"sort"
	"time"

	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/test/e2e/framework"
	"k8s.io/kubernetes/test/e2e/perftype"

	. "github.com/onsi/gomega"
)

const (
	// TODO(coufon): be consistent with perf_util.go version
	currentDataVersion = "v1"
	TimeSeriesTag      = "[Result:TimeSeries]"
	TimeSeriesEnd      = "[Finish:TimeSeries]"
)

type NodeTimeSeries struct {
	// value in OperationData is an array of timestamps
	OperationData map[string][]int64         `json:"op_series,omitempty"`
	ResourceData  map[string]*ResourceSeries `json:"resource_series,omitempty"`
	Labels        map[string]string          `json:"labels"`
	Version       string                     `json:"version"`
}

// logDensityTimeSeries logs the time series data of operation and resource usage
func logDensityTimeSeries(rc *ResourceCollector, create, watch map[string]unversioned.Time, testInfo map[string]string) {
	timeSeries := &NodeTimeSeries{
		Labels:  testInfo,
		Version: currentDataVersion,
	}
	// Attach operation time series.
	timeSeries.OperationData = map[string][]int64{
		"create":  getCumulatedPodTimeSeries(create),
		"running": getCumulatedPodTimeSeries(watch),
	}
	// Attach resource time series.
	timeSeries.ResourceData = rc.GetResourceTimeSeries()
	// Log time series with tags
	framework.Logf("%s %s\n%s", TimeSeriesTag, framework.PrettyPrintJSON(timeSeries), TimeSeriesEnd)
}

type int64arr []int64

func (a int64arr) Len() int           { return len(a) }
func (a int64arr) Swap(i, j int)      { a[i], a[j] = a[j], a[i] }
func (a int64arr) Less(i, j int) bool { return a[i] < a[j] }

// getCumulatedPodTimeSeries gets the cumulative pod number time series.
func getCumulatedPodTimeSeries(timePerPod map[string]unversioned.Time) []int64 {
	timeSeries := make(int64arr, 0)
	for _, ts := range timePerPod {
		timeSeries = append(timeSeries, ts.Time.UnixNano())
	}
	// Sort all timestamps.
	sort.Sort(timeSeries)
	return timeSeries
}

// getLatencyPerfData returns perf data of pod startup latency.
func getLatencyPerfData(latency framework.LatencyMetric, testInfo map[string]string) *perftype.PerfData {
	return &perftype.PerfData{
		Version: currentDataVersion,
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
func getThroughputPerfData(batchLag time.Duration, e2eLags []framework.PodLatencyData, podsNr int, testInfo map[string]string) *perftype.PerfData {
	return &perftype.PerfData{
		Version: currentDataVersion,
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

// getTestNodeInfo fetches the capacity of a node from API server and returns a map of labels.
func getTestNodeInfo(f *framework.Framework, testName string) map[string]string {
	nodeName := framework.TestContext.NodeName
	node, err := f.ClientSet.Core().Nodes().Get(nodeName)
	Expect(err).NotTo(HaveOccurred())

	cpu, ok := node.Status.Capacity["cpu"]
	if !ok {
		framework.Failf("Fail to fetch CPU capacity value of test node.")
	}

	memory, ok := node.Status.Capacity["memory"]
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

	return map[string]string{
		"node":    nodeName,
		"test":    testName,
		"image":   node.Status.NodeInfo.OSImage,
		"machine": fmt.Sprintf("cpu:%dcore,memory:%.1fGB", cpuValue, float32(memoryValue)/(1024*1024*1024)),
	}
}
