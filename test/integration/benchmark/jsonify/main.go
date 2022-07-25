/*
Copyright 2018 The Kubernetes Authors.

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

package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os"

	benchparse "golang.org/x/tools/benchmark/parse"
)

// TODO(random-liu): Replace this with prometheus' data model.

// The following performance data structures are generalized and well-formatted.
// They can be pretty printed in json format and be analyzed by other performance
// analyzing tools, such as Perfdash (k8s.io/contrib/perfdash).

// DataItem is the data point.
type DataItem struct {
	// Data is a map from bucket to real data point (e.g. "Perc90" -> 23.5). Notice
	// that all data items with the same label combination should have the same buckets.
	Data map[string]float64 `json:"data"`
	// Unit is the data unit. Notice that all data items with the same label combination
	// should have the same unit.
	Unit string `json:"unit"`
	// Labels is the labels of the data item.
	Labels map[string]string `json:"labels,omitempty"`
}

// PerfData contains all data items generated in current test.
type PerfData struct {
	// Version is the version of the metrics. The metrics consumer could use the version
	// to detect metrics version change and decide what version to support.
	Version   string     `json:"version"`
	DataItems []DataItem `json:"dataItems"`
	// Labels is the labels of the dataset.
	Labels map[string]string `json:"labels,omitempty"`
}

func main() {
	err := run()
	if err != nil {
		panic(err)
	}
}

func run() error {
	if len(os.Args) < 2 {
		return fmt.Errorf("output filename is a required argument")
	}
	benchmarkSet, err := benchparse.ParseSet(os.Stdin)
	if err != nil {
		return err
	}
	data := PerfData{Version: "v1"}
	for _, benchMarks := range benchmarkSet {
		for _, benchMark := range benchMarks {
			data.DataItems = appendIfMeasured(data.DataItems, benchMark, benchparse.NsPerOp, "time", "μs", benchMark.NsPerOp/1000.0)
			data.DataItems = appendIfMeasured(data.DataItems, benchMark, benchparse.MBPerS, "throughput", "MBps", benchMark.MBPerS)
			data.DataItems = appendIfMeasured(data.DataItems, benchMark, benchparse.AllocedBytesPerOp, "allocated", "bytes", float64(benchMark.AllocedBytesPerOp))
			data.DataItems = appendIfMeasured(data.DataItems, benchMark, benchparse.AllocsPerOp, "allocations", "1", float64(benchMark.AllocsPerOp))
			data.DataItems = appendIfMeasured(data.DataItems, benchMark, 0, "iterations", "1", float64(benchMark.N))
		}
	}
	output := &bytes.Buffer{}
	if err := json.NewEncoder(output).Encode(data); err != nil {
		return err
	}
	formatted := &bytes.Buffer{}
	if err := json.Indent(formatted, output.Bytes(), "", "  "); err != nil {
		return err
	}
	return os.WriteFile(os.Args[1], formatted.Bytes(), 0664)
}

func appendIfMeasured(items []DataItem, benchmark *benchparse.Benchmark, metricType int, metricName string, unit string, value float64) []DataItem {
	if metricType != 0 && (benchmark.Measured&metricType) == 0 {
		return items
	}
	return append(items, DataItem{
		Unit: unit,
		Labels: map[string]string{
			"benchmark":  benchmark.Name,
			"metricName": metricName},
		Data: map[string]float64{
			"value": value}})
}
