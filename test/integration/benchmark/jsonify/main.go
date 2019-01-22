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
	"io/ioutil"
	"os"

	benchparse "golang.org/x/tools/benchmark/parse"
	"k8s.io/kubernetes/test/e2e/perftype"
)

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
	data := perftype.PerfData{Version: "v1"}
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
	return ioutil.WriteFile(os.Args[1], formatted.Bytes(), 0664)
}

func appendIfMeasured(items []perftype.DataItem, benchmark *benchparse.Benchmark, metricType int, metricName string, unit string, value float64) []perftype.DataItem {
	if metricType != 0 && (benchmark.Measured&metricType) == 0 {
		return items
	}
	return append(items, perftype.DataItem{
		Unit: unit,
		Labels: map[string]string{
			"benchmark":  benchmark.Name,
			"metricName": metricName},
		Data: map[string]float64{
			"value": value}})
}
