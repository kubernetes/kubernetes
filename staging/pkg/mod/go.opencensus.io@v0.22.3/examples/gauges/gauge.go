// Copyright 2019, OpenCensus Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// START entire

// This example shows how to use gauge metrics. The program records two gauges, one to demonstrate
// a gauge with int64 value and the other to demonstrate a gauge with float64 value.
//
// Metrics
//
// 1. process_heap_alloc (int64): Total bytes used by objects allocated in the heap.
// It includes objects currently used and objects that are freed but not garbage collected.
//
// 2. process_heap_idle_to_alloc_ratio (float64): It is the ratio of Idle bytes to allocated
// bytes in the heap.
//
// It periodically runs a function that retrieves the memory stats and updates the above two
// metrics. These metrics are then exported using log exporter.
// The program lets you choose the amount of memory (in MB) to consume. Choose different values
// and query the metrics to see the change in metrics.
package main

import (
	"bufio"
	"fmt"
	"log"
	"os"
	"runtime"
	"strconv"
	"strings"
	"time"

	"go.opencensus.io/examples/exporter"
	"go.opencensus.io/metric"
	"go.opencensus.io/metric/metricdata"
	"go.opencensus.io/metric/metricproducer"
)

const (
	metricsLogFile = "/tmp/metrics.log"
)

var (
	mem = &runtime.MemStats{}
)

type memObj struct {
	size int
	b    []byte
}

func newMemObj(size int) *memObj {
	n := &memObj{size: size, b: make([]byte, size)}
	for i := 0; i < n.size; i++ {
		n.b[i] = byte(i)
	}
	return n
}

var allocEntry *metric.Int64GaugeEntry
var ratioEntry *metric.Float64Entry
var arr []*memObj

func getAlloc() uint64 {
	runtime.ReadMemStats(mem)
	return mem.HeapAlloc
}

func getIdleToAllocRatio() float64 {
	runtime.ReadMemStats(mem)
	return float64(mem.HeapIdle) / float64(mem.HeapAlloc)
}

func consumeMem(sizeMB int) {
	arr = make([]*memObj, sizeMB)
	for i := 0; i < sizeMB; i++ {
		arr = append(arr, newMemObj(1000000))
	}
}

func doSomeWork(sizeMB int) {
	// do some work
	consumeMem(sizeMB)
}

func recordMetrics(delay int, done chan int) {
	tick := time.NewTicker(time.Duration(delay) * time.Second)
	for {
		select {
		case <-done:
			return
		case <-tick.C:
			// record heap allocation and idle to allocation ratio.
			// START record
			allocEntry.Set(int64(getAlloc()))     // int64 gauge
			ratioEntry.Set(getIdleToAllocRatio()) // float64 gauge
			// END record
		}
	}
}

func getInput() int {
	reader := bufio.NewReader(os.Stdin)
	limit := 50
	for {
		fmt.Printf("Enter memory (in MB between 1-%d): ", limit)
		text, _ := reader.ReadString('\n')
		sizeMB, err := strconv.Atoi(strings.TrimSuffix(text, "\n"))
		if err == nil {
			if sizeMB < 1 || sizeMB > limit {
				fmt.Printf("invalid value %s\n", text)
				continue
			}
			fmt.Printf("consuming %dMB\n", sizeMB)
			return sizeMB
		}
		fmt.Printf("error %v\n", err)
	}
}

func work() {
	fmt.Printf("Program periodically records following gauge metrics.\n")
	fmt.Printf("   1. process_heap_alloc = the heap allocation (used + freed but not garbage collected)\n")
	fmt.Printf("   2. process_idle_to_alloc_ratio = heap idle (unused) /allocation ratio\n")
	fmt.Printf("\nGo to file://%s to see the metrics. OR do `tail -f %s` in another terminal\n\n\n",
		metricsLogFile, metricsLogFile)
	fmt.Printf("Enter memory you would like to allocate in MB to change the value of above metrics.\n")

	// Do some work and record gauge metrics.
	for {
		sizeMB := getInput()
		doSomeWork(sizeMB)
		fmt.Printf("press CTRL+C to terminate the program\n")
	}
}

func main() {
	// Using log exporter to export metrics but you can choose any supported exporter.
	exporter, err := exporter.NewLogExporter(exporter.Options{
		ReportingInterval: 10 * time.Second,
		MetricsLogFile:    metricsLogFile,
	})
	if err != nil {
		log.Fatalf("Error creating log exporter: %v", err)
	}
	exporter.Start()
	defer exporter.Stop()
	defer exporter.Close()

	// Create metric registry and register it with global producer manager.
	// START reg
	r := metric.NewRegistry()
	metricproducer.GlobalManager().AddProducer(r)
	// END reg

	// Create Int64Gauge to report memory usage of a process.
	// START alloc
	allocGauge, err := r.AddInt64Gauge(
		"process_heap_alloc",
		metric.WithDescription("Process heap allocation"),
		metric.WithUnit(metricdata.UnitBytes))
	if err != nil {
		log.Fatalf("error creating heap allocation gauge, error %v\n", err)
	}
	// END alloc

	// START entryAlloc
	allocEntry, err = allocGauge.GetEntry()
	if err != nil {
		log.Fatalf("error getting heap allocation gauge entry, error %v\n", err)
	}
	// END entryAlloc

	// Create Float64Gauge to report fractional cpu consumed by Garbage Collection.
	// START idle
	ratioGauge, err := r.AddFloat64Gauge(
		"process_heap_idle_to_alloc_ratio",
		metric.WithDescription("process heap idle to allocate ratio"),
		metric.WithUnit(metricdata.UnitDimensionless))
	if err != nil {
		log.Fatalf("error creating process heap idle to allocate ratio gauge, error %v\n", err)
	}
	// END idle

	// START entryIdle
	ratioEntry, err = ratioGauge.GetEntry()
	if err != nil {
		log.Fatalf("error getting process heap idle to allocate ratio gauge entry, error %v\n", err)
	}
	// END entryIdle

	// record gauge metrics every 5 seconds. This example records the gauges periodically. However,
	// depending on the application it can be non-periodic and can be recorded at any time.
	done := make(chan int)
	defer close(done)
	go recordMetrics(1, done)

	// do your work.
	work()

}

// END entire
