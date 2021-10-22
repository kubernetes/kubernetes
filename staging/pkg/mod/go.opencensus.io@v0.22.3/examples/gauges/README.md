# Gauges Example


Table of Contents
=================

- [Summary](#summary)
- [Run the example](#run-the-example)
- [How to use gauges?](#how-to-use-gauges-)
  * [Initialize Metric Registry](#initialize-metric-registry)
  * [Create gauge metric](#create-gauge-metric)
  * [Create gauge entry](#create-gauge-entry)
  * [Set gauge values](#set-gauge-values)
  * [Complete Example](#complete-example)

## Summary 
[top](#Table-of-Contents)

This example shows how to use gauge metrics. The program records two gauges.

1. **process_heap_alloc (int64)**: Total bytes used by objects allocated in the heap. It includes objects currently used and objects that are freed but not garbage collected.
1. **process_heap_idle_to_alloc_ratio (float64)**: It is the ratio of Idle bytes to allocated bytes in the heap.

It periodically runs a function that retrieves the memory stats and updates the above two metrics.
These metrics are then exported using log exporter. Metrics can be viewed at
[file:///tmp/metrics.log](file:///tmp/metrics.log) 
once the program is running. Alternatively you could do `tail -f /tmp/metrics.log` on Linux/OSx.

The program lets you choose the amount of memory (in MB) to consume. Choose different values and query the metrics to see the change in metrics.

## Run the example

```
$ go get go.opencensus.io/examples/gauges/...
```

then:

```
$ go run $(go env GOPATH)/src/go.opencensus.io/examples/gauges/gauge.go
```

## How to use gauges?

### Initialize Metric Registry
Create a new metric registry for all your metrics.
This step is a general step for any kind of metrics and not specific to gauges.
Register newly created registry with global producer manager.

[embedmd]:# (gauge.go reg)
```go
r := metric.NewRegistry()
metricproducer.GlobalManager().AddProducer(r)
```


### Create gauge metric
Create a gauge metric. In this example we have two metrics.

**process_heap_alloc**

[embedmd]:# (gauge.go alloc)
```go
allocGauge, err := r.AddInt64Gauge(
	"process_heap_alloc",
	metric.WithDescription("Process heap allocation"),
	metric.WithUnit(metricdata.UnitBytes))
if err != nil {
	log.Fatalf("error creating heap allocation gauge, error %v\n", err)
}
```

**process_heap_idle_to_alloc_ratio**

[embedmd]:# (gauge.go idle)
```go
ratioGauge, err := r.AddFloat64Gauge(
	"process_heap_idle_to_alloc_ratio",
	metric.WithDescription("process heap idle to allocate ratio"),
	metric.WithUnit(metricdata.UnitDimensionless))
if err != nil {
	log.Fatalf("error creating process heap idle to allocate ratio gauge, error %v\n", err)
}
```

### Create gauge entry
Now, create or get a unique entry (equivalent of a row in a table) for a given set of tags. Since we are not using any tags in this example we only have one entry for each gauge metric.

**entry for process_heap_alloc**

[embedmd]:# (gauge.go entryAlloc)
```go
allocEntry, err = allocGauge.GetEntry()
if err != nil {
	log.Fatalf("error getting heap allocation gauge entry, error %v\n", err)
}
```

**entry for process_heap_idle_to_alloc_ratio**

[embedmd]:# (gauge.go entryIdle)
```go
ratioEntry, err = ratioGauge.GetEntry()
if err != nil {
	log.Fatalf("error getting process heap idle to allocate ratio gauge entry, error %v\n", err)
}
```


### Set gauge values
Use `Set` or `Add` function to update the value of gauge entries. You can call these methods anytime based on your metric and your application. In this example, `Set` is called periodically.

[embedmd]:# (gauge.go record)
```go
			allocEntry.Set(int64(getAlloc()))     // int64 gauge
			ratioEntry.Set(getIdleToAllocRatio()) // float64 gauge
```

### Complete Example

[embedmd]:# (gauge.go entire)
```go

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
			allocEntry.Set(int64(getAlloc()))     // int64 gauge
			ratioEntry.Set(getIdleToAllocRatio()) // float64 gauge
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
	r := metric.NewRegistry()
	metricproducer.GlobalManager().AddProducer(r)

	// Create Int64Gauge to report memory usage of a process.
	allocGauge, err := r.AddInt64Gauge(
		"process_heap_alloc",
		metric.WithDescription("Process heap allocation"),
		metric.WithUnit(metricdata.UnitBytes))
	if err != nil {
		log.Fatalf("error creating heap allocation gauge, error %v\n", err)
	}

	allocEntry, err = allocGauge.GetEntry()
	if err != nil {
		log.Fatalf("error getting heap allocation gauge entry, error %v\n", err)
	}

	// Create Float64Gauge to report fractional cpu consumed by Garbage Collection.
	ratioGauge, err := r.AddFloat64Gauge(
		"process_heap_idle_to_alloc_ratio",
		metric.WithDescription("process heap idle to allocate ratio"),
		metric.WithUnit(metricdata.UnitDimensionless))
	if err != nil {
		log.Fatalf("error creating process heap idle to allocate ratio gauge, error %v\n", err)
	}

	ratioEntry, err = ratioGauge.GetEntry()
	if err != nil {
		log.Fatalf("error getting process heap idle to allocate ratio gauge entry, error %v\n", err)
	}

	// record gauge metrics every 5 seconds. This example records the gauges periodically. However,
	// depending on the application it can be non-periodic and can be recorded at any time.
	done := make(chan int)
	defer close(done)
	go recordMetrics(1, done)

	// do your work.
	work()

}

```
