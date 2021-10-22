# Derived Gauge Example

Table of Contents
=================
- [Summary](#summary)
- [Run the example](#run-the-example)
- [How to use derived gauges?](#how-to-use-derived-gauges-)
  * [Initialize Metric Registry](#initialize-metric-registry)
  * [Create derived gauge metric](#create-derived-gauge-metric)
  * [Create derived gauge entry](#create-derived-gauge-entry)
  * [Implement derived gauge interface](#implement-derived-gauge-interface)
  * [Complete Example](#complete-example)



## Summary 
[top](#Table-of-Contents)

This example demonstrates the use of derived gauges. It is a simple interactive program of consumer
and producer. User can input number of items to produce. Producer produces specified number of
items. Consumer consumes randomly consumes 1-5 items in each attempt. It then sleeps randomly
between 1-10 seconds before the next attempt.

There are two metrics collected to monitor the queue.
1. **queue_size**: It is an instantaneous queue size represented using derived gauge int64.
1. **queue_seconds_since_processed_last**: It is the time elaspsed in seconds since the last time
   when the queue was consumed. It is represented using derived gauge float64.
This example shows how to use gauge metrics. The program records two gauges.

These metrics are read when exporter scrapes them. In this example log exporter is used to
log the data into a file. Metrics can be viewed at [file:///tmp/metrics.log](file:///tmp/metrics.log) 
once the program is running. Alternatively you could do `tail -f /tmp/metrics.log` on Linux/OSx.

Enter different value for number of items to queue and fetch the metrics using above url to see the variation in the metrics.

## Run the example

```
$ go get go.opencensus.io/examples/derived_gauges/...
```

then:

```
$ go run $(go env GOPATH)/src/go.opencensus.io/examples/derived_gauges/derived_gauge.go
```

## How to use derived gauges?

### Initialize Metric Registry
Create a new metric registry for all your metrics.
This step is a general step for any kind of metrics and not specific to gauges.
Register newly created registry with global producer manager.

[embedmd]:# (derived_gauge.go reg)
```go
r := metric.NewRegistry()
metricproducer.GlobalManager().AddProducer(r)
```


### Create derived gauge metric
Create a gauge metric. In this example we have two metrics.

**queue_size**

[embedmd]:# (derived_gauge.go size)
```go
queueSizeGauge, err := r.AddInt64DerivedGauge(
	"queue_size",
	metric.WithDescription("Instantaneous queue size"),
	metric.WithUnit(metricdata.UnitDimensionless))
if err != nil {
	log.Fatalf("error creating queue size derived gauge, error %v\n", err)
}
```

**queue_seconds_since_processed_last**

[embedmd]:# (derived_gauge.go elapsed)
```go
elapsedSeconds, err := r.AddFloat64DerivedGauge(
	"queue_seconds_since_processed_last",
	metric.WithDescription("time elapsed since last time the queue was processed"),
	metric.WithUnit(metricdata.UnitDimensionless))
if err != nil {
	log.Fatalf("error creating queue_seconds_since_processed_last derived gauge, error %v\n", err)
}
```

### Create derived gauge entry
Now, create or insert a unique entry an interface `ToInt64` for a given set of tags. Since we are not using any tags in this example we only insert one entry for each derived gauge metric.

**insert interface for queue_size**

[embedmd]:# (derived_gauge.go entrySize)
```go
err = queueSizeGauge.UpsertEntry(q.Size)
if err != nil {
	log.Fatalf("error getting queue size derived gauge entry, error %v\n", err)
}
```

**insert interface for queue_seconds_since_processed_lasto**

[embedmd]:# (derived_gauge.go entryElapsed)
```go
err = elapsedSeconds.UpsertEntry(q.Elapsed)
if err != nil {
	log.Fatalf("error getting queue_seconds_since_processed_last derived gauge entry, error %v\n", err)
}
```


### Implement derived gauge interface
In order for metrics reader to read the value of your dervied gauge it must
implement ToFloat64 or ToInt64

[embedmd]:# (derived_gauge.go toint64)
```go
func (q *queue) Size() int64 {
	q.mu.Lock()
	defer q.mu.Unlock()
	return int64(q.size)
}

```

[embedmd]:# (derived_gauge.go tofloat64)
```go
func (q *queue) Elapsed() float64 {
	q.mu.Lock()
	defer q.mu.Unlock()
	return time.Since(q.lastConsumed).Seconds()
}

```


### Complete Example

[embedmd]:# (derived_gauge.go entire)
```go

// This example demonstrates the use of derived gauges. It is a simple interactive program of consumer
// and producer. User can input number of items to produce. Producer produces specified number of
// items. Consumer randomly consumes 1-5 items in each attempt. It then sleeps randomly
// between 1-10 seconds before the next attempt. Two metrics collected to monitor the queue.
//
// Metrics
//
// * queue_size: It is an instantaneous queue size represented using derived gauge int64.
//
// * queue_seconds_since_processed_last: It is the time elaspsed in seconds since the last time
// when the queue was consumed. It is represented using derived gauge float64.
package main

import (
	"bufio"
	"fmt"
	"log"
	"math/rand"
	"os"
	"strconv"
	"strings"
	"sync"
	"time"

	"go.opencensus.io/examples/exporter"
	"go.opencensus.io/metric"
	"go.opencensus.io/metric/metricdata"
	"go.opencensus.io/metric/metricproducer"
)

const (
	metricsLogFile = "/tmp/metrics.log"
)

type queue struct {
	size         int
	lastConsumed time.Time

	mu sync.Mutex
	q  []int
}

var q = &queue{}

const (
	maxItemsToConsumePerAttempt = 25
)

func init() {
	q.q = make([]int, 100)
}

// consume randomly dequeues upto 5 items from the queue
func (q *queue) consume() {
	q.mu.Lock()
	defer q.mu.Unlock()

	consumeCount := rand.Int() % maxItemsToConsumePerAttempt
	i := 0
	for i = 0; i < consumeCount; i++ {
		if q.size > 0 {
			q.q = q.q[1:]
			q.size--
		} else {
			break
		}
	}
	if i > 0 {
		q.lastConsumed = time.Now()
	}
}

// produce randomly enqueues upto 5 items from the queue
func (q *queue) produce(count int) {
	q.mu.Lock()
	defer q.mu.Unlock()

	for i := 0; i < count; i++ {
		v := rand.Int() % 100
		q.q = append(q.q, v)
		q.size++
	}
	fmt.Printf("queued %d items, queue size is %d\n", count, q.size)
}

func (q *queue) runConsumer(interval time.Duration, cQuit chan bool) {
	t := time.NewTicker(interval)
	for {
		select {
		case <-t.C:
			q.consume()
		case <-cQuit:
			t.Stop()
			return
		}
	}
}

// Size reports instantaneous queue size.
// This is the interface supplied while creating an entry for derived gauge int64.
func (q *queue) Size() int64 {
	q.mu.Lock()
	defer q.mu.Unlock()
	return int64(q.size)
}


// Elapsed reports time elapsed since the last time an item was consumed from the queue.
// This is the interface supplied while creating an entry for derived gauge float64.
func (q *queue) Elapsed() float64 {
	q.mu.Lock()
	defer q.mu.Unlock()
	return time.Since(q.lastConsumed).Seconds()
}


func getInput() int {
	reader := bufio.NewReader(os.Stdin)
	limit := 100
	for {
		fmt.Printf("Enter number of items to put in consumer queue? [1-%d]: ", limit)
		text, _ := reader.ReadString('\n')
		count, err := strconv.Atoi(strings.TrimSuffix(text, "\n"))
		if err == nil {
			if count < 1 || count > limit {
				fmt.Printf("invalid value %s\n", text)
				continue
			}
			return count
		}
		fmt.Printf("error %v\n", err)
	}
}

func doWork() {
	fmt.Printf("Program monitors queue using two derived gauge metrics.\n")
	fmt.Printf("   1. queue_size = the instantaneous size of the queue.\n")
	fmt.Printf("   2. queue_seconds_since_processed_last = the number of seconds elapsed since last time the queue was processed.\n")
	fmt.Printf("\nGo to file://%s to see the metrics. OR do `tail -f %s` in another terminal\n\n\n",
		metricsLogFile, metricsLogFile)

	// Take a number of items to queue as an input from the user
	// and enqueue the same number of items on to the consumer queue.
	for {
		count := getInput()
		q.produce(count)
		fmt.Printf("press CTRL+C to terminate the program\n")
	}
}

func main() {
	// Using logexporter but you can choose any supported exporter.
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

	// Create Int64DerviedGauge
	queueSizeGauge, err := r.AddInt64DerivedGauge(
		"queue_size",
		metric.WithDescription("Instantaneous queue size"),
		metric.WithUnit(metricdata.UnitDimensionless))
	if err != nil {
		log.Fatalf("error creating queue size derived gauge, error %v\n", err)
	}

	err = queueSizeGauge.UpsertEntry(q.Size)
	if err != nil {
		log.Fatalf("error getting queue size derived gauge entry, error %v\n", err)
	}

	// Create Float64DerviedGauge
	elapsedSeconds, err := r.AddFloat64DerivedGauge(
		"queue_seconds_since_processed_last",
		metric.WithDescription("time elapsed since last time the queue was processed"),
		metric.WithUnit(metricdata.UnitDimensionless))
	if err != nil {
		log.Fatalf("error creating queue_seconds_since_processed_last derived gauge, error %v\n", err)
	}

	err = elapsedSeconds.UpsertEntry(q.Elapsed)
	if err != nil {
		log.Fatalf("error getting queue_seconds_since_processed_last derived gauge entry, error %v\n", err)
	}

	quit := make(chan bool)
	defer func() {
		close(quit)
	}()

	// Run consumer and producer
	go q.runConsumer(5*time.Second, quit)

	for {
		doWork()
	}
}

```
