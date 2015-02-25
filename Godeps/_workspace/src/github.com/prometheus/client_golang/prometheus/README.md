# Overview
This is the [Prometheus](http://www.prometheus.io) telemetric
instrumentation client [Go](http://golang.org) client library.  It
enable authors to define process-space metrics for their servers and
expose them through a web service interface for extraction,
aggregation, and a whole slew of other post processing techniques.

# Installing
    $ go get github.com/prometheus/client_golang/prometheus

# Example
```go
package main

import (
	"net/http"

	"github.com/prometheus/client_golang/prometheus"
)

var (
	indexed = prometheus.NewCounter(prometheus.CounterOpts{
		Namespace: "my_company",
		Subsystem: "indexer",
		Name:      "documents_indexed",
		Help:      "The number of documents indexed.",
	})
	size = prometheus.NewGauge(prometheus.GaugeOpts{
		Namespace: "my_company",
		Subsystem: "storage",
		Name:      "documents_total_size_bytes",
		Help:      "The total size of all documents in the storage."}})
)
 
func main() {
	http.Handle("/metrics", prometheus.Handler())

	indexed.Inc()
	size.Set(5)

	http.ListenAndServe(":8080", nil)
}

func init() {
	prometheus.MustRegister(indexed)
	prometheus.MustRegister(size)
}
```

# Documentation

[![GoDoc](https://godoc.org/github.com/prometheus/client_golang?status.png)](https://godoc.org/github.com/prometheus/client_golang)

