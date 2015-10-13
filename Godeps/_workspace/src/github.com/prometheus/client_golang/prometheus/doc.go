// Copyright 2014 The Prometheus Authors
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Package prometheus provides embeddable metric primitives for servers and
// standardized exposition of telemetry through a web services interface.
//
// All exported functions and methods are safe to be used concurrently unless
// specified otherwise.
//
// To expose metrics registered with the Prometheus registry, an HTTP server
// needs to know about the Prometheus handler. The usual endpoint is "/metrics".
//
//     http.Handle("/metrics", prometheus.Handler())
//
// As a starting point a very basic usage example:
//
//    package main
//
//    import (
//    	"net/http"
//
//    	"github.com/prometheus/client_golang/prometheus"
//    )
//
//    var (
//    	cpuTemp = prometheus.NewGauge(prometheus.GaugeOpts{
//    		Name: "cpu_temperature_celsius",
//    		Help: "Current temperature of the CPU.",
//    	})
//    	hdFailures = prometheus.NewCounter(prometheus.CounterOpts{
//    		Name: "hd_errors_total",
//    		Help: "Number of hard-disk errors.",
//    	})
//    )
//
//    func init() {
//    	prometheus.MustRegister(cpuTemp)
//    	prometheus.MustRegister(hdFailures)
//    }
//
//    func main() {
//    	cpuTemp.Set(65.3)
//    	hdFailures.Inc()
//
//    	http.Handle("/metrics", prometheus.Handler())
//    	http.ListenAndServe(":8080", nil)
//    }
//
//
// This is a complete program that exports two metrics, a Gauge and a Counter.
// It also exports some stats about the HTTP usage of the /metrics
// endpoint. (See the Handler function for more detail.)
//
// Two more advanced metric types are the Summary and Histogram.
//
// In addition to the fundamental metric types Gauge, Counter, Summary, and
// Histogram, a very important part of the Prometheus data model is the
// partitioning of samples along dimensions called labels, which results in
// metric vectors. The fundamental types are GaugeVec, CounterVec, SummaryVec,
// and HistogramVec.
//
// Those are all the parts needed for basic usage. Detailed documentation and
// examples are provided below.
//
// Everything else this package offers is essentially for "power users" only. A
// few pointers to "power user features":
//
// All the various ...Opts structs have a ConstLabels field for labels that
// never change their value (which is only useful under special circumstances,
// see documentation of the Opts type).
//
// The Untyped metric behaves like a Gauge, but signals the Prometheus server
// not to assume anything about its type.
//
// Functions to fine-tune how the metric registry works: EnableCollectChecks,
// PanicOnCollectError, Register, Unregister, SetMetricFamilyInjectionHook.
//
// For custom metric collection, there are two entry points: Custom Metric
// implementations and custom Collector implementations. A Metric is the
// fundamental unit in the Prometheus data model: a sample at a point in time
// together with its meta-data (like its fully-qualified name and any number of
// pairs of label name and label value) that knows how to marshal itself into a
// data transfer object (aka DTO, implemented as a protocol buffer). A Collector
// gets registered with the Prometheus registry and manages the collection of
// one or more Metrics. Many parts of this package are building blocks for
// Metrics and Collectors. Desc is the metric descriptor, actually used by all
// metrics under the hood, and by Collectors to describe the Metrics to be
// collected, but only to be dealt with by users if they implement their own
// Metrics or Collectors. To create a Desc, the BuildFQName function will come
// in handy. Other useful components for Metric and Collector implementation
// include: LabelPairSorter to sort the DTO version of label pairs,
// NewConstMetric and MustNewConstMetric to create "throw away" Metrics at
// collection time, MetricVec to bundle custom Metrics into a metric vector
// Collector, SelfCollector to make a custom Metric collect itself.
//
// A good example for a custom Collector is the ExpVarCollector included in this
// package, which exports variables exported via the "expvar" package as
// Prometheus metrics.
package prometheus
