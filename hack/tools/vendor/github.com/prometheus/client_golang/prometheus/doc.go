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

// Package prometheus is the core instrumentation package. It provides metrics
// primitives to instrument code for monitoring. It also offers a registry for
// metrics. Sub-packages allow to expose the registered metrics via HTTP
// (package promhttp) or push them to a Pushgateway (package push). There is
// also a sub-package promauto, which provides metrics constructors with
// automatic registration.
//
// All exported functions and methods are safe to be used concurrently unless
// specified otherwise.
//
// A Basic Example
//
// As a starting point, a very basic usage example:
//
//    package main
//
//    import (
//    	"log"
//    	"net/http"
//
//    	"github.com/prometheus/client_golang/prometheus"
//    	"github.com/prometheus/client_golang/prometheus/promhttp"
//    )
//
//    var (
//    	cpuTemp = prometheus.NewGauge(prometheus.GaugeOpts{
//    		Name: "cpu_temperature_celsius",
//    		Help: "Current temperature of the CPU.",
//    	})
//    	hdFailures = prometheus.NewCounterVec(
//    		prometheus.CounterOpts{
//    			Name: "hd_errors_total",
//    			Help: "Number of hard-disk errors.",
//    		},
//    		[]string{"device"},
//    	)
//    )
//
//    func init() {
//    	// Metrics have to be registered to be exposed:
//    	prometheus.MustRegister(cpuTemp)
//    	prometheus.MustRegister(hdFailures)
//    }
//
//    func main() {
//    	cpuTemp.Set(65.3)
//    	hdFailures.With(prometheus.Labels{"device":"/dev/sda"}).Inc()
//
//    	// The Handler function provides a default handler to expose metrics
//    	// via an HTTP server. "/metrics" is the usual endpoint for that.
//    	http.Handle("/metrics", promhttp.Handler())
//    	log.Fatal(http.ListenAndServe(":8080", nil))
//    }
//
//
// This is a complete program that exports two metrics, a Gauge and a Counter,
// the latter with a label attached to turn it into a (one-dimensional) vector.
//
// Metrics
//
// The number of exported identifiers in this package might appear a bit
// overwhelming. However, in addition to the basic plumbing shown in the example
// above, you only need to understand the different metric types and their
// vector versions for basic usage. Furthermore, if you are not concerned with
// fine-grained control of when and how to register metrics with the registry,
// have a look at the promauto package, which will effectively allow you to
// ignore registration altogether in simple cases.
//
// Above, you have already touched the Counter and the Gauge. There are two more
// advanced metric types: the Summary and Histogram. A more thorough description
// of those four metric types can be found in the Prometheus docs:
// https://prometheus.io/docs/concepts/metric_types/
//
// In addition to the fundamental metric types Gauge, Counter, Summary, and
// Histogram, a very important part of the Prometheus data model is the
// partitioning of samples along dimensions called labels, which results in
// metric vectors. The fundamental types are GaugeVec, CounterVec, SummaryVec,
// and HistogramVec.
//
// While only the fundamental metric types implement the Metric interface, both
// the metrics and their vector versions implement the Collector interface. A
// Collector manages the collection of a number of Metrics, but for convenience,
// a Metric can also “collect itself”. Note that Gauge, Counter, Summary, and
// Histogram are interfaces themselves while GaugeVec, CounterVec, SummaryVec,
// and HistogramVec are not.
//
// To create instances of Metrics and their vector versions, you need a suitable
// …Opts struct, i.e. GaugeOpts, CounterOpts, SummaryOpts, or HistogramOpts.
//
// Custom Collectors and constant Metrics
//
// While you could create your own implementations of Metric, most likely you
// will only ever implement the Collector interface on your own. At a first
// glance, a custom Collector seems handy to bundle Metrics for common
// registration (with the prime example of the different metric vectors above,
// which bundle all the metrics of the same name but with different labels).
//
// There is a more involved use case, too: If you already have metrics
// available, created outside of the Prometheus context, you don't need the
// interface of the various Metric types. You essentially want to mirror the
// existing numbers into Prometheus Metrics during collection. An own
// implementation of the Collector interface is perfect for that. You can create
// Metric instances “on the fly” using NewConstMetric, NewConstHistogram, and
// NewConstSummary (and their respective Must… versions). NewConstMetric is used
// for all metric types with just a float64 as their value: Counter, Gauge, and
// a special “type” called Untyped. Use the latter if you are not sure if the
// mirrored metric is a Counter or a Gauge. Creation of the Metric instance
// happens in the Collect method. The Describe method has to return separate
// Desc instances, representative of the “throw-away” metrics to be created
// later.  NewDesc comes in handy to create those Desc instances. Alternatively,
// you could return no Desc at all, which will mark the Collector “unchecked”.
// No checks are performed at registration time, but metric consistency will
// still be ensured at scrape time, i.e. any inconsistencies will lead to scrape
// errors. Thus, with unchecked Collectors, the responsibility to not collect
// metrics that lead to inconsistencies in the total scrape result lies with the
// implementer of the Collector. While this is not a desirable state, it is
// sometimes necessary. The typical use case is a situation where the exact
// metrics to be returned by a Collector cannot be predicted at registration
// time, but the implementer has sufficient knowledge of the whole system to
// guarantee metric consistency.
//
// The Collector example illustrates the use case. You can also look at the
// source code of the processCollector (mirroring process metrics), the
// goCollector (mirroring Go metrics), or the expvarCollector (mirroring expvar
// metrics) as examples that are used in this package itself.
//
// If you just need to call a function to get a single float value to collect as
// a metric, GaugeFunc, CounterFunc, or UntypedFunc might be interesting
// shortcuts.
//
// Advanced Uses of the Registry
//
// While MustRegister is the by far most common way of registering a Collector,
// sometimes you might want to handle the errors the registration might cause.
// As suggested by the name, MustRegister panics if an error occurs. With the
// Register function, the error is returned and can be handled.
//
// An error is returned if the registered Collector is incompatible or
// inconsistent with already registered metrics. The registry aims for
// consistency of the collected metrics according to the Prometheus data model.
// Inconsistencies are ideally detected at registration time, not at collect
// time. The former will usually be detected at start-up time of a program,
// while the latter will only happen at scrape time, possibly not even on the
// first scrape if the inconsistency only becomes relevant later. That is the
// main reason why a Collector and a Metric have to describe themselves to the
// registry.
//
// So far, everything we did operated on the so-called default registry, as it
// can be found in the global DefaultRegisterer variable. With NewRegistry, you
// can create a custom registry, or you can even implement the Registerer or
// Gatherer interfaces yourself. The methods Register and Unregister work in the
// same way on a custom registry as the global functions Register and Unregister
// on the default registry.
//
// There are a number of uses for custom registries: You can use registries with
// special properties, see NewPedanticRegistry. You can avoid global state, as
// it is imposed by the DefaultRegisterer. You can use multiple registries at
// the same time to expose different metrics in different ways.  You can use
// separate registries for testing purposes.
//
// Also note that the DefaultRegisterer comes registered with a Collector for Go
// runtime metrics (via NewGoCollector) and a Collector for process metrics (via
// NewProcessCollector). With a custom registry, you are in control and decide
// yourself about the Collectors to register.
//
// HTTP Exposition
//
// The Registry implements the Gatherer interface. The caller of the Gather
// method can then expose the gathered metrics in some way. Usually, the metrics
// are served via HTTP on the /metrics endpoint. That's happening in the example
// above. The tools to expose metrics via HTTP are in the promhttp sub-package.
//
// Pushing to the Pushgateway
//
// Function for pushing to the Pushgateway can be found in the push sub-package.
//
// Graphite Bridge
//
// Functions and examples to push metrics from a Gatherer to Graphite can be
// found in the graphite sub-package.
//
// Other Means of Exposition
//
// More ways of exposing metrics can easily be added by following the approaches
// of the existing implementations.
package prometheus
