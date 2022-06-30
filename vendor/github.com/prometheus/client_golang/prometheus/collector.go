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

package prometheus

// Collector is the interface implemented by anything that can be used by
// Prometheus to collect metrics. A Collector has to be registered for
// collection. See Registerer.Register.
//
// The stock metrics provided by this package (Gauge, Counter, Summary,
// Histogram, Untyped) are also Collectors (which only ever collect one metric,
// namely itself). An implementer of Collector may, however, collect multiple
// metrics in a coordinated fashion and/or create metrics on the fly. Examples
// for collectors already implemented in this library are the metric vectors
// (i.e. collection of multiple instances of the same Metric but with different
// label values) like GaugeVec or SummaryVec, and the ExpvarCollector.
type Collector interface {
	// Describe sends the super-set of all possible descriptors of metrics
	// collected by this Collector to the provided channel and returns once
	// the last descriptor has been sent. The sent descriptors fulfill the
	// consistency and uniqueness requirements described in the Desc
	// documentation.
	//
	// It is valid if one and the same Collector sends duplicate
	// descriptors. Those duplicates are simply ignored. However, two
	// different Collectors must not send duplicate descriptors.
	//
	// Sending no descriptor at all marks the Collector as “unchecked”,
	// i.e. no checks will be performed at registration time, and the
	// Collector may yield any Metric it sees fit in its Collect method.
	//
	// This method idempotently sends the same descriptors throughout the
	// lifetime of the Collector. It may be called concurrently and
	// therefore must be implemented in a concurrency safe way.
	//
	// If a Collector encounters an error while executing this method, it
	// must send an invalid descriptor (created with NewInvalidDesc) to
	// signal the error to the registry.
	Describe(chan<- *Desc)
	// Collect is called by the Prometheus registry when collecting
	// metrics. The implementation sends each collected metric via the
	// provided channel and returns once the last metric has been sent. The
	// descriptor of each sent metric is one of those returned by Describe
	// (unless the Collector is unchecked, see above). Returned metrics that
	// share the same descriptor must differ in their variable label
	// values.
	//
	// This method may be called concurrently and must therefore be
	// implemented in a concurrency safe way. Blocking occurs at the expense
	// of total performance of rendering all registered metrics. Ideally,
	// Collector implementations support concurrent readers.
	Collect(chan<- Metric)
}

// DescribeByCollect is a helper to implement the Describe method of a custom
// Collector. It collects the metrics from the provided Collector and sends
// their descriptors to the provided channel.
//
// If a Collector collects the same metrics throughout its lifetime, its
// Describe method can simply be implemented as:
//
//   func (c customCollector) Describe(ch chan<- *Desc) {
//   	DescribeByCollect(c, ch)
//   }
//
// However, this will not work if the metrics collected change dynamically over
// the lifetime of the Collector in a way that their combined set of descriptors
// changes as well. The shortcut implementation will then violate the contract
// of the Describe method. If a Collector sometimes collects no metrics at all
// (for example vectors like CounterVec, GaugeVec, etc., which only collect
// metrics after a metric with a fully specified label set has been accessed),
// it might even get registered as an unchecked Collector (cf. the Register
// method of the Registerer interface). Hence, only use this shortcut
// implementation of Describe if you are certain to fulfill the contract.
//
// The Collector example demonstrates a use of DescribeByCollect.
func DescribeByCollect(c Collector, descs chan<- *Desc) {
	metrics := make(chan Metric)
	go func() {
		c.Collect(metrics)
		close(metrics)
	}()
	for m := range metrics {
		descs <- m.Desc()
	}
}

// selfCollector implements Collector for a single Metric so that the Metric
// collects itself. Add it as an anonymous field to a struct that implements
// Metric, and call init with the Metric itself as an argument.
type selfCollector struct {
	self Metric
}

// init provides the selfCollector with a reference to the metric it is supposed
// to collect. It is usually called within the factory function to create a
// metric. See example.
func (c *selfCollector) init(self Metric) {
	c.self = self
}

// Describe implements Collector.
func (c *selfCollector) Describe(ch chan<- *Desc) {
	ch <- c.self.Desc()
}

// Collect implements Collector.
func (c *selfCollector) Collect(ch chan<- Metric) {
	ch <- c.self
}

// collectorMetric is a metric that is also a collector.
// Because of selfCollector, most (if not all) Metrics in
// this package are also collectors.
type collectorMetric interface {
	Metric
	Collector
}
