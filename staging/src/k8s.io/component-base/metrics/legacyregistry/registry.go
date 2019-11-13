/*
Copyright 2019 The Kubernetes Authors.

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

package legacyregistry

import (
	"net/http"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promhttp"

	"k8s.io/component-base/metrics"
)

var (
	defaultRegistry = metrics.NewKubeRegistry()
	// DefaultGatherer exposes the global registry gatherer
	DefaultGatherer metrics.Gatherer = defaultRegistry
)

func init() {
	RawMustRegister(prometheus.NewProcessCollector(prometheus.ProcessCollectorOpts{}))
	RawMustRegister(prometheus.NewGoCollector())
}

// Handler returns an HTTP handler for the DefaultGatherer. It is
// already instrumented with InstrumentHandler (using "prometheus" as handler
// name).
//
// Deprecated: Please note the issues described in the doc comment of
// InstrumentHandler. You might want to consider using promhttp.Handler instead.
func Handler() http.Handler {
	return promhttp.InstrumentMetricHandler(prometheus.DefaultRegisterer, promhttp.HandlerFor(defaultRegistry, promhttp.HandlerOpts{}))
}

// Register registers a collectable metric but uses the global registry
func Register(c metrics.Registerable) error {
	err := defaultRegistry.Register(c)
	return err
}

// MustRegister registers registerable metrics but uses the global registry.
func MustRegister(cs ...metrics.Registerable) {
	defaultRegistry.MustRegister(cs...)
}

// RawMustRegister registers prometheus collectors but uses the global registry, this
// bypasses the metric stability framework
//
// Deprecated
func RawMustRegister(cs ...prometheus.Collector) {
	defaultRegistry.RawMustRegister(cs...)
}

// RawRegister registers a prometheus collector but uses the global registry, this
// bypasses the metric stability framework
//
// Deprecated
func RawRegister(c prometheus.Collector) error {
	err := defaultRegistry.RawRegister(c)
	return err
}

// CustomRegister registers a custom collector but uses the global registry.
func CustomRegister(c metrics.StableCollector) error {
	err := defaultRegistry.CustomRegister(c)

	//TODO(RainbowMango): Maybe we can wrap this error by error wrapping.(Golang 1.13)
	_ = prometheus.Register(c)

	return err
}

// CustomMustRegister registers custom collectors but uses the global registry.
func CustomMustRegister(cs ...metrics.StableCollector) {
	defaultRegistry.CustomMustRegister(cs...)

	for _, c := range cs {
		prometheus.MustRegister(c)
	}
}
