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

package prometheus_test

import (
	"runtime"

	"github.com/golang/protobuf/proto"

	dto "github.com/prometheus/client_model/go"

	"github.com/prometheus/client_golang/prometheus"
)

func NewCallbackMetric(desc *prometheus.Desc, callback func() float64) *CallbackMetric {
	result := &CallbackMetric{desc: desc, callback: callback}
	result.Init(result) // Initialize the SelfCollector.
	return result
}

// TODO: Come up with a better example.

// CallbackMetric is an example for a user-defined Metric that exports the
// result of a function call as a metric of type "untyped" without any
// labels. It uses SelfCollector to turn the Metric into a Collector so that it
// can be registered with Prometheus.
//
// Note that this example is pretty much academic as the prometheus package
// already provides an UntypedFunc type.
type CallbackMetric struct {
	prometheus.SelfCollector

	desc     *prometheus.Desc
	callback func() float64
}

func (cm *CallbackMetric) Desc() *prometheus.Desc {
	return cm.desc
}

func (cm *CallbackMetric) Write(m *dto.Metric) error {
	m.Untyped = &dto.Untyped{Value: proto.Float64(cm.callback())}
	return nil
}

func ExampleSelfCollector() {
	m := NewCallbackMetric(
		prometheus.NewDesc(
			"runtime_goroutines_count",
			"Total number of goroutines that currently exist.",
			nil, nil, // No labels, these must be nil.
		),
		func() float64 {
			return float64(runtime.NumGoroutine())
		},
	)
	prometheus.MustRegister(m)
}
