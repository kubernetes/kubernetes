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
	"os"

	"github.com/prometheus/client_golang/prometheus"
)

var (
	// If a function is called rarely (i.e. not more often than scrapes
	// happen) or ideally only once (like in a batch job), it can make sense
	// to use a Gauge for timing the function call. For timing a batch job
	// and pushing the result to a Pushgateway, see also the comprehensive
	// example in the push package.
	funcDuration = prometheus.NewGauge(prometheus.GaugeOpts{
		Name: "example_function_duration_seconds",
		Help: "Duration of the last call of an example function.",
	})
)

func run() error {
	// The Set method of the Gauge is used to observe the duration.
	timer := prometheus.NewTimer(prometheus.ObserverFunc(funcDuration.Set))
	defer timer.ObserveDuration()

	// Do something. Return errors as encountered. The use of 'defer' above
	// makes sure the function is still timed properly.
	return nil
}

func ExampleTimer_gauge() {
	if err := run(); err != nil {
		os.Exit(1)
	}
}
