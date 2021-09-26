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
	"math/rand"
	"time"

	"github.com/prometheus/client_golang/prometheus"
)

var (
	requestDuration = prometheus.NewHistogram(prometheus.HistogramOpts{
		Name:    "example_request_duration_seconds",
		Help:    "Histogram for the runtime of a simple example function.",
		Buckets: prometheus.LinearBuckets(0.01, 0.01, 10),
	})
)

func ExampleTimer() {
	// timer times this example function. It uses a Histogram, but a Summary
	// would also work, as both implement Observer. Check out
	// https://prometheus.io/docs/practices/histograms/ for differences.
	timer := prometheus.NewTimer(requestDuration)
	defer timer.ObserveDuration()

	// Do something here that takes time.
	time.Sleep(time.Duration(rand.NormFloat64()*10000+50000) * time.Microsecond)
}
