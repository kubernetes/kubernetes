/*
Copyright 2017 The Kubernetes Authors.

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

package value

import (
	"sync"
	"time"

	"github.com/prometheus/client_golang/prometheus"
)

var (
	transformerLatencies = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Namespace: "apiserver",
			Subsystem: "storage",
			Name:      "transformation_latencies_microseconds",
			Help:      "Latencies in microseconds of value transformation operations.",
			// In-process transformations (ex. AES CBC) complete on the order of 20 microseconds. However, when
			// external KMS is involved latencies may climb into milliseconds.
			Buckets: prometheus.ExponentialBuckets(5, 2, 14),
		},
		[]string{"transformation_type"},
	)
)

var registerMetrics sync.Once

func RegisterMetrics() {
	registerMetrics.Do(func() {
		prometheus.MustRegister(transformerLatencies)
	})
}

func RecordTransformation(transformationType string, start time.Time) {
	since := sinceInMicroseconds(start)
	transformerLatencies.WithLabelValues(transformationType).Observe(float64(since))
}

func sinceInMicroseconds(start time.Time) int64 {
	elapsedNanoseconds := time.Since(start).Nanoseconds()
	return elapsedNanoseconds / int64(time.Microsecond)
}
