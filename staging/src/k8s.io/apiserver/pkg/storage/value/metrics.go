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

const (
	valueSubsystem = "value"
)

var (
	TransformerOperationalLatencies = prometheus.NewSummaryVec(
		prometheus.SummaryOpts{
			Subsystem: valueSubsystem,
			Name:      "apiserver_storage_transformation_latency_microseconds",
			Help:      "Latency in microseconds of value transformation operations.",
		},
		[]string{"transformation_type"},
	)
)

var registerMetrics sync.Once

func RegisterMetrics() {
	registerMetrics.Do(func() {
		prometheus.MustRegister(TransformerOperationalLatencies)
	})
}

func RecordTransformation(transformationType string, start time.Time) {
	TransformerOperationalLatencies.WithLabelValues(transformationType).Observe(float64(sinceInMicroseconds(start)))
}

func sinceInMicroseconds(start time.Time) time.Duration {
	return time.Since(start) / time.Microsecond
}
