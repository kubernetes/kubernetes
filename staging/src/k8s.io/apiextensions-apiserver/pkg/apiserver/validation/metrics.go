/*
Copyright 2023 The Kubernetes Authors.

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

package validation

import (
	"time"

	"k8s.io/component-base/metrics"
	"k8s.io/component-base/metrics/legacyregistry"
)

const (
	namespace = "apiextensions_apiserver"
	subsystem = "validation"
)

// Interface to stub for tests
type ValidationMetrics interface {
	ObserveRatchetingTime(d time.Duration)
}

var Metrics ValidationMetrics = &validationMetrics{
	RatchetingTime: metrics.NewHistogram(&metrics.HistogramOpts{
		Namespace:      namespace,
		Subsystem:      subsystem,
		Name:           "ratcheting_seconds",
		Help:           "Time for comparison of old to new for the purposes of CRDValidationRatcheting during an UPDATE in seconds.",
		StabilityLevel: metrics.ALPHA,
		// Start 0.01ms with the last bucket being [~2.5s, +Inf)
		Buckets: metrics.ExponentialBuckets(0.00001, 4, 10),
	}),
}

func init() {
	legacyregistry.MustRegister(Metrics.(*validationMetrics).RatchetingTime)
}

type validationMetrics struct {
	RatchetingTime *metrics.Histogram
}

// ObserveRatchetingTime records the time spent on ratcheting
func (m *validationMetrics) ObserveRatchetingTime(d time.Duration) {
	m.RatchetingTime.Observe(d.Seconds())
}

// Reset resets the metrics. This is meant to be used for testing. Panics
// if the metrics cannot be re-registered. Returns all the reset metrics
func (m *validationMetrics) Reset() []metrics.Registerable {
	m.RatchetingTime = metrics.NewHistogram(m.RatchetingTime.HistogramOpts)
	return []metrics.Registerable{m.RatchetingTime}
}
