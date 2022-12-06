/*
Copyright 2022 The Kubernetes Authors.

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

package metrics

import (
	"time"

	"k8s.io/component-base/metrics"
	"k8s.io/component-base/metrics/legacyregistry"
)

// TODO(jiahuif) CEL is to be used in multiple components, revise naming when that happens.
const (
	namespace = "apiserver"
	subsystem = "cel"
)

// Metrics provides access to CEL metrics.
var Metrics = newCelMetrics()

type CelMetrics struct {
	compilationTime *metrics.Histogram
	evaluationTime  *metrics.Histogram
}

func newCelMetrics() *CelMetrics {
	m := &CelMetrics{
		compilationTime: metrics.NewHistogram(&metrics.HistogramOpts{
			Namespace:      namespace,
			Subsystem:      subsystem,
			Name:           "compilation_duration_seconds",
			StabilityLevel: metrics.ALPHA,
		}),
		evaluationTime: metrics.NewHistogram(&metrics.HistogramOpts{
			Namespace:      namespace,
			Subsystem:      subsystem,
			Name:           "evaluation_duration_seconds",
			StabilityLevel: metrics.ALPHA,
		}),
	}

	legacyregistry.MustRegister(m.compilationTime)
	legacyregistry.MustRegister(m.evaluationTime)

	return m
}

// ObserveCompilation records a CEL compilation with the time the compilation took.
func (m *CelMetrics) ObserveCompilation(elapsed time.Duration) {
	seconds := elapsed.Seconds()
	m.compilationTime.Observe(seconds)
}

// ObserveEvaluation records a CEL evaluation with the time the evaluation took.
func (m *CelMetrics) ObserveEvaluation(elapsed time.Duration) {
	seconds := elapsed.Seconds()
	m.evaluationTime.Observe(seconds)
}
