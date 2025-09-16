/*
Copyright 2025 The Kubernetes Authors.

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
	"sync"

	"k8s.io/component-base/metrics"
	"k8s.io/component-base/metrics/legacyregistry"
)

// controllerSubsystem must be kept in sync with the controller name in cmd/kube-controller-manager/names.
const controllerSubsystem = "device_taint_eviction_controller"

var (
	Global = New()
)

var registerMetrics sync.Once

// Register registers TaintEvictionController metrics.
func Register() {
	registerMetrics.Do(func() {
		legacyregistry.MustRegister(Global.PodDeletionsTotal)
		legacyregistry.MustRegister(Global.PodDeletionsLatency)
	})
}

// New returns new instances of all metrics for testing in parallel.
// Optionally, buckets for the histograms can be specified.
func New(buckets ...float64) Metrics {
	if len(buckets) == 0 {
		buckets = []float64{0.005, 0.025, 0.1, 0.5, 1, 2.5, 10, 30, 60, 120, 180, 240} // 5ms to 4m
	}
	m := Metrics{
		KubeRegistry: metrics.NewKubeRegistry(),
		PodDeletionsTotal: metrics.NewCounter(
			&metrics.CounterOpts{
				Subsystem:      controllerSubsystem,
				Name:           "pod_deletions_total",
				Help:           "Total number of Pods deleted by DeviceTaintEvictionController since its start.",
				StabilityLevel: metrics.ALPHA,
			},
		),
		PodDeletionsLatency: metrics.NewHistogram(
			&metrics.HistogramOpts{
				Subsystem:      controllerSubsystem,
				Name:           "pod_deletion_duration_seconds",
				Help:           "Latency, in seconds, between the time when a device taint effect has been activated and a Pod's deletion via DeviceTaintEvictionController.",
				Buckets:        []float64{0.005, 0.025, 0.1, 0.5, 1, 2.5, 10, 30, 60, 120, 180, 240}, // 5ms to 4m,
				StabilityLevel: metrics.ALPHA,
			},
		),
	}

	// This has to be done after construction, otherwise ./hack/update-generated-stable-metrics.sh
	// fails to find the default buckets.
	if len(buckets) > 0 {
		m.PodDeletionsLatency.HistogramOpts.Buckets = buckets
	}

	m.KubeRegistry.MustRegister(m.PodDeletionsTotal, m.PodDeletionsLatency)
	return m
}

// Metrics contains all metrics supported by the device taint eviction controller.
// It implements [metrics.Gatherer].
type Metrics struct {
	metrics.KubeRegistry
	PodDeletionsTotal   *metrics.Counter
	PodDeletionsLatency *metrics.Histogram
}

var _ metrics.Gatherer = Metrics{}
