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

package metrics

import (
	"k8s.io/component-base/metrics"
)

// MetricRecorder represents a metric recorder which takes action when the
// metric Inc(), Dec() and Clear()
type MetricRecorder interface {
	Inc()
	Dec()
	Clear()
}

var _ MetricRecorder = &PendingPodsRecorder{}

// PendingPodsRecorder is an implementation of MetricRecorder
type PendingPodsRecorder struct {
	recorder metrics.GaugeMetric
}

// NewActivePodsRecorder returns ActivePods in a Prometheus metric fashion
func NewActivePodsRecorder() *PendingPodsRecorder {
	return &PendingPodsRecorder{
		recorder: ActivePods(),
	}
}

// NewUnschedulablePodsRecorder returns UnschedulablePods in a Prometheus metric fashion
func NewUnschedulablePodsRecorder() *PendingPodsRecorder {
	return &PendingPodsRecorder{
		recorder: UnschedulablePods(),
	}
}

// NewBackoffPodsRecorder returns BackoffPods in a Prometheus metric fashion
func NewBackoffPodsRecorder() *PendingPodsRecorder {
	return &PendingPodsRecorder{
		recorder: BackoffPods(),
	}
}

// Inc increases a metric counter by 1, in an atomic way
func (r *PendingPodsRecorder) Inc() {
	r.recorder.Inc()
}

// Dec decreases a metric counter by 1, in an atomic way
func (r *PendingPodsRecorder) Dec() {
	r.recorder.Dec()
}

// Clear set a metric counter to 0, in an atomic way
func (r *PendingPodsRecorder) Clear() {
	r.recorder.Set(float64(0))
}
