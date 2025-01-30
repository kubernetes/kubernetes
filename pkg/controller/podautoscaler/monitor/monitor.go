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

package monitor

import (
	"time"

	v2 "k8s.io/api/autoscaling/v2"
)

type ActionLabel string
type ErrorLabel string

const (
	ActionLabelScaleUp   ActionLabel = "scale_up"
	ActionLabelScaleDown ActionLabel = "scale_down"
	ActionLabelNone      ActionLabel = "none"

	// ErrorLabelSpec represents an error due to an invalid spec of HPA object.
	ErrorLabelSpec ErrorLabel = "spec"
	// ErrorLabelInternal represents an error from an internal computation or communication with other component.
	ErrorLabelInternal ErrorLabel = "internal"
	ErrorLabelNone     ErrorLabel = "none"
)

// Monitor records some metrics so that people can monitor HPA controller.
type Monitor interface {
	ObserveReconciliationResult(action ActionLabel, err ErrorLabel, duration time.Duration)
	ObserveMetricComputationResult(action ActionLabel, err ErrorLabel, duration time.Duration, metricType v2.MetricSourceType)
}

type monitor struct{}

func New() Monitor {
	return &monitor{}
}

// ObserveReconciliationResult observes some metrics from a reconciliation result.
func (r *monitor) ObserveReconciliationResult(action ActionLabel, err ErrorLabel, duration time.Duration) {
	reconciliationsTotal.WithLabelValues(string(action), string(err)).Inc()
	reconciliationsDuration.WithLabelValues(string(action), string(err)).Observe(duration.Seconds())
}

// ObserveMetricComputationResult observes some metrics from a metric computation result.
func (r *monitor) ObserveMetricComputationResult(action ActionLabel, err ErrorLabel, duration time.Duration, metricType v2.MetricSourceType) {
	metricComputationTotal.WithLabelValues(string(action), string(err), string(metricType)).Inc()
	metricComputationDuration.WithLabelValues(string(action), string(err), string(metricType)).Observe(duration.Seconds())
}
