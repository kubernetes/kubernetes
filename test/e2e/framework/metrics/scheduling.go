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

// SchedulingMetrics is a struct for managing scheduling metrics.
type SchedulingMetrics struct {
	PredicateEvaluationLatency  LatencyMetric `json:"predicateEvaluationLatency"`
	PriorityEvaluationLatency   LatencyMetric `json:"priorityEvaluationLatency"`
	PreemptionEvaluationLatency LatencyMetric `json:"preemptionEvaluationLatency"`
	BindingLatency              LatencyMetric `json:"bindingLatency"`
	ThroughputAverage           float64       `json:"throughputAverage"`
	ThroughputPerc50            float64       `json:"throughputPerc50"`
	ThroughputPerc90            float64       `json:"throughputPerc90"`
	ThroughputPerc99            float64       `json:"throughputPerc99"`
}

// SummaryKind returns the summary of scheduling metrics.
func (l *SchedulingMetrics) SummaryKind() string {
	return "SchedulingMetrics"
}

// PrintHumanReadable returns scheduling metrics with JSON format.
func (l *SchedulingMetrics) PrintHumanReadable() string {
	return PrettyPrintJSON(l)
}

// PrintJSON returns scheduling metrics with JSON format.
func (l *SchedulingMetrics) PrintJSON() string {
	return PrettyPrintJSON(l)
}
