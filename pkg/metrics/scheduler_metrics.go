/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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
	"k8s.io/kubernetes/pkg/util/sets"

	"github.com/prometheus/common/model"
)

var KnownSchedulerMetrics = map[string][]string{
	"rest_client_request_latency_microseconds":                   {"url", "verb", "quantile"},
	"rest_client_request_latency_microseconds_count":             {"url", "verb"},
	"rest_client_request_latency_microseconds_sum":               {"url", "verb"},
	"rest_client_request_status_codes":                           {"code", "host", "method"},
	"scheduler_binding_latency_microseconds_bucket":              {"le"},
	"scheduler_binding_latency_microseconds_count":               {},
	"scheduler_binding_latency_microseconds_sum":                 {},
	"scheduler_e2e_scheduling_latency_microseconds_bucket":       {"le"},
	"scheduler_e2e_scheduling_latency_microseconds_count":        {},
	"scheduler_e2e_scheduling_latency_microseconds_sum":          {},
	"scheduler_scheduling_algorithm_latency_microseconds_bucket": {"le"},
	"scheduler_scheduling_algorithm_latency_microseconds_count":  {},
	"scheduler_scheduling_algorithm_latency_microseconds_sum":    {},
}

type SchedulerMetrics Metrics

func (m *SchedulerMetrics) Equal(o SchedulerMetrics) bool {
	return (*Metrics)(m).Equal(Metrics(o))
}

func NewSchedulerMetrics() SchedulerMetrics {
	result := NewMetrics()
	for metric := range KnownSchedulerMetrics {
		result[metric] = make(model.Samples, 0)
	}
	return SchedulerMetrics(result)
}

func parseSchedulerMetrics(data string, unknownMetrics sets.String) (SchedulerMetrics, error) {
	result := NewSchedulerMetrics()
	if err := parseMetrics(data, KnownSchedulerMetrics, (*Metrics)(&result), unknownMetrics); err != nil {
		return SchedulerMetrics{}, err
	}
	return result, nil
}
