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

var KnownControllerManagerMetrics = map[string][]string{
	"etcd_helper_cache_entry_count":                  {},
	"etcd_helper_cache_hit_count":                    {},
	"etcd_helper_cache_miss_count":                   {},
	"etcd_request_cache_add_latencies_summary":       {"quantile"},
	"etcd_request_cache_add_latencies_summary_count": {},
	"etcd_request_cache_add_latencies_summary_sum":   {},
	"etcd_request_cache_get_latencies_summary":       {"quantile"},
	"etcd_request_cache_get_latencies_summary_count": {},
	"etcd_request_cache_get_latencies_summary_sum":   {},
	"get_token_count":                                {},
	"get_token_fail_count":                           {},
	"rest_client_request_latency_microseconds":       {"url", "verb", "quantile"},
	"rest_client_request_latency_microseconds_count": {"url", "verb"},
	"rest_client_request_latency_microseconds_sum":   {"url", "verb"},
	"rest_client_request_status_codes":               {"method", "code", "host"},
}

type ControllerManagerMetrics Metrics

func (m *ControllerManagerMetrics) Equal(o ControllerManagerMetrics) bool {
	return (*Metrics)(m).Equal(Metrics(o))
}

func NewControllerManagerMetrics() ControllerManagerMetrics {
	result := NewMetrics()
	for metric := range KnownControllerManagerMetrics {
		result[metric] = make(model.Samples, 0)
	}
	return ControllerManagerMetrics(result)
}

func parseControllerManagerMetrics(data string, unknownMetrics sets.String) (ControllerManagerMetrics, error) {
	result := NewControllerManagerMetrics()
	if err := parseMetrics(data, KnownControllerManagerMetrics, (*Metrics)(&result), unknownMetrics); err != nil {
		return ControllerManagerMetrics{}, err
	}
	return result, nil
}
