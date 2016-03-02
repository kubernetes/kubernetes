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

var KnownApiServerMetrics = map[string][]string{
	"apiserver_request_count":                        {"verb", "resource", "client", "code"},
	"apiserver_request_latencies_bucket":             {"verb", "resource", "le"},
	"apiserver_request_latencies_count":              {"verb", "resource"},
	"apiserver_request_latencies_sum":                {"verb", "resource"},
	"apiserver_request_latencies_summary":            {"verb", "resource", "quantile"},
	"apiserver_request_latencies_summary_count":      {"verb", "resource"},
	"apiserver_request_latencies_summary_sum":        {"verb", "resource"},
	"etcd_helper_cache_entry_count":                  {},
	"etcd_helper_cache_hit_count":                    {},
	"etcd_helper_cache_miss_count":                   {},
	"etcd_request_cache_add_latencies_summary":       {"quantile"},
	"etcd_request_cache_add_latencies_summary_count": {},
	"etcd_request_cache_add_latencies_summary_sum":   {},
	"etcd_request_cache_get_latencies_summary":       {"quantile"},
	"etcd_request_cache_get_latencies_summary_count": {},
	"etcd_request_cache_get_latencies_summary_sum":   {},
	"etcd_request_latencies_summary":                 {"operation", "type", "quantile"},
	"etcd_request_latencies_summary_count":           {"operation", "type"},
	"etcd_request_latencies_summary_sum":             {"operation", "type"},
	"rest_client_request_latency_microseconds":       {"url", "verb", "quantile"},
	"rest_client_request_latency_microseconds_count": {"url", "verb"},
	"rest_client_request_latency_microseconds_sum":   {"url", "verb"},
	"rest_client_request_status_codes":               {"code", "host", "method"},
}

type ApiServerMetrics Metrics

func (m *ApiServerMetrics) Equal(o ApiServerMetrics) bool {
	return (*Metrics)(m).Equal(Metrics(o))
}

func NewApiServerMetrics() ApiServerMetrics {
	result := NewMetrics()
	for metric := range KnownApiServerMetrics {
		result[metric] = make(model.Samples, 0)
	}
	return ApiServerMetrics(result)
}

func parseApiServerMetrics(data string, unknownMetrics sets.String) (ApiServerMetrics, error) {
	result := NewApiServerMetrics()
	if err := parseMetrics(data, KnownApiServerMetrics, (*Metrics)(&result), unknownMetrics); err != nil {
		return ApiServerMetrics{}, err
	}
	return result, nil
}

func (g *MetricsGrabber) getMetricsFromApiServer() (string, error) {
	rawOutput, err := g.client.Get().RequestURI("/metrics").Do().Raw()
	if err != nil {
		return "", err
	}
	return string(rawOutput), nil
}
