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
	"fmt"
	"io/ioutil"
	"net/http"
	"time"

	"k8s.io/kubernetes/pkg/util/sets"

	"github.com/prometheus/common/model"
)

var NecessaryKubeletMetrics = map[string][]string{
	"cadvisor_version_info":                                  {"cadvisorRevision", "cadvisorVersion", "dockerVersion", "kernelVersion", "osVersion"},
	"container_cpu_system_seconds_total":                     {"id", "image", "kubernetes_container_name", "kubernetes_namespace", "kubernetes_pod_name", "name"},
	"container_cpu_usage_seconds_total":                      {"id", "image", "kubernetes_container_name", "kubernetes_namespace", "kubernetes_pod_name", "name", "cpu"},
	"container_cpu_user_seconds_total":                       {"id", "image", "kubernetes_container_name", "kubernetes_namespace", "kubernetes_pod_name", "name"},
	"container_fs_io_current":                                {"device", "id", "image", "kubernetes_container_name", "kubernetes_namespace", "kubernetes_pod_name", "name"},
	"container_fs_io_time_seconds_total":                     {"device", "id", "image", "kubernetes_container_name", "kubernetes_namespace", "kubernetes_pod_name", "name"},
	"container_fs_io_time_weighted_seconds_total":            {"device", "id", "image", "kubernetes_container_name", "kubernetes_namespace", "kubernetes_pod_name", "name"},
	"container_fs_limit_bytes":                               {"device", "id", "image", "kubernetes_container_name", "kubernetes_namespace", "kubernetes_pod_name", "name"},
	"container_fs_read_seconds_total":                        {"device", "id", "image", "kubernetes_container_name", "kubernetes_namespace", "kubernetes_pod_name", "name"},
	"container_fs_reads_merged_total":                        {"device", "id", "image", "kubernetes_container_name", "kubernetes_namespace", "kubernetes_pod_name", "name"},
	"container_fs_reads_total":                               {"device", "id", "image", "kubernetes_container_name", "kubernetes_namespace", "kubernetes_pod_name", "name"},
	"container_fs_sector_reads_total":                        {"device", "id", "image", "kubernetes_container_name", "kubernetes_namespace", "kubernetes_pod_name", "name"},
	"container_fs_sector_writes_total":                       {"device", "id", "image", "kubernetes_container_name", "kubernetes_namespace", "kubernetes_pod_name", "name"},
	"container_fs_usage_bytes":                               {"device", "id", "image", "kubernetes_container_name", "kubernetes_namespace", "kubernetes_pod_name", "name"},
	"container_fs_write_seconds_total":                       {"device", "id", "image", "kubernetes_container_name", "kubernetes_namespace", "kubernetes_pod_name", "name"},
	"container_fs_writes_merged_total":                       {"device", "id", "image", "kubernetes_container_name", "kubernetes_namespace", "kubernetes_pod_name", "name"},
	"container_fs_writes_total":                              {"device", "id", "image", "kubernetes_container_name", "kubernetes_namespace", "kubernetes_pod_name", "name"},
	"container_last_seen":                                    {"id", "image", "kubernetes_container_name", "kubernetes_namespace", "kubernetes_pod_name", "name"},
	"container_memory_cache":                                 {},
	"container_memory_rss":                                   {},
	"container_memory_failcnt":                               {"id", "image", "kubernetes_container_name", "kubernetes_namespace", "kubernetes_pod_name", "name"},
	"container_memory_failures_total":                        {"id", "image", "kubernetes_container_name", "kubernetes_namespace", "kubernetes_pod_name", "name", "scope", "type"},
	"container_memory_usage_bytes":                           {"id", "image", "kubernetes_container_name", "kubernetes_namespace", "kubernetes_pod_name", "name"},
	"container_memory_working_set_bytes":                     {"id", "image", "kubernetes_container_name", "kubernetes_namespace", "kubernetes_pod_name", "name"},
	"container_network_receive_bytes_total":                  {"id", "interface", "image", "kubernetes_container_name", "kubernetes_namespace", "kubernetes_pod_name", "name"},
	"container_network_receive_errors_total":                 {"id", "image", "interface", "kubernetes_container_name", "kubernetes_namespace", "kubernetes_pod_name", "name"},
	"container_network_receive_packets_dropped_total":        {"id", "image", "interface", "kubernetes_container_name", "kubernetes_namespace", "kubernetes_pod_name", "name"},
	"container_network_receive_packets_total":                {"id", "image", "interface", "kubernetes_container_name", "kubernetes_namespace", "kubernetes_pod_name", "name"},
	"container_network_transmit_bytes_total":                 {"id", "interface", "image", "kubernetes_container_name", "kubernetes_namespace", "kubernetes_pod_name", "name"},
	"container_network_transmit_errors_total":                {"id", "interface", "image", "kubernetes_container_name", "kubernetes_namespace", "kubernetes_pod_name", "name"},
	"container_network_transmit_packets_dropped_total":       {"id", "interface", "image", "kubernetes_container_name", "kubernetes_namespace", "kubernetes_pod_name", "name"},
	"container_network_transmit_packets_total":               {"id", "interface", "image", "kubernetes_container_name", "kubernetes_namespace", "kubernetes_pod_name", "name"},
	"container_scrape_error":                                 {},
	"container_spec_cpu_period":                              {"id", "image", "kubernetes_container_name", "kubernetes_namespace", "kubernetes_pod_name", "name"},
	"container_spec_cpu_shares":                              {"id", "image", "kubernetes_container_name", "kubernetes_namespace", "kubernetes_pod_name", "name"},
	"container_spec_memory_limit_bytes":                      {"id", "image", "kubernetes_container_name", "kubernetes_namespace", "kubernetes_pod_name", "name"},
	"container_spec_memory_swap_limit_bytes":                 {"id", "image", "kubernetes_container_name", "kubernetes_namespace", "kubernetes_pod_name", "name"},
	"container_start_time_seconds":                           {"id", "image", "kubernetes_container_name", "kubernetes_namespace", "kubernetes_pod_name", "name"},
	"container_tasks_state":                                  {"id", "image", "kubernetes_container_name", "kubernetes_namespace", "kubernetes_pod_name", "name", "state"},
	"kubelet_container_manager_latency_microseconds":         {"operation_type", "quantile"},
	"kubelet_container_manager_latency_microseconds_count":   {"operation_type"},
	"kubelet_container_manager_latency_microseconds_sum":     {"operation_type"},
	"kubelet_containers_per_pod_count":                       {"quantile"},
	"kubelet_containers_per_pod_count_count":                 {},
	"kubelet_containers_per_pod_count_sum":                   {},
	"kubelet_docker_operations":                              {"operation_type"},
	"kubelet_docker_operations_errors":                       {"operation_type"},
	"kubelet_docker_operations_timeout":                      {"operation_type"},
	"kubelet_docker_operations_latency_microseconds":         {"operation_type", "quantile"},
	"kubelet_docker_operations_latency_microseconds_count":   {"operation_type"},
	"kubelet_docker_operations_latency_microseconds_sum":     {"operation_type"},
	"kubelet_generate_pod_status_latency_microseconds":       {"quantile"},
	"kubelet_generate_pod_status_latency_microseconds_count": {},
	"kubelet_generate_pod_status_latency_microseconds_sum":   {},
	"kubelet_pleg_relist_latency_microseconds":               {"quantile"},
	"kubelet_pleg_relist_latency_microseconds_sum":           {},
	"kubelet_pleg_relist_latency_microseconds_count":         {},
	"kubelet_pleg_relist_interval_microseconds":              {"quantile"},
	"kubelet_pleg_relist_interval_microseconds_sum":          {},
	"kubelet_pleg_relist_interval_microseconds_count":        {},
	"kubelet_pod_start_latency_microseconds":                 {"quantile"},
	"kubelet_pod_start_latency_microseconds_count":           {},
	"kubelet_pod_start_latency_microseconds_sum":             {},
	"kubelet_pod_worker_latency_microseconds":                {"operation_type", "quantile"},
	"kubelet_pod_worker_latency_microseconds_count":          {"operation_type"},
	"kubelet_pod_worker_latency_microseconds_sum":            {"operation_type"},
	"kubelet_pod_worker_start_latency_microseconds":          {"quantile"},
	"kubelet_pod_worker_start_latency_microseconds_count":    {},
	"kubelet_pod_worker_start_latency_microseconds_sum":      {},
	"kubelet_running_container_count":                        {},
	"kubelet_running_pod_count":                              {},
	"kubelet_sync_pods_latency_microseconds":                 {"quantile"},
	"kubelet_sync_pods_latency_microseconds_count":           {},
	"kubelet_sync_pods_latency_microseconds_sum":             {},
	"machine_cpu_cores":                                      {},
	"machine_memory_bytes":                                   {},
	"rest_client_request_latency_microseconds":               {"quantile", "url", "verb"},
	"rest_client_request_latency_microseconds_count":         {"url", "verb"},
	"rest_client_request_latency_microseconds_sum":           {"url", "verb"},
	"rest_client_request_status_codes":                       {"code", "host", "method"},
}

var KubeletMetricsLabelsToSkip = sets.NewString(
	"kubernetes_namespace",
	"image",
	"name",
)

type KubeletMetrics Metrics

func (m *KubeletMetrics) Equal(o KubeletMetrics) bool {
	return (*Metrics)(m).Equal(Metrics(o))
}

func NewKubeletMetrics() KubeletMetrics {
	result := NewMetrics()
	for metric := range NecessaryKubeletMetrics {
		result[metric] = make(model.Samples, 0)
	}
	return KubeletMetrics(result)
}

// GrabKubeletMetricsWithoutProxy retrieve metrics from the kubelet on the given node using a simple GET over http.
// Currently only used in integration tests.
func GrabKubeletMetricsWithoutProxy(nodeName string) (KubeletMetrics, error) {
	metricsEndpoint := "http://%s/metrics"
	resp, err := http.Get(fmt.Sprintf(metricsEndpoint, nodeName))
	if err != nil {
		return KubeletMetrics{}, err
	}
	defer resp.Body.Close()
	body, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		return KubeletMetrics{}, err
	}
	return parseKubeletMetrics(string(body))
}

func parseKubeletMetrics(data string) (KubeletMetrics, error) {
	result := NewKubeletMetrics()
	if err := parseMetrics(data, NecessaryKubeletMetrics, (*Metrics)(&result), nil); err != nil {
		return KubeletMetrics{}, err
	}
	return result, nil
}

func (g *MetricsGrabber) getMetricsFromNode(nodeName string, kubeletPort int) (string, error) {
	// There's a problem with timing out during proxy. Wrapping this in a goroutine to prevent deadlock.
	// Hanging goroutine will be leaked.
	finished := make(chan struct{})
	var err error
	var rawOutput []byte
	go func() {
		rawOutput, err = g.client.Get().
			Prefix("proxy").
			Resource("nodes").
			Name(fmt.Sprintf("%v:%v", nodeName, kubeletPort)).
			Suffix("metrics").
			Do().Raw()
		finished <- struct{}{}
	}()
	select {
	case <-time.After(ProxyTimeout):
		return "", fmt.Errorf("Timed out when waiting for proxy to gather metrics from %v", nodeName)
	case <-finished:
		if err != nil {
			return "", err
		}
		return string(rawOutput), nil
	}
}
