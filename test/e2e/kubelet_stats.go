/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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

package e2e

import (
	"fmt"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/kubelet/metrics"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/master/ports"
)

// KubeletMetric stores metrics scraped from the kubelet server's /metric endpoint.
// TODO: Get some more structure aroud the metrics and this type
type KubeletMetric struct {
	// eg: list, info, create
	Operation string
	// eg: sync_pods, pod_worker
	Method string
	// 0 <= quantile <=1, e.g. 0.95 is 95%tile, 0.5 is median.
	Quantile float64
	Latency  time.Duration
}

// KubeletMetricByLatency implements sort.Interface for []KubeletMetric based on
// the latency field.
type KubeletMetricByLatency []KubeletMetric

func (a KubeletMetricByLatency) Len() int           { return len(a) }
func (a KubeletMetricByLatency) Swap(i, j int)      { a[i], a[j] = a[j], a[i] }
func (a KubeletMetricByLatency) Less(i, j int) bool { return a[i].Latency > a[j].Latency }

// ReadKubeletMetrics reads metrics from the kubelet server running on the given node
func ReadKubeletMetrics(c *client.Client, nodeName string) ([]KubeletMetric, error) {
	body, err := getKubeletMetrics(c, nodeName)
	if err != nil {
		return nil, err
	}

	metric := make([]KubeletMetric, 0)
	for _, line := range strings.Split(string(body), "\n") {

		// A kubelet stats line starts with the KubeletSubsystem marker, followed by a stat name, followed by fields
		// that vary by stat described on a case by case basis below.
		// TODO: String parsing is such a hack, but getting our rest client/proxy to cooperate with prometheus
		// client is weird, we should eventually invest some time in doing this the right way.
		if !strings.HasPrefix(line, fmt.Sprintf("%v_", metrics.KubeletSubsystem)) {
			continue
		}
		keyVal := strings.Split(line, " ")
		if len(keyVal) != 2 {
			return nil, fmt.Errorf("Error parsing metric %q", line)
		}
		keyElems := strings.Split(line, "\"")

		latency, err := strconv.ParseFloat(keyVal[1], 64)
		if err != nil {
			continue
		}

		methodLine := strings.Split(keyElems[0], "{")
		methodList := strings.Split(methodLine[0], "_")
		if len(methodLine) != 2 || len(methodList) == 1 {
			continue
		}
		method := strings.Join(methodList[1:], "_")

		var operation, rawQuantile string
		var quantile float64

		switch method {
		case metrics.PodWorkerLatencyKey:
			// eg: kubelet_pod_worker_latency_microseconds{operation_type="create",pod_name="foopause3_default",quantile="0.99"} 1344
			if len(keyElems) != 7 {
				continue
			}
			operation = keyElems[1]
			rawQuantile = keyElems[5]
			break

		case metrics.SyncPodsLatencyKey:
			// eg:  kubelet_sync_pods_latency_microseconds{quantile="0.5"} 9949
			fallthrough

		case metrics.PodStartLatencyKey:
			// eg: kubelet_pod_start_latency_microseconds{quantile="0.5"} 123
			fallthrough

		case metrics.PodStatusLatencyKey:
			// eg: kubelet_generate_pod_status_latency_microseconds{quantile="0.5"} 12715
			if len(keyElems) != 3 {
				continue
			}
			operation = ""
			rawQuantile = keyElems[1]
			break

		case metrics.ContainerManagerOperationsKey:
			// eg: kubelet_container_manager_latency_microseconds{operation_type="SyncPod",quantile="0.5"} 6705
			fallthrough

		case metrics.DockerOperationsKey:
			// eg: kubelet_docker_operations_latency_microseconds{operation_type="info",quantile="0.5"} 31590
			if len(keyElems) != 5 {
				continue
			}
			operation = keyElems[1]
			rawQuantile = keyElems[3]
			break

		case metrics.DockerErrorsKey:
			Logf("ERROR %v", line)

		default:
			continue
		}
		quantile, err = strconv.ParseFloat(rawQuantile, 64)
		if err != nil {
			continue
		}
		metric = append(metric, KubeletMetric{operation, method, quantile, time.Duration(int64(latency)) * time.Microsecond})
	}
	return metric, nil
}

// HighLatencyKubeletOperations logs and counts the high latency metrics exported by the kubelet server via /metrics.
func HighLatencyKubeletOperations(c *client.Client, threshold time.Duration, nodeName string) ([]KubeletMetric, error) {
	metric, err := ReadKubeletMetrics(c, nodeName)
	if err != nil {
		return []KubeletMetric{}, err
	}
	sort.Sort(KubeletMetricByLatency(metric))
	var badMetrics []KubeletMetric
	Logf("Latency metrics for node %v", nodeName)
	for _, m := range metric {
		if m.Latency > threshold {
			badMetrics = append(badMetrics, m)
			Logf("%+v", m)
		}
	}
	return badMetrics, nil
}

// Retrieve metrics from the kubelet server of the given node.
func getKubeletMetrics(c *client.Client, node string) (string, error) {
	metric, err := c.Get().
		Prefix("proxy").
		Resource("nodes").
		Name(fmt.Sprintf("%v:%v", node, ports.KubeletPort)).
		Suffix("metrics").
		Do().
		Raw()
	if err != nil {
		return "", err
	}
	return string(metric), nil
}
