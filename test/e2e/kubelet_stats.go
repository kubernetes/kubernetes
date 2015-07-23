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
	"io/ioutil"
	"net/http"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/kubelet/metrics"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/master/ports"
)

type MetricsType int

const (
	MetricsTypeLatency MetricsType = iota
	MetricsTypeSum
	MetricsTypeCount
)

// KubeletMetric stores metrics scraped from the kubelet server's /metric endpoint.
// TODO: Get some more structure around the metrics and this type
type KubeletMetric struct {
	Type MetricsType
	// eg: list, info, create
	Operation string
	// eg: sync_pods, pod_worker
	Method string

	// The following fields are set for MetricsTypeLatency:
	// 0 <= quantile <=1, e.g. 0.95 is 95%tile, 0.5 is median.
	Quantile float64
	Latency  time.Duration

	// The following fields are set for MetricsTypeSum:
	// Sum of all latencies for this type of operation.
	Sum time.Duration

	// The following fields are set for MetricsTypeCount:
	// Number of operations observed.
	Count int64
}

// KubeletMetricByLatency implements sort.Interface for []KubeletMetric based on
// the latency field.
type KubeletMetricByLatency []KubeletMetric

func (a KubeletMetricByLatency) Len() int           { return len(a) }
func (a KubeletMetricByLatency) Swap(i, j int)      { a[i], a[j] = a[j], a[i] }
func (a KubeletMetricByLatency) Less(i, j int) bool { return a[i].Latency > a[j].Latency }

// parseKubeletmetricsLine parse a single line in the kubelet /metrics output.
func parseKubeletMetricsLine(line string, prefix string) (string, string, map[string]string, error) {
	// Extract method and value from the line.
	// E.g., kubelet_pod_worker_latency_microseconds{operation_type="create", quantile="0.99"} 1344
	// Method is "kubelet_pod_worker_latency_microseconds"; value is "1344"
	keyValMap := make(map[string]string)
	r, _ := regexp.Compile(`(\w+)(?:{.*})? ([\w.]+)`)
	matches := r.FindAllStringSubmatch(line, -1)
	if len(matches) != 1 {
		return "", "", keyValMap, fmt.Errorf("found zero or muiltiple matches. line: %s, matches: %v", line, matches)
	}
	method, value := matches[0][1], matches[0][2]
	// Extract key value pairs from the line.
	// E.g. {"operation_type": "create", "quantile": "0.99"}
	r, _ = regexp.Compile(`(\w+)="([\w.]+)"`)
	matches = r.FindAllStringSubmatch(line, -1)
	for _, match := range matches {
		keyValMap[match[1]] = match[2]
	}
	return method, value, keyValMap, nil
}

const (
	metricsOpKey       = "operation_type"
	metricsQuantileKey = "quantile"
	metricsSumSuffix   = "_sum"
	metricsCountSuffix = "_count"
)

// ReadKubeletMetrics reads metrics from the kubelet server running on the given node
// TODO: String parsing is such a hack, but getting our rest client/proxy to cooperate with prometheus
// client is weird, we should eventually invest some time in doing this the right way.
func ParseKubeletMetrics(metricsBlob string) ([]KubeletMetric, error) {
	metric := make([]KubeletMetric, 0)
	for _, line := range strings.Split(metricsBlob, "\n") {
		// We are interested in kubelet stats lines, which start with the
		// KubeletSubsystem marker. E.g., kubelet_pod_worker_latency_microseconds
		prefix := fmt.Sprintf("%v_", metrics.KubeletSubsystem)
		if !strings.HasPrefix(line, prefix) {
			continue
		}
		method, value, keyValMap, err := parseKubeletMetricsLine(line, prefix)
		if err != nil {
			return nil, fmt.Errorf("Error parsing metric %q, err: %v", line, err)
		}
		// Trim the kubelet prefix.
		method = strings.TrimPrefix(method, prefix)
		if method == metrics.DockerErrorsKey {
			Logf("ERROR %v", line)
			continue
		}
		var metricsType MetricsType
		var operation, rawQuantile string
		var sum, quantile, latency float64
		var count int64
		var ok bool
		// Operation is optional.
		operation, _ = keyValMap[metricsOpKey]

		if strings.HasSuffix(method, metricsSumSuffix) {
			metricsType = MetricsTypeSum
			sum, err = strconv.ParseFloat(value, 64)
			if err != nil {
				continue
			}
		} else if strings.HasSuffix(method, metricsCountSuffix) {
			metricsType = MetricsTypeCount
			count, err = strconv.ParseInt(value, 10, 64)
			if err != nil {
				continue
			}
		} else {
			metricsType = MetricsTypeLatency
			latency, err = strconv.ParseFloat(value, 64)
			if err != nil {
				continue
			}
			rawQuantile, ok = keyValMap[metricsQuantileKey]
			if !ok {
				continue
			}
			quantile, err = strconv.ParseFloat(rawQuantile, 64)
			if err != nil {
				continue
			}
		}
		metric = append(metric, KubeletMetric{metricsType, operation, method, quantile,
			time.Duration(int64(latency)) * time.Microsecond, time.Duration(int64(sum)) * time.Microsecond, count})
	}
	return metric, nil
}

// HighLatencyKubeletOperations logs and counts the high latency metrics exported by the kubelet server via /metrics.
func HighLatencyKubeletOperations(c *client.Client, threshold time.Duration, nodeName string) ([]KubeletMetric, error) {
	var metricsBlob string
	var err error
	// If we haven't been given a client try scraping the nodename directly for a /metrics endpoint.
	if c == nil {
		metricsBlob, err = getKubeletMetricsThroughNode(nodeName)
	} else {
		metricsBlob, err = getKubeletMetricsThroughProxy(c, nodeName)
	}
	if err != nil {
		return []KubeletMetric{}, err
	}
	allMetric, err := ParseKubeletMetrics(metricsBlob)
	// Only look at metric with latency information.
	var metric []KubeletMetric
	for _, m := range allMetric {
		if m.Type == MetricsTypeLatency {
			metric = append(metric, m)
		}
	}
	if err != nil {
		return []KubeletMetric{}, err
	}
	sort.Sort(KubeletMetricByLatency(metric))
	var badMetrics []KubeletMetric
	Logf("\nLatency metrics for node %v", nodeName)
	for _, m := range metric {
		if m.Latency > threshold {
			badMetrics = append(badMetrics, m)
			Logf("%+v", m)
		}
	}
	return badMetrics, nil
}

// Performs a get on a node proxy endpoint given the nodename and rest client.
func nodeProxyRequest(c *client.Client, node, endpoint string) client.Result {
	return c.Get().
		Prefix("proxy").
		Resource("nodes").
		Name(fmt.Sprintf("%v:%v", node, ports.KubeletPort)).
		Suffix(endpoint).
		Do()
}

// Retrieve metrics from the kubelet server of the given node.
func getKubeletMetricsThroughProxy(c *client.Client, node string) (string, error) {
	metric, err := nodeProxyRequest(c, node, "metrics").Raw()
	if err != nil {
		return "", err
	}
	return string(metric), nil
}

// Retrieve metrics from the kubelet on the given node using a simple GET over http.
// Currently only used in integration tests.
func getKubeletMetricsThroughNode(nodeName string) (string, error) {
	resp, err := http.Get(fmt.Sprintf("http://%v/metrics", nodeName))
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()
	body, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		return "", err
	}
	return string(body), nil
}

// GetKubeletPods retrieves the list of running pods on the kubelet. The pods
// includes necessary information (e.g., UID, name, namespace for
// pods/containers), but do not contain the full spec.
func GetKubeletPods(c *client.Client, node string) (*api.PodList, error) {
	result := &api.PodList{}
	if err := nodeProxyRequest(c, node, "runningpods").Into(result); err != nil {
		return &api.PodList{}, err
	}
	return result, nil
}
