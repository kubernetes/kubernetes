/*
Copyright 2015 The Kubernetes Authors.

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
	"context"
	"fmt"
	"io/ioutil"
	"net/http"
	"sort"
	"strconv"
	"strings"
	"time"

	"k8s.io/apimachinery/pkg/util/sets"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/component-base/metrics/testutil"
	dockermetrics "k8s.io/kubernetes/pkg/kubelet/dockershim/metrics"
	kubeletmetrics "k8s.io/kubernetes/pkg/kubelet/metrics"
	e2elog "k8s.io/kubernetes/test/e2e/framework/log"
)

const (
	proxyTimeout = 2 * time.Minute
)

// KubeletMetrics is metrics for kubelet
type KubeletMetrics testutil.Metrics

// Equal returns true if all metrics are the same as the arguments.
func (m *KubeletMetrics) Equal(o KubeletMetrics) bool {
	return (*testutil.Metrics)(m).Equal(testutil.Metrics(o))
}

// NewKubeletMetrics returns new metrics which are initialized.
func NewKubeletMetrics() KubeletMetrics {
	result := testutil.NewMetrics()
	return KubeletMetrics(result)
}

// GrabKubeletMetricsWithoutProxy retrieve metrics from the kubelet on the given node using a simple GET over http.
// Currently only used in integration tests.
func GrabKubeletMetricsWithoutProxy(nodeName, path string) (KubeletMetrics, error) {
	resp, err := http.Get(fmt.Sprintf("http://%s%s", nodeName, path))
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
	if err := testutil.ParseMetrics(data, (*testutil.Metrics)(&result)); err != nil {
		return KubeletMetrics{}, err
	}
	return result, nil
}

func (g *Grabber) getMetricsFromNode(nodeName string, kubeletPort int) (string, error) {
	// There's a problem with timing out during proxy. Wrapping this in a goroutine to prevent deadlock.
	finished := make(chan struct{}, 1)
	var err error
	var rawOutput []byte
	go func() {
		rawOutput, err = g.client.CoreV1().RESTClient().Get().
			Resource("nodes").
			SubResource("proxy").
			Name(fmt.Sprintf("%v:%v", nodeName, kubeletPort)).
			Suffix("metrics").
			Do(context.TODO()).Raw()
		finished <- struct{}{}
	}()
	select {
	case <-time.After(proxyTimeout):
		return "", fmt.Errorf("Timed out when waiting for proxy to gather metrics from %v", nodeName)
	case <-finished:
		if err != nil {
			return "", err
		}
		return string(rawOutput), nil
	}
}

// KubeletLatencyMetric stores metrics scraped from the kubelet server's /metric endpoint.
// TODO: Get some more structure around the metrics and this type
type KubeletLatencyMetric struct {
	// eg: list, info, create
	Operation string
	// eg: sync_pods, pod_worker
	Method string
	// 0 <= quantile <=1, e.g. 0.95 is 95%tile, 0.5 is median.
	Quantile float64
	Latency  time.Duration
}

// KubeletLatencyMetrics implements sort.Interface for []KubeletMetric based on
// the latency field.
type KubeletLatencyMetrics []KubeletLatencyMetric

func (a KubeletLatencyMetrics) Len() int           { return len(a) }
func (a KubeletLatencyMetrics) Swap(i, j int)      { a[i], a[j] = a[j], a[i] }
func (a KubeletLatencyMetrics) Less(i, j int) bool { return a[i].Latency > a[j].Latency }

// If a apiserver client is passed in, the function will try to get kubelet metrics from metrics grabber;
// or else, the function will try to get kubelet metrics directly from the node.
func getKubeletMetricsFromNode(c clientset.Interface, nodeName string) (KubeletMetrics, error) {
	if c == nil {
		return GrabKubeletMetricsWithoutProxy(nodeName, "/metrics")
	}
	grabber, err := NewMetricsGrabber(c, nil, true, false, false, false, false)
	if err != nil {
		return KubeletMetrics{}, err
	}
	return grabber.GrabFromKubelet(nodeName)
}

// GetKubeletMetrics gets all metrics in kubelet subsystem from specified node and trims
// the subsystem prefix.
func GetKubeletMetrics(c clientset.Interface, nodeName string) (KubeletMetrics, error) {
	ms, err := getKubeletMetricsFromNode(c, nodeName)
	if err != nil {
		return KubeletMetrics{}, err
	}

	kubeletMetrics := make(KubeletMetrics)
	for name, samples := range ms {
		const prefix = kubeletmetrics.KubeletSubsystem + "_"
		if !strings.HasPrefix(name, prefix) {
			// Not a kubelet metric.
			continue
		}
		method := strings.TrimPrefix(name, prefix)
		kubeletMetrics[method] = samples
	}
	return kubeletMetrics, nil
}

// GetDefaultKubeletLatencyMetrics calls GetKubeletLatencyMetrics with a set of default metricNames
// identifying common latency metrics.
// Note that the KubeletMetrics passed in should not contain subsystem prefix.
func GetDefaultKubeletLatencyMetrics(ms KubeletMetrics) KubeletLatencyMetrics {
	latencyMetricNames := sets.NewString(
		kubeletmetrics.PodWorkerDurationKey,
		kubeletmetrics.PodWorkerStartDurationKey,
		kubeletmetrics.PodStartDurationKey,
		kubeletmetrics.CgroupManagerOperationsKey,
		dockermetrics.DockerOperationsLatencyKey,
		kubeletmetrics.PodWorkerStartDurationKey,
		kubeletmetrics.PLEGRelistDurationKey,
	)
	return GetKubeletLatencyMetrics(ms, latencyMetricNames)
}

// GetKubeletLatencyMetrics filters ms to include only those contained in the metricNames set,
// then constructs a KubeletLatencyMetrics list based on the samples associated with those metrics.
func GetKubeletLatencyMetrics(ms KubeletMetrics, filterMetricNames sets.String) KubeletLatencyMetrics {
	var latencyMetrics KubeletLatencyMetrics
	for name, samples := range ms {
		if !filterMetricNames.Has(name) {
			continue
		}
		for _, sample := range samples {
			latency := sample.Value
			operation := string(sample.Metric["operation_type"])
			var quantile float64
			if val, ok := sample.Metric[testutil.QuantileLabel]; ok {
				var err error
				if quantile, err = strconv.ParseFloat(string(val), 64); err != nil {
					continue
				}
			}

			latencyMetrics = append(latencyMetrics, KubeletLatencyMetric{
				Operation: operation,
				Method:    name,
				Quantile:  quantile,
				Latency:   time.Duration(int64(latency)) * time.Microsecond,
			})
		}
	}
	return latencyMetrics
}

// HighLatencyKubeletOperations logs and counts the high latency metrics exported by the kubelet server via /metrics.
func HighLatencyKubeletOperations(c clientset.Interface, threshold time.Duration, nodeName string, logFunc func(fmt string, args ...interface{})) (KubeletLatencyMetrics, error) {
	ms, err := GetKubeletMetrics(c, nodeName)
	if err != nil {
		return KubeletLatencyMetrics{}, err
	}
	latencyMetrics := GetDefaultKubeletLatencyMetrics(ms)
	sort.Sort(latencyMetrics)
	var badMetrics KubeletLatencyMetrics
	logFunc("\nLatency metrics for node %v", nodeName)
	for _, m := range latencyMetrics {
		if m.Latency > threshold {
			badMetrics = append(badMetrics, m)
			e2elog.Logf("%+v", m)
		}
	}
	return badMetrics, nil
}
