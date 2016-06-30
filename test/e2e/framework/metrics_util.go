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

package framework

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"sort"
	"strconv"
	"strings"
	"time"

	"k8s.io/kubernetes/pkg/api"
	client "k8s.io/kubernetes/pkg/client/unversioned"
	"k8s.io/kubernetes/pkg/master/ports"
	"k8s.io/kubernetes/pkg/metrics"
	"k8s.io/kubernetes/pkg/util/sets"

	"github.com/prometheus/common/expfmt"
	"github.com/prometheus/common/model"
)

const (
	// NodeStartupThreshold is a rough estimate of the time allocated for a pod to start on a node.
	NodeStartupThreshold = 4 * time.Second

	podStartupThreshold time.Duration = 5 * time.Second
	// We are setting 1s threshold for apicalls even in small clusters to avoid flakes.
	// The problem is that if long GC is happening in small clusters (where we have e.g.
	// 1-core master machines) and tests are pretty short, it may consume significant
	// portion of CPU and basically stop all the real work.
	// Increasing threshold to 1s is within our SLO and should solve this problem.
	apiCallLatencyThreshold time.Duration = 1 * time.Second
)

type MetricsForE2E metrics.MetricsCollection

func (m *MetricsForE2E) filterMetrics() {
	interestingApiServerMetrics := make(metrics.ApiServerMetrics)
	for _, metric := range InterestingApiServerMetrics {
		interestingApiServerMetrics[metric] = (*m).ApiServerMetrics[metric]
	}
	interestingControllerManagerMetrics := make(metrics.ControllerManagerMetrics)
	for _, metric := range InterestingControllerManagerMetrics {
		interestingControllerManagerMetrics[metric] = (*m).ControllerManagerMetrics[metric]
	}
	interestingKubeletMetrics := make(map[string]metrics.KubeletMetrics)
	for kubelet, grabbed := range (*m).KubeletMetrics {
		interestingKubeletMetrics[kubelet] = make(metrics.KubeletMetrics)
		for _, metric := range InterestingKubeletMetrics {
			interestingKubeletMetrics[kubelet][metric] = grabbed[metric]
		}
	}
	(*m).ApiServerMetrics = interestingApiServerMetrics
	(*m).ControllerManagerMetrics = interestingControllerManagerMetrics
	(*m).KubeletMetrics = interestingKubeletMetrics
}

func (m *MetricsForE2E) PrintHumanReadable() string {
	buf := bytes.Buffer{}
	for _, interestingMetric := range InterestingApiServerMetrics {
		buf.WriteString(fmt.Sprintf("For %v:\n", interestingMetric))
		for _, sample := range (*m).ApiServerMetrics[interestingMetric] {
			buf.WriteString(fmt.Sprintf("\t%v\n", metrics.PrintSample(sample)))
		}
	}
	for _, interestingMetric := range InterestingControllerManagerMetrics {
		buf.WriteString(fmt.Sprintf("For %v:\n", interestingMetric))
		for _, sample := range (*m).ControllerManagerMetrics[interestingMetric] {
			buf.WriteString(fmt.Sprintf("\t%v\n", metrics.PrintSample(sample)))
		}
	}
	for kubelet, grabbed := range (*m).KubeletMetrics {
		buf.WriteString(fmt.Sprintf("For %v:\n", kubelet))
		for _, interestingMetric := range InterestingKubeletMetrics {
			buf.WriteString(fmt.Sprintf("\tFor %v:\n", interestingMetric))
			for _, sample := range grabbed[interestingMetric] {
				buf.WriteString(fmt.Sprintf("\t\t%v\n", metrics.PrintSample(sample)))
			}
		}
	}
	return buf.String()
}

func (m *MetricsForE2E) PrintJSON() string {
	m.filterMetrics()
	return PrettyPrintJSON(*m)
}

var InterestingApiServerMetrics = []string{
	"apiserver_request_count",
	"apiserver_request_latencies_summary",
	"etcd_helper_cache_entry_count",
	"etcd_helper_cache_hit_count",
	"etcd_helper_cache_miss_count",
	"etcd_request_cache_add_latencies_summary",
	"etcd_request_cache_get_latencies_summary",
	"etcd_request_latencies_summary",
}

var InterestingControllerManagerMetrics = []string{
	"garbage_collector_event_processing_latency_microseconds",
	"garbage_collector_dirty_processing_latency_microseconds",
	"garbage_collector_orphan_processing_latency_microseconds",
}

var InterestingKubeletMetrics = []string{
	"kubelet_container_manager_latency_microseconds",
	"kubelet_docker_errors",
	"kubelet_docker_operations_latency_microseconds",
	"kubelet_generate_pod_status_latency_microseconds",
	"kubelet_pod_start_latency_microseconds",
	"kubelet_pod_worker_latency_microseconds",
	"kubelet_pod_worker_start_latency_microseconds",
	"kubelet_sync_pods_latency_microseconds",
}

// Dashboard metrics
type LatencyMetric struct {
	Perc50 time.Duration `json:"Perc50"`
	Perc90 time.Duration `json:"Perc90"`
	Perc99 time.Duration `json:"Perc99"`
}

type PodStartupLatency struct {
	Latency LatencyMetric `json:"latency"`
}

type SchedulingLatency struct {
	Scheduling LatencyMetric `json:"scheduling"`
	Binding    LatencyMetric `json:"binding"`
	Total      LatencyMetric `json:"total"`
}

type SaturationTime struct {
	TimeToSaturate time.Duration `json:"timeToStaturate"`
	NumberOfNodes  int           `json:"numberOfNodes"`
	NumberOfPods   int           `json:"numberOfPods"`
	Throughput     float32       `json:"throughput"`
}

type APICall struct {
	Resource string        `json:"resource"`
	Verb     string        `json:"verb"`
	Latency  LatencyMetric `json:"latency"`
}

type APIResponsiveness struct {
	APICalls []APICall `json:"apicalls"`
}

func (a APIResponsiveness) Len() int      { return len(a.APICalls) }
func (a APIResponsiveness) Swap(i, j int) { a.APICalls[i], a.APICalls[j] = a.APICalls[j], a.APICalls[i] }
func (a APIResponsiveness) Less(i, j int) bool {
	return a.APICalls[i].Latency.Perc99 < a.APICalls[j].Latency.Perc99
}

// 0 <= quantile <=1 (e.g. 0.95 is 95%tile, 0.5 is median)
// Only 0.5, 0.9 and 0.99 quantiles are supported.
func (a *APIResponsiveness) addMetric(resource, verb string, quantile float64, latency time.Duration) {
	for i, apicall := range a.APICalls {
		if apicall.Resource == resource && apicall.Verb == verb {
			a.APICalls[i] = setQuantileAPICall(apicall, quantile, latency)
			return
		}
	}
	apicall := setQuantileAPICall(APICall{Resource: resource, Verb: verb}, quantile, latency)
	a.APICalls = append(a.APICalls, apicall)
}

// 0 <= quantile <=1 (e.g. 0.95 is 95%tile, 0.5 is median)
// Only 0.5, 0.9 and 0.99 quantiles are supported.
func setQuantileAPICall(apicall APICall, quantile float64, latency time.Duration) APICall {
	setQuantile(&apicall.Latency, quantile, latency)
	return apicall
}

// Only 0.5, 0.9 and 0.99 quantiles are supported.
func setQuantile(metric *LatencyMetric, quantile float64, latency time.Duration) {
	switch quantile {
	case 0.5:
		metric.Perc50 = latency
	case 0.9:
		metric.Perc90 = latency
	case 0.99:
		metric.Perc99 = latency
	}
}

func readLatencyMetrics(c *client.Client) (APIResponsiveness, error) {
	var a APIResponsiveness

	body, err := getMetrics(c)
	if err != nil {
		return a, err
	}

	samples, err := extractMetricSamples(body)
	if err != nil {
		return a, err
	}

	ignoredResources := sets.NewString("events")
	// TODO: figure out why we're getting non-capitalized proxy and fix this.
	ignoredVerbs := sets.NewString("WATCHLIST", "PROXY", "proxy", "CONNECT")

	for _, sample := range samples {
		// Example line:
		// apiserver_request_latencies_summary{resource="namespaces",verb="LIST",quantile="0.99"} 908
		if sample.Metric[model.MetricNameLabel] != "apiserver_request_latencies_summary" {
			continue
		}

		resource := string(sample.Metric["resource"])
		verb := string(sample.Metric["verb"])
		if ignoredResources.Has(resource) || ignoredVerbs.Has(verb) {
			continue
		}
		latency := sample.Value
		quantile, err := strconv.ParseFloat(string(sample.Metric[model.QuantileLabel]), 64)
		if err != nil {
			return a, err
		}
		a.addMetric(resource, verb, quantile, time.Duration(int64(latency))*time.Microsecond)
	}

	return a, err
}

// Prints top five summary metrics for request types with latency and returns
// number of such request types above threshold.
func HighLatencyRequests(c *client.Client) (int, error) {
	metrics, err := readLatencyMetrics(c)
	if err != nil {
		return 0, err
	}
	sort.Sort(sort.Reverse(metrics))
	badMetrics := 0
	top := 5
	for _, metric := range metrics.APICalls {
		isBad := false
		if metric.Latency.Perc99 > apiCallLatencyThreshold {
			badMetrics++
			isBad = true
		}
		if top > 0 || isBad {
			top--
			prefix := ""
			if isBad {
				prefix = "WARNING "
			}
			Logf("%vTop latency metric: %+v", prefix, metric)
		}
	}

	// TODO(random-liu): Remove the log when we migrate to new perfdash
	Logf("API calls latencies: %s", PrettyPrintJSON(metrics))
	// Log perf data
	PrintPerfData(ApiCallToPerfData(metrics))

	return badMetrics, nil
}

// Verifies whether 50, 90 and 99th percentiles of PodStartupLatency are
// within the threshold.
func VerifyPodStartupLatency(latency PodStartupLatency) error {
	Logf("Pod startup latency: %s", PrettyPrintJSON(latency))

	if latency.Latency.Perc50 > podStartupThreshold {
		return fmt.Errorf("too high pod startup latency 50th percentile: %v", latency.Latency.Perc50)
	}
	if latency.Latency.Perc90 > podStartupThreshold {
		return fmt.Errorf("too high pod startup latency 90th percentile: %v", latency.Latency.Perc90)
	}
	if latency.Latency.Perc99 > podStartupThreshold {
		return fmt.Errorf("too high pod startup latency 99th percentil: %v", latency.Latency.Perc99)
	}
	return nil
}

// Resets latency metrics in apiserver.
func ResetMetrics(c *client.Client) error {
	Logf("Resetting latency metrics in apiserver...")
	body, err := c.Delete().AbsPath("/metrics").DoRaw()
	if err != nil {
		return err
	}
	if string(body) != "metrics reset\n" {
		return fmt.Errorf("Unexpected response: %q", string(body))
	}
	return nil
}

// Retrieves metrics information.
func getMetrics(c *client.Client) (string, error) {
	body, err := c.Get().AbsPath("/metrics").DoRaw()
	if err != nil {
		return "", err
	}
	return string(body), nil
}

// Retrieves scheduler metrics information.
func getSchedulingLatency(c *client.Client) (SchedulingLatency, error) {
	result := SchedulingLatency{}

	// Check if master Node is registered
	nodes, err := c.Nodes().List(api.ListOptions{})
	ExpectNoError(err)

	var data string
	var masterRegistered = false
	for _, node := range nodes.Items {
		if strings.HasSuffix(node.Name, "master") {
			masterRegistered = true
		}
	}
	if masterRegistered {
		rawData, err := c.Get().
			Prefix("proxy").
			Namespace(api.NamespaceSystem).
			Resource("pods").
			Name(fmt.Sprintf("kube-scheduler-%v:%v", TestContext.CloudConfig.MasterName, ports.SchedulerPort)).
			Suffix("metrics").
			Do().Raw()

		ExpectNoError(err)
		data = string(rawData)
	} else {
		// If master is not registered fall back to old method of using SSH.
		cmd := "curl http://localhost:10251/metrics"
		sshResult, err := SSH(cmd, GetMasterHost()+":22", TestContext.Provider)
		if err != nil || sshResult.Code != 0 {
			return result, fmt.Errorf("unexpected error (code: %d) in ssh connection to master: %#v", sshResult.Code, err)
		}
		data = sshResult.Stdout
	}
	samples, err := extractMetricSamples(data)
	if err != nil {
		return result, err
	}

	for _, sample := range samples {
		var metric *LatencyMetric = nil
		switch sample.Metric[model.MetricNameLabel] {
		case "scheduler_scheduling_algorithm_latency_microseconds":
			metric = &result.Scheduling
		case "scheduler_binding_latency_microseconds":
			metric = &result.Binding
		case "scheduler_e2e_scheduling_latency_microseconds":
			metric = &result.Total
		}
		if metric == nil {
			continue
		}

		latency := sample.Value
		quantile, err := strconv.ParseFloat(string(sample.Metric[model.QuantileLabel]), 64)
		if err != nil {
			return result, err
		}
		setQuantile(metric, quantile, time.Duration(int64(latency))*time.Microsecond)
	}
	return result, nil
}

// Verifies (currently just by logging them) the scheduling latencies.
func VerifySchedulerLatency(c *client.Client) error {
	latency, err := getSchedulingLatency(c)
	if err != nil {
		return err
	}
	Logf("Scheduling latency: %s", PrettyPrintJSON(latency))

	// TODO: Add some reasonable checks once we know more about the values.
	return nil
}

func PrettyPrintJSON(metrics interface{}) string {
	output := &bytes.Buffer{}
	if err := json.NewEncoder(output).Encode(metrics); err != nil {
		Logf("Error building encoder: %v", err)
		return ""
	}
	formatted := &bytes.Buffer{}
	if err := json.Indent(formatted, output.Bytes(), "", "  "); err != nil {
		Logf("Error indenting: %v", err)
		return ""
	}
	return string(formatted.Bytes())
}

// extractMetricSamples parses the prometheus metric samples from the input string.
func extractMetricSamples(metricsBlob string) ([]*model.Sample, error) {
	dec := expfmt.NewDecoder(strings.NewReader(metricsBlob), expfmt.FmtText)
	decoder := expfmt.SampleDecoder{
		Dec:  dec,
		Opts: &expfmt.DecodeOptions{},
	}

	var samples []*model.Sample
	for {
		var v model.Vector
		if err := decoder.Decode(&v); err != nil {
			if err == io.EOF {
				// Expected loop termination condition.
				return samples, nil
			}
			return nil, err
		}
		samples = append(samples, v...)
	}
}

// PodLatencyData encapsulates pod startup latency information.
type PodLatencyData struct {
	// Name of the pod
	Name string
	// Node this pod was running on
	Node string
	// Latency information related to pod startuptime
	Latency time.Duration
}

type LatencySlice []PodLatencyData

func (a LatencySlice) Len() int           { return len(a) }
func (a LatencySlice) Swap(i, j int)      { a[i], a[j] = a[j], a[i] }
func (a LatencySlice) Less(i, j int) bool { return a[i].Latency < a[j].Latency }

func ExtractLatencyMetrics(latencies []PodLatencyData) LatencyMetric {
	length := len(latencies)
	perc50 := latencies[int(math.Ceil(float64(length*50)/100))-1].Latency
	perc90 := latencies[int(math.Ceil(float64(length*90)/100))-1].Latency
	perc99 := latencies[int(math.Ceil(float64(length*99)/100))-1].Latency
	return LatencyMetric{Perc50: perc50, Perc90: perc90, Perc99: perc99}
}

// LogSuspiciousLatency logs metrics/docker errors from all nodes that had slow startup times
// If latencyDataLag is nil then it will be populated from latencyData
func LogSuspiciousLatency(latencyData []PodLatencyData, latencyDataLag []PodLatencyData, nodeCount int, c *client.Client) {
	if latencyDataLag == nil {
		latencyDataLag = latencyData
	}
	for _, l := range latencyData {
		if l.Latency > NodeStartupThreshold {
			HighLatencyKubeletOperations(c, 1*time.Second, l.Node)
		}
	}
	Logf("Approx throughput: %v pods/min",
		float64(nodeCount)/(latencyDataLag[len(latencyDataLag)-1].Latency.Minutes()))
}

// testMaximumLatencyValue verifies the highest latency value is less than or equal to
// the given time.Duration. Since the arrays are sorted we are looking at the last
// element which will always be the highest. If the latency is higher than the max Failf
// is called.
func testMaximumLatencyValue(latencies []PodLatencyData, max time.Duration, name string) {
	highestLatency := latencies[len(latencies)-1]
	if !(highestLatency.Latency <= max) {
		Failf("%s were not all under %s: %#v", name, max.String(), latencies)
	}
}

func PrintLatencies(latencies []PodLatencyData, header string) {
	metrics := ExtractLatencyMetrics(latencies)
	Logf("10%% %s: %v", header, latencies[(len(latencies)*9)/10:])
	Logf("perc50: %v, perc90: %v, perc99: %v", metrics.Perc50, metrics.Perc90, metrics.Perc99)
}
