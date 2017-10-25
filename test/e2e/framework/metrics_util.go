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
	"context"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"sort"
	"strconv"
	"strings"
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/sets"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/pkg/master/ports"
	"k8s.io/kubernetes/pkg/util/system"
	"k8s.io/kubernetes/test/e2e/framework/metrics"

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

	// We use a higher threshold for list apicalls if the cluster is big (i.e having > 500 nodes)
	// as list response sizes are bigger in general for big clusters. We also use a higher threshold
	// for list calls at cluster scope (this includes non-namespaced and all-namespaced calls).
	apiListCallLatencyThreshold      time.Duration = 5 * time.Second
	apiClusterScopeListCallThreshold time.Duration = 10 * time.Second
	bigClusterNodeCountThreshold                   = 500

	// Cluster Autoscaler metrics names
	caFunctionMetric      = "cluster_autoscaler_function_duration_seconds_bucket"
	caFunctionMetricLabel = "function"
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
	interestingClusterAutoscalerMetrics := make(metrics.ClusterAutoscalerMetrics)
	for _, metric := range InterestingClusterAutoscalerMetrics {
		interestingClusterAutoscalerMetrics[metric] = (*m).ClusterAutoscalerMetrics[metric]
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
	for _, interestingMetric := range InterestingClusterAutoscalerMetrics {
		buf.WriteString(fmt.Sprintf("For %v:\n", interestingMetric))
		for _, sample := range (*m).ClusterAutoscalerMetrics[interestingMetric] {
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
	return PrettyPrintJSON(m)
}

func (m *MetricsForE2E) SummaryKind() string {
	return "MetricsForE2E"
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
	"garbage_collector_attempt_to_delete_queue_latency",
	"garbage_collector_attempt_to_delete_work_duration",
	"garbage_collector_attempt_to_orphan_queue_latency",
	"garbage_collector_attempt_to_orphan_work_duration",
	"garbage_collector_dirty_processing_latency_microseconds",
	"garbage_collector_event_processing_latency_microseconds",
	"garbage_collector_graph_changes_queue_latency",
	"garbage_collector_graph_changes_work_duration",
	"garbage_collector_orphan_processing_latency_microseconds",

	"namespace_queue_latency",
	"namespace_queue_latency_sum",
	"namespace_queue_latency_count",
	"namespace_retries",
	"namespace_work_duration",
	"namespace_work_duration_sum",
	"namespace_work_duration_count",
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

var InterestingClusterAutoscalerMetrics = []string{
	"function_duration_seconds",
	"errors_total",
	"evicted_pods_total",
}

// Dashboard metrics
type LatencyMetric struct {
	Perc50  time.Duration `json:"Perc50"`
	Perc90  time.Duration `json:"Perc90"`
	Perc99  time.Duration `json:"Perc99"`
	Perc100 time.Duration `json:"Perc100"`
}

type PodStartupLatency struct {
	Latency LatencyMetric `json:"latency"`
}

func (l *PodStartupLatency) SummaryKind() string {
	return "PodStartupLatency"
}

func (l *PodStartupLatency) PrintHumanReadable() string {
	return PrettyPrintJSON(l)
}

func (l *PodStartupLatency) PrintJSON() string {
	return PrettyPrintJSON(PodStartupLatencyToPerfData(l))
}

type SchedulingLatency struct {
	Scheduling LatencyMetric `json:"scheduling"`
	Binding    LatencyMetric `json:"binding"`
	Total      LatencyMetric `json:"total"`
}

func (l *SchedulingLatency) SummaryKind() string {
	return "SchedulingLatency"
}

func (l *SchedulingLatency) PrintHumanReadable() string {
	return PrettyPrintJSON(l)
}

func (l *SchedulingLatency) PrintJSON() string {
	return PrettyPrintJSON(l)
}

type SaturationTime struct {
	TimeToSaturate time.Duration `json:"timeToSaturate"`
	NumberOfNodes  int           `json:"numberOfNodes"`
	NumberOfPods   int           `json:"numberOfPods"`
	Throughput     float32       `json:"throughput"`
}

type APICall struct {
	Resource    string        `json:"resource"`
	Subresource string        `json:"subresource"`
	Verb        string        `json:"verb"`
	Scope       string        `json:"scope"`
	Latency     LatencyMetric `json:"latency"`
	Count       int           `json:"count"`
}

type APIResponsiveness struct {
	APICalls []APICall `json:"apicalls"`
}

func (a *APIResponsiveness) SummaryKind() string {
	return "APIResponsiveness"
}

func (a *APIResponsiveness) PrintHumanReadable() string {
	return PrettyPrintJSON(a)
}

func (a *APIResponsiveness) PrintJSON() string {
	return PrettyPrintJSON(ApiCallToPerfData(a))
}

func (a *APIResponsiveness) Len() int { return len(a.APICalls) }
func (a *APIResponsiveness) Swap(i, j int) {
	a.APICalls[i], a.APICalls[j] = a.APICalls[j], a.APICalls[i]
}
func (a *APIResponsiveness) Less(i, j int) bool {
	return a.APICalls[i].Latency.Perc99 < a.APICalls[j].Latency.Perc99
}

// Set request latency for a particular quantile in the APICall metric entry (creating one if necessary).
// 0 <= quantile <=1 (e.g. 0.95 is 95%tile, 0.5 is median)
// Only 0.5, 0.9 and 0.99 quantiles are supported.
func (a *APIResponsiveness) addMetricRequestLatency(resource, subresource, verb, scope string, quantile float64, latency time.Duration) {
	for i, apicall := range a.APICalls {
		if apicall.Resource == resource && apicall.Subresource == subresource && apicall.Verb == verb && apicall.Scope == scope {
			a.APICalls[i] = setQuantileAPICall(apicall, quantile, latency)
			return
		}
	}
	apicall := setQuantileAPICall(APICall{Resource: resource, Subresource: subresource, Verb: verb, Scope: scope}, quantile, latency)
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

// Add request count to the APICall metric entry (creating one if necessary).
func (a *APIResponsiveness) addMetricRequestCount(resource, subresource, verb, scope string, count int) {
	for i, apicall := range a.APICalls {
		if apicall.Resource == resource && apicall.Subresource == subresource && apicall.Verb == verb && apicall.Scope == scope {
			a.APICalls[i].Count += count
			return
		}
	}
	apicall := APICall{Resource: resource, Subresource: subresource, Verb: verb, Count: count, Scope: scope}
	a.APICalls = append(a.APICalls, apicall)
}

func readLatencyMetrics(c clientset.Interface) (*APIResponsiveness, error) {
	var a APIResponsiveness

	body, err := getMetrics(c)
	if err != nil {
		return nil, err
	}

	samples, err := extractMetricSamples(body)
	if err != nil {
		return nil, err
	}

	ignoredResources := sets.NewString("events")
	// TODO: figure out why we're getting non-capitalized proxy and fix this.
	ignoredVerbs := sets.NewString("WATCH", "WATCHLIST", "PROXY", "proxy", "CONNECT")

	for _, sample := range samples {
		// Example line:
		// apiserver_request_latencies_summary{resource="namespaces",verb="LIST",quantile="0.99"} 908
		// apiserver_request_count{resource="pods",verb="LIST",client="kubectl",code="200",contentType="json"} 233
		if sample.Metric[model.MetricNameLabel] != "apiserver_request_latencies_summary" &&
			sample.Metric[model.MetricNameLabel] != "apiserver_request_count" {
			continue
		}

		resource := string(sample.Metric["resource"])
		subresource := string(sample.Metric["subresource"])
		verb := string(sample.Metric["verb"])
		scope := string(sample.Metric["scope"])
		if ignoredResources.Has(resource) || ignoredVerbs.Has(verb) {
			continue
		}

		switch sample.Metric[model.MetricNameLabel] {
		case "apiserver_request_latencies_summary":
			latency := sample.Value
			quantile, err := strconv.ParseFloat(string(sample.Metric[model.QuantileLabel]), 64)
			if err != nil {
				return nil, err
			}
			a.addMetricRequestLatency(resource, subresource, verb, scope, quantile, time.Duration(int64(latency))*time.Microsecond)
		case "apiserver_request_count":
			count := sample.Value
			a.addMetricRequestCount(resource, subresource, verb, scope, int(count))

		}
	}

	return &a, err
}

// Prints top five summary metrics for request types with latency and returns
// number of such request types above threshold. We use a higher threshold for
// list calls if nodeCount is above a given threshold (i.e. cluster is big).
func HighLatencyRequests(c clientset.Interface, nodeCount int) (int, *APIResponsiveness, error) {
	isBigCluster := (nodeCount > bigClusterNodeCountThreshold)
	metrics, err := readLatencyMetrics(c)
	if err != nil {
		return 0, metrics, err
	}
	sort.Sort(sort.Reverse(metrics))
	badMetrics := 0
	top := 5
	for i := range metrics.APICalls {
		latency := metrics.APICalls[i].Latency.Perc99
		isListCall := (metrics.APICalls[i].Verb == "LIST")
		isClusterScopedCall := (metrics.APICalls[i].Scope == "cluster")
		isBad := false
		latencyThreshold := apiCallLatencyThreshold
		if isListCall && isBigCluster {
			latencyThreshold = apiListCallLatencyThreshold
			if isClusterScopedCall {
				latencyThreshold = apiClusterScopeListCallThreshold
			}
		}
		if latency > latencyThreshold {
			isBad = true
			badMetrics++
		}
		if top > 0 || isBad {
			top--
			prefix := ""
			if isBad {
				prefix = "WARNING "
			}
			Logf("%vTop latency metric: %+v", prefix, metrics.APICalls[i])
		}
	}
	return badMetrics, metrics, nil
}

// Verifies whether 50, 90 and 99th percentiles of PodStartupLatency are
// within the threshold.
func VerifyPodStartupLatency(latency *PodStartupLatency) error {
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
func ResetMetrics(c clientset.Interface) error {
	Logf("Resetting latency metrics in apiserver...")
	body, err := c.CoreV1().RESTClient().Delete().AbsPath("/metrics").DoRaw()
	if err != nil {
		return err
	}
	if string(body) != "metrics reset\n" {
		return fmt.Errorf("Unexpected response: %q", string(body))
	}
	return nil
}

// Retrieves metrics information.
func getMetrics(c clientset.Interface) (string, error) {
	body, err := c.CoreV1().RESTClient().Get().AbsPath("/metrics").DoRaw()
	if err != nil {
		return "", err
	}
	return string(body), nil
}

// Retrieves scheduler metrics information.
func getSchedulingLatency(c clientset.Interface) (*SchedulingLatency, error) {
	result := SchedulingLatency{}

	// Check if master Node is registered
	nodes, err := c.CoreV1().Nodes().List(metav1.ListOptions{})
	ExpectNoError(err)

	subResourceProxyAvailable, err := ServerVersionGTE(SubResourcePodProxyVersion, c.Discovery())
	if err != nil {
		return nil, err
	}

	var data string
	var masterRegistered = false
	for _, node := range nodes.Items {
		if system.IsMasterNode(node.Name) {
			masterRegistered = true
		}
	}
	if masterRegistered {
		ctx, cancel := context.WithTimeout(context.Background(), SingleCallTimeout)
		defer cancel()

		var rawData []byte
		if subResourceProxyAvailable {
			rawData, err = c.CoreV1().RESTClient().Get().
				Context(ctx).
				Namespace(metav1.NamespaceSystem).
				Resource("pods").
				Name(fmt.Sprintf("kube-scheduler-%v:%v", TestContext.CloudConfig.MasterName, ports.SchedulerPort)).
				SubResource("proxy").
				Suffix("metrics").
				Do().Raw()
		} else {
			rawData, err = c.CoreV1().RESTClient().Get().
				Context(ctx).
				Prefix("proxy").
				Namespace(metav1.NamespaceSystem).
				SubResource("pods").
				Name(fmt.Sprintf("kube-scheduler-%v:%v", TestContext.CloudConfig.MasterName, ports.SchedulerPort)).
				Suffix("metrics").
				Do().Raw()
		}

		ExpectNoError(err)
		data = string(rawData)
	} else {
		// If master is not registered fall back to old method of using SSH.
		if TestContext.Provider == "gke" {
			Logf("Not grabbing scheduler metrics through master SSH: unsupported for gke")
			return nil, nil
		}
		cmd := "curl http://localhost:10251/metrics"
		sshResult, err := SSH(cmd, GetMasterHost()+":22", TestContext.Provider)
		if err != nil || sshResult.Code != 0 {
			return &result, fmt.Errorf("unexpected error (code: %d) in ssh connection to master: %#v", sshResult.Code, err)
		}
		data = sshResult.Stdout
	}
	samples, err := extractMetricSamples(data)
	if err != nil {
		return nil, err
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
			return nil, err
		}
		setQuantile(metric, quantile, time.Duration(int64(latency))*time.Microsecond)
	}
	return &result, nil
}

// Verifies (currently just by logging them) the scheduling latencies.
func VerifySchedulerLatency(c clientset.Interface) (*SchedulingLatency, error) {
	latency, err := getSchedulingLatency(c)
	if err != nil {
		return nil, err
	}
	return latency, nil
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
	perc100 := latencies[length-1].Latency
	return LatencyMetric{Perc50: perc50, Perc90: perc90, Perc99: perc99, Perc100: perc100}
}

// LogSuspiciousLatency logs metrics/docker errors from all nodes that had slow startup times
// If latencyDataLag is nil then it will be populated from latencyData
func LogSuspiciousLatency(latencyData []PodLatencyData, latencyDataLag []PodLatencyData, nodeCount int, c clientset.Interface) {
	if latencyDataLag == nil {
		latencyDataLag = latencyData
	}
	for _, l := range latencyData {
		if l.Latency > NodeStartupThreshold {
			HighLatencyKubeletOperations(c, 1*time.Second, l.Node, Logf)
		}
	}
	Logf("Approx throughput: %v pods/min",
		float64(nodeCount)/(latencyDataLag[len(latencyDataLag)-1].Latency.Minutes()))
}

func PrintLatencies(latencies []PodLatencyData, header string) {
	metrics := ExtractLatencyMetrics(latencies)
	Logf("10%% %s: %v", header, latencies[(len(latencies)*9)/10:])
	Logf("perc50: %v, perc90: %v, perc99: %v", metrics.Perc50, metrics.Perc90, metrics.Perc99)
}

func (m *MetricsForE2E) computeClusterAutoscalerMetricsDelta(before metrics.MetricsCollection) {
	if beforeSamples, found := before.ClusterAutoscalerMetrics[caFunctionMetric]; found {
		if afterSamples, found := m.ClusterAutoscalerMetrics[caFunctionMetric]; found {
			beforeSamplesMap := make(map[string]*model.Sample)
			for _, bSample := range beforeSamples {
				beforeSamplesMap[makeKey(bSample.Metric[caFunctionMetricLabel], bSample.Metric["le"])] = bSample
			}
			for _, aSample := range afterSamples {
				if bSample, found := beforeSamplesMap[makeKey(aSample.Metric[caFunctionMetricLabel], aSample.Metric["le"])]; found {
					aSample.Value = aSample.Value - bSample.Value
				}

			}
		}
	}
}

func makeKey(a, b model.LabelValue) string {
	return string(a) + "___" + string(b)
}
