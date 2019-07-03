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
	"reflect"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/sets"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/pkg/master/ports"
	"k8s.io/kubernetes/pkg/util/system"
	e2elog "k8s.io/kubernetes/test/e2e/framework/log"
	e2essh "k8s.io/kubernetes/test/e2e/framework/ssh"

	"github.com/prometheus/common/expfmt"
	"github.com/prometheus/common/model"
)

const (
	// NodeStartupThreshold is a rough estimate of the time allocated for a pod to start on a node.
	NodeStartupThreshold = 4 * time.Second

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

// LatencyMetric is a struct for dashboard metrics.
type LatencyMetric struct {
	Perc50  time.Duration `json:"Perc50"`
	Perc90  time.Duration `json:"Perc90"`
	Perc99  time.Duration `json:"Perc99"`
	Perc100 time.Duration `json:"Perc100"`
}

// Histogram is a struct for managing histogram.
type Histogram struct {
	Labels  map[string]string `json:"labels"`
	Buckets map[string]int    `json:"buckets"`
}

// HistogramVec is an array of Histogram.
type HistogramVec []Histogram

func newHistogram(labels map[string]string) *Histogram {
	return &Histogram{
		Labels:  labels,
		Buckets: make(map[string]int),
	}
}

// EtcdMetrics is a struct for managing etcd metrics.
type EtcdMetrics struct {
	BackendCommitDuration     HistogramVec `json:"backendCommitDuration"`
	SnapshotSaveTotalDuration HistogramVec `json:"snapshotSaveTotalDuration"`
	PeerRoundTripTime         HistogramVec `json:"peerRoundTripTime"`
	WalFsyncDuration          HistogramVec `json:"walFsyncDuration"`
	MaxDatabaseSize           float64      `json:"maxDatabaseSize"`
}

func newEtcdMetrics() *EtcdMetrics {
	return &EtcdMetrics{
		BackendCommitDuration:     make(HistogramVec, 0),
		SnapshotSaveTotalDuration: make(HistogramVec, 0),
		PeerRoundTripTime:         make(HistogramVec, 0),
		WalFsyncDuration:          make(HistogramVec, 0),
	}
}

// SummaryKind returns the summary of etcd metrics.
func (l *EtcdMetrics) SummaryKind() string {
	return "EtcdMetrics"
}

// PrintHumanReadable returns etcd metrics with JSON format.
func (l *EtcdMetrics) PrintHumanReadable() string {
	return PrettyPrintJSON(l)
}

// PrintJSON returns etcd metrics with JSON format.
func (l *EtcdMetrics) PrintJSON() string {
	return PrettyPrintJSON(l)
}

// EtcdMetricsCollector is a struct for managing etcd metrics collector.
type EtcdMetricsCollector struct {
	stopCh  chan struct{}
	wg      *sync.WaitGroup
	metrics *EtcdMetrics
}

// NewEtcdMetricsCollector creates a new etcd metrics collector.
func NewEtcdMetricsCollector() *EtcdMetricsCollector {
	return &EtcdMetricsCollector{
		stopCh:  make(chan struct{}),
		wg:      &sync.WaitGroup{},
		metrics: newEtcdMetrics(),
	}
}

func getEtcdMetrics() ([]*model.Sample, error) {
	// Etcd is only exposed on localhost level. We are using ssh method
	if TestContext.Provider == "gke" || TestContext.Provider == "eks" {
		e2elog.Logf("Not grabbing etcd metrics through master SSH: unsupported for %s", TestContext.Provider)
		return nil, nil
	}

	cmd := "curl http://localhost:2379/metrics"
	sshResult, err := e2essh.SSH(cmd, GetMasterHost()+":22", TestContext.Provider)
	if err != nil || sshResult.Code != 0 {
		return nil, fmt.Errorf("unexpected error (code: %d) in ssh connection to master: %#v", sshResult.Code, err)
	}
	data := sshResult.Stdout

	return extractMetricSamples(data)
}

func getEtcdDatabaseSize() (float64, error) {
	samples, err := getEtcdMetrics()
	if err != nil {
		return 0, err
	}
	for _, sample := range samples {
		if sample.Metric[model.MetricNameLabel] == "etcd_debugging_mvcc_db_total_size_in_bytes" {
			return float64(sample.Value), nil
		}
	}
	return 0, fmt.Errorf("Couldn't find etcd database size metric")
}

// StartCollecting starts to collect etcd db size metric periodically
// and updates MaxDatabaseSize accordingly.
func (mc *EtcdMetricsCollector) StartCollecting(interval time.Duration) {
	mc.wg.Add(1)
	go func() {
		defer mc.wg.Done()
		for {
			select {
			case <-time.After(interval):
				dbSize, err := getEtcdDatabaseSize()
				if err != nil {
					e2elog.Logf("Failed to collect etcd database size")
					continue
				}
				mc.metrics.MaxDatabaseSize = math.Max(mc.metrics.MaxDatabaseSize, dbSize)
			case <-mc.stopCh:
				return
			}
		}
	}()
}

// StopAndSummarize stops etcd metrics collector and summarizes the metrics.
func (mc *EtcdMetricsCollector) StopAndSummarize() error {
	close(mc.stopCh)
	mc.wg.Wait()

	// Do some one-off collection of metrics.
	samples, err := getEtcdMetrics()
	if err != nil {
		return err
	}
	for _, sample := range samples {
		switch sample.Metric[model.MetricNameLabel] {
		case "etcd_disk_backend_commit_duration_seconds_bucket":
			convertSampleToBucket(sample, &mc.metrics.BackendCommitDuration)
		case "etcd_debugging_snap_save_total_duration_seconds_bucket":
			convertSampleToBucket(sample, &mc.metrics.SnapshotSaveTotalDuration)
		case "etcd_disk_wal_fsync_duration_seconds_bucket":
			convertSampleToBucket(sample, &mc.metrics.WalFsyncDuration)
		case "etcd_network_peer_round_trip_time_seconds_bucket":
			convertSampleToBucket(sample, &mc.metrics.PeerRoundTripTime)
		}
	}
	return nil
}

// GetMetrics returns metrics of etcd metrics collector.
func (mc *EtcdMetricsCollector) GetMetrics() *EtcdMetrics {
	return mc.metrics
}

// APICall is a struct for managing API call.
type APICall struct {
	Resource    string        `json:"resource"`
	Subresource string        `json:"subresource"`
	Verb        string        `json:"verb"`
	Scope       string        `json:"scope"`
	Latency     LatencyMetric `json:"latency"`
	Count       int           `json:"count"`
}

// APIResponsiveness is a struct for managing multiple API calls.
type APIResponsiveness struct {
	APICalls []APICall `json:"apicalls"`
}

// SummaryKind returns the summary of API responsiveness.
func (a *APIResponsiveness) SummaryKind() string {
	return "APIResponsiveness"
}

// PrintHumanReadable returns metrics with JSON format.
func (a *APIResponsiveness) PrintHumanReadable() string {
	return PrettyPrintJSON(a)
}

// PrintJSON returns metrics of PerfData(50, 90 and 99th percentiles) with JSON format.
func (a *APIResponsiveness) PrintJSON() string {
	return PrettyPrintJSON(APICallToPerfData(a))
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
		// apiserver_request_total{resource="pods",verb="LIST",client="kubectl",code="200",contentType="json"} 233
		if sample.Metric[model.MetricNameLabel] != "apiserver_request_latencies_summary" &&
			sample.Metric[model.MetricNameLabel] != "apiserver_request_total" {
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
		case "apiserver_request_total":
			count := sample.Value
			a.addMetricRequestCount(resource, subresource, verb, scope, int(count))

		}
	}

	return &a, err
}

// HighLatencyRequests prints top five summary metrics for request types with latency and returns
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
			e2elog.Logf("%vTop latency metric: %+v", prefix, metrics.APICalls[i])
		}
	}
	return badMetrics, metrics, nil
}

// VerifyLatencyWithinThreshold verifies whether 50, 90 and 99th percentiles of a latency metric are
// within the expected threshold.
func VerifyLatencyWithinThreshold(threshold, actual LatencyMetric, metricName string) error {
	if actual.Perc50 > threshold.Perc50 {
		return fmt.Errorf("too high %v latency 50th percentile: %v", metricName, actual.Perc50)
	}
	if actual.Perc90 > threshold.Perc90 {
		return fmt.Errorf("too high %v latency 90th percentile: %v", metricName, actual.Perc90)
	}
	if actual.Perc99 > threshold.Perc99 {
		return fmt.Errorf("too high %v latency 99th percentile: %v", metricName, actual.Perc99)
	}
	return nil
}

// ResetMetrics resets latency metrics in apiserver.
func ResetMetrics(c clientset.Interface) error {
	e2elog.Logf("Resetting latency metrics in apiserver...")
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

// Sends REST request to kube scheduler metrics
func sendRestRequestToScheduler(c clientset.Interface, op string) (string, error) {
	opUpper := strings.ToUpper(op)
	if opUpper != "GET" && opUpper != "DELETE" {
		return "", fmt.Errorf("Unknown REST request")
	}

	nodes, err := c.CoreV1().Nodes().List(metav1.ListOptions{})
	ExpectNoError(err)

	var masterRegistered = false
	for _, node := range nodes.Items {
		if system.IsMasterNode(node.Name) {
			masterRegistered = true
		}
	}

	var responseText string
	if masterRegistered {
		ctx, cancel := context.WithTimeout(context.Background(), SingleCallTimeout)
		defer cancel()

		body, err := c.CoreV1().RESTClient().Verb(opUpper).
			Context(ctx).
			Namespace(metav1.NamespaceSystem).
			Resource("pods").
			Name(fmt.Sprintf("kube-scheduler-%v:%v", TestContext.CloudConfig.MasterName, ports.InsecureSchedulerPort)).
			SubResource("proxy").
			Suffix("metrics").
			Do().Raw()

		ExpectNoError(err)
		responseText = string(body)
	} else {
		// If master is not registered fall back to old method of using SSH.
		if TestContext.Provider == "gke" || TestContext.Provider == "eks" {
			e2elog.Logf("Not grabbing scheduler metrics through master SSH: unsupported for %s", TestContext.Provider)
			return "", nil
		}

		cmd := "curl -X " + opUpper + " http://localhost:10251/metrics"
		sshResult, err := e2essh.SSH(cmd, GetMasterHost()+":22", TestContext.Provider)
		if err != nil || sshResult.Code != 0 {
			return "", fmt.Errorf("unexpected error (code: %d) in ssh connection to master: %#v", sshResult.Code, err)
		}
		responseText = sshResult.Stdout
	}
	return responseText, nil
}

// ResetSchedulerMetrics sends a DELETE request to kube-scheduler for resetting metrics.
func ResetSchedulerMetrics(c clientset.Interface) error {
	responseText, err := sendRestRequestToScheduler(c, "DELETE")
	if err != nil {
		return fmt.Errorf("Unexpected response: %q", responseText)
	}
	return nil
}

func convertSampleToBucket(sample *model.Sample, h *HistogramVec) {
	labels := make(map[string]string)
	for k, v := range sample.Metric {
		if k != "le" {
			labels[string(k)] = string(v)
		}
	}
	var hist *Histogram
	for i := range *h {
		if reflect.DeepEqual(labels, (*h)[i].Labels) {
			hist = &((*h)[i])
			break
		}
	}
	if hist == nil {
		hist = newHistogram(labels)
		*h = append(*h, *hist)
	}
	hist.Buckets[string(sample.Metric["le"])] = int(sample.Value)
}

// PrettyPrintJSON converts metrics to JSON format.
func PrettyPrintJSON(metrics interface{}) string {
	output := &bytes.Buffer{}
	if err := json.NewEncoder(output).Encode(metrics); err != nil {
		e2elog.Logf("Error building encoder: %v", err)
		return ""
	}
	formatted := &bytes.Buffer{}
	if err := json.Indent(formatted, output.Bytes(), "", "  "); err != nil {
		e2elog.Logf("Error indenting: %v", err)
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
