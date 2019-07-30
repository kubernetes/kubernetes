/*
Copyright 2019 The Kubernetes Authors.

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
	"math"
	"sort"
	"strconv"
	"strings"
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/sets"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/pkg/master/ports"
	schedulermetric "k8s.io/kubernetes/pkg/scheduler/metrics"
	"k8s.io/kubernetes/pkg/util/system"
	e2elog "k8s.io/kubernetes/test/e2e/framework/log"
	e2essh "k8s.io/kubernetes/test/e2e/framework/ssh"

	"github.com/onsi/gomega"

	"github.com/prometheus/common/model"
)

const (
	// SingleCallTimeout is how long to try single API calls (like 'get' or 'list'). Used to prevent
	// transient failures from failing tests.
	// TODO: client should not apply this timeout to Watch calls. Increased from 30s until that is fixed.
	SingleCallTimeout = 5 * time.Minute

	// nodeStartupThreshold is a rough estimate of the time allocated for a pod to start on a node.
	nodeStartupThreshold = 4 * time.Second

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
)

var schedulingLatencyMetricName = model.LabelValue(schedulermetric.SchedulerSubsystem + "_" + schedulermetric.SchedulingLatencyName)

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
func sendRestRequestToScheduler(c clientset.Interface, op, provider, cloudMasterName, masterHostname string) (string, error) {
	opUpper := strings.ToUpper(op)
	if opUpper != "GET" && opUpper != "DELETE" {
		return "", fmt.Errorf("Unknown REST request")
	}

	nodes, err := c.CoreV1().Nodes().List(metav1.ListOptions{})
	// The following 4 lines are intended to replace framework.ExpectNoError(err)
	if err != nil {
		e2elog.Logf("Unexpected error occurred: %v", err)
	}
	gomega.ExpectWithOffset(1, err).NotTo(gomega.HaveOccurred())

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
			Name(fmt.Sprintf("kube-scheduler-%v:%v", cloudMasterName, ports.InsecureSchedulerPort)).
			SubResource("proxy").
			Suffix("metrics").
			Do().Raw()

		// The following 4 lines are intended to replace
		// framework.ExpectNoError(err).
		if err != nil {
			e2elog.Logf("Unexpected error occurred: %v", err)
		}
		gomega.ExpectWithOffset(1, err).NotTo(gomega.HaveOccurred())
		responseText = string(body)
	} else {
		// If master is not registered fall back to old method of using SSH.
		if provider == "gke" || provider == "eks" {
			e2elog.Logf("Not grabbing scheduler metrics through master SSH: unsupported for %s", provider)
			return "", nil
		}

		cmd := "curl -X " + opUpper + " http://localhost:10251/metrics"
		sshResult, err := e2essh.SSH(cmd, masterHostname+":22", provider)
		if err != nil || sshResult.Code != 0 {
			return "", fmt.Errorf("unexpected error (code: %d) in ssh connection to master: %#v", sshResult.Code, err)
		}
		responseText = sshResult.Stdout
	}
	return responseText, nil
}

// Retrieves scheduler latency metrics.
func getSchedulingLatency(c clientset.Interface, provider, cloudMasterName, masterHostname string) (*SchedulingMetrics, error) {
	result := SchedulingMetrics{}
	data, err := sendRestRequestToScheduler(c, "GET", provider, cloudMasterName, masterHostname)
	if err != nil {
		return nil, err
	}

	samples, err := extractMetricSamples(data)
	if err != nil {
		return nil, err
	}

	for _, sample := range samples {
		if sample.Metric[model.MetricNameLabel] != schedulingLatencyMetricName {
			continue
		}

		var metric *LatencyMetric
		switch sample.Metric[schedulermetric.OperationLabel] {
		case schedulermetric.PredicateEvaluation:
			metric = &result.PredicateEvaluationLatency
		case schedulermetric.PriorityEvaluation:
			metric = &result.PriorityEvaluationLatency
		case schedulermetric.PreemptionEvaluation:
			metric = &result.PreemptionEvaluationLatency
		case schedulermetric.Binding:
			metric = &result.BindingLatency
		}
		if metric == nil {
			continue
		}

		quantile, err := strconv.ParseFloat(string(sample.Metric[model.QuantileLabel]), 64)
		if err != nil {
			return nil, err
		}
		setQuantile(metric, quantile, time.Duration(int64(float64(sample.Value)*float64(time.Second))))
	}
	return &result, nil
}

// VerifySchedulerLatency verifies (currently just by logging them) the scheduling latencies.
func VerifySchedulerLatency(c clientset.Interface, provider, cloudMasterName, masterHostname string) (*SchedulingMetrics, error) {
	latency, err := getSchedulingLatency(c, provider, cloudMasterName, masterHostname)
	if err != nil {
		return nil, err
	}
	return latency, nil
}

// ResetSchedulerMetrics sends a DELETE request to kube-scheduler for resetting metrics.
func ResetSchedulerMetrics(c clientset.Interface, provider, cloudMasterName, masterHostname string) error {
	responseText, err := sendRestRequestToScheduler(c, "DELETE", provider, cloudMasterName, masterHostname)
	if err != nil {
		return fmt.Errorf("Unexpected response: %q, %v", responseText, err)
	}
	return nil
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

// LatencySlice is an array of PodLatencyData which encapsulates pod startup latency information.
type LatencySlice []PodLatencyData

func (a LatencySlice) Len() int           { return len(a) }
func (a LatencySlice) Swap(i, j int)      { a[i], a[j] = a[j], a[i] }
func (a LatencySlice) Less(i, j int) bool { return a[i].Latency < a[j].Latency }

// ExtractLatencyMetrics returns latency metrics for each percentile(50th, 90th and 99th).
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
		if l.Latency > nodeStartupThreshold {
			HighLatencyKubeletOperations(c, 1*time.Second, l.Node, e2elog.Logf)
		}
	}
	e2elog.Logf("Approx throughput: %v pods/min",
		float64(nodeCount)/(latencyDataLag[len(latencyDataLag)-1].Latency.Minutes()))
}

// PrintLatencies outputs latencies to log with readable format.
func PrintLatencies(latencies []PodLatencyData, header string) {
	metrics := ExtractLatencyMetrics(latencies)
	e2elog.Logf("10%% %s: %v", header, latencies[(len(latencies)*9)/10:])
	e2elog.Logf("perc50: %v, perc90: %v, perc99: %v", metrics.Perc50, metrics.Perc90, metrics.Perc99)
}
