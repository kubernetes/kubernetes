/*
Copyright 2018 The Kubernetes Authors.

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

package monitoring

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"time"

	"github.com/prometheus/common/model"

	"github.com/onsi/ginkgo"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/test/e2e/common"
	"k8s.io/kubernetes/test/e2e/framework"
	instrumentation "k8s.io/kubernetes/test/e2e/instrumentation/common"
)

const (
	prometheusQueryStep                = time.Minute
	prometheusMetricErrorTolerance     = 0.25
	prometheusMetricValidationDuration = time.Minute * 2
	prometheusRate                     = time.Minute * 2
	prometheusRequiredNodesUpDuration  = time.Minute * 5
	prometheusService                  = "prometheus"
	prometheusSleepBetweenAttempts     = time.Second * 30
	prometheusTestTimeout              = time.Minute * 5
	customMetricValue                  = 1000
	targetCPUUsage                     = 0.1
)

var _ = instrumentation.SIGDescribe("[Feature:PrometheusMonitoring] Prometheus", func() {
	ginkgo.BeforeEach(func() {
		framework.SkipUnlessPrometheusMonitoringIsEnabled()
	})

	f := framework.NewDefaultFramework("prometheus-monitoring")
	ginkgo.It("should scrape container metrics from all nodes.", func() {
		expectedNodes, err := getAllNodes(f.ClientSet)
		framework.ExpectNoError(err)
		retryUntilSucceeds(func() error {
			return validateMetricAvailableForAllNodes(f.ClientSet, `container_cpu_usage_seconds_total`, expectedNodes)
		}, prometheusTestTimeout)
	})
	ginkgo.It("should successfully scrape all targets", func() {
		retryUntilSucceeds(func() error {
			return validateAllActiveTargetsAreHealthy(f.ClientSet)
		}, prometheusTestTimeout)
	})
	ginkgo.It("should contain correct container CPU metric.", func() {
		query := prometheusCPUQuery(f.Namespace.Name, "prometheus-cpu-consumer", prometheusRate)
		consumer := consumeCPUResources(f, "prometheus-cpu-consumer", targetCPUUsage*1000)
		defer consumer.CleanUp()
		retryUntilSucceeds(func() error {
			return validateQueryReturnsCorrectValues(f.ClientSet, query, targetCPUUsage, 3, prometheusMetricErrorTolerance)
		}, prometheusTestTimeout)
	})
	ginkgo.It("should scrape metrics from annotated pods.", func() {
		query := prometheusPodCustomMetricQuery(f.Namespace.Name, "prometheus-custom-pod-metric")
		consumer := exportCustomMetricFromPod(f, "prometheus-custom-pod-metric", customMetricValue)
		defer consumer.CleanUp()
		retryUntilSucceeds(func() error {
			return validateQueryReturnsCorrectValues(f.ClientSet, query, customMetricValue, 1, prometheusMetricErrorTolerance)
		}, prometheusTestTimeout)
	})
	ginkgo.It("should scrape metrics from annotated services.", func() {
		query := prometheusServiceCustomMetricQuery(f.Namespace.Name, "prometheus-custom-service-metric")
		consumer := exportCustomMetricFromService(f, "prometheus-custom-service-metric", customMetricValue)
		defer consumer.CleanUp()
		retryUntilSucceeds(func() error {
			return validateQueryReturnsCorrectValues(f.ClientSet, query, customMetricValue, 1, prometheusMetricErrorTolerance)
		}, prometheusTestTimeout)
	})
})

func prometheusCPUQuery(namespace, podNamePrefix string, rate time.Duration) string {
	return fmt.Sprintf(`sum(irate(container_cpu_usage_seconds_total{namespace="%v",pod_name=~"%v.*",image!=""}[%vm]))`,
		namespace, podNamePrefix, int64(rate.Minutes()))
}

func prometheusServiceCustomMetricQuery(namespace, service string) string {
	return fmt.Sprintf(`sum(QPS{kubernetes_namespace="%v",kubernetes_name="%v"})`, namespace, service)
}

func prometheusPodCustomMetricQuery(namespace, podNamePrefix string) string {
	return fmt.Sprintf(`sum(QPS{kubernetes_namespace="%s",kubernetes_pod_name=~"%s.*"})`, namespace, podNamePrefix)
}

func consumeCPUResources(f *framework.Framework, consumerName string, cpuUsage int) *common.ResourceConsumer {
	return common.NewDynamicResourceConsumer(consumerName, f.Namespace.Name, common.KindDeployment, 1, cpuUsage,
		memoryUsed, 0, int64(cpuUsage), memoryLimit, f.ClientSet, f.InternalClientset, f.ScalesGetter)
}

func exportCustomMetricFromPod(f *framework.Framework, consumerName string, metricValue int) *common.ResourceConsumer {
	podAnnotations := map[string]string{
		"prometheus.io/scrape": "true",
		"prometheus.io/path":   "/metrics",
		"prometheus.io/port":   "8080",
	}
	return common.NewMetricExporter(consumerName, f.Namespace.Name, podAnnotations, nil, metricValue, f.ClientSet, f.InternalClientset, f.ScalesGetter)
}

func exportCustomMetricFromService(f *framework.Framework, consumerName string, metricValue int) *common.ResourceConsumer {
	serviceAnnotations := map[string]string{
		"prometheus.io/scrape": "true",
		"prometheus.io/path":   "/metrics",
		"prometheus.io/port":   "8080",
	}
	return common.NewMetricExporter(consumerName, f.Namespace.Name, nil, serviceAnnotations, metricValue, f.ClientSet, f.InternalClientset, f.ScalesGetter)
}

func validateMetricAvailableForAllNodes(c clientset.Interface, metric string, expectedNodesNames []string) error {
	instanceLabels, err := getInstanceLabelsAvailableForMetric(c, prometheusRequiredNodesUpDuration, metric)
	if err != nil {
		return err
	}
	nodesWithMetric := make(map[string]bool)
	for _, instance := range instanceLabels {
		nodesWithMetric[instance] = true
	}
	missedNodesCount := 0
	for _, nodeName := range expectedNodesNames {
		if _, found := nodesWithMetric[nodeName]; !found {
			missedNodesCount++
		}
	}
	if missedNodesCount > 0 {
		return fmt.Errorf("Metric not found for %v out of %v nodes", missedNodesCount, len(expectedNodesNames))
	}
	return nil
}

func validateAllActiveTargetsAreHealthy(c clientset.Interface) error {
	discovery, err := fetchPrometheusTargetDiscovery(c)
	if err != nil {
		return err
	}
	if len(discovery.ActiveTargets) == 0 {
		return fmt.Errorf("Prometheus is not scraping any targets, at least one target is required")
	}
	for _, target := range discovery.ActiveTargets {
		if target.Health != HealthGood {
			return fmt.Errorf("Target health not good. Target: %v", target)
		}
	}
	return nil
}

func validateQueryReturnsCorrectValues(c clientset.Interface, query string, expectedValue float64, minSamplesCount int, errorTolerance float64) error {
	samples, err := fetchQueryValues(c, query, prometheusMetricValidationDuration)
	if err != nil {
		return err
	}
	if len(samples) < minSamplesCount {
		return fmt.Errorf("Not enough samples for query '%v', got %v", query, samples)
	}
	framework.Logf("Executed query '%v' returned %v", query, samples)
	for _, value := range samples {
		error := math.Abs(value-expectedValue) / expectedValue
		if error >= errorTolerance {
			return fmt.Errorf("Query result values outside expected value tolerance. Expected error below %v, got %v", errorTolerance, error)
		}
	}
	return nil
}

func fetchQueryValues(c clientset.Interface, query string, duration time.Duration) ([]float64, error) {
	now := time.Now()
	response, err := queryPrometheus(c, query, now.Add(-duration), now, prometheusQueryStep)
	if err != nil {
		return nil, err
	}
	m, ok := response.(model.Matrix)
	if !ok {
		return nil, fmt.Errorf("Expected matric response, got: %T", response)
	}
	values := make([]float64, 0)
	for _, stream := range m {
		for _, sample := range stream.Values {
			values = append(values, float64(sample.Value))
		}
	}
	return values, nil
}

func getInstanceLabelsAvailableForMetric(c clientset.Interface, duration time.Duration, metric string) ([]string, error) {
	var instance model.LabelValue
	now := time.Now()
	query := fmt.Sprintf(`sum(%v)by(instance)`, metric)
	result, err := queryPrometheus(c, query, now.Add(-duration), now, prometheusQueryStep)
	if err != nil {
		return nil, err
	}
	instanceLabels := make([]string, 0)
	m, ok := result.(model.Matrix)
	if !ok {
		framework.Failf("Expected matrix response for query '%v', got: %T", query, result)
		return instanceLabels, nil
	}
	for _, stream := range m {
		if instance, ok = stream.Metric["instance"]; !ok {
			continue
		}
		instanceLabels = append(instanceLabels, string(instance))
	}
	return instanceLabels, nil
}

func fetchPrometheusTargetDiscovery(c clientset.Interface) (TargetDiscovery, error) {
	ctx, cancel := context.WithTimeout(context.Background(), framework.SingleCallTimeout)
	defer cancel()

	response, err := c.CoreV1().RESTClient().Get().
		Context(ctx).
		Namespace("kube-system").
		Resource("services").
		Name(prometheusService+":9090").
		SubResource("proxy").
		Suffix("api", "v1", "targets").
		Do().
		Raw()
	var qres promTargetsResponse
	if err != nil {
		framework.Logf(string(response))
		return qres.Data, err
	}
	err = json.Unmarshal(response, &qres)

	return qres.Data, nil
}

type promTargetsResponse struct {
	Status string          `json:"status"`
	Data   TargetDiscovery `json:"data"`
}

// TargetDiscovery has all the active targets.
type TargetDiscovery struct {
	ActiveTargets  []*Target        `json:"activeTargets"`
	DroppedTargets []*DroppedTarget `json:"droppedTargets"`
}

// Target has the information for one target.
type Target struct {
	DiscoveredLabels map[string]string `json:"discoveredLabels"`
	Labels           map[string]string `json:"labels"`

	ScrapeURL string `json:"scrapeUrl"`

	LastError  string       `json:"lastError"`
	LastScrape time.Time    `json:"lastScrape"`
	Health     TargetHealth `json:"health"`
}

// DroppedTarget has the information for one target that was dropped during relabelling.
type DroppedTarget struct {
	// Labels before any processing.
	DiscoveredLabels map[string]string `json:"discoveredLabels"`
}

// The possible health states of a target based on the last performed scrape.
const (
	HealthUnknown TargetHealth = "unknown"
	HealthGood    TargetHealth = "up"
	HealthBad     TargetHealth = "down"
)

// TargetHealth describes the health state of a target.
type TargetHealth string

func queryPrometheus(c clientset.Interface, query string, start, end time.Time, step time.Duration) (model.Value, error) {
	ctx, cancel := context.WithTimeout(context.Background(), framework.SingleCallTimeout)
	defer cancel()

	response, err := c.CoreV1().RESTClient().Get().
		Context(ctx).
		Namespace("kube-system").
		Resource("services").
		Name(prometheusService+":9090").
		SubResource("proxy").
		Suffix("api", "v1", "query_range").
		Param("query", query).
		Param("start", fmt.Sprintf("%v", start.Unix())).
		Param("end", fmt.Sprintf("%v", end.Unix())).
		Param("step", fmt.Sprintf("%vs", step.Seconds())).
		Do().
		Raw()
	if err != nil {
		framework.Logf(string(response))
		return nil, err
	}
	var qres promQueryResponse
	err = json.Unmarshal(response, &qres)

	return model.Value(qres.Data.v), err
}

type promQueryResponse struct {
	Status string       `json:"status"`
	Data   responseData `json:"data"`
}

type responseData struct {
	Type   model.ValueType `json:"resultType"`
	Result interface{}     `json:"result"`

	// The decoded value.
	v model.Value
}

func (qr *responseData) UnmarshalJSON(b []byte) error {
	v := struct {
		Type   model.ValueType `json:"resultType"`
		Result json.RawMessage `json:"result"`
	}{}

	err := json.Unmarshal(b, &v)
	if err != nil {
		return err
	}

	switch v.Type {
	case model.ValScalar:
		var sv model.Scalar
		err = json.Unmarshal(v.Result, &sv)
		qr.v = &sv

	case model.ValVector:
		var vv model.Vector
		err = json.Unmarshal(v.Result, &vv)
		qr.v = vv

	case model.ValMatrix:
		var mv model.Matrix
		err = json.Unmarshal(v.Result, &mv)
		qr.v = mv

	default:
		err = fmt.Errorf("unexpected value type %q", v.Type)
	}
	return err
}

func retryUntilSucceeds(validator func() error, timeout time.Duration) {
	startTime := time.Now()
	var err error
	for {
		err = validator()
		if err == nil {
			return
		}
		if time.Since(startTime) >= timeout {
			break
		}
		framework.Logf(err.Error())
		time.Sleep(prometheusSleepBetweenAttempts)
	}
	framework.Failf(err.Error())
}

func getAllNodes(c clientset.Interface) ([]string, error) {
	nodeList, err := c.CoreV1().Nodes().List(metav1.ListOptions{})
	if err != nil {
		return nil, err
	}
	result := []string{}
	for _, node := range nodeList.Items {
		result = append(result, node.Name)
	}
	return result, nil
}
