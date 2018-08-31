/*
Copyright 2017 The Kubernetes Authors.

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
	"testing"
	"time"

	autoscalingapi "k8s.io/api/autoscaling/v2beta2"
	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/meta/testrestmapper"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	core "k8s.io/client-go/testing"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	_ "k8s.io/kubernetes/pkg/apis/extensions/install"
	cmapi "k8s.io/metrics/pkg/apis/custom_metrics/v1beta2"
	emapi "k8s.io/metrics/pkg/apis/external_metrics/v1beta1"
	metricsapi "k8s.io/metrics/pkg/apis/metrics/v1beta1"
	metricsfake "k8s.io/metrics/pkg/client/clientset/versioned/fake"
	cmfake "k8s.io/metrics/pkg/client/custom_metrics/fake"
	emfake "k8s.io/metrics/pkg/client/external_metrics/fake"

	"github.com/stretchr/testify/assert"
)

type restClientTestCase struct {
	desiredMetricValues PodMetricsInfo
	desiredError        error

	// "timestamps" here are actually the offset in minutes from a base timestamp
	targetTimestamp      int
	window               time.Duration
	reportedMetricPoints []metricPoint
	reportedPodMetrics   [][]int64
	singleObject         *autoscalingapi.CrossVersionObjectReference

	namespace           string
	selector            labels.Selector
	resourceName        v1.ResourceName
	metricName          string
	metricSelector      *metav1.LabelSelector
	metricLabelSelector labels.Selector
}

func (tc *restClientTestCase) prepareTestClient(t *testing.T) (*metricsfake.Clientset, *cmfake.FakeCustomMetricsClient, *emfake.FakeExternalMetricsClient) {
	namespace := "test-namespace"
	tc.namespace = namespace
	podNamePrefix := "test-pod"
	podLabels := map[string]string{"name": podNamePrefix}
	tc.selector = labels.SelectorFromSet(podLabels)

	// it's a resource test if we have a resource name
	isResource := len(tc.resourceName) > 0
	// it's an external test if we have a metric selector
	isExternal := tc.metricSelector != nil

	fakeMetricsClient := &metricsfake.Clientset{}
	fakeCMClient := &cmfake.FakeCustomMetricsClient{}
	fakeEMClient := &emfake.FakeExternalMetricsClient{}

	if isResource {
		fakeMetricsClient.AddReactor("list", "pods", func(action core.Action) (handled bool, ret runtime.Object, err error) {
			metrics := &metricsapi.PodMetricsList{}
			for i, containers := range tc.reportedPodMetrics {
				metric := metricsapi.PodMetrics{
					ObjectMeta: metav1.ObjectMeta{
						Name:      fmt.Sprintf("%s-%d", podNamePrefix, i),
						Namespace: namespace,
						Labels:    podLabels,
					},
					Timestamp:  metav1.Time{Time: offsetTimestampBy(tc.targetTimestamp)},
					Window:     metav1.Duration{Duration: tc.window},
					Containers: []metricsapi.ContainerMetrics{},
				}
				for j, cpu := range containers {
					cm := metricsapi.ContainerMetrics{
						Name: fmt.Sprintf("%s-%d-container-%d", podNamePrefix, i, j),
						Usage: v1.ResourceList{
							v1.ResourceCPU: *resource.NewMilliQuantity(
								cpu,
								resource.DecimalSI),
							v1.ResourceMemory: *resource.NewQuantity(
								int64(1024*1024),
								resource.BinarySI),
						},
					}
					metric.Containers = append(metric.Containers, cm)
				}
				metrics.Items = append(metrics.Items, metric)
			}
			return true, metrics, nil
		})
	} else if isExternal {
		fakeEMClient.AddReactor("list", "*", func(action core.Action) (handled bool, ret runtime.Object, err error) {
			listAction := action.(core.ListAction)
			assert.Equal(t, tc.metricName, listAction.GetResource().Resource, "the metric requested should have matched the one specified.")
			assert.Equal(t, tc.metricLabelSelector, listAction.GetListRestrictions().Labels, "the metric selector should have matched the one specified")

			metrics := emapi.ExternalMetricValueList{}
			for _, metricPoint := range tc.reportedMetricPoints {
				timestamp := offsetTimestampBy(metricPoint.timestamp)
				metric := emapi.ExternalMetricValue{
					Value:      *resource.NewMilliQuantity(int64(metricPoint.level), resource.DecimalSI),
					Timestamp:  metav1.Time{Time: timestamp},
					MetricName: tc.metricName,
				}
				metrics.Items = append(metrics.Items, metric)
			}
			return true, &metrics, nil
		})
	} else {
		fakeCMClient.AddReactor("get", "*", func(action core.Action) (handled bool, ret runtime.Object, err error) {
			getForAction := action.(cmfake.GetForAction)
			assert.Equal(t, tc.metricName, getForAction.GetMetricName(), "the metric requested should have matched the one specified")

			if getForAction.GetName() == "*" {
				// multiple objects
				metrics := cmapi.MetricValueList{}
				assert.Equal(t, "pods", getForAction.GetResource().Resource, "type of object that we requested multiple metrics for should have been pods")

				for i, metricPoint := range tc.reportedMetricPoints {
					timestamp := offsetTimestampBy(metricPoint.timestamp)
					metric := cmapi.MetricValue{
						DescribedObject: v1.ObjectReference{
							Kind:       "Pod",
							APIVersion: "v1",
							Name:       fmt.Sprintf("%s-%d", podNamePrefix, i),
						},
						Value:     *resource.NewMilliQuantity(int64(metricPoint.level), resource.DecimalSI),
						Timestamp: metav1.Time{Time: timestamp},
						Metric: cmapi.MetricIdentifier{
							Name: tc.metricName,
						},
					}

					metrics.Items = append(metrics.Items, metric)
				}

				return true, &metrics, nil
			} else {
				name := getForAction.GetName()
				mapper := testrestmapper.TestOnlyStaticRESTMapper(legacyscheme.Scheme)
				assert.NotNil(t, tc.singleObject, "should have only requested a single-object metric when we asked for metrics for a single object")
				gk := schema.FromAPIVersionAndKind(tc.singleObject.APIVersion, tc.singleObject.Kind).GroupKind()
				mapping, err := mapper.RESTMapping(gk)
				if err != nil {
					return true, nil, fmt.Errorf("unable to get mapping for %s: %v", gk.String(), err)
				}
				groupResource := mapping.Resource.GroupResource()

				assert.Equal(t, groupResource.String(), getForAction.GetResource().Resource, "should have requested metrics for the resource matching the GroupKind passed in")
				assert.Equal(t, tc.singleObject.Name, name, "should have requested metrics for the object matching the name passed in")
				metricPoint := tc.reportedMetricPoints[0]
				timestamp := offsetTimestampBy(metricPoint.timestamp)

				metrics := &cmapi.MetricValueList{
					Items: []cmapi.MetricValue{
						{
							DescribedObject: v1.ObjectReference{
								Kind:       tc.singleObject.Kind,
								APIVersion: tc.singleObject.APIVersion,
								Name:       tc.singleObject.Name,
							},
							Timestamp: metav1.Time{Time: timestamp},
							Metric: cmapi.MetricIdentifier{
								Name: tc.metricName,
							},
							Value: *resource.NewMilliQuantity(int64(metricPoint.level), resource.DecimalSI),
						},
					},
				}

				return true, metrics, nil
			}
		})
	}

	return fakeMetricsClient, fakeCMClient, fakeEMClient
}

func (tc *restClientTestCase) verifyResults(t *testing.T, metrics PodMetricsInfo, timestamp time.Time, err error) {
	if tc.desiredError != nil {
		assert.Error(t, err, "there should be an error retrieving the metrics")
		assert.Contains(t, fmt.Sprintf("%v", err), fmt.Sprintf("%v", tc.desiredError), "the error message should be as expected")
		return
	}
	assert.NoError(t, err, "there should be no error retrieving the metrics")
	assert.NotNil(t, metrics, "there should be metrics returned")

	if len(metrics) != len(tc.desiredMetricValues) {
		t.Errorf("Not equal:\nexpected: %v\nactual: %v", tc.desiredMetricValues, metrics)
	} else {
		for k, m := range metrics {
			if !m.Timestamp.Equal(tc.desiredMetricValues[k].Timestamp) ||
				m.Window != tc.desiredMetricValues[k].Window ||
				m.Value != tc.desiredMetricValues[k].Value {
				t.Errorf("Not equal:\nexpected: %v\nactual: %v", tc.desiredMetricValues, metrics)
				break
			}
		}
	}

	targetTimestamp := offsetTimestampBy(tc.targetTimestamp)
	assert.True(t, targetTimestamp.Equal(timestamp), fmt.Sprintf("the timestamp should be as expected (%s) but was %s", targetTimestamp, timestamp))
}

func (tc *restClientTestCase) runTest(t *testing.T) {
	var err error
	testMetricsClient, testCMClient, testEMClient := tc.prepareTestClient(t)
	metricsClient := NewRESTMetricsClient(testMetricsClient.MetricsV1beta1(), testCMClient, testEMClient)
	isResource := len(tc.resourceName) > 0
	isExternal := tc.metricSelector != nil
	if isResource {
		info, timestamp, err := metricsClient.GetResourceMetric(v1.ResourceName(tc.resourceName), tc.namespace, tc.selector)
		tc.verifyResults(t, info, timestamp, err)
	} else if isExternal {
		tc.metricLabelSelector, err = metav1.LabelSelectorAsSelector(tc.metricSelector)
		if err != nil {
			t.Errorf("invalid metric selector: %+v", tc.metricSelector)
		}
		val, timestamp, err := metricsClient.GetExternalMetric(tc.metricName, tc.namespace, tc.metricLabelSelector)
		info := make(PodMetricsInfo, len(val))
		for i, metricVal := range val {
			info[fmt.Sprintf("%v-val-%v", tc.metricName, i)] = PodMetric{Value: metricVal}
		}
		tc.verifyResults(t, info, timestamp, err)
	} else if tc.singleObject == nil {
		info, timestamp, err := metricsClient.GetRawMetric(tc.metricName, tc.namespace, tc.selector, tc.metricLabelSelector)
		tc.verifyResults(t, info, timestamp, err)
	} else {
		val, timestamp, err := metricsClient.GetObjectMetric(tc.metricName, tc.namespace, tc.singleObject, tc.metricLabelSelector)
		info := PodMetricsInfo{tc.singleObject.Name: {Value: val}}
		tc.verifyResults(t, info, timestamp, err)
	}
}

func TestRESTClientCPU(t *testing.T) {
	targetTimestamp := 1
	window := 30 * time.Second
	tc := restClientTestCase{
		desiredMetricValues: PodMetricsInfo{
			"test-pod-0": {Value: 5000, Timestamp: offsetTimestampBy(targetTimestamp), Window: window},
			"test-pod-1": {Value: 5000, Timestamp: offsetTimestampBy(targetTimestamp), Window: window},
			"test-pod-2": {Value: 5000, Timestamp: offsetTimestampBy(targetTimestamp), Window: window},
		},
		resourceName:       v1.ResourceCPU,
		targetTimestamp:    targetTimestamp,
		window:             window,
		reportedPodMetrics: [][]int64{{5000}, {5000}, {5000}},
	}
	tc.runTest(t)
}

func TestRESTClientExternal(t *testing.T) {
	tc := restClientTestCase{
		desiredMetricValues: PodMetricsInfo{
			"external-val-0": {Value: 10000}, "external-val-1": {Value: 20000}, "external-val-2": {Value: 10000},
		},
		metricSelector:       &metav1.LabelSelector{MatchLabels: map[string]string{"label": "value"}},
		metricName:           "external",
		targetTimestamp:      1,
		reportedMetricPoints: []metricPoint{{10000, 1}, {20000, 1}, {10000, 1}},
	}
	tc.runTest(t)
}

func TestRESTClientQPS(t *testing.T) {
	targetTimestamp := 1
	tc := restClientTestCase{
		desiredMetricValues: PodMetricsInfo{
			"test-pod-0": {Value: 10000, Timestamp: offsetTimestampBy(targetTimestamp), Window: metricServerDefaultMetricWindow},
			"test-pod-1": {Value: 20000, Timestamp: offsetTimestampBy(targetTimestamp), Window: metricServerDefaultMetricWindow},
			"test-pod-2": {Value: 10000, Timestamp: offsetTimestampBy(targetTimestamp), Window: metricServerDefaultMetricWindow},
		},
		metricName:           "qps",
		targetTimestamp:      targetTimestamp,
		reportedMetricPoints: []metricPoint{{10000, 1}, {20000, 1}, {10000, 1}},
	}
	tc.runTest(t)
}

func TestRESTClientSingleObject(t *testing.T) {
	tc := restClientTestCase{
		desiredMetricValues:  PodMetricsInfo{"some-dep": {Value: 10}},
		metricName:           "queue-length",
		targetTimestamp:      1,
		reportedMetricPoints: []metricPoint{{10, 1}},
		singleObject: &autoscalingapi.CrossVersionObjectReference{
			APIVersion: "extensions/v1beta1",
			Kind:       "Deployment",
			Name:       "some-dep",
		},
	}
	tc.runTest(t)
}

func TestRESTClientQpsSumEqualZero(t *testing.T) {
	targetTimestamp := 0
	tc := restClientTestCase{
		desiredMetricValues: PodMetricsInfo{
			"test-pod-0": {Value: 0, Timestamp: offsetTimestampBy(targetTimestamp), Window: metricServerDefaultMetricWindow},
			"test-pod-1": {Value: 0, Timestamp: offsetTimestampBy(targetTimestamp), Window: metricServerDefaultMetricWindow},
			"test-pod-2": {Value: 0, Timestamp: offsetTimestampBy(targetTimestamp), Window: metricServerDefaultMetricWindow},
		},
		metricName:           "qps",
		targetTimestamp:      targetTimestamp,
		reportedMetricPoints: []metricPoint{{0, 0}, {0, 0}, {0, 0}},
	}
	tc.runTest(t)
}

func TestRESTClientExternalSumEqualZero(t *testing.T) {
	tc := restClientTestCase{
		desiredMetricValues: PodMetricsInfo{
			"external-val-0": {Value: 0}, "external-val-1": {Value: 0}, "external-val-2": {Value: 0},
		},
		metricSelector:       &metav1.LabelSelector{MatchLabels: map[string]string{"label": "value"}},
		metricName:           "external",
		targetTimestamp:      0,
		reportedMetricPoints: []metricPoint{{0, 0}, {0, 0}, {0, 0}},
	}
	tc.runTest(t)
}

func TestRESTClientQpsEmptyMetrics(t *testing.T) {
	tc := restClientTestCase{
		metricName:           "qps",
		desiredError:         fmt.Errorf("no metrics returned from custom metrics API"),
		reportedMetricPoints: []metricPoint{},
	}

	tc.runTest(t)
}

func TestRESTClientExternalEmptyMetrics(t *testing.T) {
	tc := restClientTestCase{
		metricName:           "external",
		metricSelector:       &metav1.LabelSelector{MatchLabels: map[string]string{"label": "value"}},
		desiredError:         fmt.Errorf("no metrics returned from external metrics API"),
		reportedMetricPoints: []metricPoint{},
	}

	tc.runTest(t)
}

func TestRESTClientCPUEmptyMetrics(t *testing.T) {
	tc := restClientTestCase{
		resourceName:         v1.ResourceCPU,
		desiredError:         fmt.Errorf("no metrics returned from resource metrics API"),
		reportedMetricPoints: []metricPoint{},
		reportedPodMetrics:   [][]int64{},
	}
	tc.runTest(t)
}

func TestRESTClientCPUEmptyMetricsForOnePod(t *testing.T) {
	targetTimestamp := 1
	window := 30 * time.Second
	tc := restClientTestCase{
		resourceName: v1.ResourceCPU,
		desiredMetricValues: PodMetricsInfo{
			"test-pod-0": {Value: 100, Timestamp: offsetTimestampBy(targetTimestamp), Window: window},
			"test-pod-1": {Value: 700, Timestamp: offsetTimestampBy(targetTimestamp), Window: window},
		},
		targetTimestamp:    targetTimestamp,
		window:             window,
		reportedPodMetrics: [][]int64{{100}, {300, 400}, {}},
	}
	tc.runTest(t)
}
