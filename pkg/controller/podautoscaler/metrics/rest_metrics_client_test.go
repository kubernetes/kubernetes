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

	autoscalingapi "k8s.io/api/autoscaling/v2beta1"
	"k8s.io/api/core/v1"
	kv1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	core "k8s.io/client-go/testing"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	_ "k8s.io/kubernetes/pkg/apis/extensions/install"
	cmapi "k8s.io/metrics/pkg/apis/custom_metrics/v1beta1"
	metricsapi "k8s.io/metrics/pkg/apis/metrics/v1beta1"
	metricsfake "k8s.io/metrics/pkg/client/clientset_generated/clientset/fake"
	cmfake "k8s.io/metrics/pkg/client/custom_metrics/fake"

	"github.com/stretchr/testify/assert"
)

type restClientTestCase struct {
	desiredMetricValues PodMetricsInfo
	desiredError        error

	// "timestamps" here are actually the offset in minutes from a base timestamp
	targetTimestamp      int
	reportedMetricPoints []metricPoint
	reportedPodMetrics   [][]int64
	singleObject         *autoscalingapi.CrossVersionObjectReference

	namespace    string
	selector     labels.Selector
	resourceName v1.ResourceName
	metricName   string
}

func (tc *restClientTestCase) prepareTestClient(t *testing.T) (*metricsfake.Clientset, *cmfake.FakeCustomMetricsClient) {
	namespace := "test-namespace"
	tc.namespace = namespace
	podNamePrefix := "test-pod"
	podLabels := map[string]string{"name": podNamePrefix}
	tc.selector = labels.SelectorFromSet(podLabels)

	// it's a resource test if we have a resource name
	isResource := len(tc.resourceName) > 0

	fakeMetricsClient := &metricsfake.Clientset{}
	fakeCMClient := &cmfake.FakeCustomMetricsClient{}

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
					Timestamp:  metav1.Time{Time: fixedTimestamp.Add(time.Duration(tc.targetTimestamp) * time.Minute)},
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
	} else {
		fakeCMClient.AddReactor("get", "*", func(action core.Action) (handled bool, ret runtime.Object, err error) {
			getForAction := action.(cmfake.GetForAction)
			assert.Equal(t, tc.metricName, getForAction.GetMetricName(), "the metric requested should have matched the one specified")

			if getForAction.GetName() == "*" {
				// multiple objects
				metrics := cmapi.MetricValueList{}
				assert.Equal(t, "pods", getForAction.GetResource().Resource, "type of object that we requested multiple metrics for should have been pods")

				for i, metricPoint := range tc.reportedMetricPoints {
					timestamp := fixedTimestamp.Add(time.Duration(metricPoint.timestamp) * time.Minute)
					metric := cmapi.MetricValue{
						DescribedObject: v1.ObjectReference{
							Kind:       "Pod",
							APIVersion: "v1",
							Name:       fmt.Sprintf("%s-%d", podNamePrefix, i),
						},
						Value:      *resource.NewMilliQuantity(int64(metricPoint.level), resource.DecimalSI),
						Timestamp:  metav1.Time{Time: timestamp},
						MetricName: tc.metricName,
					}

					metrics.Items = append(metrics.Items, metric)
				}

				return true, &metrics, nil
			} else {
				name := getForAction.GetName()
				mapper := legacyscheme.Registry.RESTMapper()
				assert.NotNil(t, tc.singleObject, "should have only requested a single-object metric when we asked for metrics for a single object")
				gk := schema.FromAPIVersionAndKind(tc.singleObject.APIVersion, tc.singleObject.Kind).GroupKind()
				mapping, err := mapper.RESTMapping(gk)
				if err != nil {
					return true, nil, fmt.Errorf("unable to get mapping for %s: %v", gk.String(), err)
				}
				groupResource := schema.GroupResource{Group: mapping.GroupVersionKind.Group, Resource: mapping.Resource}

				assert.Equal(t, groupResource.String(), getForAction.GetResource().Resource, "should have requested metrics for the resource matching the GroupKind passed in")
				assert.Equal(t, tc.singleObject.Name, name, "should have requested metrics for the object matching the name passed in")
				metricPoint := tc.reportedMetricPoints[0]
				timestamp := fixedTimestamp.Add(time.Duration(metricPoint.timestamp) * time.Minute)

				metrics := &cmapi.MetricValueList{
					Items: []cmapi.MetricValue{
						{
							DescribedObject: v1.ObjectReference{
								Kind:       tc.singleObject.Kind,
								APIVersion: tc.singleObject.APIVersion,
								Name:       tc.singleObject.Name,
							},
							Timestamp:  metav1.Time{Time: timestamp},
							MetricName: tc.metricName,
							Value:      *resource.NewMilliQuantity(int64(metricPoint.level), resource.DecimalSI),
						},
					},
				}

				return true, metrics, nil
			}
		})
	}

	return fakeMetricsClient, fakeCMClient
}

func (tc *restClientTestCase) verifyResults(t *testing.T, metrics PodMetricsInfo, timestamp time.Time, err error) {
	if tc.desiredError != nil {
		assert.Error(t, err, "there should be an error retrieving the metrics")
		assert.Contains(t, fmt.Sprintf("%v", err), fmt.Sprintf("%v", tc.desiredError), "the error message should be eas expected")
		return
	}
	assert.NoError(t, err, "there should be no error retrieving the metrics")
	assert.NotNil(t, metrics, "there should be metrics returned")

	assert.Equal(t, tc.desiredMetricValues, metrics, "the metrics values should be as expected")

	targetTimestamp := fixedTimestamp.Add(time.Duration(tc.targetTimestamp) * time.Minute)
	assert.True(t, targetTimestamp.Equal(timestamp), fmt.Sprintf("the timestamp should be as expected (%s) but was %s", targetTimestamp, timestamp))
}

func (tc *restClientTestCase) runTest(t *testing.T) {
	testMetricsClient, testCMClient := tc.prepareTestClient(t)
	metricsClient := NewRESTMetricsClient(testMetricsClient.MetricsV1beta1(), testCMClient)
	isResource := len(tc.resourceName) > 0
	if isResource {
		info, timestamp, err := metricsClient.GetResourceMetric(kv1.ResourceName(tc.resourceName), tc.namespace, tc.selector)
		tc.verifyResults(t, info, timestamp, err)
	} else if tc.singleObject == nil {
		info, timestamp, err := metricsClient.GetRawMetric(tc.metricName, tc.namespace, tc.selector)
		tc.verifyResults(t, info, timestamp, err)
	} else {
		val, timestamp, err := metricsClient.GetObjectMetric(tc.metricName, tc.namespace, tc.singleObject)
		info := PodMetricsInfo{tc.singleObject.Name: val}
		tc.verifyResults(t, info, timestamp, err)
	}
}

func TestRESTClientCPU(t *testing.T) {
	tc := restClientTestCase{
		desiredMetricValues: PodMetricsInfo{
			"test-pod-0": 5000, "test-pod-1": 5000, "test-pod-2": 5000,
		},
		resourceName:       v1.ResourceCPU,
		targetTimestamp:    1,
		reportedPodMetrics: [][]int64{{5000}, {5000}, {5000}},
	}
	tc.runTest(t)
}

func TestRESTClientQPS(t *testing.T) {
	tc := restClientTestCase{
		desiredMetricValues: PodMetricsInfo{
			"test-pod-0": 10000, "test-pod-1": 20000, "test-pod-2": 10000,
		},
		metricName:           "qps",
		targetTimestamp:      1,
		reportedMetricPoints: []metricPoint{{10000, 1}, {20000, 1}, {10000, 1}},
	}
	tc.runTest(t)
}

func TestRESTClientSingleObject(t *testing.T) {
	tc := restClientTestCase{
		desiredMetricValues:  PodMetricsInfo{"some-dep": 10},
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
	tc := restClientTestCase{
		desiredMetricValues: PodMetricsInfo{
			"test-pod-0": 0, "test-pod-1": 0, "test-pod-2": 0,
		},
		metricName:           "qps",
		targetTimestamp:      0,
		reportedMetricPoints: []metricPoint{{0, 0}, {0, 0}, {0, 0}},
	}
	tc.runTest(t)
}

func TestRESTClientCPUEmptyMetrics(t *testing.T) {
	tc := restClientTestCase{
		resourceName:         v1.ResourceCPU,
		desiredError:         fmt.Errorf("no metrics returned from heapster"),
		reportedMetricPoints: []metricPoint{},
		reportedPodMetrics:   [][]int64{},
	}
	tc.runTest(t)
}

func TestRESTClientCPUEmptyMetricsForOnePod(t *testing.T) {
	tc := restClientTestCase{
		resourceName: v1.ResourceCPU,
		desiredMetricValues: PodMetricsInfo{
			"test-pod-0": 100, "test-pod-1": 700,
		},
		reportedPodMetrics: [][]int64{{100}, {300, 400}, {}},
	}
	tc.runTest(t)
}
