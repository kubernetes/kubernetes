/*
Copyright 2016 The Kubernetes Authors.

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

package podautoscaler

import (
	"fmt"
	"math"
	"testing"
	"time"

	autoscalingv2 "k8s.io/api/autoscaling/v2beta1"
	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/client-go/kubernetes/fake"
	core "k8s.io/client-go/testing"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/controller/podautoscaler/metrics"
	metricsfake "k8s.io/metrics/pkg/client/clientset_generated/clientset/fake"
	cmfake "k8s.io/metrics/pkg/client/custom_metrics/fake"

	cmapi "k8s.io/metrics/pkg/apis/custom_metrics/v1beta1"
	metricsapi "k8s.io/metrics/pkg/apis/metrics/v1beta1"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

type resourceInfo struct {
	name     v1.ResourceName
	requests []resource.Quantity
	levels   []int64
	// only applies to pod names returned from "heapster"
	podNames []string

	targetUtilization   int32
	expectedUtilization int32
	expectedValue       int64
}

type metricInfo struct {
	name         string
	levels       []int64
	singleObject *autoscalingv2.CrossVersionObjectReference

	targetUtilization   int64
	expectedUtilization int64
}

type replicaCalcTestCase struct {
	currentReplicas  int32
	expectedReplicas int32
	expectedError    error

	timestamp time.Time

	resource *resourceInfo
	metric   *metricInfo

	podReadiness []v1.ConditionStatus
}

const (
	testNamespace       = "test-namespace"
	podNamePrefix       = "test-pod"
	numContainersPerPod = 2
)

func (tc *replicaCalcTestCase) prepareTestClient(t *testing.T) (*fake.Clientset, *metricsfake.Clientset, *cmfake.FakeCustomMetricsClient) {

	fakeClient := &fake.Clientset{}
	fakeClient.AddReactor("list", "pods", func(action core.Action) (handled bool, ret runtime.Object, err error) {
		obj := &v1.PodList{}
		for i := 0; i < int(tc.currentReplicas); i++ {
			podReadiness := v1.ConditionTrue
			if tc.podReadiness != nil {
				podReadiness = tc.podReadiness[i]
			}
			podName := fmt.Sprintf("%s-%d", podNamePrefix, i)
			pod := v1.Pod{
				Status: v1.PodStatus{
					Phase: v1.PodRunning,
					Conditions: []v1.PodCondition{
						{
							Type:   v1.PodReady,
							Status: podReadiness,
						},
					},
				},
				ObjectMeta: metav1.ObjectMeta{
					Name:      podName,
					Namespace: testNamespace,
					Labels: map[string]string{
						"name": podNamePrefix,
					},
				},
				Spec: v1.PodSpec{
					Containers: []v1.Container{{}, {}},
				},
			}

			if tc.resource != nil && i < len(tc.resource.requests) {
				pod.Spec.Containers[0].Resources = v1.ResourceRequirements{
					Requests: v1.ResourceList{
						tc.resource.name: tc.resource.requests[i],
					},
				}
				pod.Spec.Containers[1].Resources = v1.ResourceRequirements{
					Requests: v1.ResourceList{
						tc.resource.name: tc.resource.requests[i],
					},
				}
			}
			obj.Items = append(obj.Items, pod)
		}
		return true, obj, nil
	})

	fakeMetricsClient := &metricsfake.Clientset{}
	// NB: we have to sound like Gollum due to gengo's inability to handle already-plural resource names
	fakeMetricsClient.AddReactor("list", "pods", func(action core.Action) (handled bool, ret runtime.Object, err error) {
		if tc.resource != nil {
			metrics := &metricsapi.PodMetricsList{}
			for i, resValue := range tc.resource.levels {
				podName := fmt.Sprintf("%s-%d", podNamePrefix, i)
				if len(tc.resource.podNames) > i {
					podName = tc.resource.podNames[i]
				}
				// NB: the list reactor actually does label selector filtering for us,
				// so we have to make sure our results match the label selector
				podMetric := metricsapi.PodMetrics{
					ObjectMeta: metav1.ObjectMeta{
						Name:      podName,
						Namespace: testNamespace,
						Labels:    map[string]string{"name": podNamePrefix},
					},
					Timestamp:  metav1.Time{Time: tc.timestamp},
					Containers: make([]metricsapi.ContainerMetrics, numContainersPerPod),
				}

				for i := 0; i < numContainersPerPod; i++ {
					podMetric.Containers[i] = metricsapi.ContainerMetrics{
						Name: fmt.Sprintf("container%v", i),
						Usage: v1.ResourceList{
							v1.ResourceName(tc.resource.name): *resource.NewMilliQuantity(
								int64(resValue),
								resource.DecimalSI),
						},
					}
				}
				metrics.Items = append(metrics.Items, podMetric)
			}
			return true, metrics, nil
		}

		return true, nil, fmt.Errorf("no pod resource metrics specified in test client")
	})

	fakeCMClient := &cmfake.FakeCustomMetricsClient{}
	fakeCMClient.AddReactor("get", "*", func(action core.Action) (handled bool, ret runtime.Object, err error) {
		getForAction, wasGetFor := action.(cmfake.GetForAction)
		if !wasGetFor {
			return true, nil, fmt.Errorf("expected a get-for action, got %v instead", action)
		}

		if tc.metric == nil {
			return true, nil, fmt.Errorf("no custom metrics specified in test client")
		}

		assert.Equal(t, tc.metric.name, getForAction.GetMetricName(), "the metric requested should have matched the one specified")

		if getForAction.GetName() == "*" {
			metrics := cmapi.MetricValueList{}

			// multiple objects
			assert.Equal(t, "pods", getForAction.GetResource().Resource, "the type of object that we requested multiple metrics for should have been pods")

			for i, level := range tc.metric.levels {
				podMetric := cmapi.MetricValue{
					DescribedObject: v1.ObjectReference{
						Kind:      "Pod",
						Name:      fmt.Sprintf("%s-%d", podNamePrefix, i),
						Namespace: testNamespace,
					},
					Timestamp:  metav1.Time{Time: tc.timestamp},
					MetricName: tc.metric.name,
					Value:      *resource.NewMilliQuantity(level, resource.DecimalSI),
				}
				metrics.Items = append(metrics.Items, podMetric)
			}

			return true, &metrics, nil
		} else {
			name := getForAction.GetName()
			mapper := api.Registry.RESTMapper()
			metrics := &cmapi.MetricValueList{}
			assert.NotNil(t, tc.metric.singleObject, "should have only requested a single-object metric when calling GetObjectMetricReplicas")
			gk := schema.FromAPIVersionAndKind(tc.metric.singleObject.APIVersion, tc.metric.singleObject.Kind).GroupKind()
			mapping, err := mapper.RESTMapping(gk)
			if err != nil {
				return true, nil, fmt.Errorf("unable to get mapping for %s: %v", gk.String(), err)
			}
			groupResource := schema.GroupResource{Group: mapping.GroupVersionKind.Group, Resource: mapping.Resource}

			assert.Equal(t, groupResource.String(), getForAction.GetResource().Resource, "should have requested metrics for the resource matching the GroupKind passed in")
			assert.Equal(t, tc.metric.singleObject.Name, name, "should have requested metrics for the object matching the name passed in")

			metrics.Items = []cmapi.MetricValue{
				{
					DescribedObject: v1.ObjectReference{
						Kind:       tc.metric.singleObject.Kind,
						APIVersion: tc.metric.singleObject.APIVersion,
						Name:       name,
					},
					Timestamp:  metav1.Time{Time: tc.timestamp},
					MetricName: tc.metric.name,
					Value:      *resource.NewMilliQuantity(int64(tc.metric.levels[0]), resource.DecimalSI),
				},
			}

			return true, metrics, nil
		}
	})

	return fakeClient, fakeMetricsClient, fakeCMClient
}

func (tc *replicaCalcTestCase) runTest(t *testing.T) {
	testClient, testMetricsClient, testCMClient := tc.prepareTestClient(t)
	metricsClient := metrics.NewRESTMetricsClient(testMetricsClient.MetricsV1beta1(), testCMClient)

	replicaCalc := &ReplicaCalculator{
		metricsClient: metricsClient,
		podsGetter:    testClient.Core(),
	}

	selector, err := metav1.LabelSelectorAsSelector(&metav1.LabelSelector{
		MatchLabels: map[string]string{"name": podNamePrefix},
	})
	if err != nil {
		require.Nil(t, err, "something went horribly wrong...")
	}

	if tc.resource != nil {
		outReplicas, outUtilization, outRawValue, outTimestamp, err := replicaCalc.GetResourceReplicas(tc.currentReplicas, tc.resource.targetUtilization, tc.resource.name, testNamespace, selector)

		if tc.expectedError != nil {
			require.Error(t, err, "there should be an error calculating the replica count")
			assert.Contains(t, err.Error(), tc.expectedError.Error(), "the error message should have contained the expected error message")
			return
		}
		require.NoError(t, err, "there should not have been an error calculating the replica count")
		assert.Equal(t, tc.expectedReplicas, outReplicas, "replicas should be as expected")
		assert.Equal(t, tc.resource.expectedUtilization, outUtilization, "utilization should be as expected")
		assert.Equal(t, tc.resource.expectedValue, outRawValue, "raw value should be as expected")
		assert.True(t, tc.timestamp.Equal(outTimestamp), "timestamp should be as expected")

	} else {
		var outReplicas int32
		var outUtilization int64
		var outTimestamp time.Time
		var err error
		if tc.metric.singleObject != nil {
			outReplicas, outUtilization, outTimestamp, err = replicaCalc.GetObjectMetricReplicas(tc.currentReplicas, tc.metric.targetUtilization, tc.metric.name, testNamespace, tc.metric.singleObject)
		} else {
			outReplicas, outUtilization, outTimestamp, err = replicaCalc.GetMetricReplicas(tc.currentReplicas, tc.metric.targetUtilization, tc.metric.name, testNamespace, selector)
		}

		if tc.expectedError != nil {
			require.Error(t, err, "there should be an error calculating the replica count")
			assert.Contains(t, err.Error(), tc.expectedError.Error(), "the error message should have contained the expected error message")
			return
		}
		require.NoError(t, err, "there should not have been an error calculating the replica count")
		assert.Equal(t, tc.expectedReplicas, outReplicas, "replicas should be as expected")
		assert.Equal(t, tc.metric.expectedUtilization, outUtilization, "utilization should be as expected")
		assert.True(t, tc.timestamp.Equal(outTimestamp), "timestamp should be as expected")
	}
}

func TestReplicaCalcDisjointResourcesMetrics(t *testing.T) {
	tc := replicaCalcTestCase{
		currentReplicas: 1,
		expectedError:   fmt.Errorf("no metrics returned matched known pods"),
		resource: &resourceInfo{
			name:     v1.ResourceCPU,
			requests: []resource.Quantity{resource.MustParse("1.0")},
			levels:   []int64{100},
			podNames: []string{"an-older-pod-name"},

			targetUtilization: 100,
		},
	}
	tc.runTest(t)
}

func TestReplicaCalcScaleUp(t *testing.T) {
	tc := replicaCalcTestCase{
		currentReplicas:  3,
		expectedReplicas: 5,
		resource: &resourceInfo{
			name:     v1.ResourceCPU,
			requests: []resource.Quantity{resource.MustParse("1.0"), resource.MustParse("1.0"), resource.MustParse("1.0")},
			levels:   []int64{300, 500, 700},

			targetUtilization:   30,
			expectedUtilization: 50,
			expectedValue:       numContainersPerPod * 500,
		},
	}
	tc.runTest(t)
}

func TestReplicaCalcScaleUpUnreadyLessScale(t *testing.T) {
	tc := replicaCalcTestCase{
		currentReplicas:  3,
		expectedReplicas: 4,
		podReadiness:     []v1.ConditionStatus{v1.ConditionFalse, v1.ConditionTrue, v1.ConditionTrue},
		resource: &resourceInfo{
			name:     v1.ResourceCPU,
			requests: []resource.Quantity{resource.MustParse("1.0"), resource.MustParse("1.0"), resource.MustParse("1.0")},
			levels:   []int64{300, 500, 700},

			targetUtilization:   30,
			expectedUtilization: 60,
			expectedValue:       numContainersPerPod * 600,
		},
	}
	tc.runTest(t)
}

func TestReplicaCalcScaleUpUnreadyNoScale(t *testing.T) {
	tc := replicaCalcTestCase{
		currentReplicas:  3,
		expectedReplicas: 3,
		podReadiness:     []v1.ConditionStatus{v1.ConditionTrue, v1.ConditionFalse, v1.ConditionFalse},
		resource: &resourceInfo{
			name:     v1.ResourceCPU,
			requests: []resource.Quantity{resource.MustParse("1.0"), resource.MustParse("1.0"), resource.MustParse("1.0")},
			levels:   []int64{400, 500, 700},

			targetUtilization:   30,
			expectedUtilization: 40,
			expectedValue:       numContainersPerPod * 400,
		},
	}
	tc.runTest(t)
}

func TestReplicaCalcScaleUpCM(t *testing.T) {
	tc := replicaCalcTestCase{
		currentReplicas:  3,
		expectedReplicas: 4,
		metric: &metricInfo{
			name:                "qps",
			levels:              []int64{20000, 10000, 30000},
			targetUtilization:   15000,
			expectedUtilization: 20000,
		},
	}
	tc.runTest(t)
}

func TestReplicaCalcScaleUpCMUnreadyLessScale(t *testing.T) {
	tc := replicaCalcTestCase{
		currentReplicas:  3,
		expectedReplicas: 4,
		podReadiness:     []v1.ConditionStatus{v1.ConditionTrue, v1.ConditionTrue, v1.ConditionFalse},
		metric: &metricInfo{
			name:                "qps",
			levels:              []int64{50000, 10000, 30000},
			targetUtilization:   15000,
			expectedUtilization: 30000,
		},
	}
	tc.runTest(t)
}

func TestReplicaCalcScaleUpCMUnreadyNoScaleWouldScaleDown(t *testing.T) {
	tc := replicaCalcTestCase{
		currentReplicas:  3,
		expectedReplicas: 3,
		podReadiness:     []v1.ConditionStatus{v1.ConditionFalse, v1.ConditionTrue, v1.ConditionFalse},
		metric: &metricInfo{
			name:                "qps",
			levels:              []int64{50000, 15000, 30000},
			targetUtilization:   15000,
			expectedUtilization: 15000,
		},
	}
	tc.runTest(t)
}

func TestReplicaCalcScaleUpCMObject(t *testing.T) {
	tc := replicaCalcTestCase{
		currentReplicas:  3,
		expectedReplicas: 4,
		metric: &metricInfo{
			name:                "qps",
			levels:              []int64{20000},
			targetUtilization:   15000,
			expectedUtilization: 20000,
			singleObject: &autoscalingv2.CrossVersionObjectReference{
				Kind:       "Deployment",
				APIVersion: "extensions/v1beta1",
				Name:       "some-deployment",
			},
		},
	}
	tc.runTest(t)
}

func TestReplicaCalcScaleDown(t *testing.T) {
	tc := replicaCalcTestCase{
		currentReplicas:  5,
		expectedReplicas: 3,
		resource: &resourceInfo{
			name:     v1.ResourceCPU,
			requests: []resource.Quantity{resource.MustParse("1.0"), resource.MustParse("1.0"), resource.MustParse("1.0"), resource.MustParse("1.0"), resource.MustParse("1.0")},
			levels:   []int64{100, 300, 500, 250, 250},

			targetUtilization:   50,
			expectedUtilization: 28,
			expectedValue:       numContainersPerPod * 280,
		},
	}
	tc.runTest(t)
}

func TestReplicaCalcScaleDownCM(t *testing.T) {
	tc := replicaCalcTestCase{
		currentReplicas:  5,
		expectedReplicas: 3,
		metric: &metricInfo{
			name:                "qps",
			levels:              []int64{12000, 12000, 12000, 12000, 12000},
			targetUtilization:   20000,
			expectedUtilization: 12000,
		},
	}
	tc.runTest(t)
}

func TestReplicaCalcScaleDownCMObject(t *testing.T) {
	tc := replicaCalcTestCase{
		currentReplicas:  5,
		expectedReplicas: 3,
		metric: &metricInfo{
			name:                "qps",
			levels:              []int64{12000},
			targetUtilization:   20000,
			expectedUtilization: 12000,
			singleObject: &autoscalingv2.CrossVersionObjectReference{
				Kind:       "Deployment",
				APIVersion: "extensions/v1beta1",
				Name:       "some-deployment",
			},
		},
	}
	tc.runTest(t)
}

func TestReplicaCalcScaleDownIgnoresUnreadyPods(t *testing.T) {
	tc := replicaCalcTestCase{
		currentReplicas:  5,
		expectedReplicas: 2,
		podReadiness:     []v1.ConditionStatus{v1.ConditionTrue, v1.ConditionTrue, v1.ConditionTrue, v1.ConditionFalse, v1.ConditionFalse},
		resource: &resourceInfo{
			name:     v1.ResourceCPU,
			requests: []resource.Quantity{resource.MustParse("1.0"), resource.MustParse("1.0"), resource.MustParse("1.0"), resource.MustParse("1.0"), resource.MustParse("1.0")},
			levels:   []int64{100, 300, 500, 250, 250},

			targetUtilization:   50,
			expectedUtilization: 30,
			expectedValue:       numContainersPerPod * 300,
		},
	}
	tc.runTest(t)
}

func TestReplicaCalcTolerance(t *testing.T) {
	tc := replicaCalcTestCase{
		currentReplicas:  3,
		expectedReplicas: 3,
		resource: &resourceInfo{
			name:     v1.ResourceCPU,
			requests: []resource.Quantity{resource.MustParse("0.9"), resource.MustParse("1.0"), resource.MustParse("1.1")},
			levels:   []int64{1010, 1030, 1020},

			targetUtilization:   100,
			expectedUtilization: 102,
			expectedValue:       numContainersPerPod * 1020,
		},
	}
	tc.runTest(t)
}

func TestReplicaCalcToleranceCM(t *testing.T) {
	tc := replicaCalcTestCase{
		currentReplicas:  3,
		expectedReplicas: 3,
		metric: &metricInfo{
			name:                "qps",
			levels:              []int64{20000, 21000, 21000},
			targetUtilization:   20000,
			expectedUtilization: 20666,
		},
	}
	tc.runTest(t)
}

func TestReplicaCalcToleranceCMObject(t *testing.T) {
	tc := replicaCalcTestCase{
		currentReplicas:  3,
		expectedReplicas: 3,
		metric: &metricInfo{
			name:                "qps",
			levels:              []int64{20666},
			targetUtilization:   20000,
			expectedUtilization: 20666,
			singleObject: &autoscalingv2.CrossVersionObjectReference{
				Kind:       "Deployment",
				APIVersion: "extensions/v1beta1",
				Name:       "some-deployment",
			},
		},
	}
	tc.runTest(t)
}

func TestReplicaCalcSuperfluousMetrics(t *testing.T) {
	tc := replicaCalcTestCase{
		currentReplicas:  4,
		expectedReplicas: 24,
		resource: &resourceInfo{
			name:                v1.ResourceCPU,
			requests:            []resource.Quantity{resource.MustParse("1.0"), resource.MustParse("1.0"), resource.MustParse("1.0"), resource.MustParse("1.0")},
			levels:              []int64{4000, 9500, 3000, 7000, 3200, 2000},
			targetUtilization:   100,
			expectedUtilization: 587,
			expectedValue:       numContainersPerPod * 5875,
		},
	}
	tc.runTest(t)
}

func TestReplicaCalcMissingMetrics(t *testing.T) {
	tc := replicaCalcTestCase{
		currentReplicas:  4,
		expectedReplicas: 3,
		resource: &resourceInfo{
			name:     v1.ResourceCPU,
			requests: []resource.Quantity{resource.MustParse("1.0"), resource.MustParse("1.0"), resource.MustParse("1.0"), resource.MustParse("1.0")},
			levels:   []int64{400, 95},

			targetUtilization:   100,
			expectedUtilization: 24,
			expectedValue:       495, // numContainersPerPod * 247, for sufficiently large values of 247
		},
	}
	tc.runTest(t)
}

func TestReplicaCalcEmptyMetrics(t *testing.T) {
	tc := replicaCalcTestCase{
		currentReplicas: 4,
		expectedError:   fmt.Errorf("unable to get metrics for resource cpu: no metrics returned from heapster"),
		resource: &resourceInfo{
			name:     v1.ResourceCPU,
			requests: []resource.Quantity{resource.MustParse("1.0"), resource.MustParse("1.0"), resource.MustParse("1.0")},
			levels:   []int64{},

			targetUtilization: 100,
		},
	}
	tc.runTest(t)
}

func TestReplicaCalcEmptyCPURequest(t *testing.T) {
	tc := replicaCalcTestCase{
		currentReplicas: 1,
		expectedError:   fmt.Errorf("missing request for"),
		resource: &resourceInfo{
			name:     v1.ResourceCPU,
			requests: []resource.Quantity{},
			levels:   []int64{200},

			targetUtilization: 100,
		},
	}
	tc.runTest(t)
}

func TestReplicaCalcMissingMetricsNoChangeEq(t *testing.T) {
	tc := replicaCalcTestCase{
		currentReplicas:  2,
		expectedReplicas: 2,
		resource: &resourceInfo{
			name:     v1.ResourceCPU,
			requests: []resource.Quantity{resource.MustParse("1.0"), resource.MustParse("1.0")},
			levels:   []int64{1000},

			targetUtilization:   100,
			expectedUtilization: 100,
			expectedValue:       numContainersPerPod * 1000,
		},
	}
	tc.runTest(t)
}

func TestReplicaCalcMissingMetricsNoChangeGt(t *testing.T) {
	tc := replicaCalcTestCase{
		currentReplicas:  2,
		expectedReplicas: 2,
		resource: &resourceInfo{
			name:     v1.ResourceCPU,
			requests: []resource.Quantity{resource.MustParse("1.0"), resource.MustParse("1.0")},
			levels:   []int64{1900},

			targetUtilization:   100,
			expectedUtilization: 190,
			expectedValue:       numContainersPerPod * 1900,
		},
	}
	tc.runTest(t)
}

func TestReplicaCalcMissingMetricsNoChangeLt(t *testing.T) {
	tc := replicaCalcTestCase{
		currentReplicas:  2,
		expectedReplicas: 2,
		resource: &resourceInfo{
			name:     v1.ResourceCPU,
			requests: []resource.Quantity{resource.MustParse("1.0"), resource.MustParse("1.0")},
			levels:   []int64{600},

			targetUtilization:   100,
			expectedUtilization: 60,
			expectedValue:       numContainersPerPod * 600,
		},
	}
	tc.runTest(t)
}

func TestReplicaCalcMissingMetricsUnreadyNoChange(t *testing.T) {
	tc := replicaCalcTestCase{
		currentReplicas:  3,
		expectedReplicas: 3,
		podReadiness:     []v1.ConditionStatus{v1.ConditionFalse, v1.ConditionTrue, v1.ConditionTrue},
		resource: &resourceInfo{
			name:     v1.ResourceCPU,
			requests: []resource.Quantity{resource.MustParse("1.0"), resource.MustParse("1.0"), resource.MustParse("1.0")},
			levels:   []int64{100, 450},

			targetUtilization:   50,
			expectedUtilization: 45,
			expectedValue:       numContainersPerPod * 450,
		},
	}
	tc.runTest(t)
}

func TestReplicaCalcMissingMetricsUnreadyScaleUp(t *testing.T) {
	tc := replicaCalcTestCase{
		currentReplicas:  3,
		expectedReplicas: 4,
		podReadiness:     []v1.ConditionStatus{v1.ConditionFalse, v1.ConditionTrue, v1.ConditionTrue},
		resource: &resourceInfo{
			name:     v1.ResourceCPU,
			requests: []resource.Quantity{resource.MustParse("1.0"), resource.MustParse("1.0"), resource.MustParse("1.0")},
			levels:   []int64{100, 2000},

			targetUtilization:   50,
			expectedUtilization: 200,
			expectedValue:       numContainersPerPod * 2000,
		},
	}
	tc.runTest(t)
}

func TestReplicaCalcMissingMetricsUnreadyScaleDown(t *testing.T) {
	tc := replicaCalcTestCase{
		currentReplicas:  4,
		expectedReplicas: 3,
		podReadiness:     []v1.ConditionStatus{v1.ConditionFalse, v1.ConditionTrue, v1.ConditionTrue, v1.ConditionTrue},
		resource: &resourceInfo{
			name:     v1.ResourceCPU,
			requests: []resource.Quantity{resource.MustParse("1.0"), resource.MustParse("1.0"), resource.MustParse("1.0"), resource.MustParse("1.0")},
			levels:   []int64{100, 100, 100},

			targetUtilization:   50,
			expectedUtilization: 10,
			expectedValue:       numContainersPerPod * 100,
		},
	}
	tc.runTest(t)
}

// TestComputedToleranceAlgImplementation is a regression test which
// back-calculates a minimal percentage for downscaling based on a small percentage
// increase in pod utilization which is calibrated against the tolerance value.
func TestReplicaCalcComputedToleranceAlgImplementation(t *testing.T) {

	startPods := int32(10)
	// 150 mCPU per pod.
	totalUsedCPUOfAllPods := int64(startPods * 150)
	// Each pod starts out asking for 2X what is really needed.
	// This means we will have a 50% ratio of used/requested
	totalRequestedCPUOfAllPods := int32(2 * totalUsedCPUOfAllPods)
	requestedToUsed := float64(totalRequestedCPUOfAllPods / int32(totalUsedCPUOfAllPods))
	// Spread the amount we ask over 10 pods.  We can add some jitter later in reportedLevels.
	perPodRequested := totalRequestedCPUOfAllPods / startPods

	// Force a minimal scaling event by satisfying  (tolerance < 1 - resourcesUsedRatio).
	target := math.Abs(1/(requestedToUsed*(1-tolerance))) + .01
	finalCpuPercentTarget := int32(target * 100)
	resourcesUsedRatio := float64(totalUsedCPUOfAllPods) / float64(float64(totalRequestedCPUOfAllPods)*target)

	// i.e. .60 * 20 -> scaled down expectation.
	finalPods := int32(math.Ceil(resourcesUsedRatio * float64(startPods)))

	// To breach tolerance we will create a utilization ratio difference of tolerance to usageRatioToleranceValue)
	tc := replicaCalcTestCase{
		currentReplicas:  startPods,
		expectedReplicas: finalPods,
		resource: &resourceInfo{
			name: v1.ResourceCPU,
			levels: []int64{
				totalUsedCPUOfAllPods / 10,
				totalUsedCPUOfAllPods / 10,
				totalUsedCPUOfAllPods / 10,
				totalUsedCPUOfAllPods / 10,
				totalUsedCPUOfAllPods / 10,
				totalUsedCPUOfAllPods / 10,
				totalUsedCPUOfAllPods / 10,
				totalUsedCPUOfAllPods / 10,
				totalUsedCPUOfAllPods / 10,
				totalUsedCPUOfAllPods / 10,
			},
			requests: []resource.Quantity{
				resource.MustParse(fmt.Sprint(perPodRequested+100) + "m"),
				resource.MustParse(fmt.Sprint(perPodRequested-100) + "m"),
				resource.MustParse(fmt.Sprint(perPodRequested+10) + "m"),
				resource.MustParse(fmt.Sprint(perPodRequested-10) + "m"),
				resource.MustParse(fmt.Sprint(perPodRequested+2) + "m"),
				resource.MustParse(fmt.Sprint(perPodRequested-2) + "m"),
				resource.MustParse(fmt.Sprint(perPodRequested+1) + "m"),
				resource.MustParse(fmt.Sprint(perPodRequested-1) + "m"),
				resource.MustParse(fmt.Sprint(perPodRequested) + "m"),
				resource.MustParse(fmt.Sprint(perPodRequested) + "m"),
			},

			targetUtilization:   finalCpuPercentTarget,
			expectedUtilization: int32(totalUsedCPUOfAllPods*100) / totalRequestedCPUOfAllPods,
			expectedValue:       numContainersPerPod * totalUsedCPUOfAllPods / 10,
		},
	}

	tc.runTest(t)

	// Reuse the data structure above, now testing "unscaling".
	// Now, we test that no scaling happens if we are in a very close margin to the tolerance
	target = math.Abs(1/(requestedToUsed*(1-tolerance))) + .004
	finalCpuPercentTarget = int32(target * 100)
	tc.resource.targetUtilization = finalCpuPercentTarget
	tc.currentReplicas = startPods
	tc.expectedReplicas = startPods
	tc.runTest(t)
}

// TODO: add more tests
