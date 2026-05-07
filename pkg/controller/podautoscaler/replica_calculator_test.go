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
	"context"
	"fmt"
	"math"
	"testing"
	"time"

	autoscalingv2 "k8s.io/api/autoscaling/v2"
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/meta/testrestmapper"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/sets"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes/fake"
	core "k8s.io/client-go/testing"
	"k8s.io/client-go/tools/cache"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	"k8s.io/kubernetes/pkg/controller"
	metricsclient "k8s.io/kubernetes/pkg/controller/podautoscaler/metrics"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/test/utils/ktesting"
	cmapi "k8s.io/metrics/pkg/apis/custom_metrics/v1beta2"
	emapi "k8s.io/metrics/pkg/apis/external_metrics/v1beta1"
	metricsapi "k8s.io/metrics/pkg/apis/metrics/v1beta1"
	metricsfake "k8s.io/metrics/pkg/client/clientset/versioned/fake"
	cmfake "k8s.io/metrics/pkg/client/custom_metrics/fake"
	emfake "k8s.io/metrics/pkg/client/external_metrics/fake"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

const (
	testNamespace       = "test-namespace"
	podNamePrefix       = "test-pod"
	numContainersPerPod = 2
)

// TODO(omerap12): this will be removed for refactoring
type metricInfo struct {
	name         string
	levels       []int64
	singleObject *autoscalingv2.CrossVersionObjectReference
	selector     *metav1.LabelSelector
	metricType   metricType

	targetUsage       int64
	perPodTargetUsage int64
	expectedUsage     int64
}

// TODO(omerap12): this will be removed for refactoring
type replicaCalcTestCase struct {
	currentReplicas  int32
	expectedReplicas int32
	expectedError    error

	timestamp time.Time

	tolerances *Tolerances
	resource   *resourceInfo
	metric     *metricInfo
	container  string

	podReadiness         []v1.ConditionStatus
	podStartTime         []metav1.Time
	podPhase             []v1.PodPhase
	podDeletionTimestamp []bool
}

// TODO(omerap12): this will be removed for refactoring
type resourceInfo struct {
	name     v1.ResourceName
	requests []resource.Quantity
	levels   [][]int64
	// only applies to pod names returned from "heapster"
	podNames []string

	targetUtilization   int32
	expectedUtilization int32
	expectedValue       int64
}

// TODO(omerap12): this will be removed for refactoring
type metricType int

// TODO(omerap12): this will be removed for refactoring
const (
	objectMetric metricType = iota
	objectPerPodMetric
	externalMetric
	externalPerPodMetric
	podMetric
)

// cpuResource describes the per-container CPU requests and the pod-level metric
// levels that shape the fake pod/metrics clients for resource-based tests.
type cpuResource struct {
	// requests is the per-pod CPU request. len(requests) is the number of pods.
	requests []resource.Quantity
	// levels[i][j] is the usage of container j in pod i. Out-of-range values
	// simulate pods with missing metrics.
	levels [][]int64
	// podNames, if non-empty, overrides the pod name returned by the fake
	// metrics client for index i.
	podNames []string
}

// customMetric describes how the fake custom - or external-metrics client should
// respond in metric-based tests.
type customMetric struct {
	name string
	// levels are the values returned by the metrics client, one per pod
	// (or one per external-metric sample).
	levels []int64
	// singleObject is required for object and objectPerPod metric tests.
	singleObject *autoscalingv2.CrossVersionObjectReference
	// selector is required for external and externalPerPod metric tests.
	selector *metav1.LabelSelector
}

// calcScenario is a pure input bundle: it describes what pods exist, what
// metrics they report, and what readiness/phase they are in.
type calcScenario struct {
	currentReplicas int32

	// Exactly one of resource or metric is set.
	resource *cpuResource
	metric   *customMetric

	// container, if set, scopes GetResourceReplicas to a specific container.
	container string

	// timestamp used by the fake metrics clients for all returned samples.
	timestamp time.Time

	// Per-pod state. When non-nil, each slice must be at least currentReplicas
	// long (or longer if the fixture is also describing failed/deleted pods
	// that aren't counted in currentReplicas).
	podReadiness         []v1.ConditionStatus
	podStartTime         []metav1.Time
	podPhase             []v1.PodPhase
	podDeletionTimestamp []bool
}

// cpuRequests returns n identical CPU resource quantities parsed from val.
func cpuRequests(n int, val string) []resource.Quantity {
	requests := make([]resource.Quantity, n)
	for i := range requests {
		requests[i] = resource.MustParse(val)
	}
	return requests
}

// makePodMetricLevels returns a [pod][container]int64 where every container in
// pod i reports the same usage containerMetric[i].
func makePodMetricLevels(containerMetric ...int64) [][]int64 {
	metrics := make([][]int64, len(containerMetric))
	for i := range containerMetric {
		metrics[i] = make([]int64, numContainersPerPod)
		for j := range numContainersPerPod {
			metrics[i][j] = containerMetric[i]
		}
	}
	return metrics
}

// replicaCalcSetup holds a ReplicaCalculator and the fake clients it uses.
// Tests call methods on .calc and check the result.
type replicaCalcSetup struct {
	calc       *ReplicaCalculator
	ctx        context.Context
	selector   labels.Selector
	namespace  string
	tolerances Tolerances
}

// resourceCase bundles a single input fixture with expected outputs for the resource-based replica tests.
type resourceCase struct {
	name    string
	fixture calcScenario

	targetUtilization   int32
	expectedReplicas    int32
	expectedUtilization int32
	expectedRawValue    int64
	expectedError       error

	// tolerances, if non-nil, overrides the default setup tolerances.
	tolerances *Tolerances
}

// newReplicaCalcSetup creates fake pod, metrics, custom-metrics, and
// external-metrics clients and adds them to ReplicaCalculator within replicaCalcSetup.
func newReplicaCalcSetup(t *testing.T, f *calcScenario) *replicaCalcSetup {
	t.Helper()

	tCtx := ktesting.Init(t)

	podClient := newFakePodClient(f)
	metricsClient := metricsclient.NewRESTMetricsClient(
		newFakeResourceMetricsClient(f).MetricsV1beta1(),
		newFakeCustomMetricsClient(t, f),
		newFakeExternalMetricsClient(t, f),
	)

	informerFactory := informers.NewSharedInformerFactory(podClient, controller.NoResyncPeriodFunc())
	informer := informerFactory.Core().V1().Pods()

	calc := NewReplicaCalculator(metricsClient, informer.Lister(),
		defaultTestingCPUInitializationPeriod, defaultTestingDelayOfInitialReadinessStatus)

	informerFactory.Start(tCtx.Done())

	syncCtx, cancel := context.WithTimeout(tCtx, 10*time.Second)
	defer cancel()
	if !cache.WaitForNamedCacheSyncWithContext(syncCtx, informer.Informer().HasSynced) {
		tCtx.Fatal("Failed to sync informer cache within the 10s timeout")
	}

	selector, err := metav1.LabelSelectorAsSelector(&metav1.LabelSelector{
		MatchLabels: map[string]string{"name": podNamePrefix},
	})
	require.NoError(t, err, "test pod label selector should parse")

	return &replicaCalcSetup{
		calc:       calc,
		ctx:        tCtx,
		selector:   selector,
		namespace:  testNamespace,
		tolerances: Tolerances{defaultTestingTolerance, defaultTestingTolerance},
	}
}

// newFakePodClient returns a fake Kubernetes clientset that lists pods.
func newFakePodClient(f *calcScenario) *fake.Clientset {
	fakeClient := &fake.Clientset{}
	fakeClient.AddReactor("list", "pods", func(action core.Action) (handled bool, ret runtime.Object, err error) {
		obj := &v1.PodList{}
		podsCount := int(f.currentReplicas)
		// Failed pods aren't included in currentReplicas.
		if f.podPhase != nil && len(f.podPhase) > podsCount {
			podsCount = len(f.podPhase)
		}
		for i := 0; i < podsCount; i++ {
			obj.Items = append(obj.Items, buildFakePod(f, i))
		}
		return true, obj, nil
	})
	return fakeClient
}

// buildFakePod build a fake pod based on calcScenario's pod index.
func buildFakePod(f *calcScenario, i int) v1.Pod {
	podReadiness := v1.ConditionTrue
	if f.podReadiness != nil && i < len(f.podReadiness) {
		podReadiness = f.podReadiness[i]
	}
	var podStartTime metav1.Time
	if f.podStartTime != nil {
		podStartTime = f.podStartTime[i]
	}
	podPhase := v1.PodRunning
	if f.podPhase != nil {
		podPhase = f.podPhase[i]
	}
	deleted := f.podDeletionTimestamp != nil && f.podDeletionTimestamp[i]

	pod := v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      fmt.Sprintf("%s-%d", podNamePrefix, i),
			Namespace: testNamespace,
			Labels:    map[string]string{"name": podNamePrefix},
		},
		Status: v1.PodStatus{
			Phase:     podPhase,
			StartTime: &podStartTime,
			Conditions: []v1.PodCondition{
				{Type: v1.PodReady, Status: podReadiness},
			},
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{{Name: "container1"}, {Name: "container2"}},
		},
	}
	if deleted {
		pod.DeletionTimestamp = &metav1.Time{Time: time.Now()}
	}
	if f.resource != nil && i < len(f.resource.requests) {
		req := v1.ResourceRequirements{
			Requests: v1.ResourceList{v1.ResourceCPU: f.resource.requests[i]},
		}
		pod.Spec.Containers[0].Resources = req
		pod.Spec.Containers[1].Resources = req
	}
	return pod
}

// newFakeResourceMetricsClient returns a fake metrics.k8s.io client.
func newFakeResourceMetricsClient(f *calcScenario) *metricsfake.Clientset {
	fakeMetricsClient := &metricsfake.Clientset{}
	fakeMetricsClient.AddReactor("list", "pods", func(action core.Action) (handled bool, ret runtime.Object, err error) {
		if f.resource == nil {
			return true, nil, fmt.Errorf("no pod resource metrics specified in test client")
		}
		metrics := &metricsapi.PodMetricsList{}
		for i, resValue := range f.resource.levels {
			podName := fmt.Sprintf("%s-%d", podNamePrefix, i)
			if len(f.resource.podNames) > i {
				podName = f.resource.podNames[i]
			}
			// The list reactor does label-selector filtering for us, so the
			// returned pod metrics must match the selector.
			podMetric := metricsapi.PodMetrics{
				ObjectMeta: metav1.ObjectMeta{
					Name:      podName,
					Namespace: testNamespace,
					Labels:    map[string]string{"name": podNamePrefix},
				},
				Timestamp:  metav1.Time{Time: f.timestamp},
				Window:     metav1.Duration{Duration: time.Minute},
				Containers: make([]metricsapi.ContainerMetrics, numContainersPerPod),
			}
			for j, m := range resValue {
				podMetric.Containers[j] = metricsapi.ContainerMetrics{
					Name: fmt.Sprintf("container%v", j+1),
					Usage: v1.ResourceList{
						v1.ResourceCPU: *resource.NewMilliQuantity(m, resource.DecimalSI),
					},
				}
			}
			metrics.Items = append(metrics.Items, podMetric)
		}
		return true, metrics, nil
	})
	return fakeMetricsClient
}

// newFakeCustomMetricsClient returns a fake custom-metrics client.
func newFakeCustomMetricsClient(t *testing.T, f *calcScenario) *cmfake.FakeCustomMetricsClient {
	fakeCMClient := &cmfake.FakeCustomMetricsClient{}
	fakeCMClient.AddReactor("get", "*", func(action core.Action) (handled bool, ret runtime.Object, err error) {
		getForAction, wasGetFor := action.(cmfake.GetForAction)
		if !wasGetFor {
			return true, nil, fmt.Errorf("expected a get-for action, got %v instead", action)
		}
		if f.metric == nil {
			return true, nil, fmt.Errorf("no custom metrics specified in test client")
		}
		assert.Equal(t, f.metric.name, getForAction.GetMetricName(), "the metric requested should have matched the one specified")

		if getForAction.GetName() == "*" {
			// Multiple-pod query.
			assert.Equal(t, "pods", getForAction.GetResource().Resource, "the type of object that we requested multiple metrics for should have been pods")
			metrics := cmapi.MetricValueList{}
			for i, level := range f.metric.levels {
				metrics.Items = append(metrics.Items, cmapi.MetricValue{
					DescribedObject: v1.ObjectReference{
						Kind:      "Pod",
						Name:      fmt.Sprintf("%s-%d", podNamePrefix, i),
						Namespace: testNamespace,
					},
					Timestamp: metav1.Time{Time: f.timestamp},
					Metric:    cmapi.MetricIdentifier{Name: f.metric.name},
					Value:     *resource.NewMilliQuantity(level, resource.DecimalSI),
				})
			}
			return true, &metrics, nil
		}

		// Single-object query.
		name := getForAction.GetName()
		assert.NotNil(t, f.metric.singleObject, "should have only requested a single-object metric when calling GetObjectMetricReplicas")
		mapper := testrestmapper.TestOnlyStaticRESTMapper(legacyscheme.Scheme)
		gk := schema.FromAPIVersionAndKind(f.metric.singleObject.APIVersion, f.metric.singleObject.Kind).GroupKind()
		mapping, err := mapper.RESTMapping(gk)
		if err != nil {
			return true, nil, fmt.Errorf("unable to get mapping for %s: %w", gk.String(), err)
		}
		assert.Equal(t, mapping.Resource.GroupResource().String(), getForAction.GetResource().Resource, "should have requested metrics for the resource matching the GroupKind passed in")
		assert.Equal(t, f.metric.singleObject.Name, name, "should have requested metrics for the object matching the name passed in")

		return true, &cmapi.MetricValueList{
			Items: []cmapi.MetricValue{{
				DescribedObject: v1.ObjectReference{
					Kind:       f.metric.singleObject.Kind,
					APIVersion: f.metric.singleObject.APIVersion,
					Name:       name,
				},
				Timestamp: metav1.Time{Time: f.timestamp},
				Metric:    cmapi.MetricIdentifier{Name: f.metric.name},
				Value:     *resource.NewMilliQuantity(f.metric.levels[0], resource.DecimalSI),
			}},
		}, nil
	})
	return fakeCMClient
}

// newFakeExternalMetricsClient returns a fake external-metrics client.
func newFakeExternalMetricsClient(t *testing.T, f *calcScenario) *emfake.FakeExternalMetricsClient {
	fakeEMClient := &emfake.FakeExternalMetricsClient{}
	fakeEMClient.AddReactor("list", "*", func(action core.Action) (handled bool, ret runtime.Object, err error) {
		listAction, wasList := action.(core.ListAction)
		if !wasList {
			return true, nil, fmt.Errorf("expected a list-for action, got %v instead", action)
		}
		if f.metric == nil {
			return true, nil, fmt.Errorf("no external metrics specified in test client")
		}
		assert.Equal(t, f.metric.name, listAction.GetResource().Resource, "the metric requested should have matched the one specified")

		selector, err := metav1.LabelSelectorAsSelector(f.metric.selector)
		if err != nil {
			return true, nil, fmt.Errorf("failed to convert label selector specified in test client")
		}
		assert.Equal(t, selector, listAction.GetListRestrictions().Labels, "the metric selector should have matched the one specified")

		metrics := emapi.ExternalMetricValueList{}
		for _, level := range f.metric.levels {
			metrics.Items = append(metrics.Items, emapi.ExternalMetricValue{
				Timestamp:  metav1.Time{Time: f.timestamp},
				MetricName: f.metric.name,
				Value:      *resource.NewMilliQuantity(level, resource.DecimalSI),
			})
		}
		return true, &metrics, nil
	})
	return fakeEMClient
}

// assertResourceReplicas checks the result of GetResourceReplicas against
// expected values (or expected error) and verifies the metric timestamp.
func assertResourceReplicas(t *testing.T, wantReplicas int32, wantUtilization int32, wantRawValue int64, wantTimestamp time.Time, wantErr error,
	gotReplicas int32, gotUtilization int32, gotRawValue int64, gotTimestamp time.Time, gotErr error) {
	t.Helper()
	if wantErr != nil {
		require.Error(t, gotErr, "there should be an error calculating the replica count")
		assert.ErrorContains(t, gotErr, wantErr.Error(), "error message should contain expected text")
		return
	}
	require.NoError(t, gotErr, "there should not have been an error calculating the replica count")
	assert.Equal(t, wantReplicas, gotReplicas, "replicas should be as expected")
	assert.Equal(t, wantUtilization, gotUtilization, "utilization should be as expected")
	assert.Equal(t, wantRawValue, gotRawValue, "raw value should be as expected")
	assert.True(t, wantTimestamp.Equal(gotTimestamp), "timestamp should be as expected")
}

// TestReplicaCalcResourceScale covers scale-up, scale-down, and no-scale scenarios for GetResourceReplicas.
func TestReplicaCalcResourceScale(t *testing.T) {
	testCases := []resourceCase{
		{
			name: "scale up",
			fixture: calcScenario{
				currentReplicas: 3,
				resource: &cpuResource{
					requests: cpuRequests(3, "1.0"),
					levels:   makePodMetricLevels(300, 500, 700),
				},
			},
			targetUtilization:   30,
			expectedReplicas:    5,
			expectedUtilization: 50,
			expectedRawValue:    numContainersPerPod * 500,
		},
		{
			name: "scale up: container metric",
			fixture: calcScenario{
				currentReplicas: 3,
				container:       "container2",
				resource: &cpuResource{
					requests: cpuRequests(3, "1.0"),
					levels:   [][]int64{{1000, 300}, {1000, 500}, {1000, 700}},
				},
			},
			targetUtilization:   30,
			expectedReplicas:    5,
			expectedUtilization: 50,
			expectedRawValue:    500,
		},
		{
			name: "scale up: unready pod scales less",
			fixture: calcScenario{
				currentReplicas: 3,
				podReadiness:    []v1.ConditionStatus{v1.ConditionFalse, v1.ConditionTrue, v1.ConditionTrue},
				resource: &cpuResource{
					requests: cpuRequests(3, "1.0"),
					levels:   makePodMetricLevels(300, 500, 700),
				},
			},
			targetUtilization:   30,
			expectedReplicas:    4,
			expectedUtilization: 60,
			expectedRawValue:    numContainersPerPod * 600,
		},
		{
			name: "scale up: hot-CPU container scales less",
			fixture: calcScenario{
				currentReplicas: 3,
				podStartTime:    []metav1.Time{hotCPUCreationTime(), coolCPUCreationTime(), coolCPUCreationTime()},
				container:       "container2",
				resource: &cpuResource{
					requests: cpuRequests(3, "1.0"),
					levels:   [][]int64{{0, 300}, {0, 500}, {0, 700}},
				},
			},
			targetUtilization:   30,
			expectedReplicas:    4,
			expectedUtilization: 60,
			expectedRawValue:    600,
		},
		{
			name: "scale up: hot-CPU pod scales less",
			fixture: calcScenario{
				currentReplicas: 3,
				podStartTime:    []metav1.Time{hotCPUCreationTime(), coolCPUCreationTime(), coolCPUCreationTime()},
				resource: &cpuResource{
					requests: cpuRequests(3, "1.0"),
					levels:   makePodMetricLevels(300, 500, 700),
				},
			},
			targetUtilization:   30,
			expectedReplicas:    4,
			expectedUtilization: 60,
			expectedRawValue:    numContainersPerPod * 600,
		},
		{
			name: "no scale: unready pods ignored",
			fixture: calcScenario{
				currentReplicas: 3,
				podReadiness:    []v1.ConditionStatus{v1.ConditionTrue, v1.ConditionFalse, v1.ConditionFalse},
				resource: &cpuResource{
					requests: cpuRequests(3, "1.0"),
					levels:   makePodMetricLevels(400, 500, 700),
				},
			},
			targetUtilization:   30,
			expectedReplicas:    3,
			expectedUtilization: 40,
			expectedRawValue:    numContainersPerPod * 400,
		},
		{
			name: "no scale: hot-CPU pods ignored",
			fixture: calcScenario{
				currentReplicas: 3,
				podReadiness:    []v1.ConditionStatus{v1.ConditionTrue, v1.ConditionFalse, v1.ConditionFalse},
				podStartTime:    []metav1.Time{coolCPUCreationTime(), hotCPUCreationTime(), hotCPUCreationTime()},
				resource: &cpuResource{
					requests: cpuRequests(3, "1.0"),
					levels:   makePodMetricLevels(400, 500, 700),
				},
			},
			targetUtilization:   30,
			expectedReplicas:    3,
			expectedUtilization: 40,
			expectedRawValue:    numContainersPerPod * 400,
		},
		{
			name: "scale up: failed pods ignored",
			fixture: calcScenario{
				currentReplicas: 2,
				podReadiness:    []v1.ConditionStatus{v1.ConditionTrue, v1.ConditionTrue, v1.ConditionFalse, v1.ConditionFalse},
				podPhase:        []v1.PodPhase{v1.PodRunning, v1.PodRunning, v1.PodFailed, v1.PodFailed},
				resource: &cpuResource{
					requests: cpuRequests(4, "1.0"),
					levels:   makePodMetricLevels(500, 700),
				},
			},
			targetUtilization:   30,
			expectedReplicas:    4,
			expectedUtilization: 60,
			expectedRawValue:    numContainersPerPod * 600,
		},
		{
			name: "scale up: failed pods ignored with missing container metric",
			fixture: calcScenario{
				currentReplicas: 2,
				podReadiness:    []v1.ConditionStatus{v1.ConditionTrue, v1.ConditionTrue, v1.ConditionFalse, v1.ConditionFalse},
				podPhase:        []v1.PodPhase{v1.PodRunning, v1.PodRunning, v1.PodFailed, v1.PodFailed},
				container:       "container2",
				resource: &cpuResource{
					requests: cpuRequests(4, "1.0"),
					levels:   [][]int64{{1000, 500}, {9000, 700}, {1000}, {9000}},
				},
			},
			targetUtilization:   30,
			expectedReplicas:    4,
			expectedUtilization: 60,
			expectedRawValue:    600,
		},
		{
			name: "scale up: container metrics with failed pods ignored",
			fixture: calcScenario{
				currentReplicas: 2,
				podReadiness:    []v1.ConditionStatus{v1.ConditionTrue, v1.ConditionTrue, v1.ConditionFalse, v1.ConditionFalse},
				podPhase:        []v1.PodPhase{v1.PodRunning, v1.PodRunning, v1.PodFailed, v1.PodFailed},
				container:       "container2",
				resource: &cpuResource{
					requests: cpuRequests(4, "1.0"),
					levels:   [][]int64{{1000, 500}, {9000, 700}},
				},
			},
			targetUtilization:   30,
			expectedReplicas:    4,
			expectedUtilization: 60,
			expectedRawValue:    600,
		},
		{
			name: "scale up: pods being deleted are ignored",
			fixture: calcScenario{
				currentReplicas:      2,
				podReadiness:         []v1.ConditionStatus{v1.ConditionTrue, v1.ConditionTrue, v1.ConditionFalse, v1.ConditionFalse},
				podPhase:             []v1.PodPhase{v1.PodRunning, v1.PodRunning, v1.PodRunning, v1.PodRunning},
				podDeletionTimestamp: []bool{false, false, true, true},
				resource: &cpuResource{
					requests: cpuRequests(4, "1.0"),
					levels:   makePodMetricLevels(500, 700),
				},
			},
			targetUtilization:   30,
			expectedReplicas:    4,
			expectedUtilization: 60,
			expectedRawValue:    numContainersPerPod * 600,
		},
		{
			name: "scale up: container metrics with pods being deleted ignored",
			fixture: calcScenario{
				currentReplicas:      2,
				podReadiness:         []v1.ConditionStatus{v1.ConditionTrue, v1.ConditionTrue, v1.ConditionFalse, v1.ConditionFalse},
				podPhase:             []v1.PodPhase{v1.PodRunning, v1.PodRunning, v1.PodRunning, v1.PodRunning},
				podDeletionTimestamp: []bool{false, false, true, true},
				container:            "container1",
				resource: &cpuResource{
					requests: cpuRequests(4, "1.0"),
					levels:   makePodMetricLevels(500, 700), // TODO: This test is broken and works only because of missing metrics.
				},
			},
			targetUtilization:   30,
			expectedReplicas:    4,
			expectedUtilization: 60,
			expectedRawValue:    600,
		},
		{
			name: "scale down",
			fixture: calcScenario{
				currentReplicas: 5,
				resource: &cpuResource{
					requests: cpuRequests(5, "1.0"),
					levels:   makePodMetricLevels(100, 300, 500, 250, 250),
				},
			},
			targetUtilization:   50,
			expectedReplicas:    3,
			expectedUtilization: 28,
			expectedRawValue:    numContainersPerPod * 280,
		},
		{
			name: "scale down: container metric",
			fixture: calcScenario{
				currentReplicas: 5,
				container:       "container2",
				resource: &cpuResource{
					requests: cpuRequests(5, "1.0"),
					levels:   [][]int64{{1000, 100}, {1000, 300}, {1000, 500}, {1000, 250}, {1000, 250}},
				},
			},
			targetUtilization:   50,
			expectedReplicas:    3,
			expectedUtilization: 28,
			expectedRawValue:    280,
		},
		{
			name: "scale down: exclude unready pods",
			fixture: calcScenario{
				currentReplicas: 5,
				podReadiness:    []v1.ConditionStatus{v1.ConditionTrue, v1.ConditionTrue, v1.ConditionTrue, v1.ConditionFalse, v1.ConditionFalse},
				resource: &cpuResource{
					requests: cpuRequests(5, "1.0"),
					levels:   makePodMetricLevels(100, 300, 500, 250, 250),
				},
			},
			targetUtilization:   50,
			expectedReplicas:    2,
			expectedUtilization: 30,
			expectedRawValue:    numContainersPerPod * 300,
		},
		{
			name: "scale down: container metric excludes unready pods",
			fixture: calcScenario{
				currentReplicas: 5,
				podReadiness:    []v1.ConditionStatus{v1.ConditionTrue, v1.ConditionTrue, v1.ConditionTrue, v1.ConditionFalse, v1.ConditionFalse},
				container:       "container2",
				resource: &cpuResource{
					requests: cpuRequests(5, "1.0"),
					levels:   [][]int64{{1000, 100}, {1000, 300}, {1000, 500}, {1000, 250}, {1000, 250}},
				},
			},
			targetUtilization:   50,
			expectedReplicas:    2,
			expectedUtilization: 30,
			expectedRawValue:    300,
		},
		{
			name: "scale down: exclude unscheduled pods",
			fixture: calcScenario{
				currentReplicas: 5,
				podReadiness:    []v1.ConditionStatus{v1.ConditionTrue, v1.ConditionFalse, v1.ConditionFalse, v1.ConditionFalse, v1.ConditionFalse},
				podPhase:        []v1.PodPhase{v1.PodRunning, v1.PodPending, v1.PodPending, v1.PodPending, v1.PodPending},
				resource: &cpuResource{
					requests: cpuRequests(5, "1.0"),
					levels:   makePodMetricLevels(100),
				},
			},
			targetUtilization:   50,
			expectedReplicas:    1,
			expectedUtilization: 10,
			expectedRawValue:    numContainersPerPod * 100,
		},
		{
			name: "scale down: container metric excludes unscheduled pods",
			fixture: calcScenario{
				currentReplicas: 5,
				podReadiness:    []v1.ConditionStatus{v1.ConditionTrue, v1.ConditionFalse, v1.ConditionFalse, v1.ConditionFalse, v1.ConditionFalse},
				podPhase:        []v1.PodPhase{v1.PodRunning, v1.PodPending, v1.PodPending, v1.PodPending, v1.PodPending},
				container:       "container2",
				resource: &cpuResource{
					requests: cpuRequests(5, "1.0"),
					levels:   [][]int64{{1000, 100}, {1000, 300}, {1000, 500}, {1000, 250}, {1000, 250}},
				},
			},
			targetUtilization:   50,
			expectedReplicas:    1,
			expectedUtilization: 10,
			expectedRawValue:    100,
		},
		{
			name: "scale down: ignore hot-CPU pods",
			fixture: calcScenario{
				currentReplicas: 5,
				podStartTime:    []metav1.Time{coolCPUCreationTime(), coolCPUCreationTime(), coolCPUCreationTime(), hotCPUCreationTime(), hotCPUCreationTime()},
				resource: &cpuResource{
					requests: cpuRequests(5, "1.0"),
					levels:   makePodMetricLevels(100, 300, 500, 250, 250),
				},
			},
			targetUtilization:   50,
			expectedReplicas:    2,
			expectedUtilization: 30,
			expectedRawValue:    numContainersPerPod * 300,
		},
		{
			name: "scale down: container metric ignores hot-CPU pods",
			fixture: calcScenario{
				currentReplicas: 5,
				podStartTime:    []metav1.Time{coolCPUCreationTime(), coolCPUCreationTime(), coolCPUCreationTime(), hotCPUCreationTime(), hotCPUCreationTime()},
				container:       "container2",
				resource: &cpuResource{
					requests: cpuRequests(5, "1.0"),
					levels:   [][]int64{{1000, 100}, {1000, 300}, {1000, 500}, {1000, 1000}, {1000, 1000}},
				},
			},
			targetUtilization:   50,
			expectedReplicas:    2,
			expectedUtilization: 30,
			expectedRawValue:    300,
		},
		{
			name: "scale down: failed pods ignored",
			fixture: calcScenario{
				currentReplicas: 5,
				podReadiness:    []v1.ConditionStatus{v1.ConditionTrue, v1.ConditionTrue, v1.ConditionTrue, v1.ConditionTrue, v1.ConditionTrue, v1.ConditionFalse, v1.ConditionFalse},
				podPhase:        []v1.PodPhase{v1.PodRunning, v1.PodRunning, v1.PodRunning, v1.PodRunning, v1.PodRunning, v1.PodFailed, v1.PodFailed},
				resource: &cpuResource{
					requests: cpuRequests(7, "1.0"),
					levels:   makePodMetricLevels(100, 300, 500, 250, 250),
				},
			},
			targetUtilization:   50,
			expectedReplicas:    3,
			expectedUtilization: 28,
			expectedRawValue:    numContainersPerPod * 280,
		},
		{
			name: "scale down: container metric with failed pods ignored",
			fixture: calcScenario{
				currentReplicas: 5,
				podReadiness:    []v1.ConditionStatus{v1.ConditionTrue, v1.ConditionTrue, v1.ConditionTrue, v1.ConditionTrue, v1.ConditionTrue, v1.ConditionFalse, v1.ConditionFalse},
				podPhase:        []v1.PodPhase{v1.PodRunning, v1.PodRunning, v1.PodRunning, v1.PodRunning, v1.PodRunning, v1.PodFailed, v1.PodFailed},
				container:       "container2",
				resource: &cpuResource{
					requests: cpuRequests(7, "1.0"),
					levels:   [][]int64{{1000, 100}, {1000, 300}, {1000, 500}, {1000, 250}, {1000, 250}}, // TODO: Test is broken.
				},
			},
			targetUtilization:   50,
			expectedReplicas:    3,
			expectedUtilization: 28,
			expectedRawValue:    280,
		},
		{
			name: "scale down: pods being deleted are ignored",
			fixture: calcScenario{
				currentReplicas:      5,
				podReadiness:         []v1.ConditionStatus{v1.ConditionTrue, v1.ConditionTrue, v1.ConditionTrue, v1.ConditionTrue, v1.ConditionTrue, v1.ConditionFalse, v1.ConditionFalse},
				podPhase:             []v1.PodPhase{v1.PodRunning, v1.PodRunning, v1.PodRunning, v1.PodRunning, v1.PodRunning, v1.PodRunning, v1.PodRunning},
				podDeletionTimestamp: []bool{false, false, false, false, false, true, true},
				resource: &cpuResource{
					requests: cpuRequests(7, "1.0"),
					levels:   makePodMetricLevels(100, 300, 500, 250, 250),
				},
			},
			targetUtilization:   50,
			expectedReplicas:    3,
			expectedUtilization: 28,
			expectedRawValue:    numContainersPerPod * 280,
		},
		{
			// Regression test for https://github.com/kubernetes/kubernetes/issues/83561.
			name: "scale down: still-running pods being deleted are ignored",
			fixture: calcScenario{
				currentReplicas:      5,
				podReadiness:         []v1.ConditionStatus{v1.ConditionTrue, v1.ConditionTrue, v1.ConditionTrue, v1.ConditionTrue, v1.ConditionTrue, v1.ConditionFalse, v1.ConditionFalse},
				podPhase:             []v1.PodPhase{v1.PodRunning, v1.PodRunning, v1.PodRunning, v1.PodRunning, v1.PodRunning, v1.PodRunning, v1.PodRunning},
				podDeletionTimestamp: []bool{false, false, false, false, false, true, true},
				container:            "container2",
				resource: &cpuResource{
					requests: cpuRequests(7, "1.0"),
					levels:   [][]int64{{1000, 100}, {1000, 300}, {1000, 500}, {1000, 250}, {1000, 250}},
				},
			},
			targetUtilization:   50,
			expectedReplicas:    3,
			expectedUtilization: 28,
			expectedRawValue:    280,
		},
	}
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			h := newReplicaCalcSetup(t, &tc.fixture)
			if tc.tolerances != nil {
				h.tolerances = *tc.tolerances
			}
			replicas, util, raw, ts, err := h.calc.GetResourceReplicas(
				h.ctx, tc.fixture.currentReplicas, tc.targetUtilization,
				v1.ResourceCPU, h.tolerances, h.namespace, h.selector, tc.fixture.container,
			)
			assertResourceReplicas(t,
				tc.expectedReplicas, tc.expectedUtilization, tc.expectedRawValue, tc.fixture.timestamp, tc.expectedError,
				replicas, util, raw, ts, err,
			)
		})
	}
}

func (tc *replicaCalcTestCase) prepareTestClientSet() *fake.Clientset {
	fakeClient := &fake.Clientset{}
	fakeClient.AddReactor("list", "pods", func(action core.Action) (handled bool, ret runtime.Object, err error) {
		obj := &v1.PodList{}
		podsCount := int(tc.currentReplicas)
		// Failed pods are not included in tc.currentReplicas
		if tc.podPhase != nil && len(tc.podPhase) > podsCount {
			podsCount = len(tc.podPhase)
		}
		for i := 0; i < podsCount; i++ {
			podReadiness := v1.ConditionTrue
			if tc.podReadiness != nil && i < len(tc.podReadiness) {
				podReadiness = tc.podReadiness[i]
			}
			var podStartTime metav1.Time
			if tc.podStartTime != nil {
				podStartTime = tc.podStartTime[i]
			}
			podPhase := v1.PodRunning
			if tc.podPhase != nil {
				podPhase = tc.podPhase[i]
			}
			podDeletionTimestamp := false
			if tc.podDeletionTimestamp != nil {
				podDeletionTimestamp = tc.podDeletionTimestamp[i]
			}
			podName := fmt.Sprintf("%s-%d", podNamePrefix, i)
			pod := v1.Pod{
				Status: v1.PodStatus{
					Phase:     podPhase,
					StartTime: &podStartTime,
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
					Containers: []v1.Container{{Name: "container1"}, {Name: "container2"}},
				},
			}
			if podDeletionTimestamp {
				pod.DeletionTimestamp = &metav1.Time{Time: time.Now()}
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
	return fakeClient
}

func (tc *replicaCalcTestCase) prepareTestMetricsClient() *metricsfake.Clientset {
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
					Window:     metav1.Duration{Duration: time.Minute},
					Containers: make([]metricsapi.ContainerMetrics, numContainersPerPod),
				}

				for i, m := range resValue {
					podMetric.Containers[i] = metricsapi.ContainerMetrics{
						Name: fmt.Sprintf("container%v", i+1),
						Usage: v1.ResourceList{
							tc.resource.name: *resource.NewMilliQuantity(m, resource.DecimalSI),
						},
					}
				}
				metrics.Items = append(metrics.Items, podMetric)
			}
			return true, metrics, nil
		}

		return true, nil, fmt.Errorf("no pod resource metrics specified in test client")
	})
	return fakeMetricsClient
}

func (tc *replicaCalcTestCase) prepareTestCMClient(t *testing.T) *cmfake.FakeCustomMetricsClient {
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
					Timestamp: metav1.Time{Time: tc.timestamp},
					Metric: cmapi.MetricIdentifier{
						Name: tc.metric.name,
					},
					Value: *resource.NewMilliQuantity(level, resource.DecimalSI),
				}
				metrics.Items = append(metrics.Items, podMetric)
			}

			return true, &metrics, nil
		}
		name := getForAction.GetName()
		mapper := testrestmapper.TestOnlyStaticRESTMapper(legacyscheme.Scheme)
		metrics := &cmapi.MetricValueList{}
		assert.NotNil(t, tc.metric.singleObject, "should have only requested a single-object metric when calling GetObjectMetricReplicas")
		gk := schema.FromAPIVersionAndKind(tc.metric.singleObject.APIVersion, tc.metric.singleObject.Kind).GroupKind()
		mapping, err := mapper.RESTMapping(gk)
		if err != nil {
			return true, nil, fmt.Errorf("unable to get mapping for %s: %w", gk.String(), err)
		}
		groupResource := mapping.Resource.GroupResource()

		assert.Equal(t, groupResource.String(), getForAction.GetResource().Resource, "should have requested metrics for the resource matching the GroupKind passed in")
		assert.Equal(t, tc.metric.singleObject.Name, name, "should have requested metrics for the object matching the name passed in")

		metrics.Items = []cmapi.MetricValue{
			{
				DescribedObject: v1.ObjectReference{
					Kind:       tc.metric.singleObject.Kind,
					APIVersion: tc.metric.singleObject.APIVersion,
					Name:       name,
				},
				Timestamp: metav1.Time{Time: tc.timestamp},
				Metric: cmapi.MetricIdentifier{
					Name: tc.metric.name,
				},
				Value: *resource.NewMilliQuantity(tc.metric.levels[0], resource.DecimalSI),
			},
		}

		return true, metrics, nil
	})
	return fakeCMClient
}

func (tc *replicaCalcTestCase) prepareTestEMClient(t *testing.T) *emfake.FakeExternalMetricsClient {
	fakeEMClient := &emfake.FakeExternalMetricsClient{}
	fakeEMClient.AddReactor("list", "*", func(action core.Action) (handled bool, ret runtime.Object, err error) {
		listAction, wasList := action.(core.ListAction)
		if !wasList {
			return true, nil, fmt.Errorf("expected a list-for action, got %v instead", action)
		}

		if tc.metric == nil {
			return true, nil, fmt.Errorf("no external metrics specified in test client")
		}

		assert.Equal(t, tc.metric.name, listAction.GetResource().Resource, "the metric requested should have matched the one specified")

		selector, err := metav1.LabelSelectorAsSelector(tc.metric.selector)
		if err != nil {
			return true, nil, fmt.Errorf("failed to convert label selector specified in test client")
		}
		assert.Equal(t, selector, listAction.GetListRestrictions().Labels, "the metric selector should have matched the one specified")

		metrics := emapi.ExternalMetricValueList{}

		for _, level := range tc.metric.levels {
			metric := emapi.ExternalMetricValue{
				Timestamp:  metav1.Time{Time: tc.timestamp},
				MetricName: tc.metric.name,
				Value:      *resource.NewMilliQuantity(level, resource.DecimalSI),
			}
			metrics.Items = append(metrics.Items, metric)
		}

		return true, &metrics, nil
	})
	return fakeEMClient
}

func (tc *replicaCalcTestCase) prepareTestClient(t *testing.T) (*fake.Clientset, *metricsfake.Clientset, *cmfake.FakeCustomMetricsClient, *emfake.FakeExternalMetricsClient) {
	fakeClient := tc.prepareTestClientSet()
	fakeMetricsClient := tc.prepareTestMetricsClient()
	fakeCMClient := tc.prepareTestCMClient(t)
	fakeEMClient := tc.prepareTestEMClient(t)
	return fakeClient, fakeMetricsClient, fakeCMClient, fakeEMClient
}

func (tc *replicaCalcTestCase) runTest(t *testing.T) {
	// Create the special test-aware context.
	tCtx := ktesting.Init(t)

	testClient, testMetricsClient, testCMClient, testEMClient := tc.prepareTestClient(t)
	metricsClient := metricsclient.NewRESTMetricsClient(testMetricsClient.MetricsV1beta1(), testCMClient, testEMClient)

	informerFactory := informers.NewSharedInformerFactory(testClient, controller.NoResyncPeriodFunc())
	informer := informerFactory.Core().V1().Pods()

	replicaCalc := NewReplicaCalculator(metricsClient, informer.Lister(), defaultTestingCPUInitializationPeriod, defaultTestingDelayOfInitialReadinessStatus)

	// Use the test context's Done() channel to manage the informer's lifecycle.
	informerFactory.Start(tCtx.Done())

	// Create a new context specifically for the cache sync operation with a timeout.
	syncCtx, cancel := context.WithTimeout(tCtx, 10*time.Second)
	defer cancel()

	if !cache.WaitForNamedCacheSyncWithContext(syncCtx, informer.Informer().HasSynced) {
		tCtx.Fatal("Failed to sync informer cache within the 10s timeout")
	}

	selector, err := metav1.LabelSelectorAsSelector(&metav1.LabelSelector{
		MatchLabels: map[string]string{"name": podNamePrefix},
	})
	require.NoError(t, err, "something went horribly wrong...")

	// Use default if tolerances are not specified in the test case.
	tolerances := Tolerances{defaultTestingTolerance, defaultTestingTolerance}
	if tc.tolerances != nil {
		tolerances = *tc.tolerances
	}

	if tc.resource != nil {
		outReplicas, outUtilization, outRawValue, outTimestamp, err := replicaCalc.GetResourceReplicas(tCtx, tc.currentReplicas, tc.resource.targetUtilization, tc.resource.name, tolerances, testNamespace, selector, tc.container)

		if tc.expectedError != nil {
			require.Error(t, err, "there should be an error calculating the replica count")
			assert.ErrorContains(t, err, tc.expectedError.Error(), "the error message should have contained the expected error message")
			return
		}
		require.NoError(t, err, "there should not have been an error calculating the replica count")
		assert.Equal(t, tc.expectedReplicas, outReplicas, "replicas should be as expected")
		assert.Equal(t, tc.resource.expectedUtilization, outUtilization, "utilization should be as expected")
		assert.Equal(t, tc.resource.expectedValue, outRawValue, "raw value should be as expected")
		assert.True(t, tc.timestamp.Equal(outTimestamp), "timestamp should be as expected")
		return
	}

	var outReplicas int32
	var outUsage int64
	var outTimestamp time.Time
	switch tc.metric.metricType {
	case objectMetric:
		if tc.metric.singleObject == nil {
			t.Fatal("Metric specified as objectMetric but metric.singleObject is nil.")
		}
		outReplicas, outUsage, outTimestamp, err = replicaCalc.GetObjectMetricReplicas(tc.currentReplicas, tc.metric.targetUsage, tc.metric.name, tolerances, testNamespace, tc.metric.singleObject, selector, nil)
	case objectPerPodMetric:
		if tc.metric.singleObject == nil {
			t.Fatal("Metric specified as objectMetric but metric.singleObject is nil.")
		}
		outReplicas, outUsage, outTimestamp, err = replicaCalc.GetObjectPerPodMetricReplicas(tc.currentReplicas, tc.metric.perPodTargetUsage, tc.metric.name, tolerances, testNamespace, tc.metric.singleObject, nil)
	case externalMetric:
		if tc.metric.selector == nil {
			t.Fatal("Metric specified as externalMetric but metric.selector is nil.")
		}
		if tc.metric.targetUsage <= 0 {
			t.Fatalf("Metric specified as externalMetric but metric.targetUsage is %d which is <=0.", tc.metric.targetUsage)
		}
		outReplicas, outUsage, outTimestamp, err = replicaCalc.GetExternalMetricReplicas(tc.currentReplicas, tc.metric.targetUsage, tc.metric.name, tolerances, testNamespace, tc.metric.selector, selector)
	case externalPerPodMetric:
		if tc.metric.selector == nil {
			t.Fatal("Metric specified as externalPerPodMetric but metric.selector is nil.")
		}
		if tc.metric.perPodTargetUsage <= 0 {
			t.Fatalf("Metric specified as externalPerPodMetric but metric.perPodTargetUsage is %d which is <=0.", tc.metric.perPodTargetUsage)
		}

		outReplicas, outUsage, outTimestamp, err = replicaCalc.GetExternalPerPodMetricReplicas(tc.currentReplicas, tc.metric.perPodTargetUsage, tc.metric.name, tolerances, testNamespace, tc.metric.selector)
	case podMetric:
		outReplicas, outUsage, outTimestamp, err = replicaCalc.GetMetricReplicas(tc.currentReplicas, tc.metric.targetUsage, tc.metric.name, tolerances, testNamespace, selector, nil)
	default:
		t.Fatalf("Unknown metric type: %d", tc.metric.metricType)
	}

	if tc.expectedError != nil {
		require.Error(t, err, "there should be an error calculating the replica count")
		assert.ErrorContains(t, err, tc.expectedError.Error(), "the error message should have contained the expected error message")
		return
	}
	require.NoError(t, err, "there should not have been an error calculating the replica count")
	assert.Equal(t, tc.expectedReplicas, outReplicas, "replicas should be as expected")
	assert.Equal(t, tc.metric.expectedUsage, outUsage, "usage should be as expected")
	assert.True(t, tc.timestamp.Equal(outTimestamp), "timestamp should be as expected")
}
func TestReplicaCalcDisjointResourcesMetrics(t *testing.T) {
	tc := replicaCalcTestCase{
		currentReplicas: 1,
		expectedError:   fmt.Errorf("no metrics returned matched known pods"),
		resource: &resourceInfo{
			name:     v1.ResourceCPU,
			requests: []resource.Quantity{resource.MustParse("1.0")},
			levels:   makePodMetricLevels(100),
			podNames: []string{"an-older-pod-name"},

			targetUtilization: 100,
		},
	}
	tc.runTest(t)
}

func TestReplicaCalcScaleUpOverflow(t *testing.T) {
	tc := replicaCalcTestCase{
		currentReplicas:  3,
		expectedReplicas: math.MaxInt32,
		metric: &metricInfo{
			name:          "qps",
			levels:        []int64{math.MaxInt64}, // Use MaxInt64 to ensure a very large value
			targetUsage:   1,                      // Set a very low target to force high scaling
			metricType:    objectMetric,
			expectedUsage: math.MaxInt64,
			singleObject: &autoscalingv2.CrossVersionObjectReference{
				Kind:       "Deployment",
				APIVersion: "apps/v1",
				Name:       "some-deployment",
			},
		},
	}
	tc.runTest(t)
}

func TestExternalPerPodMetricReplicaOverflow(t *testing.T) {
	tc := replicaCalcTestCase{
		currentReplicas:  1,
		expectedReplicas: math.MaxInt32,
		metric: &metricInfo{
			name:              "qps",
			levels:            []int64{math.MaxInt64},
			perPodTargetUsage: 1, // Set to 1 to test replica calculation when targeting a 1:1 ratio between metric value and number of pods
			metricType:        externalPerPodMetric,
			expectedUsage:     math.MaxInt64,
			selector:          &metav1.LabelSelector{MatchLabels: map[string]string{"label": "value"}},
		},
	}
	tc.runTest(t)
}

func TestExternalPerPodMetricUsageOverflow(t *testing.T) {
	tc := replicaCalcTestCase{
		currentReplicas:  1,
		expectedReplicas: 1,
		metric: &metricInfo{
			name:              "qps",
			levels:            []int64{math.MaxInt64},
			perPodTargetUsage: math.MaxInt64, // Set to a high value to test replica calculation when targeting a high value:1 ratio between metric value and number of pods
			metricType:        externalPerPodMetric,
			expectedUsage:     math.MaxInt64,
			selector:          &metav1.LabelSelector{MatchLabels: map[string]string{"label": "value"}},
		},
	}
	tc.runTest(t)
}

func TestReplicaCalcScaleUpCM(t *testing.T) {
	tc := replicaCalcTestCase{
		currentReplicas:  3,
		expectedReplicas: 4,
		metric: &metricInfo{
			name:          "qps",
			levels:        []int64{20000, 10000, 30000},
			targetUsage:   15000,
			expectedUsage: 20000,
			metricType:    podMetric,
		},
	}
	tc.runTest(t)
}

func TestReplicaCalcScaleUpCMUnreadyHotCpuNoLessScale(t *testing.T) {
	tc := replicaCalcTestCase{
		currentReplicas:  3,
		expectedReplicas: 6,
		podReadiness:     []v1.ConditionStatus{v1.ConditionTrue, v1.ConditionTrue, v1.ConditionFalse},
		podStartTime:     []metav1.Time{coolCPUCreationTime(), coolCPUCreationTime(), hotCPUCreationTime()},
		metric: &metricInfo{
			name:          "qps",
			levels:        []int64{50000, 10000, 30000},
			targetUsage:   15000,
			expectedUsage: 30000,
			metricType:    podMetric,
		},
	}
	tc.runTest(t)
}

func TestReplicaCalcScaleUpCMUnreadyHotCpuScaleWouldScaleDown(t *testing.T) {
	tc := replicaCalcTestCase{
		currentReplicas:  3,
		expectedReplicas: 7,
		podReadiness:     []v1.ConditionStatus{v1.ConditionFalse, v1.ConditionTrue, v1.ConditionFalse},
		podStartTime:     []metav1.Time{hotCPUCreationTime(), coolCPUCreationTime(), hotCPUCreationTime()},
		metric: &metricInfo{
			name:          "qps",
			levels:        []int64{50000, 15000, 30000},
			targetUsage:   15000,
			expectedUsage: 31666,
			metricType:    podMetric,
		},
	}
	tc.runTest(t)
}

func TestReplicaCalcScaleUpCMObject(t *testing.T) {
	tc := replicaCalcTestCase{
		currentReplicas:  3,
		expectedReplicas: 4,
		metric: &metricInfo{
			name:          "qps",
			levels:        []int64{20000},
			targetUsage:   15000,
			expectedUsage: 20000,
			singleObject: &autoscalingv2.CrossVersionObjectReference{
				Kind:       "Deployment",
				APIVersion: "apps/v1",
				Name:       "some-deployment",
			},
		},
	}
	tc.runTest(t)
}

func TestReplicaCalcScaleUpCMPerPodObject(t *testing.T) {
	tc := replicaCalcTestCase{
		currentReplicas:  3,
		expectedReplicas: 4,
		metric: &metricInfo{
			metricType:        objectPerPodMetric,
			name:              "qps",
			levels:            []int64{20000},
			perPodTargetUsage: 5000,
			expectedUsage:     6667,
			singleObject: &autoscalingv2.CrossVersionObjectReference{
				Kind:       "Deployment",
				APIVersion: "apps/v1",
				Name:       "some-deployment",
			},
		},
	}
	tc.runTest(t)
}

func TestReplicaCalcScaleUpCMObjectIgnoresUnreadyPods(t *testing.T) {
	tc := replicaCalcTestCase{
		currentReplicas:  3,
		expectedReplicas: 5, // If we did not ignore unready pods, we'd expect 15 replicas.
		podReadiness:     []v1.ConditionStatus{v1.ConditionFalse, v1.ConditionTrue, v1.ConditionFalse},
		metric: &metricInfo{
			name:          "qps",
			levels:        []int64{50000},
			targetUsage:   10000,
			expectedUsage: 50000,
			singleObject: &autoscalingv2.CrossVersionObjectReference{
				Kind:       "Deployment",
				APIVersion: "apps/v1",
				Name:       "some-deployment",
			},
		},
	}
	tc.runTest(t)
}

func TestReplicaCalcScaleUpCMExternalIgnoresUnreadyPods(t *testing.T) {
	tc := replicaCalcTestCase{
		currentReplicas:  3,
		expectedReplicas: 2, // Would expect 6 if we didn't ignore unready pods
		podReadiness:     []v1.ConditionStatus{v1.ConditionFalse, v1.ConditionTrue, v1.ConditionFalse},
		metric: &metricInfo{
			name:          "qps",
			levels:        []int64{8600},
			targetUsage:   4400,
			expectedUsage: 8600,
			selector:      &metav1.LabelSelector{MatchLabels: map[string]string{"label": "value"}},
			metricType:    externalMetric,
		},
	}
	tc.runTest(t)
}

func TestReplicaCalcScaleUpPerPodCMExternal(t *testing.T) {
	tc := replicaCalcTestCase{
		currentReplicas:  3,
		expectedReplicas: 4,
		metric: &metricInfo{
			name:              "qps",
			levels:            []int64{8600},
			perPodTargetUsage: 2150,
			expectedUsage:     2867,
			selector:          &metav1.LabelSelector{MatchLabels: map[string]string{"label": "value"}},
			metricType:        externalPerPodMetric,
		},
	}
	tc.runTest(t)
}

func TestReplicaCalcScaleDownCM(t *testing.T) {
	tc := replicaCalcTestCase{
		currentReplicas:  5,
		expectedReplicas: 3,
		metric: &metricInfo{
			name:          "qps",
			levels:        []int64{12000, 12000, 12000, 12000, 12000},
			targetUsage:   20000,
			expectedUsage: 12000,
			metricType:    podMetric,
		},
	}
	tc.runTest(t)
}

func TestReplicaCalcScaleDownPerPodCMObject(t *testing.T) {
	tc := replicaCalcTestCase{
		currentReplicas:  5,
		expectedReplicas: 3,
		metric: &metricInfo{
			name:              "qps",
			levels:            []int64{6000},
			perPodTargetUsage: 2000,
			expectedUsage:     1200,
			singleObject: &autoscalingv2.CrossVersionObjectReference{
				Kind:       "Deployment",
				APIVersion: "apps/v1",
				Name:       "some-deployment",
			},
			metricType: objectPerPodMetric,
		},
	}
	tc.runTest(t)
}

func TestReplicaCalcScaleDownCMObject(t *testing.T) {
	tc := replicaCalcTestCase{
		currentReplicas:  5,
		expectedReplicas: 3,
		metric: &metricInfo{
			name:          "qps",
			levels:        []int64{12000},
			targetUsage:   20000,
			expectedUsage: 12000,
			singleObject: &autoscalingv2.CrossVersionObjectReference{
				Kind:       "Deployment",
				APIVersion: "apps/v1",
				Name:       "some-deployment",
			},
		},
	}
	tc.runTest(t)
}

func TestReplicaCalcScaleDownCMExternal(t *testing.T) {
	tc := replicaCalcTestCase{
		currentReplicas:  5,
		expectedReplicas: 3,
		metric: &metricInfo{
			name:          "qps",
			levels:        []int64{8600},
			targetUsage:   14334,
			expectedUsage: 8600,
			selector:      &metav1.LabelSelector{MatchLabels: map[string]string{"label": "value"}},
			metricType:    externalMetric,
		},
	}
	tc.runTest(t)
}

func TestReplicaCalcScaleDownPerPodCMExternal(t *testing.T) {
	tc := replicaCalcTestCase{
		currentReplicas:  5,
		expectedReplicas: 3,
		metric: &metricInfo{
			name:              "qps",
			levels:            []int64{8600},
			perPodTargetUsage: 2867,
			expectedUsage:     1720,
			selector:          &metav1.LabelSelector{MatchLabels: map[string]string{"label": "value"}},
			metricType:        externalPerPodMetric,
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
			levels:   makePodMetricLevels(1010, 1030, 1020),

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
			name:          "qps",
			levels:        []int64{20000, 21000, 21000},
			targetUsage:   20000,
			expectedUsage: 20666,
			metricType:    podMetric,
		},
	}
	tc.runTest(t)
}

func TestReplicaCalcToleranceCMObject(t *testing.T) {
	tc := replicaCalcTestCase{
		currentReplicas:  3,
		expectedReplicas: 3,
		metric: &metricInfo{
			name:          "qps",
			levels:        []int64{20666},
			targetUsage:   20000,
			expectedUsage: 20666,
			singleObject: &autoscalingv2.CrossVersionObjectReference{
				Kind:       "Deployment",
				APIVersion: "apps/v1",
				Name:       "some-deployment",
			},
		},
	}
	tc.runTest(t)
}

func TestReplicaCalcTolerancePerPodCMObject(t *testing.T) {
	tc := replicaCalcTestCase{
		currentReplicas:  4,
		expectedReplicas: 4,
		metric: &metricInfo{
			metricType:        objectPerPodMetric,
			name:              "qps",
			levels:            []int64{20166},
			perPodTargetUsage: 5000,
			expectedUsage:     5042,
			singleObject: &autoscalingv2.CrossVersionObjectReference{
				Kind:       "Deployment",
				APIVersion: "apps/v1",
				Name:       "some-deployment",
			},
		},
	}
	tc.runTest(t)
}

func TestReplicaCalcToleranceCMExternal(t *testing.T) {
	tc := replicaCalcTestCase{
		currentReplicas:  3,
		expectedReplicas: 3,
		metric: &metricInfo{
			name:          "qps",
			levels:        []int64{8600},
			targetUsage:   8888,
			expectedUsage: 8600,
			selector:      &metav1.LabelSelector{MatchLabels: map[string]string{"label": "value"}},
			metricType:    externalMetric,
		},
	}
	tc.runTest(t)
}

func TestReplicaCalcTolerancePerPodCMExternal(t *testing.T) {
	tc := replicaCalcTestCase{
		currentReplicas:  3,
		expectedReplicas: 3,
		metric: &metricInfo{
			name:              "qps",
			levels:            []int64{8600},
			perPodTargetUsage: 2900,
			expectedUsage:     2867,
			selector:          &metav1.LabelSelector{MatchLabels: map[string]string{"label": "value"}},
			metricType:        externalPerPodMetric,
		},
	}
	tc.runTest(t)
}

func TestReplicaCalcConfigurableTolerance(t *testing.T) {
	testCases := []struct {
		name string
		replicaCalcTestCase
	}{
		{
			name: "Outside of a 0% tolerance",
			replicaCalcTestCase: replicaCalcTestCase{
				tolerances:       &Tolerances{0., 0.},
				currentReplicas:  3,
				expectedReplicas: 4,
				resource: &resourceInfo{
					name:                v1.ResourceCPU,
					requests:            []resource.Quantity{resource.MustParse("0.9"), resource.MustParse("1.0"), resource.MustParse("1.1")},
					levels:              makePodMetricLevels(909, 1010, 1111),
					targetUtilization:   100,
					expectedUtilization: 101,
					expectedValue:       numContainersPerPod * 1010,
				},
			},
		},
		{
			name: "Within a 200% scale-up tolerance",
			replicaCalcTestCase: replicaCalcTestCase{
				tolerances:       &Tolerances{defaultTestingTolerance, 2.},
				currentReplicas:  3,
				expectedReplicas: 3,
				resource: &resourceInfo{
					name:                v1.ResourceCPU,
					requests:            []resource.Quantity{resource.MustParse("0.9"), resource.MustParse("1.0"), resource.MustParse("1.1")},
					levels:              makePodMetricLevels(1890, 1910, 1900),
					targetUtilization:   100,
					expectedUtilization: 190,
					expectedValue:       numContainersPerPod * 1900,
				},
			},
		},
		{
			name: "Outside 8% scale-up tolerance (and superfuous scale-down tolerance)",
			replicaCalcTestCase: replicaCalcTestCase{
				tolerances:       &Tolerances{2., .08},
				currentReplicas:  3,
				expectedReplicas: 4,
				resource: &resourceInfo{
					name:                v1.ResourceCPU,
					requests:            []resource.Quantity{resource.MustParse("0.9"), resource.MustParse("1.0"), resource.MustParse("1.1")},
					levels:              makePodMetricLevels(1100, 1080, 1090),
					targetUtilization:   100,
					expectedUtilization: 109,
					expectedValue:       numContainersPerPod * 1090,
				},
			},
		},
		{
			name: "Within a 36% scale-down tolerance",
			replicaCalcTestCase: replicaCalcTestCase{
				tolerances:       &Tolerances{.36, defaultTestingTolerance},
				currentReplicas:  3,
				expectedReplicas: 3,
				resource: &resourceInfo{
					name:                v1.ResourceCPU,
					requests:            []resource.Quantity{resource.MustParse("0.9"), resource.MustParse("1.0"), resource.MustParse("1.1")},
					levels:              makePodMetricLevels(660, 640, 650),
					targetUtilization:   100,
					expectedUtilization: 65,
					expectedValue:       numContainersPerPod * 650,
				},
			},
		},
		{
			name: "Outside a 34% scale-down tolerance",
			replicaCalcTestCase: replicaCalcTestCase{
				tolerances:       &Tolerances{.34, defaultTestingTolerance},
				currentReplicas:  3,
				expectedReplicas: 2,
				resource: &resourceInfo{
					name:                v1.ResourceCPU,
					requests:            []resource.Quantity{resource.MustParse("0.9"), resource.MustParse("1.0"), resource.MustParse("1.1")},
					levels:              makePodMetricLevels(660, 640, 650),
					targetUtilization:   100,
					expectedUtilization: 65,
					expectedValue:       numContainersPerPod * 650,
				},
			},
		},
	}
	for _, tc := range testCases {
		t.Run(tc.name, tc.runTest)
	}
}

func TestReplicaCalcConfigurableToleranceCM(t *testing.T) {
	tc := replicaCalcTestCase{
		tolerances:       &Tolerances{defaultTestingTolerance, .01},
		currentReplicas:  3,
		expectedReplicas: 4,
		metric: &metricInfo{
			name:          "qps",
			levels:        []int64{20000, 21000, 21000},
			targetUsage:   20000,
			expectedUsage: 20666,
			metricType:    podMetric,
		},
	}
	tc.runTest(t)
}

func TestReplicaCalcConfigurableToleranceCMObject(t *testing.T) {
	tc := replicaCalcTestCase{
		tolerances:       &Tolerances{defaultTestingTolerance, .01},
		currentReplicas:  3,
		expectedReplicas: 4,
		metric: &metricInfo{
			name:          "qps",
			levels:        []int64{20666},
			targetUsage:   20000,
			expectedUsage: 20666,
			singleObject: &autoscalingv2.CrossVersionObjectReference{
				Kind:       "Deployment",
				APIVersion: "apps/v1",
				Name:       "some-deployment",
			},
		},
	}
	tc.runTest(t)
}

func TestReplicaCalcConfigurableTolerancePerPodCMObject(t *testing.T) {
	tc := replicaCalcTestCase{
		tolerances:       &Tolerances{defaultTestingTolerance, .01},
		currentReplicas:  4,
		expectedReplicas: 5,
		metric: &metricInfo{
			metricType:        objectPerPodMetric,
			name:              "qps",
			levels:            []int64{20208},
			perPodTargetUsage: 5000,
			expectedUsage:     5052,
			singleObject: &autoscalingv2.CrossVersionObjectReference{
				Kind:       "Deployment",
				APIVersion: "apps/v1",
				Name:       "some-deployment",
			},
		},
	}
	tc.runTest(t)
}

func TestReplicaCalcConfigurableToleranceCMExternal(t *testing.T) {
	tc := replicaCalcTestCase{
		tolerances:       &Tolerances{defaultTestingTolerance, .01},
		currentReplicas:  3,
		expectedReplicas: 4,
		metric: &metricInfo{
			name:          "qps",
			levels:        []int64{8900},
			targetUsage:   8800,
			expectedUsage: 8900,
			selector:      &metav1.LabelSelector{MatchLabels: map[string]string{"label": "value"}},
			metricType:    externalMetric,
		},
	}
	tc.runTest(t)
}

func TestReplicaCalcConfigurableTolerancePerPodCMExternal(t *testing.T) {
	tc := replicaCalcTestCase{
		tolerances:       &Tolerances{defaultTestingTolerance, .01},
		currentReplicas:  3,
		expectedReplicas: 4,
		metric: &metricInfo{
			name:              "qps",
			levels:            []int64{8600},
			perPodTargetUsage: 2800,
			expectedUsage:     2867,
			selector:          &metav1.LabelSelector{MatchLabels: map[string]string{"label": "value"}},
			metricType:        externalPerPodMetric,
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
			levels:              makePodMetricLevels(4000, 9500, 3000, 7000, 3200, 2000),
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
			levels:   makePodMetricLevels(400, 95),

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
		expectedError:   fmt.Errorf("unable to get metrics for resource cpu: no metrics returned from resource metrics API"),
		resource: &resourceInfo{
			name:     v1.ResourceCPU,
			requests: []resource.Quantity{resource.MustParse("1.0"), resource.MustParse("1.0"), resource.MustParse("1.0")},
			levels:   makePodMetricLevels(),

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
			levels:   makePodMetricLevels(200),

			targetUtilization: 100,
		},
	}
	tc.runTest(t)
}

func TestPlainMetricReplicaCalcMissingMetricsNoChangeEq(t *testing.T) {
	tc := replicaCalcTestCase{
		currentReplicas:  5,
		expectedReplicas: 5,
		metric: &metricInfo{
			name:          "qps",
			levels:        []int64{20000, 19000, 21000},
			targetUsage:   20000,
			expectedUsage: 20000,
			metricType:    podMetric,
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
			levels:   makePodMetricLevels(1000),

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
			levels:   makePodMetricLevels(1900),

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
			levels:   makePodMetricLevels(600),

			targetUtilization:   100,
			expectedUtilization: 60,
			expectedValue:       numContainersPerPod * 600,
		},
	}
	tc.runTest(t)
}

func TestReplicaCalcMissingMetricsUnreadyChange(t *testing.T) {
	tc := replicaCalcTestCase{
		currentReplicas:  3,
		expectedReplicas: 3,
		podReadiness:     []v1.ConditionStatus{v1.ConditionFalse, v1.ConditionTrue, v1.ConditionTrue},
		resource: &resourceInfo{
			name:     v1.ResourceCPU,
			requests: []resource.Quantity{resource.MustParse("1.0"), resource.MustParse("1.0"), resource.MustParse("1.0")},
			levels:   makePodMetricLevels(100, 450),

			targetUtilization:   50,
			expectedUtilization: 45,
			expectedValue:       numContainersPerPod * 450,
		},
	}
	tc.runTest(t)
}

func TestReplicaCalcMissingMetricsHotCpuNoChange(t *testing.T) {
	tc := replicaCalcTestCase{
		currentReplicas:  3,
		expectedReplicas: 3,
		podStartTime:     []metav1.Time{hotCPUCreationTime(), coolCPUCreationTime(), coolCPUCreationTime()},
		resource: &resourceInfo{
			name:     v1.ResourceCPU,
			requests: []resource.Quantity{resource.MustParse("1.0"), resource.MustParse("1.0"), resource.MustParse("1.0")},
			levels:   makePodMetricLevels(100, 450),

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
			levels:   makePodMetricLevels(100, 2000),

			targetUtilization:   50,
			expectedUtilization: 200,
			expectedValue:       numContainersPerPod * 2000,
		},
	}
	tc.runTest(t)
}

func TestReplicaCalcMissingMetricsHotCpuScaleUp(t *testing.T) {
	tc := replicaCalcTestCase{
		currentReplicas:  3,
		expectedReplicas: 4,
		podReadiness:     []v1.ConditionStatus{v1.ConditionFalse, v1.ConditionTrue, v1.ConditionTrue},
		podStartTime:     []metav1.Time{hotCPUCreationTime(), coolCPUCreationTime(), coolCPUCreationTime()},
		resource: &resourceInfo{
			name:     v1.ResourceCPU,
			requests: []resource.Quantity{resource.MustParse("1.0"), resource.MustParse("1.0"), resource.MustParse("1.0")},
			levels:   makePodMetricLevels(100, 2000),

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
			levels:   makePodMetricLevels(100, 100, 100),

			targetUtilization:   50,
			expectedUtilization: 10,
			expectedValue:       numContainersPerPod * 100,
		},
	}
	tc.runTest(t)
}

func TestReplicaCalcMissingMetricsScaleDownTargetOver100(t *testing.T) {
	tc := replicaCalcTestCase{
		currentReplicas:  4,
		expectedReplicas: 2,
		podReadiness:     []v1.ConditionStatus{v1.ConditionFalse, v1.ConditionTrue, v1.ConditionTrue, v1.ConditionTrue},
		resource: &resourceInfo{
			name:     v1.ResourceCPU,
			requests: []resource.Quantity{resource.MustParse("1.0"), resource.MustParse("1.0"), resource.MustParse("2.0"), resource.MustParse("2.0")},
			levels:   makePodMetricLevels(200, 100, 100),

			targetUtilization:   300,
			expectedUtilization: 6,
			expectedValue:       numContainersPerPod * 100,
		},
	}
	tc.runTest(t)
}

func TestReplicaCalcDuringRollingUpdateWithMaxSurge(t *testing.T) {
	tc := replicaCalcTestCase{
		currentReplicas:  2,
		expectedReplicas: 2,
		podPhase:         []v1.PodPhase{v1.PodRunning, v1.PodRunning, v1.PodRunning},
		resource: &resourceInfo{
			name:     v1.ResourceCPU,
			requests: []resource.Quantity{resource.MustParse("1.0"), resource.MustParse("1.0"), resource.MustParse("1.0")},
			levels:   makePodMetricLevels(100, 100),

			targetUtilization:   50,
			expectedUtilization: 10,
			expectedValue:       numContainersPerPod * 100,
		},
	}
	tc.runTest(t)
}

func TestReplicaCalcDuringRollingUpdateWithMaxSurgeCM(t *testing.T) {
	tc := replicaCalcTestCase{
		currentReplicas:  2,
		expectedReplicas: 2,
		podPhase:         []v1.PodPhase{v1.PodRunning, v1.PodRunning, v1.PodRunning},
		metric: &metricInfo{
			name:          "qps",
			levels:        []int64{10000, 10000},
			targetUsage:   17000,
			expectedUsage: 10000,
			metricType:    podMetric,
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
	target := math.Abs(1/(requestedToUsed*(1-defaultTestingTolerance))) + .01
	finalCPUPercentTarget := int32(target * 100)
	resourcesUsedRatio := float64(totalUsedCPUOfAllPods) / float64(float64(totalRequestedCPUOfAllPods)*target)

	// i.e. .60 * 20 -> scaled down expectation.
	finalPods := int32(math.Ceil(resourcesUsedRatio * float64(startPods)))

	// To breach tolerance we will create a utilization ratio difference of tolerance to usageRatioToleranceValue)
	tc := replicaCalcTestCase{
		currentReplicas:  startPods,
		expectedReplicas: finalPods,
		resource: &resourceInfo{
			name: v1.ResourceCPU,
			levels: makePodMetricLevels(
				totalUsedCPUOfAllPods/10,
				totalUsedCPUOfAllPods/10,
				totalUsedCPUOfAllPods/10,
				totalUsedCPUOfAllPods/10,
				totalUsedCPUOfAllPods/10,
				totalUsedCPUOfAllPods/10,
				totalUsedCPUOfAllPods/10,
				totalUsedCPUOfAllPods/10,
				totalUsedCPUOfAllPods/10,
				totalUsedCPUOfAllPods/10,
			),
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

			targetUtilization:   finalCPUPercentTarget,
			expectedUtilization: int32(totalUsedCPUOfAllPods*100) / totalRequestedCPUOfAllPods,
			expectedValue:       numContainersPerPod * totalUsedCPUOfAllPods / 10,
		},
	}

	tc.runTest(t)

	// Reuse the data structure above, now testing "unscaling".
	// Now, we test that no scaling happens if we are in a very close margin to the tolerance
	target = math.Abs(1/(requestedToUsed*(1-defaultTestingTolerance))) + .004
	finalCPUPercentTarget = int32(target * 100)
	tc.resource.targetUtilization = finalCPUPercentTarget
	tc.currentReplicas = startPods
	tc.expectedReplicas = startPods
	tc.runTest(t)
}

func TestGroupPods(t *testing.T) {
	tests := []struct {
		name                string
		pods                []*v1.Pod
		metrics             metricsclient.PodMetricsInfo
		resource            v1.ResourceName
		expectReadyPodCount int
		expectUnreadyPods   sets.Set[string]
		expectMissingPods   sets.Set[string]
		expectIgnoredPods   sets.Set[string]
	}{
		{
			name:                "void",
			pods:                []*v1.Pod{},
			metrics:             metricsclient.PodMetricsInfo{},
			resource:            v1.ResourceCPU,
			expectReadyPodCount: 0,
			expectUnreadyPods:   sets.New[string](),
			expectMissingPods:   sets.New[string](),
			expectIgnoredPods:   sets.New[string](),
		}, {
			name: "count in a ready pod - memory",
			pods: []*v1.Pod{
				{
					ObjectMeta: metav1.ObjectMeta{
						Name: "bentham",
					},
					Status: v1.PodStatus{
						Phase: v1.PodSucceeded,
					},
				},
			},
			metrics: metricsclient.PodMetricsInfo{
				"bentham": metricsclient.PodMetric{Value: 1, Timestamp: time.Now(), Window: time.Minute},
			},
			resource:            v1.ResourceMemory,
			expectReadyPodCount: 1,
			expectUnreadyPods:   sets.New[string](),
			expectMissingPods:   sets.New[string](),
			expectIgnoredPods:   sets.New[string](),
		}, {
			name: "unready a pod without ready condition - CPU",
			pods: []*v1.Pod{
				{
					ObjectMeta: metav1.ObjectMeta{
						Name: "lucretius",
					},
					Status: v1.PodStatus{
						Phase: v1.PodSucceeded,
						StartTime: &metav1.Time{
							Time: time.Now(),
						},
					},
				},
			},
			metrics: metricsclient.PodMetricsInfo{
				"lucretius": metricsclient.PodMetric{Value: 1},
			},
			resource:            v1.ResourceCPU,
			expectReadyPodCount: 0,
			expectUnreadyPods:   sets.New[string]("lucretius"),
			expectMissingPods:   sets.New[string](),
			expectIgnoredPods:   sets.New[string](),
		}, {
			name: "count in a ready pod with fresh metrics during initialization period - CPU",
			pods: []*v1.Pod{
				{
					ObjectMeta: metav1.ObjectMeta{
						Name: "bentham",
					},
					Status: v1.PodStatus{
						Phase: v1.PodSucceeded,
						StartTime: &metav1.Time{
							Time: time.Now().Add(-1 * time.Minute),
						},
						Conditions: []v1.PodCondition{
							{
								Type:               v1.PodReady,
								LastTransitionTime: metav1.Time{Time: time.Now().Add(-30 * time.Second)},
								Status:             v1.ConditionTrue,
							},
						},
					},
				},
			},
			metrics: metricsclient.PodMetricsInfo{
				"bentham": metricsclient.PodMetric{Value: 1, Timestamp: time.Now(), Window: 30 * time.Second},
			},
			resource:            v1.ResourceCPU,
			expectReadyPodCount: 1,
			expectUnreadyPods:   sets.New[string](),
			expectMissingPods:   sets.New[string](),
			expectIgnoredPods:   sets.New[string](),
		}, {
			name: "unready a ready pod without fresh metrics during initialization period - CPU",
			pods: []*v1.Pod{
				{
					ObjectMeta: metav1.ObjectMeta{
						Name: "bentham",
					},
					Status: v1.PodStatus{
						Phase: v1.PodSucceeded,
						StartTime: &metav1.Time{
							Time: time.Now().Add(-1 * time.Minute),
						},
						Conditions: []v1.PodCondition{
							{
								Type:               v1.PodReady,
								LastTransitionTime: metav1.Time{Time: time.Now().Add(-30 * time.Second)},
								Status:             v1.ConditionTrue,
							},
						},
					},
				},
			},
			metrics: metricsclient.PodMetricsInfo{
				"bentham": metricsclient.PodMetric{Value: 1, Timestamp: time.Now(), Window: 60 * time.Second},
			},
			resource:            v1.ResourceCPU,
			expectReadyPodCount: 0,
			expectUnreadyPods:   sets.New[string]("bentham"),
			expectMissingPods:   sets.New[string](),
			expectIgnoredPods:   sets.New[string](),
		}, {
			name: "unready an unready pod during initialization period - CPU",
			pods: []*v1.Pod{
				{
					ObjectMeta: metav1.ObjectMeta{
						Name: "lucretius",
					},
					Status: v1.PodStatus{
						Phase: v1.PodSucceeded,
						StartTime: &metav1.Time{
							Time: time.Now().Add(-10 * time.Minute),
						},
						Conditions: []v1.PodCondition{
							{
								Type:               v1.PodReady,
								LastTransitionTime: metav1.Time{Time: time.Now().Add(-9*time.Minute - 54*time.Second)},
								Status:             v1.ConditionFalse,
							},
						},
					},
				},
			},
			metrics: metricsclient.PodMetricsInfo{
				"lucretius": metricsclient.PodMetric{Value: 1},
			},
			resource:            v1.ResourceCPU,
			expectReadyPodCount: 0,
			expectUnreadyPods:   sets.New[string]("lucretius"),
			expectMissingPods:   sets.New[string](),
			expectIgnoredPods:   sets.New[string](),
		}, {
			name: "count in a ready pod without fresh metrics after initialization period - CPU",
			pods: []*v1.Pod{
				{
					ObjectMeta: metav1.ObjectMeta{
						Name: "bentham",
					},
					Status: v1.PodStatus{
						Phase: v1.PodSucceeded,
						StartTime: &metav1.Time{
							Time: time.Now().Add(-3 * time.Minute),
						},
						Conditions: []v1.PodCondition{
							{
								Type:               v1.PodReady,
								LastTransitionTime: metav1.Time{Time: time.Now().Add(-3 * time.Minute)},
								Status:             v1.ConditionTrue,
							},
						},
					},
				},
			},
			metrics: metricsclient.PodMetricsInfo{
				"bentham": metricsclient.PodMetric{Value: 1, Timestamp: time.Now().Add(-2 * time.Minute), Window: time.Minute},
			},
			resource:            v1.ResourceCPU,
			expectReadyPodCount: 1,
			expectUnreadyPods:   sets.New[string](),
			expectMissingPods:   sets.New[string](),
			expectIgnoredPods:   sets.New[string](),
		}, {
			name: "count in an unready pod that was ready after initialization period - CPU",
			pods: []*v1.Pod{
				{
					ObjectMeta: metav1.ObjectMeta{
						Name: "lucretius",
					},
					Status: v1.PodStatus{
						Phase: v1.PodSucceeded,
						StartTime: &metav1.Time{
							Time: time.Now().Add(-10 * time.Minute),
						},
						Conditions: []v1.PodCondition{
							{
								Type:               v1.PodReady,
								LastTransitionTime: metav1.Time{Time: time.Now().Add(-9 * time.Minute)},
								Status:             v1.ConditionFalse,
							},
						},
					},
				},
			},
			metrics: metricsclient.PodMetricsInfo{
				"lucretius": metricsclient.PodMetric{Value: 1},
			},
			resource:            v1.ResourceCPU,
			expectReadyPodCount: 1,
			expectUnreadyPods:   sets.New[string](),
			expectMissingPods:   sets.New[string](),
			expectIgnoredPods:   sets.New[string](),
		}, {
			name: "unready pod that has never been ready after initialization period - CPU",
			pods: []*v1.Pod{
				{
					ObjectMeta: metav1.ObjectMeta{
						Name: "lucretius",
					},
					Status: v1.PodStatus{
						Phase: v1.PodSucceeded,
						StartTime: &metav1.Time{
							Time: time.Now().Add(-10 * time.Minute),
						},
						Conditions: []v1.PodCondition{
							{
								Type:               v1.PodReady,
								LastTransitionTime: metav1.Time{Time: time.Now().Add(-9*time.Minute - 50*time.Second)},
								Status:             v1.ConditionFalse,
							},
						},
					},
				},
			},
			metrics: metricsclient.PodMetricsInfo{
				"lucretius": metricsclient.PodMetric{Value: 1},
			},
			resource:            v1.ResourceCPU,
			expectReadyPodCount: 1,
			expectUnreadyPods:   sets.New[string](),
			expectMissingPods:   sets.New[string](),
			expectIgnoredPods:   sets.New[string](),
		}, {
			name: "a missing pod",
			pods: []*v1.Pod{
				{
					ObjectMeta: metav1.ObjectMeta{
						Name: "epicurus",
					},
					Status: v1.PodStatus{
						Phase: v1.PodSucceeded,
						StartTime: &metav1.Time{
							Time: time.Now().Add(-3 * time.Minute),
						},
					},
				},
			},
			metrics:             metricsclient.PodMetricsInfo{},
			resource:            v1.ResourceCPU,
			expectReadyPodCount: 0,
			expectUnreadyPods:   sets.New[string](),
			expectMissingPods:   sets.New[string]("epicurus"),
			expectIgnoredPods:   sets.New[string](),
		}, {
			name: "several pods",
			pods: []*v1.Pod{
				{
					ObjectMeta: metav1.ObjectMeta{
						Name: "lucretius",
					},
					Status: v1.PodStatus{
						Phase: v1.PodSucceeded,
						StartTime: &metav1.Time{
							Time: time.Now(),
						},
					},
				},
				{
					ObjectMeta: metav1.ObjectMeta{
						Name: "niccolo",
					},
					Status: v1.PodStatus{
						Phase: v1.PodSucceeded,
						StartTime: &metav1.Time{
							Time: time.Now().Add(-3 * time.Minute),
						},
						Conditions: []v1.PodCondition{
							{
								Type:               v1.PodReady,
								LastTransitionTime: metav1.Time{Time: time.Now().Add(-3 * time.Minute)},
								Status:             v1.ConditionTrue,
							},
						},
					},
				},
				{
					ObjectMeta: metav1.ObjectMeta{
						Name: "epicurus",
					},
					Status: v1.PodStatus{
						Phase: v1.PodSucceeded,
						StartTime: &metav1.Time{
							Time: time.Now().Add(-3 * time.Minute),
						},
					},
				},
			},
			metrics: metricsclient.PodMetricsInfo{
				"lucretius": metricsclient.PodMetric{Value: 1},
				"niccolo":   metricsclient.PodMetric{Value: 1},
			},
			resource:            v1.ResourceCPU,
			expectReadyPodCount: 1,
			expectUnreadyPods:   sets.New[string]("lucretius"),
			expectMissingPods:   sets.New[string]("epicurus"),
			expectIgnoredPods:   sets.New[string](),
		}, {
			name: "pending pods are unreadied",
			pods: []*v1.Pod{
				{
					ObjectMeta: metav1.ObjectMeta{
						Name: "unscheduled",
					},
					Status: v1.PodStatus{
						Phase: v1.PodPending,
					},
				},
			},
			metrics:             metricsclient.PodMetricsInfo{},
			resource:            v1.ResourceCPU,
			expectReadyPodCount: 0,
			expectUnreadyPods:   sets.New[string]("unscheduled"),
			expectMissingPods:   sets.New[string](),
			expectIgnoredPods:   sets.New[string](),
		}, {
			name: "ignore pods with deletion timestamps",
			pods: []*v1.Pod{
				{
					ObjectMeta: metav1.ObjectMeta{
						Name:              "deleted",
						DeletionTimestamp: &metav1.Time{Time: time.Unix(1, 0)},
					},
					Status: v1.PodStatus{
						Phase: v1.PodPending,
					},
				},
			},
			metrics: metricsclient.PodMetricsInfo{
				"deleted": metricsclient.PodMetric{Value: 1},
			},
			resource:            v1.ResourceCPU,
			expectReadyPodCount: 0,
			expectUnreadyPods:   sets.New[string](),
			expectMissingPods:   sets.New[string](),
			expectIgnoredPods:   sets.New[string]("deleted"),
		}, {
			name: "ignore pods in a failed state",
			pods: []*v1.Pod{
				{
					ObjectMeta: metav1.ObjectMeta{
						Name: "failed",
					},
					Status: v1.PodStatus{
						Phase: v1.PodFailed,
					},
				},
			},
			metrics: metricsclient.PodMetricsInfo{
				"failed": metricsclient.PodMetric{Value: 1},
			},
			resource:            v1.ResourceCPU,
			expectReadyPodCount: 0,
			expectUnreadyPods:   sets.New[string](),
			expectMissingPods:   sets.New[string](),
			expectIgnoredPods:   sets.New[string]("failed"),
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			readyPodCount, unreadyPods, missingPods, ignoredPods := groupPods(tc.pods, tc.metrics, tc.resource, defaultTestingCPUInitializationPeriod, defaultTestingDelayOfInitialReadinessStatus)
			if readyPodCount != tc.expectReadyPodCount {
				t.Errorf("%s got readyPodCount %d, expected %d", tc.name, readyPodCount, tc.expectReadyPodCount)
			}
			if !unreadyPods.Equal(tc.expectUnreadyPods) {
				t.Errorf("%s got unreadyPods %v, expected %v", tc.name, unreadyPods, tc.expectUnreadyPods)
			}
			if !missingPods.Equal(tc.expectMissingPods) {
				t.Errorf("%s got missingPods %v, expected %v", tc.name, missingPods, tc.expectMissingPods)
			}
			if !ignoredPods.Equal(tc.expectIgnoredPods) {
				t.Errorf("%s got ignoredPods %v, expected %v", tc.name, ignoredPods, tc.expectIgnoredPods)
			}
		})
	}
}

func TestCalculateRequests(t *testing.T) {
	containerRestartPolicyAlways := v1.ContainerRestartPolicyAlways
	testPod := "test-pod"

	tests := []struct {
		name                    string
		pods                    []*v1.Pod
		container               string
		resource                v1.ResourceName
		enablePodLevelResources bool
		expectedRequests        map[string]int64
		expectedError           error
	}{
		{
			name:             "void",
			pods:             []*v1.Pod{},
			container:        "",
			resource:         v1.ResourceCPU,
			expectedRequests: map[string]int64{},
			expectedError:    nil,
		},
		{
			name: "Sum container requests if pod-level feature is disabled",
			pods: []*v1.Pod{{
				ObjectMeta: metav1.ObjectMeta{
					Name:      testPod,
					Namespace: testNamespace,
				},
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{Name: "container1", Resources: v1.ResourceRequirements{Requests: v1.ResourceList{v1.ResourceCPU: *resource.NewMilliQuantity(100, resource.DecimalSI)}}},
						{Name: "container2", Resources: v1.ResourceRequirements{Requests: v1.ResourceList{v1.ResourceCPU: *resource.NewMilliQuantity(50, resource.DecimalSI)}}},
					},
				},
			}},
			container:        "",
			resource:         v1.ResourceCPU,
			expectedRequests: map[string]int64{testPod: 150},
			expectedError:    nil,
		},
		{
			name:                    "Pod-level resources are enabled, but not set: fallback to sum container requests",
			enablePodLevelResources: true,
			pods: []*v1.Pod{{
				ObjectMeta: metav1.ObjectMeta{
					Name:      testPod,
					Namespace: testNamespace,
				},
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{Name: "container1", Resources: v1.ResourceRequirements{Requests: v1.ResourceList{v1.ResourceCPU: *resource.NewMilliQuantity(100, resource.DecimalSI)}}},
						{Name: "container2", Resources: v1.ResourceRequirements{Requests: v1.ResourceList{v1.ResourceCPU: *resource.NewMilliQuantity(50, resource.DecimalSI)}}},
					},
				},
			}},
			container:        "",
			resource:         v1.ResourceCPU,
			expectedRequests: map[string]int64{testPod: 150},
			expectedError:    nil,
		},
		{
			name:                    "Pod-level resources override container requests when feature enabled and pod resources specified",
			enablePodLevelResources: true,
			pods: []*v1.Pod{{

				ObjectMeta: metav1.ObjectMeta{
					Name:      testPod,
					Namespace: testNamespace,
				},
				Spec: v1.PodSpec{
					Resources: &v1.ResourceRequirements{
						Requests: v1.ResourceList{v1.ResourceCPU: *resource.NewMilliQuantity(800, resource.DecimalSI)},
					},
					Containers: []v1.Container{
						{Name: "container1", Resources: v1.ResourceRequirements{Requests: v1.ResourceList{v1.ResourceCPU: *resource.NewMilliQuantity(100, resource.DecimalSI)}}},
						{Name: "container2", Resources: v1.ResourceRequirements{Requests: v1.ResourceList{v1.ResourceCPU: *resource.NewMilliQuantity(50, resource.DecimalSI)}}},
					},
				},
			}},
			container:        "",
			resource:         v1.ResourceCPU,
			expectedRequests: map[string]int64{testPod: 800},
			expectedError:    nil,
		},
		{
			name: "Fail if at least one of the containers is missing requests and pod-level feature/requests are not set",
			pods: []*v1.Pod{{
				ObjectMeta: metav1.ObjectMeta{
					Name:      testPod,
					Namespace: testNamespace,
				},
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{Name: "container1"},
						{Name: "container2", Resources: v1.ResourceRequirements{Requests: v1.ResourceList{v1.ResourceCPU: *resource.NewMilliQuantity(50, resource.DecimalSI)}}},
					},
				},
			}},
			container:        "",
			resource:         v1.ResourceCPU,
			expectedRequests: nil,
			expectedError:    fmt.Errorf("missing request for %s in container %s of Pod %s", v1.ResourceCPU, "container1", testPod),
		},
		{
			name:                    "Pod-level resources override missing container requests when feature enabled and pod resources specified",
			enablePodLevelResources: true,
			pods: []*v1.Pod{{
				ObjectMeta: metav1.ObjectMeta{
					Name:      testPod,
					Namespace: testNamespace,
				},
				Spec: v1.PodSpec{
					Resources: &v1.ResourceRequirements{
						Requests: v1.ResourceList{v1.ResourceCPU: *resource.NewMilliQuantity(800, resource.DecimalSI)},
					},
					Containers: []v1.Container{
						{Name: "container1"},
						{Name: "container2", Resources: v1.ResourceRequirements{Requests: v1.ResourceList{v1.ResourceCPU: *resource.NewMilliQuantity(50, resource.DecimalSI)}}},
					},
				},
			}},
			container:        "",
			resource:         v1.ResourceCPU,
			expectedRequests: map[string]int64{testPod: 800},
			expectedError:    nil,
		},
		{
			name: "Container: if a container name is specified, calculate requests only for that container",
			pods: []*v1.Pod{{
				ObjectMeta: metav1.ObjectMeta{
					Name:      testPod,
					Namespace: testNamespace,
				},
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{Name: "container1", Resources: v1.ResourceRequirements{Requests: v1.ResourceList{v1.ResourceCPU: *resource.NewMilliQuantity(100, resource.DecimalSI)}}},
						{Name: "container2", Resources: v1.ResourceRequirements{Requests: v1.ResourceList{v1.ResourceCPU: *resource.NewMilliQuantity(50, resource.DecimalSI)}}},
					},
				},
			}},
			container:        "container1",
			resource:         v1.ResourceCPU,
			expectedRequests: map[string]int64{testPod: 100},
			expectedError:    nil,
		},
		{
			name:                    "Container: if a container name is specified, calculate requests only for that container and ignore pod-level requests",
			enablePodLevelResources: true,
			pods: []*v1.Pod{{
				ObjectMeta: metav1.ObjectMeta{
					Name:      testPod,
					Namespace: testNamespace,
				},
				Spec: v1.PodSpec{
					Resources: &v1.ResourceRequirements{
						Requests: v1.ResourceList{v1.ResourceCPU: *resource.NewMilliQuantity(800, resource.DecimalSI)},
					},
					Containers: []v1.Container{
						{Name: "container1", Resources: v1.ResourceRequirements{Requests: v1.ResourceList{v1.ResourceCPU: *resource.NewMilliQuantity(100, resource.DecimalSI)}}},
						{Name: "container2", Resources: v1.ResourceRequirements{Requests: v1.ResourceList{v1.ResourceCPU: *resource.NewMilliQuantity(50, resource.DecimalSI)}}},
					},
				},
			}},
			container:        "container1",
			resource:         v1.ResourceCPU,
			expectedRequests: map[string]int64{testPod: 100},
			expectedError:    nil,
		},
		{
			name: "pod with restartable init containers",
			pods: []*v1.Pod{{
				ObjectMeta: metav1.ObjectMeta{
					Name:      testPod,
					Namespace: testNamespace,
				},
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{Name: "container1", Resources: v1.ResourceRequirements{Requests: v1.ResourceList{v1.ResourceCPU: *resource.NewMilliQuantity(100, resource.DecimalSI)}}},
					},
					InitContainers: []v1.Container{
						{Name: "init-container1", Resources: v1.ResourceRequirements{Requests: v1.ResourceList{v1.ResourceCPU: *resource.NewMilliQuantity(20, resource.DecimalSI)}}},
						{Name: "restartable-container1", RestartPolicy: &containerRestartPolicyAlways, Resources: v1.ResourceRequirements{Requests: v1.ResourceList{v1.ResourceCPU: *resource.NewMilliQuantity(50, resource.DecimalSI)}}},
					},
				},
			}},
			container:        "",
			resource:         v1.ResourceCPU,
			expectedRequests: map[string]int64{testPod: 150},
			expectedError:    nil,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.PodLevelResources, tc.enablePodLevelResources)

			requests, err := calculateRequests(tc.pods, tc.container, tc.resource)
			assert.Equal(t, tc.expectedRequests, requests, "requests should be as expected")
			assert.Equal(t, tc.expectedError, err, "error should be as expected")
		})
	}
}
func TestCalculatePodRequestsFromContainers_NonExistentContainer(t *testing.T) {
	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-pod",
			Namespace: testNamespace,
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Name: "container1",
					Resources: v1.ResourceRequirements{
						Requests: v1.ResourceList{
							v1.ResourceCPU: resource.MustParse("100m"),
						},
					},
				},
			},
		},
	}

	request, err := calculatePodRequestsFromContainers(pod, "non-existent-container", v1.ResourceCPU)

	require.Error(t, err, "expected error for non-existent container")
	expectedErr := "container non-existent-container not found in Pod test-pod"
	assert.Equal(t, expectedErr, err.Error(), "error message should match expected format")
	assert.Equal(t, int64(0), request, "request should be 0 when container does not exist")
}

func TestReplicaCalcExternalMetricUsageOverflow(t *testing.T) {
	tc := replicaCalcTestCase{
		currentReplicas:  3,
		expectedReplicas: 3,
		metric: &metricInfo{
			name: "qps",
			// Two values that when added together will overflow int64
			levels:        []int64{math.MaxInt64/2 + 1, math.MaxInt64/2 + 1},
			targetUsage:   math.MaxInt64, // Set high target
			metricType:    externalMetric,
			selector:      &metav1.LabelSelector{MatchLabels: map[string]string{"label": "value"}},
			expectedUsage: math.MaxInt64, // expect capped value

		},
	}
	tc.runTest(t)
}
