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

package podautoscaler

import (
	"fmt"
	"sync"
	"testing"
	"time"

	autoscalingv1 "k8s.io/api/autoscaling/v1"
	autoscalingv2 "k8s.io/api/autoscaling/v2"
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/client-go/kubernetes/fake"
	scalefake "k8s.io/client-go/scale/fake"
	core "k8s.io/client-go/testing"
	autoscalingapiv2 "k8s.io/kubernetes/pkg/apis/autoscaling/v2"
	"k8s.io/kubernetes/pkg/controller/podautoscaler/monitor"
	"k8s.io/kubernetes/pkg/controller/util/selectors"
	metricsfake "k8s.io/metrics/pkg/client/clientset/versioned/fake"
	cmfake "k8s.io/metrics/pkg/client/custom_metrics/fake"
	emfake "k8s.io/metrics/pkg/client/external_metrics/fake"

	"github.com/stretchr/testify/assert"

	_ "k8s.io/kubernetes/pkg/apis/apps/install"
	_ "k8s.io/kubernetes/pkg/apis/autoscaling/install"

	"k8s.io/client-go/tools/cache"
	watchtools "k8s.io/client-go/tools/watch"
)

// From now on, the HPA controller does have history in it (scaleUpEvents, scaleDownEvents)
// Hence the second HPA controller reconcile cycle might return different result (comparing with the first run).
// Current test infrastructure has a race condition, when several reconcile cycles will be performed
//    while it should be stopped right after the first one. And the second will raise an exception
//    because of different result.

// This comment has more info: https://github.com/kubernetes/kubernetes/pull/74525#issuecomment-502653106
// We need to rework this infrastructure:  https://github.com/kubernetes/kubernetes/issues/79222

var statusOk = []autoscalingv2.HorizontalPodAutoscalerCondition{
	{Type: autoscalingv2.AbleToScale, Status: v1.ConditionTrue, Reason: "SucceededRescale"},
	{Type: autoscalingv2.ScalingActive, Status: v1.ConditionTrue, Reason: "ValidMetricFound"},
	{Type: autoscalingv2.ScalingLimited, Status: v1.ConditionFalse, Reason: "DesiredWithinRange"},
}

// statusOkWithOverrides returns the "ok" status with the given conditions as overridden
func statusOkWithOverrides(overrides ...autoscalingv2.HorizontalPodAutoscalerCondition) []autoscalingv2.HorizontalPodAutoscalerCondition {
	resv2 := make([]autoscalingv2.HorizontalPodAutoscalerCondition, len(statusOk))
	copy(resv2, statusOk)
	for _, override := range overrides {
		resv2 = setConditionInList(resv2, override.Type, override.Status, override.Reason, "%s", override.Message)
	}

	// copy to a v1 slice
	resv1 := make([]autoscalingv2.HorizontalPodAutoscalerCondition, len(resv2))
	for i, cond := range resv2 {
		resv1[i] = autoscalingv2.HorizontalPodAutoscalerCondition{
			Type:   autoscalingv2.HorizontalPodAutoscalerConditionType(cond.Type),
			Status: cond.Status,
			Reason: cond.Reason,
		}
	}

	return resv1
}

func alwaysReady() bool { return true }

type fakeResource struct {
	name       string
	apiVersion string
	kind       string
}

type testCase struct {
	sync.Mutex
	minReplicas     int32
	maxReplicas     int32
	specReplicas    int32
	statusReplicas  int32
	initialReplicas int32
	scaleUpRules    *autoscalingv2.HPAScalingRules
	scaleDownRules  *autoscalingv2.HPAScalingRules

	// CPU target utilization as a percentage of the requested resources.
	CPUTarget                    int32
	CPUCurrent                   int32
	verifyCPUCurrent             bool
	reportedLevels               []uint64
	reportedCPURequests          []resource.Quantity
	reportedPodReadiness         []v1.ConditionStatus
	reportedPodStartTime         []metav1.Time
	reportedPodPhase             []v1.PodPhase
	reportedPodDeletionTimestamp []bool
	scaleUpdated                 bool
	statusUpdated                bool
	eventCreated                 bool
	verifyEvents                 bool
	useMetricsAPI                bool
	metricsTarget                []autoscalingv2.MetricSpec
	expectedDesiredReplicas      int32
	expectedConditions           []autoscalingv2.HorizontalPodAutoscalerCondition
	// Channel with names of HPA objects which we have reconciled.
	processed chan string

	// expected results reported to the mock monitor at first.
	expectedReportedReconciliationActionLabel     monitor.ActionLabel
	expectedReportedReconciliationErrorLabel      monitor.ErrorLabel
	expectedReportedMetricComputationActionLabels map[autoscalingv2.MetricSourceType]monitor.ActionLabel
	expectedReportedMetricComputationErrorLabels  map[autoscalingv2.MetricSourceType]monitor.ErrorLabel

	// Target resource information.
	resource *fakeResource

	// Last scale time
	lastScaleTime *metav1.Time

	// override the test clients
	testClient        *fake.Clientset
	testMetricsClient *metricsfake.Clientset
	testCMClient      *cmfake.FakeCustomMetricsClient
	testEMClient      *emfake.FakeExternalMetricsClient
	testScaleClient   *scalefake.FakeScaleClient

	recommendations []timestampedRecommendation
	hpaSelectors    *selectors.BiMultimap
}

// Needs to be called under a lock.
func (tc *testCase) computeCPUCurrent() {
	if len(tc.reportedLevels) != len(tc.reportedCPURequests) || len(tc.reportedLevels) == 0 {
		return
	}
	reported := 0
	for _, r := range tc.reportedLevels {
		reported += int(r)
	}
	requested := 0
	for _, req := range tc.reportedCPURequests {
		requested += int(req.MilliValue())
	}
	tc.CPUCurrent = int32(100 * reported / requested)
}

func init() {
	// set this high so we don't accidentally run into it when testing
	scaleUpLimitFactor = 8
}

func findCpuUtilization(metrics []autoscalingv2.MetricStatus) *int32 {
	for _, metric := range metrics {
		if metric.Type == autoscalingv2.ResourceMetricSourceType && metric.Resource != nil && metric.Resource.Name == v1.ResourceCPU {
			return metric.Resource.Current.AverageUtilization
		}
	}
	return nil
}

func (tc *testCase) prepareTestClient(t *testing.T) (*fake.Clientset, *metricsfake.Clientset, *cmfake.FakeCustomMetricsClient, *emfake.FakeExternalMetricsClient, *scalefake.FakeScaleClient) {
	namespace := "test-namespace"
	hpaName := "test-hpa"
	podNamePrefix := "test-pod"
	labelSet := map[string]string{"name": podNamePrefix}
	selector := labels.SelectorFromSet(labelSet).String()

	tc.Lock()

	tc.scaleUpdated = false
	tc.statusUpdated = false
	tc.eventCreated = false
	tc.processed = make(chan string, 100)
	if tc.CPUCurrent == 0 {
		tc.computeCPUCurrent()
	}

	if tc.resource == nil {
		tc.resource = &fakeResource{
			name:       "test-rc",
			apiVersion: "v1",
			kind:       "ReplicationController",
		}
	}
	tc.Unlock()

	fakeClient := &fake.Clientset{}
	fakeWatch := watch.NewFake()
	fakeClient.AddWatchReactor("pods", func(action core.Action) (handled bool, ret watch.Interface, err error) {
		// Create a new watch each time to avoid the "Failed to watch" errors
		fakeWatch := watch.NewFake()
		lw := &cache.ListWatch{
			WatchFunc: func(options metav1.ListOptions) (watch.Interface, error) {
				return fakeWatch, nil
			},
		}
		w, err := watchtools.NewRetryWatcher("1", lw)
		return true, w, err
	})
	fakeClient.AddWatchReactor("*", core.DefaultWatchReactor(fakeWatch, nil))
	fakeClient.AddReactor("list", "horizontalpodautoscalers", func(action core.Action) (handled bool, ret runtime.Object, err error) {
		tc.Lock()
		defer tc.Unlock()
		var behavior *autoscalingv2.HorizontalPodAutoscalerBehavior
		if tc.scaleUpRules != nil || tc.scaleDownRules != nil {
			behavior = &autoscalingv2.HorizontalPodAutoscalerBehavior{
				ScaleUp:   tc.scaleUpRules,
				ScaleDown: tc.scaleDownRules,
			}
		}
		hpa := autoscalingv2.HorizontalPodAutoscaler{
			ObjectMeta: metav1.ObjectMeta{
				Name:      hpaName,
				Namespace: namespace,
			},
			Spec: autoscalingv2.HorizontalPodAutoscalerSpec{
				ScaleTargetRef: autoscalingv2.CrossVersionObjectReference{
					Kind:       tc.resource.kind,
					Name:       tc.resource.name,
					APIVersion: tc.resource.apiVersion,
				},
				MinReplicas: &tc.minReplicas,
				MaxReplicas: tc.maxReplicas,
				Behavior:    behavior,
			},
			Status: autoscalingv2.HorizontalPodAutoscalerStatus{
				CurrentReplicas: tc.specReplicas,
				DesiredReplicas: tc.specReplicas,
				LastScaleTime:   tc.lastScaleTime,
			},
		}
		// Initialize default values
		autoscalingapiv2.SetDefaults_HorizontalPodAutoscalerBehavior(&hpa)

		obj := &autoscalingv2.HorizontalPodAutoscalerList{
			Items: []autoscalingv2.HorizontalPodAutoscaler{hpa},
		}

		if tc.CPUTarget > 0 {
			obj.Items[0].Spec.Metrics = []autoscalingv2.MetricSpec{
				{
					Type: autoscalingv2.ResourceMetricSourceType,
					Resource: &autoscalingv2.ResourceMetricSource{
						Name: v1.ResourceCPU,
						Target: autoscalingv2.MetricTarget{
							Type:               autoscalingv2.UtilizationMetricType,
							AverageUtilization: &tc.CPUTarget,
						},
					},
				},
			}
		}
		if len(tc.metricsTarget) > 0 {
			obj.Items[0].Spec.Metrics = append(obj.Items[0].Spec.Metrics, tc.metricsTarget...)
		}

		if len(obj.Items[0].Spec.Metrics) == 0 {
			// manually add in the defaulting logic
			obj.Items[0].Spec.Metrics = []autoscalingv2.MetricSpec{
				{
					Type: autoscalingv2.ResourceMetricSourceType,
					Resource: &autoscalingv2.ResourceMetricSource{
						Name: v1.ResourceCPU,
					},
				},
			}
		}

		return true, obj, nil
	})

	fakeClient.AddReactor("list", "pods", func(action core.Action) (handled bool, ret runtime.Object, err error) {
		tc.Lock()
		defer tc.Unlock()

		obj := &v1.PodList{}

		specifiedCPURequests := tc.reportedCPURequests != nil

		numPodsToCreate := int(tc.statusReplicas)
		if specifiedCPURequests {
			numPodsToCreate = len(tc.reportedCPURequests)
		}

		for i := 0; i < numPodsToCreate; i++ {
			podReadiness := v1.ConditionTrue
			if tc.reportedPodReadiness != nil {
				podReadiness = tc.reportedPodReadiness[i]
			}
			var podStartTime metav1.Time
			if tc.reportedPodStartTime != nil {
				podStartTime = tc.reportedPodStartTime[i]
			}

			podPhase := v1.PodRunning
			if tc.reportedPodPhase != nil {
				podPhase = tc.reportedPodPhase[i]
			}

			podDeletionTimestamp := false
			if tc.reportedPodDeletionTimestamp != nil {
				podDeletionTimestamp = tc.reportedPodDeletionTimestamp[i]
			}

			podName := fmt.Sprintf("%s-%d", podNamePrefix, i)

			reportedCPURequest := resource.MustParse("1.0")
			if specifiedCPURequests {
				reportedCPURequest = tc.reportedCPURequests[i]
			}

			pod := v1.Pod{
				Status: v1.PodStatus{
					Phase: podPhase,
					Conditions: []v1.PodCondition{
						{
							Type:               v1.PodReady,
							Status:             podReadiness,
							LastTransitionTime: podStartTime,
						},
					},
					StartTime: &podStartTime,
				},
				ObjectMeta: metav1.ObjectMeta{
					Name:      podName,
					Namespace: namespace,
					Labels: map[string]string{
						"name": podNamePrefix,
					},
				},

				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							Name: "container1",
							Resources: v1.ResourceRequirements{
								Requests: v1.ResourceList{
									v1.ResourceCPU: *resource.NewMilliQuantity(reportedCPURequest.MilliValue()/2, resource.DecimalSI),
								},
							},
						},
						{
							Name: "container2",
							Resources: v1.ResourceRequirements{
								Requests: v1.ResourceList{
									v1.ResourceCPU: *resource.NewMilliQuantity(reportedCPURequest.MilliValue()/2, resource.DecimalSI),
								},
							},
						},
					},
				},
			}
			if podDeletionTimestamp {
				pod.DeletionTimestamp = &metav1.Time{Time: time.Now()}
			}
			obj.Items = append(obj.Items, pod)
		}
		return true, obj, nil
	})

	fakeClient.AddReactor("update", "horizontalpodautoscalers", func(action core.Action) (handled bool, ret runtime.Object, err error) {
		handled, obj, err := func() (handled bool, ret *autoscalingv2.HorizontalPodAutoscaler, err error) {
			tc.Lock()
			defer tc.Unlock()

			obj := action.(core.UpdateAction).GetObject().(*autoscalingv2.HorizontalPodAutoscaler)
			assert.Equal(t, namespace, obj.Namespace, "the HPA namespace should be as expected")
			assert.Equal(t, hpaName, obj.Name, "the HPA name should be as expected")
			assert.Equal(t, tc.expectedDesiredReplicas, obj.Status.DesiredReplicas, "the desired replica count reported in the object status should be as expected")
			if tc.verifyCPUCurrent {
				if utilization := findCpuUtilization(obj.Status.CurrentMetrics); assert.NotNil(t, utilization, "the reported CPU utilization percentage should be non-nil") {
					assert.Equal(t, tc.CPUCurrent, *utilization, "the report CPU utilization percentage should be as expected")
				}
			}

			if len(obj.Spec.Metrics) > 0 && obj.Spec.Metrics[0].Object != nil && len(obj.Status.CurrentMetrics) > 0 && obj.Status.CurrentMetrics[0].Object != nil {
				assert.Equal(t, obj.Spec.Metrics[0].Object.DescribedObject.APIVersion, obj.Status.CurrentMetrics[0].Object.DescribedObject.APIVersion)
				assert.Equal(t, obj.Spec.Metrics[0].Object.DescribedObject.Kind, obj.Status.CurrentMetrics[0].Object.DescribedObject.Kind)
				assert.Equal(t, obj.Spec.Metrics[0].Object.DescribedObject.Name, obj.Status.CurrentMetrics[0].Object.DescribedObject.Name)
			}

			actualConditions := obj.Status.Conditions
			// TODO: it's ok not to sort these because statusOk
			// contains all the conditions, so we'll never be appending.
			// Default to statusOk when missing any specific conditions
			if tc.expectedConditions == nil {
				tc.expectedConditions = statusOkWithOverrides()
			}
			// clear the message so that we can easily compare
			for i := range actualConditions {
				actualConditions[i].Message = ""
				actualConditions[i].LastTransitionTime = metav1.Time{}
			}
			assert.Equal(t, tc.expectedConditions, actualConditions, "the status conditions should have been as expected")
			tc.statusUpdated = true
			// Every time we reconcile HPA object we are updating status.
			return true, obj, nil
		}()
		if obj != nil {
			tc.processed <- obj.Name
		}
		return handled, obj, err
	})

	fakeScaleClient := &scalefake.FakeScaleClient{}
	fakeScaleClient.AddReactor("get", "replicationcontrollers", func(action core.Action) (handled bool, ret runtime.Object, err error) {
		tc.Lock()
		defer tc.Unlock()

		obj := &autoscalingv1.Scale{
			ObjectMeta: metav1.ObjectMeta{
				Name:      tc.resource.name,
				Namespace: namespace,
			},
			Spec: autoscalingv1.ScaleSpec{
				Replicas: tc.specReplicas,
			},
			Status: autoscalingv1.ScaleStatus{
				Replicas: tc.statusReplicas,
				Selector: selector,
			},
		}
		return true, obj, nil
	})

	fakeScaleClient.AddReactor("get", "deployments", func(action core.Action) (handled bool, ret runtime.Object, err error) {
		tc.Lock()
		defer tc.Unlock()

		obj := &autoscalingv1.Scale{
			ObjectMeta: metav1.ObjectMeta{
				Name:      tc.resource.name,
				Namespace: namespace,
			},
			Spec: autoscalingv1.ScaleSpec{
				Replicas: tc.specReplicas,
			},
			Status: autoscalingv1.ScaleStatus{
				Replicas: tc.statusReplicas,
				Selector: selector,
			},
		}
		return true, obj, nil
	})

	fakeScaleClient.AddReactor("get", "replicasets", func(action core.Action) (handled bool, ret runtime.Object, err error) {
		tc.Lock()
		defer tc.Unlock()

		obj := &autoscalingv1.Scale{
			ObjectMeta: metav1.ObjectMeta{
				Name:      tc.resource.name,
				Namespace: namespace,
			},
			Spec: autoscalingv1.ScaleSpec{
				Replicas: tc.specReplicas,
			},
			Status: autoscalingv1.ScaleStatus{
				Replicas: tc.statusReplicas,
				Selector: selector,
			},
		}
		return true, obj, nil
	})

	fakeScaleClient.AddReactor("update", "replicationcontrollers", func(action core.Action) (handled bool, ret runtime.Object, err error) {
		tc.Lock()
		defer tc.Unlock()

		obj := action.(core.UpdateAction).GetObject().(*autoscalingv1.Scale)
		replicas := action.(core.UpdateAction).GetObject().(*autoscalingv1.Scale).Spec.Replicas
		assert.Equal(t, tc.expectedDesiredReplicas, replicas, "the replica count of the RC should be as expected")
		tc.scaleUpdated = true
		return true, obj, nil
	})

	fakeScaleClient.AddReactor("update", "deployments", func(action core.Action) (handled bool, ret runtime.Object, err error) {
		tc.Lock()
		defer tc.Unlock()

		obj := action.(core.UpdateAction).GetObject().(*autoscalingv1.Scale)
		replicas := action.(core.UpdateAction).GetObject().(*autoscalingv1.Scale).Spec.Replicas
		assert.Equal(t, tc.expectedDesiredReplicas, replicas, "the replica count of the deployment should be as expected")
		tc.scaleUpdated = true
		return true, obj, nil
	})

	fakeScaleClient.AddReactor("update", "replicasets", func(action core.Action) (handled bool, ret runtime.Object, err error) {
		tc.Lock()
		defer tc.Unlock()

		obj := action.(core.UpdateAction).GetObject().(*autoscalingv1.Scale)
		replicas := action.(core.UpdateAction).GetObject().(*autoscalingv1.Scale).Spec.Replicas
		assert.Equal(t, tc.expectedDesiredReplicas, replicas, "the replica count of the replicaset should be as expected")
		tc.scaleUpdated = true
		return true, obj, nil
	})

	fakeMetricsClient := &metricsfake.Clientset{}
	fakeMetricsClient.AddReactor("list", "pods", func(action core.Action) (handled bool, ret runtime.Object, err error) {
		tc.Lock()
		defer tc.Unlock()

		obj := &v1.PodList{}

		specifiedCPURequests := tc.reportedCPURequests != nil

		numPodsToCreate := int(tc.statusReplicas)
		if specifiedCPURequests {
			numPodsToCreate = len(tc.reportedCPURequests)
		}

		for i := 0; i < numPodsToCreate; i++ {
			podReadiness := v1.ConditionTrue
			if tc.reportedPodReadiness != nil {
				podReadiness = tc.reportedPodReadiness[i]
			}
			var podStartTime metav1.Time
			if tc.reportedPodStartTime != nil {
				podStartTime = tc.reportedPodStartTime[i]
			}

			podPhase := v1.PodRunning
			if tc.reportedPodPhase != nil {
				podPhase = tc.reportedPodPhase[i]
			}

			podDeletionTimestamp := false
			if tc.reportedPodDeletionTimestamp != nil {
				podDeletionTimestamp = tc.reportedPodDeletionTimestamp[i]
			}

			podName := fmt.Sprintf("%s-%d", podNamePrefix, i)

			reportedCPURequest := resource.MustParse("1.0")
			if specifiedCPURequests {
				reportedCPURequest = tc.reportedCPURequests[i]
			}

			pod := v1.Pod{
				Status: v1.PodStatus{
					Phase: podPhase,
					Conditions: []v1.PodCondition{
						{
							Type:               v1.PodReady,
							Status:             podReadiness,
							LastTransitionTime: podStartTime,
						},
					},
					StartTime: &podStartTime,
				},
				ObjectMeta: metav1.ObjectMeta{
					Name:      podName,
					Namespace: namespace,
					Labels: map[string]string{
						"name": podNamePrefix,
					},
				},

				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							Name: "container1",
							Resources: v1.ResourceRequirements{
								Requests: v1.ResourceList{
									v1.ResourceCPU: *resource.NewMilliQuantity(reportedCPURequest.MilliValue()/2, resource.DecimalSI),
								},
							},
						},
						{
							Name: "container2",
							Resources: v1.ResourceRequirements{
								Requests: v1.ResourceList{
									v1.ResourceCPU: *resource.NewMilliQuantity(reportedCPURequest.MilliValue()/2, resource.DecimalSI),
								},
							},
						},
					},
				},
			}
			if podDeletionTimestamp {
				pod.DeletionTimestamp = &metav1.Time{Time: time.Now()}
			}
			obj.Items = append(obj.Items, pod)
		}
		return true, obj, nil
	})

	fakeCMClient := &cmfake.FakeCustomMetricsClient{}
	fakeEMClient := &emfake.FakeExternalMetricsClient{}

	return fakeClient, fakeMetricsClient, fakeCMClient, fakeEMClient, fakeScaleClient
}
