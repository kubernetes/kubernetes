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
	"encoding/json"
	"fmt"
	"math"
	"sync"
	"testing"
	"time"

	autoscalingv1 "k8s.io/api/autoscaling/v1"
	autoscalingv2 "k8s.io/api/autoscaling/v2beta2"
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/meta/testrestmapper"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes/fake"
	scalefake "k8s.io/client-go/scale/fake"
	core "k8s.io/client-go/testing"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	"k8s.io/kubernetes/pkg/apis/autoscaling"
	autoscalingapiv2beta2 "k8s.io/kubernetes/pkg/apis/autoscaling/v2beta2"
	"k8s.io/kubernetes/pkg/controller"
	"k8s.io/kubernetes/pkg/controller/podautoscaler/metrics"
	cmapi "k8s.io/metrics/pkg/apis/custom_metrics/v1beta2"
	emapi "k8s.io/metrics/pkg/apis/external_metrics/v1beta1"
	metricsapi "k8s.io/metrics/pkg/apis/metrics/v1beta1"
	metricsfake "k8s.io/metrics/pkg/client/clientset/versioned/fake"
	cmfake "k8s.io/metrics/pkg/client/custom_metrics/fake"
	emfake "k8s.io/metrics/pkg/client/external_metrics/fake"
	utilpointer "k8s.io/utils/pointer"

	"github.com/stretchr/testify/assert"

	_ "k8s.io/kubernetes/pkg/apis/apps/install"
	_ "k8s.io/kubernetes/pkg/apis/autoscaling/install"
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
func statusOkWithOverrides(overrides ...autoscalingv2.HorizontalPodAutoscalerCondition) []autoscalingv1.HorizontalPodAutoscalerCondition {
	resv2 := make([]autoscalingv2.HorizontalPodAutoscalerCondition, len(statusOk))
	copy(resv2, statusOk)
	for _, override := range overrides {
		resv2 = setConditionInList(resv2, override.Type, override.Status, override.Reason, override.Message)
	}

	// copy to a v1 slice
	resv1 := make([]autoscalingv1.HorizontalPodAutoscalerCondition, len(resv2))
	for i, cond := range resv2 {
		resv1[i] = autoscalingv1.HorizontalPodAutoscalerCondition{
			Type:   autoscalingv1.HorizontalPodAutoscalerConditionType(cond.Type),
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
	expectedConditions           []autoscalingv1.HorizontalPodAutoscalerCondition
	// Channel with names of HPA objects which we have reconciled.
	processed chan string

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
				SelfLink:  "experimental/v1/namespaces/" + namespace + "/horizontalpodautoscalers/" + hpaName,
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
		autoscalingapiv2beta2.SetDefaults_HorizontalPodAutoscalerBehavior(&hpa)

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

		// and... convert to autoscaling v1 to return the right type
		objv1, err := unsafeConvertToVersionVia(obj, autoscalingv1.SchemeGroupVersion)
		if err != nil {
			return true, nil, err
		}

		return true, objv1, nil
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
		handled, obj, err := func() (handled bool, ret *autoscalingv1.HorizontalPodAutoscaler, err error) {
			tc.Lock()
			defer tc.Unlock()

			obj := action.(core.UpdateAction).GetObject().(*autoscalingv1.HorizontalPodAutoscaler)
			assert.Equal(t, namespace, obj.Namespace, "the HPA namespace should be as expected")
			assert.Equal(t, hpaName, obj.Name, "the HPA name should be as expected")
			assert.Equal(t, tc.expectedDesiredReplicas, obj.Status.DesiredReplicas, "the desired replica count reported in the object status should be as expected")
			if tc.verifyCPUCurrent {
				if assert.NotNil(t, obj.Status.CurrentCPUUtilizationPercentage, "the reported CPU utilization percentage should be non-nil") {
					assert.Equal(t, tc.CPUCurrent, *obj.Status.CurrentCPUUtilizationPercentage, "the report CPU utilization percentage should be as expected")
				}
			}
			var actualConditions []autoscalingv1.HorizontalPodAutoscalerCondition
			if err := json.Unmarshal([]byte(obj.ObjectMeta.Annotations[autoscaling.HorizontalPodAutoscalerConditionsAnnotation]), &actualConditions); err != nil {
				return true, nil, err
			}
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

	fakeWatch := watch.NewFake()
	fakeClient.AddWatchReactor("*", core.DefaultWatchReactor(fakeWatch, nil))

	fakeMetricsClient := &metricsfake.Clientset{}
	fakeMetricsClient.AddReactor("list", "pods", func(action core.Action) (handled bool, ret runtime.Object, err error) {
		tc.Lock()
		defer tc.Unlock()

		metrics := &metricsapi.PodMetricsList{}
		for i, cpu := range tc.reportedLevels {
			// NB: the list reactor actually does label selector filtering for us,
			// so we have to make sure our results match the label selector
			podMetric := metricsapi.PodMetrics{
				ObjectMeta: metav1.ObjectMeta{
					Name:      fmt.Sprintf("%s-%d", podNamePrefix, i),
					Namespace: namespace,
					Labels:    labelSet,
				},
				Timestamp: metav1.Time{Time: time.Now()},
				Window:    metav1.Duration{Duration: time.Minute},
				Containers: []metricsapi.ContainerMetrics{
					{
						Name: "container1",
						Usage: v1.ResourceList{
							v1.ResourceCPU: *resource.NewMilliQuantity(
								int64(cpu/2),
								resource.DecimalSI),
							v1.ResourceMemory: *resource.NewQuantity(
								int64(1024*1024/2),
								resource.BinarySI),
						},
					},
					{
						Name: "container2",
						Usage: v1.ResourceList{
							v1.ResourceCPU: *resource.NewMilliQuantity(
								int64(cpu/2),
								resource.DecimalSI),
							v1.ResourceMemory: *resource.NewQuantity(
								int64(1024*1024/2),
								resource.BinarySI),
						},
					},
				},
			}
			metrics.Items = append(metrics.Items, podMetric)
		}

		return true, metrics, nil
	})

	fakeCMClient := &cmfake.FakeCustomMetricsClient{}
	fakeCMClient.AddReactor("get", "*", func(action core.Action) (handled bool, ret runtime.Object, err error) {
		tc.Lock()
		defer tc.Unlock()

		getForAction, wasGetFor := action.(cmfake.GetForAction)
		if !wasGetFor {
			return true, nil, fmt.Errorf("expected a get-for action, got %v instead", action)
		}

		if getForAction.GetName() == "*" {
			metrics := &cmapi.MetricValueList{}

			// multiple objects
			assert.Equal(t, "pods", getForAction.GetResource().Resource, "the type of object that we requested multiple metrics for should have been pods")
			assert.Equal(t, "qps", getForAction.GetMetricName(), "the metric name requested should have been qps, as specified in the metric spec")

			for i, level := range tc.reportedLevels {
				podMetric := cmapi.MetricValue{
					DescribedObject: v1.ObjectReference{
						Kind:      "Pod",
						Name:      fmt.Sprintf("%s-%d", podNamePrefix, i),
						Namespace: namespace,
					},
					Timestamp: metav1.Time{Time: time.Now()},
					Metric: cmapi.MetricIdentifier{
						Name: "qps",
					},
					Value: *resource.NewMilliQuantity(int64(level), resource.DecimalSI),
				}
				metrics.Items = append(metrics.Items, podMetric)
			}

			return true, metrics, nil
		}

		name := getForAction.GetName()
		mapper := testrestmapper.TestOnlyStaticRESTMapper(legacyscheme.Scheme)
		metrics := &cmapi.MetricValueList{}
		var matchedTarget *autoscalingv2.MetricSpec
		for i, target := range tc.metricsTarget {
			if target.Type == autoscalingv2.ObjectMetricSourceType && name == target.Object.DescribedObject.Name {
				gk := schema.FromAPIVersionAndKind(target.Object.DescribedObject.APIVersion, target.Object.DescribedObject.Kind).GroupKind()
				mapping, err := mapper.RESTMapping(gk)
				if err != nil {
					t.Logf("unable to get mapping for %s: %v", gk.String(), err)
					continue
				}
				groupResource := mapping.Resource.GroupResource()

				if getForAction.GetResource().Resource == groupResource.String() {
					matchedTarget = &tc.metricsTarget[i]
				}
			}
		}
		assert.NotNil(t, matchedTarget, "this request should have matched one of the metric specs")
		assert.Equal(t, "qps", getForAction.GetMetricName(), "the metric name requested should have been qps, as specified in the metric spec")

		metrics.Items = []cmapi.MetricValue{
			{
				DescribedObject: v1.ObjectReference{
					Kind:       matchedTarget.Object.DescribedObject.Kind,
					APIVersion: matchedTarget.Object.DescribedObject.APIVersion,
					Name:       name,
				},
				Timestamp: metav1.Time{Time: time.Now()},
				Metric: cmapi.MetricIdentifier{
					Name: "qps",
				},
				Value: *resource.NewMilliQuantity(int64(tc.reportedLevels[0]), resource.DecimalSI),
			},
		}

		return true, metrics, nil
	})

	fakeEMClient := &emfake.FakeExternalMetricsClient{}

	fakeEMClient.AddReactor("list", "*", func(action core.Action) (handled bool, ret runtime.Object, err error) {
		tc.Lock()
		defer tc.Unlock()

		listAction, wasList := action.(core.ListAction)
		if !wasList {
			return true, nil, fmt.Errorf("expected a list action, got %v instead", action)
		}

		metrics := &emapi.ExternalMetricValueList{}

		assert.Equal(t, "qps", listAction.GetResource().Resource, "the metric name requested should have been qps, as specified in the metric spec")

		for _, level := range tc.reportedLevels {
			metric := emapi.ExternalMetricValue{
				Timestamp:  metav1.Time{Time: time.Now()},
				MetricName: "qps",
				Value:      *resource.NewMilliQuantity(int64(level), resource.DecimalSI),
			}
			metrics.Items = append(metrics.Items, metric)
		}

		return true, metrics, nil
	})

	return fakeClient, fakeMetricsClient, fakeCMClient, fakeEMClient, fakeScaleClient
}

func (tc *testCase) verifyResults(t *testing.T) {
	tc.Lock()
	defer tc.Unlock()

	assert.Equal(t, tc.specReplicas != tc.expectedDesiredReplicas, tc.scaleUpdated, "the scale should only be updated if we expected a change in replicas")
	assert.True(t, tc.statusUpdated, "the status should have been updated")
	if tc.verifyEvents {
		assert.Equal(t, tc.specReplicas != tc.expectedDesiredReplicas, tc.eventCreated, "an event should have been created only if we expected a change in replicas")
	}
}

func (tc *testCase) setupController(t *testing.T) (*HorizontalController, informers.SharedInformerFactory) {
	testClient, testMetricsClient, testCMClient, testEMClient, testScaleClient := tc.prepareTestClient(t)
	if tc.testClient != nil {
		testClient = tc.testClient
	}
	if tc.testMetricsClient != nil {
		testMetricsClient = tc.testMetricsClient
	}
	if tc.testCMClient != nil {
		testCMClient = tc.testCMClient
	}
	if tc.testEMClient != nil {
		testEMClient = tc.testEMClient
	}
	if tc.testScaleClient != nil {
		testScaleClient = tc.testScaleClient
	}
	metricsClient := metrics.NewRESTMetricsClient(
		testMetricsClient.MetricsV1beta1(),
		testCMClient,
		testEMClient,
	)

	eventClient := &fake.Clientset{}
	eventClient.AddReactor("create", "events", func(action core.Action) (handled bool, ret runtime.Object, err error) {
		tc.Lock()
		defer tc.Unlock()

		obj := action.(core.CreateAction).GetObject().(*v1.Event)
		if tc.verifyEvents {
			switch obj.Reason {
			case "SuccessfulRescale":
				assert.Equal(t, fmt.Sprintf("New size: %d; reason: cpu resource utilization (percentage of request) above target", tc.expectedDesiredReplicas), obj.Message)
			case "DesiredReplicasComputed":
				assert.Equal(t, fmt.Sprintf(
					"Computed the desired num of replicas: %d (avgCPUutil: %d, current replicas: %d)",
					tc.expectedDesiredReplicas,
					(int64(tc.reportedLevels[0])*100)/tc.reportedCPURequests[0].MilliValue(), tc.specReplicas), obj.Message)
			default:
				assert.False(t, true, fmt.Sprintf("Unexpected event: %s / %s", obj.Reason, obj.Message))
			}
		}
		tc.eventCreated = true
		return true, obj, nil
	})

	informerFactory := informers.NewSharedInformerFactory(testClient, controller.NoResyncPeriodFunc())
	defaultDownscalestabilizationWindow := 5 * time.Minute

	hpaController := NewHorizontalController(
		eventClient.CoreV1(),
		testScaleClient,
		testClient.AutoscalingV1(),
		testrestmapper.TestOnlyStaticRESTMapper(legacyscheme.Scheme),
		metricsClient,
		informerFactory.Autoscaling().V1().HorizontalPodAutoscalers(),
		informerFactory.Core().V1().Pods(),
		100*time.Millisecond, // we need non-zero resync period to avoid race conditions
		defaultDownscalestabilizationWindow,
		defaultTestingTolerance,
		defaultTestingCPUInitializationPeriod,
		defaultTestingDelayOfInitialReadinessStatus,
	)
	hpaController.hpaListerSynced = alwaysReady
	if tc.recommendations != nil {
		hpaController.recommendations["test-namespace/test-hpa"] = tc.recommendations
	}

	return hpaController, informerFactory
}

func hotCPUCreationTime() metav1.Time {
	return metav1.Time{Time: time.Now()}
}

func coolCPUCreationTime() metav1.Time {
	return metav1.Time{Time: time.Now().Add(-3 * time.Minute)}
}

func (tc *testCase) runTestWithController(t *testing.T, hpaController *HorizontalController, informerFactory informers.SharedInformerFactory) {
	stop := make(chan struct{})
	defer close(stop)
	informerFactory.Start(stop)
	go hpaController.Run(stop)

	tc.Lock()
	shouldWait := tc.verifyEvents
	tc.Unlock()

	if shouldWait {
		// We need to wait for events to be broadcasted (sleep for longer than record.sleepDuration).
		timeoutTime := time.Now().Add(2 * time.Second)
		for now := time.Now(); timeoutTime.After(now); now = time.Now() {
			sleepUntil := timeoutTime.Sub(now)
			select {
			case <-tc.processed:
				// drain the chan of any sent events to keep it from filling before the timeout
			case <-time.After(sleepUntil):
				// timeout reached, ready to verifyResults
			}
		}
	} else {
		// Wait for HPA to be processed.
		<-tc.processed
	}
	tc.verifyResults(t)
}

func (tc *testCase) runTest(t *testing.T) {
	hpaController, informerFactory := tc.setupController(t)
	tc.runTestWithController(t, hpaController, informerFactory)
}

func TestScaleUp(t *testing.T) {
	tc := testCase{
		minReplicas:             2,
		maxReplicas:             6,
		specReplicas:            3,
		statusReplicas:          3,
		expectedDesiredReplicas: 5,
		CPUTarget:               30,
		verifyCPUCurrent:        true,
		reportedLevels:          []uint64{300, 500, 700},
		reportedCPURequests:     []resource.Quantity{resource.MustParse("1.0"), resource.MustParse("1.0"), resource.MustParse("1.0")},
		useMetricsAPI:           true,
	}
	tc.runTest(t)
}

func TestScaleUpContainer(t *testing.T) {
	tc := testCase{
		minReplicas:             2,
		maxReplicas:             6,
		specReplicas:            3,
		statusReplicas:          3,
		expectedDesiredReplicas: 5,
		metricsTarget: []autoscalingv2.MetricSpec{{
			Type: autoscalingv2.ContainerResourceMetricSourceType,
			ContainerResource: &autoscalingv2.ContainerResourceMetricSource{
				Name: v1.ResourceCPU,
				Target: autoscalingv2.MetricTarget{
					Type:               autoscalingv2.UtilizationMetricType,
					AverageUtilization: utilpointer.Int32Ptr(30),
				},
				Container: "container1",
			},
		}},
		reportedLevels:      []uint64{300, 500, 700},
		reportedCPURequests: []resource.Quantity{resource.MustParse("1.0"), resource.MustParse("1.0"), resource.MustParse("1.0")},
		useMetricsAPI:       true,
	}
	tc.runTest(t)
}

func TestScaleUpUnreadyLessScale(t *testing.T) {
	tc := testCase{
		minReplicas:             2,
		maxReplicas:             6,
		specReplicas:            3,
		statusReplicas:          3,
		expectedDesiredReplicas: 4,
		CPUTarget:               30,
		CPUCurrent:              60,
		verifyCPUCurrent:        true,
		reportedLevels:          []uint64{300, 500, 700},
		reportedCPURequests:     []resource.Quantity{resource.MustParse("1.0"), resource.MustParse("1.0"), resource.MustParse("1.0")},
		reportedPodReadiness:    []v1.ConditionStatus{v1.ConditionFalse, v1.ConditionTrue, v1.ConditionTrue},
		useMetricsAPI:           true,
	}
	tc.runTest(t)
}

func TestScaleUpHotCpuLessScale(t *testing.T) {
	tc := testCase{
		minReplicas:             2,
		maxReplicas:             6,
		specReplicas:            3,
		statusReplicas:          3,
		expectedDesiredReplicas: 4,
		CPUTarget:               30,
		CPUCurrent:              60,
		verifyCPUCurrent:        true,
		reportedLevels:          []uint64{300, 500, 700},
		reportedCPURequests:     []resource.Quantity{resource.MustParse("1.0"), resource.MustParse("1.0"), resource.MustParse("1.0")},
		reportedPodStartTime:    []metav1.Time{hotCPUCreationTime(), coolCPUCreationTime(), coolCPUCreationTime()},
		useMetricsAPI:           true,
	}
	tc.runTest(t)
}

func TestScaleUpUnreadyNoScale(t *testing.T) {
	tc := testCase{
		minReplicas:             2,
		maxReplicas:             6,
		specReplicas:            3,
		statusReplicas:          3,
		expectedDesiredReplicas: 3,
		CPUTarget:               30,
		CPUCurrent:              40,
		verifyCPUCurrent:        true,
		reportedLevels:          []uint64{400, 500, 700},
		reportedCPURequests:     []resource.Quantity{resource.MustParse("1.0"), resource.MustParse("1.0"), resource.MustParse("1.0")},
		reportedPodReadiness:    []v1.ConditionStatus{v1.ConditionTrue, v1.ConditionFalse, v1.ConditionFalse},
		useMetricsAPI:           true,
		expectedConditions: statusOkWithOverrides(autoscalingv2.HorizontalPodAutoscalerCondition{
			Type:   autoscalingv2.AbleToScale,
			Status: v1.ConditionTrue,
			Reason: "ReadyForNewScale",
		}),
	}
	tc.runTest(t)
}

func TestScaleUpHotCpuNoScale(t *testing.T) {
	tc := testCase{
		minReplicas:             2,
		maxReplicas:             6,
		specReplicas:            3,
		statusReplicas:          3,
		expectedDesiredReplicas: 3,
		CPUTarget:               30,
		CPUCurrent:              40,
		verifyCPUCurrent:        true,
		reportedLevels:          []uint64{400, 500, 700},
		reportedCPURequests:     []resource.Quantity{resource.MustParse("1.0"), resource.MustParse("1.0"), resource.MustParse("1.0")},
		reportedPodReadiness:    []v1.ConditionStatus{v1.ConditionTrue, v1.ConditionFalse, v1.ConditionFalse},
		reportedPodStartTime:    []metav1.Time{coolCPUCreationTime(), hotCPUCreationTime(), hotCPUCreationTime()},
		useMetricsAPI:           true,
		expectedConditions: statusOkWithOverrides(autoscalingv2.HorizontalPodAutoscalerCondition{
			Type:   autoscalingv2.AbleToScale,
			Status: v1.ConditionTrue,
			Reason: "ReadyForNewScale",
		}),
	}
	tc.runTest(t)
}

func TestScaleUpIgnoresFailedPods(t *testing.T) {
	tc := testCase{
		minReplicas:             2,
		maxReplicas:             6,
		specReplicas:            2,
		statusReplicas:          2,
		expectedDesiredReplicas: 4,
		CPUTarget:               30,
		CPUCurrent:              60,
		verifyCPUCurrent:        true,
		reportedLevels:          []uint64{500, 700},
		reportedCPURequests:     []resource.Quantity{resource.MustParse("1.0"), resource.MustParse("1.0"), resource.MustParse("1.0"), resource.MustParse("1.0")},
		reportedPodReadiness:    []v1.ConditionStatus{v1.ConditionTrue, v1.ConditionTrue, v1.ConditionFalse, v1.ConditionFalse},
		reportedPodPhase:        []v1.PodPhase{v1.PodRunning, v1.PodRunning, v1.PodFailed, v1.PodFailed},
		useMetricsAPI:           true,
	}
	tc.runTest(t)
}

func TestScaleUpIgnoresDeletionPods(t *testing.T) {
	tc := testCase{
		minReplicas:                  2,
		maxReplicas:                  6,
		specReplicas:                 2,
		statusReplicas:               2,
		expectedDesiredReplicas:      4,
		CPUTarget:                    30,
		CPUCurrent:                   60,
		verifyCPUCurrent:             true,
		reportedLevels:               []uint64{500, 700},
		reportedCPURequests:          []resource.Quantity{resource.MustParse("1.0"), resource.MustParse("1.0"), resource.MustParse("1.0"), resource.MustParse("1.0")},
		reportedPodReadiness:         []v1.ConditionStatus{v1.ConditionTrue, v1.ConditionTrue, v1.ConditionFalse, v1.ConditionFalse},
		reportedPodPhase:             []v1.PodPhase{v1.PodRunning, v1.PodRunning, v1.PodRunning, v1.PodRunning},
		reportedPodDeletionTimestamp: []bool{false, false, true, true},
		useMetricsAPI:                true,
	}
	tc.runTest(t)
}

func TestScaleUpDeployment(t *testing.T) {
	tc := testCase{
		minReplicas:             2,
		maxReplicas:             6,
		specReplicas:            3,
		statusReplicas:          3,
		expectedDesiredReplicas: 5,
		CPUTarget:               30,
		verifyCPUCurrent:        true,
		reportedLevels:          []uint64{300, 500, 700},
		reportedCPURequests:     []resource.Quantity{resource.MustParse("1.0"), resource.MustParse("1.0"), resource.MustParse("1.0")},
		useMetricsAPI:           true,
		resource: &fakeResource{
			name:       "test-dep",
			apiVersion: "apps/v1",
			kind:       "Deployment",
		},
	}
	tc.runTest(t)
}

func TestScaleUpReplicaSet(t *testing.T) {
	tc := testCase{
		minReplicas:             2,
		maxReplicas:             6,
		specReplicas:            3,
		statusReplicas:          3,
		expectedDesiredReplicas: 5,
		CPUTarget:               30,
		verifyCPUCurrent:        true,
		reportedLevels:          []uint64{300, 500, 700},
		reportedCPURequests:     []resource.Quantity{resource.MustParse("1.0"), resource.MustParse("1.0"), resource.MustParse("1.0")},
		useMetricsAPI:           true,
		resource: &fakeResource{
			name:       "test-replicaset",
			apiVersion: "apps/v1",
			kind:       "ReplicaSet",
		},
	}
	tc.runTest(t)
}

func TestScaleUpCM(t *testing.T) {
	averageValue := resource.MustParse("15.0")
	tc := testCase{
		minReplicas:             2,
		maxReplicas:             6,
		specReplicas:            3,
		statusReplicas:          3,
		expectedDesiredReplicas: 4,
		CPUTarget:               0,
		metricsTarget: []autoscalingv2.MetricSpec{
			{
				Type: autoscalingv2.PodsMetricSourceType,
				Pods: &autoscalingv2.PodsMetricSource{
					Metric: autoscalingv2.MetricIdentifier{
						Name: "qps",
					},
					Target: autoscalingv2.MetricTarget{
						AverageValue: &averageValue,
					},
				},
			},
		},
		reportedLevels:      []uint64{20000, 10000, 30000},
		reportedCPURequests: []resource.Quantity{resource.MustParse("1.0"), resource.MustParse("1.0"), resource.MustParse("1.0")},
	}
	tc.runTest(t)
}

func TestScaleUpCMUnreadyAndHotCpuNoLessScale(t *testing.T) {
	averageValue := resource.MustParse("15.0")
	tc := testCase{
		minReplicas:             2,
		maxReplicas:             6,
		specReplicas:            3,
		statusReplicas:          3,
		expectedDesiredReplicas: 6,
		CPUTarget:               0,
		metricsTarget: []autoscalingv2.MetricSpec{
			{
				Type: autoscalingv2.PodsMetricSourceType,
				Pods: &autoscalingv2.PodsMetricSource{
					Metric: autoscalingv2.MetricIdentifier{
						Name: "qps",
					},
					Target: autoscalingv2.MetricTarget{
						AverageValue: &averageValue,
					},
				},
			},
		},
		reportedLevels:       []uint64{50000, 10000, 30000},
		reportedPodReadiness: []v1.ConditionStatus{v1.ConditionTrue, v1.ConditionTrue, v1.ConditionFalse},
		reportedPodStartTime: []metav1.Time{coolCPUCreationTime(), coolCPUCreationTime(), hotCPUCreationTime()},
		reportedCPURequests:  []resource.Quantity{resource.MustParse("1.0"), resource.MustParse("1.0"), resource.MustParse("1.0")},
	}
	tc.runTest(t)
}

func TestScaleUpCMUnreadyandCpuHot(t *testing.T) {
	averageValue := resource.MustParse("15.0")
	tc := testCase{
		minReplicas:             2,
		maxReplicas:             6,
		specReplicas:            3,
		statusReplicas:          3,
		expectedDesiredReplicas: 6,
		CPUTarget:               0,
		metricsTarget: []autoscalingv2.MetricSpec{
			{
				Type: autoscalingv2.PodsMetricSourceType,
				Pods: &autoscalingv2.PodsMetricSource{
					Metric: autoscalingv2.MetricIdentifier{
						Name: "qps",
					},
					Target: autoscalingv2.MetricTarget{
						AverageValue: &averageValue,
					},
				},
			},
		},
		reportedLevels:       []uint64{50000, 15000, 30000},
		reportedPodReadiness: []v1.ConditionStatus{v1.ConditionFalse, v1.ConditionTrue, v1.ConditionFalse},
		reportedPodStartTime: []metav1.Time{hotCPUCreationTime(), coolCPUCreationTime(), hotCPUCreationTime()},
		reportedCPURequests:  []resource.Quantity{resource.MustParse("1.0"), resource.MustParse("1.0"), resource.MustParse("1.0")},
		expectedConditions: statusOkWithOverrides(autoscalingv2.HorizontalPodAutoscalerCondition{
			Type:   autoscalingv2.AbleToScale,
			Status: v1.ConditionTrue,
			Reason: "SucceededRescale",
		}, autoscalingv2.HorizontalPodAutoscalerCondition{
			Type:   autoscalingv2.ScalingLimited,
			Status: v1.ConditionTrue,
			Reason: "TooManyReplicas",
		}),
	}
	tc.runTest(t)
}

func TestScaleUpHotCpuNoScaleWouldScaleDown(t *testing.T) {
	averageValue := resource.MustParse("15.0")
	tc := testCase{
		minReplicas:             2,
		maxReplicas:             6,
		specReplicas:            3,
		statusReplicas:          3,
		expectedDesiredReplicas: 6,
		CPUTarget:               0,
		metricsTarget: []autoscalingv2.MetricSpec{
			{
				Type: autoscalingv2.PodsMetricSourceType,
				Pods: &autoscalingv2.PodsMetricSource{
					Metric: autoscalingv2.MetricIdentifier{
						Name: "qps",
					},
					Target: autoscalingv2.MetricTarget{
						AverageValue: &averageValue,
					},
				},
			},
		},
		reportedLevels:       []uint64{50000, 15000, 30000},
		reportedCPURequests:  []resource.Quantity{resource.MustParse("1.0"), resource.MustParse("1.0"), resource.MustParse("1.0")},
		reportedPodStartTime: []metav1.Time{hotCPUCreationTime(), coolCPUCreationTime(), hotCPUCreationTime()},
		expectedConditions: statusOkWithOverrides(autoscalingv2.HorizontalPodAutoscalerCondition{
			Type:   autoscalingv2.AbleToScale,
			Status: v1.ConditionTrue,
			Reason: "SucceededRescale",
		}, autoscalingv2.HorizontalPodAutoscalerCondition{
			Type:   autoscalingv2.ScalingLimited,
			Status: v1.ConditionTrue,
			Reason: "TooManyReplicas",
		}),
	}
	tc.runTest(t)
}

func TestScaleUpCMObject(t *testing.T) {
	targetValue := resource.MustParse("15.0")
	tc := testCase{
		minReplicas:             2,
		maxReplicas:             6,
		specReplicas:            3,
		statusReplicas:          3,
		expectedDesiredReplicas: 4,
		CPUTarget:               0,
		metricsTarget: []autoscalingv2.MetricSpec{
			{
				Type: autoscalingv2.ObjectMetricSourceType,
				Object: &autoscalingv2.ObjectMetricSource{
					DescribedObject: autoscalingv2.CrossVersionObjectReference{
						APIVersion: "apps/v1",
						Kind:       "Deployment",
						Name:       "some-deployment",
					},
					Metric: autoscalingv2.MetricIdentifier{
						Name: "qps",
					},
					Target: autoscalingv2.MetricTarget{
						Value: &targetValue,
					},
				},
			},
		},
		reportedLevels: []uint64{20000},
	}
	tc.runTest(t)
}

func TestScaleUpFromZeroCMObject(t *testing.T) {
	targetValue := resource.MustParse("15.0")
	tc := testCase{
		minReplicas:             0,
		maxReplicas:             6,
		specReplicas:            0,
		statusReplicas:          0,
		expectedDesiredReplicas: 2,
		CPUTarget:               0,
		metricsTarget: []autoscalingv2.MetricSpec{
			{
				Type: autoscalingv2.ObjectMetricSourceType,
				Object: &autoscalingv2.ObjectMetricSource{
					DescribedObject: autoscalingv2.CrossVersionObjectReference{
						APIVersion: "apps/v1",
						Kind:       "Deployment",
						Name:       "some-deployment",
					},
					Metric: autoscalingv2.MetricIdentifier{
						Name: "qps",
					},
					Target: autoscalingv2.MetricTarget{
						Value: &targetValue,
					},
				},
			},
		},
		reportedLevels: []uint64{20000},
	}
	tc.runTest(t)
}

func TestScaleUpFromZeroIgnoresToleranceCMObject(t *testing.T) {
	targetValue := resource.MustParse("1.0")
	tc := testCase{
		minReplicas:             0,
		maxReplicas:             6,
		specReplicas:            0,
		statusReplicas:          0,
		expectedDesiredReplicas: 1,
		CPUTarget:               0,
		metricsTarget: []autoscalingv2.MetricSpec{
			{
				Type: autoscalingv2.ObjectMetricSourceType,
				Object: &autoscalingv2.ObjectMetricSource{
					DescribedObject: autoscalingv2.CrossVersionObjectReference{
						APIVersion: "apps/v1",
						Kind:       "Deployment",
						Name:       "some-deployment",
					},
					Metric: autoscalingv2.MetricIdentifier{
						Name: "qps",
					},
					Target: autoscalingv2.MetricTarget{
						Value: &targetValue,
					},
				},
			},
		},
		reportedLevels: []uint64{1000},
	}
	tc.runTest(t)
}

func TestScaleUpPerPodCMObject(t *testing.T) {
	targetAverageValue := resource.MustParse("10.0")
	tc := testCase{
		minReplicas:             2,
		maxReplicas:             6,
		specReplicas:            3,
		statusReplicas:          3,
		expectedDesiredReplicas: 4,
		CPUTarget:               0,
		metricsTarget: []autoscalingv2.MetricSpec{
			{
				Type: autoscalingv2.ObjectMetricSourceType,
				Object: &autoscalingv2.ObjectMetricSource{
					DescribedObject: autoscalingv2.CrossVersionObjectReference{
						APIVersion: "apps/v1",
						Kind:       "Deployment",
						Name:       "some-deployment",
					},
					Metric: autoscalingv2.MetricIdentifier{
						Name: "qps",
					},
					Target: autoscalingv2.MetricTarget{
						AverageValue: &targetAverageValue,
					},
				},
			},
		},
		reportedLevels: []uint64{40000},
	}
	tc.runTest(t)
}

func TestScaleUpCMExternal(t *testing.T) {
	tc := testCase{
		minReplicas:             2,
		maxReplicas:             6,
		specReplicas:            3,
		statusReplicas:          3,
		expectedDesiredReplicas: 4,
		metricsTarget: []autoscalingv2.MetricSpec{
			{
				Type: autoscalingv2.ExternalMetricSourceType,
				External: &autoscalingv2.ExternalMetricSource{
					Metric: autoscalingv2.MetricIdentifier{
						Name:     "qps",
						Selector: &metav1.LabelSelector{},
					},
					Target: autoscalingv2.MetricTarget{
						Value: resource.NewMilliQuantity(6666, resource.DecimalSI),
					},
				},
			},
		},
		reportedLevels: []uint64{8600},
	}
	tc.runTest(t)
}

func TestScaleUpPerPodCMExternal(t *testing.T) {
	tc := testCase{
		minReplicas:             2,
		maxReplicas:             6,
		specReplicas:            3,
		statusReplicas:          3,
		expectedDesiredReplicas: 4,
		metricsTarget: []autoscalingv2.MetricSpec{
			{
				Type: autoscalingv2.ExternalMetricSourceType,
				External: &autoscalingv2.ExternalMetricSource{
					Metric: autoscalingv2.MetricIdentifier{
						Name:     "qps",
						Selector: &metav1.LabelSelector{},
					},
					Target: autoscalingv2.MetricTarget{
						AverageValue: resource.NewMilliQuantity(2222, resource.DecimalSI),
					},
				},
			},
		},
		reportedLevels: []uint64{8600},
	}
	tc.runTest(t)
}

func TestScaleDown(t *testing.T) {
	tc := testCase{
		minReplicas:             2,
		maxReplicas:             6,
		specReplicas:            5,
		statusReplicas:          5,
		expectedDesiredReplicas: 3,
		CPUTarget:               50,
		verifyCPUCurrent:        true,
		reportedLevels:          []uint64{100, 300, 500, 250, 250},
		reportedCPURequests:     []resource.Quantity{resource.MustParse("1.0"), resource.MustParse("1.0"), resource.MustParse("1.0"), resource.MustParse("1.0"), resource.MustParse("1.0")},
		useMetricsAPI:           true,
		recommendations:         []timestampedRecommendation{},
	}
	tc.runTest(t)
}

func TestScaleDownContainerResource(t *testing.T) {
	tc := testCase{
		minReplicas:             2,
		maxReplicas:             6,
		specReplicas:            5,
		statusReplicas:          5,
		expectedDesiredReplicas: 3,
		reportedLevels:          []uint64{100, 300, 500, 250, 250},
		reportedCPURequests:     []resource.Quantity{resource.MustParse("1.0"), resource.MustParse("1.0"), resource.MustParse("1.0"), resource.MustParse("1.0"), resource.MustParse("1.0")},
		metricsTarget: []autoscalingv2.MetricSpec{{
			Type: autoscalingv2.ContainerResourceMetricSourceType,
			ContainerResource: &autoscalingv2.ContainerResourceMetricSource{
				Container: "container2",
				Name:      v1.ResourceCPU,
				Target: autoscalingv2.MetricTarget{
					Type:               autoscalingv2.UtilizationMetricType,
					AverageUtilization: utilpointer.Int32Ptr(50),
				},
			},
		}},
		useMetricsAPI:   true,
		recommendations: []timestampedRecommendation{},
	}
	tc.runTest(t)
}

func TestScaleDownWithScalingRules(t *testing.T) {
	tc := testCase{
		minReplicas:             2,
		maxReplicas:             6,
		scaleUpRules:            generateScalingRules(0, 0, 100, 15, 30),
		specReplicas:            5,
		statusReplicas:          5,
		expectedDesiredReplicas: 3,
		CPUTarget:               50,
		verifyCPUCurrent:        true,
		reportedLevels:          []uint64{100, 300, 500, 250, 250},
		reportedCPURequests:     []resource.Quantity{resource.MustParse("1.0"), resource.MustParse("1.0"), resource.MustParse("1.0"), resource.MustParse("1.0"), resource.MustParse("1.0")},
		useMetricsAPI:           true,
		recommendations:         []timestampedRecommendation{},
	}
	tc.runTest(t)
}

func TestScaleUpOneMetricInvalid(t *testing.T) {
	tc := testCase{
		minReplicas:             2,
		maxReplicas:             6,
		specReplicas:            3,
		statusReplicas:          3,
		expectedDesiredReplicas: 4,
		CPUTarget:               30,
		verifyCPUCurrent:        true,
		metricsTarget: []autoscalingv2.MetricSpec{
			{
				Type: "CheddarCheese",
			},
		},
		reportedLevels:      []uint64{300, 400, 500},
		reportedCPURequests: []resource.Quantity{resource.MustParse("1.0"), resource.MustParse("1.0"), resource.MustParse("1.0")},
	}
	tc.runTest(t)
}

func TestScaleUpFromZeroOneMetricInvalid(t *testing.T) {
	tc := testCase{
		minReplicas:             0,
		maxReplicas:             6,
		specReplicas:            0,
		statusReplicas:          0,
		expectedDesiredReplicas: 4,
		CPUTarget:               30,
		verifyCPUCurrent:        true,
		metricsTarget: []autoscalingv2.MetricSpec{
			{
				Type: "CheddarCheese",
			},
		},
		reportedLevels:      []uint64{300, 400, 500},
		reportedCPURequests: []resource.Quantity{resource.MustParse("1.0"), resource.MustParse("1.0"), resource.MustParse("1.0")},
		recommendations:     []timestampedRecommendation{},
	}
	tc.runTest(t)
}

func TestScaleUpBothMetricsEmpty(t *testing.T) { // Switch to missing
	tc := testCase{
		minReplicas:             2,
		maxReplicas:             6,
		specReplicas:            3,
		statusReplicas:          3,
		expectedDesiredReplicas: 3,
		CPUTarget:               0,
		metricsTarget: []autoscalingv2.MetricSpec{
			{
				Type: "CheddarCheese",
			},
		},
		reportedLevels:      []uint64{},
		reportedCPURequests: []resource.Quantity{resource.MustParse("1.0"), resource.MustParse("1.0"), resource.MustParse("1.0")},
		expectedConditions: []autoscalingv1.HorizontalPodAutoscalerCondition{
			{Type: autoscalingv1.AbleToScale, Status: v1.ConditionTrue, Reason: "SucceededGetScale"},
			{Type: autoscalingv1.ScalingActive, Status: v1.ConditionFalse, Reason: "InvalidMetricSourceType"},
		},
	}
	tc.runTest(t)
}

func TestScaleDownStabilizeInitialSize(t *testing.T) {
	tc := testCase{
		minReplicas:             2,
		maxReplicas:             6,
		specReplicas:            5,
		statusReplicas:          5,
		expectedDesiredReplicas: 5,
		CPUTarget:               50,
		verifyCPUCurrent:        true,
		reportedLevels:          []uint64{100, 300, 500, 250, 250},
		reportedCPURequests:     []resource.Quantity{resource.MustParse("1.0"), resource.MustParse("1.0"), resource.MustParse("1.0"), resource.MustParse("1.0"), resource.MustParse("1.0")},
		useMetricsAPI:           true,
		recommendations:         nil,
		expectedConditions: statusOkWithOverrides(autoscalingv2.HorizontalPodAutoscalerCondition{
			Type:   autoscalingv2.AbleToScale,
			Status: v1.ConditionTrue,
			Reason: "ReadyForNewScale",
		}, autoscalingv2.HorizontalPodAutoscalerCondition{
			Type:   autoscalingv2.AbleToScale,
			Status: v1.ConditionTrue,
			Reason: "ScaleDownStabilized",
		}),
	}
	tc.runTest(t)
}

func TestScaleDownCM(t *testing.T) {
	averageValue := resource.MustParse("20.0")
	tc := testCase{
		minReplicas:             2,
		maxReplicas:             6,
		specReplicas:            5,
		statusReplicas:          5,
		expectedDesiredReplicas: 3,
		CPUTarget:               0,
		metricsTarget: []autoscalingv2.MetricSpec{
			{
				Type: autoscalingv2.PodsMetricSourceType,
				Pods: &autoscalingv2.PodsMetricSource{
					Metric: autoscalingv2.MetricIdentifier{
						Name: "qps",
					},
					Target: autoscalingv2.MetricTarget{
						AverageValue: &averageValue,
					},
				},
			},
		},
		reportedLevels:      []uint64{12000, 12000, 12000, 12000, 12000},
		reportedCPURequests: []resource.Quantity{resource.MustParse("1.0"), resource.MustParse("1.0"), resource.MustParse("1.0"), resource.MustParse("1.0"), resource.MustParse("1.0")},
		recommendations:     []timestampedRecommendation{},
	}
	tc.runTest(t)
}

func TestScaleDownCMObject(t *testing.T) {
	targetValue := resource.MustParse("20.0")
	tc := testCase{
		minReplicas:             2,
		maxReplicas:             6,
		specReplicas:            5,
		statusReplicas:          5,
		expectedDesiredReplicas: 3,
		CPUTarget:               0,
		metricsTarget: []autoscalingv2.MetricSpec{
			{
				Type: autoscalingv2.ObjectMetricSourceType,
				Object: &autoscalingv2.ObjectMetricSource{
					DescribedObject: autoscalingv2.CrossVersionObjectReference{
						APIVersion: "apps/v1",
						Kind:       "Deployment",
						Name:       "some-deployment",
					},
					Metric: autoscalingv2.MetricIdentifier{
						Name: "qps",
					},
					Target: autoscalingv2.MetricTarget{
						Value: &targetValue,
					},
				},
			},
		},
		reportedLevels:      []uint64{12000},
		reportedCPURequests: []resource.Quantity{resource.MustParse("1.0"), resource.MustParse("1.0"), resource.MustParse("1.0"), resource.MustParse("1.0"), resource.MustParse("1.0")},
		recommendations:     []timestampedRecommendation{},
	}
	tc.runTest(t)
}

func TestScaleDownToZeroCMObject(t *testing.T) {
	targetValue := resource.MustParse("20.0")
	tc := testCase{
		minReplicas:             0,
		maxReplicas:             6,
		specReplicas:            5,
		statusReplicas:          5,
		expectedDesiredReplicas: 0,
		CPUTarget:               0,
		metricsTarget: []autoscalingv2.MetricSpec{
			{
				Type: autoscalingv2.ObjectMetricSourceType,
				Object: &autoscalingv2.ObjectMetricSource{
					DescribedObject: autoscalingv2.CrossVersionObjectReference{
						APIVersion: "apps/v1",
						Kind:       "Deployment",
						Name:       "some-deployment",
					},
					Metric: autoscalingv2.MetricIdentifier{
						Name: "qps",
					},
					Target: autoscalingv2.MetricTarget{
						Value: &targetValue,
					},
				},
			},
		},
		reportedLevels:      []uint64{0},
		reportedCPURequests: []resource.Quantity{resource.MustParse("1.0"), resource.MustParse("1.0"), resource.MustParse("1.0"), resource.MustParse("1.0"), resource.MustParse("1.0")},
		recommendations:     []timestampedRecommendation{},
	}
	tc.runTest(t)
}

func TestScaleDownPerPodCMObject(t *testing.T) {
	targetAverageValue := resource.MustParse("20.0")
	tc := testCase{
		minReplicas:             2,
		maxReplicas:             6,
		specReplicas:            5,
		statusReplicas:          5,
		expectedDesiredReplicas: 3,
		CPUTarget:               0,
		metricsTarget: []autoscalingv2.MetricSpec{
			{
				Type: autoscalingv2.ObjectMetricSourceType,
				Object: &autoscalingv2.ObjectMetricSource{
					DescribedObject: autoscalingv2.CrossVersionObjectReference{
						APIVersion: "apps/v1",
						Kind:       "Deployment",
						Name:       "some-deployment",
					},
					Metric: autoscalingv2.MetricIdentifier{
						Name: "qps",
					},
					Target: autoscalingv2.MetricTarget{
						AverageValue: &targetAverageValue,
					},
				},
			},
		},
		reportedLevels:      []uint64{60000},
		reportedCPURequests: []resource.Quantity{resource.MustParse("1.0"), resource.MustParse("1.0"), resource.MustParse("1.0"), resource.MustParse("1.0"), resource.MustParse("1.0")},
		recommendations:     []timestampedRecommendation{},
	}
	tc.runTest(t)
}

func TestScaleDownCMExternal(t *testing.T) {
	tc := testCase{
		minReplicas:             2,
		maxReplicas:             6,
		specReplicas:            5,
		statusReplicas:          5,
		expectedDesiredReplicas: 3,
		metricsTarget: []autoscalingv2.MetricSpec{
			{
				Type: autoscalingv2.ExternalMetricSourceType,
				External: &autoscalingv2.ExternalMetricSource{
					Metric: autoscalingv2.MetricIdentifier{
						Name:     "qps",
						Selector: &metav1.LabelSelector{},
					},
					Target: autoscalingv2.MetricTarget{
						Value: resource.NewMilliQuantity(14400, resource.DecimalSI),
					},
				},
			},
		},
		reportedLevels:  []uint64{8600},
		recommendations: []timestampedRecommendation{},
	}
	tc.runTest(t)
}

func TestScaleDownToZeroCMExternal(t *testing.T) {
	tc := testCase{
		minReplicas:             0,
		maxReplicas:             6,
		specReplicas:            5,
		statusReplicas:          5,
		expectedDesiredReplicas: 0,
		metricsTarget: []autoscalingv2.MetricSpec{
			{
				Type: autoscalingv2.ExternalMetricSourceType,
				External: &autoscalingv2.ExternalMetricSource{
					Metric: autoscalingv2.MetricIdentifier{
						Name:     "qps",
						Selector: &metav1.LabelSelector{},
					},
					Target: autoscalingv2.MetricTarget{
						Value: resource.NewMilliQuantity(14400, resource.DecimalSI),
					},
				},
			},
		},
		reportedLevels:  []uint64{0},
		recommendations: []timestampedRecommendation{},
	}
	tc.runTest(t)
}

func TestScaleDownPerPodCMExternal(t *testing.T) {
	tc := testCase{
		minReplicas:             2,
		maxReplicas:             6,
		specReplicas:            5,
		statusReplicas:          5,
		expectedDesiredReplicas: 3,
		metricsTarget: []autoscalingv2.MetricSpec{
			{
				Type: autoscalingv2.ExternalMetricSourceType,
				External: &autoscalingv2.ExternalMetricSource{
					Metric: autoscalingv2.MetricIdentifier{
						Name:     "qps",
						Selector: &metav1.LabelSelector{},
					},
					Target: autoscalingv2.MetricTarget{
						AverageValue: resource.NewMilliQuantity(3000, resource.DecimalSI),
					},
				},
			},
		},
		reportedLevels:  []uint64{8600},
		recommendations: []timestampedRecommendation{},
	}
	tc.runTest(t)
}

func TestScaleDownIncludeUnreadyPods(t *testing.T) {
	tc := testCase{
		minReplicas:             2,
		maxReplicas:             6,
		specReplicas:            5,
		statusReplicas:          5,
		expectedDesiredReplicas: 2,
		CPUTarget:               50,
		CPUCurrent:              30,
		verifyCPUCurrent:        true,
		reportedLevels:          []uint64{100, 300, 500, 250, 250},
		reportedCPURequests:     []resource.Quantity{resource.MustParse("1.0"), resource.MustParse("1.0"), resource.MustParse("1.0"), resource.MustParse("1.0"), resource.MustParse("1.0")},
		useMetricsAPI:           true,
		reportedPodReadiness:    []v1.ConditionStatus{v1.ConditionTrue, v1.ConditionTrue, v1.ConditionTrue, v1.ConditionFalse, v1.ConditionFalse},
		recommendations:         []timestampedRecommendation{},
	}
	tc.runTest(t)
}

func TestScaleDownOneMetricInvalid(t *testing.T) {
	tc := testCase{
		minReplicas:             2,
		maxReplicas:             6,
		specReplicas:            5,
		statusReplicas:          5,
		expectedDesiredReplicas: 3,
		CPUTarget:               50,
		metricsTarget: []autoscalingv2.MetricSpec{
			{
				Type: "CheddarCheese",
			},
		},
		reportedLevels:      []uint64{100, 300, 500, 250, 250},
		reportedCPURequests: []resource.Quantity{resource.MustParse("1.0"), resource.MustParse("1.0"), resource.MustParse("1.0"), resource.MustParse("1.0"), resource.MustParse("1.0")},
		useMetricsAPI:       true,
		recommendations:     []timestampedRecommendation{},
	}

	tc.runTest(t)
}

func TestScaleDownOneMetricEmpty(t *testing.T) {
	tc := testCase{
		minReplicas:             2,
		maxReplicas:             6,
		specReplicas:            5,
		statusReplicas:          5,
		expectedDesiredReplicas: 3,
		CPUTarget:               50,
		metricsTarget: []autoscalingv2.MetricSpec{
			{
				Type: autoscalingv2.ExternalMetricSourceType,
				External: &autoscalingv2.ExternalMetricSource{
					Metric: autoscalingv2.MetricIdentifier{
						Name:     "qps",
						Selector: &metav1.LabelSelector{},
					},
					Target: autoscalingv2.MetricTarget{
						Type:         autoscalingv2.AverageValueMetricType,
						AverageValue: resource.NewMilliQuantity(1000, resource.DecimalSI),
					},
				},
			},
		},
		reportedLevels:      []uint64{100, 300, 500, 250, 250},
		reportedCPURequests: []resource.Quantity{resource.MustParse("1.0"), resource.MustParse("1.0"), resource.MustParse("1.0"), resource.MustParse("1.0"), resource.MustParse("1.0")},
		useMetricsAPI:       true,
		recommendations:     []timestampedRecommendation{},
	}
	_, _, _, testEMClient, _ := tc.prepareTestClient(t)
	testEMClient.PrependReactor("list", "*", func(action core.Action) (handled bool, ret runtime.Object, err error) {
		return true, &emapi.ExternalMetricValueList{}, fmt.Errorf("something went wrong")
	})
	tc.testEMClient = testEMClient
	tc.runTest(t)
}

func TestScaleDownIgnoreHotCpuPods(t *testing.T) {
	tc := testCase{
		minReplicas:             2,
		maxReplicas:             6,
		specReplicas:            5,
		statusReplicas:          5,
		expectedDesiredReplicas: 2,
		CPUTarget:               50,
		CPUCurrent:              30,
		verifyCPUCurrent:        true,
		reportedLevels:          []uint64{100, 300, 500, 250, 250},
		reportedCPURequests:     []resource.Quantity{resource.MustParse("1.0"), resource.MustParse("1.0"), resource.MustParse("1.0"), resource.MustParse("1.0"), resource.MustParse("1.0")},
		useMetricsAPI:           true,
		reportedPodStartTime:    []metav1.Time{coolCPUCreationTime(), coolCPUCreationTime(), coolCPUCreationTime(), hotCPUCreationTime(), hotCPUCreationTime()},
		recommendations:         []timestampedRecommendation{},
	}
	tc.runTest(t)
}

func TestScaleDownIgnoresFailedPods(t *testing.T) {
	tc := testCase{
		minReplicas:             2,
		maxReplicas:             6,
		specReplicas:            5,
		statusReplicas:          5,
		expectedDesiredReplicas: 3,
		CPUTarget:               50,
		CPUCurrent:              28,
		verifyCPUCurrent:        true,
		reportedLevels:          []uint64{100, 300, 500, 250, 250},
		reportedCPURequests:     []resource.Quantity{resource.MustParse("1.0"), resource.MustParse("1.0"), resource.MustParse("1.0"), resource.MustParse("1.0"), resource.MustParse("1.0"), resource.MustParse("1.0"), resource.MustParse("1.0")},
		useMetricsAPI:           true,
		reportedPodReadiness:    []v1.ConditionStatus{v1.ConditionTrue, v1.ConditionTrue, v1.ConditionTrue, v1.ConditionTrue, v1.ConditionTrue, v1.ConditionFalse, v1.ConditionFalse},
		reportedPodPhase:        []v1.PodPhase{v1.PodRunning, v1.PodRunning, v1.PodRunning, v1.PodRunning, v1.PodRunning, v1.PodFailed, v1.PodFailed},
		recommendations:         []timestampedRecommendation{},
	}
	tc.runTest(t)
}

func TestScaleDownIgnoresDeletionPods(t *testing.T) {
	tc := testCase{
		minReplicas:                  2,
		maxReplicas:                  6,
		specReplicas:                 5,
		statusReplicas:               5,
		expectedDesiredReplicas:      3,
		CPUTarget:                    50,
		CPUCurrent:                   28,
		verifyCPUCurrent:             true,
		reportedLevels:               []uint64{100, 300, 500, 250, 250},
		reportedCPURequests:          []resource.Quantity{resource.MustParse("1.0"), resource.MustParse("1.0"), resource.MustParse("1.0"), resource.MustParse("1.0"), resource.MustParse("1.0"), resource.MustParse("1.0"), resource.MustParse("1.0")},
		useMetricsAPI:                true,
		reportedPodReadiness:         []v1.ConditionStatus{v1.ConditionTrue, v1.ConditionTrue, v1.ConditionTrue, v1.ConditionTrue, v1.ConditionTrue, v1.ConditionFalse, v1.ConditionFalse},
		reportedPodPhase:             []v1.PodPhase{v1.PodRunning, v1.PodRunning, v1.PodRunning, v1.PodRunning, v1.PodRunning, v1.PodRunning, v1.PodRunning},
		reportedPodDeletionTimestamp: []bool{false, false, false, false, false, true, true},
		recommendations:              []timestampedRecommendation{},
	}
	tc.runTest(t)
}

func TestTolerance(t *testing.T) {
	tc := testCase{
		minReplicas:             1,
		maxReplicas:             5,
		specReplicas:            3,
		statusReplicas:          3,
		expectedDesiredReplicas: 3,
		CPUTarget:               100,
		reportedLevels:          []uint64{1010, 1030, 1020},
		reportedCPURequests:     []resource.Quantity{resource.MustParse("0.9"), resource.MustParse("1.0"), resource.MustParse("1.1")},
		useMetricsAPI:           true,
		expectedConditions: statusOkWithOverrides(autoscalingv2.HorizontalPodAutoscalerCondition{
			Type:   autoscalingv2.AbleToScale,
			Status: v1.ConditionTrue,
			Reason: "ReadyForNewScale",
		}),
	}
	tc.runTest(t)
}

func TestToleranceCM(t *testing.T) {
	averageValue := resource.MustParse("20.0")
	tc := testCase{
		minReplicas:             1,
		maxReplicas:             5,
		specReplicas:            3,
		statusReplicas:          3,
		expectedDesiredReplicas: 3,
		metricsTarget: []autoscalingv2.MetricSpec{
			{
				Type: autoscalingv2.PodsMetricSourceType,
				Pods: &autoscalingv2.PodsMetricSource{
					Metric: autoscalingv2.MetricIdentifier{
						Name: "qps",
					},
					Target: autoscalingv2.MetricTarget{
						AverageValue: &averageValue,
					},
				},
			},
		},
		reportedLevels:      []uint64{20000, 20001, 21000},
		reportedCPURequests: []resource.Quantity{resource.MustParse("0.9"), resource.MustParse("1.0"), resource.MustParse("1.1")},
		expectedConditions: statusOkWithOverrides(autoscalingv2.HorizontalPodAutoscalerCondition{
			Type:   autoscalingv2.AbleToScale,
			Status: v1.ConditionTrue,
			Reason: "ReadyForNewScale",
		}),
	}
	tc.runTest(t)
}

func TestToleranceCMObject(t *testing.T) {
	targetValue := resource.MustParse("20.0")
	tc := testCase{
		minReplicas:             1,
		maxReplicas:             5,
		specReplicas:            3,
		statusReplicas:          3,
		expectedDesiredReplicas: 3,
		metricsTarget: []autoscalingv2.MetricSpec{
			{
				Type: autoscalingv2.ObjectMetricSourceType,
				Object: &autoscalingv2.ObjectMetricSource{
					DescribedObject: autoscalingv2.CrossVersionObjectReference{
						APIVersion: "apps/v1",
						Kind:       "Deployment",
						Name:       "some-deployment",
					},
					Metric: autoscalingv2.MetricIdentifier{
						Name: "qps",
					},
					Target: autoscalingv2.MetricTarget{
						Value: &targetValue,
					},
				},
			},
		},
		reportedLevels:      []uint64{20050},
		reportedCPURequests: []resource.Quantity{resource.MustParse("0.9"), resource.MustParse("1.0"), resource.MustParse("1.1")},
		expectedConditions: statusOkWithOverrides(autoscalingv2.HorizontalPodAutoscalerCondition{
			Type:   autoscalingv2.AbleToScale,
			Status: v1.ConditionTrue,
			Reason: "ReadyForNewScale",
		}),
	}
	tc.runTest(t)
}

func TestToleranceCMExternal(t *testing.T) {
	tc := testCase{
		minReplicas:             2,
		maxReplicas:             6,
		specReplicas:            4,
		statusReplicas:          4,
		expectedDesiredReplicas: 4,
		metricsTarget: []autoscalingv2.MetricSpec{
			{
				Type: autoscalingv2.ExternalMetricSourceType,
				External: &autoscalingv2.ExternalMetricSource{
					Metric: autoscalingv2.MetricIdentifier{
						Name:     "qps",
						Selector: &metav1.LabelSelector{},
					},
					Target: autoscalingv2.MetricTarget{
						Value: resource.NewMilliQuantity(8666, resource.DecimalSI),
					},
				},
			},
		},
		reportedLevels: []uint64{8600},
		expectedConditions: statusOkWithOverrides(autoscalingv2.HorizontalPodAutoscalerCondition{
			Type:   autoscalingv2.AbleToScale,
			Status: v1.ConditionTrue,
			Reason: "ReadyForNewScale",
		}),
	}
	tc.runTest(t)
}

func TestTolerancePerPodCMObject(t *testing.T) {
	tc := testCase{
		minReplicas:             2,
		maxReplicas:             6,
		specReplicas:            4,
		statusReplicas:          4,
		expectedDesiredReplicas: 4,
		metricsTarget: []autoscalingv2.MetricSpec{
			{
				Type: autoscalingv2.ObjectMetricSourceType,
				Object: &autoscalingv2.ObjectMetricSource{
					DescribedObject: autoscalingv2.CrossVersionObjectReference{
						APIVersion: "apps/v1",
						Kind:       "Deployment",
						Name:       "some-deployment",
					},
					Metric: autoscalingv2.MetricIdentifier{
						Name:     "qps",
						Selector: &metav1.LabelSelector{},
					},
					Target: autoscalingv2.MetricTarget{
						AverageValue: resource.NewMilliQuantity(2200, resource.DecimalSI),
					},
				},
			},
		},
		reportedLevels: []uint64{8600},
		expectedConditions: statusOkWithOverrides(autoscalingv2.HorizontalPodAutoscalerCondition{
			Type:   autoscalingv2.AbleToScale,
			Status: v1.ConditionTrue,
			Reason: "ReadyForNewScale",
		}),
	}
	tc.runTest(t)
}

func TestTolerancePerPodCMExternal(t *testing.T) {
	tc := testCase{
		minReplicas:             2,
		maxReplicas:             6,
		specReplicas:            4,
		statusReplicas:          4,
		expectedDesiredReplicas: 4,
		metricsTarget: []autoscalingv2.MetricSpec{
			{
				Type: autoscalingv2.ExternalMetricSourceType,
				External: &autoscalingv2.ExternalMetricSource{
					Metric: autoscalingv2.MetricIdentifier{
						Name:     "qps",
						Selector: &metav1.LabelSelector{},
					},
					Target: autoscalingv2.MetricTarget{
						AverageValue: resource.NewMilliQuantity(2200, resource.DecimalSI),
					},
				},
			},
		},
		reportedLevels: []uint64{8600},
		expectedConditions: statusOkWithOverrides(autoscalingv2.HorizontalPodAutoscalerCondition{
			Type:   autoscalingv2.AbleToScale,
			Status: v1.ConditionTrue,
			Reason: "ReadyForNewScale",
		}),
	}
	tc.runTest(t)
}

func TestMinReplicas(t *testing.T) {
	tc := testCase{
		minReplicas:             2,
		maxReplicas:             5,
		specReplicas:            3,
		statusReplicas:          3,
		expectedDesiredReplicas: 2,
		CPUTarget:               90,
		reportedLevels:          []uint64{10, 95, 10},
		reportedCPURequests:     []resource.Quantity{resource.MustParse("0.9"), resource.MustParse("1.0"), resource.MustParse("1.1")},
		useMetricsAPI:           true,
		expectedConditions: statusOkWithOverrides(autoscalingv2.HorizontalPodAutoscalerCondition{
			Type:   autoscalingv2.ScalingLimited,
			Status: v1.ConditionTrue,
			Reason: "TooFewReplicas",
		}),
		recommendations: []timestampedRecommendation{},
	}
	tc.runTest(t)
}

func TestZeroMinReplicasDesiredZero(t *testing.T) {
	tc := testCase{
		minReplicas:             0,
		maxReplicas:             5,
		specReplicas:            3,
		statusReplicas:          3,
		expectedDesiredReplicas: 0,
		CPUTarget:               90,
		reportedLevels:          []uint64{0, 0, 0},
		reportedCPURequests:     []resource.Quantity{resource.MustParse("0.9"), resource.MustParse("1.0"), resource.MustParse("1.1")},
		useMetricsAPI:           true,
		expectedConditions: statusOkWithOverrides(autoscalingv2.HorizontalPodAutoscalerCondition{
			Type:   autoscalingv2.ScalingLimited,
			Status: v1.ConditionFalse,
			Reason: "DesiredWithinRange",
		}),
		recommendations: []timestampedRecommendation{},
	}
	tc.runTest(t)
}

func TestMinReplicasDesiredZero(t *testing.T) {
	tc := testCase{
		minReplicas:             2,
		maxReplicas:             5,
		specReplicas:            3,
		statusReplicas:          3,
		expectedDesiredReplicas: 2,
		CPUTarget:               90,
		reportedLevels:          []uint64{0, 0, 0},
		reportedCPURequests:     []resource.Quantity{resource.MustParse("0.9"), resource.MustParse("1.0"), resource.MustParse("1.1")},
		useMetricsAPI:           true,
		expectedConditions: statusOkWithOverrides(autoscalingv2.HorizontalPodAutoscalerCondition{
			Type:   autoscalingv2.ScalingLimited,
			Status: v1.ConditionTrue,
			Reason: "TooFewReplicas",
		}),
		recommendations: []timestampedRecommendation{},
	}
	tc.runTest(t)
}

func TestZeroReplicas(t *testing.T) {
	tc := testCase{
		minReplicas:             3,
		maxReplicas:             5,
		specReplicas:            0,
		statusReplicas:          0,
		expectedDesiredReplicas: 0,
		CPUTarget:               90,
		reportedLevels:          []uint64{},
		reportedCPURequests:     []resource.Quantity{},
		useMetricsAPI:           true,
		expectedConditions: []autoscalingv1.HorizontalPodAutoscalerCondition{
			{Type: autoscalingv1.AbleToScale, Status: v1.ConditionTrue, Reason: "SucceededGetScale"},
			{Type: autoscalingv1.ScalingActive, Status: v1.ConditionFalse, Reason: "ScalingDisabled"},
		},
	}
	tc.runTest(t)
}

func TestTooFewReplicas(t *testing.T) {
	tc := testCase{
		minReplicas:             3,
		maxReplicas:             5,
		specReplicas:            2,
		statusReplicas:          2,
		expectedDesiredReplicas: 3,
		CPUTarget:               90,
		reportedLevels:          []uint64{},
		reportedCPURequests:     []resource.Quantity{},
		useMetricsAPI:           true,
		expectedConditions: []autoscalingv1.HorizontalPodAutoscalerCondition{
			{Type: autoscalingv1.AbleToScale, Status: v1.ConditionTrue, Reason: "SucceededRescale"},
		},
	}
	tc.runTest(t)
}

func TestTooManyReplicas(t *testing.T) {
	tc := testCase{
		minReplicas:             3,
		maxReplicas:             5,
		specReplicas:            10,
		statusReplicas:          10,
		expectedDesiredReplicas: 5,
		CPUTarget:               90,
		reportedLevels:          []uint64{},
		reportedCPURequests:     []resource.Quantity{},
		useMetricsAPI:           true,
		expectedConditions: []autoscalingv1.HorizontalPodAutoscalerCondition{
			{Type: autoscalingv1.AbleToScale, Status: v1.ConditionTrue, Reason: "SucceededRescale"},
		},
	}
	tc.runTest(t)
}

func TestMaxReplicas(t *testing.T) {
	tc := testCase{
		minReplicas:             2,
		maxReplicas:             5,
		specReplicas:            3,
		statusReplicas:          3,
		expectedDesiredReplicas: 5,
		CPUTarget:               90,
		reportedLevels:          []uint64{8000, 9500, 1000},
		reportedCPURequests:     []resource.Quantity{resource.MustParse("0.9"), resource.MustParse("1.0"), resource.MustParse("1.1")},
		useMetricsAPI:           true,
		expectedConditions: statusOkWithOverrides(autoscalingv2.HorizontalPodAutoscalerCondition{
			Type:   autoscalingv2.ScalingLimited,
			Status: v1.ConditionTrue,
			Reason: "TooManyReplicas",
		}),
	}
	tc.runTest(t)
}

func TestSuperfluousMetrics(t *testing.T) {
	tc := testCase{
		minReplicas:             2,
		maxReplicas:             6,
		specReplicas:            4,
		statusReplicas:          4,
		expectedDesiredReplicas: 6,
		CPUTarget:               100,
		reportedLevels:          []uint64{4000, 9500, 3000, 7000, 3200, 2000},
		reportedCPURequests:     []resource.Quantity{resource.MustParse("1.0"), resource.MustParse("1.0"), resource.MustParse("1.0"), resource.MustParse("1.0")},
		useMetricsAPI:           true,
		expectedConditions: statusOkWithOverrides(autoscalingv2.HorizontalPodAutoscalerCondition{
			Type:   autoscalingv2.ScalingLimited,
			Status: v1.ConditionTrue,
			Reason: "TooManyReplicas",
		}),
	}
	tc.runTest(t)
}

func TestMissingMetrics(t *testing.T) {
	tc := testCase{
		minReplicas:             2,
		maxReplicas:             6,
		specReplicas:            4,
		statusReplicas:          4,
		expectedDesiredReplicas: 3,
		CPUTarget:               100,
		reportedLevels:          []uint64{400, 95},
		reportedCPURequests:     []resource.Quantity{resource.MustParse("1.0"), resource.MustParse("1.0"), resource.MustParse("1.0"), resource.MustParse("1.0")},
		useMetricsAPI:           true,
		recommendations:         []timestampedRecommendation{},
	}
	tc.runTest(t)
}

func TestEmptyMetrics(t *testing.T) {
	tc := testCase{
		minReplicas:             2,
		maxReplicas:             6,
		specReplicas:            4,
		statusReplicas:          4,
		expectedDesiredReplicas: 4,
		CPUTarget:               100,
		reportedLevels:          []uint64{},
		reportedCPURequests:     []resource.Quantity{resource.MustParse("1.0"), resource.MustParse("1.0"), resource.MustParse("1.0"), resource.MustParse("1.0")},
		useMetricsAPI:           true,
		expectedConditions: []autoscalingv1.HorizontalPodAutoscalerCondition{
			{Type: autoscalingv1.AbleToScale, Status: v1.ConditionTrue, Reason: "SucceededGetScale"},
			{Type: autoscalingv1.ScalingActive, Status: v1.ConditionFalse, Reason: "FailedGetResourceMetric"},
		},
	}
	tc.runTest(t)
}

func TestEmptyCPURequest(t *testing.T) {
	tc := testCase{
		minReplicas:             1,
		maxReplicas:             5,
		specReplicas:            1,
		statusReplicas:          1,
		expectedDesiredReplicas: 1,
		CPUTarget:               100,
		reportedLevels:          []uint64{200},
		reportedCPURequests:     []resource.Quantity{},
		useMetricsAPI:           true,
		expectedConditions: []autoscalingv1.HorizontalPodAutoscalerCondition{
			{Type: autoscalingv1.AbleToScale, Status: v1.ConditionTrue, Reason: "SucceededGetScale"},
			{Type: autoscalingv1.ScalingActive, Status: v1.ConditionFalse, Reason: "FailedGetResourceMetric"},
		},
	}
	tc.runTest(t)
}

func TestEventCreated(t *testing.T) {
	tc := testCase{
		minReplicas:             1,
		maxReplicas:             5,
		specReplicas:            1,
		statusReplicas:          1,
		expectedDesiredReplicas: 2,
		CPUTarget:               50,
		reportedLevels:          []uint64{200},
		reportedCPURequests:     []resource.Quantity{resource.MustParse("0.2")},
		verifyEvents:            true,
		useMetricsAPI:           true,
	}
	tc.runTest(t)
}

func TestEventNotCreated(t *testing.T) {
	tc := testCase{
		minReplicas:             1,
		maxReplicas:             5,
		specReplicas:            2,
		statusReplicas:          2,
		expectedDesiredReplicas: 2,
		CPUTarget:               50,
		reportedLevels:          []uint64{200, 200},
		reportedCPURequests:     []resource.Quantity{resource.MustParse("0.4"), resource.MustParse("0.4")},
		verifyEvents:            true,
		useMetricsAPI:           true,
		expectedConditions: statusOkWithOverrides(autoscalingv2.HorizontalPodAutoscalerCondition{
			Type:   autoscalingv2.AbleToScale,
			Status: v1.ConditionTrue,
			Reason: "ReadyForNewScale",
		}),
	}
	tc.runTest(t)
}

func TestMissingReports(t *testing.T) {
	tc := testCase{
		minReplicas:             1,
		maxReplicas:             5,
		specReplicas:            4,
		statusReplicas:          4,
		expectedDesiredReplicas: 2,
		CPUTarget:               50,
		reportedLevels:          []uint64{200},
		reportedCPURequests:     []resource.Quantity{resource.MustParse("0.2")},
		useMetricsAPI:           true,
		recommendations:         []timestampedRecommendation{},
	}
	tc.runTest(t)
}

func TestUpscaleCap(t *testing.T) {
	tc := testCase{
		minReplicas:             1,
		maxReplicas:             100,
		specReplicas:            3,
		statusReplicas:          3,
		scaleUpRules:            generateScalingRules(0, 0, 700, 60, 0),
		initialReplicas:         3,
		expectedDesiredReplicas: 24,
		CPUTarget:               10,
		reportedLevels:          []uint64{100, 200, 300},
		reportedCPURequests:     []resource.Quantity{resource.MustParse("0.1"), resource.MustParse("0.1"), resource.MustParse("0.1")},
		useMetricsAPI:           true,
		expectedConditions: statusOkWithOverrides(autoscalingv2.HorizontalPodAutoscalerCondition{
			Type:   autoscalingv2.ScalingLimited,
			Status: v1.ConditionTrue,
			Reason: "ScaleUpLimit",
		}),
	}
	tc.runTest(t)
}

func TestUpscaleCapGreaterThanMaxReplicas(t *testing.T) {
	tc := testCase{
		minReplicas:     1,
		maxReplicas:     20,
		specReplicas:    3,
		statusReplicas:  3,
		scaleUpRules:    generateScalingRules(0, 0, 700, 60, 0),
		initialReplicas: 3,
		// expectedDesiredReplicas would be 24 without maxReplicas
		expectedDesiredReplicas: 20,
		CPUTarget:               10,
		reportedLevels:          []uint64{100, 200, 300},
		reportedCPURequests:     []resource.Quantity{resource.MustParse("0.1"), resource.MustParse("0.1"), resource.MustParse("0.1")},
		useMetricsAPI:           true,
		expectedConditions: statusOkWithOverrides(autoscalingv2.HorizontalPodAutoscalerCondition{
			Type:   autoscalingv2.ScalingLimited,
			Status: v1.ConditionTrue,
			Reason: "TooManyReplicas",
		}),
	}
	tc.runTest(t)
}

func TestMoreReplicasThanSpecNoScale(t *testing.T) {
	tc := testCase{
		minReplicas:             1,
		maxReplicas:             8,
		specReplicas:            4,
		statusReplicas:          5, // Deployment update with 25% surge.
		expectedDesiredReplicas: 4,
		CPUTarget:               50,
		reportedLevels:          []uint64{500, 500, 500, 500, 500},
		reportedCPURequests: []resource.Quantity{
			resource.MustParse("1"),
			resource.MustParse("1"),
			resource.MustParse("1"),
			resource.MustParse("1"),
			resource.MustParse("1"),
		},
		useMetricsAPI: true,
		expectedConditions: statusOkWithOverrides(autoscalingv2.HorizontalPodAutoscalerCondition{
			Type:   autoscalingv2.AbleToScale,
			Status: v1.ConditionTrue,
			Reason: "ReadyForNewScale",
		}),
	}
	tc.runTest(t)
}

func TestConditionInvalidSelectorMissing(t *testing.T) {
	tc := testCase{
		minReplicas:             1,
		maxReplicas:             100,
		specReplicas:            3,
		statusReplicas:          3,
		expectedDesiredReplicas: 3,
		CPUTarget:               10,
		reportedLevels:          []uint64{100, 200, 300},
		reportedCPURequests:     []resource.Quantity{resource.MustParse("0.1"), resource.MustParse("0.1"), resource.MustParse("0.1")},
		useMetricsAPI:           true,
		expectedConditions: []autoscalingv1.HorizontalPodAutoscalerCondition{
			{
				Type:   autoscalingv1.AbleToScale,
				Status: v1.ConditionTrue,
				Reason: "SucceededGetScale",
			},
			{
				Type:   autoscalingv1.ScalingActive,
				Status: v1.ConditionFalse,
				Reason: "InvalidSelector",
			},
		},
	}

	_, _, _, _, testScaleClient := tc.prepareTestClient(t)
	tc.testScaleClient = testScaleClient

	testScaleClient.PrependReactor("get", "replicationcontrollers", func(action core.Action) (handled bool, ret runtime.Object, err error) {
		obj := &autoscalingv1.Scale{
			ObjectMeta: metav1.ObjectMeta{
				Name: tc.resource.name,
			},
			Spec: autoscalingv1.ScaleSpec{
				Replicas: tc.specReplicas,
			},
			Status: autoscalingv1.ScaleStatus{
				Replicas: tc.specReplicas,
			},
		}
		return true, obj, nil
	})

	tc.runTest(t)
}

func TestConditionInvalidSelectorUnparsable(t *testing.T) {
	tc := testCase{
		minReplicas:             1,
		maxReplicas:             100,
		specReplicas:            3,
		statusReplicas:          3,
		expectedDesiredReplicas: 3,
		CPUTarget:               10,
		reportedLevels:          []uint64{100, 200, 300},
		reportedCPURequests:     []resource.Quantity{resource.MustParse("0.1"), resource.MustParse("0.1"), resource.MustParse("0.1")},
		useMetricsAPI:           true,
		expectedConditions: []autoscalingv1.HorizontalPodAutoscalerCondition{
			{
				Type:   autoscalingv1.AbleToScale,
				Status: v1.ConditionTrue,
				Reason: "SucceededGetScale",
			},
			{
				Type:   autoscalingv1.ScalingActive,
				Status: v1.ConditionFalse,
				Reason: "InvalidSelector",
			},
		},
	}

	_, _, _, _, testScaleClient := tc.prepareTestClient(t)
	tc.testScaleClient = testScaleClient

	testScaleClient.PrependReactor("get", "replicationcontrollers", func(action core.Action) (handled bool, ret runtime.Object, err error) {
		obj := &autoscalingv1.Scale{
			ObjectMeta: metav1.ObjectMeta{
				Name: tc.resource.name,
			},
			Spec: autoscalingv1.ScaleSpec{
				Replicas: tc.specReplicas,
			},
			Status: autoscalingv1.ScaleStatus{
				Replicas: tc.specReplicas,
				Selector: "cheddar cheese",
			},
		}
		return true, obj, nil
	})

	tc.runTest(t)
}

func TestConditionFailedGetMetrics(t *testing.T) {
	targetValue := resource.MustParse("15.0")
	averageValue := resource.MustParse("15.0")
	metricsTargets := map[string][]autoscalingv2.MetricSpec{
		"FailedGetResourceMetric": nil,
		"FailedGetPodsMetric": {
			{
				Type: autoscalingv2.PodsMetricSourceType,
				Pods: &autoscalingv2.PodsMetricSource{
					Metric: autoscalingv2.MetricIdentifier{
						Name: "qps",
					},
					Target: autoscalingv2.MetricTarget{
						AverageValue: &averageValue,
					},
				},
			},
		},
		"FailedGetObjectMetric": {
			{
				Type: autoscalingv2.ObjectMetricSourceType,
				Object: &autoscalingv2.ObjectMetricSource{
					DescribedObject: autoscalingv2.CrossVersionObjectReference{
						APIVersion: "apps/v1",
						Kind:       "Deployment",
						Name:       "some-deployment",
					},
					Metric: autoscalingv2.MetricIdentifier{
						Name: "qps",
					},
					Target: autoscalingv2.MetricTarget{
						Value: &targetValue,
					},
				},
			},
		},
		"FailedGetExternalMetric": {
			{
				Type: autoscalingv2.ExternalMetricSourceType,
				External: &autoscalingv2.ExternalMetricSource{
					Metric: autoscalingv2.MetricIdentifier{
						Name:     "qps",
						Selector: &metav1.LabelSelector{},
					},
					Target: autoscalingv2.MetricTarget{
						Value: resource.NewMilliQuantity(300, resource.DecimalSI),
					},
				},
			},
		},
	}

	for reason, specs := range metricsTargets {
		tc := testCase{
			minReplicas:             1,
			maxReplicas:             100,
			specReplicas:            3,
			statusReplicas:          3,
			expectedDesiredReplicas: 3,
			CPUTarget:               10,
			reportedLevels:          []uint64{100, 200, 300},
			reportedCPURequests:     []resource.Quantity{resource.MustParse("0.1"), resource.MustParse("0.1"), resource.MustParse("0.1")},
			useMetricsAPI:           true,
		}
		_, testMetricsClient, testCMClient, testEMClient, _ := tc.prepareTestClient(t)
		tc.testMetricsClient = testMetricsClient
		tc.testCMClient = testCMClient
		tc.testEMClient = testEMClient

		testMetricsClient.PrependReactor("list", "pods", func(action core.Action) (handled bool, ret runtime.Object, err error) {
			return true, &metricsapi.PodMetricsList{}, fmt.Errorf("something went wrong")
		})
		testCMClient.PrependReactor("get", "*", func(action core.Action) (handled bool, ret runtime.Object, err error) {
			return true, &cmapi.MetricValueList{}, fmt.Errorf("something went wrong")
		})
		testEMClient.PrependReactor("list", "*", func(action core.Action) (handled bool, ret runtime.Object, err error) {
			return true, &emapi.ExternalMetricValueList{}, fmt.Errorf("something went wrong")
		})

		tc.expectedConditions = []autoscalingv1.HorizontalPodAutoscalerCondition{
			{Type: autoscalingv1.AbleToScale, Status: v1.ConditionTrue, Reason: "SucceededGetScale"},
			{Type: autoscalingv1.ScalingActive, Status: v1.ConditionFalse, Reason: reason},
		}
		if specs != nil {
			tc.CPUTarget = 0
		} else {
			tc.CPUTarget = 10
		}
		tc.metricsTarget = specs
		tc.runTest(t)
	}
}

func TestConditionInvalidSourceType(t *testing.T) {
	tc := testCase{
		minReplicas:             2,
		maxReplicas:             6,
		specReplicas:            3,
		statusReplicas:          3,
		expectedDesiredReplicas: 3,
		CPUTarget:               0,
		metricsTarget: []autoscalingv2.MetricSpec{
			{
				Type: "CheddarCheese",
			},
		},
		reportedLevels: []uint64{20000},
		expectedConditions: []autoscalingv1.HorizontalPodAutoscalerCondition{
			{
				Type:   autoscalingv1.AbleToScale,
				Status: v1.ConditionTrue,
				Reason: "SucceededGetScale",
			},
			{
				Type:   autoscalingv1.ScalingActive,
				Status: v1.ConditionFalse,
				Reason: "InvalidMetricSourceType",
			},
		},
	}
	tc.runTest(t)
}

func TestConditionFailedGetScale(t *testing.T) {
	tc := testCase{
		minReplicas:             1,
		maxReplicas:             100,
		specReplicas:            3,
		statusReplicas:          3,
		expectedDesiredReplicas: 3,
		CPUTarget:               10,
		reportedLevels:          []uint64{100, 200, 300},
		reportedCPURequests:     []resource.Quantity{resource.MustParse("0.1"), resource.MustParse("0.1"), resource.MustParse("0.1")},
		useMetricsAPI:           true,
		expectedConditions: []autoscalingv1.HorizontalPodAutoscalerCondition{
			{
				Type:   autoscalingv1.AbleToScale,
				Status: v1.ConditionFalse,
				Reason: "FailedGetScale",
			},
		},
	}

	_, _, _, _, testScaleClient := tc.prepareTestClient(t)
	tc.testScaleClient = testScaleClient

	testScaleClient.PrependReactor("get", "replicationcontrollers", func(action core.Action) (handled bool, ret runtime.Object, err error) {
		return true, &autoscalingv1.Scale{}, fmt.Errorf("something went wrong")
	})

	tc.runTest(t)
}

func TestConditionFailedUpdateScale(t *testing.T) {
	tc := testCase{
		minReplicas:             1,
		maxReplicas:             5,
		specReplicas:            3,
		statusReplicas:          3,
		expectedDesiredReplicas: 3,
		CPUTarget:               100,
		reportedLevels:          []uint64{150, 150, 150},
		reportedCPURequests:     []resource.Quantity{resource.MustParse("0.1"), resource.MustParse("0.1"), resource.MustParse("0.1")},
		useMetricsAPI:           true,
		expectedConditions: statusOkWithOverrides(autoscalingv2.HorizontalPodAutoscalerCondition{
			Type:   autoscalingv2.AbleToScale,
			Status: v1.ConditionFalse,
			Reason: "FailedUpdateScale",
		}),
	}

	_, _, _, _, testScaleClient := tc.prepareTestClient(t)
	tc.testScaleClient = testScaleClient

	testScaleClient.PrependReactor("update", "replicationcontrollers", func(action core.Action) (handled bool, ret runtime.Object, err error) {
		return true, &autoscalingv1.Scale{}, fmt.Errorf("something went wrong")
	})

	tc.runTest(t)
}

func TestNoBackoffUpscaleCM(t *testing.T) {
	averageValue := resource.MustParse("15.0")
	time := metav1.Time{Time: time.Now()}
	tc := testCase{
		minReplicas:             1,
		maxReplicas:             5,
		specReplicas:            3,
		statusReplicas:          3,
		expectedDesiredReplicas: 4,
		CPUTarget:               0,
		metricsTarget: []autoscalingv2.MetricSpec{
			{
				Type: autoscalingv2.PodsMetricSourceType,
				Pods: &autoscalingv2.PodsMetricSource{
					Metric: autoscalingv2.MetricIdentifier{
						Name: "qps",
					},
					Target: autoscalingv2.MetricTarget{
						AverageValue: &averageValue,
					},
				},
			},
		},
		reportedLevels:      []uint64{20000, 10000, 30000},
		reportedCPURequests: []resource.Quantity{resource.MustParse("1.0"), resource.MustParse("1.0"), resource.MustParse("1.0")},
		//useMetricsAPI:       true,
		lastScaleTime: &time,
		expectedConditions: statusOkWithOverrides(autoscalingv2.HorizontalPodAutoscalerCondition{
			Type:   autoscalingv2.AbleToScale,
			Status: v1.ConditionTrue,
			Reason: "ReadyForNewScale",
		}, autoscalingv2.HorizontalPodAutoscalerCondition{
			Type:   autoscalingv2.AbleToScale,
			Status: v1.ConditionTrue,
			Reason: "SucceededRescale",
		}, autoscalingv2.HorizontalPodAutoscalerCondition{
			Type:   autoscalingv2.ScalingLimited,
			Status: v1.ConditionFalse,
			Reason: "DesiredWithinRange",
		}),
	}
	tc.runTest(t)
}

func TestNoBackoffUpscaleCMNoBackoffCpu(t *testing.T) {
	averageValue := resource.MustParse("15.0")
	time := metav1.Time{Time: time.Now()}
	tc := testCase{
		minReplicas:             1,
		maxReplicas:             5,
		specReplicas:            3,
		statusReplicas:          3,
		expectedDesiredReplicas: 5,
		CPUTarget:               10,
		metricsTarget: []autoscalingv2.MetricSpec{
			{
				Type: autoscalingv2.PodsMetricSourceType,
				Pods: &autoscalingv2.PodsMetricSource{
					Metric: autoscalingv2.MetricIdentifier{
						Name: "qps",
					},
					Target: autoscalingv2.MetricTarget{
						AverageValue: &averageValue,
					},
				},
			},
		},
		reportedLevels:      []uint64{20000, 10000, 30000},
		reportedCPURequests: []resource.Quantity{resource.MustParse("1.0"), resource.MustParse("1.0"), resource.MustParse("1.0")},
		useMetricsAPI:       true,
		lastScaleTime:       &time,
		expectedConditions: statusOkWithOverrides(autoscalingv2.HorizontalPodAutoscalerCondition{
			Type:   autoscalingv2.AbleToScale,
			Status: v1.ConditionTrue,
			Reason: "ReadyForNewScale",
		}, autoscalingv2.HorizontalPodAutoscalerCondition{
			Type:   autoscalingv2.AbleToScale,
			Status: v1.ConditionTrue,
			Reason: "SucceededRescale",
		}, autoscalingv2.HorizontalPodAutoscalerCondition{
			Type:   autoscalingv2.ScalingLimited,
			Status: v1.ConditionTrue,
			Reason: "TooManyReplicas",
		}),
	}
	tc.runTest(t)
}

func TestStabilizeDownscale(t *testing.T) {
	tc := testCase{
		minReplicas:             1,
		maxReplicas:             5,
		specReplicas:            4,
		statusReplicas:          4,
		expectedDesiredReplicas: 4,
		CPUTarget:               100,
		reportedLevels:          []uint64{50, 50, 50},
		reportedCPURequests:     []resource.Quantity{resource.MustParse("0.1"), resource.MustParse("0.1"), resource.MustParse("0.1")},
		useMetricsAPI:           true,
		expectedConditions: statusOkWithOverrides(autoscalingv2.HorizontalPodAutoscalerCondition{
			Type:   autoscalingv2.AbleToScale,
			Status: v1.ConditionTrue,
			Reason: "ReadyForNewScale",
		}, autoscalingv2.HorizontalPodAutoscalerCondition{
			Type:   autoscalingv2.AbleToScale,
			Status: v1.ConditionTrue,
			Reason: "ScaleDownStabilized",
		}),
		recommendations: []timestampedRecommendation{
			{10, time.Now().Add(-10 * time.Minute)},
			{4, time.Now().Add(-1 * time.Minute)},
		},
	}
	tc.runTest(t)
}

// TestComputedToleranceAlgImplementation is a regression test which
// back-calculates a minimal percentage for downscaling based on a small percentage
// increase in pod utilization which is calibrated against the tolerance value.
func TestComputedToleranceAlgImplementation(t *testing.T) {

	startPods := int32(10)
	// 150 mCPU per pod.
	totalUsedCPUOfAllPods := uint64(startPods * 150)
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
	tc1 := testCase{
		minReplicas:             0,
		maxReplicas:             1000,
		specReplicas:            startPods,
		statusReplicas:          startPods,
		expectedDesiredReplicas: finalPods,
		CPUTarget:               finalCPUPercentTarget,
		reportedLevels: []uint64{
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
		reportedCPURequests: []resource.Quantity{
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
		useMetricsAPI:   true,
		recommendations: []timestampedRecommendation{},
	}
	tc1.runTest(t)

	target = math.Abs(1/(requestedToUsed*(1-defaultTestingTolerance))) + .004
	finalCPUPercentTarget = int32(target * 100)
	tc2 := testCase{
		minReplicas:             0,
		maxReplicas:             1000,
		specReplicas:            startPods,
		statusReplicas:          startPods,
		expectedDesiredReplicas: startPods,
		CPUTarget:               finalCPUPercentTarget,
		reportedLevels: []uint64{
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
		reportedCPURequests: []resource.Quantity{
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
		useMetricsAPI:   true,
		recommendations: []timestampedRecommendation{},
		expectedConditions: statusOkWithOverrides(autoscalingv2.HorizontalPodAutoscalerCondition{
			Type:   autoscalingv2.AbleToScale,
			Status: v1.ConditionTrue,
			Reason: "ReadyForNewScale",
		}),
	}
	tc2.runTest(t)
}

func TestScaleUpRCImmediately(t *testing.T) {
	time := metav1.Time{Time: time.Now()}
	tc := testCase{
		minReplicas:             2,
		maxReplicas:             6,
		specReplicas:            1,
		statusReplicas:          1,
		expectedDesiredReplicas: 2,
		verifyCPUCurrent:        false,
		reportedLevels:          []uint64{0, 0, 0, 0},
		reportedCPURequests:     []resource.Quantity{resource.MustParse("1.0"), resource.MustParse("1.0"), resource.MustParse("1.0"), resource.MustParse("1.0")},
		useMetricsAPI:           true,
		lastScaleTime:           &time,
		expectedConditions: []autoscalingv1.HorizontalPodAutoscalerCondition{
			{Type: autoscalingv1.AbleToScale, Status: v1.ConditionTrue, Reason: "SucceededRescale"},
		},
	}
	tc.runTest(t)
}

func TestScaleDownRCImmediately(t *testing.T) {
	time := metav1.Time{Time: time.Now()}
	tc := testCase{
		minReplicas:             2,
		maxReplicas:             5,
		specReplicas:            6,
		statusReplicas:          6,
		expectedDesiredReplicas: 5,
		CPUTarget:               50,
		reportedLevels:          []uint64{8000, 9500, 1000},
		reportedCPURequests:     []resource.Quantity{resource.MustParse("0.9"), resource.MustParse("1.0"), resource.MustParse("1.1")},
		useMetricsAPI:           true,
		lastScaleTime:           &time,
		expectedConditions: []autoscalingv1.HorizontalPodAutoscalerCondition{
			{Type: autoscalingv1.AbleToScale, Status: v1.ConditionTrue, Reason: "SucceededRescale"},
		},
	}
	tc.runTest(t)
}

func TestAvoidUnnecessaryUpdates(t *testing.T) {
	now := metav1.Time{Time: time.Now().Add(-time.Hour)}
	tc := testCase{
		minReplicas:             2,
		maxReplicas:             6,
		specReplicas:            2,
		statusReplicas:          2,
		expectedDesiredReplicas: 2,
		CPUTarget:               30,
		CPUCurrent:              40,
		verifyCPUCurrent:        true,
		reportedLevels:          []uint64{400, 500, 700},
		reportedCPURequests:     []resource.Quantity{resource.MustParse("1.0"), resource.MustParse("1.0"), resource.MustParse("1.0")},
		reportedPodStartTime:    []metav1.Time{coolCPUCreationTime(), hotCPUCreationTime(), hotCPUCreationTime()},
		useMetricsAPI:           true,
		lastScaleTime:           &now,
		recommendations:         []timestampedRecommendation{},
	}
	testClient, _, _, _, _ := tc.prepareTestClient(t)
	tc.testClient = testClient
	testClient.PrependReactor("list", "horizontalpodautoscalers", func(action core.Action) (handled bool, ret runtime.Object, err error) {
		tc.Lock()
		defer tc.Unlock()
		// fake out the verification logic and mark that we're done processing
		go func() {
			// wait a tick and then mark that we're finished (otherwise, we have no
			// way to indicate that we're finished, because the function decides not to do anything)
			time.Sleep(1 * time.Second)
			tc.Lock()
			tc.statusUpdated = true
			tc.Unlock()
			tc.processed <- "test-hpa"
		}()

		quantity := resource.MustParse("400m")
		obj := &autoscalingv2.HorizontalPodAutoscalerList{
			Items: []autoscalingv2.HorizontalPodAutoscaler{
				{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "test-hpa",
						Namespace: "test-namespace",
						SelfLink:  "experimental/v1/namespaces/test-namespace/horizontalpodautoscalers/test-hpa",
					},
					Spec: autoscalingv2.HorizontalPodAutoscalerSpec{
						ScaleTargetRef: autoscalingv2.CrossVersionObjectReference{
							Kind:       "ReplicationController",
							Name:       "test-rc",
							APIVersion: "v1",
						},

						MinReplicas: &tc.minReplicas,
						MaxReplicas: tc.maxReplicas,
					},
					Status: autoscalingv2.HorizontalPodAutoscalerStatus{
						CurrentReplicas: tc.specReplicas,
						DesiredReplicas: tc.specReplicas,
						LastScaleTime:   tc.lastScaleTime,
						CurrentMetrics: []autoscalingv2.MetricStatus{
							{
								Type: autoscalingv2.ResourceMetricSourceType,
								Resource: &autoscalingv2.ResourceMetricStatus{
									Name: v1.ResourceCPU,
									Current: autoscalingv2.MetricValueStatus{
										AverageValue:       &quantity,
										AverageUtilization: &tc.CPUCurrent,
									},
								},
							},
						},
						Conditions: []autoscalingv2.HorizontalPodAutoscalerCondition{
							{
								Type:               autoscalingv2.AbleToScale,
								Status:             v1.ConditionTrue,
								LastTransitionTime: *tc.lastScaleTime,
								Reason:             "ReadyForNewScale",
								Message:            "recommended size matches current size",
							},
							{
								Type:               autoscalingv2.ScalingActive,
								Status:             v1.ConditionTrue,
								LastTransitionTime: *tc.lastScaleTime,
								Reason:             "ValidMetricFound",
								Message:            "the HPA was able to successfully calculate a replica count from cpu resource utilization (percentage of request)",
							},
							{
								Type:               autoscalingv2.ScalingLimited,
								Status:             v1.ConditionTrue,
								LastTransitionTime: *tc.lastScaleTime,
								Reason:             "TooFewReplicas",
								Message:            "the desired replica count is less than the minimum replica count",
							},
						},
					},
				},
			},
		}
		// and... convert to autoscaling v1 to return the right type
		objv1, err := unsafeConvertToVersionVia(obj, autoscalingv1.SchemeGroupVersion)
		if err != nil {
			return true, nil, err
		}

		return true, objv1, nil
	})
	testClient.PrependReactor("update", "horizontalpodautoscalers", func(action core.Action) (handled bool, ret runtime.Object, err error) {
		assert.Fail(t, "should not have attempted to update the HPA when nothing changed")
		// mark that we've processed this HPA
		tc.processed <- ""
		return true, nil, fmt.Errorf("unexpected call")
	})

	controller, informerFactory := tc.setupController(t)
	tc.runTestWithController(t, controller, informerFactory)
}

func TestConvertDesiredReplicasWithRules(t *testing.T) {
	conversionTestCases := []struct {
		currentReplicas                  int32
		expectedDesiredReplicas          int32
		hpaMinReplicas                   int32
		hpaMaxReplicas                   int32
		expectedConvertedDesiredReplicas int32
		expectedCondition                string
		annotation                       string
	}{
		{
			currentReplicas:                  5,
			expectedDesiredReplicas:          7,
			hpaMinReplicas:                   3,
			hpaMaxReplicas:                   8,
			expectedConvertedDesiredReplicas: 7,
			expectedCondition:                "DesiredWithinRange",
			annotation:                       "prenormalized desired replicas within range",
		},
		{
			currentReplicas:                  3,
			expectedDesiredReplicas:          1,
			hpaMinReplicas:                   2,
			hpaMaxReplicas:                   8,
			expectedConvertedDesiredReplicas: 2,
			expectedCondition:                "TooFewReplicas",
			annotation:                       "prenormalized desired replicas < minReplicas",
		},
		{
			currentReplicas:                  1,
			expectedDesiredReplicas:          0,
			hpaMinReplicas:                   0,
			hpaMaxReplicas:                   10,
			expectedConvertedDesiredReplicas: 0,
			expectedCondition:                "DesiredWithinRange",
			annotation:                       "prenormalized desired zeroed replicas within range",
		},
		{
			currentReplicas:                  20,
			expectedDesiredReplicas:          1000,
			hpaMinReplicas:                   1,
			hpaMaxReplicas:                   10,
			expectedConvertedDesiredReplicas: 10,
			expectedCondition:                "TooManyReplicas",
			annotation:                       "maxReplicas is the limit because maxReplicas < scaleUpLimit",
		},
		{
			currentReplicas:                  3,
			expectedDesiredReplicas:          1000,
			hpaMinReplicas:                   1,
			hpaMaxReplicas:                   2000,
			expectedConvertedDesiredReplicas: calculateScaleUpLimit(3),
			expectedCondition:                "ScaleUpLimit",
			annotation:                       "scaleUpLimit is the limit because scaleUpLimit < maxReplicas",
		},
	}

	for _, ctc := range conversionTestCases {
		t.Run(ctc.annotation, func(t *testing.T) {
			actualConvertedDesiredReplicas, actualCondition, _ := convertDesiredReplicasWithRules(
				ctc.currentReplicas, ctc.expectedDesiredReplicas, ctc.hpaMinReplicas, ctc.hpaMaxReplicas,
			)

			assert.Equal(t, ctc.expectedConvertedDesiredReplicas, actualConvertedDesiredReplicas, ctc.annotation)
			assert.Equal(t, ctc.expectedCondition, actualCondition, ctc.annotation)
		})
	}
}

func TestCalculateScaleUpLimitWithScalingRules(t *testing.T) {
	policy := autoscalingv2.MinPolicySelect

	calculated := calculateScaleUpLimitWithScalingRules(1, []timestampedScaleEvent{}, &autoscalingv2.HPAScalingRules{
		StabilizationWindowSeconds: utilpointer.Int32Ptr(300),
		SelectPolicy:               &policy,
		Policies: []autoscalingv2.HPAScalingPolicy{
			{
				Type:          autoscalingv2.PodsScalingPolicy,
				Value:         2,
				PeriodSeconds: 60,
			},
			{
				Type:          autoscalingv2.PercentScalingPolicy,
				Value:         50,
				PeriodSeconds: 60,
			},
		},
	})
	assert.Equal(t, calculated, int32(2))
}

func TestCalculateScaleDownLimitWithBehaviors(t *testing.T) {
	policy := autoscalingv2.MinPolicySelect

	calculated := calculateScaleDownLimitWithBehaviors(5, []timestampedScaleEvent{}, &autoscalingv2.HPAScalingRules{
		StabilizationWindowSeconds: utilpointer.Int32Ptr(300),
		SelectPolicy:               &policy,
		Policies: []autoscalingv2.HPAScalingPolicy{
			{
				Type:          autoscalingv2.PodsScalingPolicy,
				Value:         2,
				PeriodSeconds: 60,
			},
			{
				Type:          autoscalingv2.PercentScalingPolicy,
				Value:         50,
				PeriodSeconds: 60,
			},
		},
	})
	assert.Equal(t, calculated, int32(3))
}

func generateScalingRules(pods, podsPeriod, percent, percentPeriod, stabilizationWindow int32) *autoscalingv2.HPAScalingRules {
	policy := autoscalingv2.MaxPolicySelect
	directionBehavior := autoscalingv2.HPAScalingRules{
		StabilizationWindowSeconds: utilpointer.Int32Ptr(stabilizationWindow),
		SelectPolicy:               &policy,
	}
	if pods != 0 {
		directionBehavior.Policies = append(directionBehavior.Policies,
			autoscalingv2.HPAScalingPolicy{Type: autoscalingv2.PodsScalingPolicy, Value: pods, PeriodSeconds: podsPeriod})
	}
	if percent != 0 {
		directionBehavior.Policies = append(directionBehavior.Policies,
			autoscalingv2.HPAScalingPolicy{Type: autoscalingv2.PercentScalingPolicy, Value: percent, PeriodSeconds: percentPeriod})
	}
	return &directionBehavior
}

// generateEventsUniformDistribution generates events that uniformly spread in the time window
//    time.Now()-periodSeconds  ; time.Now()
// It split the time window into several segments (by the number of events) and put the event in the center of the segment
// it is needed if you want to create events for several policies (to check how "outdated" flag is set).
// E.g. generateEventsUniformDistribution([]int{1,2,3,4}, 120) will spread events uniformly for the last 120 seconds:
//
//       1          2          3          4
// -----------------------------------------------
//  ^          ^          ^          ^          ^
// -120s      -90s       -60s       -30s       now()
// And we can safely have two different stabilizationWindows:
//  - 60s (guaranteed to have last half of events)
//  - 120s (guaranteed to have all events)
func generateEventsUniformDistribution(rawEvents []int, periodSeconds int) []timestampedScaleEvent {
	events := make([]timestampedScaleEvent, len(rawEvents))
	segmentDuration := float64(periodSeconds) / float64(len(rawEvents))
	for idx, event := range rawEvents {
		segmentBoundary := time.Duration(float64(periodSeconds) - segmentDuration*float64(idx+1) + segmentDuration/float64(2))
		events[idx] = timestampedScaleEvent{
			replicaChange: int32(event),
			timestamp:     time.Now().Add(-time.Second * segmentBoundary),
		}
	}
	return events
}

func TestNormalizeDesiredReplicas(t *testing.T) {
	tests := []struct {
		name                         string
		key                          string
		recommendations              []timestampedRecommendation
		prenormalizedDesiredReplicas int32
		expectedStabilizedReplicas   int32
		expectedLogLength            int
	}{
		{
			"empty log",
			"",
			[]timestampedRecommendation{},
			5,
			5,
			1,
		},
		{
			"stabilize",
			"",
			[]timestampedRecommendation{
				{4, time.Now().Add(-2 * time.Minute)},
				{5, time.Now().Add(-1 * time.Minute)},
			},
			3,
			5,
			3,
		},
		{
			"no stabilize",
			"",
			[]timestampedRecommendation{
				{1, time.Now().Add(-2 * time.Minute)},
				{2, time.Now().Add(-1 * time.Minute)},
			},
			3,
			3,
			3,
		},
		{
			"no stabilize - old recommendations",
			"",
			[]timestampedRecommendation{
				{10, time.Now().Add(-10 * time.Minute)},
				{9, time.Now().Add(-9 * time.Minute)},
			},
			3,
			3,
			2,
		},
		{
			"stabilize - old recommendations",
			"",
			[]timestampedRecommendation{
				{10, time.Now().Add(-10 * time.Minute)},
				{4, time.Now().Add(-1 * time.Minute)},
				{5, time.Now().Add(-2 * time.Minute)},
				{9, time.Now().Add(-9 * time.Minute)},
			},
			3,
			5,
			4,
		},
	}
	for _, tc := range tests {
		hc := HorizontalController{
			downscaleStabilisationWindow: 5 * time.Minute,
			recommendations: map[string][]timestampedRecommendation{
				tc.key: tc.recommendations,
			},
		}
		r := hc.stabilizeRecommendation(tc.key, tc.prenormalizedDesiredReplicas)
		if r != tc.expectedStabilizedReplicas {
			t.Errorf("[%s] got %d stabilized replicas, expected %d", tc.name, r, tc.expectedStabilizedReplicas)
		}
		if len(hc.recommendations[tc.key]) != tc.expectedLogLength {
			t.Errorf("[%s] after  stabilization recommendations log has %d entries, expected %d", tc.name, len(hc.recommendations[tc.key]), tc.expectedLogLength)
		}
	}
}

func TestScalingWithRules(t *testing.T) {
	type TestCase struct {
		name string
		key  string
		// controller arguments
		scaleUpEvents   []timestampedScaleEvent
		scaleDownEvents []timestampedScaleEvent
		// HPA Spec arguments
		specMinReplicas int32
		specMaxReplicas int32
		scaleUpRules    *autoscalingv2.HPAScalingRules
		scaleDownRules  *autoscalingv2.HPAScalingRules
		// external world state
		currentReplicas              int32
		prenormalizedDesiredReplicas int32
		// test expected result
		expectedReplicas  int32
		expectedCondition string

		testThis bool
	}

	tests := []TestCase{
		{
			currentReplicas:              5,
			prenormalizedDesiredReplicas: 7,
			specMinReplicas:              3,
			specMaxReplicas:              8,
			expectedReplicas:             7,
			expectedCondition:            "DesiredWithinRange",
			name:                         "prenormalized desired replicas within range",
		},
		{
			currentReplicas:              3,
			prenormalizedDesiredReplicas: 1,
			specMinReplicas:              2,
			specMaxReplicas:              8,
			expectedReplicas:             2,
			expectedCondition:            "TooFewReplicas",
			name:                         "prenormalized desired replicas < minReplicas",
		},
		{
			currentReplicas:              1,
			prenormalizedDesiredReplicas: 0,
			specMinReplicas:              0,
			specMaxReplicas:              10,
			expectedReplicas:             0,
			expectedCondition:            "DesiredWithinRange",
			name:                         "prenormalized desired replicas within range when minReplicas is 0",
		},
		{
			currentReplicas:              20,
			prenormalizedDesiredReplicas: 1000,
			specMinReplicas:              1,
			specMaxReplicas:              10,
			expectedReplicas:             10,
			expectedCondition:            "TooManyReplicas",
			name:                         "maxReplicas is the limit because maxReplicas < scaleUpLimit",
		},
		{
			currentReplicas:              100,
			prenormalizedDesiredReplicas: 1000,
			specMinReplicas:              100,
			specMaxReplicas:              150,
			expectedReplicas:             150,
			expectedCondition:            "TooManyReplicas",
			name:                         "desired replica count is more than the maximum replica count",
		},
		{
			currentReplicas:              3,
			prenormalizedDesiredReplicas: 1000,
			specMinReplicas:              1,
			specMaxReplicas:              2000,
			expectedReplicas:             4,
			expectedCondition:            "ScaleUpLimit",
			scaleUpRules:                 generateScalingRules(0, 0, 1, 60, 0),
			name:                         "scaleUpLimit is the limit because scaleUpLimit < maxReplicas with user policies",
		},
		{
			currentReplicas:              1000,
			prenormalizedDesiredReplicas: 3,
			specMinReplicas:              3,
			specMaxReplicas:              2000,
			scaleDownRules:               generateScalingRules(20, 60, 0, 0, 0),
			expectedReplicas:             980,
			expectedCondition:            "ScaleDownLimit",
			name:                         "scaleDownLimit is the limit because scaleDownLimit > minReplicas with user defined policies",
			testThis:                     true,
		},
		// ScaleUp without PeriodSeconds usage
		{
			name:                         "scaleUp with default behavior",
			specMinReplicas:              1,
			specMaxReplicas:              1000,
			currentReplicas:              10,
			prenormalizedDesiredReplicas: 50,
			expectedReplicas:             20,
			expectedCondition:            "ScaleUpLimit",
		},
		{
			name:                         "scaleUp with pods policy larger than percent policy",
			specMinReplicas:              1,
			specMaxReplicas:              1000,
			scaleUpRules:                 generateScalingRules(100, 60, 100, 60, 0),
			currentReplicas:              10,
			prenormalizedDesiredReplicas: 500,
			expectedReplicas:             110,
			expectedCondition:            "ScaleUpLimit",
		},
		{
			name:                         "scaleUp with percent policy larger than pods policy",
			specMinReplicas:              1,
			specMaxReplicas:              1000,
			scaleUpRules:                 generateScalingRules(2, 60, 100, 60, 0),
			currentReplicas:              10,
			prenormalizedDesiredReplicas: 500,
			expectedReplicas:             20,
			expectedCondition:            "ScaleUpLimit",
		},
		{
			name:                         "scaleUp with spec MaxReplicas limitation with large pod policy",
			specMinReplicas:              1,
			specMaxReplicas:              1000,
			scaleUpRules:                 generateScalingRules(100, 60, 0, 0, 0),
			currentReplicas:              10,
			prenormalizedDesiredReplicas: 50,
			expectedReplicas:             50,
			expectedCondition:            "DesiredWithinRange",
		},
		{
			name:                         "scaleUp with spec MaxReplicas limitation with large percent policy",
			specMinReplicas:              1,
			specMaxReplicas:              1000,
			scaleUpRules:                 generateScalingRules(10000, 60, 0, 0, 0),
			currentReplicas:              10,
			prenormalizedDesiredReplicas: 50,
			expectedReplicas:             50,
			expectedCondition:            "DesiredWithinRange",
		},
		{
			name:                         "scaleUp with pod policy limitation",
			specMinReplicas:              1,
			specMaxReplicas:              1000,
			scaleUpRules:                 generateScalingRules(30, 60, 0, 0, 0),
			currentReplicas:              10,
			prenormalizedDesiredReplicas: 50,
			expectedReplicas:             40,
			expectedCondition:            "ScaleUpLimit",
		},
		{
			name:                         "scaleUp with percent policy limitation",
			specMinReplicas:              1,
			specMaxReplicas:              1000,
			scaleUpRules:                 generateScalingRules(0, 0, 200, 60, 0),
			currentReplicas:              10,
			prenormalizedDesiredReplicas: 50,
			expectedReplicas:             30,
			expectedCondition:            "ScaleUpLimit",
		},
		{
			name:                         "scaleDown with percent policy larger than pod policy",
			specMinReplicas:              1,
			specMaxReplicas:              1000,
			scaleDownRules:               generateScalingRules(20, 60, 1, 60, 300),
			currentReplicas:              100,
			prenormalizedDesiredReplicas: 2,
			expectedReplicas:             80,
			expectedCondition:            "ScaleDownLimit",
		},
		{
			name:                         "scaleDown with pod policy larger than percent policy",
			specMinReplicas:              1,
			specMaxReplicas:              1000,
			scaleDownRules:               generateScalingRules(2, 60, 1, 60, 300),
			currentReplicas:              100,
			prenormalizedDesiredReplicas: 2,
			expectedReplicas:             98,
			expectedCondition:            "ScaleDownLimit",
		},
		{
			name:                         "scaleDown with spec MinReplicas=nil limitation with large pod policy",
			specMinReplicas:              1,
			specMaxReplicas:              1000,
			scaleDownRules:               generateScalingRules(100, 60, 0, 0, 300),
			currentReplicas:              10,
			prenormalizedDesiredReplicas: 0,
			expectedReplicas:             1,
			expectedCondition:            "TooFewReplicas",
		},
		{
			name:                         "scaleDown with spec MinReplicas limitation with large pod policy",
			specMinReplicas:              1,
			specMaxReplicas:              1000,
			scaleDownRules:               generateScalingRules(100, 60, 0, 0, 300),
			currentReplicas:              10,
			prenormalizedDesiredReplicas: 0,
			expectedReplicas:             1,
			expectedCondition:            "TooFewReplicas",
		},
		{
			name:                         "scaleDown with spec MinReplicas limitation with large percent policy",
			specMinReplicas:              5,
			specMaxReplicas:              1000,
			scaleDownRules:               generateScalingRules(0, 0, 100, 60, 300),
			currentReplicas:              10,
			prenormalizedDesiredReplicas: 2,
			expectedReplicas:             5,
			expectedCondition:            "TooFewReplicas",
		},
		{
			name:                         "scaleDown with pod policy limitation",
			specMinReplicas:              1,
			specMaxReplicas:              1000,
			scaleDownRules:               generateScalingRules(5, 60, 0, 0, 300),
			currentReplicas:              10,
			prenormalizedDesiredReplicas: 2,
			expectedReplicas:             5,
			expectedCondition:            "ScaleDownLimit",
		},
		{
			name:                         "scaleDown with percent policy limitation",
			specMinReplicas:              1,
			specMaxReplicas:              1000,
			scaleDownRules:               generateScalingRules(0, 0, 50, 60, 300),
			currentReplicas:              10,
			prenormalizedDesiredReplicas: 5,
			expectedReplicas:             5,
			expectedCondition:            "DesiredWithinRange",
		},
		{
			name:                         "scaleUp with spec MaxReplicas limitation with large pod policy and events",
			scaleUpEvents:                generateEventsUniformDistribution([]int{1, 5, 9}, 120),
			specMinReplicas:              1,
			specMaxReplicas:              200,
			scaleUpRules:                 generateScalingRules(300, 60, 0, 0, 0),
			currentReplicas:              100,
			prenormalizedDesiredReplicas: 500,
			expectedReplicas:             200, // 200 < 100 - 15 + 300
			expectedCondition:            "TooManyReplicas",
		},
		{
			name:                         "scaleUp with spec MaxReplicas limitation with large percent policy and events",
			scaleUpEvents:                generateEventsUniformDistribution([]int{1, 5, 9}, 120),
			specMinReplicas:              1,
			specMaxReplicas:              200,
			scaleUpRules:                 generateScalingRules(0, 0, 10000, 60, 0),
			currentReplicas:              100,
			prenormalizedDesiredReplicas: 500,
			expectedReplicas:             200,
			expectedCondition:            "TooManyReplicas",
		},
		{
			// corner case for calculating the scaleUpLimit, when we changed pod policy after a lot of scaleUp events
			// in this case we shouldn't allow scale up, though, the naive formula will suggest that scaleUplimit is less then CurrentReplicas (100-15+5 < 100)
			name:                         "scaleUp with currentReplicas limitation with rate.PeriodSeconds with a lot of recent scale up events",
			scaleUpEvents:                generateEventsUniformDistribution([]int{1, 5, 9}, 120),
			specMinReplicas:              1,
			specMaxReplicas:              1000,
			scaleUpRules:                 generateScalingRules(5, 120, 0, 0, 0),
			currentReplicas:              100,
			prenormalizedDesiredReplicas: 500,
			expectedReplicas:             100, // 120 seconds ago we had (100 - 15) replicas, now the rate.Pods = 5,
			expectedCondition:            "ScaleUpLimit",
		},
		{
			name:                         "scaleUp with pod policy and previous scale up events",
			scaleUpEvents:                generateEventsUniformDistribution([]int{1, 5, 9}, 120),
			specMinReplicas:              1,
			specMaxReplicas:              1000,
			scaleUpRules:                 generateScalingRules(150, 120, 0, 0, 0),
			currentReplicas:              100,
			prenormalizedDesiredReplicas: 500,
			expectedReplicas:             235, // 100 - 15 + 150
			expectedCondition:            "ScaleUpLimit",
		},
		{
			name:                         "scaleUp with percent policy and previous scale up events",
			scaleUpEvents:                generateEventsUniformDistribution([]int{1, 5, 9}, 120),
			specMinReplicas:              1,
			specMaxReplicas:              1000,
			scaleUpRules:                 generateScalingRules(0, 0, 200, 120, 0),
			currentReplicas:              100,
			prenormalizedDesiredReplicas: 500,
			expectedReplicas:             255, // (100 - 15) + 200%
			expectedCondition:            "ScaleUpLimit",
		},
		// ScaleDown with PeriodSeconds usage
		{
			name:                         "scaleDown with default policy and previous events",
			scaleDownEvents:              generateEventsUniformDistribution([]int{1, 5, 9}, 120),
			specMinReplicas:              1,
			specMaxReplicas:              1000,
			currentReplicas:              10,
			prenormalizedDesiredReplicas: 5,
			expectedReplicas:             5, // without scaleDown rate limitations the PeriodSeconds does not influence anything
			expectedCondition:            "DesiredWithinRange",
		},
		{
			name:                         "scaleDown with spec MinReplicas=nil limitation with large pod policy and previous events",
			scaleDownEvents:              generateEventsUniformDistribution([]int{1, 5, 9}, 120),
			specMinReplicas:              1,
			specMaxReplicas:              1000,
			scaleDownRules:               generateScalingRules(115, 120, 0, 0, 300),
			currentReplicas:              100,
			prenormalizedDesiredReplicas: 0,
			expectedReplicas:             1,
			expectedCondition:            "TooFewReplicas",
		},
		{
			name:                         "scaleDown with spec MinReplicas limitation with large pod policy and previous events",
			scaleDownEvents:              generateEventsUniformDistribution([]int{1, 5, 9}, 120),
			specMinReplicas:              5,
			specMaxReplicas:              1000,
			scaleDownRules:               generateScalingRules(130, 120, 0, 0, 300),
			currentReplicas:              100,
			prenormalizedDesiredReplicas: 0,
			expectedReplicas:             5,
			expectedCondition:            "TooFewReplicas",
		},
		{
			name:                         "scaleDown with spec MinReplicas limitation with large percent policy and previous events",
			scaleDownEvents:              generateEventsUniformDistribution([]int{1, 5, 9}, 120),
			specMinReplicas:              5,
			specMaxReplicas:              1000,
			scaleDownRules:               generateScalingRules(0, 0, 100, 120, 300), // 100% removal - is always to 0 => limited by MinReplicas
			currentReplicas:              100,
			prenormalizedDesiredReplicas: 2,
			expectedReplicas:             5,
			expectedCondition:            "TooFewReplicas",
		},
		{
			name:                         "scaleDown with pod policy limitation and previous events",
			scaleDownEvents:              generateEventsUniformDistribution([]int{1, 5, 9}, 120),
			specMinReplicas:              1,
			specMaxReplicas:              1000,
			scaleDownRules:               generateScalingRules(5, 120, 0, 0, 300),
			currentReplicas:              100,
			prenormalizedDesiredReplicas: 2,
			expectedReplicas:             100, // 100 + 15 - 5
			expectedCondition:            "ScaleDownLimit",
		},
		{
			name:                         "scaleDown with percent policy limitation and previous events",
			scaleDownEvents:              generateEventsUniformDistribution([]int{2, 4, 6}, 120),
			specMinReplicas:              1,
			specMaxReplicas:              1000,
			scaleDownRules:               generateScalingRules(0, 0, 50, 120, 300),
			currentReplicas:              100,
			prenormalizedDesiredReplicas: 0,
			expectedReplicas:             56, // (100 + 12) - 50%
			expectedCondition:            "ScaleDownLimit",
		},
		{
			// corner case for calculating the scaleDownLimit, when we changed pod or percent policy after a lot of scaleDown events
			// in this case we shouldn't allow scale down, though, the naive formula will suggest that scaleDownlimit is more then CurrentReplicas (100+30-10% > 100)
			name:                         "scaleDown with previous events preventing further scale down",
			scaleDownEvents:              generateEventsUniformDistribution([]int{10, 10, 10}, 120),
			specMinReplicas:              1,
			specMaxReplicas:              1000,
			scaleDownRules:               generateScalingRules(0, 0, 10, 120, 300),
			currentReplicas:              100,
			prenormalizedDesiredReplicas: 0,
			expectedReplicas:             100, // (100 + 30) - 10% = 117 is more then 100 (currentReplicas), keep 100
			expectedCondition:            "ScaleDownLimit",
		},
		{
			// corner case, the same as above, but calculation shows that we should go below zero
			name:                         "scaleDown with with previous events still allowing more scale down",
			scaleDownEvents:              generateEventsUniformDistribution([]int{10, 10, 10}, 120),
			specMinReplicas:              1,
			specMaxReplicas:              1000,
			scaleDownRules:               generateScalingRules(0, 0, 1000, 120, 300),
			currentReplicas:              10,
			prenormalizedDesiredReplicas: 5,
			expectedReplicas:             5, // (10 + 30) - 1000% = -360 is less than 0 and less then 5 (desired by metrics), set 5
			expectedCondition:            "DesiredWithinRange",
		},
		{
			name:                         "check 'outdated' flag for events for one behavior for up",
			scaleUpEvents:                generateEventsUniformDistribution([]int{8, 12, 9, 11}, 120),
			specMinReplicas:              1,
			specMaxReplicas:              1000,
			scaleUpRules:                 generateScalingRules(1000, 60, 0, 0, 0),
			currentReplicas:              100,
			prenormalizedDesiredReplicas: 200,
			expectedReplicas:             200,
			expectedCondition:            "DesiredWithinRange",
		},
		{
			name:                         "check that events were not marked 'outdated' for two different policies in the behavior for up",
			scaleUpEvents:                generateEventsUniformDistribution([]int{8, 12, 9, 11}, 120),
			specMinReplicas:              1,
			specMaxReplicas:              1000,
			scaleUpRules:                 generateScalingRules(1000, 120, 100, 60, 0),
			currentReplicas:              100,
			prenormalizedDesiredReplicas: 200,
			expectedReplicas:             200,
			expectedCondition:            "DesiredWithinRange",
		},
		{
			name:                         "check that events were marked 'outdated' for two different policies in the behavior for up",
			scaleUpEvents:                generateEventsUniformDistribution([]int{8, 12, 9, 11}, 120),
			specMinReplicas:              1,
			specMaxReplicas:              1000,
			scaleUpRules:                 generateScalingRules(1000, 30, 100, 60, 0),
			currentReplicas:              100,
			prenormalizedDesiredReplicas: 200,
			expectedReplicas:             200,
			expectedCondition:            "DesiredWithinRange",
		},
		{
			name:                         "check 'outdated' flag for events for one behavior for down",
			scaleDownEvents:              generateEventsUniformDistribution([]int{8, 12, 9, 11}, 120),
			specMinReplicas:              1,
			specMaxReplicas:              1000,
			scaleDownRules:               generateScalingRules(1000, 60, 0, 0, 300),
			currentReplicas:              100,
			prenormalizedDesiredReplicas: 5,
			expectedReplicas:             5,
			expectedCondition:            "DesiredWithinRange",
		},
		{
			name:                         "check that events were not marked 'outdated' for two different policies in the behavior for down",
			scaleDownEvents:              generateEventsUniformDistribution([]int{8, 12, 9, 11}, 120),
			specMinReplicas:              1,
			specMaxReplicas:              1000,
			scaleDownRules:               generateScalingRules(1000, 120, 100, 60, 300),
			currentReplicas:              100,
			prenormalizedDesiredReplicas: 5,
			expectedReplicas:             5,
			expectedCondition:            "DesiredWithinRange",
		},
		{
			name:                         "check that events were marked 'outdated' for two different policies in the behavior for down",
			scaleDownEvents:              generateEventsUniformDistribution([]int{8, 12, 9, 11}, 120),
			specMinReplicas:              1,
			specMaxReplicas:              1000,
			scaleDownRules:               generateScalingRules(1000, 30, 100, 60, 300),
			currentReplicas:              100,
			prenormalizedDesiredReplicas: 5,
			expectedReplicas:             5,
			expectedCondition:            "DesiredWithinRange",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {

			if tc.testThis {
				return
			}
			hc := HorizontalController{
				scaleUpEvents: map[string][]timestampedScaleEvent{
					tc.key: tc.scaleUpEvents,
				},
				scaleDownEvents: map[string][]timestampedScaleEvent{
					tc.key: tc.scaleDownEvents,
				},
			}
			arg := NormalizationArg{
				Key:               tc.key,
				ScaleUpBehavior:   autoscalingapiv2beta2.GenerateHPAScaleUpRules(tc.scaleUpRules),
				ScaleDownBehavior: autoscalingapiv2beta2.GenerateHPAScaleDownRules(tc.scaleDownRules),
				MinReplicas:       tc.specMinReplicas,
				MaxReplicas:       tc.specMaxReplicas,
				DesiredReplicas:   tc.prenormalizedDesiredReplicas,
				CurrentReplicas:   tc.currentReplicas,
			}

			replicas, condition, _ := hc.convertDesiredReplicasWithBehaviorRate(arg)
			assert.Equal(t, tc.expectedReplicas, replicas, "expected replicas do not match with converted replicas")
			assert.Equal(t, tc.expectedCondition, condition, "HPA condition does not match with expected condition")
		})
	}

}

// TestStoreScaleEvents tests events storage and usage
func TestStoreScaleEvents(t *testing.T) {
	type TestCase struct {
		name                   string
		key                    string
		replicaChange          int32
		prevScaleEvents        []timestampedScaleEvent
		newScaleEvents         []timestampedScaleEvent
		scalingRules           *autoscalingv2.HPAScalingRules
		expectedReplicasChange int32
	}
	tests := []TestCase{
		{
			name:                   "empty entries with default behavior",
			replicaChange:          5,
			prevScaleEvents:        []timestampedScaleEvent{}, // no history -> 0 replica change
			newScaleEvents:         []timestampedScaleEvent{}, // no behavior -> no events are stored
			expectedReplicasChange: 0,
		},
		{
			name:                   "empty entries with two-policy-behavior",
			replicaChange:          5,
			prevScaleEvents:        []timestampedScaleEvent{}, // no history -> 0 replica change
			newScaleEvents:         []timestampedScaleEvent{{5, time.Now(), false}},
			scalingRules:           generateScalingRules(10, 60, 100, 60, 0),
			expectedReplicasChange: 0,
		},
		{
			name:          "one outdated entry to be kept untouched without behavior",
			replicaChange: 5,
			prevScaleEvents: []timestampedScaleEvent{
				{7, time.Now().Add(-time.Second * time.Duration(61)), false}, // outdated event, should be replaced
			},
			newScaleEvents: []timestampedScaleEvent{
				{7, time.Now(), false}, // no behavior -> we don't touch stored events
			},
			expectedReplicasChange: 0,
		},
		{
			name:          "one outdated entry to be replaced with behavior",
			replicaChange: 5,
			prevScaleEvents: []timestampedScaleEvent{
				{7, time.Now().Add(-time.Second * time.Duration(61)), false}, // outdated event, should be replaced
			},
			newScaleEvents: []timestampedScaleEvent{
				{5, time.Now(), false},
			},
			scalingRules:           generateScalingRules(10, 60, 100, 60, 0),
			expectedReplicasChange: 0,
		},
		{
			name:          "one actual entry to be not touched with behavior",
			replicaChange: 5,
			prevScaleEvents: []timestampedScaleEvent{
				{7, time.Now().Add(-time.Second * time.Duration(58)), false},
			},
			newScaleEvents: []timestampedScaleEvent{
				{7, time.Now(), false},
				{5, time.Now(), false},
			},
			scalingRules:           generateScalingRules(10, 60, 100, 60, 0),
			expectedReplicasChange: 7,
		},
		{
			name:          "two entries, one of them to be replaced",
			replicaChange: 5,
			prevScaleEvents: []timestampedScaleEvent{
				{7, time.Now().Add(-time.Second * time.Duration(61)), false}, // outdated event, should be replaced
				{6, time.Now().Add(-time.Second * time.Duration(59)), false},
			},
			newScaleEvents: []timestampedScaleEvent{
				{5, time.Now(), false},
				{6, time.Now(), false},
			},
			scalingRules:           generateScalingRules(10, 60, 0, 0, 0),
			expectedReplicasChange: 6,
		},
		{
			name:          "replace one entry, use policies with different periods",
			replicaChange: 5,
			prevScaleEvents: []timestampedScaleEvent{
				{8, time.Now().Add(-time.Second * time.Duration(29)), false},
				{6, time.Now().Add(-time.Second * time.Duration(59)), false},
				{7, time.Now().Add(-time.Second * time.Duration(61)), false}, // outdated event, should be marked as outdated
				{9, time.Now().Add(-time.Second * time.Duration(61)), false}, // outdated event, should be replaced
			},
			newScaleEvents: []timestampedScaleEvent{
				{8, time.Now(), false},
				{6, time.Now(), false},
				{7, time.Now(), true},
				{5, time.Now(), false},
			},
			scalingRules:           generateScalingRules(10, 60, 100, 30, 0),
			expectedReplicasChange: 14,
		},
		{
			name:          "two entries, both actual",
			replicaChange: 5,
			prevScaleEvents: []timestampedScaleEvent{
				{7, time.Now().Add(-time.Second * time.Duration(58)), false},
				{6, time.Now().Add(-time.Second * time.Duration(59)), false},
			},
			newScaleEvents: []timestampedScaleEvent{
				{7, time.Now(), false},
				{6, time.Now(), false},
				{5, time.Now(), false},
			},
			scalingRules:           generateScalingRules(10, 120, 100, 30, 0),
			expectedReplicasChange: 13,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			// testing scale up
			var behaviorUp *autoscalingv2.HorizontalPodAutoscalerBehavior
			if tc.scalingRules != nil {
				behaviorUp = &autoscalingv2.HorizontalPodAutoscalerBehavior{
					ScaleUp: tc.scalingRules,
				}
			}
			hcUp := HorizontalController{
				scaleUpEvents: map[string][]timestampedScaleEvent{
					tc.key: append([]timestampedScaleEvent{}, tc.prevScaleEvents...),
				},
			}
			gotReplicasChangeUp := getReplicasChangePerPeriod(60, hcUp.scaleUpEvents[tc.key])
			assert.Equal(t, tc.expectedReplicasChange, gotReplicasChangeUp)
			hcUp.storeScaleEvent(behaviorUp, tc.key, 10, 10+tc.replicaChange)
			if !assert.Len(t, hcUp.scaleUpEvents[tc.key], len(tc.newScaleEvents), "up: scale events differ in length") {
				return
			}
			for i, gotEvent := range hcUp.scaleUpEvents[tc.key] {
				expEvent := tc.newScaleEvents[i]
				assert.Equal(t, expEvent.replicaChange, gotEvent.replicaChange, "up: idx:%v replicaChange", i)
				assert.Equal(t, expEvent.outdated, gotEvent.outdated, "up: idx:%v outdated", i)
			}
			// testing scale down
			var behaviorDown *autoscalingv2.HorizontalPodAutoscalerBehavior
			if tc.scalingRules != nil {
				behaviorDown = &autoscalingv2.HorizontalPodAutoscalerBehavior{
					ScaleDown: tc.scalingRules,
				}
			}
			hcDown := HorizontalController{
				scaleDownEvents: map[string][]timestampedScaleEvent{
					tc.key: append([]timestampedScaleEvent{}, tc.prevScaleEvents...),
				},
			}
			gotReplicasChangeDown := getReplicasChangePerPeriod(60, hcDown.scaleDownEvents[tc.key])
			assert.Equal(t, tc.expectedReplicasChange, gotReplicasChangeDown)
			hcDown.storeScaleEvent(behaviorDown, tc.key, 10, 10-tc.replicaChange)
			if !assert.Len(t, hcDown.scaleDownEvents[tc.key], len(tc.newScaleEvents), "down: scale events differ in length") {
				return
			}
			for i, gotEvent := range hcDown.scaleDownEvents[tc.key] {
				expEvent := tc.newScaleEvents[i]
				assert.Equal(t, expEvent.replicaChange, gotEvent.replicaChange, "down: idx:%v replicaChange", i)
				assert.Equal(t, expEvent.outdated, gotEvent.outdated, "down: idx:%v outdated", i)
			}
		})
	}
}

func TestNormalizeDesiredReplicasWithBehavior(t *testing.T) {
	type TestCase struct {
		name                                string
		key                                 string
		recommendations                     []timestampedRecommendation
		currentReplicas                     int32
		prenormalizedDesiredReplicas        int32
		expectedStabilizedReplicas          int32
		expectedRecommendations             []timestampedRecommendation
		scaleUpStabilizationWindowSeconds   int32
		scaleDownStabilizationWindowSeconds int32
	}
	tests := []TestCase{
		{
			name:                         "empty recommendations for scaling down",
			key:                          "",
			recommendations:              []timestampedRecommendation{},
			currentReplicas:              100,
			prenormalizedDesiredReplicas: 5,
			expectedStabilizedReplicas:   5,
			expectedRecommendations: []timestampedRecommendation{
				{5, time.Now()},
			},
		},
		{
			name: "simple scale down stabilization",
			key:  "",
			recommendations: []timestampedRecommendation{
				{4, time.Now().Add(-2 * time.Minute)},
				{5, time.Now().Add(-1 * time.Minute)}},
			currentReplicas:              100,
			prenormalizedDesiredReplicas: 3,
			expectedStabilizedReplicas:   5,
			expectedRecommendations: []timestampedRecommendation{
				{4, time.Now()},
				{5, time.Now()},
				{3, time.Now()},
			},
			scaleDownStabilizationWindowSeconds: 60 * 3,
		},
		{
			name: "simple scale up stabilization",
			key:  "",
			recommendations: []timestampedRecommendation{
				{4, time.Now().Add(-2 * time.Minute)},
				{5, time.Now().Add(-1 * time.Minute)}},
			currentReplicas:              1,
			prenormalizedDesiredReplicas: 7,
			expectedStabilizedReplicas:   4,
			expectedRecommendations: []timestampedRecommendation{
				{4, time.Now()},
				{5, time.Now()},
				{7, time.Now()},
			},
			scaleUpStabilizationWindowSeconds: 60 * 5,
		},
		{
			name: "no scale down stabilization",
			key:  "",
			recommendations: []timestampedRecommendation{
				{1, time.Now().Add(-2 * time.Minute)},
				{2, time.Now().Add(-1 * time.Minute)}},
			currentReplicas:              100, // to apply scaleDown delay we should have current > desired
			prenormalizedDesiredReplicas: 3,
			expectedStabilizedReplicas:   3,
			expectedRecommendations: []timestampedRecommendation{
				{1, time.Now()},
				{2, time.Now()},
				{3, time.Now()},
			},
			scaleUpStabilizationWindowSeconds: 60 * 5,
		},
		{
			name: "no scale up stabilization",
			key:  "",
			recommendations: []timestampedRecommendation{
				{4, time.Now().Add(-2 * time.Minute)},
				{5, time.Now().Add(-1 * time.Minute)}},
			currentReplicas:              1, // to apply scaleDown delay we should have current > desired
			prenormalizedDesiredReplicas: 3,
			expectedStabilizedReplicas:   3,
			expectedRecommendations: []timestampedRecommendation{
				{4, time.Now()},
				{5, time.Now()},
				{3, time.Now()},
			},
			scaleDownStabilizationWindowSeconds: 60 * 5,
		},
		{
			name: "no scale down stabilization, reuse recommendation element",
			key:  "",
			recommendations: []timestampedRecommendation{
				{10, time.Now().Add(-10 * time.Minute)},
				{9, time.Now().Add(-9 * time.Minute)}},
			currentReplicas:              100, // to apply scaleDown delay we should have current > desired
			prenormalizedDesiredReplicas: 3,
			expectedStabilizedReplicas:   3,
			expectedRecommendations: []timestampedRecommendation{
				{10, time.Now()},
				{3, time.Now()},
			},
		},
		{
			name: "no scale up stabilization, reuse recommendation element",
			key:  "",
			recommendations: []timestampedRecommendation{
				{10, time.Now().Add(-10 * time.Minute)},
				{9, time.Now().Add(-9 * time.Minute)}},
			currentReplicas:              1,
			prenormalizedDesiredReplicas: 100,
			expectedStabilizedReplicas:   100,
			expectedRecommendations: []timestampedRecommendation{
				{10, time.Now()},
				{100, time.Now()},
			},
		},
		{
			name: "scale down stabilization, reuse one of obsolete recommendation element",
			key:  "",
			recommendations: []timestampedRecommendation{
				{10, time.Now().Add(-10 * time.Minute)},
				{4, time.Now().Add(-1 * time.Minute)},
				{5, time.Now().Add(-2 * time.Minute)},
				{9, time.Now().Add(-9 * time.Minute)}},
			currentReplicas:              100,
			prenormalizedDesiredReplicas: 3,
			expectedStabilizedReplicas:   5,
			expectedRecommendations: []timestampedRecommendation{
				{10, time.Now()},
				{4, time.Now()},
				{5, time.Now()},
				{3, time.Now()},
			},
			scaleDownStabilizationWindowSeconds: 3 * 60,
		},
		{
			// we can reuse only the first recommendation element
			// as the scale up delay = 150 (set in test), scale down delay = 300 (by default)
			// hence, only the first recommendation is obsolete for both scale up and scale down
			name: "scale up stabilization, reuse one of obsolete recommendation element",
			key:  "",
			recommendations: []timestampedRecommendation{
				{10, time.Now().Add(-100 * time.Minute)},
				{6, time.Now().Add(-1 * time.Minute)},
				{5, time.Now().Add(-2 * time.Minute)},
				{9, time.Now().Add(-3 * time.Minute)}},
			currentReplicas:              1,
			prenormalizedDesiredReplicas: 100,
			expectedStabilizedReplicas:   5,
			expectedRecommendations: []timestampedRecommendation{
				{100, time.Now()},
				{6, time.Now()},
				{5, time.Now()},
				{9, time.Now()},
			},
			scaleUpStabilizationWindowSeconds: 300,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			hc := HorizontalController{
				recommendations: map[string][]timestampedRecommendation{
					tc.key: tc.recommendations,
				},
			}
			arg := NormalizationArg{
				Key:             tc.key,
				DesiredReplicas: tc.prenormalizedDesiredReplicas,
				CurrentReplicas: tc.currentReplicas,
				ScaleUpBehavior: &autoscalingv2.HPAScalingRules{
					StabilizationWindowSeconds: &tc.scaleUpStabilizationWindowSeconds,
				},
				ScaleDownBehavior: &autoscalingv2.HPAScalingRules{
					StabilizationWindowSeconds: &tc.scaleDownStabilizationWindowSeconds,
				},
			}
			r, _, _ := hc.stabilizeRecommendationWithBehaviors(arg)
			assert.Equal(t, tc.expectedStabilizedReplicas, r, "expected replicas do not match")
			if !assert.Len(t, hc.recommendations[tc.key], len(tc.expectedRecommendations), "stored recommendations differ in length") {
				return
			}
			for i, r := range hc.recommendations[tc.key] {
				expectedRecommendation := tc.expectedRecommendations[i]
				assert.Equal(t, expectedRecommendation.recommendation, r.recommendation, "stored recommendation differs at position %d", i)
			}
		})
	}
}

func TestScaleUpOneMetricEmpty(t *testing.T) {
	tc := testCase{
		minReplicas:             2,
		maxReplicas:             6,
		specReplicas:            3,
		statusReplicas:          3,
		expectedDesiredReplicas: 4,
		CPUTarget:               30,
		verifyCPUCurrent:        true,
		metricsTarget: []autoscalingv2.MetricSpec{
			{
				Type: autoscalingv2.ExternalMetricSourceType,
				External: &autoscalingv2.ExternalMetricSource{
					Metric: autoscalingv2.MetricIdentifier{
						Name:     "qps",
						Selector: &metav1.LabelSelector{},
					},
					Target: autoscalingv2.MetricTarget{
						Value: resource.NewMilliQuantity(100, resource.DecimalSI),
					},
				},
			},
		},
		reportedLevels:      []uint64{300, 400, 500},
		reportedCPURequests: []resource.Quantity{resource.MustParse("1.0"), resource.MustParse("1.0"), resource.MustParse("1.0")},
	}
	_, _, _, testEMClient, _ := tc.prepareTestClient(t)
	testEMClient.PrependReactor("list", "*", func(action core.Action) (handled bool, ret runtime.Object, err error) {
		return true, &emapi.ExternalMetricValueList{}, fmt.Errorf("something went wrong")
	})
	tc.testEMClient = testEMClient
	tc.runTest(t)
}

func TestNoScaleDownOneMetricInvalid(t *testing.T) {
	tc := testCase{
		minReplicas:             2,
		maxReplicas:             6,
		specReplicas:            5,
		statusReplicas:          5,
		expectedDesiredReplicas: 5,
		CPUTarget:               50,
		metricsTarget: []autoscalingv2.MetricSpec{
			{
				Type: "CheddarCheese",
			},
		},
		reportedLevels:      []uint64{100, 300, 500, 250, 250},
		reportedCPURequests: []resource.Quantity{resource.MustParse("1.0"), resource.MustParse("1.0"), resource.MustParse("1.0"), resource.MustParse("1.0"), resource.MustParse("1.0")},
		useMetricsAPI:       true,
		expectedConditions: []autoscalingv1.HorizontalPodAutoscalerCondition{
			{Type: autoscalingv1.AbleToScale, Status: v1.ConditionTrue, Reason: "ScaleDownStabilized"},
			{Type: autoscalingv1.ScalingActive, Status: v1.ConditionTrue, Reason: "ValidMetricFound"},
			{Type: autoscalingv1.ScalingLimited, Status: v1.ConditionFalse, Reason: "DesiredWithinRange"},
		},
	}

	tc.runTest(t)
}

func TestNoScaleDownOneMetricEmpty(t *testing.T) {
	tc := testCase{
		minReplicas:             2,
		maxReplicas:             6,
		specReplicas:            5,
		statusReplicas:          5,
		expectedDesiredReplicas: 5,
		CPUTarget:               50,
		metricsTarget: []autoscalingv2.MetricSpec{
			{
				Type: autoscalingv2.ExternalMetricSourceType,
				External: &autoscalingv2.ExternalMetricSource{
					Metric: autoscalingv2.MetricIdentifier{
						Name:     "qps",
						Selector: &metav1.LabelSelector{},
					},
					Target: autoscalingv2.MetricTarget{
						Value: resource.NewMilliQuantity(1000, resource.DecimalSI),
					},
				},
			},
		},
		reportedLevels:      []uint64{100, 300, 500, 250, 250},
		reportedCPURequests: []resource.Quantity{resource.MustParse("1.0"), resource.MustParse("1.0"), resource.MustParse("1.0"), resource.MustParse("1.0"), resource.MustParse("1.0")},
		useMetricsAPI:       true,
		expectedConditions: []autoscalingv1.HorizontalPodAutoscalerCondition{
			{Type: autoscalingv1.AbleToScale, Status: v1.ConditionTrue, Reason: "ScaleDownStabilized"},
			{Type: autoscalingv1.ScalingActive, Status: v1.ConditionTrue, Reason: "ValidMetricFound"},
			{Type: autoscalingv1.ScalingLimited, Status: v1.ConditionFalse, Reason: "DesiredWithinRange"},
		},
	}
	_, _, _, testEMClient, _ := tc.prepareTestClient(t)
	testEMClient.PrependReactor("list", "*", func(action core.Action) (handled bool, ret runtime.Object, err error) {
		return true, &emapi.ExternalMetricValueList{}, fmt.Errorf("something went wrong")
	})
	tc.testEMClient = testEMClient
	tc.runTest(t)
}
