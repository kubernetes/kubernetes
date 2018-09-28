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
	"k8s.io/api/core/v1"
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
	"k8s.io/kubernetes/pkg/controller"
	"k8s.io/kubernetes/pkg/controller/podautoscaler/metrics"
	cmapi "k8s.io/metrics/pkg/apis/custom_metrics/v1beta2"
	emapi "k8s.io/metrics/pkg/apis/external_metrics/v1beta1"
	metricsapi "k8s.io/metrics/pkg/apis/metrics/v1beta1"
	metricsfake "k8s.io/metrics/pkg/client/clientset/versioned/fake"
	cmfake "k8s.io/metrics/pkg/client/custom_metrics/fake"
	emfake "k8s.io/metrics/pkg/client/external_metrics/fake"

	"github.com/stretchr/testify/assert"

	_ "k8s.io/kubernetes/pkg/apis/autoscaling/install"
	_ "k8s.io/kubernetes/pkg/apis/extensions/install"
)

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
	initialReplicas int32

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

		obj := &autoscalingv2.HorizontalPodAutoscalerList{
			Items: []autoscalingv2.HorizontalPodAutoscaler{
				{
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
					},
					Status: autoscalingv2.HorizontalPodAutoscalerStatus{
						CurrentReplicas: tc.initialReplicas,
						DesiredReplicas: tc.initialReplicas,
						LastScaleTime:   tc.lastScaleTime,
					},
				},
			},
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

		numPodsToCreate := int(tc.initialReplicas)
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
							Resources: v1.ResourceRequirements{
								Requests: v1.ResourceList{
									v1.ResourceCPU: reportedCPURequest,
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
		// TODO: it's ok not to sort these becaues statusOk
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
		tc.processed <- obj.Name
		return true, obj, nil
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
				Replicas: tc.initialReplicas,
			},
			Status: autoscalingv1.ScaleStatus{
				Replicas: tc.initialReplicas,
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
				Replicas: tc.initialReplicas,
			},
			Status: autoscalingv1.ScaleStatus{
				Replicas: tc.initialReplicas,
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
				Replicas: tc.initialReplicas,
			},
			Status: autoscalingv1.ScaleStatus{
				Replicas: tc.initialReplicas,
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
						Name: "container",
						Usage: v1.ResourceList{
							v1.ResourceCPU: *resource.NewMilliQuantity(
								int64(cpu),
								resource.DecimalSI),
							v1.ResourceMemory: *resource.NewQuantity(
								int64(1024*1024),
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

	assert.Equal(t, tc.initialReplicas != tc.expectedDesiredReplicas, tc.scaleUpdated, "the scale should only be updated if we expected a change in replicas")
	assert.True(t, tc.statusUpdated, "the status should have been updated")
	if tc.verifyEvents {
		assert.Equal(t, tc.initialReplicas != tc.expectedDesiredReplicas, tc.eventCreated, "an event should have been created only if we expected a change in replicas")
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
					(int64(tc.reportedLevels[0])*100)/tc.reportedCPURequests[0].MilliValue(), tc.initialReplicas), obj.Message)
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
		eventClient.Core(),
		testScaleClient,
		testClient.Autoscaling(),
		testrestmapper.TestOnlyStaticRESTMapper(legacyscheme.Scheme),
		metricsClient,
		informerFactory.Autoscaling().V1().HorizontalPodAutoscalers(),
		informerFactory.Core().V1().Pods(),
		controller.NoResyncPeriodFunc(),
		defaultDownscalestabilizationWindow,
		defaultTestingTolerance,
		defaultTestingCpuInitializationPeriod,
		defaultTestingDelayOfInitialReadinessStatus,
	)
	hpaController.hpaListerSynced = alwaysReady
	if tc.recommendations != nil {
		hpaController.recommendations["test-namespace/test-hpa"] = tc.recommendations
	}

	return hpaController, informerFactory
}

func hotCpuCreationTime() metav1.Time {
	return metav1.Time{Time: time.Now()}
}

func coolCpuCreationTime() metav1.Time {
	return metav1.Time{Time: time.Now().Add(-3 * time.Minute)}
}

func (tc *testCase) runTestWithController(t *testing.T, hpaController *HorizontalController, informerFactory informers.SharedInformerFactory) {
	stop := make(chan struct{})
	defer close(stop)
	informerFactory.Start(stop)
	go hpaController.Run(stop)

	tc.Lock()
	if tc.verifyEvents {
		tc.Unlock()
		// We need to wait for events to be broadcasted (sleep for longer than record.sleepDuration).
		time.Sleep(2 * time.Second)
	} else {
		tc.Unlock()
	}
	// Wait for HPA to be processed.
	<-tc.processed
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
		initialReplicas:         3,
		expectedDesiredReplicas: 5,
		CPUTarget:               30,
		verifyCPUCurrent:        true,
		reportedLevels:          []uint64{300, 500, 700},
		reportedCPURequests:     []resource.Quantity{resource.MustParse("1.0"), resource.MustParse("1.0"), resource.MustParse("1.0")},
		useMetricsAPI:           true,
	}
	tc.runTest(t)
}

func TestScaleUpUnreadyLessScale(t *testing.T) {
	tc := testCase{
		minReplicas:             2,
		maxReplicas:             6,
		initialReplicas:         3,
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
		initialReplicas:         3,
		expectedDesiredReplicas: 4,
		CPUTarget:               30,
		CPUCurrent:              60,
		verifyCPUCurrent:        true,
		reportedLevels:          []uint64{300, 500, 700},
		reportedCPURequests:     []resource.Quantity{resource.MustParse("1.0"), resource.MustParse("1.0"), resource.MustParse("1.0")},
		reportedPodStartTime:    []metav1.Time{hotCpuCreationTime(), coolCpuCreationTime(), coolCpuCreationTime()},
		useMetricsAPI:           true,
	}
	tc.runTest(t)
}

func TestScaleUpUnreadyNoScale(t *testing.T) {
	tc := testCase{
		minReplicas:             2,
		maxReplicas:             6,
		initialReplicas:         3,
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
		initialReplicas:         3,
		expectedDesiredReplicas: 3,
		CPUTarget:               30,
		CPUCurrent:              40,
		verifyCPUCurrent:        true,
		reportedLevels:          []uint64{400, 500, 700},
		reportedCPURequests:     []resource.Quantity{resource.MustParse("1.0"), resource.MustParse("1.0"), resource.MustParse("1.0")},
		reportedPodReadiness:    []v1.ConditionStatus{v1.ConditionTrue, v1.ConditionFalse, v1.ConditionFalse},
		reportedPodStartTime:    []metav1.Time{coolCpuCreationTime(), hotCpuCreationTime(), hotCpuCreationTime()},
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
		initialReplicas:         2,
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
		initialReplicas:              2,
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
		initialReplicas:         3,
		expectedDesiredReplicas: 5,
		CPUTarget:               30,
		verifyCPUCurrent:        true,
		reportedLevels:          []uint64{300, 500, 700},
		reportedCPURequests:     []resource.Quantity{resource.MustParse("1.0"), resource.MustParse("1.0"), resource.MustParse("1.0")},
		useMetricsAPI:           true,
		resource: &fakeResource{
			name:       "test-dep",
			apiVersion: "extensions/v1beta1",
			kind:       "Deployment",
		},
	}
	tc.runTest(t)
}

func TestScaleUpReplicaSet(t *testing.T) {
	tc := testCase{
		minReplicas:             2,
		maxReplicas:             6,
		initialReplicas:         3,
		expectedDesiredReplicas: 5,
		CPUTarget:               30,
		verifyCPUCurrent:        true,
		reportedLevels:          []uint64{300, 500, 700},
		reportedCPURequests:     []resource.Quantity{resource.MustParse("1.0"), resource.MustParse("1.0"), resource.MustParse("1.0")},
		useMetricsAPI:           true,
		resource: &fakeResource{
			name:       "test-replicaset",
			apiVersion: "extensions/v1beta1",
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
		initialReplicas:         3,
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
		initialReplicas:         3,
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
		reportedPodStartTime: []metav1.Time{coolCpuCreationTime(), coolCpuCreationTime(), hotCpuCreationTime()},
		reportedCPURequests:  []resource.Quantity{resource.MustParse("1.0"), resource.MustParse("1.0"), resource.MustParse("1.0")},
	}
	tc.runTest(t)
}

func TestScaleUpCMUnreadyandCpuHot(t *testing.T) {
	averageValue := resource.MustParse("15.0")
	tc := testCase{
		minReplicas:             2,
		maxReplicas:             6,
		initialReplicas:         3,
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
		reportedPodStartTime: []metav1.Time{hotCpuCreationTime(), coolCpuCreationTime(), hotCpuCreationTime()},
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
		initialReplicas:         3,
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
		reportedPodStartTime: []metav1.Time{hotCpuCreationTime(), coolCpuCreationTime(), hotCpuCreationTime()},
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
		initialReplicas:         3,
		expectedDesiredReplicas: 4,
		CPUTarget:               0,
		metricsTarget: []autoscalingv2.MetricSpec{
			{
				Type: autoscalingv2.ObjectMetricSourceType,
				Object: &autoscalingv2.ObjectMetricSource{
					DescribedObject: autoscalingv2.CrossVersionObjectReference{
						APIVersion: "extensions/v1beta1",
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

func TestScaleUpCMExternal(t *testing.T) {
	tc := testCase{
		minReplicas:             2,
		maxReplicas:             6,
		initialReplicas:         3,
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
		initialReplicas:         3,
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
		initialReplicas:         5,
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

func TestScaleDownStabilizeInitialSize(t *testing.T) {
	tc := testCase{
		minReplicas:             2,
		maxReplicas:             6,
		initialReplicas:         5,
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
		initialReplicas:         5,
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
		initialReplicas:         5,
		expectedDesiredReplicas: 3,
		CPUTarget:               0,
		metricsTarget: []autoscalingv2.MetricSpec{
			{
				Type: autoscalingv2.ObjectMetricSourceType,
				Object: &autoscalingv2.ObjectMetricSource{
					DescribedObject: autoscalingv2.CrossVersionObjectReference{
						APIVersion: "extensions/v1beta1",
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

func TestScaleDownCMExternal(t *testing.T) {
	tc := testCase{
		minReplicas:             2,
		maxReplicas:             6,
		initialReplicas:         5,
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

func TestScaleDownPerPodCMExternal(t *testing.T) {
	tc := testCase{
		minReplicas:             2,
		maxReplicas:             6,
		initialReplicas:         5,
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
		initialReplicas:         5,
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

func TestScaleDownIgnoreHotCpuPods(t *testing.T) {
	tc := testCase{
		minReplicas:             2,
		maxReplicas:             6,
		initialReplicas:         5,
		expectedDesiredReplicas: 2,
		CPUTarget:               50,
		CPUCurrent:              30,
		verifyCPUCurrent:        true,
		reportedLevels:          []uint64{100, 300, 500, 250, 250},
		reportedCPURequests:     []resource.Quantity{resource.MustParse("1.0"), resource.MustParse("1.0"), resource.MustParse("1.0"), resource.MustParse("1.0"), resource.MustParse("1.0")},
		useMetricsAPI:           true,
		reportedPodStartTime:    []metav1.Time{coolCpuCreationTime(), coolCpuCreationTime(), coolCpuCreationTime(), hotCpuCreationTime(), hotCpuCreationTime()},
		recommendations:         []timestampedRecommendation{},
	}
	tc.runTest(t)
}

func TestScaleDownIgnoresFailedPods(t *testing.T) {
	tc := testCase{
		minReplicas:             2,
		maxReplicas:             6,
		initialReplicas:         5,
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
		initialReplicas:              5,
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
		initialReplicas:         3,
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
		initialReplicas:         3,
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
		initialReplicas:         3,
		expectedDesiredReplicas: 3,
		metricsTarget: []autoscalingv2.MetricSpec{
			{
				Type: autoscalingv2.ObjectMetricSourceType,
				Object: &autoscalingv2.ObjectMetricSource{
					DescribedObject: autoscalingv2.CrossVersionObjectReference{
						APIVersion: "extensions/v1beta1",
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
		initialReplicas:         4,
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

func TestTolerancePerPodCMExternal(t *testing.T) {
	tc := testCase{
		minReplicas:             2,
		maxReplicas:             6,
		initialReplicas:         4,
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
		initialReplicas:         3,
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

func TestMinReplicasDesiredZero(t *testing.T) {
	tc := testCase{
		minReplicas:             2,
		maxReplicas:             5,
		initialReplicas:         3,
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
		initialReplicas:         0,
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
		initialReplicas:         2,
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
		initialReplicas:         10,
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
		initialReplicas:         3,
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
		initialReplicas:         4,
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
		initialReplicas:         4,
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
		initialReplicas:         4,
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
		initialReplicas:         1,
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
		initialReplicas:         1,
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
		initialReplicas:         2,
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
		initialReplicas:         4,
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

func TestConditionInvalidSelectorMissing(t *testing.T) {
	tc := testCase{
		minReplicas:             1,
		maxReplicas:             100,
		initialReplicas:         3,
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
				Replicas: tc.initialReplicas,
			},
			Status: autoscalingv1.ScaleStatus{
				Replicas: tc.initialReplicas,
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
		initialReplicas:         3,
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
				Replicas: tc.initialReplicas,
			},
			Status: autoscalingv1.ScaleStatus{
				Replicas: tc.initialReplicas,
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
						APIVersion: "extensions/v1beta1",
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
			initialReplicas:         3,
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
		initialReplicas:         3,
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
		initialReplicas:         3,
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
		initialReplicas:         3,
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

func NoTestBackoffUpscale(t *testing.T) {
	time := metav1.Time{Time: time.Now()}
	tc := testCase{
		minReplicas:             1,
		maxReplicas:             5,
		initialReplicas:         3,
		expectedDesiredReplicas: 3,
		CPUTarget:               100,
		reportedLevels:          []uint64{150, 150, 150},
		reportedCPURequests:     []resource.Quantity{resource.MustParse("0.1"), resource.MustParse("0.1"), resource.MustParse("0.1")},
		useMetricsAPI:           true,
		lastScaleTime:           &time,
		expectedConditions: statusOkWithOverrides(autoscalingv2.HorizontalPodAutoscalerCondition{
			Type:   autoscalingv2.AbleToScale,
			Status: v1.ConditionTrue,
			Reason: "ReadyForNewScale",
		}, autoscalingv2.HorizontalPodAutoscalerCondition{
			Type:   autoscalingv2.AbleToScale,
			Status: v1.ConditionTrue,
			Reason: "SucceededRescale",
		}),
	}
	tc.runTest(t)
}

func TestNoBackoffUpscaleCM(t *testing.T) {
	averageValue := resource.MustParse("15.0")
	time := metav1.Time{Time: time.Now()}
	tc := testCase{
		minReplicas:             1,
		maxReplicas:             5,
		initialReplicas:         3,
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
		initialReplicas:         3,
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
		initialReplicas:         4,
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
	tc := testCase{
		minReplicas:             0,
		maxReplicas:             1000,
		initialReplicas:         startPods,
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

	tc.runTest(t)

	// Reuse the data structure above, now testing "unscaling".
	// Now, we test that no scaling happens if we are in a very close margin to the tolerance
	target = math.Abs(1/(requestedToUsed*(1-defaultTestingTolerance))) + .004
	finalCPUPercentTarget = int32(target * 100)
	tc.CPUTarget = finalCPUPercentTarget
	tc.initialReplicas = startPods
	tc.expectedDesiredReplicas = startPods
	tc.expectedConditions = statusOkWithOverrides(autoscalingv2.HorizontalPodAutoscalerCondition{
		Type:   autoscalingv2.AbleToScale,
		Status: v1.ConditionTrue,
		Reason: "ReadyForNewScale",
	})
	tc.runTest(t)
}

func TestScaleUpRCImmediately(t *testing.T) {
	time := metav1.Time{Time: time.Now()}
	tc := testCase{
		minReplicas:             2,
		maxReplicas:             6,
		initialReplicas:         1,
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
		initialReplicas:         6,
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

func TestAvoidUncessaryUpdates(t *testing.T) {
	now := metav1.Time{Time: time.Now().Add(-time.Hour)}
	tc := testCase{
		minReplicas:             2,
		maxReplicas:             6,
		initialReplicas:         2,
		expectedDesiredReplicas: 2,
		CPUTarget:               30,
		CPUCurrent:              40,
		verifyCPUCurrent:        true,
		reportedLevels:          []uint64{400, 500, 700},
		reportedCPURequests:     []resource.Quantity{resource.MustParse("1.0"), resource.MustParse("1.0"), resource.MustParse("1.0")},
		reportedPodStartTime:    []metav1.Time{coolCpuCreationTime(), hotCpuCreationTime(), hotCpuCreationTime()},
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
			tc.statusUpdated = true
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
						CurrentReplicas: tc.initialReplicas,
						DesiredReplicas: tc.initialReplicas,
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
								Message:            "the desired replica count is more than the maximum replica count",
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
		tc.Lock()
		defer tc.Unlock()
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
			expectedConvertedDesiredReplicas: 1,
			expectedCondition:                "TooFewReplicas",
			annotation:                       "1 is minLimit because hpaMinReplicas < 1",
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
		actualConvertedDesiredReplicas, actualCondition, _ := convertDesiredReplicasWithRules(
			ctc.currentReplicas, ctc.expectedDesiredReplicas, ctc.hpaMinReplicas, ctc.hpaMaxReplicas,
		)

		assert.Equal(t, ctc.expectedConvertedDesiredReplicas, actualConvertedDesiredReplicas, ctc.annotation)
		assert.Equal(t, ctc.expectedCondition, actualCondition, ctc.annotation)
	}
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

// TODO: add more tests
