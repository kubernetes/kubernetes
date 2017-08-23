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

package podtolerationrestriction

import (
	"encoding/json"
	"testing"
	"time"

	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apiserver/pkg/admission"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/kubernetes/pkg/api"
	clientset "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"
	"k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/fake"
	informers "k8s.io/kubernetes/pkg/client/informers/informers_generated/internalversion"
	kubeadmission "k8s.io/kubernetes/pkg/kubeapiserver/admission"
	"k8s.io/kubernetes/pkg/util/tolerations"
	pluginapi "k8s.io/kubernetes/plugin/pkg/admission/podtolerationrestriction/apis/podtolerationrestriction"
	"k8s.io/kubernetes/plugin/pkg/scheduler/algorithm"
)

// TestPodAdmission verifies various scenarios involving pod/namespace tolerations
func TestPodAdmission(t *testing.T) {
	namespace := &api.Namespace{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "testNamespace",
			Namespace: "",
		},
	}

	mockClient := &fake.Clientset{}
	handler, informerFactory, err := newHandlerForTest(mockClient)
	if err != nil {
		t.Errorf("unexpected error initializing handler: %v", err)
	}
	stopCh := make(chan struct{})
	defer close(stopCh)
	informerFactory.Start(stopCh)

	CPU1000m := resource.MustParse("1000m")
	CPU500m := resource.MustParse("500m")

	burstablePod := &api.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: "testPod", Namespace: "testNamespace"},
		Spec: api.PodSpec{
			Containers: []api.Container{
				{
					Name: "test",
					Resources: api.ResourceRequirements{
						Limits:   api.ResourceList{api.ResourceCPU: CPU1000m},
						Requests: api.ResourceList{api.ResourceCPU: CPU500m},
					},
				},
			},
		},
	}

	guaranteedPod := &api.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: "testPod", Namespace: "testNamespace"},
		Spec: api.PodSpec{
			Containers: []api.Container{
				{
					Name: "test",
					Resources: api.ResourceRequirements{
						Limits:   api.ResourceList{api.ResourceCPU: CPU1000m},
						Requests: api.ResourceList{api.ResourceCPU: CPU1000m},
					},
				},
			},
		},
	}

	bestEffortPod := &api.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: "testPod", Namespace: "testNamespace"},
		Spec: api.PodSpec{
			Containers: []api.Container{
				{
					Name: "test",
				},
			},
		},
	}

	if err := utilfeature.DefaultFeatureGate.Set("TaintNodesByCondition=true"); err != nil {
		t.Errorf("Failed to enable TaintByCondition feature: %v.", err)
	}

	tests := []struct {
		pod                       *api.Pod
		defaultClusterTolerations []api.Toleration
		namespaceTolerations      []api.Toleration
		whitelist                 []api.Toleration
		clusterWhitelist          []api.Toleration
		podTolerations            []api.Toleration
		mergedTolerations         []api.Toleration
		admit                     bool
		testName                  string
	}{
		{
			pod: bestEffortPod,
			defaultClusterTolerations: []api.Toleration{{Key: "testKey", Operator: "Equal", Value: "testValue", Effect: "NoSchedule", TolerationSeconds: nil}},
			namespaceTolerations:      []api.Toleration{},
			podTolerations:            []api.Toleration{},
			mergedTolerations:         []api.Toleration{{Key: "testKey", Operator: "Equal", Value: "testValue", Effect: "NoSchedule", TolerationSeconds: nil}},
			admit:                     true,
			testName:                  "default cluster tolerations with empty pod tolerations",
		},
		{
			pod: bestEffortPod,
			defaultClusterTolerations: []api.Toleration{{Key: "testKey", Operator: "Equal", Value: "testValue", Effect: "NoSchedule", TolerationSeconds: nil}},
			namespaceTolerations:      []api.Toleration{},
			podTolerations:            []api.Toleration{{Key: "testKey", Operator: "Equal", Value: "testValue", Effect: "NoSchedule", TolerationSeconds: nil}},
			mergedTolerations:         []api.Toleration{{Key: "testKey", Operator: "Equal", Value: "testValue", Effect: "NoSchedule", TolerationSeconds: nil}},
			admit:                     true,
			testName:                  "default cluster tolerations with pod tolerations specified",
		},
		{
			pod: bestEffortPod,
			defaultClusterTolerations: []api.Toleration{},
			namespaceTolerations:      []api.Toleration{{Key: "testKey", Operator: "Equal", Value: "testValue", Effect: "NoSchedule", TolerationSeconds: nil}},
			podTolerations:            []api.Toleration{{Key: "testKey", Operator: "Equal", Value: "testValue", Effect: "NoSchedule", TolerationSeconds: nil}},
			mergedTolerations:         []api.Toleration{{Key: "testKey", Operator: "Equal", Value: "testValue", Effect: "NoSchedule", TolerationSeconds: nil}},
			admit:                     true,
			testName:                  "namespace tolerations",
		},
		{
			pod: bestEffortPod,
			defaultClusterTolerations: []api.Toleration{},
			namespaceTolerations:      []api.Toleration{{Key: "testKey", Operator: "Equal", Value: "testValue", Effect: "NoSchedule", TolerationSeconds: nil}},
			podTolerations:            []api.Toleration{},
			mergedTolerations:         []api.Toleration{{Key: "testKey", Operator: "Equal", Value: "testValue", Effect: "NoSchedule", TolerationSeconds: nil}},
			admit:                     true,
			testName:                  "no pod tolerations",
		},
		{
			pod: bestEffortPod,
			defaultClusterTolerations: []api.Toleration{},
			namespaceTolerations:      []api.Toleration{{Key: "testKey", Operator: "Equal", Value: "testValue", Effect: "NoSchedule", TolerationSeconds: nil}},
			podTolerations:            []api.Toleration{{Key: "testKey", Operator: "Equal", Value: "testValue1", Effect: "NoSchedule", TolerationSeconds: nil}},
			admit:                     false,
			testName:                  "conflicting pod and namespace tolerations",
		},
		{
			pod: bestEffortPod,
			defaultClusterTolerations: []api.Toleration{{Key: "testKey", Operator: "Equal", Value: "testValue2", Effect: "NoSchedule", TolerationSeconds: nil}},
			namespaceTolerations:      []api.Toleration{},
			podTolerations:            []api.Toleration{{Key: "testKey", Operator: "Equal", Value: "testValue1", Effect: "NoSchedule", TolerationSeconds: nil}},
			admit:                     false,
			testName:                  "conflicting pod and default cluster tolerations",
		},
		{
			pod: bestEffortPod,
			defaultClusterTolerations: []api.Toleration{},
			namespaceTolerations:      []api.Toleration{{Key: "testKey", Operator: "Equal", Value: "testValue", Effect: "NoSchedule", TolerationSeconds: nil}},
			whitelist:                 []api.Toleration{{Key: "testKey", Operator: "Equal", Value: "testValue", Effect: "NoSchedule", TolerationSeconds: nil}},
			podTolerations:            []api.Toleration{},
			mergedTolerations:         []api.Toleration{{Key: "testKey", Operator: "Equal", Value: "testValue", Effect: "NoSchedule", TolerationSeconds: nil}},
			admit:                     true,
			testName:                  "merged pod tolerations satisfy whitelist",
		},
		{
			pod: bestEffortPod,
			defaultClusterTolerations: []api.Toleration{},
			namespaceTolerations:      []api.Toleration{{Key: "testKey", Operator: "Equal", Value: "testValue", Effect: "NoSchedule", TolerationSeconds: nil}},
			whitelist:                 []api.Toleration{{Key: "testKey", Operator: "Equal", Value: "testValue1", Effect: "NoSchedule", TolerationSeconds: nil}},
			podTolerations:            []api.Toleration{},
			admit:                     false,
			testName:                  "merged pod tolerations conflict with the whitelist",
		},
		{
			pod: burstablePod,
			defaultClusterTolerations: []api.Toleration{},
			namespaceTolerations:      []api.Toleration{{Key: "testKey", Operator: "Equal", Value: "testValue", Effect: "NoSchedule", TolerationSeconds: nil}},
			whitelist:                 []api.Toleration{},
			podTolerations:            []api.Toleration{},
			mergedTolerations: []api.Toleration{
				{Key: algorithm.TaintNodeMemoryPressure, Operator: api.TolerationOpExists, Effect: api.TaintEffectNoSchedule, TolerationSeconds: nil},
				{Key: "testKey", Operator: "Equal", Value: "testValue", Effect: "NoSchedule", TolerationSeconds: nil},
			},
			admit:    true,
			testName: "added memoryPressure/DiskPressure for Burstable pod",
		},
		{
			pod: guaranteedPod,
			defaultClusterTolerations: []api.Toleration{},
			namespaceTolerations:      []api.Toleration{{Key: "testKey", Operator: "Equal", Value: "testValue", Effect: "NoSchedule", TolerationSeconds: nil}},
			whitelist:                 []api.Toleration{},
			podTolerations:            []api.Toleration{},
			mergedTolerations: []api.Toleration{
				{Key: algorithm.TaintNodeMemoryPressure, Operator: api.TolerationOpExists, Effect: api.TaintEffectNoSchedule, TolerationSeconds: nil},
				{Key: "testKey", Operator: "Equal", Value: "testValue", Effect: "NoSchedule", TolerationSeconds: nil},
			},
			admit:    true,
			testName: "added memoryPressure/DiskPressure for Guaranteed pod",
		},
	}
	for _, test := range tests {
		if len(test.namespaceTolerations) > 0 {
			tolerationStr, err := json.Marshal(test.namespaceTolerations)
			if err != nil {
				t.Errorf("error in marshalling namespace tolerations %v", test.namespaceTolerations)
			}
			namespace.Annotations = map[string]string{NSDefaultTolerations: string(tolerationStr)}
		}

		if len(test.whitelist) > 0 {
			tolerationStr, err := json.Marshal(test.whitelist)
			if err != nil {
				t.Errorf("error in marshalling namespace whitelist %v", test.whitelist)
			}
			namespace.Annotations[NSWLTolerations] = string(tolerationStr)
		}

		informerFactory.Core().InternalVersion().Namespaces().Informer().GetStore().Update(namespace)

		handler.pluginConfig = &pluginapi.Configuration{Default: test.defaultClusterTolerations, Whitelist: test.clusterWhitelist}
		pod := test.pod
		pod.Spec.Tolerations = test.podTolerations

		// copy the original pod for tests of uninitialized pod updates.
		oldPod := *pod
		oldPod.Initializers = &metav1.Initializers{Pending: []metav1.Initializer{{Name: "init"}}}
		oldPod.Spec.Tolerations = []api.Toleration{{Key: "testKey", Operator: "Equal", Value: "testValue1", Effect: "NoSchedule", TolerationSeconds: nil}}

		err := handler.Admit(admission.NewAttributesRecord(pod, nil, api.Kind("Pod").WithVersion("version"), "testNamespace", namespace.ObjectMeta.Name, api.Resource("pods").WithVersion("version"), "", admission.Create, nil))
		if test.admit && err != nil {
			t.Errorf("Test: %s, expected no error but got: %s", test.testName, err)
		} else if !test.admit && err == nil {
			t.Errorf("Test: %s, expected an error", test.testName)
		}

		updatedPodTolerations := pod.Spec.Tolerations
		if test.admit && !tolerations.EqualTolerations(updatedPodTolerations, test.mergedTolerations) {
			t.Errorf("Test: %s, expected: %#v but got: %#v", test.testName, test.mergedTolerations, updatedPodTolerations)
		}

		// handles update of uninitialized pod like it's newly created.
		err = handler.Admit(admission.NewAttributesRecord(pod, &oldPod, api.Kind("Pod").WithVersion("version"), "testNamespace", namespace.ObjectMeta.Name, api.Resource("pods").WithVersion("version"), "", admission.Update, nil))
		if test.admit && err != nil {
			t.Errorf("Test: %s, expected no error but got: %s", test.testName, err)
		} else if !test.admit && err == nil {
			t.Errorf("Test: %s, expected an error", test.testName)
		}

		updatedPodTolerations = pod.Spec.Tolerations
		if test.admit && !tolerations.EqualTolerations(updatedPodTolerations, test.mergedTolerations) {
			t.Errorf("Test: %s, expected: %#v but got: %#v", test.testName, test.mergedTolerations, updatedPodTolerations)
		}
	}
}

func TestHandles(t *testing.T) {
	for op, shouldHandle := range map[admission.Operation]bool{
		admission.Create:  true,
		admission.Update:  true,
		admission.Connect: false,
		admission.Delete:  false,
	} {

		pluginConfig, err := loadConfiguration(nil)
		// must not fail
		if err != nil {
			t.Errorf("%v: error reading default configuration", op)
		}
		ptPlugin := NewPodTolerationsPlugin(pluginConfig)
		if e, a := shouldHandle, ptPlugin.Handles(op); e != a {
			t.Errorf("%v: shouldHandle=%t, handles=%t", op, e, a)
		}
	}
}

func TestIgnoreUpdatingInitializedPod(t *testing.T) {
	mockClient := &fake.Clientset{}
	handler, informerFactory, err := newHandlerForTest(mockClient)
	if err != nil {
		t.Errorf("unexpected error initializing handler: %v", err)
	}
	handler.SetReadyFunc(func() bool { return true })

	pod := &api.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: "testPod", Namespace: "testNamespace"},
		Spec:       api.PodSpec{},
	}
	podToleration := api.Toleration{
		Key:               "testKey",
		Operator:          "Equal",
		Value:             "testValue1",
		Effect:            "NoSchedule",
		TolerationSeconds: nil,
	}
	pod.Spec.Tolerations = []api.Toleration{podToleration}

	// this conflicts with pod's Tolerations
	namespaceToleration := podToleration
	namespaceToleration.Value = "testValue2"
	namespaceTolerations := []api.Toleration{namespaceToleration}
	tolerationsStr, err := json.Marshal(namespaceTolerations)
	if err != nil {
		t.Errorf("error in marshalling namespace tolerations %v", namespaceTolerations)
	}
	namespace := &api.Namespace{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "testNamespace",
			Namespace: "",
		},
	}
	namespace.Annotations = map[string]string{NSDefaultTolerations: string(tolerationsStr)}
	err = informerFactory.Core().InternalVersion().Namespaces().Informer().GetStore().Update(namespace)
	if err != nil {
		t.Fatal(err)
	}

	// if the update of initialized pod is not ignored, an error will be returned because the pod's Tolerations conflicts with namespace's Tolerations.
	err = handler.Admit(admission.NewAttributesRecord(pod, pod, api.Kind("Pod").WithVersion("version"), "testNamespace", pod.ObjectMeta.Name, api.Resource("pods").WithVersion("version"), "", admission.Update, nil))
	if err != nil {
		t.Errorf("expected no error, got: %v", err)
	}
}

// newHandlerForTest returns the admission controller configured for testing.
func newHandlerForTest(c clientset.Interface) (*podTolerationsPlugin, informers.SharedInformerFactory, error) {
	f := informers.NewSharedInformerFactory(c, 5*time.Minute)
	pluginConfig, err := loadConfiguration(nil)
	// must not fail
	if err != nil {
		return nil, nil, err
	}
	handler := NewPodTolerationsPlugin(pluginConfig)
	pluginInitializer := kubeadmission.NewPluginInitializer(c, nil, f, nil, nil, nil, nil)
	pluginInitializer.Initialize(handler)
	err = admission.Validate(handler)
	return handler, f, err
}
