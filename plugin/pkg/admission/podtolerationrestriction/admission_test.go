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

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apiserver/pkg/admission"
	"k8s.io/kubernetes/pkg/api"
	clientset "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"
	"k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/fake"
	informers "k8s.io/kubernetes/pkg/client/informers/informers_generated/internalversion"
	kubeadmission "k8s.io/kubernetes/pkg/kubeapiserver/admission"
	"k8s.io/kubernetes/pkg/util/tolerations"
	pluginapi "k8s.io/kubernetes/plugin/pkg/admission/podtolerationrestriction/apis/podtolerationrestriction"
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

	pod := &api.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: "testPod", Namespace: "testNamespace"},
	}

	tests := []struct {
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
			defaultClusterTolerations: []api.Toleration{{Key: "testKey", Operator: "Equal", Value: "testValue", Effect: "NoSchedule", TolerationSeconds: nil}},
			namespaceTolerations:      []api.Toleration{},
			podTolerations:            []api.Toleration{},
			mergedTolerations:         []api.Toleration{{Key: "testKey", Operator: "Equal", Value: "testValue", Effect: "NoSchedule", TolerationSeconds: nil}},
			admit:                     true,
			testName:                  "default cluster tolerations with empty pod tolerations",
		},
		{
			defaultClusterTolerations: []api.Toleration{{Key: "testKey", Operator: "Equal", Value: "testValue", Effect: "NoSchedule", TolerationSeconds: nil}},
			namespaceTolerations:      []api.Toleration{},
			podTolerations:            []api.Toleration{{Key: "testKey", Operator: "Equal", Value: "testValue", Effect: "NoSchedule", TolerationSeconds: nil}},
			mergedTolerations:         []api.Toleration{{Key: "testKey", Operator: "Equal", Value: "testValue", Effect: "NoSchedule", TolerationSeconds: nil}},
			admit:                     true,
			testName:                  "default cluster tolerations with pod tolerations specified",
		},
		{
			defaultClusterTolerations: []api.Toleration{},
			namespaceTolerations:      []api.Toleration{{Key: "testKey", Operator: "Equal", Value: "testValue", Effect: "NoSchedule", TolerationSeconds: nil}},
			podTolerations:            []api.Toleration{{Key: "testKey", Operator: "Equal", Value: "testValue", Effect: "NoSchedule", TolerationSeconds: nil}},
			mergedTolerations:         []api.Toleration{{Key: "testKey", Operator: "Equal", Value: "testValue", Effect: "NoSchedule", TolerationSeconds: nil}},
			admit:                     true,
			testName:                  "namespace tolerations",
		},
		{
			defaultClusterTolerations: []api.Toleration{},
			namespaceTolerations:      []api.Toleration{{Key: "testKey", Operator: "Equal", Value: "testValue", Effect: "NoSchedule", TolerationSeconds: nil}},
			podTolerations:            []api.Toleration{},
			mergedTolerations:         []api.Toleration{{Key: "testKey", Operator: "Equal", Value: "testValue", Effect: "NoSchedule", TolerationSeconds: nil}},
			admit:                     true,
			testName:                  "no pod tolerations",
		},
		{
			defaultClusterTolerations: []api.Toleration{},
			namespaceTolerations:      []api.Toleration{{Key: "testKey", Operator: "Equal", Value: "testValue", Effect: "NoSchedule", TolerationSeconds: nil}},
			podTolerations:            []api.Toleration{{Key: "testKey", Operator: "Equal", Value: "testValue1", Effect: "NoSchedule", TolerationSeconds: nil}},
			admit:                     false,
			testName:                  "conflicting pod and namespace tolerations",
		},
		{
			defaultClusterTolerations: []api.Toleration{{Key: "testKey", Operator: "Equal", Value: "testValue2", Effect: "NoSchedule", TolerationSeconds: nil}},
			namespaceTolerations:      []api.Toleration{},
			podTolerations:            []api.Toleration{{Key: "testKey", Operator: "Equal", Value: "testValue1", Effect: "NoSchedule", TolerationSeconds: nil}},
			admit:                     false,
			testName:                  "conflicting pod and default cluster tolerations",
		},
		{
			defaultClusterTolerations: []api.Toleration{},
			namespaceTolerations:      []api.Toleration{{Key: "testKey", Operator: "Equal", Value: "testValue", Effect: "NoSchedule", TolerationSeconds: nil}},
			whitelist:                 []api.Toleration{{Key: "testKey", Operator: "Equal", Value: "testValue", Effect: "NoSchedule", TolerationSeconds: nil}},
			podTolerations:            []api.Toleration{},
			mergedTolerations:         []api.Toleration{{Key: "testKey", Operator: "Equal", Value: "testValue", Effect: "NoSchedule", TolerationSeconds: nil}},
			admit:                     true,
			testName:                  "merged pod tolerations satisfy whitelist",
		},
		{
			defaultClusterTolerations: []api.Toleration{},
			namespaceTolerations:      []api.Toleration{{Key: "testKey", Operator: "Equal", Value: "testValue", Effect: "NoSchedule", TolerationSeconds: nil}},
			whitelist:                 []api.Toleration{{Key: "testKey", Operator: "Equal", Value: "testValue1", Effect: "NoSchedule", TolerationSeconds: nil}},
			podTolerations:            []api.Toleration{},
			admit:                     false,
			testName:                  "merged pod tolerations conflict with the whitelist",
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
		pod.Spec.Tolerations = test.podTolerations

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
