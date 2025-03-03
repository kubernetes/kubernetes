/*
Copyright 2024 The Kubernetes Authors.

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

package podtopologylabels

import (
	"context"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"

	corev1 "k8s.io/api/core/v1"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/version"
	"k8s.io/apiserver/pkg/admission"
	genericadmissioninitializer "k8s.io/apiserver/pkg/admission/initializer"
	admissiontesting "k8s.io/apiserver/pkg/admission/testing"
	"k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/kubernetes/fake"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	api "k8s.io/kubernetes/pkg/apis/core"
	kubefeatures "k8s.io/kubernetes/pkg/features"
)

// TestPodTopology verifies the pod topology admission plugin works as expected.
func TestPodTopology(t *testing.T) {
	tests := []struct {
		name                  string               // name of the test case.
		bindingTarget         *api.ObjectReference // target being bound to. Defaults to a valid node with the provided labels.
		targetNodeLabels      map[string]string    // list of labels set on the node being bound to.
		existingBindingLabels map[string]string    // list of labels that are set on the Binding prior to admission (aka by the client/scheduler)
		expectedBindingLabels map[string]string    // list of labels that we expect to be set on the Binding after admission.
		featureDisabled       bool                 // configure whether the SetPodTopologyLabels feature gate should be disabled.
	}{
		{
			name: "copies topology.k8s.io/zone and region labels to binding annotations",
			targetNodeLabels: map[string]string{
				"topology.k8s.io/zone":      "zone1",
				"topology.k8s.io/region":    "region1",
				"non-topology.k8s.io/label": "something", // verify we don't unexpectedly copy non topology.k8s.io labels.
			},
			expectedBindingLabels: map[string]string{
				"topology.k8s.io/zone":   "zone1",
				"topology.k8s.io/region": "region1",
			},
		},
		{
			name: "copies arbitrary topology labels",
			targetNodeLabels: map[string]string{
				"topology.k8s.io/arbitrary": "something",
			},
			expectedBindingLabels: map[string]string{
				"topology.k8s.io/arbitrary": "something",
			},
		},
		{
			name: "copies topology labels that use a subdomain",
			targetNodeLabels: map[string]string{
				"something.topology.k8s.io/a-thing": "value",
			},
			expectedBindingLabels: map[string]string{
				"something.topology.k8s.io/a-thing": "value",
			},
		},
		{
			name: "does not copy label keys that don't contain a / character",
			targetNodeLabels: map[string]string{
				"topology.k8s.io": "value",
			},
			existingBindingLabels: map[string]string{},
		},
		{
			name: "does not overwrite existing topology labels on Binding objects",
			existingBindingLabels: map[string]string{
				"topology.k8s.io/zone": "oldValue",
			},
			targetNodeLabels: map[string]string{
				"topology.k8s.io/zone": "newValue",
			},
			expectedBindingLabels: map[string]string{
				"topology.k8s.io/zone": "oldValue",
			},
		},
		{
			name: "does nothing if the SetPodTopologyLabels feature gate is disabled",
			targetNodeLabels: map[string]string{
				"topology.k8s.io/zone":   "zone1",
				"topology.k8s.io/region": "region1",
			},
			expectedBindingLabels: map[string]string{},
			featureDisabled:       true,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			namespace := &corev1.Namespace{
				ObjectMeta: metav1.ObjectMeta{Name: "test-ns"},
			}
			node := &corev1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Name:   "valid-test-node",
					Labels: test.targetNodeLabels,
				},
			}
			// Pod we bind during test cases.
			pod := &corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: "testPod", Namespace: namespace.Name},
				Spec:       corev1.PodSpec{},
			}
			featuregatetesting.SetFeatureGateEmulationVersionDuringTest(t, feature.DefaultFeatureGate, version.MustParse("1.33"))
			featuregatetesting.SetFeatureGateDuringTest(t, feature.DefaultFeatureGate, kubefeatures.PodTopologyLabelsAdmission,
				!test.featureDisabled)
			mockClient := fake.NewSimpleClientset(namespace, node, pod)
			handler, informerFactory, err := newHandlerForTest(mockClient)
			if err != nil {
				t.Fatalf("unexpected error initializing handler: %v", err)
			}
			stopCh := make(chan struct{})
			defer close(stopCh)
			informerFactory.Start(stopCh)

			target := test.bindingTarget
			if target == nil {
				target = &api.ObjectReference{
					Kind: "Node",
					Name: node.Name,
				}
			}
			binding := &api.Binding{
				ObjectMeta: metav1.ObjectMeta{
					Name:      pod.Name,
					Namespace: pod.Namespace,
					Labels:    test.existingBindingLabels,
				},
				Target: *target,
			}
			if err := admissiontesting.WithReinvocationTesting(t, handler).
				Admit(context.TODO(), admission.NewAttributesRecord(binding, nil, api.Kind("Binding").WithVersion("version"), pod.Namespace, pod.Name, api.Resource("pods").WithVersion("version"), "binding", admission.Create, &metav1.CreateOptions{}, false, nil), nil); err != nil {
				t.Errorf("failed running admission plugin: %v", err)
			}
			updatedBindingLabels := binding.Labels
			if !apiequality.Semantic.DeepEqual(updatedBindingLabels, test.expectedBindingLabels) {
				t.Errorf("Unexpected label values: %v", cmp.Diff(updatedBindingLabels, test.expectedBindingLabels))
			}
		})
	}
}

// newHandlerForTest returns the admission controller configured for testing.
func newHandlerForTest(c kubernetes.Interface) (*Plugin, informers.SharedInformerFactory, error) {
	factory := informers.NewSharedInformerFactory(c, 5*time.Minute)
	handler := NewPodTopologyPlugin()
	pluginInitializer := genericadmissioninitializer.New(c, nil, factory, nil, feature.DefaultFeatureGate, nil, nil)
	pluginInitializer.Initialize(handler)
	return handler, factory, admission.ValidateInitialization(handler)
}
