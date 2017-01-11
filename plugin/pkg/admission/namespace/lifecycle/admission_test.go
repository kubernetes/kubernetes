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

package lifecycle

import (
	"fmt"
	"testing"
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/kubernetes/pkg/admission"
	"k8s.io/kubernetes/pkg/api"
	clientset "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"
	"k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/fake"
	"k8s.io/kubernetes/pkg/client/testing/core"
	"k8s.io/kubernetes/pkg/controller/informers"
	kubeadmission "k8s.io/kubernetes/pkg/kubeapiserver/admission"
	"k8s.io/kubernetes/pkg/util/clock"
)

// newHandlerForTest returns a configured handler for testing.
func newHandlerForTest(c clientset.Interface) (admission.Interface, informers.SharedInformerFactory, error) {
	return newHandlerForTestWithClock(c, clock.RealClock{})
}

// newHandlerForTestWithClock returns a configured handler for testing.
func newHandlerForTestWithClock(c clientset.Interface, cacheClock clock.Clock) (admission.Interface, informers.SharedInformerFactory, error) {
	f := informers.NewSharedInformerFactory(nil, c, 5*time.Minute)
	handler, err := newLifecycleWithClock(sets.NewString(api.NamespaceDefault, api.NamespaceSystem), cacheClock)
	if err != nil {
		return nil, f, err
	}
	pluginInitializer := kubeadmission.NewPluginInitializer(c, f, nil)
	pluginInitializer.Initialize(handler)
	err = admission.Validate(handler)
	return handler, f, err
}

// newMockClientForTest creates a mock client that returns a client configured for the specified list of namespaces with the specified phase.
func newMockClientForTest(namespaces map[string]api.NamespacePhase) *fake.Clientset {
	mockClient := &fake.Clientset{}
	mockClient.AddReactor("list", "namespaces", func(action core.Action) (bool, runtime.Object, error) {
		namespaceList := &api.NamespaceList{
			ListMeta: metav1.ListMeta{
				ResourceVersion: fmt.Sprintf("%d", len(namespaces)),
			},
		}
		index := 0
		for name, phase := range namespaces {
			namespaceList.Items = append(namespaceList.Items, api.Namespace{
				ObjectMeta: api.ObjectMeta{
					Name:            name,
					ResourceVersion: fmt.Sprintf("%d", index),
				},
				Status: api.NamespaceStatus{
					Phase: phase,
				},
			})
			index++
		}
		return true, namespaceList, nil
	})
	return mockClient
}

// newPod returns a new pod for the specified namespace
func newPod(namespace string) api.Pod {
	return api.Pod{
		ObjectMeta: api.ObjectMeta{Name: "123", Namespace: namespace},
		Spec: api.PodSpec{
			Volumes:    []api.Volume{{Name: "vol"}},
			Containers: []api.Container{{Name: "ctr", Image: "image"}},
		},
	}
}

// TestAdmissionNamespaceDoesNotExist verifies pod is not admitted if namespace does not exist.
func TestAdmissionNamespaceDoesNotExist(t *testing.T) {
	namespace := "test"
	mockClient := newMockClientForTest(map[string]api.NamespacePhase{})
	mockClient.AddReactor("get", "namespaces", func(action core.Action) (bool, runtime.Object, error) {
		return true, nil, fmt.Errorf("nope, out of luck")
	})
	handler, informerFactory, err := newHandlerForTest(mockClient)
	if err != nil {
		t.Errorf("unexpected error initializing handler: %v", err)
	}
	informerFactory.Start(wait.NeverStop)

	pod := newPod(namespace)
	err = handler.Admit(admission.NewAttributesRecord(&pod, nil, api.Kind("Pod").WithVersion("version"), pod.Namespace, pod.Name, api.Resource("pods").WithVersion("version"), "", admission.Create, nil))
	if err == nil {
		actions := ""
		for _, action := range mockClient.Actions() {
			actions = actions + action.GetVerb() + ":" + action.GetResource().Resource + ":" + action.GetSubresource() + ", "
		}
		t.Errorf("expected error returned from admission handler: %v", actions)
	}
}

// TestAdmissionNamespaceActive verifies a resource is admitted when the namespace is active.
func TestAdmissionNamespaceActive(t *testing.T) {
	namespace := "test"
	mockClient := newMockClientForTest(map[string]api.NamespacePhase{
		namespace: api.NamespaceActive,
	})

	handler, informerFactory, err := newHandlerForTest(mockClient)
	if err != nil {
		t.Errorf("unexpected error initializing handler: %v", err)
	}
	informerFactory.Start(wait.NeverStop)

	pod := newPod(namespace)
	err = handler.Admit(admission.NewAttributesRecord(&pod, nil, api.Kind("Pod").WithVersion("version"), pod.Namespace, pod.Name, api.Resource("pods").WithVersion("version"), "", admission.Create, nil))
	if err != nil {
		t.Errorf("unexpected error returned from admission handler")
	}
}

// TestAdmissionNamespaceTerminating verifies a resource is not created when the namespace is active.
func TestAdmissionNamespaceTerminating(t *testing.T) {
	namespace := "test"
	mockClient := newMockClientForTest(map[string]api.NamespacePhase{
		namespace: api.NamespaceTerminating,
	})

	handler, informerFactory, err := newHandlerForTest(mockClient)
	if err != nil {
		t.Errorf("unexpected error initializing handler: %v", err)
	}
	informerFactory.Start(wait.NeverStop)

	pod := newPod(namespace)
	// verify create operations in the namespace cause an error
	err = handler.Admit(admission.NewAttributesRecord(&pod, nil, api.Kind("Pod").WithVersion("version"), pod.Namespace, pod.Name, api.Resource("pods").WithVersion("version"), "", admission.Create, nil))
	if err == nil {
		t.Errorf("Expected error rejecting creates in a namespace when it is terminating")
	}

	// verify update operations in the namespace can proceed
	err = handler.Admit(admission.NewAttributesRecord(&pod, nil, api.Kind("Pod").WithVersion("version"), pod.Namespace, pod.Name, api.Resource("pods").WithVersion("version"), "", admission.Update, nil))
	if err != nil {
		t.Errorf("Unexpected error returned from admission handler: %v", err)
	}

	// verify delete operations in the namespace can proceed
	err = handler.Admit(admission.NewAttributesRecord(nil, nil, api.Kind("Pod").WithVersion("version"), pod.Namespace, pod.Name, api.Resource("pods").WithVersion("version"), "", admission.Delete, nil))
	if err != nil {
		t.Errorf("Unexpected error returned from admission handler: %v", err)
	}

	// verify delete of namespace default can never proceed
	err = handler.Admit(admission.NewAttributesRecord(nil, nil, api.Kind("Namespace").WithVersion("version"), "", api.NamespaceDefault, api.Resource("namespaces").WithVersion("version"), "", admission.Delete, nil))
	if err == nil {
		t.Errorf("Expected an error that this namespace can never be deleted")
	}

	// verify delete of namespace other than default can proceed
	err = handler.Admit(admission.NewAttributesRecord(nil, nil, api.Kind("Namespace").WithVersion("version"), "", "other", api.Resource("namespaces").WithVersion("version"), "", admission.Delete, nil))
	if err != nil {
		t.Errorf("Did not expect an error %v", err)
	}
}

// TestAdmissionNamespaceForceLiveLookup verifies live lookups are done after deleting a namespace
func TestAdmissionNamespaceForceLiveLookup(t *testing.T) {
	namespace := "test"
	getCalls := int64(0)
	phases := map[string]api.NamespacePhase{namespace: api.NamespaceActive}
	mockClient := newMockClientForTest(phases)
	mockClient.AddReactor("get", "namespaces", func(action core.Action) (bool, runtime.Object, error) {
		getCalls++
		return true, &api.Namespace{ObjectMeta: api.ObjectMeta{Name: namespace}, Status: api.NamespaceStatus{Phase: phases[namespace]}}, nil
	})

	fakeClock := clock.NewFakeClock(time.Now())

	handler, informerFactory, err := newHandlerForTestWithClock(mockClient, fakeClock)
	if err != nil {
		t.Errorf("unexpected error initializing handler: %v", err)
	}
	informerFactory.Start(wait.NeverStop)

	pod := newPod(namespace)
	// verify create operations in the namespace is allowed
	err = handler.Admit(admission.NewAttributesRecord(&pod, nil, api.Kind("Pod").WithVersion("version"), pod.Namespace, pod.Name, api.Resource("pods").WithVersion("version"), "", admission.Create, nil))
	if err != nil {
		t.Errorf("Unexpected error rejecting creates in an active namespace")
	}
	if getCalls != 0 {
		t.Errorf("Expected no live lookups of the namespace, got %d", getCalls)
	}
	getCalls = 0

	// verify delete of namespace can proceed
	err = handler.Admit(admission.NewAttributesRecord(nil, nil, api.Kind("Namespace").WithVersion("version"), "", namespace, api.Resource("namespaces").WithVersion("version"), "", admission.Delete, nil))
	if err != nil {
		t.Errorf("Expected namespace deletion to be allowed")
	}
	if getCalls != 0 {
		t.Errorf("Expected no live lookups of the namespace, got %d", getCalls)
	}
	getCalls = 0

	// simulate the phase changing
	phases[namespace] = api.NamespaceTerminating

	// verify create operations in the namespace cause an error
	err = handler.Admit(admission.NewAttributesRecord(&pod, nil, api.Kind("Pod").WithVersion("version"), pod.Namespace, pod.Name, api.Resource("pods").WithVersion("version"), "", admission.Create, nil))
	if err == nil {
		t.Errorf("Expected error rejecting creates in a namespace right after deleting it")
	}
	if getCalls != 1 {
		t.Errorf("Expected a live lookup of the namespace at t=0, got %d", getCalls)
	}
	getCalls = 0

	// Ensure the live lookup is still forced up to forceLiveLookupTTL
	fakeClock.Step(forceLiveLookupTTL)

	// verify create operations in the namespace cause an error
	err = handler.Admit(admission.NewAttributesRecord(&pod, nil, api.Kind("Pod").WithVersion("version"), pod.Namespace, pod.Name, api.Resource("pods").WithVersion("version"), "", admission.Create, nil))
	if err == nil {
		t.Errorf("Expected error rejecting creates in a namespace right after deleting it")
	}
	if getCalls != 1 {
		t.Errorf("Expected a live lookup of the namespace at t=forceLiveLookupTTL, got %d", getCalls)
	}
	getCalls = 0

	// Ensure the live lookup expires
	fakeClock.Step(time.Millisecond)

	// verify create operations in the namespace don't force a live lookup after the timeout
	handler.Admit(admission.NewAttributesRecord(&pod, nil, api.Kind("Pod").WithVersion("version"), pod.Namespace, pod.Name, api.Resource("pods").WithVersion("version"), "", admission.Create, nil))
	if getCalls != 0 {
		t.Errorf("Expected no live lookup of the namespace at t=forceLiveLookupTTL+1ms, got %d", getCalls)
	}
	getCalls = 0
}
