/*
Copyright 2014 The Kubernetes Authors.

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

package autoprovision

import (
	"context"
	"fmt"
	"testing"
	"time"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apiserver/pkg/admission"
	genericadmissioninitializer "k8s.io/apiserver/pkg/admission/initializer"
	admissiontesting "k8s.io/apiserver/pkg/admission/testing"
	"k8s.io/client-go/informers"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/kubernetes/fake"
	core "k8s.io/client-go/testing"
	api "k8s.io/kubernetes/pkg/apis/core"
)

// newHandlerForTest returns the admission controller configured for testing.
func newHandlerForTest(c clientset.Interface) (admission.MutationInterface, informers.SharedInformerFactory, error) {
	f := informers.NewSharedInformerFactory(c, 5*time.Minute)
	handler := NewProvision()
	pluginInitializer := genericadmissioninitializer.New(c, nil, f, nil, nil, nil, nil)
	pluginInitializer.Initialize(handler)
	err := admission.ValidateInitialization(handler)
	return handler, f, err
}

// newMockClientForTest creates a mock client that returns a client configured for the specified list of namespaces.
func newMockClientForTest(namespaces []string) *fake.Clientset {
	mockClient := &fake.Clientset{}
	mockClient.AddReactor("list", "namespaces", func(action core.Action) (bool, runtime.Object, error) {
		namespaceList := &corev1.NamespaceList{
			ListMeta: metav1.ListMeta{
				ResourceVersion: fmt.Sprintf("%d", len(namespaces)),
			},
		}
		for i, ns := range namespaces {
			namespaceList.Items = append(namespaceList.Items, corev1.Namespace{
				ObjectMeta: metav1.ObjectMeta{
					Name:            ns,
					ResourceVersion: fmt.Sprintf("%d", i),
				},
			})
		}
		return true, namespaceList, nil
	})
	return mockClient
}

// newPod returns a new pod for the specified namespace
func newPod(namespace string) api.Pod {
	return api.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: "123", Namespace: namespace},
		Spec: api.PodSpec{
			Volumes:    []api.Volume{{Name: "vol"}},
			Containers: []api.Container{{Name: "ctr", Image: "image"}},
		},
	}
}

// hasCreateNamespaceAction returns true if it has the create namespace action
func hasCreateNamespaceAction(mockClient *fake.Clientset) bool {
	for _, action := range mockClient.Actions() {
		if action.GetVerb() == "create" && action.GetResource().Resource == "namespaces" {
			return true
		}
	}
	return false
}

// TestAdmission verifies a namespace is created on create requests for namespace managed resources
func TestAdmission(t *testing.T) {
	namespace := "test"
	mockClient := newMockClientForTest([]string{})
	handler, informerFactory, err := newHandlerForTest(mockClient)
	if err != nil {
		t.Errorf("unexpected error initializing handler: %v", err)
	}
	informerFactory.Start(wait.NeverStop)

	pod := newPod(namespace)
	err = admissiontesting.WithReinvocationTesting(t, handler).Admit(context.TODO(), admission.NewAttributesRecord(&pod, nil, api.Kind("Pod").WithVersion("version"), pod.Namespace, pod.Name, api.Resource("pods").WithVersion("version"), "", admission.Create, &metav1.CreateOptions{}, false, nil), nil)
	if err != nil {
		t.Errorf("unexpected error returned from admission handler")
	}
	if !hasCreateNamespaceAction(mockClient) {
		t.Errorf("expected create namespace action")
	}
}

// TestAdmissionNamespaceExists verifies that no client call is made when a namespace already exists
func TestAdmissionNamespaceExists(t *testing.T) {
	namespace := "test"
	mockClient := newMockClientForTest([]string{namespace})
	handler, informerFactory, err := newHandlerForTest(mockClient)
	if err != nil {
		t.Errorf("unexpected error initializing handler: %v", err)
	}
	informerFactory.Start(wait.NeverStop)

	pod := newPod(namespace)
	err = admissiontesting.WithReinvocationTesting(t, handler).Admit(context.TODO(), admission.NewAttributesRecord(&pod, nil, api.Kind("Pod").WithVersion("version"), pod.Namespace, pod.Name, api.Resource("pods").WithVersion("version"), "", admission.Create, &metav1.CreateOptions{}, false, nil), nil)
	if err != nil {
		t.Errorf("unexpected error returned from admission handler")
	}
	if hasCreateNamespaceAction(mockClient) {
		t.Errorf("unexpected create namespace action")
	}
}

// TestAdmissionDryRun verifies that no client call is made on a dry run request
func TestAdmissionDryRun(t *testing.T) {
	namespace := "test"
	mockClient := newMockClientForTest([]string{})
	handler, informerFactory, err := newHandlerForTest(mockClient)
	if err != nil {
		t.Errorf("unexpected error initializing handler: %v", err)
	}
	informerFactory.Start(wait.NeverStop)

	pod := newPod(namespace)
	err = admissiontesting.WithReinvocationTesting(t, handler).Admit(context.TODO(), admission.NewAttributesRecord(&pod, nil, api.Kind("Pod").WithVersion("version"), pod.Namespace, pod.Name, api.Resource("pods").WithVersion("version"), "", admission.Create, &metav1.CreateOptions{}, true, nil), nil)
	if err != nil {
		t.Errorf("unexpected error returned from admission handler")
	}
	if hasCreateNamespaceAction(mockClient) {
		t.Errorf("unexpected create namespace action")
	}
}

// TestIgnoreAdmission validates that a request is ignored if its not a create
func TestIgnoreAdmission(t *testing.T) {
	namespace := "test"
	mockClient := newMockClientForTest([]string{})
	handler, informerFactory, err := newHandlerForTest(mockClient)
	if err != nil {
		t.Errorf("unexpected error initializing handler: %v", err)
	}
	informerFactory.Start(wait.NeverStop)
	chainHandler := admissiontesting.WithReinvocationTesting(t, admission.NewChainHandler(handler))

	pod := newPod(namespace)
	err = admissiontesting.WithReinvocationTesting(t, chainHandler).Admit(context.TODO(), admission.NewAttributesRecord(&pod, nil, api.Kind("Pod").WithVersion("version"), pod.Namespace, pod.Name, api.Resource("pods").WithVersion("version"), "", admission.Update, &metav1.UpdateOptions{}, false, nil), nil)
	if err != nil {
		t.Errorf("unexpected error returned from admission handler")
	}
	if hasCreateNamespaceAction(mockClient) {
		t.Errorf("unexpected create namespace action")
	}
}

func TestAdmissionWithLatentCache(t *testing.T) {
	namespace := "test"
	mockClient := newMockClientForTest([]string{})
	mockClient.AddReactor("create", "namespaces", func(action core.Action) (bool, runtime.Object, error) {
		return true, nil, errors.NewAlreadyExists(api.Resource("namespaces"), namespace)
	})
	handler, informerFactory, err := newHandlerForTest(mockClient)
	if err != nil {
		t.Errorf("unexpected error initializing handler: %v", err)
	}
	informerFactory.Start(wait.NeverStop)

	pod := newPod(namespace)
	err = admissiontesting.WithReinvocationTesting(t, handler).Admit(context.TODO(), admission.NewAttributesRecord(&pod, nil, api.Kind("Pod").WithVersion("version"), pod.Namespace, pod.Name, api.Resource("pods").WithVersion("version"), "", admission.Create, &metav1.CreateOptions{}, false, nil), nil)
	if err != nil {
		t.Errorf("unexpected error returned from admission handler")
	}

	if !hasCreateNamespaceAction(mockClient) {
		t.Errorf("expected create namespace action")
	}
}
