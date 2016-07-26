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
	"testing"
	"time"

	"k8s.io/kubernetes/pkg/admission"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/errors"
	"k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/fake"
	"k8s.io/kubernetes/pkg/client/testing/core"
	"k8s.io/kubernetes/pkg/controller/framework/informers"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/util/wait"
)

// TestAdmission verifies a namespace is created on create requests for namespace managed resources
func TestAdmission(t *testing.T) {
	namespace := "test"
	mockClient := &fake.Clientset{}
	informerFactory := informers.NewSharedInformerFactory(mockClient, 5*time.Minute)
	informerFactory.Namespaces()
	informerFactory.Start(wait.NeverStop)
	handler := &provision{
		client:          mockClient,
		informerFactory: informerFactory,
	}
	pod := api.Pod{
		ObjectMeta: api.ObjectMeta{Name: "123", Namespace: namespace},
		Spec: api.PodSpec{
			Volumes:    []api.Volume{{Name: "vol"}},
			Containers: []api.Container{{Name: "ctr", Image: "image"}},
		},
	}
	err := handler.Admit(admission.NewAttributesRecord(&pod, nil, api.Kind("Pod").WithVersion("version"), pod.Namespace, pod.Name, api.Resource("pods").WithVersion("version"), "", admission.Create, nil))
	if err != nil {
		t.Errorf("Unexpected error returned from admission handler")
	}
	actions := mockClient.Actions()
	if len(actions) != 1 {
		t.Errorf("Expected a create-namespace request")
	}
	if !actions[0].Matches("create", "namespaces") {
		t.Errorf("Expected a create-namespace request to be made via the client")
	}
}

// TestAdmissionNamespaceExists verifies that no client call is made when a namespace already exists
// func TestAdmissionNamespaceExists(t *testing.T) {
// 	namespace := "test"
// 	mockClient := &fake.Clientset{}
// 	informerFactory := informers.NewSharedInformerFactory(mockClient, 5*time.Minute)
// 	informerFactory.Namespaces().Informer().GetStore().Add(&api.Namespace{
// 		ObjectMeta: api.ObjectMeta{Name: namespace},
// 	})
// 	informerFactory.Start(wait.NeverStop)
// 	handler := &provision{
// 		client:          mockClient,
// 		informerFactory: informerFactory,
// 	}
// 	pod := api.Pod{
// 		ObjectMeta: api.ObjectMeta{Name: "123", Namespace: namespace},
// 		Spec: api.PodSpec{
// 			Volumes:    []api.Volume{{Name: "vol"}},
// 			Containers: []api.Container{{Name: "ctr", Image: "image"}},
// 		},
// 	}
// 	err := handler.Admit(admission.NewAttributesRecord(&pod, nil, api.Kind("Pod").WithVersion("version"), pod.Namespace, pod.Name, api.Resource("pods").WithVersion("version"), "", admission.Create, nil))
// 	if err != nil {
// 		t.Errorf("Unexpected error returned from admission handler")
// 	}
// 	if len(mockClient.Actions()) != 0 {
// 		t.Errorf("No client request should have been made")
// 	}
// }

// TestIgnoreAdmission validates that a request is ignored if its not a create
func TestIgnoreAdmission(t *testing.T) {
	namespace := "test"
	mockClient := &fake.Clientset{}
	handler := admission.NewChainHandler(NewProvision(mockClient))
	pod := api.Pod{
		ObjectMeta: api.ObjectMeta{Name: "123", Namespace: namespace},
		Spec: api.PodSpec{
			Volumes:    []api.Volume{{Name: "vol"}},
			Containers: []api.Container{{Name: "ctr", Image: "image"}},
		},
	}
	err := handler.Admit(admission.NewAttributesRecord(&pod, nil, api.Kind("Pod").WithVersion("version"), pod.Namespace, pod.Name, api.Resource("pods").WithVersion("version"), "", admission.Update, nil))
	if err != nil {
		t.Errorf("Unexpected error returned from admission handler")
	}
	if len(mockClient.Actions()) != 0 {
		t.Errorf("No client request should have been made")
	}
}

// TestAdmissionNamespaceExistsUnknownToHandler
func TestAdmissionNamespaceExistsUnknownToHandler(t *testing.T) {
	namespace := "test"
	mockClient := &fake.Clientset{}
	mockClient.AddReactor("create", "namespaces", func(action core.Action) (bool, runtime.Object, error) {
		return true, nil, errors.NewAlreadyExists(api.Resource("namespaces"), namespace)
	})
	informerFactory := informers.NewSharedInformerFactory(mockClient, 5*time.Minute)
	informerFactory.Namespaces()
	informerFactory.Start(wait.NeverStop)
	handler := &provision{
		client:          mockClient,
		informerFactory: informerFactory,
	}
	pod := api.Pod{
		ObjectMeta: api.ObjectMeta{Name: "123", Namespace: namespace},
		Spec: api.PodSpec{
			Volumes:    []api.Volume{{Name: "vol"}},
			Containers: []api.Container{{Name: "ctr", Image: "image"}},
		},
	}
	err := handler.Admit(admission.NewAttributesRecord(&pod, nil, api.Kind("Pod").WithVersion("version"), pod.Namespace, pod.Name, api.Resource("pods").WithVersion("version"), "", admission.Create, nil))
	if err != nil {
		t.Errorf("Unexpected error returned from admission handler")
	}
}

// TestAdmissionNamespaceValidate
func TestAdmissionNamespaceValidate(t *testing.T) {
	mockClient := &fake.Clientset{}
	informerFactory := informers.NewSharedInformerFactory(mockClient, 5*time.Minute)
	handler := &provision{
		client: mockClient,
	}
	handler.SetInformerFactory(informerFactory)
	err := handler.Validate()
	if err != nil {
		t.Errorf("Failed to initialize informer")
	}
}
