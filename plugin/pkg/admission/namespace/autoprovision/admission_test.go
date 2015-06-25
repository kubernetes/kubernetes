/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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

	"github.com/GoogleCloudPlatform/kubernetes/pkg/admission"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/errors"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client/cache"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client/testclient"
)

// TestAdmission verifies a namespace is created on create requests for namespace managed resources
func TestAdmission(t *testing.T) {
	namespace := "test"
	mockClient := &testclient.Fake{}
	handler := &provision{
		client: mockClient,
		store:  cache.NewStore(cache.MetaNamespaceKeyFunc),
	}
	pod := api.Pod{
		ObjectMeta: api.ObjectMeta{Name: "123", Namespace: namespace},
		Spec: api.PodSpec{
			Volumes:    []api.Volume{{Name: "vol"}},
			Containers: []api.Container{{Name: "ctr", Image: "image"}},
		},
	}
	err := handler.Admit(admission.NewAttributesRecord(&pod, "Pod", pod.Namespace, pod.Name, "pods", "", admission.Create, nil))
	if err != nil {
		t.Errorf("Unexpected error returned from admission handler")
	}
	if len(mockClient.Actions) != 1 {
		t.Errorf("Expected a create-namespace request")
	}
	if mockClient.Actions[0].Action != "create-namespace" {
		t.Errorf("Expected a create-namespace request to be made via the client")
	}
}

// TestAdmissionNamespaceExists verifies that no client call is made when a namespace already exists
func TestAdmissionNamespaceExists(t *testing.T) {
	namespace := "test"
	mockClient := &testclient.Fake{}
	store := cache.NewStore(cache.MetaNamespaceKeyFunc)
	store.Add(&api.Namespace{
		ObjectMeta: api.ObjectMeta{Name: namespace},
	})
	handler := &provision{
		client: mockClient,
		store:  store,
	}
	pod := api.Pod{
		ObjectMeta: api.ObjectMeta{Name: "123", Namespace: namespace},
		Spec: api.PodSpec{
			Volumes:    []api.Volume{{Name: "vol"}},
			Containers: []api.Container{{Name: "ctr", Image: "image"}},
		},
	}
	err := handler.Admit(admission.NewAttributesRecord(&pod, "Pod", pod.Namespace, pod.Name, "pods", "", admission.Create, nil))
	if err != nil {
		t.Errorf("Unexpected error returned from admission handler")
	}
	if len(mockClient.Actions) != 0 {
		t.Errorf("No client request should have been made")
	}
}

// TestIgnoreAdmission validates that a request is ignored if its not a create
func TestIgnoreAdmission(t *testing.T) {
	namespace := "test"
	mockClient := &testclient.Fake{}
	handler := admission.NewChainHandler(createProvision(mockClient, nil))
	pod := api.Pod{
		ObjectMeta: api.ObjectMeta{Name: "123", Namespace: namespace},
		Spec: api.PodSpec{
			Volumes:    []api.Volume{{Name: "vol"}},
			Containers: []api.Container{{Name: "ctr", Image: "image"}},
		},
	}
	err := handler.Admit(admission.NewAttributesRecord(&pod, "Pod", pod.Namespace, pod.Name, "pods", "", admission.Update, nil))
	if err != nil {
		t.Errorf("Unexpected error returned from admission handler")
	}
	if len(mockClient.Actions) != 0 {
		t.Errorf("No client request should have been made")
	}
}

// TestAdmissionNamespaceExistsUnknownToHandler
func TestAdmissionNamespaceExistsUnknownToHandler(t *testing.T) {
	namespace := "test"
	mockClient := &testclient.Fake{
		Err: errors.NewAlreadyExists("namespaces", namespace),
	}
	store := cache.NewStore(cache.MetaNamespaceKeyFunc)
	handler := &provision{
		client: mockClient,
		store:  store,
	}
	pod := api.Pod{
		ObjectMeta: api.ObjectMeta{Name: "123", Namespace: namespace},
		Spec: api.PodSpec{
			Volumes:    []api.Volume{{Name: "vol"}},
			Containers: []api.Container{{Name: "ctr", Image: "image"}},
		},
	}
	err := handler.Admit(admission.NewAttributesRecord(&pod, "Pod", pod.Namespace, pod.Name, "pods", "", admission.Create, nil))
	if err != nil {
		t.Errorf("Unexpected error returned from admission handler")
	}
}
