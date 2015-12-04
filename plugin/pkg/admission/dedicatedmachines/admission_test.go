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

package dedicatedmachines

import (
	"testing"

	"k8s.io/kubernetes/pkg/admission"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/apis/extensions"
	"k8s.io/kubernetes/pkg/client/cache"
	"k8s.io/kubernetes/pkg/client/unversioned/testclient"
)

// TestAdmission verifies a dedicated machine matches the pod to be created.
func TestAdmission(t *testing.T) {
	namespaceName := "test"

	namespaceList := api.NamespaceList{
		Items: []api.Namespace{
			{
				ObjectMeta: api.ObjectMeta{
					Name: namespaceName,
				},
				Spec: api.NamespaceSpec{},
				Status: api.NamespaceStatus{
					Phase: api.NamespaceActive,
				},
			},
		},
	}

	dedicatedMachineList := extensions.DedicatedMachineList{
		Items: []extensions.DedicatedMachine{
			{
				ObjectMeta: api.ObjectMeta{
					Name:      "foo",
					Namespace: namespaceName,
				},
				Spec: extensions.DedicatedMachineSpec{
					LabelValue: "bar",
				},
			},
		},
	}
	mockClient := testclient.NewSimpleFake(&namespaceList, &dedicatedMachineList)
	mockClientExperimental := &testclient.FakeExperimental{
		Fake: mockClient,
	}
	handler := &dedicated{
		client:          mockClient,
		extensionClient: mockClientExperimental,
		store:           cache.NewStore(cache.MetaNamespaceKeyFunc),
	}
	pod := api.Pod{
		ObjectMeta: api.ObjectMeta{Name: "123", Namespace: namespaceName},
		Spec: api.PodSpec{
			Volumes:    []api.Volume{{Name: "vol"}},
			Containers: []api.Container{{Name: "ctr", Image: "image"}},
		},
	}

	err := handler.Admit(admission.NewAttributesRecord(&pod, "Pod", pod.Namespace, pod.Name, "pods", "", admission.Create, nil))
	if err != nil {
		t.Errorf("Unexpected error returned from admission handler")
	}

	if pod.Spec.NodeSelector == nil || pod.Spec.NodeSelector["dedicated"] != "bar" {
		t.Error("Expected NodeSelector contains dedicated=bar")
	}
}

// TestAdmission verifies no dedicated machine matches the pod to be created.
func TestAdmissionNoDedicatedMachineInTargetNamespace(t *testing.T) {
	namespaceName := "test"

	namespaceList := api.NamespaceList{
		Items: []api.Namespace{
			{
				ObjectMeta: api.ObjectMeta{
					Name: namespaceName,
				},
				Spec: api.NamespaceSpec{},
				Status: api.NamespaceStatus{
					Phase: api.NamespaceActive,
				},
			},
		},
	}

	mockClient := testclient.NewSimpleFake(&namespaceList)
	mockClientExperimental := &testclient.FakeExperimental{
		Fake: mockClient,
	}
	handler := &dedicated{
		client:          mockClient,
		extensionClient: mockClientExperimental,
		store:           cache.NewStore(cache.MetaNamespaceKeyFunc),
	}
	pod := api.Pod{
		ObjectMeta: api.ObjectMeta{Name: "123", Namespace: namespaceName},
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
