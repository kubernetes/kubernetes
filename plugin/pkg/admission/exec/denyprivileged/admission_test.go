/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package denyprivileged

import (
	"testing"

	"k8s.io/kubernetes/pkg/admission"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/rest"
	"k8s.io/kubernetes/pkg/client/unversioned/testclient"
	"k8s.io/kubernetes/pkg/runtime"
)

// TestAdmission verifies a namespace is created on create requests for namespace managed resources
func TestAdmissionAccept(t *testing.T) {
	testAdmission(t, acceptPod("podname"), true)
}

func TestAdmissionDeny(t *testing.T) {
	testAdmission(t, denyPod("podname"), false)
}

func testAdmission(t *testing.T, pod *api.Pod, shouldAccept bool) {
	mockClient := &testclient.Fake{}
	mockClient.AddReactor("get", "pods", func(action testclient.Action) (bool, runtime.Object, error) {
		if action.(testclient.GetAction).GetName() == pod.Name {
			return true, pod, nil
		}
		t.Errorf("Unexpected API call: %#v", action)
		return true, nil, nil
	})
	handler := &denyExecOnPrivileged{
		client: mockClient,
	}

	// pods/exec
	{
		req := &rest.ConnectRequest{Name: pod.Name, ResourcePath: "pods/exec"}
		err := handler.Admit(admission.NewAttributesRecord(req, "Pod", "test", "name", "pods", "exec", admission.Connect, nil))
		if shouldAccept && err != nil {
			t.Errorf("Unexpected error returned from admission handler: %v", err)
		}
		if !shouldAccept && err == nil {
			t.Errorf("An error was expected from the admission handler. Received nil")
		}
	}

	// pods/attach
	{
		req := &rest.ConnectRequest{Name: pod.Name, ResourcePath: "pods/attach"}
		err := handler.Admit(admission.NewAttributesRecord(req, "Pod", "test", "name", "pods", "attach", admission.Connect, nil))
		if shouldAccept && err != nil {
			t.Errorf("Unexpected error returned from admission handler: %v", err)
		}
		if !shouldAccept && err == nil {
			t.Errorf("An error was expected from the admission handler. Received nil")
		}
	}
}

func acceptPod(name string) *api.Pod {
	return &api.Pod{
		ObjectMeta: api.ObjectMeta{Name: name, Namespace: "test"},
		Spec: api.PodSpec{
			Containers: []api.Container{
				{Name: "ctr1", Image: "image"},
				{Name: "ctr2", Image: "image2"},
			},
		},
	}
}

func denyPod(name string) *api.Pod {
	privileged := true
	return &api.Pod{
		ObjectMeta: api.ObjectMeta{Name: name, Namespace: "test"},
		Spec: api.PodSpec{
			Containers: []api.Container{
				{Name: "ctr1", Image: "image"},
				{
					Name:  "ctr2",
					Image: "image2",
					SecurityContext: &api.SecurityContext{
						Privileged: &privileged,
					},
				},
			},
		},
	}
}
