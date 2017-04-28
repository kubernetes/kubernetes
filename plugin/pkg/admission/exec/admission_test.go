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

package exec

import (
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apiserver/pkg/admission"
	"k8s.io/apiserver/pkg/registry/rest"
	core "k8s.io/client-go/testing"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/fake"
)

// newAllowEscalatingExec returns `admission.Interface` that allows execution on
// "hostIPC", "hostPID" and "privileged".
func newAllowEscalatingExec() admission.Interface {
	return &denyExec{
		Handler:    admission.NewHandler(admission.Connect),
		hostIPC:    false,
		hostPID:    false,
		privileged: false,
	}
}

func TestAdmission(t *testing.T) {
	privPod := validPod("privileged")
	priv := true
	privPod.Spec.Containers[0].SecurityContext = &api.SecurityContext{
		Privileged: &priv,
	}

	hostPIDPod := validPod("hostPID")
	hostPIDPod.Spec.SecurityContext = &api.PodSecurityContext{}
	hostPIDPod.Spec.SecurityContext.HostPID = true

	hostIPCPod := validPod("hostIPC")
	hostIPCPod.Spec.SecurityContext = &api.PodSecurityContext{}
	hostIPCPod.Spec.SecurityContext.HostIPC = true

	testCases := map[string]struct {
		pod          *api.Pod
		shouldAccept bool
	}{
		"priv": {
			shouldAccept: false,
			pod:          privPod,
		},
		"hostPID": {
			shouldAccept: false,
			pod:          hostPIDPod,
		},
		"hostIPC": {
			shouldAccept: false,
			pod:          hostIPCPod,
		},
		"non privileged": {
			shouldAccept: true,
			pod:          validPod("nonPrivileged"),
		},
	}

	// Get the direct object though to allow testAdmission to inject the client
	handler := NewDenyEscalatingExec().(*denyExec)

	for _, tc := range testCases {
		testAdmission(t, tc.pod, handler, tc.shouldAccept)
	}

	// run with a permissive config and all cases should pass
	handler = newAllowEscalatingExec().(*denyExec)

	for _, tc := range testCases {
		testAdmission(t, tc.pod, handler, true)
	}

	// run against an init container
	handler = NewDenyEscalatingExec().(*denyExec)

	for _, tc := range testCases {
		tc.pod.Spec.InitContainers = tc.pod.Spec.Containers
		tc.pod.Spec.Containers = nil
		testAdmission(t, tc.pod, handler, tc.shouldAccept)
	}

	// run with a permissive config and all cases should pass
	handler = newAllowEscalatingExec().(*denyExec)

	for _, tc := range testCases {
		testAdmission(t, tc.pod, handler, true)
	}
}

func testAdmission(t *testing.T, pod *api.Pod, handler *denyExec, shouldAccept bool) {
	mockClient := &fake.Clientset{}
	mockClient.AddReactor("get", "pods", func(action core.Action) (bool, runtime.Object, error) {
		if action.(core.GetAction).GetName() == pod.Name {
			return true, pod, nil
		}
		t.Errorf("Unexpected API call: %#v", action)
		return true, nil, nil
	})

	handler.SetInternalKubeClientSet(mockClient)
	admission.Validate(handler)

	// pods/exec
	{
		req := &rest.ConnectRequest{Name: pod.Name, ResourcePath: "pods/exec"}
		err := handler.Admit(admission.NewAttributesRecord(req, nil, api.Kind("Pod").WithVersion("version"), "test", "name", api.Resource("pods").WithVersion("version"), "exec", admission.Connect, nil))
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
		err := handler.Admit(admission.NewAttributesRecord(req, nil, api.Kind("Pod").WithVersion("version"), "test", "name", api.Resource("pods").WithVersion("version"), "attach", admission.Connect, nil))
		if shouldAccept && err != nil {
			t.Errorf("Unexpected error returned from admission handler: %v", err)
		}
		if !shouldAccept && err == nil {
			t.Errorf("An error was expected from the admission handler. Received nil")
		}
	}
}

// Test to ensure legacy admission controller works as expected.
func TestDenyExecOnPrivileged(t *testing.T) {
	privPod := validPod("privileged")
	priv := true
	privPod.Spec.Containers[0].SecurityContext = &api.SecurityContext{
		Privileged: &priv,
	}

	hostPIDPod := validPod("hostPID")
	hostPIDPod.Spec.SecurityContext = &api.PodSecurityContext{}
	hostPIDPod.Spec.SecurityContext.HostPID = true

	hostIPCPod := validPod("hostIPC")
	hostIPCPod.Spec.SecurityContext = &api.PodSecurityContext{}
	hostIPCPod.Spec.SecurityContext.HostIPC = true

	testCases := map[string]struct {
		pod          *api.Pod
		shouldAccept bool
	}{
		"priv": {
			shouldAccept: false,
			pod:          privPod,
		},
		"hostPID": {
			shouldAccept: true,
			pod:          hostPIDPod,
		},
		"hostIPC": {
			shouldAccept: true,
			pod:          hostIPCPod,
		},
		"non privileged": {
			shouldAccept: true,
			pod:          validPod("nonPrivileged"),
		},
	}

	// Get the direct object though to allow testAdmission to inject the client
	handler := NewDenyExecOnPrivileged().(*denyExec)

	for _, tc := range testCases {
		testAdmission(t, tc.pod, handler, tc.shouldAccept)
	}

	// test init containers
	for _, tc := range testCases {
		tc.pod.Spec.InitContainers = tc.pod.Spec.Containers
		tc.pod.Spec.Containers = nil
		testAdmission(t, tc.pod, handler, tc.shouldAccept)
	}
}

func validPod(name string) *api.Pod {
	return &api.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: name, Namespace: "test"},
		Spec: api.PodSpec{
			Containers: []api.Container{
				{Name: "ctr1", Image: "image"},
				{Name: "ctr2", Image: "image2"},
			},
		},
	}
}
