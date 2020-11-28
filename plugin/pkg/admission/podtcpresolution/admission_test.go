/*
Copyright 2020 The Kubernetes Authors.

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

package podtcpresolution

import (
	"context"
	"strings"
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apiserver/pkg/admission"
	admissiontesting "k8s.io/apiserver/pkg/admission/testing"
	api "k8s.io/kubernetes/pkg/apis/core"
)

// TestAdmission verifies all create requests for pods result in every container has
// the environment variable RES_OPTION=use-vc
func TestAdmission(t *testing.T) {
	namespace := "test"
	handler := admissiontesting.WithReinvocationTesting(t, &PodTCPResolution{})
	pod := api.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: "123", Namespace: namespace},
		Spec: api.PodSpec{
			InitContainers: []api.Container{
				{Name: "init1", Image: "image"},
				{Name: "init2", Image: "image", Env: []api.EnvVar{{Name: "POD_IP", Value: "192.168.2.1"}}},
				{Name: "init3", Image: "image"},
				{Name: "init4", Image: "image"},
			},
			Containers: []api.Container{
				{Name: "ctr1", Image: "image"},
				{Name: "ctr2", Image: "image", Env: []api.EnvVar{{Name: "RES_OPTIONS", Value: "use-vc"}}},
				{Name: "ctr3", Image: "image", Env: []api.EnvVar{{Name: "RES_OPTIONS", Value: "ndots:2"}}},
				{Name: "ctr4", Image: "image", Env: []api.EnvVar{{Name: "RES_OPTIONS", Value: "ndots:2 use-vc"}}},
			},
		},
	}
	err := handler.Admit(context.TODO(), admission.NewAttributesRecord(&pod, nil, api.Kind("Pod").WithVersion("version"), pod.Namespace, pod.Name, api.Resource("pods").WithVersion("version"), "", admission.Create, &metav1.CreateOptions{}, false, nil), nil)
	if err != nil {
		t.Errorf("Unexpected error returned from admission handler")
	}
	for _, c := range pod.Spec.InitContainers {
		_, value := getEnvValueByName("RES_OPTIONS", c.Env)
		if !strings.Contains(value, "use-vc") {
			t.Errorf("Container %v: not expected env variable RES_OPTIONS=use-vc, got %v", c, c.Env)
		}
	}
	for _, c := range pod.Spec.Containers {
		_, value := getEnvValueByName("RES_OPTIONS", c.Env)
		if !strings.Contains(value, "use-vc") {
			t.Errorf("Container %v: expected RES_OPTIONS=use-vc, got %v", c, c.Env)
		}
	}
}

func TestValidate(t *testing.T) {
	namespace := "test"
	handler := &PodTCPResolution{}
	pod := api.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: "123", Namespace: namespace},
		Spec: api.PodSpec{
			InitContainers: []api.Container{
				{Name: "init1", Image: "image"},
				{Name: "init2", Image: "image", Env: []api.EnvVar{{Name: "POD_IP", Value: "192.168.2.1"}}},
				{Name: "init3", Image: "image"},
				{Name: "init4", Image: "image"},
			},
			Containers: []api.Container{
				{Name: "ctr1", Image: "image"},
				{Name: "ctr2", Image: "image", Env: []api.EnvVar{{Name: "RES_OPTIONS", Value: "use-vc"}}},
				{Name: "ctr3", Image: "image", Env: []api.EnvVar{{Name: "RES_OPTIONS", Value: "ndots:2"}}},
				{Name: "ctr4", Image: "image", Env: []api.EnvVar{{Name: "RES_OPTIONS", Value: "ndots:2 use-vc"}}},
			},
		},
	}
	expectedError := `[` +
		`spec.initContainers[0].env: Not found: "RES_OPTIONS=\"use-vc\" to force tcp dns resolution", ` +
		`spec.initContainers[1].env: Not found: "RES_OPTIONS=\"use-vc\" to force tcp dns resolution", ` +
		`spec.initContainers[2].env: Not found: "RES_OPTIONS=\"use-vc\" to force tcp dns resolution", ` +
		`spec.initContainers[3].env: Not found: "RES_OPTIONS=\"use-vc\" to force tcp dns resolution", ` +
		`spec.containers[0].env: Not found: "RES_OPTIONS=\"use-vc\" to force tcp dns resolution", ` +
		`spec.containers[2].env: Not found: "RES_OPTIONS=\"use-vc\" to force tcp dns resolution"]`
	err := handler.Validate(context.TODO(), admission.NewAttributesRecord(&pod, nil, api.Kind("Pod").WithVersion("version"), pod.Namespace, pod.Name, api.Resource("pods").WithVersion("version"), "", admission.Create, &metav1.CreateOptions{}, false, nil), nil)
	if err == nil {
		t.Fatal("missing expected error")
	}
	if err.Error() != expectedError {
		t.Fatal(err)
	}
}

// TestOtherResources ensures that this admission controller is a no-op for other resources,
// subresources, and non-pods.
func TestOtherResources(t *testing.T) {
	namespace := "testnamespace"
	name := "testname"
	pod := &api.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: name, Namespace: namespace},
		Spec: api.PodSpec{
			Containers: []api.Container{
				{Name: "ctr2", Image: "image"},
			},
		},
	}
	tests := []struct {
		name        string
		kind        string
		resource    string
		subresource string
		object      runtime.Object
		expectError bool
	}{
		{
			name:     "non-pod resource",
			kind:     "Foo",
			resource: "foos",
			object:   pod,
		},
		{
			name:        "pod subresource",
			kind:        "Pod",
			resource:    "pods",
			subresource: "exec",
			object:      pod,
		},
		{
			name:        "non-pod object",
			kind:        "Pod",
			resource:    "pods",
			object:      &api.Service{},
			expectError: true,
		},
	}

	for _, tc := range tests {
		handler := admissiontesting.WithReinvocationTesting(t, &PodTCPResolution{})

		err := handler.Admit(context.TODO(), admission.NewAttributesRecord(tc.object, nil, api.Kind(tc.kind).WithVersion("version"), namespace, name, api.Resource(tc.resource).WithVersion("version"), tc.subresource, admission.Create, &metav1.CreateOptions{}, false, nil), nil)

		if tc.expectError {
			if err == nil {
				t.Errorf("%s: unexpected nil error", tc.name)
			}
			continue
		}

		if err != nil {
			t.Errorf("%s: unexpected error: %v", tc.name, err)
			continue
		}

		if len(pod.Spec.Containers[0].Env) > 0 {
			t.Errorf("%s: container environment variables has changed %v", tc.name, pod.Spec.Containers[0].Env)
		}
	}
}
