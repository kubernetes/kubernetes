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

package alwayspullimages

import (
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apiserver/pkg/admission"
	"k8s.io/kubernetes/pkg/api"
)

// TestAdmission verifies all create requests for pods result in every container's image pull policy
// set to Always
func TestAdmission(t *testing.T) {
	namespace := "test"
	handler := &alwaysPullImages{}
	pod := api.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: "123", Namespace: namespace},
		Spec: api.PodSpec{
			InitContainers: []api.Container{
				{Name: "init1", Image: "image"},
				{Name: "init2", Image: "image", ImagePullPolicy: api.PullNever},
				{Name: "init3", Image: "image", ImagePullPolicy: api.PullIfNotPresent},
				{Name: "init4", Image: "image", ImagePullPolicy: api.PullAlways},
			},
			Containers: []api.Container{
				{Name: "ctr1", Image: "image"},
				{Name: "ctr2", Image: "image", ImagePullPolicy: api.PullNever},
				{Name: "ctr3", Image: "image", ImagePullPolicy: api.PullIfNotPresent},
				{Name: "ctr4", Image: "image", ImagePullPolicy: api.PullAlways},
			},
		},
	}
	err := handler.Admit(admission.NewAttributesRecord(&pod, nil, api.Kind("Pod").WithVersion("version"), pod.Namespace, pod.Name, api.Resource("pods").WithVersion("version"), "", admission.Create, nil))
	if err != nil {
		t.Errorf("Unexpected error returned from admission handler")
	}
	for _, c := range pod.Spec.InitContainers {
		if c.ImagePullPolicy != api.PullAlways {
			t.Errorf("Container %v: expected pull always, got %v", c, c.ImagePullPolicy)
		}
	}
	for _, c := range pod.Spec.Containers {
		if c.ImagePullPolicy != api.PullAlways {
			t.Errorf("Container %v: expected pull always, got %v", c, c.ImagePullPolicy)
		}
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
				{Name: "ctr2", Image: "image", ImagePullPolicy: api.PullNever},
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
		handler := &alwaysPullImages{}

		err := handler.Admit(admission.NewAttributesRecord(tc.object, nil, api.Kind(tc.kind).WithVersion("version"), namespace, name, api.Resource(tc.resource).WithVersion("version"), tc.subresource, admission.Create, nil))

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

		if e, a := api.PullNever, pod.Spec.Containers[0].ImagePullPolicy; e != a {
			t.Errorf("%s: image pull policy was changed to %s", tc.name, a)
		}
	}
}
