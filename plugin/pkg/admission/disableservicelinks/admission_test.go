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

package disableservicelinks

import (
	"context"
	"testing"

	"k8s.io/apiserver/pkg/admission"
	admissiontesting "k8s.io/apiserver/pkg/admission/testing"
	"k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/apis/core/helper"
	"k8s.io/utils/ptr"
)

func TestAdmit(t *testing.T) {

	plugin := admissiontesting.WithReinvocationTesting(t, newDisableServiceLinks())

	tests := []struct {
		description  string
		requestedPod core.Pod
		expectedPod  core.Pod
		operation    admission.Operation
	}{
		{
			description: "Create empty pod with default value of Spec.EnableServiceLinks",
			requestedPod: core.Pod{
				Spec: core.PodSpec{},
			},
			expectedPod: core.Pod{
				Spec: core.PodSpec{EnableServiceLinks: ptr.To(false)},
			},
			operation: admission.Create,
		},
		{
			description: "Create empty pod with Spec.EnableServiceLinks set to true",
			requestedPod: core.Pod{
				Spec: core.PodSpec{EnableServiceLinks: ptr.To(true)},
			},
			expectedPod: core.Pod{
				Spec: core.PodSpec{EnableServiceLinks: ptr.To(false)},
			},
			operation: admission.Create,
		},
		{
			description: "Update empty pod with Spec.EnableServiceLinks set to true",
			requestedPod: core.Pod{
				Spec: core.PodSpec{EnableServiceLinks: ptr.To(true)},
			},
			expectedPod: core.Pod{
				Spec: core.PodSpec{EnableServiceLinks: ptr.To(true)},
			},
			operation: admission.Update,
		},
	}
	for i, test := range tests {
		err := plugin.Admit(context.TODO(), admission.NewAttributesRecord(&test.requestedPod, nil,
			core.Kind("Pod").WithVersion("version"), "foo", "name",
			core.Resource("pods").WithVersion("version"), "", test.operation,
			nil, false, nil), nil)

		if err != nil {
			t.Errorf("[%d: %s] unexpected error %v for pod %+v", i, test.description, err, test.requestedPod)
		}

		if !helper.Semantic.DeepEqual(test.expectedPod.Spec, test.requestedPod.Spec) {
			t.Errorf("[%d: %s] expected %t got %t", i, test.description, *test.expectedPod.Spec.EnableServiceLinks, *test.requestedPod.Spec.EnableServiceLinks)
		}
	}
}

func TestHandles(t *testing.T) {
	plugin := newDisableServiceLinks()
	tests := map[admission.Operation]bool{
		admission.Create:  true,
		admission.Update:  true,
		admission.Delete:  false,
		admission.Connect: false,
	}
	for op, expected := range tests {
		result := plugin.Handles(op)
		if result != expected {
			t.Errorf("Unexpected result for operation %s: %v\n", op, result)
		}
	}
}
