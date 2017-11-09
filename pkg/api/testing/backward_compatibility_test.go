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

package testing

import (
	"testing"

	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/kubernetes/pkg/api/testing/compat"
	api "k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/apis/core/validation"

	_ "k8s.io/kubernetes/pkg/apis/core/install"
)

func TestCompatibility_v1_PodSecurityContext(t *testing.T) {
	cases := []struct {
		name         string
		input        string
		expectedKeys map[string]string
		absentKeys   []string
	}{
		{
			name: "hostNetwork = true",
			input: `
{
	"kind":"Pod",
	"apiVersion":"v1",
	"metadata":{"name":"my-pod-name", "namespace":"my-pod-namespace"},
	"spec": {
		"hostNetwork": true,
		"containers":[{
			"name":"a",
			"image":"my-container-image"
		}]
	}
}
`,
			expectedKeys: map[string]string{
				"spec.hostNetwork": "true",
			},
		},
		{
			name: "hostNetwork = false",
			input: `
{
	"kind":"Pod",
	"apiVersion":"v1",
	"metadata":{"name":"my-pod-name", "namespace":"my-pod-namespace"},
	"spec": {
		"hostNetwork": false,
		"containers":[{
			"name":"a",
			"image":"my-container-image"
		}]
	}
}
`,
			absentKeys: []string{
				"spec.hostNetwork",
			},
		},
		{
			name: "hostIPC = true",
			input: `
{
	"kind":"Pod",
	"apiVersion":"v1",
	"metadata":{"name":"my-pod-name", "namespace":"my-pod-namespace"},
	"spec": {
		"hostIPC": true,
		"containers":[{
			"name":"a",
			"image":"my-container-image"
		}]
	}
}
`,
			expectedKeys: map[string]string{
				"spec.hostIPC": "true",
			},
		},
		{
			name: "hostIPC = false",
			input: `
{
	"kind":"Pod",
	"apiVersion":"v1",
	"metadata":{"name":"my-pod-name", "namespace":"my-pod-namespace"},
	"spec": {
		"hostIPC": false,
		"containers":[{
			"name":"a",
			"image":"my-container-image"
		}]
	}
}
`,
			absentKeys: []string{
				"spec.hostIPC",
			},
		},
		{
			name: "hostPID = true",
			input: `
{
	"kind":"Pod",
	"apiVersion":"v1",
	"metadata":{"name":"my-pod-name", "namespace":"my-pod-namespace"},
	"spec": {
		"hostPID": true,
		"containers":[{
			"name":"a",
			"image":"my-container-image"
		}]
	}
}
`,
			expectedKeys: map[string]string{
				"spec.hostPID": "true",
			},
		},
		{
			name: "hostPID = false",
			input: `
{
	"kind":"Pod",
	"apiVersion":"v1",
	"metadata":{"name":"my-pod-name", "namespace":"my-pod-namespace"},
	"spec": {
		"hostPID": false,
		"containers":[{
			"name":"a",
			"image":"my-container-image"
		}]
	}
}
`,
			absentKeys: []string{
				"spec.hostPID",
			},
		},
	}

	validator := func(obj runtime.Object) field.ErrorList {
		return validation.ValidatePodSpec(&(obj.(*api.Pod).Spec), field.NewPath("spec"))
	}

	for _, tc := range cases {
		t.Logf("Testing 1.0.0 backward compatibility for %v", tc.name)
		compat.TestCompatibility(t, v1.SchemeGroupVersion, []byte(tc.input), validator, tc.expectedKeys, tc.absentKeys)
	}
}
