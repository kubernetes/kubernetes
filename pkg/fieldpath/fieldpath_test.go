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

package fieldpath

import (
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/resource"
)

func TestExtractFieldPathAsString(t *testing.T) {
	cases := []struct {
		name                    string
		fieldPath               string
		obj                     interface{}
		expectedValue           string
		expectedMessageFragment string
	}{
		{
			name:      "not an API object",
			fieldPath: "metadata.name",
			obj:       "",
			expectedMessageFragment: "expected struct",
		},
		{
			name:      "ok - namespace",
			fieldPath: "metadata.namespace",
			obj: &api.Pod{
				ObjectMeta: api.ObjectMeta{
					Namespace: "object-namespace",
				},
			},
			expectedValue: "object-namespace",
		},
		{
			name:      "ok - name",
			fieldPath: "metadata.name",
			obj: &api.Pod{
				ObjectMeta: api.ObjectMeta{
					Name: "object-name",
				},
			},
			expectedValue: "object-name",
		},
		{
			name:      "ok - labels",
			fieldPath: "metadata.labels",
			obj: &api.Pod{
				ObjectMeta: api.ObjectMeta{
					Labels: map[string]string{"key": "value"},
				},
			},
			expectedValue: "key=\"value\"",
		},
		{
			name:      "ok - labels bslash n",
			fieldPath: "metadata.labels",
			obj: &api.Pod{
				ObjectMeta: api.ObjectMeta{
					Labels: map[string]string{"key": "value\n"},
				},
			},
			expectedValue: "key=\"value\\n\"",
		},
		{
			name:      "ok - annotations",
			fieldPath: "metadata.annotations",
			obj: &api.Pod{
				ObjectMeta: api.ObjectMeta{
					Annotations: map[string]string{"builder": "john-doe"},
				},
			},
			expectedValue: "builder=\"john-doe\"",
		},

		{
			name:      "invalid expression",
			fieldPath: "metadata.whoops",
			obj: &api.Pod{
				ObjectMeta: api.ObjectMeta{
					Namespace: "object-namespace",
				},
			},
			expectedMessageFragment: "Unsupported fieldPath",
		},
	}

	for _, tc := range cases {
		actual, err := ExtractFieldPathAsString(tc.obj, tc.fieldPath)
		if err != nil {
			if tc.expectedMessageFragment != "" {
				if !strings.Contains(err.Error(), tc.expectedMessageFragment) {
					t.Errorf("%v: Unexpected error message: %q, expected to contain %q", tc.name, err, tc.expectedMessageFragment)
				}
			} else {
				t.Errorf("%v: unexpected error: %v", tc.name, err)
			}
		} else if e := tc.expectedValue; e != "" && e != actual {
			t.Errorf("%v: Unexpected result; got %q, expected %q", tc.name, actual, e)
		}
	}
}

func getPod(cname, cpuRequest, cpuLimit, memoryRequest, memoryLimit string) *api.Pod {
	resources := api.ResourceRequirements{
		Limits:   make(api.ResourceList),
		Requests: make(api.ResourceList),
	}
	if cpuLimit != "" {
		resources.Limits[api.ResourceCPU] = resource.MustParse(cpuLimit)
	}
	if memoryLimit != "" {
		resources.Limits[api.ResourceMemory] = resource.MustParse(memoryLimit)
	}
	if cpuRequest != "" {
		resources.Requests[api.ResourceCPU] = resource.MustParse(cpuRequest)
	}
	if memoryRequest != "" {
		resources.Requests[api.ResourceMemory] = resource.MustParse(memoryRequest)
	}
	return &api.Pod{
		Spec: api.PodSpec{
			Containers: []api.Container{
				{
					Name:      cname,
					Resources: resources,
				},
			},
		},
	}
}

func TestExtractResourceValue(t *testing.T) {
	cases := []struct {
		fs            *api.ResourceFieldSelector
		pod           *api.Pod
		cName         string
		expectedValue string
		expectedError error
	}{
		{
			fs: &api.ResourceFieldSelector{
				Resource: "limits.cpu",
			},
			cName:         "foo",
			pod:           getPod("foo", "", "9", "", ""),
			expectedValue: "9",
		},
		{
			fs: &api.ResourceFieldSelector{
				Resource: "requests.cpu",
			},
			cName:         "foo",
			pod:           getPod("foo", "", "", "", ""),
			expectedValue: "0",
		},
		{
			fs: &api.ResourceFieldSelector{
				Resource: "requests.cpu",
			},
			cName:         "foo",
			pod:           getPod("foo", "8", "", "", ""),
			expectedValue: "8",
		},
		{
			fs: &api.ResourceFieldSelector{
				Resource: "requests.cpu",
			},
			cName:         "foo",
			pod:           getPod("foo", "100m", "", "", ""),
			expectedValue: "1",
		},
		{
			fs: &api.ResourceFieldSelector{
				Resource: "requests.cpu",
				Divisor:  resource.MustParse("100m"),
			},
			cName:         "foo",
			pod:           getPod("foo", "1200m", "", "", ""),
			expectedValue: "12",
		},
		{
			fs: &api.ResourceFieldSelector{
				Resource: "requests.memory",
			},
			cName:         "foo",
			pod:           getPod("foo", "", "", "100Mi", ""),
			expectedValue: "104857600",
		},
		{
			fs: &api.ResourceFieldSelector{
				Resource: "requests.memory",
				Divisor:  resource.MustParse("1Mi"),
			},
			cName:         "foo",
			pod:           getPod("foo", "", "", "100Mi", "1Gi"),
			expectedValue: "100",
		},
		{
			fs: &api.ResourceFieldSelector{
				Resource: "limits.memory",
			},
			cName:         "foo",
			pod:           getPod("foo", "", "", "10Mi", "100Mi"),
			expectedValue: "104857600",
		},
	}
	as := assert.New(t)
	for idx, tc := range cases {
		actual, err := ExtractResourceValueByContainerName(tc.fs, tc.pod, tc.cName)
		if tc.expectedError != nil {
			as.Equal(tc.expectedError, err, "expected test case [%d] to fail with error %v; got %v", idx, tc.expectedError, err)
		} else {
			as.Nil(err, "expected test case [%d] to not return an error; got %v", idx, err)
			as.Equal(tc.expectedValue, actual, "expected test case [%d] to return %q; got %q instead", idx, tc.expectedValue, actual)
		}
	}
}
