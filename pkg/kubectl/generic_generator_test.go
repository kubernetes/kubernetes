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

package kubectl

import (
	"reflect"
	"testing"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/testapi"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/util/sets"
)

func TestGetParams(t *testing.T) {
	tests := []struct {
		template   string
		paramNames sets.String
	}{
		{
			template:   "{{ foo }} {{ bar }}",
			paramNames: sets.NewString("foo", "bar"),
		},
		{
			template: `{{ foo }}
			           {{ bar }} {{ baz }}`,
			paramNames: sets.NewString("foo", "bar", "baz"),
		},
	}

	for _, test := range tests {
		params := getParams([]byte(test.template))
		if len(params) != len(test.paramNames) {
			t.Errorf("unexpected params: %v, for %s", params, test.template)
		}
		for ix := range params {
			if !test.paramNames.Has(params[ix].Name) {
				t.Errorf("unexpected param: %v", params[ix])
			}
		}
	}
}

func TestValidateTemplate(t *testing.T) {
	tests := []struct {
		template  string
		valid     bool
		errorLine int
	}{
		{
			template: `{{ ok }}
					   {{ template }}`,
			valid: true,
		},
		{
			template:  "{{ invalid }} {{ template",
			valid:     false,
			errorLine: 1,
		},
		{
			template: `{{ ok-line-one }}
			           {{ error-line-two`,
			valid:     false,
			errorLine: 2,
		},
	}
	for _, test := range tests {
		err := validateTemplate([]byte(test.template))
		if test.valid {
			if err != nil {
				t.Errorf("unexpected error: %v", err)
			}
		} else {
			if err == nil {
				t.Error("unexpected non error")
			}
			if err.(*ValidationError).LineNumber != test.errorLine {
				t.Errorf("unexpected error line: %d, expected: %d for %s", err.(*ValidationError).LineNumber, test.errorLine, test.template)
			}
		}
	}
}

func TestRegexReplace(t *testing.T) {
	tests := []struct {
		input     string
		expected  string
		params    map[string]interface{}
		expectErr bool
	}{
		{
			input:    "{{ simple }}",
			params:   map[string]interface{}{"simple": "foo"},
			expected: "foo",
		},
		{
			input:     "{{ missing }}",
			params:    map[string]interface{}{"simple": "foo"},
			expectErr: true,
		},
		{
			input: "{{multiple }} {{  values }} {{with}} {{ different }} {{	spacing	}}",
			params: map[string]interface{}{
				"multiple":  "this",
				"values":    "is",
				"with":      "the",
				"different": "expected",
				"spacing":   "output",
			},
			expected: "this is the expected output",
		},
		{
			input: `{{ handles }}
{{ newlines }}`,
			params: map[string]interface{}{"handles": "foo", "newlines": "bar"},
			expected: `foo
bar`,
		},
	}

	for _, test := range tests {
		data, err := regexpReplace([]byte(test.input), test.params)
		if test.expectErr {
			if err == nil {
				t.Error("unexpected non-error")
			}
			continue
		}
		if err != nil {
			t.Errorf("unexpected error: %v", err)
		}
		output := string(data)
		if output != test.expected {
			t.Errorf("expected: %s, saw: %s", test.expected, output)
		}
	}
}

const podTemplate = `apiVersion: v1
kind: Pod
metadata:
  name: {{ name }}
  labels:
    name: {{ name }}
spec:
  containers:
  - name: {{ name }}
    image: {{ image }}
    ports:
    - containerPort: {{ port }}`

func int64Ptr(val int64) *int64 {
	return &val
}

func TestDecode(t *testing.T) {
	tests := []struct {
		template string
		params   map[string]interface{}
		expected runtime.Object
	}{
		{
			template: podTemplate,
			params: map[string]interface{}{
				"name":  "foo",
				"image": "bar",
				"port":  "80",
			},
			expected: &api.Pod{
				ObjectMeta: api.ObjectMeta{
					Name:   "foo",
					Labels: map[string]string{"name": "foo"},
				},
				Spec: api.PodSpec{
					Containers: []api.Container{
						{
							Name:  "foo",
							Image: "bar",
							Ports: []api.ContainerPort{
								{
									ContainerPort: 80,
									Protocol:      "TCP",
								},
							},
							TerminationMessagePath: "/dev/termination-log",
							ImagePullPolicy:        api.PullIfNotPresent,
						},
					},
					RestartPolicy:                 api.RestartPolicyAlways,
					DNSPolicy:                     api.DNSClusterFirst,
					TerminationGracePeriodSeconds: int64Ptr(30),
					SecurityContext:               &api.PodSecurityContext{false, false, false},
				},
			},
		},
	}

	for _, test := range tests {
		generator, err := NewGenericGeneratorFromBytes([]byte(test.template), testapi.Default.Codec())
		if err != nil {
			t.Errorf("unexpected error: %v", err)
		}
		obj, err := generator.Generate(test.params)
		if err != nil {
			t.Errorf("unexpected error: %v", err)
		}
		if !reflect.DeepEqual(obj, test.expected) {
			t.Errorf("expected:\n%#v\nsaw:\n%#v\n%v", test.expected, obj, *obj.(*api.Pod).Spec.SecurityContext)
		}
	}
}
