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

package options

import (
	"reflect"
	"testing"

	kubeadmapiv1 "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/v1beta4"
)

func TestArgSliceStringSet(t *testing.T) {
	tests := []struct {
		name   string
		input  *[]kubeadmapiv1.Arg
		output string
	}{
		// a test case with nil is input is not needed because ExtraArgs are never nil in ClusterConfiguration
		{
			name:   "no args",
			input:  &[]kubeadmapiv1.Arg{},
			output: "",
		},
		{
			name: "one arg",
			input: &[]kubeadmapiv1.Arg{
				{Name: "foo", Value: "bar"},
			},
			output: "foo=bar",
		},
		{
			name: "two args",
			input: &[]kubeadmapiv1.Arg{
				{Name: "foo", Value: "bar"},
				{Name: "baz", Value: "qux"},
			},
			output: "foo=bar,baz=qux",
		},
		{
			name: "three args with a duplicate",
			input: &[]kubeadmapiv1.Arg{
				{Name: "foo", Value: "bar"},
				{Name: "foo", Value: "qux"},
				{Name: "baz", Value: "qux"},
			},
			output: "foo=bar,foo=qux,baz=qux",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			s := newArgSlice(tt.input)
			gotString := s.String()
			if ok := reflect.DeepEqual(gotString, tt.output); !ok {
				t.Errorf("String()\ngot: %v\noutput: %v", gotString, tt.output)
			}
			_ = s.Set(gotString)
			if ok := reflect.DeepEqual(s.args, tt.input); !ok {
				t.Errorf("Set()\ngot: %+v\noutput: %+v", s.args, tt.input)
			}
		})
	}
}
