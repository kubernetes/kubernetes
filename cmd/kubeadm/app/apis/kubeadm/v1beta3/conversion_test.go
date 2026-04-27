/*
Copyright 2023 The Kubernetes Authors.

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

package v1beta3

import (
	"reflect"
	"testing"

	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
)

func TestConvertToArgs(t *testing.T) {
	var tests = []struct {
		name         string
		args         map[string]string
		expectedArgs []kubeadmapi.Arg
	}{
		{
			name:         "nil map returns nil args",
			args:         nil,
			expectedArgs: nil,
		},
		{
			name: "valid args are parsed (sorted)",
			args: map[string]string{"c": "d", "a": "b"},
			expectedArgs: []kubeadmapi.Arg{
				{Name: "a", Value: "b"},
				{Name: "c", Value: "d"},
			},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			actual := convertToArgs(tc.args)
			if !reflect.DeepEqual(tc.expectedArgs, actual) {
				t.Errorf("expected args: %v\n\t got: %v\n\t", tc.expectedArgs, actual)
			}
		})
	}
}

func TestConvertFromArgs(t *testing.T) {
	var tests = []struct {
		name         string
		args         []kubeadmapi.Arg
		expectedArgs map[string]string
	}{
		{
			name:         "nil args return nil map",
			args:         nil,
			expectedArgs: nil,
		},
		{
			name: "valid args are parsed",
			args: []kubeadmapi.Arg{
				{Name: "a", Value: "b"},
				{Name: "c", Value: "d"},
			},
			expectedArgs: map[string]string{"a": "b", "c": "d"},
		},
		{
			name: "duplicates are dropped",
			args: []kubeadmapi.Arg{
				{Name: "a", Value: "b"},
				{Name: "c", Value: "d1"},
				{Name: "c", Value: "d2"},
			},
			expectedArgs: map[string]string{"a": "b", "c": "d2"},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			actual := convertFromArgs(tc.args)
			if !reflect.DeepEqual(tc.expectedArgs, actual) {
				t.Errorf("expected args: %v\n\t got: %v\n\t", tc.expectedArgs, actual)
			}
		})
	}
}
