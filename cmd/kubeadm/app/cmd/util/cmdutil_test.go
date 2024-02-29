/*
Copyright 2017 The Kubernetes Authors.

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

package util

import (
	"fmt"
	"testing"

	"github.com/spf13/pflag"
	"k8s.io/client-go/tools/clientcmd"
)

func TestValidateExactArgNumber(t *testing.T) {
	var tests = []struct {
		name                string
		args, supportedArgs []string
		expectedErr         bool
	}{
		{
			name:          "one arg given and one arg expected",
			args:          []string{"my-node-1234"},
			supportedArgs: []string{"node-name"},
			expectedErr:   false,
		},
		{
			name:          "two args given and two args expected",
			args:          []string{"my-node-1234", "foo"},
			supportedArgs: []string{"node-name", "second-toplevel-arg"},
			expectedErr:   false,
		},
		{
			name:          "too few supplied args",
			args:          []string{},
			supportedArgs: []string{"node-name"},
			expectedErr:   true,
		},
		{
			name:          "too few non-empty args",
			args:          []string{""},
			supportedArgs: []string{"node-name"},
			expectedErr:   true,
		},
		{
			name:          "too many args",
			args:          []string{"my-node-1234", "foo"},
			supportedArgs: []string{"node-name"},
			expectedErr:   true,
		},
	}
	for _, rt := range tests {
		t.Run(rt.name, func(t *testing.T) {
			actual := ValidateExactArgNumber(rt.args, rt.supportedArgs)
			if (actual != nil) != rt.expectedErr {
				t.Errorf(
					"failed ValidateExactArgNumber:\n\texpected error: %t\n\t  actual error: %t",
					rt.expectedErr,
					(actual != nil),
				)
			}
		})
	}
}

func TestGetKubeConfigPath(t *testing.T) {
	var tests = []struct {
		name     string
		file     string
		expected string
	}{
		{
			name:     "provide an empty value",
			file:     "",
			expected: clientcmd.NewDefaultClientConfigLoadingRules().GetDefaultFilename(),
		},
		{
			name:     "provide a non-empty value",
			file:     "kubelet.kubeconfig",
			expected: "kubelet.kubeconfig",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			actualResult := GetKubeConfigPath(tt.file)
			if actualResult != tt.expected {
				t.Errorf(
					"failed GetKubeConfigPath:\n\texpected: %s\n\t  actual: %s",
					tt.expected,
					actualResult,
				)
			}
		})
	}
}

func TestValueFromFlagsOrConfig(t *testing.T) {
	var tests = []struct {
		name      string
		flag      string
		cfg       interface{}
		flagValue interface{}
		expected  interface{}
	}{
		{
			name:      "string: config is overridden by the flag",
			flag:      "foo",
			cfg:       "foo_cfg",
			flagValue: "foo_flag",
			expected:  "foo_flag",
		},
		{
			name:      "bool: config is overridden by the flag",
			flag:      "bar",
			cfg:       true,
			flagValue: false,
			expected:  false,
		},
		{
			name:      "nil bool is converted to false",
			cfg:       (*bool)(nil),
			flagValue: false,
			expected:  false,
		},
	}
	for _, tt := range tests {
		type options struct {
			foo string
			bar bool
		}
		fakeOptions := &options{}
		fs := pflag.FlagSet{}
		fs.StringVar(&fakeOptions.foo, "foo", "", "")
		fs.BoolVar(&fakeOptions.bar, "bar", false, "")

		t.Run(tt.name, func(t *testing.T) {
			if tt.flag != "" {
				if err := fs.Set(tt.flag, fmt.Sprintf("%v", tt.flagValue)); err != nil {
					t.Fatalf("failed to set the value of the flag %v", tt.flagValue)
				}
			}
			actualResult := ValueFromFlagsOrConfig(&fs, tt.flag, tt.cfg, tt.flagValue)
			if result, ok := actualResult.(*bool); ok {
				actualResult = *result
			}
			if actualResult != tt.expected {
				t.Errorf(
					"failed ValueFromFlagsOrConfig:\n\texpected: %s\n\t  actual: %s",
					tt.expected,
					actualResult,
				)
			}
		})
	}
}
