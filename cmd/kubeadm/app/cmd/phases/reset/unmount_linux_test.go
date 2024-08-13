//go:build linux
// +build linux

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

package phases

import (
	"testing"

	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
)

func TestFlagsToInt(t *testing.T) {

	tests := []struct {
		name           string
		input          []string
		expectedOutput int
	}{
		{
			name:           "nil input",
			input:          nil,
			expectedOutput: 0,
		},
		{
			name:           "no flags",
			input:          []string{},
			expectedOutput: 0,
		},
		{
			name: "all flags",
			input: []string{
				kubeadmapi.UnmountFlagMNTForce,
				kubeadmapi.UnmountFlagMNTDetach,
				kubeadmapi.UnmountFlagMNTExpire,
				kubeadmapi.UnmountFlagUmountNoFollow,
			},
			expectedOutput: 15,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			out := flagsToInt(tc.input)
			if tc.expectedOutput != out {
				t.Errorf("expected output %d, got %d", tc.expectedOutput, out)
			}
		})
	}
}
