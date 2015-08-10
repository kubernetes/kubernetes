/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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

package cmd

import (
	"testing"

	"github.com/spf13/cobra"
	"k8s.io/kubernetes/pkg/api"
)

func TestGetRestartPolicy(t *testing.T) {
	tests := []struct {
		input       string
		interactive bool
		expected    api.RestartPolicy
		expectErr   bool
	}{
		{
			input:    "",
			expected: api.RestartPolicyAlways,
		},
		{
			input:       "",
			interactive: true,
			expected:    api.RestartPolicyOnFailure,
		},
		{
			input:       string(api.RestartPolicyAlways),
			interactive: true,
			expected:    api.RestartPolicyAlways,
		},
		{
			input:       string(api.RestartPolicyNever),
			interactive: true,
			expected:    api.RestartPolicyNever,
		},
		{
			input:    string(api.RestartPolicyAlways),
			expected: api.RestartPolicyAlways,
		},
		{
			input:    string(api.RestartPolicyNever),
			expected: api.RestartPolicyNever,
		},
		{
			input:     "foo",
			expectErr: true,
		},
	}
	for _, test := range tests {
		cmd := &cobra.Command{}
		cmd.Flags().String("restart", "", "dummy restart flag")
		cmd.Flags().Lookup("restart").Value.Set(test.input)
		policy, err := getRestartPolicy(cmd, test.interactive)
		if test.expectErr && err == nil {
			t.Error("unexpected non-error")
		}
		if !test.expectErr && err != nil {
			t.Errorf("unexpected error: %v", err)
		}
		if !test.expectErr && policy != test.expected {
			t.Errorf("expected: %s, saw: %s (%s:%v)", test.expected, policy, test.input, test.interactive)
		}
	}
}
