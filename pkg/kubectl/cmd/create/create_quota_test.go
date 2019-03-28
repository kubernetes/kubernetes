/*
Copyright 2016 The Kubernetes Authors.

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

package create

import (
	"testing"

	"k8s.io/api/core/v1"
	"k8s.io/cli-runtime/pkg/genericclioptions"
	cmdtesting "k8s.io/kubectl/pkg/cmd/testing"
)

func TestCreateQuota(t *testing.T) {
	resourceQuotaObject := &v1.ResourceQuota{}
	resourceQuotaObject.Name = "my-quota"

	tests := map[string]struct {
		flags          []string
		expectedOutput string
	}{
		"single resource": {
			flags:          []string{"--hard=cpu=1"},
			expectedOutput: "resourcequota/" + resourceQuotaObject.Name + "\n",
		},
		"single resource with a scope": {
			flags:          []string{"--hard=cpu=1", "--scopes=BestEffort"},
			expectedOutput: "resourcequota/" + resourceQuotaObject.Name + "\n",
		},
		"multiple resources": {
			flags:          []string{"--hard=cpu=1,pods=42", "--scopes=BestEffort"},
			expectedOutput: "resourcequota/" + resourceQuotaObject.Name + "\n",
		},
		"single resource with multiple scopes": {
			flags:          []string{"--hard=cpu=1", "--scopes=BestEffort,NotTerminating"},
			expectedOutput: "resourcequota/" + resourceQuotaObject.Name + "\n",
		},
	}
	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			tf := cmdtesting.NewTestFactory().WithNamespace("test")
			defer tf.Cleanup()

			ioStreams, _, buf, _ := genericclioptions.NewTestIOStreams()
			cmd := NewCmdCreateQuota(tf, ioStreams)
			cmd.Flags().Parse(test.flags)
			cmd.Flags().Set("output", "name")
			cmd.Run(cmd, []string{resourceQuotaObject.Name})

			if buf.String() != test.expectedOutput {
				t.Errorf("%s: expected output: %s, but got: %s", name, test.expectedOutput, buf.String())
			}
		})
	}
}
