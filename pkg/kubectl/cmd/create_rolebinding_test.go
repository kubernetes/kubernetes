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

package cmd

import (
	"bytes"
	"testing"

	cmdtesting "k8s.io/kubernetes/pkg/kubectl/cmd/testing"
)

func TestCreateRoleBinding(t *testing.T) {
	roleBindingName := "my-rolebinding"
	f, tf, _, _ := cmdtesting.NewAPIFactory()
	tf.Printer = &testPrinter{}
	tf.Namespace = "test"

	tests := map[string]struct {
		flags map[string]string
	}{
		"bind role to user": {
			flags: map[string]string{"role": "my-role", "user": "fake-user"},
		},
		"bind role to group": {
			flags: map[string]string{"role": "my-role", "group": "fake-group"},
		},
		"bind role to service account": {
			flags: map[string]string{"role": "my-role", "serviceaccount": "fake-ns:fake-account"},
		},
		"bind cluster role": {
			flags: map[string]string{"clusterrole": "my-clusterrole", "user": "fake-user"},
		},
	}
	expectedOutput := "rolebinding/" + roleBindingName + "\n"

	for name, test := range tests {
		buf := bytes.NewBuffer([]byte{})
		cmd := NewCmdCreateRoleBinding(f, buf)
		cmd.Flags().Set("output", "name")
		cmd.Flags().Set("dry-run", "true")

		for k, v := range test.flags {
			cmd.Flags().Set(k, v)
		}

		cmd.Run(cmd, []string{roleBindingName})

		if buf.String() != expectedOutput {
			t.Errorf("%s: expected output: %s, but got: %s", name, expectedOutput, buf.String())
		}
	}
}
