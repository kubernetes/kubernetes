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

	"github.com/stretchr/testify/assert"

	"k8s.io/kubernetes/pkg/apis/rbac"
	cmdtesting "k8s.io/kubernetes/pkg/kubectl/cmd/testing"
)

func TestCreateRole(t *testing.T) {
	roleObject := &rbac.Role{}
	roleObject.Name = "my-role"

	f, tf, _, _ := cmdtesting.NewAPIFactory()
	tf.Printer = &testPrinter{}
	tf.Namespace = "test"

	buf := bytes.NewBuffer([]byte{})
	cmd := NewCmdCreateRole(f, buf)
	cmd.Flags().Set("dry-run", "true")
	cmd.Flags().Set("output", "name")
	cmd.Flags().Set("verb", "get")
	cmd.Flags().Set("resource", "pods")
	cmd.Run(cmd, []string{roleObject.Name})

	expectedOutput := "role/" + roleObject.Name + "\n"
	if buf.String() != expectedOutput {
		t.Errorf("expected output: %s, but got: %s", expectedOutput, buf.String())
	}
}

func TestCreateRoleWithoutVerb(t *testing.T) {
	roleObject := &rbac.Role{}
	roleObject.Name = "my-role"

	f, tf, _, _ := cmdtesting.NewAPIFactory()
	tf.Printer = &testPrinter{}
	tf.Namespace = "test"

	buf := bytes.NewBuffer([]byte{})
	cmd := NewCmdCreateRole(f, buf)
	cmd.Flags().Set("dry-run", "true")
	cmd.Flags().Set("output", "name")
	cmd.Flags().Set("resource", "pods")

	err := CreateRole(f, buf, cmd, []string{"my-role"})
	assert.Error(t, err, "at least one verb must be specified")
}

func TestCreateRoleWithoutResource(t *testing.T) {
	roleObject := &rbac.Role{}
	roleObject.Name = "my-role"

	f, tf, _, _ := cmdtesting.NewAPIFactory()
	tf.Printer = &testPrinter{}
	tf.Namespace = "test"

	buf := bytes.NewBuffer([]byte{})
	cmd := NewCmdCreateRole(f, buf)
	cmd.Flags().Set("dry-run", "true")
	cmd.Flags().Set("output", "name")
	cmd.Flags().Set("verb", "get")

	err := CreateRole(f, buf, cmd, []string{"my-role"})
	assert.Error(t, err, "at least one resource must be specified")
}
