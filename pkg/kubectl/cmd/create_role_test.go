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
	"io"
	"reflect"
	"testing"

	"k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/client-go/rest/fake"
	"k8s.io/kubernetes/pkg/apis/rbac"
	cmdtesting "k8s.io/kubernetes/pkg/kubectl/cmd/testing"
)

type testRolePrinter struct {
	CachedRole *rbac.Role
}

func (t *testRolePrinter) PrintObj(obj runtime.Object, out io.Writer) error {
	t.CachedRole = obj.(*rbac.Role)
	return nil
}

func (t *testRolePrinter) AfterPrint(output io.Writer, res string) error {
	return nil
}

func (t *testRolePrinter) HandledResources() []string {
	return []string{}
}

func (t *testRolePrinter) IsGeneric() bool {
	return true
}

func TestCreateRole(t *testing.T) {
	roleName := "my-role"

	f, tf, _, _ := cmdtesting.NewAPIFactory()
	printer := &testRolePrinter{}
	tf.Printer = printer
	tf.Namespace = "test"
	tf.Client = &fake.RESTClient{}
	tf.ClientConfig = defaultClientConfig()

	tests := map[string]struct {
		verbs         string
		resources     string
		resourceNames string
		expectedRole  *rbac.Role
	}{
		"test-duplicate-resources": {
			verbs:     "get,watch,list",
			resources: "pods,pods",
			expectedRole: &rbac.Role{
				ObjectMeta: v1.ObjectMeta{
					Name: roleName,
				},
				Rules: []rbac.PolicyRule{
					{
						Verbs:         []string{"get", "watch", "list"},
						Resources:     []string{"pods"},
						APIGroups:     []string{""},
						ResourceNames: []string{},
					},
				},
			},
		},
		"test-subresources": {
			verbs:     "get,watch,list",
			resources: "replicasets/scale",
			expectedRole: &rbac.Role{
				ObjectMeta: v1.ObjectMeta{
					Name: roleName,
				},
				Rules: []rbac.PolicyRule{
					{
						Verbs:         []string{"get", "watch", "list"},
						Resources:     []string{"replicasets/scale"},
						APIGroups:     []string{"extensions"},
						ResourceNames: []string{},
					},
				},
			},
		},
		"test-subresources-with-apigroup": {
			verbs:     "get,watch,list",
			resources: "replicasets.extensions/scale",
			expectedRole: &rbac.Role{
				ObjectMeta: v1.ObjectMeta{
					Name: roleName,
				},
				Rules: []rbac.PolicyRule{
					{
						Verbs:         []string{"get", "watch", "list"},
						Resources:     []string{"replicasets/scale"},
						APIGroups:     []string{"extensions"},
						ResourceNames: []string{},
					},
				},
			},
		},
		"test-valid-case-with-multiple-apigroups": {
			verbs:     "get,watch,list",
			resources: "pods,deployments.extensions",
			expectedRole: &rbac.Role{
				ObjectMeta: v1.ObjectMeta{
					Name: roleName,
				},
				Rules: []rbac.PolicyRule{
					{
						Verbs:         []string{"get", "watch", "list"},
						Resources:     []string{"pods"},
						APIGroups:     []string{""},
						ResourceNames: []string{},
					},
					{
						Verbs:         []string{"get", "watch", "list"},
						Resources:     []string{"deployments"},
						APIGroups:     []string{"extensions"},
						ResourceNames: []string{},
					},
				},
			},
		},
	}

	for name, test := range tests {
		buf := bytes.NewBuffer([]byte{})
		cmd := NewCmdCreateRole(f, buf)
		cmd.Flags().Set("dry-run", "true")
		cmd.Flags().Set("output", "object")
		cmd.Flags().Set("verb", test.verbs)
		cmd.Flags().Set("resource", test.resources)
		if test.resourceNames != "" {
			cmd.Flags().Set("resource-name", test.resourceNames)
		}
		cmd.Run(cmd, []string{roleName})
		if !reflect.DeepEqual(test.expectedRole, printer.CachedRole) {
			t.Errorf("%s:\nexpected:\n%#v\nsaw:\n%#v", name, test.expectedRole, printer.CachedRole)
		}
	}
}

func TestValidate(t *testing.T) {
	f, tf, _, _ := cmdtesting.NewAPIFactory()
	tf.Printer = &testPrinter{}
	tf.Namespace = "test"

	tests := map[string]struct {
		roleOptions *CreateRoleOptions
		expectErr   bool
	}{
		"test-missing-name": {
			roleOptions: &CreateRoleOptions{},
			expectErr:   true,
		},
		"test-missing-verb": {
			roleOptions: &CreateRoleOptions{
				Name: "my-role",
			},
			expectErr: true,
		},
		"test-missing-resource": {
			roleOptions: &CreateRoleOptions{
				Name:  "my-role",
				Verbs: []string{"get"},
			},
			expectErr: true,
		},
		"test-missing-resource-existing-apigroup": {
			roleOptions: &CreateRoleOptions{
				Name:  "my-role",
				Verbs: []string{"get"},
				Resources: []ResourceOptions{
					{
						Group: "extensions",
					},
				},
			},
			expectErr: true,
		},
		"test-missing-resource-existing-subresource": {
			roleOptions: &CreateRoleOptions{
				Name:  "my-role",
				Verbs: []string{"get"},
				Resources: []ResourceOptions{
					{
						SubResource: "scale",
					},
				},
			},
			expectErr: true,
		},
		"test-invalid-verb": {
			roleOptions: &CreateRoleOptions{
				Name:  "my-role",
				Verbs: []string{"invalid-verb"},
				Resources: []ResourceOptions{
					{
						Resource: "pods",
					},
				},
			},
			expectErr: true,
		},
		"test-nonresource-verb": {
			roleOptions: &CreateRoleOptions{
				Name:  "my-role",
				Verbs: []string{"post"},
				Resources: []ResourceOptions{
					{
						Resource: "pods",
					},
				},
			},
			expectErr: true,
		},
		"test-special-verb": {
			roleOptions: &CreateRoleOptions{
				Name:  "my-role",
				Verbs: []string{"use"},
				Resources: []ResourceOptions{
					{
						Resource: "pods",
					},
				},
			},
			expectErr: true,
		},
		"test-mix-verbs": {
			roleOptions: &CreateRoleOptions{
				Name:  "my-role",
				Verbs: []string{"impersonate", "use"},
				Resources: []ResourceOptions{
					{
						Resource:    "userextras",
						SubResource: "scopes",
					},
				},
			},
			expectErr: true,
		},
		"test-special-verb-with-wrong-apigroup": {
			roleOptions: &CreateRoleOptions{
				Name:  "my-role",
				Verbs: []string{"impersonate"},
				Resources: []ResourceOptions{
					{
						Resource:    "userextras",
						SubResource: "scopes",
						Group:       "extensions",
					},
				},
			},
			expectErr: true,
		},
		"test-invalid-resource": {
			roleOptions: &CreateRoleOptions{
				Name:  "my-role",
				Verbs: []string{"get"},
				Resources: []ResourceOptions{
					{
						Resource: "invalid-resource",
					},
				},
			},
			expectErr: true,
		},
		"test-resource-name-with-multiple-resources": {
			roleOptions: &CreateRoleOptions{
				Name:  "my-role",
				Verbs: []string{"get"},
				Resources: []ResourceOptions{
					{
						Resource: "pods",
					},
					{
						Resource: "deployments",
						Group:    "extensions",
					},
				},
				ResourceNames: []string{"foo"},
			},
			expectErr: false,
		},
		"test-valid-case": {
			roleOptions: &CreateRoleOptions{
				Name:  "role-binder",
				Verbs: []string{"get", "list", "bind"},
				Resources: []ResourceOptions{
					{
						Resource: "roles",
						Group:    "rbac.authorization.k8s.io",
					},
				},
				ResourceNames: []string{"foo"},
			},
			expectErr: false,
		},
		"test-valid-case-with-subresource": {
			roleOptions: &CreateRoleOptions{
				Name:  "my-role",
				Verbs: []string{"get", "list"},
				Resources: []ResourceOptions{
					{
						Resource:    "replicasets",
						SubResource: "scale",
					},
				},
				ResourceNames: []string{"bar"},
			},
			expectErr: false,
		},
		"test-valid-case-with-additional-resource": {
			roleOptions: &CreateRoleOptions{
				Name:  "my-role",
				Verbs: []string{"impersonate"},
				Resources: []ResourceOptions{
					{
						Resource:    "userextras",
						SubResource: "scopes",
						Group:       "authentication.k8s.io",
					},
				},
			},
			expectErr: false,
		},
	}

	for name, test := range tests {
		test.roleOptions.Mapper, _ = f.Object()
		err := test.roleOptions.Validate()
		if test.expectErr && err == nil {
			t.Errorf("%s: expect error happens but validate passes.", name)
		}
		if !test.expectErr && err != nil {
			t.Errorf("%s: unexpected error: %v", name, err)
		}
	}
}

func TestComplete(t *testing.T) {
	roleName := "my-role"

	f, tf, _, _ := cmdtesting.NewAPIFactory()
	tf.Printer = &testPrinter{}
	tf.Namespace = "test"
	tf.Client = &fake.RESTClient{}
	tf.ClientConfig = defaultClientConfig()

	buf := bytes.NewBuffer([]byte{})
	cmd := NewCmdCreateRole(f, buf)
	cmd.Flags().Set("resource", "pods,deployments.extensions")

	tests := map[string]struct {
		params      []string
		roleOptions *CreateRoleOptions
		expected    *CreateRoleOptions
		expectErr   bool
	}{
		"test-missing-name": {
			params:      []string{},
			roleOptions: &CreateRoleOptions{},
			expectErr:   true,
		},
		"test-duplicate-verbs": {
			params: []string{roleName},
			roleOptions: &CreateRoleOptions{
				Name: roleName,
				Verbs: []string{
					"get",
					"watch",
					"list",
					"get",
				},
			},
			expected: &CreateRoleOptions{
				Name: roleName,
				Verbs: []string{
					"get",
					"watch",
					"list",
				},
				Resources: []ResourceOptions{
					{
						Resource: "pods",
						Group:    "",
					},
					{
						Resource: "deployments",
						Group:    "extensions",
					},
				},
				ResourceNames: []string{},
			},
			expectErr: false,
		},
		"test-verball": {
			params: []string{roleName},
			roleOptions: &CreateRoleOptions{
				Name: roleName,
				Verbs: []string{
					"get",
					"watch",
					"list",
					"*",
				},
			},
			expected: &CreateRoleOptions{
				Name:  roleName,
				Verbs: []string{"*"},
				Resources: []ResourceOptions{
					{
						Resource: "pods",
						Group:    "",
					},
					{
						Resource: "deployments",
						Group:    "extensions",
					},
				},
				ResourceNames: []string{},
			},
			expectErr: false,
		},
		"test-duplicate-resourcenames": {
			params: []string{roleName},
			roleOptions: &CreateRoleOptions{
				Name:          roleName,
				Verbs:         []string{"*"},
				ResourceNames: []string{"foo", "foo"},
			},
			expected: &CreateRoleOptions{
				Name:  roleName,
				Verbs: []string{"*"},
				Resources: []ResourceOptions{
					{
						Resource: "pods",
						Group:    "",
					},
					{
						Resource: "deployments",
						Group:    "extensions",
					},
				},
				ResourceNames: []string{"foo"},
			},
			expectErr: false,
		},
		"test-valid-complete-case": {
			params: []string{roleName},
			roleOptions: &CreateRoleOptions{
				Name:          roleName,
				Verbs:         []string{"*"},
				ResourceNames: []string{"foo"},
			},
			expected: &CreateRoleOptions{
				Name:  roleName,
				Verbs: []string{"*"},
				Resources: []ResourceOptions{
					{
						Resource: "pods",
						Group:    "",
					},
					{
						Resource: "deployments",
						Group:    "extensions",
					},
				},
				ResourceNames: []string{"foo"},
			},
			expectErr: false,
		},
	}

	for name, test := range tests {
		err := test.roleOptions.Complete(f, cmd, test.params)
		if !test.expectErr && err != nil {
			t.Errorf("%s: unexpected error: %v", name, err)
		}

		if test.expectErr {
			if err != nil {
				continue
			} else {
				t.Errorf("%s: expect error happens but test passes.", name)
			}
		}

		if test.roleOptions.Name != test.expected.Name {
			t.Errorf("%s:\nexpected name:\n%#v\nsaw name:\n%#v", name, test.expected.Name, test.roleOptions.Name)
		}

		if !reflect.DeepEqual(test.roleOptions.Verbs, test.expected.Verbs) {
			t.Errorf("%s:\nexpected verbs:\n%#v\nsaw verbs:\n%#v", name, test.expected.Verbs, test.roleOptions.Verbs)
		}

		if !reflect.DeepEqual(test.roleOptions.Resources, test.expected.Resources) {
			t.Errorf("%s:\nexpected resources:\n%#v\nsaw resources:\n%#v", name, test.expected.Resources, test.roleOptions.Resources)
		}

		if !reflect.DeepEqual(test.roleOptions.ResourceNames, test.expected.ResourceNames) {
			t.Errorf("%s:\nexpected resource names:\n%#v\nsaw resource names:\n%#v", name, test.expected.ResourceNames, test.roleOptions.ResourceNames)
		}
	}
}
