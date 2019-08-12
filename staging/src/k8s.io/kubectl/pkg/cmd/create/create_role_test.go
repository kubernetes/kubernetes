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

package create

import (
	"reflect"
	"testing"

	rbac "k8s.io/api/rbac/v1"
	"k8s.io/apimachinery/pkg/api/equality"
	"k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/diff"
	"k8s.io/cli-runtime/pkg/genericclioptions"
	"k8s.io/client-go/rest/fake"
	cmdtesting "k8s.io/kubectl/pkg/cmd/testing"
	"k8s.io/kubectl/pkg/scheme"
)

func TestCreateRole(t *testing.T) {
	roleName := "my-role"

	tf := cmdtesting.NewTestFactory().WithNamespace("test")
	defer tf.Cleanup()

	tf.Client = &fake.RESTClient{}
	tf.ClientConfigVal = cmdtesting.DefaultClientConfig()

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
				TypeMeta: v1.TypeMeta{APIVersion: "rbac.authorization.k8s.io/v1", Kind: "Role"},
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
				TypeMeta: v1.TypeMeta{APIVersion: "rbac.authorization.k8s.io/v1", Kind: "Role"},
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
				TypeMeta: v1.TypeMeta{APIVersion: "rbac.authorization.k8s.io/v1", Kind: "Role"},
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
				TypeMeta: v1.TypeMeta{APIVersion: "rbac.authorization.k8s.io/v1", Kind: "Role"},
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
		t.Run(name, func(t *testing.T) {
			ioStreams, _, buf, _ := genericclioptions.NewTestIOStreams()
			cmd := NewCmdCreateRole(tf, ioStreams)
			cmd.Flags().Set("dry-run", "true")
			cmd.Flags().Set("output", "yaml")
			cmd.Flags().Set("verb", test.verbs)
			cmd.Flags().Set("resource", test.resources)
			if test.resourceNames != "" {
				cmd.Flags().Set("resource-name", test.resourceNames)
			}
			cmd.Run(cmd, []string{roleName})
			actual := &rbac.Role{}
			if err := runtime.DecodeInto(scheme.Codecs.UniversalDecoder(), buf.Bytes(), actual); err != nil {
				t.Log(string(buf.Bytes()))
				t.Fatal(err)
			}
			if !equality.Semantic.DeepEqual(test.expectedRole, actual) {
				t.Errorf("%s", diff.ObjectReflectDiff(test.expectedRole, actual))
			}
		})
	}
}

func TestValidate(t *testing.T) {
	tf := cmdtesting.NewTestFactory().WithNamespace("test")
	defer tf.Cleanup()

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
		var err error
		test.roleOptions.Mapper, err = tf.ToRESTMapper()
		if err != nil {
			t.Fatal(err)
		}
		err = test.roleOptions.Validate()
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

	tf := cmdtesting.NewTestFactory().WithNamespace("test")
	defer tf.Cleanup()

	tf.Client = &fake.RESTClient{}
	tf.ClientConfigVal = cmdtesting.DefaultClientConfig()

	defaultTestResources := "pods,deployments.extensions"

	tests := map[string]struct {
		params      []string
		resources   string
		roleOptions *CreateRoleOptions
		expected    *CreateRoleOptions
		expectErr   bool
	}{
		"test-missing-name": {
			params:    []string{},
			resources: defaultTestResources,
			roleOptions: &CreateRoleOptions{
				PrintFlags: genericclioptions.NewPrintFlags("created").WithTypeSetter(scheme.Scheme),
			},
			expectErr: true,
		},
		"test-duplicate-verbs": {
			params:    []string{roleName},
			resources: defaultTestResources,
			roleOptions: &CreateRoleOptions{
				PrintFlags: genericclioptions.NewPrintFlags("created").WithTypeSetter(scheme.Scheme),
				Name:       roleName,
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
			params:    []string{roleName},
			resources: defaultTestResources,
			roleOptions: &CreateRoleOptions{
				PrintFlags: genericclioptions.NewPrintFlags("created").WithTypeSetter(scheme.Scheme),
				Name:       roleName,
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
		"test-allresource": {
			params:    []string{roleName},
			resources: "*,pods",
			roleOptions: &CreateRoleOptions{
				PrintFlags: genericclioptions.NewPrintFlags("created"),
				Name:       roleName,
				Verbs:      []string{"*"},
			},
			expected: &CreateRoleOptions{
				Name:  roleName,
				Verbs: []string{"*"},
				Resources: []ResourceOptions{
					{
						Resource: "*",
					},
				},
				ResourceNames: []string{},
			},
			expectErr: false,
		},
		"test-allresource-subresource": {
			params:    []string{roleName},
			resources: "*/scale,pods",
			roleOptions: &CreateRoleOptions{
				PrintFlags: genericclioptions.NewPrintFlags("created"),
				Name:       roleName,
				Verbs:      []string{"*"},
			},
			expected: &CreateRoleOptions{
				Name:  roleName,
				Verbs: []string{"*"},
				Resources: []ResourceOptions{
					{
						Resource:    "*",
						SubResource: "scale",
					},
					{
						Resource: "pods",
					},
				},
				ResourceNames: []string{},
			},
			expectErr: false,
		},
		"test-allresrouce-allgroup": {
			params:    []string{roleName},
			resources: "*.*,pods",
			roleOptions: &CreateRoleOptions{
				PrintFlags: genericclioptions.NewPrintFlags("created"),
				Name:       roleName,
				Verbs:      []string{"*"},
			},
			expected: &CreateRoleOptions{
				Name:  roleName,
				Verbs: []string{"*"},
				Resources: []ResourceOptions{
					{
						Resource: "*",
						Group:    "*",
					},
					{
						Resource: "pods",
					},
				},
				ResourceNames: []string{},
			},
			expectErr: false,
		},
		"test-allresource-allgroup-subresource": {
			params:    []string{roleName},
			resources: "*.*/scale,pods",
			roleOptions: &CreateRoleOptions{
				PrintFlags: genericclioptions.NewPrintFlags("created"),
				Name:       roleName,
				Verbs:      []string{"*"},
			},
			expected: &CreateRoleOptions{
				Name:  roleName,
				Verbs: []string{"*"},
				Resources: []ResourceOptions{
					{
						Resource:    "*",
						Group:       "*",
						SubResource: "scale",
					},
					{
						Resource: "pods",
					},
				},
				ResourceNames: []string{},
			},
			expectErr: false,
		},
		"test-allresource-specificgroup": {
			params:    []string{roleName},
			resources: "*.extensions,pods",
			roleOptions: &CreateRoleOptions{
				PrintFlags: genericclioptions.NewPrintFlags("created"),
				Name:       roleName,
				Verbs:      []string{"*"},
			},
			expected: &CreateRoleOptions{
				Name:  roleName,
				Verbs: []string{"*"},
				Resources: []ResourceOptions{
					{
						Resource: "*",
						Group:    "extensions",
					},
					{
						Resource: "pods",
					},
				},
				ResourceNames: []string{},
			},
			expectErr: false,
		},
		"test-allresource-specificgroup-subresource": {
			params:    []string{roleName},
			resources: "*.extensions/scale,pods",
			roleOptions: &CreateRoleOptions{
				PrintFlags: genericclioptions.NewPrintFlags("created"),
				Name:       roleName,
				Verbs:      []string{"*"},
			},
			expected: &CreateRoleOptions{
				Name:  roleName,
				Verbs: []string{"*"},
				Resources: []ResourceOptions{
					{
						Resource:    "*",
						Group:       "extensions",
						SubResource: "scale",
					},
					{
						Resource: "pods",
					},
				},
				ResourceNames: []string{},
			},
			expectErr: false,
		},
		"test-duplicate-resourcenames": {
			params:    []string{roleName},
			resources: defaultTestResources,
			roleOptions: &CreateRoleOptions{
				PrintFlags:    genericclioptions.NewPrintFlags("created").WithTypeSetter(scheme.Scheme),
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
			params:    []string{roleName},
			resources: defaultTestResources,
			roleOptions: &CreateRoleOptions{
				PrintFlags:    genericclioptions.NewPrintFlags("created").WithTypeSetter(scheme.Scheme),
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
		cmd := NewCmdCreateRole(tf, genericclioptions.NewTestIOStreamsDiscard())
		cmd.Flags().Set("resource", test.resources)

		err := test.roleOptions.Complete(tf, cmd, test.params)
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
