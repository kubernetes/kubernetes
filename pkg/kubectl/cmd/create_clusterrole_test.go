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
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/kubernetes/pkg/apis/rbac"
	cmdtesting "k8s.io/kubernetes/pkg/kubectl/cmd/testing"
)

type testClusterRolePrinter struct {
	CachedClusterRole *rbac.ClusterRole
}

func (t *testClusterRolePrinter) PrintObj(obj runtime.Object, out io.Writer) error {
	t.CachedClusterRole = obj.(*rbac.ClusterRole)
	return nil
}

func (t *testClusterRolePrinter) AfterPrint(output io.Writer, res string) error {
	return nil
}

func (t *testClusterRolePrinter) HandledResources() []string {
	return []string{}
}

func TestCreateClusterRole(t *testing.T) {
	clusterRoleName := "my-cluster-role"

	f, tf, _, _ := cmdtesting.NewAPIFactory()
	printer := &testClusterRolePrinter{}
	tf.Printer = printer
	tf.Namespace = "test"

	tests := map[string]struct {
		verbs               string
		resources           string
		resourceNames       string
		nonResourceURLs     string
		expectedClusterRole *rbac.ClusterRole
	}{
		"test-duplicate-resources": {
			verbs:     "get,watch,list",
			resources: "pods,pods",
			expectedClusterRole: &rbac.ClusterRole{
				ObjectMeta: v1.ObjectMeta{
					Name: clusterRoleName,
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
		"test-valid-case-with-multiple-apigroups": {
			verbs:     "get,watch,list",
			resources: "pods,deployments.extensions",
			expectedClusterRole: &rbac.ClusterRole{
				ObjectMeta: v1.ObjectMeta{
					Name: clusterRoleName,
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
		"test-valid-case-with-nonresourceurl": {
			verbs:           "get,watch,list",
			resources:       "pods,deployments.extensions",
			nonResourceURLs: "/versions",
			expectedClusterRole: &rbac.ClusterRole{
				ObjectMeta: v1.ObjectMeta{
					Name: clusterRoleName,
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
					{
						Verbs:           []string{"get"},
						NonResourceURLs: []string{"/versions"},
					},
				},
			},
		},
	}

	for name, test := range tests {
		buf := bytes.NewBuffer([]byte{})
		cmd := NewCmdCreateClusterRole(f, buf)
		cmd.Flags().Set("dry-run", "true")
		cmd.Flags().Set("output", "object")
		cmd.Flags().Set("verb", test.verbs)
		cmd.Flags().Set("resource", test.resources)
		if test.resourceNames != "" {
			cmd.Flags().Set("resource-name", test.resourceNames)
		}
		if test.nonResourceURLs != "" {
			cmd.Flags().Set("non-resource-url", test.nonResourceURLs)
		}
		cmd.Run(cmd, []string{clusterRoleName})
		if !reflect.DeepEqual(test.expectedClusterRole, printer.CachedClusterRole) {
			t.Errorf("%s:\nexpected:\n%#v\nsaw:\n%#v", name, test.expectedClusterRole, printer.CachedClusterRole)
		}
	}
}

func TestClusterRoleOptoinsValidate(t *testing.T) {
	clusterRoleName := "my-cluster-role"

	f, tf, _, _ := cmdtesting.NewAPIFactory()
	tf.Printer = &testPrinter{}
	tf.Namespace = "test"

	tests := map[string]struct {
		clusterRoleOptions *CreateClusterRoleOptions
		expectErr          bool
	}{
		"test-missing-name": {
			clusterRoleOptions: &CreateClusterRoleOptions{},
			expectErr:          true,
		},
		"test-missing-resource-and-nonresourceurl": {
			clusterRoleOptions: &CreateClusterRoleOptions{
				Name:  clusterRoleName,
				Verbs: []string{"get"},
			},
			expectErr: true,
		},
		"test-invalid-verb": {
			clusterRoleOptions: &CreateClusterRoleOptions{
				Name:  clusterRoleName,
				Verbs: []string{"invalid-verb"},
				Resources: []schema.GroupVersionResource{
					{
						Resource: "pods",
					},
				},
			},
			expectErr: true,
		},
		"test-invalid-resource": {
			clusterRoleOptions: &CreateClusterRoleOptions{
				Name:  clusterRoleName,
				Verbs: []string{"get"},
				Resources: []schema.GroupVersionResource{
					{
						Resource: "invalid-resource",
					},
				},
			},
			expectErr: true,
		},
		"test-resourcename-with-no-resource-provided": {
			clusterRoleOptions: &CreateClusterRoleOptions{
				Name:            clusterRoleName,
				Verbs:           []string{"get"},
				ResourceNames:   []string{"foo"},
				NonResourceURLs: []string{"/versions"},
			},
			expectErr: true,
		},
		"test-resource-name-with-multiple-resources": {
			clusterRoleOptions: &CreateClusterRoleOptions{
				Name:  clusterRoleName,
				Verbs: []string{"get"},
				Resources: []schema.GroupVersionResource{
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
			expectErr: true,
		},
		"test-resource-with-no-resource-verb": {
			clusterRoleOptions: &CreateClusterRoleOptions{
				Name:  clusterRoleName,
				Verbs: []string{"post"},
				Resources: []schema.GroupVersionResource{
					{
						Resource: "pods",
					},
				},
			},
			expectErr: true,
		},
		"test-nonresourceurl-with-no-non-resource-verb": {
			clusterRoleOptions: &CreateClusterRoleOptions{
				Name:  clusterRoleName,
				Verbs: []string{"create"},
				Resources: []schema.GroupVersionResource{
					{
						Resource: "pods",
					},
				},
				NonResourceURLs: []string{"/api"},
			},
			expectErr: true,
		},
		"test-valid-case": {
			clusterRoleOptions: &CreateClusterRoleOptions{
				Name:  clusterRoleName,
				Verbs: []string{"get", "list"},
				Resources: []schema.GroupVersionResource{
					{
						Resource: "pods",
					},
				},
				ResourceNames:   []string{"foo"},
				NonResourceURLs: []string{"/api"},
			},
			expectErr: false,
		},
	}

	for name, test := range tests {
		err := test.clusterRoleOptions.Validate(f)
		if test.expectErr && err != nil {
			continue
		}
		if !test.expectErr && err != nil {
			t.Errorf("%s: unexpected error: %v", name, err)
		}
	}
}

func TestClusterRoleOptionsComplete(t *testing.T) {
	clusterRoleName := "my-cluster-role"

	f, tf, _, _ := cmdtesting.NewAPIFactory()
	tf.Printer = &testPrinter{}
	tf.Namespace = "test"

	buf := bytes.NewBuffer([]byte{})
	cmd := NewCmdCreateRole(f, buf)
	cmd.Flags().Set("resource", "pods,deployments.extensions")

	tests := map[string]struct {
		params             []string
		clusterRoleOptions *CreateClusterRoleOptions
		expected           *CreateClusterRoleOptions
		expectErr          bool
	}{
		"test-missing-name": {
			params:             []string{},
			clusterRoleOptions: &CreateClusterRoleOptions{},
			expectErr:          true,
		},
		"test-duplicate-verbs": {
			params: []string{clusterRoleName},
			clusterRoleOptions: &CreateClusterRoleOptions{
				Name: clusterRoleName,
				Verbs: []string{
					"get",
					"watch",
					"list",
					"get",
				},
			},
			expected: &CreateClusterRoleOptions{
				Name: clusterRoleName,
				Verbs: []string{
					"get",
					"watch",
					"list",
				},
				Resources: []schema.GroupVersionResource{
					{
						Resource: "pods",
						Group:    "",
					},
					{
						Resource: "deployments",
						Group:    "extensions",
					},
				},
				ResourceNames:   []string{},
				NonResourceURLs: []string{},
			},
			expectErr: false,
		},
		"test-verball": {
			params: []string{clusterRoleName},
			clusterRoleOptions: &CreateClusterRoleOptions{
				Name: clusterRoleName,
				Verbs: []string{
					"get",
					"watch",
					"list",
					"*",
				},
			},
			expected: &CreateClusterRoleOptions{
				Name:  clusterRoleName,
				Verbs: []string{"*"},
				Resources: []schema.GroupVersionResource{
					{
						Resource: "pods",
						Group:    "",
					},
					{
						Resource: "deployments",
						Group:    "extensions",
					},
				},
				ResourceNames:   []string{},
				NonResourceURLs: []string{},
			},
			expectErr: false,
		},
		"test-duplicate-resourcenames": {
			params: []string{clusterRoleName},
			clusterRoleOptions: &CreateClusterRoleOptions{
				Name:          clusterRoleName,
				Verbs:         []string{"*"},
				ResourceNames: []string{"foo", "foo"},
			},
			expected: &CreateClusterRoleOptions{
				Name:  clusterRoleName,
				Verbs: []string{"*"},
				Resources: []schema.GroupVersionResource{
					{
						Resource: "pods",
						Group:    "",
					},
					{
						Resource: "deployments",
						Group:    "extensions",
					},
				},
				ResourceNames:   []string{"foo"},
				NonResourceURLs: []string{},
			},
			expectErr: false,
		},
		"test-duplicate-nonresourceurl": {
			params: []string{clusterRoleName},
			clusterRoleOptions: &CreateClusterRoleOptions{
				Name:            clusterRoleName,
				Verbs:           []string{"*"},
				NonResourceURLs: []string{"/api", "/api"},
			},
			expected: &CreateClusterRoleOptions{
				Name:  clusterRoleName,
				Verbs: []string{"*"},
				Resources: []schema.GroupVersionResource{
					{
						Resource: "pods",
						Group:    "",
					},
					{
						Resource: "deployments",
						Group:    "extensions",
					},
				},
				ResourceNames:   []string{},
				NonResourceURLs: []string{"/api"},
			},
			expectErr: false,
		},
		"test-valid-complete-case": {
			params: []string{clusterRoleName},
			clusterRoleOptions: &CreateClusterRoleOptions{
				Name:            clusterRoleName,
				Verbs:           []string{"*"},
				ResourceNames:   []string{"foo"},
				NonResourceURLs: []string{"/api"},
			},
			expected: &CreateClusterRoleOptions{
				Name:  clusterRoleName,
				Verbs: []string{"*"},
				Resources: []schema.GroupVersionResource{
					{
						Resource: "pods",
						Group:    "",
					},
					{
						Resource: "deployments",
						Group:    "extensions",
					},
				},
				ResourceNames:   []string{"foo"},
				NonResourceURLs: []string{"/api"},
			},
			expectErr: false,
		},
	}

	for name, test := range tests {
		err := test.clusterRoleOptions.Complete(cmd, test.params)
		if !test.expectErr && err != nil {
			t.Errorf("%s: unexpected error: %v", name, err)
		}
		if test.expectErr && err != nil {
			continue
		}
		if !reflect.DeepEqual(test.clusterRoleOptions, test.expected) {
			t.Errorf("%s:\nexpected:\n%#v\nsaw:\n%#v", name, test.expected, test.clusterRoleOptions)
		}
	}
}
