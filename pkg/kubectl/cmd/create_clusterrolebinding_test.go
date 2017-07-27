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

type testClusterRoleBindingPrinter struct {
	CachedClusterRoleBinding *rbac.ClusterRoleBinding
}

func (t *testClusterRoleBindingPrinter) PrintObj(obj runtime.Object, out io.Writer) error {
	t.CachedClusterRoleBinding = obj.(*rbac.ClusterRoleBinding)
	return nil
}

func (t *testClusterRoleBindingPrinter) AfterPrint(output io.Writer, res string) error {
	return nil
}

func (t *testClusterRoleBindingPrinter) HandledResources() []string {
	return []string{}
}

func (t *testClusterRoleBindingPrinter) IsGeneric() bool {
	return true
}

func TestClusterRoleBinding(t *testing.T) {
	clusterRoleName := "my-cluster-role-binding"
	f, tf, _, _ := cmdtesting.NewAPIFactory()
	printer := &testClusterRoleBindingPrinter{}
	tf.Printer = printer
	tf.Namespace = "test"
	tf.Client = &fake.RESTClient{}
	tf.ClientConfig = defaultClientConfig()

	tests := map[string]struct {
		clusterrole                string
		users                      []string
		groups                     []string
		expectedClusterRoleBinding *rbac.ClusterRoleBinding
	}{
		"test-single-user": {
			clusterrole: "clusterrole-test",
			users: []string{
				"user1",
			},
			expectedClusterRoleBinding: &rbac.ClusterRoleBinding{
				ObjectMeta: v1.ObjectMeta{
					Name: clusterRoleName,
				},
				Subjects: []rbac.Subject{
					{
						Kind:     "User",
						APIGroup: "rbac.authorization.k8s.io",
						Name:     "user1",
					},
				},
				RoleRef: rbac.RoleRef{
					APIGroup: "rbac.authorization.k8s.io",
					Kind:     "ClusterRole",
					Name:     "clusterrole-test",
				},
			},
		},
		"test-signle-group": {
			clusterrole: "clusterrole-test",
			groups: []string{
				"group1",
			},
			expectedClusterRoleBinding: &rbac.ClusterRoleBinding{
				ObjectMeta: v1.ObjectMeta{
					Name: clusterRoleName,
				},
				Subjects: []rbac.Subject{
					{
						Kind:     "Group",
						APIGroup: "rbac.authorization.k8s.io",
						Name:     "group1",
					},
				},
				RoleRef: rbac.RoleRef{
					APIGroup: "rbac.authorization.k8s.io",
					Kind:     "ClusterRole",
					Name:     "clusterrole-test",
				},
			},
		},
		"test-multi-user": {
			clusterrole: "clusterrole-test",
			users: []string{
				"user1",
				"user2",
				"user3",
			},
			expectedClusterRoleBinding: &rbac.ClusterRoleBinding{
				ObjectMeta: v1.ObjectMeta{
					Name: clusterRoleName,
				},
				Subjects: []rbac.Subject{
					{
						Kind:     "User",
						APIGroup: "rbac.authorization.k8s.io",
						Name:     "user1",
					},
					{
						Kind:     "User",
						APIGroup: "rbac.authorization.k8s.io",
						Name:     "user2",
					},
					{
						Kind:     "User",
						APIGroup: "rbac.authorization.k8s.io",
						Name:     "user3",
					},
				},
				RoleRef: rbac.RoleRef{
					APIGroup: "rbac.authorization.k8s.io",
					Kind:     "ClusterRole",
					Name:     "clusterrole-test",
				},
			},
		},
		"test-multi-group": {
			clusterrole: "clusterrole-test",
			groups: []string{
				"group1",
				"group2",
				"group3",
			},
			expectedClusterRoleBinding: &rbac.ClusterRoleBinding{
				ObjectMeta: v1.ObjectMeta{
					Name: clusterRoleName,
				},
				Subjects: []rbac.Subject{
					{
						Kind:     "Group",
						APIGroup: "rbac.authorization.k8s.io",
						Name:     "group1",
					},
					{
						Kind:     "Group",
						APIGroup: "rbac.authorization.k8s.io",
						Name:     "group2",
					},
					{
						Kind:     "Group",
						APIGroup: "rbac.authorization.k8s.io",
						Name:     "group3",
					},
				},
				RoleRef: rbac.RoleRef{
					APIGroup: "rbac.authorization.k8s.io",
					Kind:     "ClusterRole",
					Name:     "clusterrole-test",
				},
			},
		},
		"test-multi-user-multi-group": {
			clusterrole: "clusterrole-test",
			users: []string{
				"user1",
				"user2",
				"user3",
			},
			groups: []string{
				"group1",
				"group2",
				"group3",
			},
			expectedClusterRoleBinding: &rbac.ClusterRoleBinding{
				ObjectMeta: v1.ObjectMeta{
					Name: clusterRoleName,
				},
				Subjects: []rbac.Subject{
					{
						Kind:     "User",
						APIGroup: "rbac.authorization.k8s.io",
						Name:     "user1",
					},
					{
						Kind:     "User",
						APIGroup: "rbac.authorization.k8s.io",
						Name:     "user2",
					},
					{
						Kind:     "User",
						APIGroup: "rbac.authorization.k8s.io",
						Name:     "user3",
					},
					{
						Kind:     "Group",
						APIGroup: "rbac.authorization.k8s.io",
						Name:     "group1",
					},
					{
						Kind:     "Group",
						APIGroup: "rbac.authorization.k8s.io",
						Name:     "group2",
					},
					{
						Kind:     "Group",
						APIGroup: "rbac.authorization.k8s.io",
						Name:     "group3",
					},
				},
				RoleRef: rbac.RoleRef{
					APIGroup: "rbac.authorization.k8s.io",
					Kind:     "ClusterRole",
					Name:     "clusterrole-test",
				},
			},
		},
		"test-dup-group": {
			clusterrole: "clusterrole-test",
			groups: []string{
				"group1",
				"group1",
				"group1",
			},
			expectedClusterRoleBinding: &rbac.ClusterRoleBinding{
				ObjectMeta: v1.ObjectMeta{
					Name: clusterRoleName,
				},
				Subjects: []rbac.Subject{
					{
						Kind:     "Group",
						APIGroup: "rbac.authorization.k8s.io",
						Name:     "group1",
					},
				},
				RoleRef: rbac.RoleRef{
					APIGroup: "rbac.authorization.k8s.io",
					Kind:     "ClusterRole",
					Name:     "clusterrole-test",
				},
			},
		},
		"test-dup-user": {
			clusterrole: "clusterrole-test",
			users: []string{
				"user1",
				"user1",
				"user1",
			},
			expectedClusterRoleBinding: &rbac.ClusterRoleBinding{
				ObjectMeta: v1.ObjectMeta{
					Name: clusterRoleName,
				},
				Subjects: []rbac.Subject{
					{
						Kind:     "User",
						APIGroup: "rbac.authorization.k8s.io",
						Name:     "user1",
					},
				},
				RoleRef: rbac.RoleRef{
					APIGroup: "rbac.authorization.k8s.io",
					Kind:     "ClusterRole",
					Name:     "clusterrole-test",
				},
			},
		},
	}

	for name, test := range tests {
		buf := bytes.NewBuffer([]byte{})
		cmd := NewCmdCreateClusterRoleBinding(f, buf)
		cmd.Flags().Set("dry-run", "true")
		cmd.Flags().Set("output", "object")
		cmd.Flags().Set("clusterrole", test.clusterrole)
		for _, user := range test.users {
			cmd.Flags().Set("user", user)
		}
		for _, group := range test.groups {
			cmd.Flags().Set("group", group)
		}
		cmd.Run(cmd, []string{clusterRoleName})
		if !reflect.DeepEqual(test.expectedClusterRoleBinding, printer.CachedClusterRoleBinding) {
			t.Errorf("%s:\nexpected:\n%#v\nsaw:\n%#v", name, test.expectedClusterRoleBinding, printer.CachedClusterRoleBinding)
		}
	}
}
