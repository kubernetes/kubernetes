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

	"github.com/spf13/cobra"
	rbacv1 "k8s.io/api/rbac/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/cli-runtime/pkg/genericclioptions"
	cmdtesting "k8s.io/kubectl/pkg/cmd/testing"
)

func TestClusterRoleBindingStrategy(t *testing.T) {
	tests := []struct {
		Args     []string
		SetFlags func(cmd *cobra.Command)
		Expected *rbacv1.ClusterRoleBinding
	}{
		{
			Args: []string{"fake-binding"},
			SetFlags: func(cmd *cobra.Command) {
				_ = cmd.Flags().Set("clusterrole", "fake-clusterrole")
				_ = cmd.Flags().Set("user", "fake-user")
				_ = cmd.Flags().Set("group", "fake-group")
				_ = cmd.Flags().Set("serviceaccount", "fake-namespace:fake-account")
			},
			Expected: &rbacv1.ClusterRoleBinding{
				ObjectMeta: v1.ObjectMeta{
					Name: "fake-binding",
				},
				TypeMeta: v1.TypeMeta{
					Kind:       "ClusterRoleBinding",
					APIVersion: "rbac.authorization.k8s.io/v1",
				},
				RoleRef: rbacv1.RoleRef{
					APIGroup: rbacv1.GroupName,
					Kind:     "ClusterRole",
					Name:     "fake-clusterrole",
				},
				Subjects: []rbacv1.Subject{
					{
						Kind:     rbacv1.UserKind,
						APIGroup: "rbac.authorization.k8s.io",
						Name:     "fake-user",
					},
					{
						Kind:     rbacv1.GroupKind,
						APIGroup: "rbac.authorization.k8s.io",
						Name:     "fake-group",
					},
					{
						Kind:      rbacv1.ServiceAccountKind,
						Namespace: "fake-namespace",
						Name:      "fake-account",
					},
				},
			},
		},
	}
	for i, tt := range tests {
		t.Run(string(i), func(t *testing.T) {
			ioStreams, _, _, _ := genericclioptions.NewTestIOStreams()
			tf := cmdtesting.NewTestFactory()
			defer tf.Cleanup()

			strategy := &ClusterRoleBindingStrategy{}
			cmd := NewCreateSubCmd(tf, ioStreams, strategy)
			cmd.Run = func(cmd *cobra.Command, args []string) {
				strategy.SetName(args[0])
				obj, err := strategy.CreateObject()
				if err != nil {
					t.Fatalf("unexpected error %v", err)
				}
				if !reflect.DeepEqual(obj, tt.Expected) {
					t.Fatalf("TestCreateClusterRoleBinding: expected:\n%#v\ngot:\n%#v", tt.Expected, obj)
				}
			}
			tt.SetFlags(cmd)
			cmd.Run(cmd, tt.Args)
		})
	}
}
