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
	"fmt"
	"strings"

	"github.com/spf13/cobra"
	rbacv1 "k8s.io/api/rbac/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/kubectl/pkg/util/i18n"
	"k8s.io/kubectl/pkg/util/templates"
)

type ClusterRoleBindingStrategy struct {
	Name            string
	ClusterRole     string
	Users           []string
	Groups          []string
	ServiceAccounts []string
}

func (s *ClusterRoleBindingStrategy) Use() string {
	return "clusterrolebinding NAME --clusterrole=NAME [--user=username] [--group=groupname] [--serviceaccount=namespace:serviceaccountname] [--dry-run=server|client|none]"
}

func (s *ClusterRoleBindingStrategy) Short() string {
	return i18n.T("Create a ClusterRoleBinding for a particular ClusterRole")
}

func (s *ClusterRoleBindingStrategy) Long() string {
	return templates.LongDesc(i18n.T(`Create a ClusterRoleBinding for a particular ClusterRole.`))
}

func (s *ClusterRoleBindingStrategy) Example() string {
	return templates.Examples(i18n.T(`
		  # Create a ClusterRoleBinding for user1, user2, and group1 using the cluster-admin ClusterRole
		  kubectl create clusterrolebinding cluster-admin --clusterrole=cluster-admin --user=user1 --user=user2 --group=group1`))
}

func (s *ClusterRoleBindingStrategy) SetCmdFlags(cmd *cobra.Command) error {
	cmd.Flags().StringVar(&s.ClusterRole, "clusterrole", "", i18n.T("ClusterRole this ClusterRoleBinding should reference"))
	_ = cmd.MarkFlagRequired("clusterrole")
	_ = cmd.MarkFlagCustom("clusterrole", "__kubectl_get_resource_clusterrole")
	cmd.Flags().StringArrayVar(&s.Users, "user", []string{}, "Usernames to bind to the clusterrole")
	cmd.Flags().StringArrayVar(&s.Groups, "group", []string{}, "Groups to bind to the clusterrole")
	cmd.Flags().StringArrayVar(&s.ServiceAccounts, "serviceaccount", []string{}, "Service accounts to bind to the clusterrole, in the format <namespace>:<name>")
	return nil
}

func (s *ClusterRoleBindingStrategy) SetName(name string) {
	s.Name = name
}

func (s *ClusterRoleBindingStrategy) GroupVersionKind() schema.GroupVersionKind {
	return schema.GroupVersionKind{}
}

func (s *ClusterRoleBindingStrategy) CreateObject() (Object, error) {
	clusterRoleBinding := &rbacv1.ClusterRoleBinding{
		TypeMeta: metav1.TypeMeta{APIVersion: rbacv1.SchemeGroupVersion.String(), Kind: "ClusterRoleBinding"},
		ObjectMeta: metav1.ObjectMeta{
			Name: s.Name,
		},
		RoleRef: rbacv1.RoleRef{
			APIGroup: rbacv1.GroupName,
			Kind:     "ClusterRole",
			Name:     s.ClusterRole,
		},
	}

	for _, user := range s.Users {
		clusterRoleBinding.Subjects = append(clusterRoleBinding.Subjects, rbacv1.Subject{
			Kind:     rbacv1.UserKind,
			APIGroup: rbacv1.GroupName,
			Name:     user,
		})
	}

	for _, group := range s.Groups {
		clusterRoleBinding.Subjects = append(clusterRoleBinding.Subjects, rbacv1.Subject{
			Kind:     rbacv1.GroupKind,
			APIGroup: rbacv1.GroupName,
			Name:     group,
		})
	}

	for _, sa := range s.ServiceAccounts {
		tokens := strings.Split(sa, ":")
		if len(tokens) != 2 || tokens[0] == "" || tokens[1] == "" {
			return nil, fmt.Errorf("serviceaccount must be <namespace>:<name>")
		}
		clusterRoleBinding.Subjects = append(clusterRoleBinding.Subjects, rbacv1.Subject{
			Kind:      rbacv1.ServiceAccountKind,
			APIGroup:  "",
			Namespace: tokens[0],
			Name:      tokens[1],
		})
	}

	return clusterRoleBinding, nil
}

var _ NameSetter = &ClusterRoleBindingStrategy{}
