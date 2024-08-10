/*
Copyright 2023 The Kubernetes Authors.

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

package node

import (
	"context"
	"testing"

	rbac "k8s.io/api/rbac/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	clientset "k8s.io/client-go/kubernetes"
	clientsetfake "k8s.io/client-go/kubernetes/fake"
	"k8s.io/kubernetes/cmd/kubeadm/app/constants"
)

func TestAllowBootstrapTokensToPostCSRs(t *testing.T) {
	tests := []struct {
		name   string
		client clientset.Interface
	}{
		{
			name:   "ClusterRoleBindings is empty",
			client: clientsetfake.NewSimpleClientset(),
		},
		{
			name: "ClusterRoleBindings already exists",
			client: newMockClusterRoleBinddingClientForTest(t, &rbac.ClusterRoleBinding{
				ObjectMeta: metav1.ObjectMeta{
					Name: constants.NodeKubeletBootstrap,
				},
				RoleRef: rbac.RoleRef{
					APIGroup: rbac.GroupName,
					Kind:     "ClusterRole",
					Name:     constants.NodeBootstrapperClusterRoleName,
				},
				Subjects: []rbac.Subject{
					{
						Kind: rbac.GroupKind,
						Name: constants.NodeBootstrapTokenAuthGroup,
					},
				},
			}),
		},
		{
			name: "Create new ClusterRoleBindings",
			client: newMockClusterRoleBinddingClientForTest(t, &rbac.ClusterRoleBinding{
				ObjectMeta: metav1.ObjectMeta{
					Name: constants.NodeKubeletBootstrap,
				},
				RoleRef: rbac.RoleRef{
					APIGroup: rbac.GroupName,
					Kind:     "ClusterRole",
					Name:     constants.NodeBootstrapperClusterRoleName,
				},
				Subjects: []rbac.Subject{
					{
						Kind: rbac.GroupKind,
						Name: constants.KubeProxyClusterRoleName,
					},
				},
			}),
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if err := AllowBootstrapTokensToPostCSRs(tt.client); err != nil {
				t.Errorf("AllowBootstrapTokensToPostCSRs() return error = %v", err)
			}
		})
	}
}

func TestAutoApproveNodeBootstrapTokens(t *testing.T) {
	tests := []struct {
		name   string
		client clientset.Interface
	}{
		{
			name:   "ClusterRoleBindings is empty",
			client: clientsetfake.NewSimpleClientset(),
		},
		{
			name: "ClusterRoleBindings already exists",
			client: newMockClusterRoleBinddingClientForTest(t, &rbac.ClusterRoleBinding{
				ObjectMeta: metav1.ObjectMeta{
					Name: constants.NodeAutoApproveBootstrapClusterRoleBinding,
				},
				RoleRef: rbac.RoleRef{
					APIGroup: rbac.GroupName,
					Kind:     "ClusterRole",
					Name:     constants.CSRAutoApprovalClusterRoleName,
				},
				Subjects: []rbac.Subject{
					{
						Kind: rbac.GroupKind,
						Name: constants.NodeBootstrapTokenAuthGroup,
					},
				},
			}),
		},
		{
			name: "Create new ClusterRoleBindings",
			client: newMockClusterRoleBinddingClientForTest(t, &rbac.ClusterRoleBinding{
				ObjectMeta: metav1.ObjectMeta{
					Name: constants.NodeAutoApproveBootstrapClusterRoleBinding,
				},
				RoleRef: rbac.RoleRef{
					APIGroup: rbac.GroupName,
					Kind:     "ClusterRole",
					Name:     constants.NodeSelfCSRAutoApprovalClusterRoleName,
				},
				Subjects: []rbac.Subject{
					{
						Kind: rbac.GroupKind,
						Name: constants.NodeBootstrapTokenAuthGroup,
					},
				},
			}),
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if err := AutoApproveNodeBootstrapTokens(tt.client); err != nil {
				t.Errorf("AutoApproveNodeBootstrapTokens() return error = %v", err)
			}
		})
	}
}

func TestAutoApproveNodeCertificateRotation(t *testing.T) {
	tests := []struct {
		name   string
		client clientset.Interface
	}{
		{
			name:   "ClusterRoleBindings is empty",
			client: clientsetfake.NewSimpleClientset(),
		},
		{
			name: "ClusterRoleBindings already exists",
			client: newMockClusterRoleBinddingClientForTest(t, &rbac.ClusterRoleBinding{
				ObjectMeta: metav1.ObjectMeta{
					Name: constants.NodeAutoApproveCertificateRotationClusterRoleBinding,
				},
				RoleRef: rbac.RoleRef{
					APIGroup: rbac.GroupName,
					Kind:     "ClusterRole",
					Name:     constants.NodeSelfCSRAutoApprovalClusterRoleName,
				},
				Subjects: []rbac.Subject{
					{
						Kind: rbac.GroupKind,
						Name: constants.NodesGroup,
					},
				},
			}),
		},
		{
			name: "Create new ClusterRoleBindings",
			client: newMockClusterRoleBinddingClientForTest(t, &rbac.ClusterRoleBinding{
				ObjectMeta: metav1.ObjectMeta{
					Name: constants.NodeAutoApproveCertificateRotationClusterRoleBinding,
				},
				RoleRef: rbac.RoleRef{
					APIGroup: rbac.GroupName,
					Kind:     "ClusterRole",
					Name:     constants.NodeSelfCSRAutoApprovalClusterRoleName,
				},
				Subjects: []rbac.Subject{
					{
						Kind: rbac.GroupKind,
						Name: constants.NodeBootstrapTokenAuthGroup,
					},
				},
			}),
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if err := AutoApproveNodeCertificateRotation(tt.client); err != nil {
				t.Errorf("AutoApproveNodeCertificateRotation() return error = %v", err)
			}
		})
	}
}

func TestAllowBootstrapTokensToGetNodes(t *testing.T) {
	tests := []struct {
		name   string
		client clientset.Interface
	}{
		{
			name:   "RBAC rules are empty",
			client: clientsetfake.NewSimpleClientset(),
		},
		{
			name: "RBAC rules already exists",
			client: newMockRbacClientForTest(t, &rbac.ClusterRole{
				ObjectMeta: metav1.ObjectMeta{
					Name: constants.GetNodesClusterRoleName,
				},
				Rules: []rbac.PolicyRule{
					{
						Verbs:     []string{"get"},
						APIGroups: []string{""},
						Resources: []string{"nodes"},
					},
				},
			}, &rbac.ClusterRoleBinding{
				ObjectMeta: metav1.ObjectMeta{
					Name: constants.GetNodesClusterRoleName,
				},
				RoleRef: rbac.RoleRef{
					APIGroup: rbac.GroupName,
					Kind:     "ClusterRole",
					Name:     constants.GetNodesClusterRoleName,
				},
				Subjects: []rbac.Subject{
					{
						Kind: rbac.GroupKind,
						Name: constants.NodeBootstrapTokenAuthGroup,
					},
				},
			}),
		},
		{
			name: "Create new RBAC rules",
			client: newMockRbacClientForTest(t, &rbac.ClusterRole{
				ObjectMeta: metav1.ObjectMeta{
					Name: constants.GetNodesClusterRoleName,
				},
				Rules: []rbac.PolicyRule{
					{
						Verbs:     []string{"create"},
						APIGroups: []string{""},
						Resources: []string{"nodes"},
					},
				},
			}, &rbac.ClusterRoleBinding{
				ObjectMeta: metav1.ObjectMeta{
					Name: constants.GetNodesClusterRoleName,
				},
				RoleRef: rbac.RoleRef{
					APIGroup: rbac.GroupName,
					Kind:     "ClusterRole",
					Name:     constants.NodeAutoApproveBootstrapClusterRoleBinding,
				},
				Subjects: []rbac.Subject{
					{
						Kind: rbac.GroupKind,
						Name: constants.NodeBootstrapTokenAuthGroup,
					},
				},
			}),
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if err := AllowBootstrapTokensToGetNodes(tt.client); err != nil {
				t.Errorf("AllowBootstrapTokensToGetNodes() return error = %v", err)
			}
		})
	}
}

func newMockClusterRoleBinddingClientForTest(t *testing.T, clusterRoleBinding *rbac.ClusterRoleBinding) *clientsetfake.Clientset {
	client := clientsetfake.NewSimpleClientset()
	_, err := client.RbacV1().ClusterRoleBindings().Create(context.TODO(), clusterRoleBinding, metav1.CreateOptions{})

	if err != nil {
		t.Fatalf("error creating ClusterRoleBindings: %v", err)
	}
	return client
}

func newMockRbacClientForTest(t *testing.T, clusterRole *rbac.ClusterRole, clusterRoleBinding *rbac.ClusterRoleBinding) *clientsetfake.Clientset {
	client := clientsetfake.NewSimpleClientset()
	_, err := client.RbacV1().ClusterRoles().Create(context.TODO(), clusterRole, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("error creating ClusterRoles: %v", err)
	}
	_, err = client.RbacV1().ClusterRoleBindings().Create(context.TODO(), clusterRoleBinding, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("error creating ClusterRoleBindings: %v", err)
	}
	return client
}
