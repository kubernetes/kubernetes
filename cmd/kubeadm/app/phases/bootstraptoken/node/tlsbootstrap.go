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

package node

import (
	"fmt"

	rbac "k8s.io/api/rbac/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/cmd/kubeadm/app/constants"
	"k8s.io/kubernetes/cmd/kubeadm/app/util/apiclient"
)

const (
	// NodeBootstrapperClusterRoleName defines the name of the auto-bootstrapped ClusterRole for letting someone post a CSR
	// TODO: This value should be defined in an other, generic authz package instead of here
	NodeBootstrapperClusterRoleName = "system:node-bootstrapper"
	// NodeKubeletBootstrap defines the name of the ClusterRoleBinding that lets kubelets post CSRs
	NodeKubeletBootstrap = "kubeadm:kubelet-bootstrap"

	// CSRAutoApprovalClusterRoleName defines the name of the auto-bootstrapped ClusterRole for making the csrapprover controller auto-approve the CSR
	// TODO: This value should be defined in an other, generic authz package instead of here
	// Starting from v1.8, CSRAutoApprovalClusterRoleName is automatically created by the API server on startup
	CSRAutoApprovalClusterRoleName = "system:certificates.k8s.io:certificatesigningrequests:nodeclient"
	// NodeSelfCSRAutoApprovalClusterRoleName is a role defined in default 1.8 RBAC policies for automatic CSR approvals for automatically rotated node certificates
	NodeSelfCSRAutoApprovalClusterRoleName = "system:certificates.k8s.io:certificatesigningrequests:selfnodeclient"
	// NodeAutoApproveBootstrapClusterRoleBinding defines the name of the ClusterRoleBinding that makes the csrapprover approve node CSRs
	NodeAutoApproveBootstrapClusterRoleBinding = "kubeadm:node-autoapprove-bootstrap"
	// NodeAutoApproveCertificateRotationClusterRoleBinding defines name of the ClusterRoleBinding that makes the csrapprover approve node auto rotated CSRs
	NodeAutoApproveCertificateRotationClusterRoleBinding = "kubeadm:node-autoapprove-certificate-rotation"
)

// AllowBootstrapTokensToPostCSRs creates RBAC rules in a way the makes Node Bootstrap Tokens able to post CSRs
func AllowBootstrapTokensToPostCSRs(client clientset.Interface) error {
	fmt.Println("[bootstraptoken] configured RBAC rules to allow Node Bootstrap tokens to post CSRs in order for nodes to get long term certificate credentials")

	return apiclient.CreateOrUpdateClusterRoleBinding(client, &rbac.ClusterRoleBinding{
		ObjectMeta: metav1.ObjectMeta{
			Name: NodeKubeletBootstrap,
		},
		RoleRef: rbac.RoleRef{
			APIGroup: rbac.GroupName,
			Kind:     "ClusterRole",
			Name:     NodeBootstrapperClusterRoleName,
		},
		Subjects: []rbac.Subject{
			{
				Kind: rbac.GroupKind,
				Name: constants.NodeBootstrapTokenAuthGroup,
			},
		},
	})
}

// AutoApproveNodeBootstrapTokens creates RBAC rules in a way that makes Node Bootstrap Tokens' CSR auto-approved by the csrapprover controller
func AutoApproveNodeBootstrapTokens(client clientset.Interface) error {
	fmt.Println("[bootstraptoken] configured RBAC rules to allow the csrapprover controller automatically approve CSRs from a Node Bootstrap Token")

	// Always create this kubeadm-specific binding though
	return apiclient.CreateOrUpdateClusterRoleBinding(client, &rbac.ClusterRoleBinding{
		ObjectMeta: metav1.ObjectMeta{
			Name: NodeAutoApproveBootstrapClusterRoleBinding,
		},
		RoleRef: rbac.RoleRef{
			APIGroup: rbac.GroupName,
			Kind:     "ClusterRole",
			Name:     CSRAutoApprovalClusterRoleName,
		},
		Subjects: []rbac.Subject{
			{
				Kind: "Group",
				Name: constants.NodeBootstrapTokenAuthGroup,
			},
		},
	})
}

// AutoApproveNodeCertificateRotation creates RBAC rules in a way that makes Node certificate rotation CSR auto-approved by the csrapprover controller
func AutoApproveNodeCertificateRotation(client clientset.Interface) error {
	fmt.Println("[bootstraptoken] configured RBAC rules to allow certificate rotation for all node client certificates in the cluster")

	return apiclient.CreateOrUpdateClusterRoleBinding(client, &rbac.ClusterRoleBinding{
		ObjectMeta: metav1.ObjectMeta{
			Name: NodeAutoApproveCertificateRotationClusterRoleBinding,
		},
		RoleRef: rbac.RoleRef{
			APIGroup: rbac.GroupName,
			Kind:     "ClusterRole",
			Name:     NodeSelfCSRAutoApprovalClusterRoleName,
		},
		Subjects: []rbac.Subject{
			{
				Kind: "Group",
				Name: constants.NodesGroup,
			},
		},
	})
}
