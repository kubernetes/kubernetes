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

	rbac "k8s.io/api/rbac/v1beta1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/cmd/kubeadm/app/constants"
	"k8s.io/kubernetes/cmd/kubeadm/app/util/apiclient"
	rbachelper "k8s.io/kubernetes/pkg/apis/rbac/v1beta1"
	"k8s.io/kubernetes/pkg/util/version"
)

const (
	// NodeBootstrapperClusterRoleName defines the name of the auto-bootstrapped ClusterRole for letting someone post a CSR
	// TODO: This value should be defined in an other, generic authz package instead of here
	NodeBootstrapperClusterRoleName = "system:node-bootstrapper"
	// NodeKubeletBootstrap defines the name of the ClusterRoleBinding that lets kubelets post CSRs
	NodeKubeletBootstrap = "kubeadm:kubelet-bootstrap"

	// CSRAutoApprovalClusterRoleName defines the name of the auto-bootstrapped ClusterRole for making the csrapprover controller auto-approve the CSR
	// TODO: This value should be defined in an other, generic authz package instead of here
	CSRAutoApprovalClusterRoleName = "system:certificates.k8s.io:certificatesigningrequests:nodeclient"
	// NodeAutoApproveBootstrap defines the name of the ClusterRoleBinding that makes the csrapprover approve node CSRs
	NodeAutoApproveBootstrap = "kubeadm:node-autoapprove-bootstrap"
)

// AllowBootstrapTokensToPostCSRs creates RBAC rules in a way the makes Node Bootstrap Tokens able to post CSRs
func AllowBootstrapTokensToPostCSRs(client clientset.Interface) error {

	fmt.Println("[bootstraptoken] Configured RBAC rules to allow Node Bootstrap tokens to post CSRs in order for nodes to get long term certificate credentials")

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
func AutoApproveNodeBootstrapTokens(client clientset.Interface, k8sVersion *version.Version) error {

	fmt.Println("[bootstraptoken] Configured RBAC rules to allow the csrapprover controller automatically approve CSRs from a Node Bootstrap Token")

	// TODO: When the v1.9 cycle starts (targeting v1.9 at HEAD) and v1.8.0 is the minimum supported version, we can remove this function as the ClusterRole will always exist
	if k8sVersion.LessThan(constants.MinimumCSRAutoApprovalClusterRolesVersion) {

		err := apiclient.CreateOrUpdateClusterRole(client, &rbac.ClusterRole{
			ObjectMeta: metav1.ObjectMeta{
				Name: CSRAutoApprovalClusterRoleName,
			},
			Rules: []rbac.PolicyRule{
				rbachelper.NewRule("create").Groups("certificates.k8s.io").Resources("certificatesigningrequests/nodeclient").RuleOrDie(),
			},
		})
		if err != nil {
			return err
		}
	}

	// Always create this kubeadm-specific binding though
	return apiclient.CreateOrUpdateClusterRoleBinding(client, &rbac.ClusterRoleBinding{
		ObjectMeta: metav1.ObjectMeta{
			Name: NodeAutoApproveBootstrap,
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
