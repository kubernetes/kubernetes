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

package apiconfig

import (
	"fmt"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/pkg/api/v1"
	rbac "k8s.io/client-go/pkg/apis/rbac/v1beta1"
	kubeadmconstants "k8s.io/kubernetes/cmd/kubeadm/app/constants"
	bootstrapapi "k8s.io/kubernetes/pkg/bootstrap/api"
)

const (
	// KubeProxyClusterRoleName sets the name for the kube-proxy ClusterRole
	KubeProxyClusterRoleName = "system:node-proxier"
	// NodeBootstrapperClusterRoleName sets the name for the TLS Node Bootstrapper ClusterRole
	NodeBootstrapperClusterRoleName = "system:node-bootstrapper"
	// BootstrapSignerClusterRoleName sets the name for the ClusterRole that allows access to ConfigMaps in the kube-public ns
	BootstrapSignerClusterRoleName = "system:bootstrap-signer-clusterinfo"

	clusterRoleKind          = "ClusterRole"
	roleKind                 = "Role"
	serviceAccountKind       = "ServiceAccount"
	rbacAPIGroup             = "rbac.authorization.k8s.io"
	anonymousUser            = "system:anonymous"
	nodeAutoApproveBootstrap = "kubeadm:node-autoapprove-bootstrap"
)

// TODO: Are there any unit tests that could be made for this file other than duplicating all values and logic in a separate file?

// CreateServiceAccounts creates the necessary serviceaccounts that kubeadm uses/might use.
func CreateServiceAccounts(clientset *clientset.Clientset) error {
	serviceAccounts := []v1.ServiceAccount{
		{
			ObjectMeta: metav1.ObjectMeta{
				Name:      kubeadmconstants.KubeDNSServiceAccountName,
				Namespace: metav1.NamespaceSystem,
			},
		},
		{
			ObjectMeta: metav1.ObjectMeta{
				Name:      kubeadmconstants.KubeProxyServiceAccountName,
				Namespace: metav1.NamespaceSystem,
			},
		},
	}

	for _, sa := range serviceAccounts {
		if _, err := clientset.CoreV1().ServiceAccounts(metav1.NamespaceSystem).Create(&sa); err != nil {
			return err
		}
	}
	return nil
}

// CreateRBACRules creates the essential RBAC rules for a minimally set-up cluster
func CreateRBACRules(clientset *clientset.Clientset) error {
	if err := createRoles(clientset); err != nil {
		return err
	}
	if err := createRoleBindings(clientset); err != nil {
		return err
	}
	if err := createClusterRoles(clientset); err != nil {
		return err
	}
	if err := createClusterRoleBindings(clientset); err != nil {
		return err
	}

	fmt.Println("[apiconfig] Created RBAC rules")
	return nil
}

func createRoles(clientset *clientset.Clientset) error {
	roles := []rbac.Role{
		{
			ObjectMeta: metav1.ObjectMeta{
				Name:      BootstrapSignerClusterRoleName,
				Namespace: metav1.NamespacePublic,
			},
			Rules: []rbac.PolicyRule{
				rbac.NewRule("get").Groups("").Resources("configmaps").RuleOrDie(),
			},
		},
	}
	for _, role := range roles {
		if _, err := clientset.RbacV1beta1().Roles(metav1.NamespacePublic).Create(&role); err != nil {
			return err
		}
	}
	return nil
}

func createRoleBindings(clientset *clientset.Clientset) error {
	roleBindings := []rbac.RoleBinding{
		{
			ObjectMeta: metav1.ObjectMeta{
				Name:      "kubeadm:bootstrap-signer-clusterinfo",
				Namespace: metav1.NamespacePublic,
			},
			RoleRef: rbac.RoleRef{
				APIGroup: rbacAPIGroup,
				Kind:     roleKind,
				Name:     BootstrapSignerClusterRoleName,
			},
			Subjects: []rbac.Subject{
				{
					Kind: "User",
					Name: anonymousUser,
				},
			},
		},
	}

	for _, roleBinding := range roleBindings {
		if _, err := clientset.RbacV1beta1().RoleBindings(metav1.NamespacePublic).Create(&roleBinding); err != nil {
			return err
		}
	}
	return nil
}

func createClusterRoles(clientset *clientset.Clientset) error {
	clusterRoles := []rbac.ClusterRole{
		{
			ObjectMeta: metav1.ObjectMeta{
				Name: nodeAutoApproveBootstrap,
			},
			Rules: []rbac.PolicyRule{
				rbac.NewRule("create").Groups("certificates.k8s.io").Resources("certificatesigningrequests/nodeclient").RuleOrDie(),
			},
		},
	}

	for _, roleBinding := range clusterRoles {
		if _, err := clientset.RbacV1beta1().ClusterRoles().Create(&roleBinding); err != nil {
			return err
		}
	}
	return nil
}

func createClusterRoleBindings(clientset *clientset.Clientset) error {
	clusterRoleBindings := []rbac.ClusterRoleBinding{
		{
			ObjectMeta: metav1.ObjectMeta{
				Name: "kubeadm:kubelet-bootstrap",
			},
			RoleRef: rbac.RoleRef{
				APIGroup: rbacAPIGroup,
				Kind:     clusterRoleKind,
				Name:     NodeBootstrapperClusterRoleName,
			},
			Subjects: []rbac.Subject{
				{
					Kind: "Group",
					Name: bootstrapapi.BootstrapGroup,
				},
			},
		},
		{
			ObjectMeta: metav1.ObjectMeta{
				Name: nodeAutoApproveBootstrap,
			},
			RoleRef: rbac.RoleRef{
				APIGroup: rbacAPIGroup,
				Kind:     clusterRoleKind,
				Name:     nodeAutoApproveBootstrap,
			},
			Subjects: []rbac.Subject{
				{
					Kind: "Group",
					Name: bootstrapapi.BootstrapGroup,
				},
			},
		},
		{
			ObjectMeta: metav1.ObjectMeta{
				Name: "kubeadm:node-proxier",
			},
			RoleRef: rbac.RoleRef{
				APIGroup: rbacAPIGroup,
				Kind:     clusterRoleKind,
				Name:     KubeProxyClusterRoleName,
			},
			Subjects: []rbac.Subject{
				{
					Kind:      serviceAccountKind,
					Name:      kubeadmconstants.KubeProxyServiceAccountName,
					Namespace: metav1.NamespaceSystem,
				},
			},
		},
	}

	for _, clusterRoleBinding := range clusterRoleBindings {
		if _, err := clientset.RbacV1beta1().ClusterRoleBindings().Create(&clusterRoleBinding); err != nil {
			return err
		}
	}
	return nil
}
