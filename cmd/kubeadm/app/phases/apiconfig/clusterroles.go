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
	kubeadmconstants "k8s.io/kubernetes/cmd/kubeadm/app/constants"
	"k8s.io/kubernetes/cmd/kubeadm/app/master"
	"k8s.io/kubernetes/pkg/api/v1"
	rbac "k8s.io/kubernetes/pkg/apis/rbac/v1beta1"
	"k8s.io/kubernetes/pkg/client/clientset_generated/clientset"
)

const (
	// TODO: This role should eventually be a system:-prefixed, automatically bootstrapped ClusterRole

	// KubeDNSClusterRoleName sets the name for the kube-dns ClusterRole
	KubeDNSClusterRoleName = "kubeadm:kube-dns"
	// KubeProxyClusterRoleName sets the name for the kube-proxy ClusterRole
	KubeProxyClusterRoleName = "system:node-proxier"
	// NodeBootstrapperClusterRoleName sets the name for the TLS Node Bootstrapper ClusterRole
	NodeBootstrapperClusterRoleName = "system:node-bootstrapper"

	// Constants
	clusterRoleKind    = "ClusterRole"
	serviceAccountKind = "ServiceAccount"
	rbacAPIGroup       = "rbac.authorization.k8s.io"
)

// TODO: Are there any unit tests that could be made for this file other than duplicating all values and logic in a separate file?

// CreateRBACRules creates the essential RBAC rules for a minimally set-up cluster
func CreateRBACRules(clientset *clientset.Clientset) error {
	// Create the ClusterRoles we need for our RBAC rules
	if err := CreateClusterRoles(clientset); err != nil {
		return err
	}
	// Create the CreateClusterRoleBindings we need for our RBAC rules
	if err := CreateClusterRoleBindings(clientset); err != nil {
		return err
	}

	fmt.Println("[apiconfig] Created RBAC rules")
	return nil
}

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

// CreateClusterRoles creates the ClusterRoles that aren't bootstrapped by the apiserver
func CreateClusterRoles(clientset *clientset.Clientset) error {
	// TODO: Remove this ClusterRole when it's automatically bootstrapped in the apiserver
	clusterRole := rbac.ClusterRole{
		ObjectMeta: metav1.ObjectMeta{Name: KubeDNSClusterRoleName},
		Rules: []rbac.PolicyRule{
			rbac.NewRule("list", "watch").Groups("").Resources("endpoints", "services").RuleOrDie(),
			// TODO: remove watch rule when https://github.com/kubernetes/kubernetes/pull/38816 gets merged
			rbac.NewRule("get", "list", "watch").Groups("").Resources("configmaps").RuleOrDie(),
		},
	}
	if _, err := clientset.Rbac().ClusterRoles().Create(&clusterRole); err != nil {
		return err
	}
	return nil
}

// CreateClusterRoleBindings creates all necessary bindings between bootstrapped & kubeadm-created ClusterRoles and subjects kubeadm is using
func CreateClusterRoleBindings(clientset *clientset.Clientset) error {
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
					Name: master.KubeletBootstrapGroup,
				},
			},
		},
		{
			ObjectMeta: metav1.ObjectMeta{
				Name: "kubeadm:kube-dns",
			},
			RoleRef: rbac.RoleRef{
				APIGroup: rbacAPIGroup,
				Kind:     clusterRoleKind,
				Name:     KubeDNSClusterRoleName,
			},
			Subjects: []rbac.Subject{
				{
					Kind:      serviceAccountKind,
					Name:      kubeadmconstants.KubeDNSServiceAccountName,
					Namespace: metav1.NamespaceSystem,
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
		if _, err := clientset.Rbac().ClusterRoleBindings().Create(&clusterRoleBinding); err != nil {
			return err
		}
	}
	return nil
}
