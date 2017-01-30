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
	"k8s.io/kubernetes/cmd/kubeadm/app/master"
	"k8s.io/kubernetes/pkg/api"
	rbac "k8s.io/kubernetes/pkg/apis/rbac/v1beta1"
	"k8s.io/kubernetes/pkg/client/clientset_generated/clientset"
)

// CreateBootstrapRBACClusterRole creates the necessary ClusterRole for bootstrapping
func CreateBootstrapRBACClusterRole(clientset *clientset.Clientset) error {
	clusterRole := rbac.ClusterRole{
		ObjectMeta: metav1.ObjectMeta{Name: "kubeadm:kubelet-bootstrap"},
		Rules: []rbac.PolicyRule{
			rbac.NewRule("get").Groups("").Resources("nodes").RuleOrDie(),
			rbac.NewRule("create", "watch").Groups("certificates.k8s.io").Resources("certificatesigningrequests").RuleOrDie(),
		},
	}
	if _, err := clientset.Rbac().ClusterRoles().Create(&clusterRole); err != nil {
		return err
	}

	subject := rbac.Subject{
		Kind: "Group",
		Name: "kubeadm:kubelet-bootstrap",
	}

	clusterRoleBinding := rbac.ClusterRoleBinding{
		ObjectMeta: metav1.ObjectMeta{
			Name: "kubeadm:kubelet-bootstrap",
		},
		RoleRef: rbac.RoleRef{
			APIGroup: "rbac.authorization.k8s.io",
			Kind:     "ClusterRole",
			Name:     "kubeadm:kubelet-bootstrap",
		},
		Subjects: []rbac.Subject{subject},
	}
	if _, err := clientset.Rbac().ClusterRoleBindings().Create(&clusterRoleBinding); err != nil {
		return err
	}
	fmt.Println("[apiconfig] Created kubelet-bootstrap RBAC rules")

	return nil
}

// CreateKubeDNSRBACClusterRole creates the necessary ClusterRole for kube-dns
func CreateKubeDNSRBACClusterRole(clientset *clientset.Clientset) error {
	clusterRole := rbac.ClusterRole{
		ObjectMeta: metav1.ObjectMeta{Name: "kubeadm:" + master.KubeDNS},
		Rules: []rbac.PolicyRule{
			rbac.NewRule("list", "watch").Groups("").Resources("endpoints", "services").RuleOrDie(),
			// TODO: remove watch rule when https://github.com/kubernetes/kubernetes/pull/38816 gets merged
			rbac.NewRule("get", "list", "watch").Groups("").Resources("configmaps").RuleOrDie(),
		},
	}
	if _, err := clientset.Rbac().ClusterRoles().Create(&clusterRole); err != nil {
		return err
	}

	subject := rbac.Subject{
		Kind:      "ServiceAccount",
		Name:      master.KubeDNS,
		Namespace: api.NamespaceSystem,
	}

	clusterRoleBinding := rbac.ClusterRoleBinding{
		ObjectMeta: metav1.ObjectMeta{
			Name: "kubeadm:" + master.KubeDNS,
		},
		RoleRef: rbac.RoleRef{
			APIGroup: "rbac.authorization.k8s.io",
			Kind:     "ClusterRole",
			Name:     "kubeadm:" + master.KubeDNS,
		},
		Subjects: []rbac.Subject{subject},
	}
	if _, err := clientset.Rbac().ClusterRoleBindings().Create(&clusterRoleBinding); err != nil {
		return err
	}
	fmt.Println("[apiconfig] Created kube-dns RBAC rules")

	return nil
}

// CreateKubeProxyClusterRoleBinding creates the necessary ClusterRole for kube-dns
func CreateKubeProxyClusterRoleBinding(clientset *clientset.Clientset) error {
	systemKubeProxySubject := rbac.Subject{
		Kind:      "User",
		Name:      "system:kube-proxy",
		Namespace: api.NamespaceSystem,
	}

	systemNodesSubject := rbac.Subject{
		Kind:      "Group",
		Name:      "system:nodes",
		Namespace: api.NamespaceSystem,
	}

	clusterRoleBinding := rbac.ClusterRoleBinding{
		ObjectMeta: metav1.ObjectMeta{
			Name: "system:node-proxier",
		},
		RoleRef: rbac.RoleRef{
			APIGroup: "rbac.authorization.k8s.io",
			Kind:     "ClusterRole",
			Name:     "system:node-proxier",
		},
		Subjects: []rbac.Subject{systemKubeProxySubject, systemNodesSubject},
	}
	if _, err := clientset.Rbac().ClusterRoleBindings().Update(&clusterRoleBinding); err != nil {
		return err
	}
	fmt.Println("[apiconfig] Created kube-proxy RBAC rules")

	return nil
}
