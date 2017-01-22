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
	rbac "k8s.io/kubernetes/pkg/apis/rbac/v1beta1"
	"k8s.io/kubernetes/pkg/client/clientset_generated/clientset"
)

// CreateBootstrapRBACClusterRole grants the system:node-bootstrapper role to the group we created the bootstrap credential with
func CreateBootstrapRBACClusterRole(clientset *clientset.Clientset) error {
	clusterRoleBinding := rbac.ClusterRoleBinding{
		ObjectMeta: metav1.ObjectMeta{
			Name: "kubeadm:kubelet-bootstrap",
		},
		RoleRef: rbac.RoleRef{
			APIGroup: "rbac.authorization.k8s.io",
			Kind:     "ClusterRole",
			Name:     "system:node-bootstrapper",
		},
		Subjects: []rbac.Subject{
			{Kind: "Group", Name: master.KubeletBootstrapGroup},
		},
	}
	if _, err := clientset.Rbac().ClusterRoleBindings().Create(&clusterRoleBinding); err != nil {
		return err
	}
	fmt.Println("[apiconfig] Created node bootstrapper RBAC rules")

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
		Namespace: metav1.NamespaceSystem,
	}

	clusterRoleBinding := rbac.ClusterRoleBinding{
		ObjectMeta: metav1.ObjectMeta{
			Name: "kubeadm:" + master.KubeDNS,
		},
		RoleRef: rbac.RoleRef{
			APIGroup: "rbac.authorization.k8s.io",
			Kind:     "ClusterRole",
			Name:     clusterRole.Name,
		},
		Subjects: []rbac.Subject{subject},
	}
	if _, err := clientset.Rbac().ClusterRoleBindings().Create(&clusterRoleBinding); err != nil {
		return err
	}
	fmt.Println("[apiconfig] Created kube-dns RBAC rules")

	return nil
}

// CreateKubeProxyClusterRoleBinding grants the system:node-proxier role to the nodes group,
// since kubelet credentials are used to run the kube-proxy
// TODO: give the kube-proxy its own credential and stop requiring this
func CreateKubeProxyClusterRoleBinding(clientset *clientset.Clientset) error {
	clusterRoleBinding := rbac.ClusterRoleBinding{
		ObjectMeta: metav1.ObjectMeta{
			Name: "kubeadm:node-proxier",
		},
		RoleRef: rbac.RoleRef{
			APIGroup: "rbac.authorization.k8s.io",
			Kind:     "ClusterRole",
			Name:     "system:node-proxier",
		},
		Subjects: []rbac.Subject{
			{Kind: "Group", Name: "system:nodes"},
		},
	}
	if _, err := clientset.Rbac().ClusterRoleBindings().Create(&clusterRoleBinding); err != nil {
		return err
	}
	fmt.Println("[apiconfig] Created kube-proxy RBAC rules")

	return nil
}
