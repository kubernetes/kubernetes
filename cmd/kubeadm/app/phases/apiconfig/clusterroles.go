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

	"k8s.io/api/core/v1"
	rbac "k8s.io/api/rbac/v1beta1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	clientset "k8s.io/client-go/kubernetes"
	kubeadmconstants "k8s.io/kubernetes/cmd/kubeadm/app/constants"
	apiclientutil "k8s.io/kubernetes/cmd/kubeadm/app/util/apiclient"
	"k8s.io/kubernetes/pkg/util/version"
)

const (
	// KubeProxyClusterRoleName sets the name for the kube-proxy ClusterRole
	KubeProxyClusterRoleName = "system:node-proxier"
)

// CreateServiceAccounts creates the necessary serviceaccounts that kubeadm uses/might use, if they don't already exist.
func CreateServiceAccounts(client clientset.Interface) error {
	// TODO: Each ServiceAccount should be created per-addon (decentralized) vs here
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
		if _, err := client.CoreV1().ServiceAccounts(metav1.NamespaceSystem).Create(&sa); err != nil {
			if !apierrors.IsAlreadyExists(err) {
				return err
			}
		}
	}
	return nil
}

// CreateRBACRules creates the essential RBAC rules for a minimally set-up cluster
func CreateRBACRules(client clientset.Interface, k8sVersion *version.Version) error {
	if err := createClusterRoleBindings(client); err != nil {
		return err
	}
	if err := deletePermissiveNodesBindingWhenUsingNodeAuthorization(client, k8sVersion); err != nil {
		return fmt.Errorf("failed to remove the permissive 'system:nodes' Group Subject in the 'system:node' ClusterRoleBinding: %v", err)
	}

	fmt.Println("[apiconfig] Created RBAC rules")
	return nil
}

func createClusterRoleBindings(client clientset.Interface) error {
	// TODO: This ClusterRoleBinding should be created by the kube-proxy phase, not here
	return apiclientutil.CreateClusterRoleBindingIfNotExists(client, &rbac.ClusterRoleBinding{
		ObjectMeta: metav1.ObjectMeta{
			Name: "kubeadm:node-proxier",
		},
		RoleRef: rbac.RoleRef{
			APIGroup: rbac.GroupName,
			Kind:     "ClusterRole",
			Name:     KubeProxyClusterRoleName,
		},
		Subjects: []rbac.Subject{
			{
				Kind:      rbac.ServiceAccountKind,
				Name:      kubeadmconstants.KubeProxyServiceAccountName,
				Namespace: metav1.NamespaceSystem,
			},
		},
	})
}

func deletePermissiveNodesBindingWhenUsingNodeAuthorization(client clientset.Interface, k8sVersion *version.Version) error {

	// TODO: When the v1.9 cycle starts (targeting v1.9 at HEAD) and v1.8.0 is the minimum supported version, we can remove this function as the ClusterRoleBinding won't exist
	// or already have no such permissive subject
	nodesRoleBinding, err := client.RbacV1beta1().ClusterRoleBindings().Get(kubeadmconstants.NodesClusterRoleBinding, metav1.GetOptions{})
	if err != nil {
		if apierrors.IsNotFound(err) {
			// Nothing to do; the RoleBinding doesn't exist
			return nil
		}
		return err
	}

	newSubjects := []rbac.Subject{}
	for _, subject := range nodesRoleBinding.Subjects {
		// Skip the subject that binds to the system:nodes group
		if subject.Name == kubeadmconstants.NodesGroup && subject.Kind == "Group" {
			continue
		}
		newSubjects = append(newSubjects, subject)
	}

	nodesRoleBinding.Subjects = newSubjects

	if _, err := client.RbacV1beta1().ClusterRoleBindings().Update(nodesRoleBinding); err != nil {
		return err
	}

	return nil
}
