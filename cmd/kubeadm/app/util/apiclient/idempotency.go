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

package util

import (
	"fmt"

	"k8s.io/api/core/v1"
	rbac "k8s.io/api/rbac/v1beta1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	clientset "k8s.io/client-go/kubernetes"
)

// TODO: We should invent a dynamic mechanism for this using the dynamic client instead of hard-coding these functions per-type

// CreateClusterRoleIfNotExists creates a ClusterRole if the target resource doesn't exist. If the resource exists already, this function will update the resource instead.
func CreateClusterRoleIfNotExists(client clientset.Interface, clusterRole *rbac.ClusterRole) error {
	if _, err := client.RbacV1beta1().ClusterRoles().Create(clusterRole); err != nil {
		if !apierrors.IsAlreadyExists(err) {
			return fmt.Errorf("unable to create RBAC clusterrole: %v", err)
		}

		if _, err := client.RbacV1beta1().ClusterRoles().Update(clusterRole); err != nil {
			return fmt.Errorf("unable to update RBAC clusterrole: %v", err)
		}
	}
	return nil
}

// CreateClusterRoleBindingIfNotExists creates a ClusterRoleBinding if the target resource doesn't exist. If the resource exists already, this function will update the resource instead.
func CreateClusterRoleBindingIfNotExists(client clientset.Interface, clusterRoleBinding *rbac.ClusterRoleBinding) error {
	if _, err := client.RbacV1beta1().ClusterRoleBindings().Create(clusterRoleBinding); err != nil {
		if !apierrors.IsAlreadyExists(err) {
			return fmt.Errorf("unable to create RBAC clusterrolebinding: %v", err)
		}

		if _, err := client.RbacV1beta1().ClusterRoleBindings().Update(clusterRoleBinding); err != nil {
			return fmt.Errorf("unable to update RBAC clusterrolebinding: %v", err)
		}
	}
	return nil
}

// CreateRoleIfNotExists creates a Role if the target resource doesn't exist. If the resource exists already, this function will update the resource instead.
func CreateRoleIfNotExists(client clientset.Interface, role *rbac.Role) error {
	if _, err := client.RbacV1beta1().Roles(role.ObjectMeta.Namespace).Create(role); err != nil {
		if !apierrors.IsAlreadyExists(err) {
			return fmt.Errorf("unable to create RBAC role: %v", err)
		}

		if _, err := client.RbacV1beta1().Roles(role.ObjectMeta.Namespace).Update(role); err != nil {
			return fmt.Errorf("unable to update RBAC role: %v", err)
		}
	}
	return nil
}

// CreateRoleBindingIfNotExists creates a RoleBinding if the target resource doesn't exist. If the resource exists already, this function will update the resource instead.
func CreateRoleBindingIfNotExists(client clientset.Interface, roleBinding *rbac.RoleBinding) error {
	if _, err := client.RbacV1beta1().RoleBindings(roleBinding.ObjectMeta.Namespace).Create(roleBinding); err != nil {
		if !apierrors.IsAlreadyExists(err) {
			return fmt.Errorf("unable to create RBAC rolebinding: %v", err)
		}

		if _, err := client.RbacV1beta1().RoleBindings(roleBinding.ObjectMeta.Namespace).Update(roleBinding); err != nil {
			return fmt.Errorf("unable to update RBAC rolebinding: %v", err)
		}
	}
	return nil
}

// CreateConfigMapIfNotExists creates a ConfigMap if the target resource doesn't exist. If the resource exists already, this function will update the resource instead.
func CreateConfigMapIfNotExists(client clientset.Interface, cm *v1.ConfigMap) error {
	if _, err := client.CoreV1().ConfigMaps(cm.ObjectMeta.Namespace).Create(cm); err != nil {
		if !apierrors.IsAlreadyExists(err) {
			return fmt.Errorf("unable to create configmap: %v", err)
		}

		if _, err := client.CoreV1().ConfigMaps(cm.ObjectMeta.Namespace).Update(cm); err != nil {
			return fmt.Errorf("unable to update configmap: %v", err)
		}
	}
	return nil
}
