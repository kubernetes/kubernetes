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

package bootstrappolicy

import (
	rbac "k8s.io/kubernetes/pkg/apis/rbac"
)

// BootstrapPolicy describes the RBAC objects necessary to bootstrap the cluster.
type BootstrapPolicy struct {
	ClusterRoles           []rbac.ClusterRole
	ClusterRoleBindings    []rbac.ClusterRoleBinding
	ControllerRoles        []rbac.ClusterRole
	ControllerRoleBindings []rbac.ClusterRoleBinding
}

// GetBootstrapPolicy returns a BootstrapPolicy for the described cluster.
func GetPolicy(pspEnabled bool) *BootstrapPolicy {
	clusterRoles := ClusterRoles(pspEnabled)
	clusterRoleBindings := ClusterRoleBindings()
	controllerRoles, controllerRoleBindings := controllerPolicy(pspEnabled)
	return &BootstrapPolicy{
		ClusterRoles:           clusterRoles,
		ClusterRoleBindings:    clusterRoleBindings,
		ControllerRoles:        controllerRoles,
		ControllerRoleBindings: controllerRoleBindings,
	}
}
