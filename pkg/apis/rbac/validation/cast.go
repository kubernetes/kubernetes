/*
Copyright 2016 The Kubernetes Authors.

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

package validation

import "k8s.io/kubernetes/pkg/apis/rbac"

// Casting utilities to and from "Cluster" level equivalents.

func toClusterRole(in *rbac.Role) *rbac.ClusterRole {
	if in == nil {
		return nil
	}

	ret := &rbac.ClusterRole{}
	ret.ObjectMeta = in.ObjectMeta
	ret.Rules = in.Rules

	return ret
}

func toClusterRoleList(in *rbac.RoleList) *rbac.ClusterRoleList {
	ret := &rbac.ClusterRoleList{}
	for _, curr := range in.Items {
		ret.Items = append(ret.Items, *toClusterRole(&curr))
	}

	return ret
}

func toClusterRoleBinding(in *rbac.RoleBinding) *rbac.ClusterRoleBinding {
	if in == nil {
		return nil
	}

	ret := &rbac.ClusterRoleBinding{}
	ret.ObjectMeta = in.ObjectMeta
	ret.Subjects = in.Subjects
	ret.RoleRef = in.RoleRef

	return ret
}

func toClusterRoleBindingList(in *rbac.RoleBindingList) *rbac.ClusterRoleBindingList {
	ret := &rbac.ClusterRoleBindingList{}
	for _, curr := range in.Items {
		ret.Items = append(ret.Items, *toClusterRoleBinding(&curr))
	}

	return ret
}

func toRole(in *rbac.ClusterRole) *rbac.Role {
	if in == nil {
		return nil
	}

	ret := &rbac.Role{}
	ret.ObjectMeta = in.ObjectMeta
	ret.Rules = in.Rules

	return ret
}

func toRoleList(in *rbac.ClusterRoleList) *rbac.RoleList {
	ret := &rbac.RoleList{}
	for _, curr := range in.Items {
		ret.Items = append(ret.Items, *toRole(&curr))
	}

	return ret
}

func toRoleBinding(in *rbac.ClusterRoleBinding) *rbac.RoleBinding {
	if in == nil {
		return nil
	}

	ret := &rbac.RoleBinding{}
	ret.ObjectMeta = in.ObjectMeta
	ret.Subjects = in.Subjects
	ret.RoleRef = in.RoleRef

	return ret
}

func toRoleBindingList(in *rbac.ClusterRoleBindingList) *rbac.RoleBindingList {
	ret := &rbac.RoleBindingList{}
	for _, curr := range in.Items {
		ret.Items = append(ret.Items, *toRoleBinding(&curr))
	}

	return ret
}
