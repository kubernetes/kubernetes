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

package v1beta1

import (
	rbacv1beta1 "k8s.io/api/rbac/v1beta1"
	"k8s.io/apimachinery/pkg/runtime"
)

func addDefaultingFuncs(scheme *runtime.Scheme) error {
	return RegisterDefaults(scheme)
}

func SetDefaults_ClusterRoleBinding(obj *rbacv1beta1.ClusterRoleBinding) {
	if len(obj.RoleRef.APIGroup) == 0 {
		obj.RoleRef.APIGroup = GroupName
	}
}
func SetDefaults_RoleBinding(obj *rbacv1beta1.RoleBinding) {
	if len(obj.RoleRef.APIGroup) == 0 {
		obj.RoleRef.APIGroup = GroupName
	}
}
func SetDefaults_Subject(obj *rbacv1beta1.Subject) {
	if len(obj.APIGroup) == 0 {
		switch obj.Kind {
		case rbacv1beta1.ServiceAccountKind:
			obj.APIGroup = ""
		case rbacv1beta1.UserKind:
			obj.APIGroup = GroupName
		case rbacv1beta1.GroupKind:
			obj.APIGroup = GroupName
		}
	}
}
