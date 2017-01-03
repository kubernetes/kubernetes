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

package adapters

import (
	"fmt"

	"k8s.io/kubernetes/pkg/apis/rbac"
	rbacv1alpha1 "k8s.io/kubernetes/pkg/apis/rbac/v1alpha1"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/runtime/schema"
)

// shaping the type like this prevents allocations
type PolicyRules interface {
	Len() int
	Get(i int) PolicyRule
}

type PolicyRule interface {
	Verbs() []string
	APIGroups() []string
	Resources() []string
	ResourceNames() []string
	NonResourceURLs() []string
}

// shaping the type like this prevents allocations
type Subjects interface {
	Len() int
	Get(i int) Subject
}

type Subject interface {
	Kind() string
	APIVersion() string
	Name() string
	Namespace() string
}

type RoleRef interface {
	APIGroup() string
	Kind() string
	Name() string
}

type Role interface {
	Rules() PolicyRules
}

type RoleBinding interface {
	Subjects() Subjects
	RoleRef() RoleRef
}

type ClusterRole interface {
	Rules() PolicyRules
}

type ClusterRoleBinding interface {
	Subjects() Subjects
	RoleRef() RoleRef
}

func ToClusterRole(obj runtime.Object) ClusterRole {
	switch castObj := obj.(type) {
	case *rbac.ClusterRole:
		return &internalversionClusterRole{obj: castObj}
	case *rbacv1alpha1.ClusterRole:
		return &v1alpha1ClusterRole{obj: castObj}
	}
	panic(fmt.Sprintf("uncastable type: %T", obj))
}

func ToClusterRoleBinding(obj runtime.Object) ClusterRoleBinding {
	switch castObj := obj.(type) {
	case *rbac.ClusterRoleBinding:
		return &internalversionClusterRoleBinding{obj: castObj}
	case *rbacv1alpha1.ClusterRoleBinding:
		return &v1alpha1ClusterRoleBinding{obj: castObj}
	}
	panic(fmt.Sprintf("uncastable type: %T", obj))
}

func ToRole(obj runtime.Object) Role {
	switch castObj := obj.(type) {
	case *rbac.Role:
		return &internalversionRole{obj: castObj}
	case *rbacv1alpha1.Role:
		return &v1alpha1Role{obj: castObj}
	}
	panic(fmt.Sprintf("uncastable type: %T", obj))
}

func ToRoleBinding(obj runtime.Object) RoleBinding {
	switch castObj := obj.(type) {
	case *rbac.RoleBinding:
		return &internalversionRoleBinding{obj: castObj}
	case *rbacv1alpha1.RoleBinding:
		return &v1alpha1RoleBinding{obj: castObj}
	}
	panic(fmt.Sprintf("uncastable type: %T", obj))
}

func ToPolicyRule(obj *rbac.PolicyRule) PolicyRule {
	return &internalversionPolicyRule{obj: obj}
}

func ToSubject(obj *rbac.Subject) Subject {
	return &internalversionSubject{obj: obj}
}

func RoleRefGroupKind(roleRef RoleRef) schema.GroupKind {
	return schema.GroupKind{Group: roleRef.APIGroup(), Kind: roleRef.Kind()}
}
