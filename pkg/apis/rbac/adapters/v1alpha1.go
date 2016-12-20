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
	rbac "k8s.io/kubernetes/pkg/apis/rbac/v1alpha1"
)

type v1alpha1PolicyRules struct {
	obj []rbac.PolicyRule
}

func (o *v1alpha1PolicyRules) Len() int {
	return len(o.obj)
}
func (o *v1alpha1PolicyRules) Get(i int) PolicyRule {
	return &v1alpha1PolicyRule{obj: &o.obj[i]}
}

type v1alpha1PolicyRule struct {
	obj *rbac.PolicyRule
}

func (o *v1alpha1PolicyRule) Verbs() []string {
	return o.obj.Verbs
}
func (o *v1alpha1PolicyRule) APIGroups() []string {
	return o.obj.APIGroups
}
func (o *v1alpha1PolicyRule) Resources() []string {
	return o.obj.Resources
}
func (o *v1alpha1PolicyRule) ResourceNames() []string {
	return o.obj.ResourceNames
}
func (o *v1alpha1PolicyRule) NonResourceURLs() []string {
	return o.obj.NonResourceURLs
}

type v1alpha1Subjects struct {
	obj []rbac.Subject
}

func (o *v1alpha1Subjects) Len() int {
	return len(o.obj)
}
func (o *v1alpha1Subjects) Get(i int) Subject {
	return &v1alpha1Subject{obj: &o.obj[i]}
}

type v1alpha1Subject struct {
	obj *rbac.Subject
}

func (o *v1alpha1Subject) Kind() string {
	return o.obj.Kind
}
func (o *v1alpha1Subject) APIVersion() string {
	return o.obj.APIVersion
}
func (o *v1alpha1Subject) Name() string {
	return o.obj.Name
}
func (o *v1alpha1Subject) Namespace() string {
	return o.obj.Namespace
}

type v1alpha1RoleRef struct {
	obj *rbac.RoleRef
}

func (o *v1alpha1RoleRef) APIGroup() string {
	return o.obj.APIGroup
}
func (o *v1alpha1RoleRef) Kind() string {
	return o.obj.Kind
}
func (o *v1alpha1RoleRef) Name() string {
	return o.obj.Name
}

type v1alpha1Role struct {
	obj *rbac.Role
}

func (o *v1alpha1Role) Rules() PolicyRules {
	return &v1alpha1PolicyRules{obj: o.obj.Rules}
}

type v1alpha1RoleBinding struct {
	obj *rbac.RoleBinding
}

func (o *v1alpha1RoleBinding) Subjects() Subjects {
	return &v1alpha1Subjects{obj: o.obj.Subjects}
}
func (o *v1alpha1RoleBinding) RoleRef() RoleRef {
	return &v1alpha1RoleRef{obj: &o.obj.RoleRef}
}

type v1alpha1ClusterRole struct {
	obj *rbac.ClusterRole
}

func (o *v1alpha1ClusterRole) Rules() PolicyRules {
	return &v1alpha1PolicyRules{obj: o.obj.Rules}
}

type v1alpha1ClusterRoleBinding struct {
	obj *rbac.ClusterRoleBinding
}

func (o *v1alpha1ClusterRoleBinding) Subjects() Subjects {
	return &v1alpha1Subjects{obj: o.obj.Subjects}
}
func (o *v1alpha1ClusterRoleBinding) RoleRef() RoleRef {
	return &v1alpha1RoleRef{obj: &o.obj.RoleRef}
}
