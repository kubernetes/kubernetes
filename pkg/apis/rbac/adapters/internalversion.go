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
	"k8s.io/kubernetes/pkg/apis/rbac"
)

type internalversionPolicyRules struct {
	obj []rbac.PolicyRule
}

func (o *internalversionPolicyRules) Len() int {
	return len(o.obj)
}
func (o *internalversionPolicyRules) Get(i int) PolicyRule {
	return &internalversionPolicyRule{obj: &o.obj[i]}
}

type internalversionPolicyRule struct {
	obj *rbac.PolicyRule
}

func (o *internalversionPolicyRule) Verbs() []string {
	return o.obj.Verbs
}
func (o *internalversionPolicyRule) APIGroups() []string {
	return o.obj.APIGroups
}
func (o *internalversionPolicyRule) Resources() []string {
	return o.obj.Resources
}
func (o *internalversionPolicyRule) ResourceNames() []string {
	return o.obj.ResourceNames
}
func (o *internalversionPolicyRule) NonResourceURLs() []string {
	return o.obj.NonResourceURLs
}

type internalversionSubjects struct {
	obj []rbac.Subject
}

func (o *internalversionSubjects) Len() int {
	return len(o.obj)
}
func (o *internalversionSubjects) Get(i int) Subject {
	return &internalversionSubject{obj: &o.obj[i]}
}

type internalversionSubject struct {
	obj *rbac.Subject
}

func (o *internalversionSubject) Kind() string {
	return o.obj.Kind
}
func (o *internalversionSubject) APIVersion() string {
	return o.obj.APIVersion
}
func (o *internalversionSubject) Name() string {
	return o.obj.Name
}
func (o *internalversionSubject) Namespace() string {
	return o.obj.Namespace
}

type internalversionRoleRef struct {
	obj *rbac.RoleRef
}

func (o *internalversionRoleRef) APIGroup() string {
	return o.obj.APIGroup
}
func (o *internalversionRoleRef) Kind() string {
	return o.obj.Kind
}
func (o *internalversionRoleRef) Name() string {
	return o.obj.Name
}

type internalversionRole struct {
	obj *rbac.Role
}

func (o *internalversionRole) Rules() PolicyRules {
	return &internalversionPolicyRules{obj: o.obj.Rules}
}

type internalversionRoleBinding struct {
	obj *rbac.RoleBinding
}

func (o *internalversionRoleBinding) Subjects() Subjects {
	return &internalversionSubjects{obj: o.obj.Subjects}
}
func (o *internalversionRoleBinding) RoleRef() RoleRef {
	return &internalversionRoleRef{obj: &o.obj.RoleRef}
}

type internalversionClusterRole struct {
	obj *rbac.ClusterRole
}

func (o *internalversionClusterRole) Rules() PolicyRules {
	return &internalversionPolicyRules{obj: o.obj.Rules}
}

type internalversionClusterRoleBinding struct {
	obj *rbac.ClusterRoleBinding
}

func (o *internalversionClusterRoleBinding) Subjects() Subjects {
	return &internalversionSubjects{obj: o.obj.Subjects}
}
func (o *internalversionClusterRoleBinding) RoleRef() RoleRef {
	return &internalversionRoleRef{obj: &o.obj.RoleRef}
}
