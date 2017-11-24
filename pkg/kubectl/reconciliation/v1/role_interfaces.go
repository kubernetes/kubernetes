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

package v1

import (
	apiv1 "k8s.io/api/core/v1"
	rbacv1 "k8s.io/api/rbac/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	core "k8s.io/client-go/kubernetes/typed/core/v1"
	clientrbacv1 "k8s.io/client-go/kubernetes/typed/rbac/v1"
)

// RoleRuleOwner is a duplication in k8s.io/kubernetes/pkg/registry/rbac/reconciliation
// +k8s:deepcopy-gen=true
// +k8s:deepcopy-gen:interfaces=k8s.io/kubernetes/pkg/kubectl/reconciliation/v1.RuleOwner
// +k8s:deepcopy-gen:nonpointer-interfaces=true
type RoleRuleOwner struct {
	Role *rbacv1.Role
}

// GetObject is a duplication in k8s.io/kubernetes/pkg/registry/rbac/reconciliation
func (o RoleRuleOwner) GetObject() runtime.Object {
	return o.Role
}

// GetNamespace is a duplication in k8s.io/kubernetes/pkg/registry/rbac/reconciliation
func (o RoleRuleOwner) GetNamespace() string {
	return o.Role.Namespace
}

// GetName is a duplication in k8s.io/kubernetes/pkg/registry/rbac/reconciliation
func (o RoleRuleOwner) GetName() string {
	return o.Role.Name
}

// GetLabels is a duplication in k8s.io/kubernetes/pkg/registry/rbac/reconciliation
func (o RoleRuleOwner) GetLabels() map[string]string {
	return o.Role.Labels
}

// SetLabels is a duplication in k8s.io/kubernetes/pkg/registry/rbac/reconciliation
func (o RoleRuleOwner) SetLabels(in map[string]string) {
	o.Role.Labels = in
}

// GetAnnotations is a duplication in k8s.io/kubernetes/pkg/registry/rbac/reconciliation
func (o RoleRuleOwner) GetAnnotations() map[string]string {
	return o.Role.Annotations
}

// SetAnnotations is a duplication in k8s.io/kubernetes/pkg/registry/rbac/reconciliation
func (o RoleRuleOwner) SetAnnotations(in map[string]string) {
	o.Role.Annotations = in
}

// GetRules is a duplication in k8s.io/kubernetes/pkg/registry/rbac/reconciliation
func (o RoleRuleOwner) GetRules() []rbacv1.PolicyRule {
	return o.Role.Rules
}

// SetRules is a duplication in k8s.io/kubernetes/pkg/registry/rbac/reconciliation
func (o RoleRuleOwner) SetRules(in []rbacv1.PolicyRule) {
	o.Role.Rules = in
}

// GetAggregationRule is a duplication in k8s.io/kubernetes/pkg/registry/rbac/reconciliation
func (o RoleRuleOwner) GetAggregationRule() *rbacv1.AggregationRule {
	return nil
}

// SetAggregationRule is a duplication in k8s.io/kubernetes/pkg/registry/rbac/reconciliation
func (o RoleRuleOwner) SetAggregationRule(in *rbacv1.AggregationRule) {
}

// RoleModifier is a duplication in k8s.io/kubernetes/pkg/registry/rbac/reconciliation
type RoleModifier struct {
	Client          clientrbacv1.RolesGetter
	NamespaceClient core.NamespaceInterface
}

// Get is a duplication in k8s.io/kubernetes/pkg/registry/rbac/reconciliation
func (c RoleModifier) Get(namespace, name string) (RuleOwner, error) {
	ret, err := c.Client.Roles(namespace).Get(name, metav1.GetOptions{})
	if err != nil {
		return nil, err
	}
	return RoleRuleOwner{Role: ret}, err
}

// Create is a duplication in k8s.io/kubernetes/pkg/registry/rbac/reconciliation
func (c RoleModifier) Create(in RuleOwner) (RuleOwner, error) {
	ns := &apiv1.Namespace{ObjectMeta: metav1.ObjectMeta{Name: in.GetNamespace()}}
	if _, err := c.NamespaceClient.Create(ns); err != nil && !apierrors.IsAlreadyExists(err) {
		return nil, err
	}

	ret, err := c.Client.Roles(in.GetNamespace()).Create(in.(RoleRuleOwner).Role)
	if err != nil {
		return nil, err
	}
	return RoleRuleOwner{Role: ret}, err
}

// Update is a duplication in k8s.io/kubernetes/pkg/registry/rbac/reconciliation
func (c RoleModifier) Update(in RuleOwner) (RuleOwner, error) {
	ret, err := c.Client.Roles(in.GetNamespace()).Update(in.(RoleRuleOwner).Role)
	if err != nil {
		return nil, err
	}
	return RoleRuleOwner{Role: ret}, err

}
