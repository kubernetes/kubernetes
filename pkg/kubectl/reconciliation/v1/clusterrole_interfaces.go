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
	rbacv1 "k8s.io/api/rbac/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	clientrbacv1 "k8s.io/client-go/kubernetes/typed/rbac/v1"
)

// ClusterRoleRuleOwner is a duplication in k8s.io/kubernetes/pkg/registry/rbac/reconciliation
// +k8s:deepcopy-gen=true
// +k8s:deepcopy-gen:interfaces=k8s.io/kubernetes/pkg/kubectl/reconciliation/v1.RuleOwner
// +k8s:deepcopy-gen:nonpointer-interfaces=true
type ClusterRoleRuleOwner struct {
	ClusterRole *rbacv1.ClusterRole
}

// GetObject is a duplication in k8s.io/kubernetes/pkg/registry/rbac/reconciliation
func (o ClusterRoleRuleOwner) GetObject() runtime.Object {
	return o.ClusterRole
}

// GetNamespace is a duplication in k8s.io/kubernetes/pkg/registry/rbac/reconciliation
func (o ClusterRoleRuleOwner) GetNamespace() string {
	return o.ClusterRole.Namespace
}

// GetName is a duplication in k8s.io/kubernetes/pkg/registry/rbac/reconciliation
func (o ClusterRoleRuleOwner) GetName() string {
	return o.ClusterRole.Name
}

// GetLabels is a duplication in k8s.io/kubernetes/pkg/registry/rbac/reconciliation
func (o ClusterRoleRuleOwner) GetLabels() map[string]string {
	return o.ClusterRole.Labels
}

// SetLabels is a duplication in k8s.io/kubernetes/pkg/registry/rbac/reconciliation
func (o ClusterRoleRuleOwner) SetLabels(in map[string]string) {
	o.ClusterRole.Labels = in
}

// GetAnnotations is a duplication in k8s.io/kubernetes/pkg/registry/rbac/reconciliation
func (o ClusterRoleRuleOwner) GetAnnotations() map[string]string {
	return o.ClusterRole.Annotations
}

// SetAnnotations is a duplication in k8s.io/kubernetes/pkg/registry/rbac/reconciliation
func (o ClusterRoleRuleOwner) SetAnnotations(in map[string]string) {
	o.ClusterRole.Annotations = in
}

// GetRules is a duplication in k8s.io/kubernetes/pkg/registry/rbac/reconciliation
func (o ClusterRoleRuleOwner) GetRules() []rbacv1.PolicyRule {
	return o.ClusterRole.Rules
}

// SetRules is a duplication in k8s.io/kubernetes/pkg/registry/rbac/reconciliation
func (o ClusterRoleRuleOwner) SetRules(in []rbacv1.PolicyRule) {
	o.ClusterRole.Rules = in
}

// GetAggregationRule is a duplication in k8s.io/kubernetes/pkg/registry/rbac/reconciliation
func (o ClusterRoleRuleOwner) GetAggregationRule() *rbacv1.AggregationRule {
	return o.ClusterRole.AggregationRule
}

// SetAggregationRule is a duplication in k8s.io/kubernetes/pkg/registry/rbac/reconciliation
func (o ClusterRoleRuleOwner) SetAggregationRule(in *rbacv1.AggregationRule) {
	o.ClusterRole.AggregationRule = in
}

// ClusterRoleModifier is a duplication in k8s.io/kubernetes/pkg/registry/rbac/reconciliation
type ClusterRoleModifier struct {
	Client clientrbacv1.ClusterRoleInterface
}

// Get is a duplication in k8s.io/kubernetes/pkg/registry/rbac/reconciliation
func (c ClusterRoleModifier) Get(namespace, name string) (RuleOwner, error) {
	ret, err := c.Client.Get(name, metav1.GetOptions{})
	if err != nil {
		return nil, err
	}
	return ClusterRoleRuleOwner{ClusterRole: ret}, err
}

// Create is a duplication in k8s.io/kubernetes/pkg/registry/rbac/reconciliation
func (c ClusterRoleModifier) Create(in RuleOwner) (RuleOwner, error) {
	ret, err := c.Client.Create(in.(ClusterRoleRuleOwner).ClusterRole)
	if err != nil {
		return nil, err
	}
	return ClusterRoleRuleOwner{ClusterRole: ret}, err
}

// Update is a duplication in k8s.io/kubernetes/pkg/registry/rbac/reconciliation
func (c ClusterRoleModifier) Update(in RuleOwner) (RuleOwner, error) {
	ret, err := c.Client.Update(in.(ClusterRoleRuleOwner).ClusterRole)
	if err != nil {
		return nil, err
	}
	return ClusterRoleRuleOwner{ClusterRole: ret}, err

}
