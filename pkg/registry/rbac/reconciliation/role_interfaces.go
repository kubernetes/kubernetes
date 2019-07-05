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

package reconciliation

import (
	rbacv1 "k8s.io/api/rbac/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	corev1client "k8s.io/client-go/kubernetes/typed/core/v1"
	rbacv1client "k8s.io/client-go/kubernetes/typed/rbac/v1"
)

// +k8s:deepcopy-gen=true
// +k8s:deepcopy-gen:interfaces=k8s.io/kubernetes/pkg/registry/rbac/reconciliation.RuleOwner
// +k8s:deepcopy-gen:nonpointer-interfaces=true
type RoleRuleOwner struct {
	Role *rbacv1.Role
}

func (o RoleRuleOwner) GetObject() runtime.Object {
	return o.Role
}

func (o RoleRuleOwner) GetNamespace() string {
	return o.Role.Namespace
}

func (o RoleRuleOwner) GetName() string {
	return o.Role.Name
}

func (o RoleRuleOwner) GetLabels() map[string]string {
	return o.Role.Labels
}

func (o RoleRuleOwner) SetLabels(in map[string]string) {
	o.Role.Labels = in
}

func (o RoleRuleOwner) GetAnnotations() map[string]string {
	return o.Role.Annotations
}

func (o RoleRuleOwner) SetAnnotations(in map[string]string) {
	o.Role.Annotations = in
}

func (o RoleRuleOwner) GetRules() []rbacv1.PolicyRule {
	return o.Role.Rules
}

func (o RoleRuleOwner) SetRules(in []rbacv1.PolicyRule) {
	o.Role.Rules = in
}

func (o RoleRuleOwner) GetAggregationRule() *rbacv1.AggregationRule {
	return nil
}

func (o RoleRuleOwner) SetAggregationRule(in *rbacv1.AggregationRule) {
}

type RoleModifier struct {
	Client          rbacv1client.RolesGetter
	NamespaceClient corev1client.NamespaceInterface
}

func (c RoleModifier) Get(namespace, name string) (RuleOwner, error) {
	ret, err := c.Client.Roles(namespace).Get(name, metav1.GetOptions{})
	if err != nil {
		return nil, err
	}
	return RoleRuleOwner{Role: ret}, err
}

func (c RoleModifier) Create(in RuleOwner) (RuleOwner, error) {
	if err := tryEnsureNamespace(c.NamespaceClient, in.GetNamespace()); err != nil {
		return nil, err
	}

	ret, err := c.Client.Roles(in.GetNamespace()).Create(in.(RoleRuleOwner).Role)
	if err != nil {
		return nil, err
	}
	return RoleRuleOwner{Role: ret}, err
}

func (c RoleModifier) Update(in RuleOwner) (RuleOwner, error) {
	ret, err := c.Client.Roles(in.GetNamespace()).Update(in.(RoleRuleOwner).Role)
	if err != nil {
		return nil, err
	}
	return RoleRuleOwner{Role: ret}, err

}
