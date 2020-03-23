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
	"context"
	rbacv1 "k8s.io/api/rbac/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	rbacv1client "k8s.io/client-go/kubernetes/typed/rbac/v1"
)

// +k8s:deepcopy-gen=true
// +k8s:deepcopy-gen:interfaces=k8s.io/kubernetes/pkg/registry/rbac/reconciliation.RuleOwner
// +k8s:deepcopy-gen:nonpointer-interfaces=true
type ClusterRoleRuleOwner struct {
	ClusterRole *rbacv1.ClusterRole
}

func (o ClusterRoleRuleOwner) GetObject() runtime.Object {
	return o.ClusterRole
}

func (o ClusterRoleRuleOwner) GetNamespace() string {
	return o.ClusterRole.Namespace
}

func (o ClusterRoleRuleOwner) GetName() string {
	return o.ClusterRole.Name
}

func (o ClusterRoleRuleOwner) GetLabels() map[string]string {
	return o.ClusterRole.Labels
}

func (o ClusterRoleRuleOwner) SetLabels(in map[string]string) {
	o.ClusterRole.Labels = in
}

func (o ClusterRoleRuleOwner) GetAnnotations() map[string]string {
	return o.ClusterRole.Annotations
}

func (o ClusterRoleRuleOwner) SetAnnotations(in map[string]string) {
	o.ClusterRole.Annotations = in
}

func (o ClusterRoleRuleOwner) GetRules() []rbacv1.PolicyRule {
	return o.ClusterRole.Rules
}

func (o ClusterRoleRuleOwner) SetRules(in []rbacv1.PolicyRule) {
	o.ClusterRole.Rules = in
}

func (o ClusterRoleRuleOwner) GetAggregationRule() *rbacv1.AggregationRule {
	return o.ClusterRole.AggregationRule
}

func (o ClusterRoleRuleOwner) SetAggregationRule(in *rbacv1.AggregationRule) {
	o.ClusterRole.AggregationRule = in
}

type ClusterRoleModifier struct {
	Client rbacv1client.ClusterRoleInterface
}

func (c ClusterRoleModifier) Get(namespace, name string) (RuleOwner, error) {
	ret, err := c.Client.Get(context.TODO(), name, metav1.GetOptions{})
	if err != nil {
		return nil, err
	}
	return ClusterRoleRuleOwner{ClusterRole: ret}, err
}

func (c ClusterRoleModifier) Create(in RuleOwner) (RuleOwner, error) {
	ret, err := c.Client.Create(context.TODO(), in.(ClusterRoleRuleOwner).ClusterRole, metav1.CreateOptions{})
	if err != nil {
		return nil, err
	}
	return ClusterRoleRuleOwner{ClusterRole: ret}, err
}

func (c ClusterRoleModifier) Update(in RuleOwner) (RuleOwner, error) {
	ret, err := c.Client.Update(context.TODO(), in.(ClusterRoleRuleOwner).ClusterRole, metav1.UpdateOptions{})
	if err != nil {
		return nil, err
	}
	return ClusterRoleRuleOwner{ClusterRole: ret}, err

}
