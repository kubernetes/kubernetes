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
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/kubernetes/pkg/apis/rbac"
	"k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/typed/rbac/internalversion"
)

// +k8s:deepcopy-gen=true
// +k8s:deepcopy-gen:interfaces=k8s.io/kubernetes/pkg/registry/rbac/reconciliation.RuleOwner
// +k8s:deepcopy-gen:nonpointer-interfaces=true
type ClusterRoleRuleOwner struct {
	ClusterRole *rbac.ClusterRole
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

func (o ClusterRoleRuleOwner) GetRules() []rbac.PolicyRule {
	return o.ClusterRole.Rules
}

func (o ClusterRoleRuleOwner) SetRules(in []rbac.PolicyRule) {
	o.ClusterRole.Rules = in
}

type ClusterRoleModifier struct {
	Client internalversion.ClusterRoleInterface
}

func (c ClusterRoleModifier) Get(namespace, name string) (RuleOwner, error) {
	ret, err := c.Client.Get(name, metav1.GetOptions{})
	if err != nil {
		return nil, err
	}
	return ClusterRoleRuleOwner{ClusterRole: ret}, err
}

func (c ClusterRoleModifier) Create(in RuleOwner) (RuleOwner, error) {
	ret, err := c.Client.Create(in.(ClusterRoleRuleOwner).ClusterRole)
	if err != nil {
		return nil, err
	}
	return ClusterRoleRuleOwner{ClusterRole: ret}, err
}

func (c ClusterRoleModifier) Update(in RuleOwner) (RuleOwner, error) {
	ret, err := c.Client.Update(in.(ClusterRoleRuleOwner).ClusterRole)
	if err != nil {
		return nil, err
	}
	return ClusterRoleRuleOwner{ClusterRole: ret}, err

}
