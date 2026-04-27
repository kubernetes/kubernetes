/*
Copyright 2018 The Kubernetes Authors.

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

import (
	"context"

	rbacv1 "k8s.io/api/rbac/v1"
	"k8s.io/kubernetes/pkg/apis/rbac"
	rbacv1helpers "k8s.io/kubernetes/pkg/apis/rbac/v1"
)

func ConfirmNoEscalationInternal(ctx context.Context, ruleResolver AuthorizationRuleResolver, inRules []rbac.PolicyRule) error {
	rules := []rbacv1.PolicyRule{}
	for i := range inRules {
		v1Rule := rbacv1.PolicyRule{}
		err := rbacv1helpers.Convert_rbac_PolicyRule_To_v1_PolicyRule(&inRules[i], &v1Rule, nil)
		if err != nil {
			return err
		}
		rules = append(rules, v1Rule)
	}

	return ConfirmNoEscalation(ctx, ruleResolver, rules)
}
