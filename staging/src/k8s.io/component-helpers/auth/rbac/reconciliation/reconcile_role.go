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
	"fmt"
	"reflect"

	rbacv1 "k8s.io/api/rbac/v1"
	"k8s.io/apimachinery/pkg/api/equality"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/component-helpers/auth/rbac/validation"
)

type ReconcileOperation string

var (
	ReconcileCreate   ReconcileOperation = "create"
	ReconcileUpdate   ReconcileOperation = "update"
	ReconcileRecreate ReconcileOperation = "recreate"
	ReconcileNone     ReconcileOperation = "none"
)

type RuleOwnerModifier interface {
	Get(namespace, name string) (RuleOwner, error)
	Create(RuleOwner) (RuleOwner, error)
	Update(RuleOwner) (RuleOwner, error)
}

type RuleOwner interface {
	GetObject() runtime.Object
	GetNamespace() string
	GetName() string
	GetLabels() map[string]string
	SetLabels(map[string]string)
	GetAnnotations() map[string]string
	SetAnnotations(map[string]string)
	GetRules() []rbacv1.PolicyRule
	SetRules([]rbacv1.PolicyRule)
	GetAggregationRule() *rbacv1.AggregationRule
	SetAggregationRule(*rbacv1.AggregationRule)
	DeepCopyRuleOwner() RuleOwner
}

type ReconcileRoleOptions struct {
	// Role is the expected role that will be reconciled
	Role RuleOwner
	// Confirm indicates writes should be performed. When false, results are returned as a dry-run.
	Confirm bool
	// RemoveExtraPermissions indicates reconciliation should remove extra permissions from an existing role
	RemoveExtraPermissions bool
	// Client is used to look up existing roles, and create/update the role when Confirm=true
	Client RuleOwnerModifier
}

type ReconcileClusterRoleResult struct {
	// Role is the reconciled role from the reconciliation operation.
	// If the reconcile was performed as a dry-run, or the existing role was protected, the reconciled role is not persisted.
	Role RuleOwner

	// MissingRules contains expected rules that were missing from the currently persisted role
	MissingRules []rbacv1.PolicyRule
	// ExtraRules contains extra permissions the currently persisted role had
	ExtraRules []rbacv1.PolicyRule

	// MissingAggregationRuleSelectors contains expected selectors that were missing from the currently persisted role
	MissingAggregationRuleSelectors []metav1.LabelSelector
	// ExtraAggregationRuleSelectors contains extra selectors the currently persisted role had
	ExtraAggregationRuleSelectors []metav1.LabelSelector

	// Operation is the API operation required to reconcile.
	// If no reconciliation was needed, it is set to ReconcileNone.
	// If options.Confirm == false, the reconcile was in dry-run mode, so the operation was not performed.
	// If result.Protected == true, the role opted out of reconciliation, so the operation was not performed.
	// Otherwise, the operation was performed.
	Operation ReconcileOperation
	// Protected indicates an existing role prevented reconciliation
	Protected bool
}

func (o *ReconcileRoleOptions) Run() (*ReconcileClusterRoleResult, error) {
	return o.run(0)
}

func (o *ReconcileRoleOptions) run(attempts int) (*ReconcileClusterRoleResult, error) {
	// This keeps us from retrying forever if a role keeps appearing and disappearing as we reconcile.
	// Conflict errors on update are handled at a higher level.
	if attempts > 2 {
		return nil, fmt.Errorf("exceeded maximum attempts")
	}

	var result *ReconcileClusterRoleResult

	existing, err := o.Client.Get(o.Role.GetNamespace(), o.Role.GetName())
	switch {
	case errors.IsNotFound(err):
		aggregationRule := o.Role.GetAggregationRule()
		if aggregationRule == nil {
			aggregationRule = &rbacv1.AggregationRule{}
		}
		result = &ReconcileClusterRoleResult{
			Role:                            o.Role,
			MissingRules:                    o.Role.GetRules(),
			MissingAggregationRuleSelectors: aggregationRule.ClusterRoleSelectors,
			Operation:                       ReconcileCreate,
		}

	case err != nil:
		return nil, err

	default:
		result, err = computeReconciledRole(existing, o.Role, o.RemoveExtraPermissions)
		if err != nil {
			return nil, err
		}
	}

	// If reconcile-protected, short-circuit
	if result.Protected {
		return result, nil
	}
	// If we're in dry-run mode, short-circuit
	if !o.Confirm {
		return result, nil
	}

	switch result.Operation {
	case ReconcileCreate:
		created, err := o.Client.Create(result.Role)
		// If created since we started this reconcile, re-run
		if errors.IsAlreadyExists(err) {
			return o.run(attempts + 1)
		}
		if err != nil {
			return nil, err
		}
		result.Role = created

	case ReconcileUpdate:
		updated, err := o.Client.Update(result.Role)
		// If deleted since we started this reconcile, re-run
		if errors.IsNotFound(err) {
			return o.run(attempts + 1)
		}
		if err != nil {
			return nil, err
		}
		result.Role = updated

	case ReconcileNone:
		// no-op

	default:
		return nil, fmt.Errorf("invalid operation: %v", result.Operation)
	}

	return result, nil
}

// computeReconciledRole returns the role that must be created and/or updated to make the
// existing role's permissions match the expected role's permissions
func computeReconciledRole(existing, expected RuleOwner, removeExtraPermissions bool) (*ReconcileClusterRoleResult, error) {
	result := &ReconcileClusterRoleResult{Operation: ReconcileNone}

	result.Protected = (existing.GetAnnotations()[rbacv1.AutoUpdateAnnotationKey] == "false")

	// Start with a copy of the existing object
	result.Role = existing.DeepCopyRuleOwner()

	// Merge expected annotations and labels
	result.Role.SetAnnotations(merge(expected.GetAnnotations(), result.Role.GetAnnotations()))
	if !reflect.DeepEqual(result.Role.GetAnnotations(), existing.GetAnnotations()) {
		result.Operation = ReconcileUpdate
	}
	result.Role.SetLabels(merge(expected.GetLabels(), result.Role.GetLabels()))
	if !reflect.DeepEqual(result.Role.GetLabels(), existing.GetLabels()) {
		result.Operation = ReconcileUpdate
	}

	// Compute extra and missing rules
	// Don't compute extra permissions if expected and existing roles are both aggregated
	if expected.GetAggregationRule() == nil || existing.GetAggregationRule() == nil {
		_, result.ExtraRules = validation.Covers(expected.GetRules(), existing.GetRules())
	}
	_, result.MissingRules = validation.Covers(existing.GetRules(), expected.GetRules())

	switch {
	case !removeExtraPermissions && len(result.MissingRules) > 0:
		// add missing rules in the union case
		result.Role.SetRules(append(result.Role.GetRules(), result.MissingRules...))
		result.Operation = ReconcileUpdate

	case removeExtraPermissions && (len(result.MissingRules) > 0 || len(result.ExtraRules) > 0):
		// stomp to expected rules in the non-union case
		result.Role.SetRules(expected.GetRules())
		result.Operation = ReconcileUpdate
	}

	// Compute extra and missing rules
	_, result.ExtraAggregationRuleSelectors = aggregationRuleCovers(expected.GetAggregationRule(), existing.GetAggregationRule())
	_, result.MissingAggregationRuleSelectors = aggregationRuleCovers(existing.GetAggregationRule(), expected.GetAggregationRule())

	switch {
	case expected.GetAggregationRule() == nil && existing.GetAggregationRule() != nil:
		// we didn't expect this to be an aggregated role at all, remove the existing aggregation
		result.Role.SetAggregationRule(nil)
		result.Operation = ReconcileUpdate

	case !removeExtraPermissions && len(result.MissingAggregationRuleSelectors) > 0:
		// add missing rules in the union case
		aggregationRule := result.Role.GetAggregationRule()
		if aggregationRule == nil {
			aggregationRule = &rbacv1.AggregationRule{}
		}
		aggregationRule.ClusterRoleSelectors = append(aggregationRule.ClusterRoleSelectors, result.MissingAggregationRuleSelectors...)
		result.Role.SetAggregationRule(aggregationRule)
		result.Operation = ReconcileUpdate

	case removeExtraPermissions && (len(result.MissingAggregationRuleSelectors) > 0 || len(result.ExtraAggregationRuleSelectors) > 0):
		result.Role.SetAggregationRule(expected.GetAggregationRule())
		result.Operation = ReconcileUpdate
	}

	return result, nil
}

// merge combines the given maps with the later annotations having higher precedence
func merge(maps ...map[string]string) map[string]string {
	var output map[string]string = nil
	for _, m := range maps {
		if m != nil && output == nil {
			output = map[string]string{}
		}
		for k, v := range m {
			output[k] = v
		}
	}
	return output
}

// aggregationRuleCovers determines whether or not the ownerSelectors cover the servantSelectors in terms of semantically
// equal label selectors.
// It returns whether or not the ownerSelectors cover and a list of the rules that the ownerSelectors do not cover.
func aggregationRuleCovers(ownerRule, servantRule *rbacv1.AggregationRule) (bool, []metav1.LabelSelector) {
	switch {
	case ownerRule == nil && servantRule == nil:
		return true, []metav1.LabelSelector{}
	case ownerRule == nil && servantRule != nil:
		return false, servantRule.ClusterRoleSelectors
	case ownerRule != nil && servantRule == nil:
		return true, []metav1.LabelSelector{}

	}

	ownerSelectors := ownerRule.ClusterRoleSelectors
	servantSelectors := servantRule.ClusterRoleSelectors
	uncoveredSelectors := []metav1.LabelSelector{}

	for _, servantSelector := range servantSelectors {
		covered := false
		for _, ownerSelector := range ownerSelectors {
			if equality.Semantic.DeepEqual(ownerSelector, servantSelector) {
				covered = true
				break
			}
		}
		if !covered {
			uncoveredSelectors = append(uncoveredSelectors, servantSelector)
		}
	}

	return (len(uncoveredSelectors) == 0), uncoveredSelectors
}
