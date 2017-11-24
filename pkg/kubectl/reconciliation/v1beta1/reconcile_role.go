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

package v1beta1

import (
	"fmt"
	"reflect"
	"strings"

	rbacv1beta1 "k8s.io/api/rbac/v1beta1"
	"k8s.io/apimachinery/pkg/api/equality"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
)

// ReconcileOperation is a duplication in k8s.io/kubernetes/pkg/registry/rbac/reconciliation
type ReconcileOperation string

var (
	// ReconcileCreate is a duplication in k8s.io/kubernetes/pkg/registry/rbac/reconciliation
	ReconcileCreate ReconcileOperation = "create"
	// ReconcileUpdate is a duplication in k8s.io/kubernetes/pkg/registry/rbac/reconciliation
	ReconcileUpdate ReconcileOperation = "update"
	// ReconcileRecreate is a duplication in k8s.io/kubernetes/pkg/registry/rbac/reconciliation
	ReconcileRecreate ReconcileOperation = "recreate"
	// ReconcileNone is a duplication in k8s.io/kubernetes/pkg/registry/rbac/reconciliation
	ReconcileNone ReconcileOperation = "none"
)

// RuleOwnerModifier is a duplication in k8s.io/kubernetes/pkg/registry/rbac/reconciliation
type RuleOwnerModifier interface {
	// Get is a duplication in k8s.io/kubernetes/pkg/registry/rbac/reconciliation
	Get(namespace, name string) (RuleOwner, error)
	// Create is a duplication in k8s.io/kubernetes/pkg/registry/rbac/reconciliation
	Create(RuleOwner) (RuleOwner, error)
	// Update is a duplication in k8s.io/kubernetes/pkg/registry/rbac/reconciliation
	Update(RuleOwner) (RuleOwner, error)
}

// RuleOwner is a duplication in k8s.io/kubernetes/pkg/registry/rbac/reconciliation
type RuleOwner interface {
	// GetObject is a duplication in k8s.io/kubernetes/pkg/registry/rbac/reconciliation
	GetObject() runtime.Object
	// GetNamespace is a duplication in k8s.io/kubernetes/pkg/registry/rbac/reconciliation
	GetNamespace() string
	// GetName is a duplication in k8s.io/kubernetes/pkg/registry/rbac/reconciliation
	GetName() string
	// GetLabels is a duplication in k8s.io/kubernetes/pkg/registry/rbac/reconciliation
	GetLabels() map[string]string
	// SetLabels is a duplication in k8s.io/kubernetes/pkg/registry/rbac/reconciliation
	SetLabels(map[string]string)
	// GetAnnotations is a duplication in k8s.io/kubernetes/pkg/registry/rbac/reconciliation
	GetAnnotations() map[string]string
	// SetAnnotations is a duplication in k8s.io/kubernetes/pkg/registry/rbac/reconciliation
	SetAnnotations(map[string]string)
	// GetRules is a duplication in k8s.io/kubernetes/pkg/registry/rbac/reconciliation
	GetRules() []rbacv1beta1.PolicyRule
	// SetRules is a duplication in k8s.io/kubernetes/pkg/registry/rbac/reconciliation
	SetRules([]rbacv1beta1.PolicyRule)
	// GetAggregationRule is a duplication in k8s.io/kubernetes/pkg/registry/rbac/reconciliation
	GetAggregationRule() *rbacv1beta1.AggregationRule
	// SetAggregationRule is a duplication in k8s.io/kubernetes/pkg/registry/rbac/reconciliation
	SetAggregationRule(*rbacv1beta1.AggregationRule)
	// DeepCopyRuleOwner is a duplication in k8s.io/kubernetes/pkg/registry/rbac/reconciliation
	DeepCopyRuleOwner() RuleOwner
}

// ReconcileRoleOptions is a duplication in k8s.io/kubernetes/pkg/registry/rbac/reconciliation
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

// ReconcileClusterRoleResult is a duplication in k8s.io/kubernetes/pkg/registry/rbac/reconciliation
type ReconcileClusterRoleResult struct {
	// Role is the reconciled role from the reconciliation operation.
	// If the reconcile was performed as a dry-run, or the existing role was protected, the reconciled role is not persisted.
	Role RuleOwner

	// MissingRules contains expected rules that were missing from the currently persisted role
	MissingRules []rbacv1beta1.PolicyRule
	// ExtraRules contains extra permissions the currently persisted role had
	ExtraRules []rbacv1beta1.PolicyRule

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

// Run is a duplication in k8s.io/kubernetes/pkg/registry/rbac/reconciliation
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
			aggregationRule = &rbacv1beta1.AggregationRule{}
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

func has(set []string, ele string) bool {
	for _, s := range set {
		if s == ele {
			return true
		}
	}
	return false
}

func hasAll(set, contains []string) bool {
	owning := make(map[string]struct{}, len(set))
	for _, ele := range set {
		owning[ele] = struct{}{}
	}
	for _, ele := range contains {
		if _, ok := owning[ele]; !ok {
			return false
		}
	}
	return true
}

func resourceCoversAll(setResources, coversResources []string) bool {
	// if we have a star or an exact match on all resources, then we match
	if has(setResources, rbacv1beta1.ResourceAll) || hasAll(setResources, coversResources) {
		return true
	}

	for _, path := range coversResources {
		// if we have an exact match, then we match.
		if has(setResources, path) {
			continue
		}
		// if we're not a subresource, then we definitely don't match.  fail.
		if !strings.Contains(path, "/") {
			return false
		}
		tokens := strings.SplitN(path, "/", 2)
		resourceToCheck := "*/" + tokens[1]
		if !has(setResources, resourceToCheck) {
			return false
		}
	}

	return true
}

func nonResourceURLsCoversAll(set, covers []string) bool {
	for _, path := range covers {
		covered := false
		for _, owner := range set {
			if nonResourceURLCovers(owner, path) {
				covered = true
				break
			}
		}
		if !covered {
			return false
		}
	}
	return true
}

func nonResourceURLCovers(ownerPath, subPath string) bool {
	if ownerPath == subPath {
		return true
	}
	return strings.HasSuffix(ownerPath, "*") && strings.HasPrefix(subPath, strings.TrimRight(ownerPath, "*"))
}

// ruleCovers determines whether the ownerRule (which may have multiple verbs, resources, and resourceNames) covers
// the subrule (which may only contain at most one verb, resource, and resourceName)
func ruleCovers(ownerRule, subRule rbacv1beta1.PolicyRule) bool {
	verbMatches := has(ownerRule.Verbs, rbacv1beta1.VerbAll) || hasAll(ownerRule.Verbs, subRule.Verbs)
	groupMatches := has(ownerRule.APIGroups, rbacv1beta1.APIGroupAll) || hasAll(ownerRule.APIGroups, subRule.APIGroups)
	resourceMatches := resourceCoversAll(ownerRule.Resources, subRule.Resources)
	nonResourceURLMatches := nonResourceURLsCoversAll(ownerRule.NonResourceURLs, subRule.NonResourceURLs)

	resourceNameMatches := false

	if len(subRule.ResourceNames) == 0 {
		resourceNameMatches = (len(ownerRule.ResourceNames) == 0)
	} else {
		resourceNameMatches = (len(ownerRule.ResourceNames) == 0) || hasAll(ownerRule.ResourceNames, subRule.ResourceNames)
	}

	return verbMatches && groupMatches && resourceMatches && resourceNameMatches && nonResourceURLMatches
}

// BreakdownRule takes a rule and builds an equivalent list of rules that each have at most one verb, one resource, and one resource name
func BreakdownRule(rule rbacv1beta1.PolicyRule) []rbacv1beta1.PolicyRule {
	subrules := []rbacv1beta1.PolicyRule{}
	for _, group := range rule.APIGroups {
		for _, resource := range rule.Resources {
			for _, verb := range rule.Verbs {
				if len(rule.ResourceNames) > 0 {
					for _, resourceName := range rule.ResourceNames {
						subrules = append(subrules, rbacv1beta1.PolicyRule{APIGroups: []string{group}, Resources: []string{resource}, Verbs: []string{verb}, ResourceNames: []string{resourceName}})
					}

				} else {
					subrules = append(subrules, rbacv1beta1.PolicyRule{APIGroups: []string{group}, Resources: []string{resource}, Verbs: []string{verb}})
				}

			}
		}
	}

	// Non-resource URLs are unique because they only combine with verbs.
	for _, nonResourceURL := range rule.NonResourceURLs {
		for _, verb := range rule.Verbs {
			subrules = append(subrules, rbacv1beta1.PolicyRule{NonResourceURLs: []string{nonResourceURL}, Verbs: []string{verb}})
		}
	}

	return subrules
}

// Covers determines whether or not the ownerRules cover the servantRules in terms of allowed actions.
// It returns whether or not the ownerRules cover and a list of the rules that the ownerRules do not cover.
func Covers(ownerRules, servantRules []rbacv1beta1.PolicyRule) (bool, []rbacv1beta1.PolicyRule) {
	// 1.  Break every servantRule into individual rule tuples: group, verb, resource, resourceName
	// 2.  Compare the mini-rules against each owner rule.  Because the breakdown is down to the most atomic level, we're guaranteed that each mini-servant rule will be either fully covered or not covered by a single owner rule
	// 3.  Any left over mini-rules means that we are not covered and we have a nice list of them.
	// TODO: it might be nice to collapse the list down into something more human readable

	subrules := []rbacv1beta1.PolicyRule{}
	for _, servantRule := range servantRules {
		subrules = append(subrules, BreakdownRule(servantRule)...)
	}

	uncoveredRules := []rbacv1beta1.PolicyRule{}
	for _, subrule := range subrules {
		covered := false
		for _, ownerRule := range ownerRules {
			if ruleCovers(ownerRule, subrule) {
				covered = true
				break
			}
		}

		if !covered {
			uncoveredRules = append(uncoveredRules, subrule)
		}
	}

	return (len(uncoveredRules) == 0), uncoveredRules
}

// computeReconciledRole returns the role that must be created and/or updated to make the
// existing role's permissions match the expected role's permissions
func computeReconciledRole(existing, expected RuleOwner, removeExtraPermissions bool) (*ReconcileClusterRoleResult, error) {
	result := &ReconcileClusterRoleResult{Operation: ReconcileNone}

	result.Protected = (existing.GetAnnotations()[rbacv1beta1.AutoUpdateAnnotationKey] == "false")

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
	_, result.ExtraRules = Covers(expected.GetRules(), existing.GetRules())
	_, result.MissingRules = Covers(existing.GetRules(), expected.GetRules())

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
	case !removeExtraPermissions && len(result.MissingAggregationRuleSelectors) > 0:
		// add missing rules in the union case
		aggregationRule := result.Role.GetAggregationRule()
		if aggregationRule == nil {
			aggregationRule = &rbacv1beta1.AggregationRule{}
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
	var output map[string]string
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
func aggregationRuleCovers(ownerRule, servantRule *rbacv1beta1.AggregationRule) (bool, []metav1.LabelSelector) {
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
