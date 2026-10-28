/*
Copyright The Kubernetes Authors.

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

package generic

import (
	"context"
	"fmt"

	admissionregistrationv1 "k8s.io/api/admissionregistration/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apiserver/pkg/admission"
	"k8s.io/apiserver/pkg/admission/coverage"
	admissionmetrics "k8s.io/apiserver/pkg/admission/metrics"
	"k8s.io/klog/v2"
)

// CoverageViolation is a (policy, binding) pair that intercepts one of a
// request's coverage sources, but not the request itself, or whose coverage
// couldn't be determined (see Err).
type CoverageViolation struct {
	Policy  PolicyAccessor
	Binding BindingAccessor
	coverage.Violation
	// Err is set if the pair's selectors couldn't be evaluated, so whether the
	// pair applies to the request is unknown. Violation.Source is then empty.
	Err error
}

// Message describes the violation. kind is the policy's kind, e.g.
// "ValidatingAdmissionPolicy".
func (v *CoverageViolation) Message(kind string) string {
	sep := " "
	if v.Err != nil {
		sep = ": "
	}
	return fmt.Sprintf("%s %q (binding %q)%s%s", kind, v.Policy.GetName(), v.Binding.GetName(), sep, v.Reason())
}

// Reason describes the violation without naming the policy and binding.
func (v *CoverageViolation) Reason() string {
	if v.Err != nil {
		return fmt.Sprintf("unable to determine admission coverage for %s: %v", v.TargetString(), v.Err)
	}
	return fmt.Sprintf("intercepts %s but not %s; add %s to the policy's matchConstraints "+
		"(and the binding's matchResources, if set), or exclude it explicitly via excludeResourceRules, "+
		"to allow this request", v.SourceString(), v.TargetString(), v.RuleHint())
}

// CoverageChecker checks requests for (policy, binding) pairs that don't cover them.
type CoverageChecker[P, B runtime.Object, E Evaluator] struct {
	matcher            PolicyMatcher
	newPolicyAccessor  func(P) PolicyAccessor
	newBindingAccessor func(B) BindingAccessor
}

// NewCoverageChecker returns a CoverageChecker that matches pairs with matcher,
// the same matcher the dispatcher uses.
func NewCoverageChecker[P, B runtime.Object, E Evaluator](
	matcher PolicyMatcher,
	newPolicyAccessor func(P) PolicyAccessor,
	newBindingAccessor func(B) BindingAccessor,
) *CoverageChecker[P, B, E] {
	return &CoverageChecker[P, B, E]{
		matcher:            matcher,
		newPolicyAccessor:  newPolicyAccessor,
		newBindingAccessor: newBindingAccessor,
	}
}

// Check returns the coverage violations of the (policy, binding) pairs in hooks
// for a. Selectors still scope the check; matchConditions, params,
// configuration errors and failurePolicy don't.
//
// Reinvocations are exempt: coverage was already enforced in the first round.
func (c *CoverageChecker[P, B, E]) Check(a admission.Attributes, o admission.ObjectInterfaces, hooks []PolicyHook[P, B, E]) []CoverageViolation {
	if a.GetReinvocationContext().IsReinvoke() {
		return nil
	}
	var violations []CoverageViolation
	for _, hook := range hooks {
		violations = append(violations, c.checkHook(a, o, hook)...)
	}
	return violations
}

func (c *CoverageChecker[P, B, E]) checkHook(a admission.Attributes, o admission.ObjectInterfaces, hook PolicyHook[P, B, E]) []CoverageViolation {
	var violations []CoverageViolation
	// Matched lazily: most pairs have no gap for a, and those cost nothing.
	var policy PolicyAccessor
	var defMatches bool
	for i, b := range hook.Bindings {
		v := hook.CoverageGapsAt(i).Check(a)
		if v == nil {
			continue
		}
		if policy == nil {
			policy = c.newPolicyAccessor(hook.Policy)
			var err error
			// A matching error is a configuration error, which the dispatcher
			// reports subject to failurePolicy.
			if defMatches, _, _, err = c.matcher.DefinitionMatches(a, o, policy); err != nil {
				return violations
			}
		}
		binding := c.newBindingAccessor(b)
		if defMatches {
			// Matching on the real request honors matchPolicy: Equivalent, which
			// the gaps ignore, so a pair that applies to a is never in violation.
			if matches, err := c.matcher.BindingMatches(a, o, binding); err == nil && matches {
				continue
			}
		}
		cv := CoverageViolation{Policy: policy, Binding: binding, Violation: *v}
		// The gaps ignore selectors. A pair whose selectors exclude a doesn't
		// apply to this object whatever its rules are.
		if matches, err := c.selectorsMatch(a, o, policy, binding, defMatches); err != nil {
			cv.Err = err
			cv.Source = coverage.Shape{}
		} else if !matches {
			continue
		}
		violations = append(violations, cv)
	}
	return violations
}

// selectorsMatch reports whether a matches the selectors of the policy and,
// if it has matchResources, the binding. defMatched reports that the policy
// is already known to match a, selectors included.
func (c *CoverageChecker[P, B, E]) selectorsMatch(a admission.Attributes, o admission.ObjectInterfaces, policy PolicyAccessor, binding BindingAccessor, defMatched bool) (bool, error) {
	if !defMatched {
		if matches, err := c.matchesSelectors(a, o, binding, policy.GetMatchConstraints()); err != nil || !matches {
			return matches, err
		}
	}
	if mr := binding.GetMatchResources(); mr != nil {
		return c.matchesSelectors(a, o, binding, mr)
	}
	return true, nil
}

// matchesSelectors evaluates only the selectors of mr, through the
// dispatcher's matcher: a clean mismatch of either selector is (false, nil),
// and a selector error is returned. A nil selector is treated as the empty
// selector it defaults to, which matches everything; PolicyMatcher would
// reject it instead.
func (c *CoverageChecker[P, B, E]) matchesSelectors(a admission.Attributes, o admission.ObjectInterfaces, binding BindingAccessor, mr *admissionregistrationv1.MatchResources) (bool, error) {
	selectors := &admissionregistrationv1.MatchResources{
		NamespaceSelector: mr.NamespaceSelector,
		ObjectSelector:    mr.ObjectSelector,
	}
	if selectors.NamespaceSelector == nil {
		selectors.NamespaceSelector = &metav1.LabelSelector{}
	}
	if selectors.ObjectSelector == nil {
		selectors.ObjectSelector = &metav1.LabelSelector{}
	}
	// No resource rules match every request, so only the selectors decide.
	return c.matcher.BindingMatches(a, o, selectorsOnlyBinding{BindingAccessor: binding, matchResources: selectors})
}

// selectorsOnlyBinding is a binding whose matchResources hold only selectors.
type selectorsOnlyBinding struct {
	BindingAccessor
	matchResources *admissionregistrationv1.MatchResources
}

func (b selectorsOnlyBinding) GetMatchResources() *admissionregistrationv1.MatchResources {
	return b.matchResources
}

// RecordCoverageViolations logs and counts violations, and records them in the
// coverage audit annotation of the plugin dispatching hooks of the given kind.
//
// Admission annotations can't be overwritten, so each plugin must call it at
// most once per request.
func RecordCoverageViolations(ctx context.Context, a admission.Attributes, kind coverage.HookKind, violations []CoverageViolation) {
	if len(violations) == 0 {
		return
	}
	stepType := "validate"
	if kind == coverage.PolicyMutating {
		stepType = "admit"
	}
	entries := make([]coverage.AuditAnnotationEntry, 0, len(violations))
	for _, v := range violations {
		admissionmetrics.Metrics.ObserveCoverageViolation(ctx, v.Policy.GetName(), admissionmetrics.KindPolicy, stepType, string(a.GetOperation()))
		klog.V(2).InfoS("Admission coverage violation", "kind", kind, "policy", v.Policy.GetName(), "binding", v.Binding.GetName(),
			"source", v.SourceString(), "target", v.TargetString(), "err", v.Err)
		entries = append(entries, coverage.AuditAnnotationEntry{
			Name:    v.Policy.GetName(),
			Binding: v.Binding.GetName(),
			Source:  v.SourceString(),
			Target:  v.TargetString(),
		})
	}
	if err := coverage.AddAuditAnnotation(a, kind, entries); err != nil {
		klog.Warningf("Failed to set admission coverage audit annotation for %s: %v", kind, err)
	}
}
