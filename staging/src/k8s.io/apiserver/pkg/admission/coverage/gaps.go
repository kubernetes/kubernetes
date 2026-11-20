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

package coverage

import (
	"slices"
	"strings"

	admissionregistrationv1 "k8s.io/api/admissionregistration/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/apiserver/pkg/admission"
	"k8s.io/apiserver/pkg/admission/plugin/policy/matching"
	"k8s.io/apiserver/pkg/admission/plugin/webhook/predicates/rules"
)

var (
	supportedOperations = []admission.Operation{admission.Create, admission.Update, admission.Delete, admission.Connect}
	namespacesGR        = schema.GroupResource{Resource: "namespaces"}
	anyVersion          = []string{"*"}
)

// targetKey identifies a requirement's target. Versions are deliberately
// absent: coverage is version-agnostic.
type targetKey struct {
	resource    schema.GroupResource
	operation   admission.Operation
	subresource string
}

func keyFor(gr schema.GroupResource, s Shape) targetKey {
	return targetKey{resource: gr, operation: s.Operation, subresource: s.Subresource}
}

// Validate checks that reqs are well-formed so mistakes in the server's
// requirement table fail at startup rather than silently leaving a target
// uncovered.
func Validate(reqs []Requirement) error {
	return validate(reqs).ToAggregate()
}

func validate(reqs []Requirement) field.ErrorList {
	var errs field.ErrorList
	seen := sets.New[targetKey]()
	for i, r := range reqs {
		fldPath := field.NewPath("requirements").Index(i)
		resPath := fldPath.Child("resource")
		if strings.Contains(r.Resource.Group, "*") {
			errs = append(errs, field.Invalid(resPath.Child("group"), r.Resource.Group, "must not contain wildcards"))
		}
		switch {
		case r.Resource.Resource == "":
			errs = append(errs, field.Required(resPath.Child("resource"), ""))
		case strings.Contains(r.Resource.Resource, "*"):
			errs = append(errs, field.Invalid(resPath.Child("resource"), r.Resource.Resource, "must not contain wildcards"))
		case strings.Contains(r.Resource.Resource, "/"):
			errs = append(errs, field.Invalid(resPath.Child("resource"), r.Resource.Resource, "must not contain '/'; set the subresource in the shape"))
		}
		errs = append(errs, validateShape(fldPath.Child("target"), r.Target)...)
		if len(r.Sources) == 0 {
			errs = append(errs, field.Required(fldPath.Child("sources"), "a requirement with no sources is never violated"))
		}
		for j, s := range r.Sources {
			srcPath := fldPath.Child("sources").Index(j)
			errs = append(errs, validateShape(srcPath, s)...)
			if s == r.Target {
				errs = append(errs, field.Invalid(srcPath, s, "must differ from the target"))
			}
		}
		k := keyFor(r.Resource, r.Target)
		if seen.Has(k) {
			errs = append(errs, field.Duplicate(fldPath.Child("target"), r.Target))
		}
		seen.Insert(k)
	}
	return errs
}

// validateShape rejects shapes that no request has. Requests name concrete
// subresources, so a wildcard would never match one.
func validateShape(fldPath *field.Path, s Shape) field.ErrorList {
	errs := validateOperation(fldPath.Child("operation"), s.Operation)
	if strings.Contains(s.Subresource, "*") {
		errs = append(errs, field.Invalid(fldPath.Child("subresource"), s.Subresource, "must not contain wildcards"))
	}
	return errs
}

func validateOperation(fldPath *field.Path, op admission.Operation) field.ErrorList {
	if !slices.Contains(supportedOperations, op) {
		return field.ErrorList{field.NotSupported(fldPath, op, supportedOperations)}
	}
	return nil
}

// Gaps holds the precomputed violated targets for one admission hook
// (a webhook or a (policy, binding) pair). A nil *Gaps means the hook
// covers all requirements (or doesn't intercept any source).
type Gaps struct {
	violatedTargets map[scopedTarget]targetViolation
}

// scopedTarget is a target together with the request's scope. Besides the
// name, scope is the only request property that rules can tell apart for a
// given (GroupResource, operation, subresource).
type scopedTarget struct {
	targetKey
	namespaced bool
}

// targetViolation records the source a hook intercepts without intercepting
// the target. A nil source means no violation.
type targetViolation struct {
	// source applies to every name the hook's resourceNames don't mention.
	source *Shape
	// byName holds the result for each name the hook's resourceNames mention.
	byName map[string]*Shape
}

// Check returns the coverage violation of this hook for attr, or nil if attr
// is not a violated target of this hook. It is safe to call on a nil *Gaps.
func (g *Gaps) Check(attr admission.Attributes) *Violation {
	if g == nil {
		return nil
	}
	gr := attr.GetResource().GroupResource()
	target := Shape{Operation: attr.GetOperation(), Subresource: attr.GetSubresource()}
	v, ok := g.violatedTargets[scopedTarget{
		targetKey:  keyFor(gr, target),
		namespaced: attr.GetNamespace() != metav1.NamespaceNone && gr != namespacesGR,
	}]
	if !ok {
		return nil
	}
	src := v.source
	if s, ok := v.byName[attr.GetName()]; ok {
		src = s
	}
	if src == nil {
		return nil
	}
	return &Violation{Resource: gr, Target: target, Source: *src}
}

// ForWebhook precomputes the coverage gaps for a webhook with the given
// rules. It returns nil if the webhook covers all requirements or intercepts
// no source.
//
// matchPolicy is ignored: Equivalent only widens matching to other groups and
// versions of the same resource, and each requirement is pinned to one
// GroupResource and is version-agnostic.
func ForWebhook(reqs []Requirement, webhookRules []admissionregistrationv1.RuleWithOperations) *Gaps {
	if len(reqs) == 0 || len(webhookRules) == 0 {
		return nil
	}
	rs := normalizeWebhookRules(webhookRules)
	intercepts := func(attr admission.Attributes) bool { return matchesAnyWebhookRule(rs, attr) }
	return newGaps(reqs, nil, intercepts, nil)
}

// ForPolicyBinding precomputes the coverage gaps for a (definition, binding)
// pair. definition is the policy's matchConstraints; binding is nil if the
// binding has no matchResources. It returns nil if the pair covers all
// requirements or intercepts no source.
func ForPolicyBinding(reqs []Requirement, definition, binding *admissionregistrationv1.MatchResources) *Gaps {
	// The policy matcher treats nil matchConstraints as a configuration error,
	// so such a definition never intercepts anything.
	if len(reqs) == 0 || definition == nil {
		return nil
	}
	def, bind := normalizeMatchResources(definition), normalizeMatchResources(binding)
	intercepts := func(attr admission.Attributes) bool {
		return matching.MatchesResourceRulesExact(def, attr) && matching.MatchesResourceRulesExact(bind, attr)
	}
	// An exclusion of the target on either the policy or the binding is an
	// explicit opt-out of the requirement for this pair.
	excludes := func(attr admission.Attributes) bool {
		return matching.ExcludedExact(def, attr) || matching.ExcludedExact(bind, attr)
	}
	return newGaps(reqs, collectResourceNames(def, bind), intercepts, excludes)
}

// newGaps precomputes, for each requirement and scope, the first source a
// hook intercepts when it neither intercepts nor excludes the target.
// intercepts and excludes evaluate the hook's resource rules; excludes may be
// nil. names are the resource names the hook's rules mention: only those can
// produce a different result than any other name.
func newGaps(reqs []Requirement, names sets.Set[string], intercepts, excludes func(admission.Attributes) bool) *Gaps {
	otherName := unmentionedName(names)
	violated := map[scopedTarget]targetViolation{}
	for _, q := range reqs {
		for _, namespaced := range []bool{false, true} {
			// Namespaces are cluster-scoped, even though their requests carry a namespace.
			if namespaced && q.Resource == namespacesGR {
				continue
			}
			violatedSource := func(name string) *Shape {
				target := syntheticAttr(q.Resource, q.Target, namespaced, name)
				if intercepts(target) || (excludes != nil && excludes(target)) {
					return nil
				}
				for _, s := range q.Sources {
					if intercepts(syntheticAttr(q.Resource, s, namespaced, name)) {
						return &s
					}
				}
				return nil
			}
			v := targetViolation{source: violatedSource(otherName)}
			found := v.source != nil
			for name := range names {
				if v.byName == nil {
					v.byName = make(map[string]*Shape, len(names))
				}
				v.byName[name] = violatedSource(name)
				found = found || v.byName[name] != nil
			}
			if found {
				violated[scopedTarget{targetKey: keyFor(q.Resource, q.Target), namespaced: namespaced}] = v
			}
		}
	}
	if len(violated) == 0 {
		return nil
	}
	return &Gaps{violatedTargets: violated}
}

// unmentionedName returns a name that isn't in names. resourceNames only
// compare names for equality, so every name a hook's rules don't mention
// matches the same way, and this one name stands in for all of them.
func unmentionedName(names sets.Set[string]) string {
	name := "\x00"
	for names.Has(name) {
		name += "\x00"
	}
	return name
}

func normalizeWebhookRules(in []admissionregistrationv1.RuleWithOperations) []admissionregistrationv1.RuleWithOperations {
	out := slices.Clone(in)
	for i := range out {
		if len(out[i].APIVersions) > 0 {
			out[i].APIVersions = anyVersion
		}
	}
	return out
}

func matchesAnyWebhookRule(rs []admissionregistrationv1.RuleWithOperations, attr admission.Attributes) bool {
	for _, r := range rs {
		m := rules.Matcher{Rule: r, Attr: attr}
		if m.Matches() {
			return true
		}
	}
	return false
}

func normalizeMatchResources(mr *admissionregistrationv1.MatchResources) *admissionregistrationv1.MatchResources {
	if mr == nil {
		return nil
	}
	out := *mr
	out.ResourceRules = normalizeNamedRules(mr.ResourceRules)
	out.ExcludeResourceRules = normalizeNamedRules(mr.ExcludeResourceRules)
	return &out
}

func normalizeNamedRules(in []admissionregistrationv1.NamedRuleWithOperations) []admissionregistrationv1.NamedRuleWithOperations {
	if len(in) == 0 {
		return nil
	}
	out := slices.Clone(in)
	for i := range out {
		if len(out[i].APIVersions) > 0 {
			out[i].APIVersions = anyVersion
		}
	}
	return out
}

func collectResourceNames(mrs ...*admissionregistrationv1.MatchResources) sets.Set[string] {
	var names sets.Set[string]
	add := func(rs []admissionregistrationv1.NamedRuleWithOperations) {
		for _, r := range rs {
			if len(r.ResourceNames) > 0 {
				if names == nil {
					names = sets.New[string]()
				}
				names.Insert(r.ResourceNames...)
			}
		}
	}
	for _, mr := range mrs {
		if mr != nil {
			add(mr.ResourceRules)
			add(mr.ExcludeResourceRules)
		}
	}
	return names
}

func syntheticAttr(gr schema.GroupResource, s Shape, namespaced bool, name string) admission.Attributes {
	ns := metav1.NamespaceNone
	if namespaced {
		// Rules only distinguish namespaced from cluster-scoped requests, not
		// particular namespaces, so any non-empty namespace will do.
		ns = metav1.NamespaceDefault
	}
	return admission.NewAttributesRecord(
		nil,
		nil,
		schema.GroupVersionKind{},
		ns,
		name,
		gr.WithVersion("v1"),
		s.Subresource,
		s.Operation,
		nil,
		false,
		nil,
	)
}
