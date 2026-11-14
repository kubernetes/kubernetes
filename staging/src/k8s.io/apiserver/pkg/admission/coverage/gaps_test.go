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
	"strings"
	"testing"

	admissionregistrationv1 "k8s.io/api/admissionregistration/v1"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apiserver/pkg/admission"
	"k8s.io/apiserver/pkg/authentication/user"
	"k8s.io/utils/ptr"
)

var (
	podsGR        = schema.GroupResource{Resource: "pods"}
	deploymentsGR = schema.GroupResource{Group: "apps", Resource: "deployments"}

	podDynamicReq = Requirement{
		Resource: podsGR,
		Target:   Shape{Operation: admission.Update, Subresource: "dynamic"},
		Sources: []Shape{
			{Operation: admission.Create},
			{Operation: admission.Update},
			{Operation: admission.Update, Subresource: "resize"},
			{Operation: admission.Update, Subresource: "ephemeralcontainers"},
		},
	}
	deploymentScaleReq = Requirement{
		Resource: deploymentsGR,
		Target:   Shape{Operation: admission.Update, Subresource: "scale"},
		Sources:  []Shape{{Operation: admission.Update}},
	}
)

func newAttr(gvr schema.GroupVersionResource, subresource string, op admission.Operation, ns, name string) admission.Attributes {
	return admission.NewAttributesRecord(
		&corev1.Pod{},
		&corev1.Pod{},
		schema.GroupVersionKind{Group: gvr.Group, Version: gvr.Version, Kind: "Object"},
		ns,
		name,
		gvr,
		subresource,
		op,
		nil,
		false,
		&user.DefaultInfo{Name: "alice"},
	)
}

func podTargetAttr() admission.Attributes {
	return newAttr(schema.GroupVersionResource{Version: "v1", Resource: "pods"}, "dynamic", admission.Update, "default", "my-pod")
}

func rule(ops []admissionregistrationv1.OperationType, groups, versions, resources []string, scope *admissionregistrationv1.ScopeType) admissionregistrationv1.RuleWithOperations {
	return admissionregistrationv1.RuleWithOperations{
		Operations: ops,
		Rule: admissionregistrationv1.Rule{
			APIGroups:   groups,
			APIVersions: versions,
			Resources:   resources,
			Scope:       scope,
		},
	}
}

func namedRule(r admissionregistrationv1.RuleWithOperations, names ...string) admissionregistrationv1.NamedRuleWithOperations {
	return admissionregistrationv1.NamedRuleWithOperations{
		ResourceNames:      names,
		RuleWithOperations: r,
	}
}

func TestValidate(t *testing.T) {
	if err := Validate(nil); err != nil {
		t.Fatalf("Validate(nil): %v", err)
	}
	if err := Validate([]Requirement{podDynamicReq, deploymentScaleReq}); err != nil {
		t.Fatalf("Validate(valid): %v", err)
	}

	tests := []struct {
		name    string
		reqs    []Requirement
		wantErr string
	}{
		{
			name: "empty resource",
			reqs: []Requirement{{
				Target:  Shape{Operation: admission.Update, Subresource: "dynamic"},
				Sources: []Shape{{Operation: admission.Create}},
			}},
			wantErr: "requirements[0].resource.resource: Required value",
		},
		{
			name: "empty target operation",
			reqs: []Requirement{{
				Resource: podsGR,
				Sources:  []Shape{{Operation: admission.Create}},
			}},
			wantErr: "requirements[0].target.operation: Unsupported value",
		},
		{
			name: "unknown target operation",
			reqs: []Requirement{{
				Resource: podsGR,
				Target:   Shape{Operation: "PATCH"},
				Sources:  []Shape{{Operation: admission.Create}},
			}},
			wantErr: "requirements[0].target.operation: Unsupported value",
		},
		{
			name: "unknown source operation",
			reqs: []Requirement{{
				Resource: podsGR,
				Target:   Shape{Operation: admission.Update, Subresource: "dynamic"},
				Sources:  []Shape{{Operation: "PATCH"}},
			}},
			wantErr: "requirements[0].sources[0].operation: Unsupported value",
		},
		{
			name: "target in its own sources",
			reqs: []Requirement{{
				Resource: podsGR,
				Target:   Shape{Operation: admission.Update, Subresource: "dynamic"},
				Sources: []Shape{
					{Operation: admission.Create},
					{Operation: admission.Update, Subresource: "dynamic"},
				},
			}},
			wantErr: "requirements[0].sources[1]: Invalid value",
		},
		{
			name:    "no sources",
			reqs:    []Requirement{{Resource: podsGR, Target: Shape{Operation: admission.Update, Subresource: "dynamic"}}},
			wantErr: "requirements[0].sources: Required value",
		},
		{
			name:    "wildcard group",
			reqs:    []Requirement{{Resource: schema.GroupResource{Group: "*", Resource: "pods"}, Target: Shape{Operation: admission.Update, Subresource: "dynamic"}, Sources: []Shape{{Operation: admission.Create}}}},
			wantErr: "requirements[0].resource.group: Invalid value",
		},
		{
			name:    "wildcard resource",
			reqs:    []Requirement{{Resource: schema.GroupResource{Resource: "*"}, Target: Shape{Operation: admission.Update, Subresource: "dynamic"}, Sources: []Shape{{Operation: admission.Create}}}},
			wantErr: "requirements[0].resource.resource: Invalid value",
		},
		{
			name:    "subresource in resource",
			reqs:    []Requirement{{Resource: schema.GroupResource{Resource: "pods/dynamic"}, Target: Shape{Operation: admission.Update}, Sources: []Shape{{Operation: admission.Create}}}},
			wantErr: "requirements[0].resource.resource: Invalid value",
		},
		{
			name:    "wildcard target subresource",
			reqs:    []Requirement{{Resource: podsGR, Target: Shape{Operation: admission.Update, Subresource: "*"}, Sources: []Shape{{Operation: admission.Create}}}},
			wantErr: "requirements[0].target.subresource: Invalid value",
		},
		{
			name:    "wildcard source subresource",
			reqs:    []Requirement{{Resource: podsGR, Target: Shape{Operation: admission.Update, Subresource: "dynamic"}, Sources: []Shape{{Operation: admission.Update, Subresource: "*"}}}},
			wantErr: "requirements[0].sources[0].subresource: Invalid value",
		},
		{
			name: "duplicate target on same resource",
			reqs: []Requirement{
				podDynamicReq,
				{
					Resource: podsGR,
					Target:   Shape{Operation: admission.Update, Subresource: "dynamic"},
					Sources:  []Shape{{Operation: admission.Delete}},
				},
			},
			wantErr: "requirements[1].target: Duplicate value",
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			err := Validate(tc.reqs)
			if err == nil {
				t.Fatal("expected error, got nil")
			}
			if !strings.Contains(err.Error(), tc.wantErr) {
				t.Errorf("err = %v, want substring %q", err, tc.wantErr)
			}
		})
	}
}

func TestCheckNilReceiver(t *testing.T) {
	var gaps *Gaps
	if v := gaps.Check(podTargetAttr()); v != nil {
		t.Fatalf("nil (*Gaps).Check = %+v, want nil", v)
	}
}

func TestForWebhook(t *testing.T) {
	reqs := []Requirement{podDynamicReq, deploymentScaleReq}

	allOps := []admissionregistrationv1.OperationType{admissionregistrationv1.OperationAll}
	createOp := []admissionregistrationv1.OperationType{admissionregistrationv1.Create}
	updateOp := []admissionregistrationv1.OperationType{admissionregistrationv1.Update}
	deleteOp := []admissionregistrationv1.OperationType{admissionregistrationv1.Delete}
	coreGroup := []string{""}
	v1Version := []string{"v1"}

	if gaps := ForWebhook(nil, []admissionregistrationv1.RuleWithOperations{rule(createOp, coreGroup, v1Version, []string{"pods"}, nil)}); gaps != nil {
		t.Fatalf("ForWebhook(nil reqs) = %+v, want nil", gaps)
	}
	if gaps := ForWebhook(reqs, nil); gaps != nil {
		t.Fatalf("ForWebhook(nil rules) = %+v, want nil", gaps)
	}

	tests := []struct {
		name       string
		rules      []admissionregistrationv1.RuleWithOperations
		attr       admission.Attributes
		noGaps     bool
		wantSource *Shape
		// wantViolation, if set, is the whole expected violation, for requests
		// other than podTargetAttr; wantSource is then ignored.
		wantViolation *Violation
	}{
		// --- Hooks that build no gaps (cover all targets or match no source) ---
		{
			name:   "unrelated resource builds no gaps",
			rules:  []admissionregistrationv1.RuleWithOperations{rule(allOps, coreGroup, v1Version, []string{"configmaps"}, nil)},
			noGaps: true,
		},
		{
			name:   "unrelated group builds no gaps",
			rules:  []admissionregistrationv1.RuleWithOperations{rule(allOps, []string{"custom.io"}, v1Version, []string{"pods"}, nil)},
			noGaps: true,
		},
		{
			name:   "target-only rule builds no gaps (directional)",
			rules:  []admissionregistrationv1.RuleWithOperations{rule(updateOp, coreGroup, v1Version, []string{"pods/dynamic"}, nil)},
			noGaps: true,
		},
		{
			name:   "unrelated subresource builds no gaps",
			rules:  []admissionregistrationv1.RuleWithOperations{rule(updateOp, coreGroup, v1Version, []string{"pods/status"}, nil)},
			noGaps: true,
		},
		{
			name:   "non-source operation on pods builds no gaps",
			rules:  []admissionregistrationv1.RuleWithOperations{rule(deleteOp, coreGroup, v1Version, []string{"pods"}, nil)},
			noGaps: true,
		},
		{
			name:   "empty APIVersions matches nothing and builds no gaps",
			rules:  []admissionregistrationv1.RuleWithOperations{rule(createOp, coreGroup, nil, []string{"pods"}, nil)},
			noGaps: true,
		},
		{
			name: "rules covering both source and target build no gaps",
			rules: []admissionregistrationv1.RuleWithOperations{
				rule(updateOp, coreGroup, v1Version, []string{"pods", "pods/dynamic"}, nil),
			},
			noGaps: true,
		},
		{
			name:   "pods/* covers both main resource and subresources so builds no gaps",
			rules:  []admissionregistrationv1.RuleWithOperations{rule(updateOp, coreGroup, v1Version, []string{"pods/*"}, nil)},
			noGaps: true,
		},
		{
			name:   "*/* covers all resources and subresources so builds no gaps",
			rules:  []admissionregistrationv1.RuleWithOperations{rule(allOps, []string{"*"}, []string{"*"}, []string{"*/*"}, nil)},
			noGaps: true,
		},

		// --- Source matching & Resource wildcards ---
		{
			name:       "CREATE pods violates UPDATE pods/dynamic",
			rules:      []admissionregistrationv1.RuleWithOperations{rule(createOp, coreGroup, v1Version, []string{"pods"}, nil)},
			wantSource: &Shape{Operation: admission.Create},
		},
		{
			name:       "UPDATE pods violates UPDATE pods/dynamic",
			rules:      []admissionregistrationv1.RuleWithOperations{rule(updateOp, coreGroup, v1Version, []string{"pods"}, nil)},
			wantSource: &Shape{Operation: admission.Update},
		},
		{
			name:       "UPDATE pods/resize violates UPDATE pods/dynamic",
			rules:      []admissionregistrationv1.RuleWithOperations{rule(updateOp, coreGroup, v1Version, []string{"pods/resize"}, nil)},
			wantSource: &Shape{Operation: admission.Update, Subresource: "resize"},
		},
		{
			name:       "UPDATE pods/ephemeralcontainers violates UPDATE pods/dynamic",
			rules:      []admissionregistrationv1.RuleWithOperations{rule(updateOp, coreGroup, v1Version, []string{"pods/ephemeralcontainers"}, nil)},
			wantSource: &Shape{Operation: admission.Update, Subresource: "ephemeralcontainers"},
		},
		{
			// In rules.Matcher, "*" has sub == "", so it matches main resources
			// (CREATE/UPDATE pods) but not subresources (pods/dynamic).
			name:       "wildcard * matches main resource only so violates subresource target",
			rules:      []admissionregistrationv1.RuleWithOperations{rule(allOps, []string{"*"}, []string{"*"}, []string{"*"}, nil)},
			wantSource: &Shape{Operation: admission.Create},
		},
		{
			name:       "CREATE pods/* covers CREATE pods/dynamic but not UPDATE pods/dynamic",
			rules:      []admissionregistrationv1.RuleWithOperations{rule(createOp, coreGroup, v1Version, []string{"pods/*"}, nil)},
			wantSource: &Shape{Operation: admission.Create},
		},
		{
			name:       "*/resize matches pods/resize source",
			rules:      []admissionregistrationv1.RuleWithOperations{rule(updateOp, coreGroup, v1Version, []string{"*/resize"}, nil)},
			wantSource: &Shape{Operation: admission.Update, Subresource: "resize"},
		},

		// --- Source ordering: first declared source wins ---
		{
			name: "first matching source in requirement order is reported",
			rules: []admissionregistrationv1.RuleWithOperations{
				// Matches ephemeralcontainers (index 3) first in the rule list,
				// and resize (index 2) second; requirement order picks resize.
				rule(updateOp, coreGroup, v1Version, []string{"pods/ephemeralcontainers"}, nil),
				rule(updateOp, coreGroup, v1Version, []string{"pods/resize"}, nil),
			},
			wantSource: &Shape{Operation: admission.Update, Subresource: "resize"},
		},

		// --- Version-agnostic matching ---
		{
			name:       "different APIVersion still matches because coverage is version-agnostic",
			rules:      []admissionregistrationv1.RuleWithOperations{rule(createOp, coreGroup, []string{"v1beta1"}, []string{"pods"}, nil)},
			wantSource: &Shape{Operation: admission.Create},
		},

		{
			name:  "non-v1 request version is checked against v1 rules",
			rules: []admissionregistrationv1.RuleWithOperations{rule(createOp, coreGroup, v1Version, []string{"pods"}, nil)},
			attr:  newAttr(schema.GroupVersionResource{Version: "v1beta1", Resource: "pods"}, "dynamic", admission.Update, "default", "my-pod"),
			wantViolation: &Violation{
				Resource: podsGR,
				Target:   podDynamicReq.Target,
				Source:   Shape{Operation: admission.Create},
			},
		},

		// --- Non-core group ---
		{
			name:  "UPDATE deployments.apps violates UPDATE deployments.apps/scale and keeps the group",
			rules: []admissionregistrationv1.RuleWithOperations{rule(updateOp, []string{"apps"}, v1Version, []string{"deployments"}, nil)},
			attr:  newAttr(schema.GroupVersionResource{Group: "apps", Version: "v1", Resource: "deployments"}, "scale", admission.Update, "default", "my-deploy"),
			wantViolation: &Violation{
				Resource: deploymentsGR,
				Target:   deploymentScaleReq.Target,
				Source:   Shape{Operation: admission.Update},
			},
		},

		// --- Scope ---
		{
			name:       "NamespacedScope rule matches namespaced request",
			rules:      []admissionregistrationv1.RuleWithOperations{rule(createOp, coreGroup, v1Version, []string{"pods"}, ptr.To(admissionregistrationv1.NamespacedScope))},
			wantSource: &Shape{Operation: admission.Create},
		},
		{
			name:       "ClusterScope rule does not match namespaced request",
			rules:      []admissionregistrationv1.RuleWithOperations{rule(createOp, coreGroup, v1Version, []string{"pods"}, ptr.To(admissionregistrationv1.ClusterScope))},
			wantSource: nil,
		},
		{
			name:       "ClusterScope rule matches cluster-scoped request",
			rules:      []admissionregistrationv1.RuleWithOperations{rule(createOp, coreGroup, v1Version, []string{"pods"}, ptr.To(admissionregistrationv1.ClusterScope))},
			attr:       newAttr(schema.GroupVersionResource{Version: "v1", Resource: "pods"}, "dynamic", admission.Update, "", "my-pod"),
			wantSource: &Shape{Operation: admission.Create},
		},

		// --- Non-target / other requirement requests ---
		{
			name:  "pod gaps return nil for source request",
			rules: []admissionregistrationv1.RuleWithOperations{rule(createOp, coreGroup, v1Version, []string{"pods"}, nil)},
			attr:  newAttr(schema.GroupVersionResource{Version: "v1", Resource: "pods"}, "", admission.Create, "default", "my-pod"),
		},
		{
			name:  "pod gaps return nil for deployment target request",
			rules: []admissionregistrationv1.RuleWithOperations{rule(createOp, coreGroup, v1Version, []string{"pods"}, nil)},
			attr:  newAttr(schema.GroupVersionResource{Group: "apps", Version: "v1", Resource: "deployments"}, "scale", admission.Update, "default", "my-deploy"),
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			gaps := ForWebhook(reqs, tc.rules)
			if tc.noGaps {
				if gaps != nil {
					t.Fatalf("ForWebhook() = %+v, want nil gaps", gaps)
				}
				return
			}
			if gaps == nil {
				t.Fatal("ForWebhook() = nil, want non-nil gaps")
			}
			attr := tc.attr
			if attr == nil {
				attr = podTargetAttr()
			}
			got := gaps.Check(attr)
			want := tc.wantViolation
			if want == nil && tc.wantSource != nil {
				want = &Violation{
					Resource: podsGR,
					Target:   podDynamicReq.Target,
					Source:   *tc.wantSource,
				}
			}
			if want == nil {
				if got != nil {
					t.Fatalf("Check() = %+v, want nil", got)
				}
				return
			}
			if got == nil || *got != *want {
				t.Errorf("Check() = %+v, want %+v", got, want)
			}
		})
	}

	t.Run("namespaces resource is always treated as cluster-scoped", func(t *testing.T) {
		nsReq := []Requirement{{
			Resource: namespacesGR,
			Target:   Shape{Operation: admission.Update, Subresource: "finalize"},
			Sources:  []Shape{{Operation: admission.Update}},
		}}
		// NamespacedScope never matches v1/namespaces in rules.Matcher.
		if gaps := ForWebhook(nsReq, []admissionregistrationv1.RuleWithOperations{
			rule(updateOp, coreGroup, v1Version, []string{"namespaces"}, ptr.To(admissionregistrationv1.NamespacedScope)),
		}); gaps != nil {
			t.Fatalf("NamespacedScope on namespaces built gaps %+v, want nil", gaps)
		}
		// ClusterScope matches v1/namespaces even though admission attributes for
		// a namespace request have GetNamespace() set to the namespace's name.
		gaps := ForWebhook(nsReq, []admissionregistrationv1.RuleWithOperations{
			rule(updateOp, coreGroup, v1Version, []string{"namespaces"}, ptr.To(admissionregistrationv1.ClusterScope)),
		})
		if gaps == nil {
			t.Fatal("ClusterScope on namespaces built nil gaps")
		}
		nsAttr := newAttr(schema.GroupVersionResource{Version: "v1", Resource: "namespaces"}, "finalize", admission.Update, "default", "default")
		got := gaps.Check(nsAttr)
		want := &Violation{
			Resource: namespacesGR,
			Target:   Shape{Operation: admission.Update, Subresource: "finalize"},
			Source:   Shape{Operation: admission.Update},
		}
		if got == nil || *got != *want {
			t.Errorf("Check(namespaces) = %+v, want %+v", got, want)
		}
	})
}

func TestForPolicyBinding(t *testing.T) {
	reqs := []Requirement{podDynamicReq, deploymentScaleReq}

	allOps := []admissionregistrationv1.OperationType{admissionregistrationv1.OperationAll}
	createOp := []admissionregistrationv1.OperationType{admissionregistrationv1.Create}
	updateOp := []admissionregistrationv1.OperationType{admissionregistrationv1.Update}
	deleteOp := []admissionregistrationv1.OperationType{admissionregistrationv1.Delete}
	coreGroup := []string{""}
	v1Version := []string{"v1"}

	podCreateRule := namedRule(rule(createOp, coreGroup, v1Version, []string{"pods"}, nil))
	podUpdateRule := namedRule(rule(updateOp, coreGroup, v1Version, []string{"pods"}, nil))
	podAllSubRule := namedRule(rule(allOps, coreGroup, v1Version, []string{"pods/*"}, nil))
	podResizeRule := namedRule(rule(updateOp, coreGroup, v1Version, []string{"pods/resize"}, nil))
	podDynamicRule := namedRule(rule(updateOp, coreGroup, v1Version, []string{"pods/dynamic"}, nil))

	if gaps := ForPolicyBinding(nil, &admissionregistrationv1.MatchResources{}, nil); gaps != nil {
		t.Fatalf("ForPolicyBinding(nil reqs) = %+v, want nil", gaps)
	}

	tests := []struct {
		name       string
		definition *admissionregistrationv1.MatchResources
		binding    *admissionregistrationv1.MatchResources
		attr       admission.Attributes
		noGaps     bool
		wantSource *Shape
	}{
		// --- Hooks that build no gaps ---
		{
			name:       "nil definition builds no gaps",
			definition: nil,
			noGaps:     true,
		},
		{
			name: "definition on unrelated resource builds no gaps",
			definition: &admissionregistrationv1.MatchResources{
				ResourceRules: []admissionregistrationv1.NamedRuleWithOperations{
					namedRule(rule(allOps, coreGroup, v1Version, []string{"configmaps"}, nil)),
				},
			},
			noGaps: true,
		},
		{
			name:       "binding on unrelated resource builds no gaps even when definition is wildcard",
			definition: &admissionregistrationv1.MatchResources{},
			binding: &admissionregistrationv1.MatchResources{
				ResourceRules: []admissionregistrationv1.NamedRuleWithOperations{
					namedRule(rule(allOps, coreGroup, v1Version, []string{"configmaps"}, nil)),
				},
			},
			noGaps: true,
		},
		{
			name: "definition on non-source operation builds no gaps",
			definition: &admissionregistrationv1.MatchResources{
				ResourceRules: []admissionregistrationv1.NamedRuleWithOperations{
					namedRule(rule(deleteOp, coreGroup, v1Version, []string{"pods"}, nil)),
				},
			},
			noGaps: true,
		},
		{
			name: "definition and binding covering both source and target build no gaps",
			definition: &admissionregistrationv1.MatchResources{
				ResourceRules: []admissionregistrationv1.NamedRuleWithOperations{podAllSubRule},
			},
			binding: nil,
			noGaps:  true,
		},
		{
			// In policy matching, empty ResourceRules means "match everything"
			// (including pods/dynamic), so when both definition and binding have
			// empty ResourceRules, the pair already covers the target!
			name:       "empty ResourceRules on both definition and binding already covers target",
			definition: &admissionregistrationv1.MatchResources{},
			binding:    nil,
			noGaps:     true,
		},
		{
			name: "disjoint sources between definition and binding build no gaps",
			definition: &admissionregistrationv1.MatchResources{
				ResourceRules: []admissionregistrationv1.NamedRuleWithOperations{podCreateRule},
			},
			binding: &admissionregistrationv1.MatchResources{
				ResourceRules: []admissionregistrationv1.NamedRuleWithOperations{podResizeRule},
			},
			noGaps: true,
		},
		{
			name: "disjoint resourceNames between definition and binding build no gaps",
			definition: &admissionregistrationv1.MatchResources{
				ResourceRules: []admissionregistrationv1.NamedRuleWithOperations{
					namedRule(rule(createOp, coreGroup, v1Version, []string{"pods"}, nil), "pod-a"),
				},
			},
			binding: &admissionregistrationv1.MatchResources{
				ResourceRules: []admissionregistrationv1.NamedRuleWithOperations{
					namedRule(rule(createOp, coreGroup, v1Version, []string{"pods"}, nil), "pod-b"),
				},
			},
			noGaps: true,
		},
		{
			name: "definition ExcludeResourceRules matching target opts out and builds no gaps",
			definition: &admissionregistrationv1.MatchResources{
				ResourceRules:        []admissionregistrationv1.NamedRuleWithOperations{podCreateRule},
				ExcludeResourceRules: []admissionregistrationv1.NamedRuleWithOperations{podDynamicRule},
			},
			binding: nil,
			noGaps:  true,
		},
		{
			name: "binding ExcludeResourceRules matching target opts out and builds no gaps",
			definition: &admissionregistrationv1.MatchResources{
				ResourceRules: []admissionregistrationv1.NamedRuleWithOperations{podCreateRule},
			},
			binding: &admissionregistrationv1.MatchResources{
				ExcludeResourceRules: []admissionregistrationv1.NamedRuleWithOperations{podDynamicRule},
			},
			noGaps: true,
		},
		{
			name: "ExcludeResourceRules excluding the only matched source builds no gaps",
			definition: &admissionregistrationv1.MatchResources{
				ResourceRules:        []admissionregistrationv1.NamedRuleWithOperations{podCreateRule},
				ExcludeResourceRules: []admissionregistrationv1.NamedRuleWithOperations{podCreateRule},
			},
			binding: nil,
			noGaps:  true,
		},

		// --- Violations ---
		{
			name: "definition intercepts CREATE pods, nil binding",
			definition: &admissionregistrationv1.MatchResources{
				ResourceRules: []admissionregistrationv1.NamedRuleWithOperations{podCreateRule},
			},
			binding:    nil,
			wantSource: &Shape{Operation: admission.Create},
		},
		{
			// An empty MatchResources{} has no ResourceRules, which in
			// policy/matching means "match everything" — so a binding with
			// only selectors still inherits the definition's violation.
			name: "definition intercepts CREATE pods, binding has empty ResourceRules",
			definition: &admissionregistrationv1.MatchResources{
				ResourceRules: []admissionregistrationv1.NamedRuleWithOperations{podCreateRule},
			},
			binding:    &admissionregistrationv1.MatchResources{},
			wantSource: &Shape{Operation: admission.Create},
		},
		{
			name:       "definition has empty ResourceRules, binding narrows to CREATE pods",
			definition: &admissionregistrationv1.MatchResources{},
			binding: &admissionregistrationv1.MatchResources{
				ResourceRules: []admissionregistrationv1.NamedRuleWithOperations{podCreateRule},
			},
			wantSource: &Shape{Operation: admission.Create},
		},
		{
			// Definition covers pods/* (including target), but binding narrows
			// to CREATE pods only, so the pair doesn't cover the target.
			name: "definition covers pods/*, binding narrows to CREATE pods",
			definition: &admissionregistrationv1.MatchResources{
				ResourceRules: []admissionregistrationv1.NamedRuleWithOperations{podAllSubRule},
			},
			binding: &admissionregistrationv1.MatchResources{
				ResourceRules: []admissionregistrationv1.NamedRuleWithOperations{podCreateRule},
			},
			wantSource: &Shape{Operation: admission.Create},
		},
		{
			name: "intersection of definition and binding picks the shared source",
			definition: &admissionregistrationv1.MatchResources{
				ResourceRules: []admissionregistrationv1.NamedRuleWithOperations{podCreateRule, podResizeRule},
			},
			binding: &admissionregistrationv1.MatchResources{
				ResourceRules: []admissionregistrationv1.NamedRuleWithOperations{podResizeRule},
			},
			wantSource: &Shape{Operation: admission.Update, Subresource: "resize"},
		},
		{
			name: "ExcludeResourceRules excluding CREATE pods falls through to UPDATE pods",
			definition: &admissionregistrationv1.MatchResources{
				ResourceRules:        []admissionregistrationv1.NamedRuleWithOperations{podCreateRule, podUpdateRule},
				ExcludeResourceRules: []admissionregistrationv1.NamedRuleWithOperations{podCreateRule},
			},
			binding:    nil,
			wantSource: &Shape{Operation: admission.Update},
		},
		{
			name: "ExcludeResourceRules on unrelated subresource does not opt out",
			definition: &admissionregistrationv1.MatchResources{
				ResourceRules: []admissionregistrationv1.NamedRuleWithOperations{podCreateRule},
				ExcludeResourceRules: []admissionregistrationv1.NamedRuleWithOperations{
					namedRule(rule(updateOp, coreGroup, v1Version, []string{"pods/status"}, nil)),
				},
			},
			binding:    nil,
			wantSource: &Shape{Operation: admission.Create},
		},
		{
			name: "different APIVersion on ResourceRules still matches (version-agnostic)",
			definition: &admissionregistrationv1.MatchResources{
				ResourceRules: []admissionregistrationv1.NamedRuleWithOperations{
					namedRule(rule(createOp, coreGroup, []string{"v1beta1"}, []string{"pods"}, nil)),
				},
			},
			binding:    nil,
			wantSource: &Shape{Operation: admission.Create},
		},
		{
			name: "different APIVersion on ExcludeResourceRules still opts out (version-agnostic)",
			definition: &admissionregistrationv1.MatchResources{
				ResourceRules: []admissionregistrationv1.NamedRuleWithOperations{podCreateRule},
				ExcludeResourceRules: []admissionregistrationv1.NamedRuleWithOperations{
					namedRule(rule(updateOp, coreGroup, []string{"v1beta1"}, []string{"pods/dynamic"}, nil)),
				},
			},
			binding: nil,
			noGaps:  true,
		},

		// --- ResourceNames ---
		{
			name: "ResourceNames on definition matches named pod",
			definition: &admissionregistrationv1.MatchResources{
				ResourceRules: []admissionregistrationv1.NamedRuleWithOperations{
					namedRule(rule(createOp, coreGroup, v1Version, []string{"pods"}, nil), "my-pod"),
				},
			},
			binding:    nil,
			wantSource: &Shape{Operation: admission.Create},
		},
		{
			name: "ResourceNames on definition does not match other pod",
			definition: &admissionregistrationv1.MatchResources{
				ResourceRules: []admissionregistrationv1.NamedRuleWithOperations{
					namedRule(rule(createOp, coreGroup, v1Version, []string{"pods"}, nil), "other-pod"),
				},
			},
			binding:    nil,
			wantSource: nil,
		},
		{
			name: "ResourceNames on binding narrows to named pod",
			definition: &admissionregistrationv1.MatchResources{
				ResourceRules: []admissionregistrationv1.NamedRuleWithOperations{podCreateRule},
			},
			binding: &admissionregistrationv1.MatchResources{
				ResourceRules: []admissionregistrationv1.NamedRuleWithOperations{
					namedRule(rule(createOp, coreGroup, v1Version, []string{"pods"}, nil), "other-pod"),
				},
			},
			wantSource: nil,
		},
		{
			name: "ExcludeResourceRules with ResourceNames opts out only for that name (matched)",
			definition: &admissionregistrationv1.MatchResources{
				ResourceRules: []admissionregistrationv1.NamedRuleWithOperations{podCreateRule},
				ExcludeResourceRules: []admissionregistrationv1.NamedRuleWithOperations{
					namedRule(rule(updateOp, coreGroup, v1Version, []string{"pods/dynamic"}, nil), "my-pod"),
				},
			},
			binding:    nil,
			wantSource: nil,
		},
		{
			name: "ExcludeResourceRules with ResourceNames does not opt out for other names",
			definition: &admissionregistrationv1.MatchResources{
				ResourceRules: []admissionregistrationv1.NamedRuleWithOperations{podCreateRule},
				ExcludeResourceRules: []admissionregistrationv1.NamedRuleWithOperations{
					namedRule(rule(updateOp, coreGroup, v1Version, []string{"pods/dynamic"}, nil), "other-pod"),
				},
			},
			binding:    nil,
			wantSource: &Shape{Operation: admission.Create},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			gaps := ForPolicyBinding(reqs, tc.definition, tc.binding)
			if tc.noGaps {
				if gaps != nil {
					t.Fatalf("ForPolicyBinding() = %+v, want nil gaps", gaps)
				}
				return
			}
			if gaps == nil {
				t.Fatal("ForPolicyBinding() = nil, want non-nil gaps")
			}
			attr := tc.attr
			if attr == nil {
				attr = podTargetAttr()
			}
			got := gaps.Check(attr)
			if tc.wantSource == nil {
				if got != nil {
					t.Fatalf("Check() = %+v, want nil", got)
				}
				return
			}
			want := &Violation{
				Resource: podsGR,
				Target:   podDynamicReq.Target,
				Source:   *tc.wantSource,
			}
			if got == nil || *got != *want {
				t.Errorf("Check() = %+v, want %+v", got, want)
			}
		})
	}
}
