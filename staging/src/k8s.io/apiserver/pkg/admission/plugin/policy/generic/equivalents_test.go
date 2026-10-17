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

package generic_test

import (
	"context"
	"strings"
	"testing"

	admissionregistrationv1 "k8s.io/api/admissionregistration/v1"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apiserver/pkg/admission"
	"k8s.io/apiserver/pkg/admission/plugin/policy/generic"
	"k8s.io/apiserver/pkg/admission/plugin/policy/matching"
	"k8s.io/apiserver/pkg/admission/plugin/policy/validating"
	admissiontesting "k8s.io/apiserver/pkg/admission/testing"
	"k8s.io/client-go/kubernetes/fake"
	"k8s.io/utils/ptr"
)

type vapHook = validating.PolicyHook

type testNamespaceLister struct {
	namespaces map[string]*corev1.Namespace
}

func (f testNamespaceLister) List(labels.Selector) ([]*corev1.Namespace, error) { return nil, nil }
func (f testNamespaceLister) Get(name string) (*corev1.Namespace, error) {
	if ns, ok := f.namespaces[name]; ok {
		return ns, nil
	}
	return nil, apierrors.NewNotFound(corev1.Resource("namespaces"), name)
}

func policyRule(resources []string, names ...string) admissionregistrationv1.NamedRuleWithOperations {
	return policyRuleVersion("v1", resources, names...)
}

func policyRuleVersion(version string, resources []string, names ...string) admissionregistrationv1.NamedRuleWithOperations {
	return admissionregistrationv1.NamedRuleWithOperations{
		ResourceNames: names,
		RuleWithOperations: admissionregistrationv1.RuleWithOperations{
			Operations: []admissionregistrationv1.OperationType{admissionregistrationv1.OperationAll},
			Rule:       admissionregistrationv1.Rule{APIGroups: []string{"example.io"}, APIVersions: []string{version}, Resources: resources, Scope: ptr.To(admissionregistrationv1.AllScopes)},
		},
	}
}

func matchResources(rules ...admissionregistrationv1.NamedRuleWithOperations) *admissionregistrationv1.MatchResources {
	return &admissionregistrationv1.MatchResources{
		NamespaceSelector: &metav1.LabelSelector{},
		ObjectSelector:    &metav1.LabelSelector{},
		MatchPolicy:       ptr.To(admissionregistrationv1.Equivalent),
		ResourceRules:     rules,
	}
}

func testPolicy(name string, mr *admissionregistrationv1.MatchResources) *admissionregistrationv1.ValidatingAdmissionPolicy {
	return &admissionregistrationv1.ValidatingAdmissionPolicy{
		ObjectMeta: metav1.ObjectMeta{Name: name},
		Spec:       admissionregistrationv1.ValidatingAdmissionPolicySpec{MatchConstraints: mr},
	}
}

func testBinding(name string, mr *admissionregistrationv1.MatchResources) *admissionregistrationv1.ValidatingAdmissionPolicyBinding {
	return &admissionregistrationv1.ValidatingAdmissionPolicyBinding{
		ObjectMeta: metav1.ObjectMeta{Name: name},
		Spec:       admissionregistrationv1.ValidatingAdmissionPolicyBindingSpec{MatchResources: mr},
	}
}

// single returns a policy "p" with a single binding "b".
func single(policy, binding *admissionregistrationv1.MatchResources) []vapHook {
	return []vapHook{{Policy: testPolicy("p", policy), Bindings: []*admissionregistrationv1.ValidatingAdmissionPolicyBinding{testBinding("b", binding)}}}
}

func testWidgetTurboUpdate(ns string) admission.Attributes {
	obj := &corev1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "w", Namespace: ns}}
	return admission.NewAttributesRecord(obj, obj.DeepCopy(), schema.GroupVersionKind{Group: "example.io", Version: "v1", Kind: "Widget"}, ns, "w",
		schema.GroupVersionResource{Group: "example.io", Version: "v1", Resource: "widgets"}, "turbo", admission.Update, &metav1.UpdateOptions{}, false, nil)
}

func TestCheckAdmissionEquivalents(t *testing.T) {
	matcher := generic.NewPolicyMatcher(matching.NewMatcher(testNamespaceLister{namespaces: map[string]*corev1.Namespace{
		"ns": {ObjectMeta: metav1.ObjectMeta{Name: "ns", Labels: map[string]string{"env": "prod"}}},
	}}, fake.NewClientset()))
	mapper := runtime.NewEquivalentResourceRegistry()
	for _, version := range []string{"v1", "v2"} {
		for _, sub := range []string{"", "turbo", "resize"} {
			mapper.RegisterKindFor(schema.GroupVersionResource{Group: "example.io", Version: version, Resource: "widgets"}, sub, schema.GroupVersionKind{Group: "example.io", Version: version, Kind: "Widget"})
		}
	}
	// Targets: widgets (UPDATE), widgets/resize (UPDATE).
	o := admissiontesting.ObjectInterfacesWithEquivalents{
		ObjectInterfaces: &admission.RuntimeObjectInterfaces{EquivalentResourceMapper: mapper},
		Equivalents:      []admission.Equivalent{{Subresource: ""}, {Subresource: "resize"}},
	}

	covering := matchResources(policyRule([]string{"widgets", "widgets/turbo"}))
	widgetsOnly := matchResources(policyRule([]string{"widgets"}))
	nilSelectors := &admissionregistrationv1.MatchResources{ResourceRules: []admissionregistrationv1.NamedRuleWithOperations{policyRule([]string{"widgets"})}}
	excludeTurbo := []admissionregistrationv1.NamedRuleWithOperations{policyRule([]string{"widgets/turbo"})}
	inEnv := func(env string) *admissionregistrationv1.MatchResources {
		return &admissionregistrationv1.MatchResources{
			NamespaceSelector: &metav1.LabelSelector{MatchLabels: map[string]string{"env": env}},
			ObjectSelector:    &metav1.LabelSelector{},
		}
	}

	violation := func(policy, binding, target string) string {
		return `ValidatingAdmissionPolicy "` + policy + `" (binding "` + binding + `") applies to ` + target + ` but not widgets/turbo (UPDATE)`
	}

	tests := []struct {
		name      string
		hooks     []vapHook
		namespace string

		wantViolations []string
	}{
		{
			name:  "covered by the policy, unscoped binding",
			hooks: single(covering, nil),
		},
		{
			name:           "policy applies to the base resource only",
			hooks:          single(widgetsOnly, nil),
			wantViolations: []string{violation("p", "b", "widgets (UPDATE)")},
		},
		{
			name:           "policy covers, binding narrows to the base resource",
			hooks:          single(matchResources(policyRule([]string{"*/*", "*"})), widgetsOnly),
			wantViolations: []string{violation("p", "b", "widgets (UPDATE)")},
		},
		{
			name: "binding excludes the source only",
			hooks: single(covering, &admissionregistrationv1.MatchResources{
				NamespaceSelector:    &metav1.LabelSelector{},
				ObjectSelector:       &metav1.LabelSelector{},
				ExcludeResourceRules: excludeTurbo,
			}),
			wantViolations: []string{violation("p", "b", "widgets (UPDATE)")},
		},
		{
			name: "policy excludes the source only",
			hooks: single(&admissionregistrationv1.MatchResources{
				NamespaceSelector:    &metav1.LabelSelector{},
				ObjectSelector:       &metav1.LabelSelector{},
				MatchPolicy:          ptr.To(admissionregistrationv1.Equivalent),
				ResourceRules:        covering.ResourceRules,
				ExcludeResourceRules: excludeTurbo,
			}, nil),
			wantViolations: []string{violation("p", "b", "widgets (UPDATE)")},
		},
		{
			name:           "matchPolicy Equivalent matches the target in another version",
			hooks:          single(matchResources(policyRuleVersion("v2", []string{"widgets/resize"})), nil),
			wantViolations: []string{violation("p", "b", "widgets/resize (UPDATE)")},
		},
		{
			name:  "policy without bindings is ignored",
			hooks: []vapHook{{Policy: testPolicy("p", widgetsOnly)}},
		},
		{
			name:  "resourceNames exclude the request",
			hooks: single(matchResources(policyRule([]string{"widgets"}, "other")), nil),
		},
		{
			name:           "resourceNames include the request",
			hooks:          single(matchResources(policyRule([]string{"widgets"}, "w")), nil),
			wantViolations: []string{violation("p", "b", "widgets (UPDATE)")},
		},
		{
			name:  "binding namespaceSelector does not match",
			hooks: single(widgetsOnly, inEnv("dev")),
		},
		{
			name:           "binding namespaceSelector error fails closed",
			namespace:      "missing",
			hooks:          single(widgetsOnly, inEnv("prod")),
			wantViolations: []string{violation("p", "b", "widgets (UPDATE)") + ` (match criteria could not be evaluated: namespaces "missing" not found)`},
		},
		{
			// The policy's selector error must not hide that the binding cleanly excludes the
			// request but not the target.
			name:      "policy namespaceSelector error, binding narrows to the base resource",
			namespace: "missing",
			hooks: []vapHook{{
				Policy: &admissionregistrationv1.ValidatingAdmissionPolicy{
					ObjectMeta: metav1.ObjectMeta{Name: "p"},
					Spec: admissionregistrationv1.ValidatingAdmissionPolicySpec{
						FailurePolicy: ptr.To(admissionregistrationv1.Ignore),
						MatchConstraints: &admissionregistrationv1.MatchResources{
							NamespaceSelector: &metav1.LabelSelector{MatchLabels: map[string]string{"env": "prod"}},
							ObjectSelector:    &metav1.LabelSelector{},
							MatchPolicy:       ptr.To(admissionregistrationv1.Equivalent),
							ResourceRules:     []admissionregistrationv1.NamedRuleWithOperations{policyRule([]string{"*/*"})},
						},
					},
				},
				Bindings: []*admissionregistrationv1.ValidatingAdmissionPolicyBinding{testBinding("b", widgetsOnly)},
			}},
			wantViolations: []string{violation("p", "b", "widgets (UPDATE)") + ` (match criteria could not be evaluated: namespaces "missing" not found)`},
		},
		{
			// Dispatch hits the same error for the request and applies the failure policy.
			name:  "nil binding selectors are left to dispatch",
			hooks: single(covering, nilSelectors),
		},
		{
			// The binding would error, but the definition cleanly matches no target, which makes
			// the pair a clean non-match.
			name:  "a definition non-match hides binding errors",
			hooks: single(matchResources(policyRule([]string{"widgets/turbo"})), nilSelectors),
		},
		{
			// Dispatch hits the same error for the request and applies the failure policy.
			name:  "policy without match constraints is left to dispatch",
			hooks: single(nil, nil),
		},
		{
			name: "each violating binding is reported, in order",
			hooks: []vapHook{
				{Policy: testPolicy("p1", matchResources(policyRule([]string{"*", "*/*"}))), Bindings: []*admissionregistrationv1.ValidatingAdmissionPolicyBinding{
					testBinding("b1", nil),
					testBinding("b2", matchResources(policyRule([]string{"widgets/resize"}))),
				}},
				{Policy: testPolicy("p2", widgetsOnly), Bindings: []*admissionregistrationv1.ValidatingAdmissionPolicyBinding{testBinding("b3", nil)}},
			},
			wantViolations: []string{
				violation("p1", "b2", "widgets/resize (UPDATE)"),
				violation("p2", "b3", "widgets (UPDATE)"),
			},
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			ns := tc.namespace
			if ns == "" {
				ns = "ns"
			}
			err := generic.CheckAdmissionEquivalents(context.Background(), "ValidatingAdmissionPolicy", testWidgetTurboUpdate(ns), o, matcher, tc.hooks,
				validating.NewValidatingAdmissionPolicyAccessor, validating.NewValidatingAdmissionPolicyBindingAccessor)

			if len(tc.wantViolations) == 0 {
				if err != nil {
					t.Fatalf("unexpected error: %v", err)
				}
				return
			}
			if err == nil {
				t.Fatalf("expected violations %q, got none", tc.wantViolations)
			}
			if want := ": " + strings.Join(tc.wantViolations, "; ") + ". Add"; !strings.Contains(err.Error(), want) {
				t.Errorf("expected exactly the violations\n\t%q\ngot\n\t%q", tc.wantViolations, err.Error())
			}
		})
	}
}
