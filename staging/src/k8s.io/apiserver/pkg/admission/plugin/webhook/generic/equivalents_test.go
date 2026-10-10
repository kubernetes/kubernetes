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
	"errors"
	"fmt"
	"strings"
	"testing"

	v1 "k8s.io/api/admissionregistration/v1"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apiserver/pkg/admission"
	"k8s.io/apiserver/pkg/admission/plugin/webhook"
	"k8s.io/apiserver/pkg/admission/plugin/webhook/predicates/namespace"
	"k8s.io/apiserver/pkg/admission/plugin/webhook/predicates/object"
	admissiontesting "k8s.io/apiserver/pkg/admission/testing"
	"k8s.io/client-go/kubernetes/fake"
	"k8s.io/utils/ptr"
)

// erroringNamespaceLister fails lookups of the "broken" namespace with a non-NotFound error.
type erroringNamespaceLister struct {
	fakeNamespaceLister
}

func (f erroringNamespaceLister) Get(name string) (*corev1.Namespace, error) {
	if name == "broken" {
		return nil, errors.New("namespace cache exploded")
	}
	return f.fakeNamespaceLister.Get(name)
}

type recordingDispatcher struct {
	called bool
}

func (d *recordingDispatcher) Dispatch(context.Context, admission.Attributes, admission.ObjectInterfaces, []webhook.WebhookAccessor) error {
	d.called = true
	return nil
}

type mockReloadableSource struct {
	mockSource
}

func (m *mockReloadableSource) RunReloadLoop(context.Context) {}

// newTestValidatingAccessor applies the API defaults for selectors, which otherwise match nothing
// when nil, and names the webhook after its uid if unnamed.
func newTestValidatingAccessor(uid, configurationName string, h *v1.ValidatingWebhook) webhook.WebhookAccessor {
	h = h.DeepCopy()
	if h.Name == "" {
		h.Name = uid
	}
	if h.NamespaceSelector == nil {
		h.NamespaceSelector = &metav1.LabelSelector{}
	}
	if h.ObjectSelector == nil {
		h.ObjectSelector = &metav1.LabelSelector{}
	}
	return webhook.NewValidatingWebhookAccessor(uid, configurationName, h)
}

func newEquivalentsTestWebhook() *Webhook {
	lister := erroringNamespaceLister{fakeNamespaceLister{namespaces: map[string]*corev1.Namespace{
		"ns":    {ObjectMeta: metav1.ObjectMeta{Name: "ns", Labels: map[string]string{"env": "prod"}}},
		"newns": {ObjectMeta: metav1.ObjectMeta{Name: "newns", Labels: map[string]string{"env": "prod"}}},
	}}}
	return &Webhook{
		Handler:          admission.NewHandler(admission.Create, admission.Update, admission.Delete, admission.Connect),
		pluginName:       "ValidatingAdmissionWebhook",
		namespaceMatcher: &namespace.Matcher{NamespaceLister: lister, Client: fake.NewClientset()},
		objectMatcher:    &object.Matcher{},
	}
}

func newEquivalentsTestInterfaces(eqs []admission.Equivalent) admission.ObjectInterfaces {
	mapper := runtime.NewEquivalentResourceRegistry()
	for _, version := range []string{"v1", "v2"} {
		for _, sub := range []string{"", "turbo", "resize", "extras"} {
			mapper.RegisterKindFor(gvr("example.io", version, "widgets"), sub, gvk("example.io", version, "Widget"))
		}
	}
	return admissiontesting.ObjectInterfacesWithEquivalents{
		ObjectInterfaces: &admission.RuntimeObjectInterfaces{EquivalentResourceMapper: mapper},
		Equivalents:      eqs,
	}
}

// Source: UPDATE widgets/turbo. Targets: widgets (CREATE), widgets (UPDATE), widgets/resize (UPDATE),
// widgets/extras (UPDATE).
var testWidgetEquivalents = []admission.Equivalent{
	{Subresource: "", Operations: []admission.Operation{admission.Create, admission.Update}},
	{Subresource: "resize"},
	{Subresource: "extras"},
}

func widgetTurboUpdate(ns string) admission.Attributes {
	obj := &corev1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "w", Namespace: ns, Labels: map[string]string{"app": "a"}}}
	return admission.NewAttributesRecord(obj, obj.DeepCopy(), gvk("example.io", "v1", "Widget"), ns, "w",
		gvr("example.io", "v1", "widgets"), "turbo", admission.Update, &metav1.UpdateOptions{}, false, nil)
}

func widgetRule(ops []v1.OperationType, resources ...string) v1.RuleWithOperations {
	return widgetRuleVersion("v1", ops, resources...)
}

func widgetRuleVersion(version string, ops []v1.OperationType, resources ...string) v1.RuleWithOperations {
	return v1.RuleWithOperations{
		Operations: ops,
		Rule:       v1.Rule{APIGroups: []string{"example.io"}, APIVersions: []string{version}, Resources: resources, Scope: ptr.To(v1.AllScopes)},
	}
}

var (
	opsAll    = []v1.OperationType{v1.OperationAll}
	opsCreate = []v1.OperationType{v1.Create}
	opsUpdate = []v1.OperationType{v1.Update}
	opsDelete = []v1.OperationType{v1.Delete}
)

func TestWebhookAdmissionEquivalents(t *testing.T) {
	testcases := []struct {
		name      string
		webhook   *v1.ValidatingWebhook
		namespace string

		expectTarget string // "" means no violation
		expectErr    string // additional expected error substring
	}{
		{
			name:    "resource and subresource listed",
			webhook: &v1.ValidatingWebhook{Rules: []v1.RuleWithOperations{widgetRule(opsAll, "widgets", "widgets/turbo")}},
		},
		{
			name:    "resource/* wildcard",
			webhook: &v1.ValidatingWebhook{Rules: []v1.RuleWithOperations{widgetRule(opsAll, "widgets", "widgets/*")}},
		},
		{
			name:    "*/* wildcard",
			webhook: &v1.ValidatingWebhook{Rules: []v1.RuleWithOperations{widgetRule(opsAll, "*/*")}},
		},
		{
			name:         "* matches resources but not subresources",
			webhook:      &v1.ValidatingWebhook{Rules: []v1.RuleWithOperations{widgetRule(opsAll, "*")}},
			expectTarget: "widgets (CREATE)",
		},
		{
			name:         "base resource only",
			webhook:      &v1.ValidatingWebhook{Rules: []v1.RuleWithOperations{widgetRule(opsUpdate, "widgets")}},
			expectTarget: "widgets (UPDATE)",
		},
		{
			name:         "subresource listed, but not for the source operation",
			webhook:      &v1.ValidatingWebhook{Rules: []v1.RuleWithOperations{widgetRule(opsCreate, "widgets", "widgets/turbo")}},
			expectTarget: "widgets (CREATE)",
		},
		{
			name: "operation-only mismatch on the source",
			webhook: &v1.ValidatingWebhook{Rules: []v1.RuleWithOperations{
				widgetRule(opsCreate, "widgets/turbo"),
				widgetRule(opsUpdate, "widgets"),
			}},
			expectTarget: "widgets (UPDATE)",
		},
		{
			name:    "target operation not declared",
			webhook: &v1.ValidatingWebhook{Rules: []v1.RuleWithOperations{widgetRule(opsDelete, "widgets")}},
		},
		{
			name:    "source subresource for another operation only",
			webhook: &v1.ValidatingWebhook{Rules: []v1.RuleWithOperations{widgetRule(opsCreate, "widgets/turbo")}},
		},
		{
			name:         "other subresource target",
			webhook:      &v1.ValidatingWebhook{Rules: []v1.RuleWithOperations{widgetRule(opsAll, "widgets/resize")}},
			expectTarget: "widgets/resize (UPDATE)",
		},
		{
			name:    "unrelated resource",
			webhook: &v1.ValidatingWebhook{Rules: []v1.RuleWithOperations{widgetRule(opsAll, "gadgets", "gadgets/turbo")}},
		},
		{
			name: "objectSelector does not match",
			webhook: &v1.ValidatingWebhook{
				ObjectSelector: &metav1.LabelSelector{MatchLabels: map[string]string{"app": "b"}},
				Rules:          []v1.RuleWithOperations{widgetRule(opsAll, "widgets")},
			},
		},
		{
			name: "objectSelector matches",
			webhook: &v1.ValidatingWebhook{
				ObjectSelector: &metav1.LabelSelector{MatchLabels: map[string]string{"app": "a"}},
				Rules:          []v1.RuleWithOperations{widgetRule(opsAll, "widgets")},
			},
			expectTarget: "widgets (CREATE)",
		},
		{
			name: "namespaceSelector does not match",
			webhook: &v1.ValidatingWebhook{
				NamespaceSelector: &metav1.LabelSelector{MatchLabels: map[string]string{"env": "dev"}},
				Rules:             []v1.RuleWithOperations{widgetRule(opsAll, "widgets")},
			},
		},
		{
			name: "namespaceSelector matches",
			webhook: &v1.ValidatingWebhook{
				NamespaceSelector: &metav1.LabelSelector{MatchLabels: map[string]string{"env": "prod"}},
				Rules:             []v1.RuleWithOperations{widgetRule(opsAll, "widgets")},
			},
			expectTarget: "widgets (CREATE)",
		},
		{
			name: "namespaceSelector error fails closed",
			webhook: &v1.ValidatingWebhook{
				NamespaceSelector: &metav1.LabelSelector{MatchLabels: map[string]string{"env": "prod"}},
				Rules:             []v1.RuleWithOperations{widgetRule(opsAll, "widgets")},
			},
			namespace:    "broken",
			expectTarget: "widgets (CREATE)",
			expectErr:    "match criteria could not be evaluated",
		},
		{
			name: "namespaceSelector error, rules match nothing",
			webhook: &v1.ValidatingWebhook{
				NamespaceSelector: &metav1.LabelSelector{MatchLabels: map[string]string{"env": "prod"}},
				Rules:             []v1.RuleWithOperations{widgetRule(opsAll, "gadgets")},
			},
			namespace: "broken",
		},
		{
			// Dispatch hits the same error for the request and rejects the request.
			name: "namespaceSelector error on a covering hook is left to dispatch",
			webhook: &v1.ValidatingWebhook{
				NamespaceSelector: &metav1.LabelSelector{MatchLabels: map[string]string{"env": "prod"}},
				Rules:             []v1.RuleWithOperations{widgetRule(opsAll, "widgets", "widgets/turbo")},
			},
			namespace: "broken",
		},
		{
			name: "matchPolicy Equivalent matches the target in another version",
			webhook: &v1.ValidatingWebhook{
				MatchPolicy: ptr.To(v1.Equivalent),
				Rules:       []v1.RuleWithOperations{widgetRuleVersion("v2", opsAll, "widgets/resize")},
			},
			expectTarget: "widgets/resize (UPDATE)",
		},
		{
			name: "matchPolicy Exact does not match the target in another version",
			webhook: &v1.ValidatingWebhook{
				MatchPolicy: ptr.To(v1.Exact),
				Rules:       []v1.RuleWithOperations{widgetRuleVersion("v2", opsAll, "widgets/resize")},
			},
		},
		{
			name: "matchConditions do not rescue a non-covering hook",
			webhook: &v1.ValidatingWebhook{
				Rules:           []v1.RuleWithOperations{widgetRule(opsAll, "widgets")},
				MatchConditions: []v1.MatchCondition{{Name: "never", Expression: "false"}},
			},
			expectTarget: "widgets (CREATE)",
		},
		{
			name: "failurePolicy Ignore",
			webhook: &v1.ValidatingWebhook{
				FailurePolicy: ptr.To(v1.Ignore),
				Rules:         []v1.RuleWithOperations{widgetRule(opsAll, "widgets")},
			},
			expectTarget: "widgets (CREATE)",
		},
	}

	for i, tc := range testcases {
		t.Run(tc.name, func(t *testing.T) {
			a := newEquivalentsTestWebhook()
			ns := tc.namespace
			if ns == "" {
				ns = "ns"
			}
			o := newEquivalentsTestInterfaces(testWidgetEquivalents)
			h := newTestValidatingAccessor(fmt.Sprintf("webhook-%d", i), "cfg", tc.webhook)

			expectViolation := ""
			if tc.expectTarget != "" {
				expectViolation = fmt.Sprintf(`ValidatingAdmissionWebhook %q (configuration "cfg") applies to %s but not widgets/turbo (UPDATE)`, h.GetName(), tc.expectTarget)
			}
			err := a.CheckAdmissionEquivalents(context.Background(), widgetTurboUpdate(ns), o, []webhook.WebhookAccessor{h})
			assertCoverageError(t, err, expectViolation, tc.expectErr)
		})
	}
}

func assertCoverageError(t *testing.T, err error, expectViolation, expectSubstring string) {
	t.Helper()
	if expectViolation == "" {
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		return
	}
	if err == nil {
		t.Fatalf("expected coverage error containing %q, got none", expectViolation)
	}
	if !strings.Contains(err.Error(), expectViolation) {
		t.Errorf("expected error containing\n\t%q\ngot\n\t%q", expectViolation, err.Error())
	}
	if expectSubstring != "" && !strings.Contains(err.Error(), expectSubstring) {
		t.Errorf("expected error containing %q, got %q", expectSubstring, err.Error())
	}
}

func TestWebhookAdmissionEquivalentsAggregated(t *testing.T) {
	a := newEquivalentsTestWebhook()
	hooks := []webhook.WebhookAccessor{
		newTestValidatingAccessor("1", "cfg", &v1.ValidatingWebhook{Name: "a.example.io", Rules: []v1.RuleWithOperations{widgetRule(opsAll, "widgets/extras")}}),
		newTestValidatingAccessor("2", "cfg", &v1.ValidatingWebhook{Name: "covering.example.io", Rules: []v1.RuleWithOperations{widgetRule(opsAll, "widgets", "widgets/turbo")}}),
		newTestValidatingAccessor("3", "cfg2", &v1.ValidatingWebhook{Name: "b.example.io", Rules: []v1.RuleWithOperations{widgetRule(opsAll, "widgets")}}),
		newTestValidatingAccessor("4", "cfg", &v1.ValidatingWebhook{Name: "unrelated.example.io", Rules: []v1.RuleWithOperations{widgetRule(opsAll, "gadgets")}}),
	}
	err := a.CheckAdmissionEquivalents(context.Background(), widgetTurboUpdate("ns"), newEquivalentsTestInterfaces(testWidgetEquivalents), hooks)
	if err == nil {
		t.Fatal("expected coverage error")
	}
	expect := []string{
		`ValidatingAdmissionWebhook "a.example.io" (configuration "cfg") applies to widgets/extras (UPDATE) but not widgets/turbo (UPDATE)`,
		`ValidatingAdmissionWebhook "b.example.io" (configuration "cfg2") applies to widgets (CREATE) but not widgets/turbo (UPDATE)`,
	}
	if !strings.Contains(err.Error(), ": "+expect[0]+"; "+expect[1]+". Add") {
		t.Errorf("expected exactly the violations %q in hook order, got %q", expect, err.Error())
	}
}

// Namespace selectors read labels from the request object, not the lister, for CREATE/UPDATE of a
// namespace itself. The check must therefore evaluate selectors for each target separately, or a
// stale lister could let a subresource request skip a hook that would apply to the main resource.
func TestWebhookAdmissionEquivalentsNamespaceSelectorPerTarget(t *testing.T) {
	a := newEquivalentsTestWebhook()
	h := newTestValidatingAccessor("1", "cfg", &v1.ValidatingWebhook{
		Name:              "ns.example.io",
		NamespaceSelector: &metav1.LabelSelector{MatchLabels: map[string]string{"env": "dev"}},
		Rules: []v1.RuleWithOperations{{
			Operations: opsAll,
			Rule:       v1.Rule{APIGroups: []string{""}, APIVersions: []string{"v1"}, Resources: []string{"namespaces"}},
		}},
	})
	// The lister has "newns" labeled env=prod; the request object says env=dev.
	obj := &corev1.Namespace{ObjectMeta: metav1.ObjectMeta{Name: "newns", Labels: map[string]string{"env": "dev"}}}
	attr := admission.NewAttributesRecord(obj, obj.DeepCopy(), corev1.SchemeGroupVersion.WithKind("Namespace"), "newns", "newns",
		corev1.SchemeGroupVersion.WithResource("namespaces"), "turbo", admission.Update, &metav1.UpdateOptions{}, false, nil)
	o := newEquivalentsTestInterfaces([]admission.Equivalent{{Subresource: ""}})

	const expect = `ValidatingAdmissionWebhook "ns.example.io" (configuration "cfg") applies to namespaces (UPDATE) but not namespaces/turbo (UPDATE)`
	assertCoverageError(t, a.CheckAdmissionEquivalents(context.Background(), attr, o, []webhook.WebhookAccessor{h}), expect, "")
}

// Selectors are evaluated before rules, as in ShouldCallHook, so that a hook that definitively
// does not apply is not reported for a rules error.
func TestWebhookAdmissionEquivalentsSelectorMismatchHidesRulesError(t *testing.T) {
	a := newEquivalentsTestWebhook()
	rules := []v1.RuleWithOperations{widgetRuleVersion("v2", opsAll, "widgets")}
	selected := newTestValidatingAccessor("1", "cfg", &v1.ValidatingWebhook{
		MatchPolicy: ptr.To(v1.Equivalent),
		Rules:       rules,
	})
	unselected := newTestValidatingAccessor("2", "cfg", &v1.ValidatingWebhook{
		MatchPolicy:       ptr.To(v1.Equivalent),
		NamespaceSelector: &metav1.LabelSelector{MatchLabels: map[string]string{"env": "dev"}},
		Rules:             rules,
	})
	// v2 widgets has no registered kind, so matching the rules against widgets fails.
	mapper := runtime.NewEquivalentResourceRegistry()
	mapper.RegisterKindFor(gvr("example.io", "v1", "widgets"), "", gvk("example.io", "v1", "Widget"))
	mapper.RegisterKindFor(gvr("example.io", "v2", "widgets"), "", schema.GroupVersionKind{})
	o := admissiontesting.ObjectInterfacesWithEquivalents{
		ObjectInterfaces: &admission.RuntimeObjectInterfaces{EquivalentResourceMapper: mapper},
		Equivalents:      []admission.Equivalent{{Subresource: ""}},
	}
	widgetUpdate := admission.NewAttributesRecord(nil, nil, gvk("example.io", "v1", "Widget"), "ns", "w", gvr("example.io", "v1", "widgets"), "", admission.Update, nil, false, nil)
	if _, err := a.matchHook(selected, widgetUpdate, o); err == nil {
		t.Fatal("expected matching the rules against widgets to fail")
	}
	assertCoverageError(t, a.CheckAdmissionEquivalents(context.Background(), widgetTurboUpdate("ns"), o, []webhook.WebhookAccessor{unselected}), "", "")
}

// Admission configuration resources are only dispatched to static hooks, which must be checked
// too. The dynamic path is covered by the integration test.
func TestWebhookDispatchAdmissionEquivalentsStaticOnly(t *testing.T) {
	d := &recordingDispatcher{}
	a := newEquivalentsTestWebhook()
	a.dispatcher = d
	a.staticSource = &mockReloadableSource{mockSource{webhooks: []webhook.WebhookAccessor{
		newTestValidatingAccessor("s", "guard.static.k8s.io", &v1.ValidatingWebhook{
			Name: "static.example.io",
			Rules: []v1.RuleWithOperations{{
				Operations: opsAll,
				Rule:       v1.Rule{APIGroups: []string{"admissionregistration.k8s.io"}, APIVersions: []string{"v1"}, Resources: []string{"validatingwebhookconfigurations"}},
			}},
		}),
	}, hasSynced: true}}
	obj := &v1.ValidatingWebhookConfiguration{ObjectMeta: metav1.ObjectMeta{Name: "c"}}
	attr := admission.NewAttributesRecord(obj, obj.DeepCopy(), v1.SchemeGroupVersion.WithKind("ValidatingWebhookConfiguration"), "", "c",
		v1.SchemeGroupVersion.WithResource("validatingwebhookconfigurations"), "turbo", admission.Update, &metav1.UpdateOptions{}, false, nil)
	err := a.Dispatch(context.Background(), attr, newEquivalentsTestInterfaces([]admission.Equivalent{{Subresource: ""}}))
	if d.called {
		t.Error("dispatcher must not be called on a coverage violation")
	}
	const expect = `ValidatingAdmissionWebhook "static.example.io" (configuration "guard.static.k8s.io") applies to validatingwebhookconfigurations (UPDATE) but not validatingwebhookconfigurations/turbo (UPDATE)`
	assertCoverageError(t, err, expect, "")
}
