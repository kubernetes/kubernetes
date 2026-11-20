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
	"testing"

	"github.com/stretchr/testify/require"

	admissionregistrationv1 "k8s.io/api/admissionregistration/v1"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apiserver/pkg/admission"
	"k8s.io/apiserver/pkg/admission/coverage"
	"k8s.io/apiserver/pkg/admission/plugin/policy/matching"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes/fake"
)

// stubPolicy and stubBinding serve as both the policy and binding types and
// their accessors.
type stubPolicy struct {
	metav1.TypeMeta
	matchConstraints *admissionregistrationv1.MatchResources
}

func (p *stubPolicy) DeepCopyObject() runtime.Object                   { c := *p; return &c }
func (p *stubPolicy) GetName() string                                  { return "policy" }
func (p *stubPolicy) GetNamespace() string                             { return "" }
func (p *stubPolicy) GetParamKind() *admissionregistrationv1.ParamKind { return nil }
func (p *stubPolicy) GetMatchConstraints() *admissionregistrationv1.MatchResources {
	return p.matchConstraints
}
func (p *stubPolicy) GetFailurePolicy() *admissionregistrationv1.FailurePolicyType { return nil }

type stubBinding struct {
	metav1.TypeMeta
	matchResources *admissionregistrationv1.MatchResources
}

func (b *stubBinding) DeepCopyObject() runtime.Object { c := *b; return &c }
func (b *stubBinding) GetName() string                { return "binding" }
func (b *stubBinding) GetNamespace() string           { return "" }
func (b *stubBinding) GetPolicyName() types.NamespacedName {
	return types.NamespacedName{Name: "policy"}
}
func (b *stubBinding) GetParamRef() *admissionregistrationv1.ParamRef { return nil }
func (b *stubBinding) GetMatchResources() *admissionregistrationv1.MatchResources {
	return b.matchResources
}

func TestCoverageChecker(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	client := fake.NewClientset(&corev1.Namespace{ObjectMeta: metav1.ObjectMeta{Name: "default", Labels: map[string]string{"env": "test"}}})
	informerFactory := informers.NewSharedInformerFactory(client, 0)
	m := matching.NewMatcher(informerFactory.Core().V1().Namespaces().Lister(), client)
	informerFactory.Start(ctx.Done())
	informerFactory.WaitForCacheSync(ctx.Done())
	checker := NewCoverageChecker[*stubPolicy, *stubBinding, Evaluator](NewPolicyMatcher(m),
		func(p *stubPolicy) PolicyAccessor { return p }, func(b *stubBinding) BindingAccessor { return b })

	pods := schema.GroupResource{Resource: "pods"}
	q := coverage.Requirement{
		Resource: pods,
		Target:   coverage.Shape{Operation: admission.Update, Subresource: "dynamic"},
		Sources:  []coverage.Shape{{Operation: admission.Create}},
	}
	rules := func(op admissionregistrationv1.OperationType, resource string) *admissionregistrationv1.MatchResources {
		return &admissionregistrationv1.MatchResources{
			NamespaceSelector: &metav1.LabelSelector{},
			ObjectSelector:    &metav1.LabelSelector{},
			ResourceRules: []admissionregistrationv1.NamedRuleWithOperations{{
				RuleWithOperations: admissionregistrationv1.RuleWithOperations{
					Operations: []admissionregistrationv1.OperationType{op},
					Rule:       admissionregistrationv1.Rule{APIGroups: []string{""}, APIVersions: []string{"v1"}, Resources: []string{resource}},
				},
			}},
		}
	}
	createPods := func() *admissionregistrationv1.MatchResources { return rules(admissionregistrationv1.Create, "pods") }
	selectorsOnly := func() *admissionregistrationv1.MatchResources {
		return &admissionregistrationv1.MatchResources{NamespaceSelector: &metav1.LabelSelector{}, ObjectSelector: &metav1.LabelSelector{}}
	}
	withObjectSelector := func(mr *admissionregistrationv1.MatchResources, app string) *admissionregistrationv1.MatchResources {
		mr.ObjectSelector = &metav1.LabelSelector{MatchLabels: map[string]string{"app": app}}
		return mr
	}
	withNamespaceSelector := func(mr *admissionregistrationv1.MatchResources, env string) *admissionregistrationv1.MatchResources {
		mr.NamespaceSelector = &metav1.LabelSelector{MatchLabels: map[string]string{"env": env}}
		return mr
	}
	bogusSelector := func() *metav1.LabelSelector {
		return &metav1.LabelSelector{MatchExpressions: []metav1.LabelSelectorRequirement{{Key: "app", Operator: "Bogus"}}}
	}
	withBogusObjectSelector := func(mr *admissionregistrationv1.MatchResources) *admissionregistrationv1.MatchResources {
		mr.ObjectSelector = bogusSelector()
		return mr
	}
	withBogusNamespaceSelector := func(mr *admissionregistrationv1.MatchResources) *admissionregistrationv1.MatchResources {
		mr.NamespaceSelector = bogusSelector()
		return mr
	}
	// Intercepts both the source and the target.
	createUpdatePodsAndDynamic := func() *admissionregistrationv1.MatchResources {
		mr := createPods()
		mr.ResourceRules = append(mr.ResourceRules, rules(admissionregistrationv1.Update, "pods/dynamic").ResourceRules...)
		return mr
	}
	excludingDynamic := func() *admissionregistrationv1.MatchResources {
		mr := createPods()
		mr.ExcludeResourceRules = rules(admissionregistrationv1.Update, "pods/dynamic").ResourceRules
		return mr
	}
	gaps := coverage.ForPolicyBinding([]coverage.Requirement{q}, createPods(), nil)
	violation := coverage.Violation{Resource: pods, Target: q.Target, Source: q.Sources[0]}

	testCases := []struct {
		name       string
		definition *admissionregistrationv1.MatchResources
		binding    *admissionregistrationv1.MatchResources
		gaps       *coverage.Gaps
		configErr  error
		reinvoke   bool
		want       *coverage.Violation
		wantErr    string
	}{
		{
			name:       "no gaps",
			definition: createPods(),
		},
		{
			name:       "violation",
			definition: createPods(),
			gaps:       gaps,
			want:       &violation,
		},
		{
			name:       "reinvocation is exempt",
			definition: createPods(),
			gaps:       gaps,
			reinvoke:   true,
		},
		{
			name:       "configuration error does not suppress coverage",
			definition: createPods(),
			gaps:       gaps,
			configErr:  errors.New("bad param kind"),
			want:       &violation,
		},
		{
			name:       "pair that applies to the request is not in violation",
			definition: rules(admissionregistrationv1.Update, "pods/dynamic"),
			gaps:       gaps,
		},
		{
			name:       "definition object selector mismatch",
			definition: withObjectSelector(createPods(), "y"),
			gaps:       gaps,
		},
		{
			name:       "definition namespace selector mismatch",
			definition: withNamespaceSelector(createPods(), "prod"),
			gaps:       gaps,
		},
		{
			name:       "definition namespace selector match",
			definition: withNamespaceSelector(createPods(), "test"),
			gaps:       gaps,
			want:       &violation,
		},
		{
			name:       "binding object selector mismatch",
			definition: createPods(),
			binding:    withObjectSelector(selectorsOnly(), "y"),
			gaps:       gaps,
		},
		{
			name:       "binding object selector mismatch once the definition matches",
			definition: rules(admissionregistrationv1.Update, "pods/dynamic"),
			binding:    withObjectSelector(selectorsOnly(), "y"),
			gaps:       gaps,
		},
		{
			name:       "binding with only selectors is checked",
			definition: createPods(),
			binding:    withObjectSelector(selectorsOnly(), "x"),
			gaps:       gaps,
			want:       &violation,
		},
		{
			name:       "definition matching error is left to the dispatcher",
			definition: &admissionregistrationv1.MatchResources{ObjectSelector: &metav1.LabelSelector{}},
			gaps:       gaps,
		},
		{
			name:       "binding with nil selectors matches every object",
			definition: createPods(),
			binding:    &admissionregistrationv1.MatchResources{},
			gaps:       gaps,
			want:       &violation,
		},
		{
			name:       "binding selector error makes coverage undetermined",
			definition: createPods(),
			binding: &admissionregistrationv1.MatchResources{ObjectSelector: &metav1.LabelSelector{
				MatchExpressions: []metav1.LabelSelectorRequirement{{Key: "app", Operator: "Bogus"}},
			}},
			gaps:    gaps,
			want:    &coverage.Violation{Resource: pods, Target: q.Target},
			wantErr: `"Bogus" is not a valid label selector operator`,
		},
		{
			// Matches swallows the selector error because the definition's
			// rules don't match the request; coverage surfaces it.
			name:       "definition selector error makes coverage undetermined",
			definition: withBogusObjectSelector(createPods()),
			gaps:       gaps,
			want:       &coverage.Violation{Resource: pods, Target: q.Target},
			wantErr:    `"Bogus" is not a valid label selector operator`,
		},
		{
			name:       "binding namespace selector mismatch wins over object selector error",
			definition: createPods(),
			binding:    withNamespaceSelector(withBogusObjectSelector(selectorsOnly()), "prod"),
			gaps:       gaps,
		},
		{
			name:       "binding object selector mismatch wins over namespace selector error",
			definition: createPods(),
			binding:    withObjectSelector(withBogusNamespaceSelector(selectorsOnly()), "y"),
			gaps:       gaps,
		},
		{
			// PolicyMatcher rejects nil selectors; coverage treats them as
			// match-all rather than skipping the pair.
			name:       "binding with nil selectors once the definition matches",
			definition: createUpdatePodsAndDynamic(),
			binding:    &admissionregistrationv1.MatchResources{ResourceRules: createPods().ResourceRules},
			gaps:       coverage.ForPolicyBinding([]coverage.Requirement{q}, createUpdatePodsAndDynamic(), &admissionregistrationv1.MatchResources{ResourceRules: createPods().ResourceRules}),
			want:       &violation,
		},
		{
			name:       "configuration error does not suppress a binding-level violation",
			definition: createUpdatePodsAndDynamic(),
			binding:    createPods(),
			gaps:       coverage.ForPolicyBinding([]coverage.Requirement{q}, createUpdatePodsAndDynamic(), createPods()),
			configErr:  errors.New("bad param kind"),
			want:       &violation,
		},
		{
			name:       "excludeResourceRules matching the target opts out",
			definition: excludingDynamic(),
			gaps:       coverage.ForPolicyBinding([]coverage.Requirement{q}, excludingDynamic(), nil),
		},
	}
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			pod := &corev1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "p", Namespace: "default", Labels: map[string]string{"app": "x"}}}
			attr := admission.NewAttributesRecord(pod, pod, corev1.SchemeGroupVersion.WithKind("Pod"), "default", "p",
				corev1.SchemeGroupVersion.WithResource("pods"), "dynamic", admission.Update, &metav1.UpdateOptions{}, false, nil)
			if tc.reinvoke {
				attr.GetReinvocationContext().SetIsReinvoke()
			}
			hook := PolicyHook[*stubPolicy, *stubBinding, Evaluator]{
				Policy:             &stubPolicy{matchConstraints: tc.definition},
				Bindings:           []*stubBinding{{matchResources: tc.binding}},
				CoverageGaps:       []*coverage.Gaps{tc.gaps},
				ConfigurationError: tc.configErr,
			}
			violations := checker.Check(attr, nil, []PolicyHook[*stubPolicy, *stubBinding, Evaluator]{hook})
			if tc.want == nil {
				require.Empty(t, violations)
				return
			}
			require.Len(t, violations, 1)
			v := violations[0]
			require.Same(t, hook.Policy, v.Policy)
			require.Same(t, hook.Bindings[0], v.Binding)
			require.Equal(t, *tc.want, v.Violation)
			if tc.wantErr == "" {
				require.NoError(t, v.Err)
			} else {
				require.ErrorContains(t, v.Err, tc.wantErr)
			}
		})
	}
}
