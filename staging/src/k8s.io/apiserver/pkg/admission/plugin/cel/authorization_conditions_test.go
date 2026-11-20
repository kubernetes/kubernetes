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

package cel

import (
	"errors"
	"testing"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apiserver/pkg/admission"
	celconfig "k8s.io/apiserver/pkg/apis/cel"
	"k8s.io/apiserver/pkg/authorization/authorizer"
	"k8s.io/apiserver/pkg/authorization/authorizerfactory"
	apiservercel "k8s.io/apiserver/pkg/cel"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes/fake"
)

var (
	podGVK = schema.GroupVersionKind{Group: "", Version: "v1", Kind: "Pod"}
	podGVR = schema.GroupVersionResource{Group: "", Version: "v1", Resource: "pods"}
)

var defaultNamespace = &corev1.Namespace{
	ObjectMeta: metav1.ObjectMeta{Name: "default", Labels: map[string]string{"env": "prod"}},
}

// newTestAuthorizationConditionsEvaluatorBuilder serves namespaces from the fake client, as
// the informers are never started and the namespace matcher falls back to a live GET.
func newTestAuthorizationConditionsEvaluatorBuilder(t *testing.T, budget int64, authz authorizer.UnconditionalAuthorizer, namespaces ...runtime.Object) authorizer.BuiltinConditionsEvaluatorBuilder {
	t.Helper()
	client := fake.NewClientset(namespaces...)
	b, err := NewAuthorizationConditionsEvaluatorBuilder(informers.NewSharedInformerFactory(client, 0), client, authz, budget)
	if err != nil {
		t.Fatalf("NewAuthorizationConditionsEvaluatorBuilder failed: %v", err)
	}
	return b
}

func versioned(t *testing.T, attrs admission.Attributes) *admission.VersionedAttributes {
	t.Helper()
	// The test objects are already in the requested GVK, so no ObjectInterfaces are needed for conversion.
	va, err := admission.NewVersionedAttributes(attrs, attrs.GetKind(), nil)
	if err != nil {
		t.Fatalf("NewVersionedAttributes failed: %v", err)
	}
	return va
}

// evalResultString converts a ConditionEvaluationResult to a human-readable string
// for use in test assertions.
func evalResultString(r authorizer.ConditionEvaluationResult) string {
	switch {
	case r.IsUnevaluatable():
		return "Unevaluatable"
	case r.IsTrue():
		return "True"
	case r.IsFalse():
		return "False"
	case r.IsError():
		return "Error"
	default:
		return "Unknown"
	}
}

func podCreateAttributes() admission.Attributes {
	object := &corev1.Pod{
		TypeMeta: metav1.TypeMeta{Kind: "Pod", APIVersion: "v1"},
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-pod",
			Namespace: "default",
			Labels:    map[string]string{"app": "web"},
		},
		Spec: corev1.PodSpec{
			NodeName: "node1",
		},
	}
	return admission.NewAttributesRecord(object, nil, podGVK, "default", "test-pod", podGVR, "", admission.Create, &metav1.CreateOptions{}, false, nil)
}

func podUpdateAttributes() admission.Attributes {
	oldObject := &corev1.Pod{
		TypeMeta: metav1.TypeMeta{Kind: "Pod", APIVersion: "v1"},
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-pod",
			Namespace: "default",
			Labels:    map[string]string{"app": "web"},
		},
		Spec: corev1.PodSpec{
			NodeName: "node1",
		},
	}
	newObject := &corev1.Pod{
		TypeMeta: metav1.TypeMeta{Kind: "Pod", APIVersion: "v1"},
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-pod",
			Namespace: "default",
			Labels:    map[string]string{"app": "web", "version": "v2"},
		},
		Spec: corev1.PodSpec{
			NodeName: "node2",
		},
	}
	return admission.NewAttributesRecord(newObject, oldObject, podGVK, "default", "test-pod", podGVR, "", admission.Update, &metav1.UpdateOptions{}, false, nil)
}

// celCond creates a CEL-typed condition for use in tests.
func celCond(id, expr string) authorizer.GenericCondition {
	return authorizer.GenericCondition{
		ID:        id,
		Type:      authorizer.ConditionTypeAuthorizationCEL,
		Condition: expr,
	}
}

// TestAuthorizationConditionsEvaluator_EvaluateCondition tests the EvaluateCondition
// method in a table-driven fashion, covering all skip/unevaluatable cases, successful
// condition evaluations, and error cases.
func TestAuthorizationConditionsEvaluator_EvaluateCondition(t *testing.T) {
	tests := []struct {
		name       string
		condition  authorizer.Condition
		attrs      admission.Attributes
		authz      authorizer.Authorizer
		costBudget int64
		wantResult string // "True", "False", "Error", "Unevaluatable"
	}{
		// --- Skip / Unevaluatable cases ---
		{
			name: "wrong condition type returns Unevaluatable",
			condition: authorizer.GenericCondition{
				Type:      "some.io/unsupported-type",
				Condition: "true",
			},
			attrs:      podCreateAttributes(),
			wantResult: "Unevaluatable",
		},
		{
			name: "empty condition string returns Unevaluatable",
			condition: authorizer.GenericCondition{
				Type:      authorizer.ConditionTypeAuthorizationCEL,
				Condition: " ",
			},
			attrs:      podCreateAttributes(),
			wantResult: "Unevaluatable",
		},

		// --- True evaluations ---
		{
			name:       "allow condition: labels.app == 'web' (true)",
			condition:  celCond("c", "'app' in object.metadata.labels && object.metadata.labels.app == 'web'"),
			attrs:      podCreateAttributes(),
			wantResult: "True",
		},
		{
			name:       "deny condition: has(object.metadata.labels.app) (true)",
			condition:  celCond("c", "has(object.metadata.labels.app)"),
			attrs:      podCreateAttributes(),
			wantResult: "True",
		},
		{
			name:       "request.kind.kind == 'Pod' (true)",
			condition:  celCond("c", "request.kind.kind == 'Pod'"),
			attrs:      podCreateAttributes(),
			wantResult: "True",
		},
		{
			name:       "request.namespace == 'default' (true)",
			condition:  celCond("c", "request.namespace == 'default'"),
			attrs:      podCreateAttributes(),
			wantResult: "True",
		},
		{
			name:       "request.operation == 'CREATE' (true)",
			condition:  celCond("c", "request.operation == 'CREATE'"),
			attrs:      podCreateAttributes(),
			wantResult: "True",
		},
		{
			name:       "oldObject == null on create (true)",
			condition:  celCond("c", "oldObject == null"),
			attrs:      podCreateAttributes(),
			wantResult: "True",
		},
		{
			name:       "object != null and oldObject != null on update (true)",
			condition:  celCond("c", "oldObject != null && object != null"),
			attrs:      podUpdateAttributes(),
			wantResult: "True",
		},
		{
			name:       "nodeName changed on update (true)",
			condition:  celCond("c", "object.spec.nodeName != oldObject.spec.nodeName"),
			attrs:      podUpdateAttributes(),
			wantResult: "True",
		},
		{
			name:       "namespaceObject labels are available (true)",
			condition:  celCond("c", "namespaceObject.metadata.labels.env == 'prod'"),
			attrs:      podCreateAttributes(),
			wantResult: "True",
		},
		{
			name:       "authorizer check is allowed (true)",
			condition:  celCond("c", "authorizer.group('').resource('pods').namespace('default').check('get').allowed()"),
			attrs:      podCreateAttributes(),
			authz:      authorizer.AuthorizerFunc(fakeAuthorizer{defaultResult: authorizerResult{decision: authorizer.DecisionAllow}}.Authorize),
			wantResult: "True",
		},

		// --- False evaluations ---
		{
			name:       "allow condition: name == 'notfound' (false)",
			condition:  celCond("c", "has(object.metadata.labels.app) && object.metadata.labels.app == 'notfound'"),
			attrs:      podCreateAttributes(),
			wantResult: "False",
		},
		{
			name:       "deny condition: has(object.metadata.labels.notexistent) (false)",
			condition:  celCond("c", "has(object.metadata.labels.notexistent)"),
			attrs:      podCreateAttributes(),
			wantResult: "False",
		},
		{
			name:       "condition never matches (false)",
			condition:  celCond("c", "object.metadata.name == 'never-matches'"),
			attrs:      podCreateAttributes(),
			wantResult: "False",
		},
		{
			name:       "authorizer check is denied (false)",
			condition:  celCond("c", "authorizer.group('').resource('pods').namespace('default').check('get').allowed()"),
			attrs:      podCreateAttributes(),
			authz:      authorizer.AuthorizerFunc(fakeAuthorizer{defaultResult: authorizerResult{decision: authorizer.DecisionDeny}}.Authorize),
			wantResult: "False",
		},

		// --- Error cases ---
		{
			name:       "invalid CEL expression returns Error",
			condition:  celCond("c", "1 < 'asdf'"),
			attrs:      podCreateAttributes(),
			wantResult: "Error",
		},
		{
			name:       "cost budget exceeded returns Error",
			condition:  celCond("c", "object.spec.nodeName == 'node1'"),
			attrs:      podCreateAttributes(),
			costBudget: 1,
			wantResult: "Error",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			budget := tc.costBudget
			if budget == 0 {
				budget = celconfig.RuntimeCELCostBudget
			}
			if tc.authz == nil {
				tc.authz = authorizerfactory.NewAlwaysDenyAuthorizer()
			}

			builder := newTestAuthorizationConditionsEvaluatorBuilder(t, budget, tc.authz, defaultNamespace)
			evaluator, err := builder.EvaluatorForData(t.Context(), versioned(t, tc.attrs))
			if err != nil {
				t.Fatalf("EvaluatorForData failed: %v", err)
			}
			result := evaluator.EvaluateCondition(t.Context(), tc.condition)

			if got := evalResultString(result); got != tc.wantResult {
				t.Errorf("got result %s, want %s (err=%v)", got, tc.wantResult, result.Error())
			}
		})
	}
}

func TestNewAuthorizationConditionsEvaluatorBuilder(t *testing.T) {
	client := fake.NewClientset()
	if b, err := NewAuthorizationConditionsEvaluatorBuilder(nil, client, nil, celconfig.RuntimeCELCostBudget); err == nil || b != nil {
		t.Errorf("expected an error and a nil builder for a nil informer factory, got err=%v, builder=%#v", err, b)
	}
	if b, err := NewAuthorizationConditionsEvaluatorBuilder(informers.NewSharedInformerFactory(client, 0), nil, nil, celconfig.RuntimeCELCostBudget); err == nil || b != nil {
		t.Errorf("expected an error and a nil builder for a nil client, got err=%v, builder=%#v", err, b)
	}
}

func TestAuthorizationConditionsEvaluatorBuilder_EvaluatorForData(t *testing.T) {
	tests := []struct {
		name       string
		data       authorizer.ConditionsData
		namespaces []runtime.Object
		wantErr    bool
	}{
		{
			name:    "nil data",
			data:    nil,
			wantErr: true,
		},
		{
			name:       "unversioned admission attributes",
			data:       podCreateAttributes(),
			namespaces: []runtime.Object{defaultNamespace},
			wantErr:    true,
		},
		{
			name:    "namespace does not exist",
			data:    versioned(t, podCreateAttributes()),
			wantErr: true,
		},
		{
			name:       "versioned admission attributes",
			data:       versioned(t, podCreateAttributes()),
			namespaces: []runtime.Object{defaultNamespace},
		},
		{
			name: "cluster-scoped resource needs no namespace",
			data: versioned(t, admission.NewAttributesRecord(
				&corev1.Namespace{TypeMeta: metav1.TypeMeta{Kind: "Namespace", APIVersion: "v1"}, ObjectMeta: metav1.ObjectMeta{Name: "new"}},
				nil, schema.GroupVersionKind{Version: "v1", Kind: "Namespace"}, "", "new",
				schema.GroupVersionResource{Version: "v1", Resource: "namespaces"}, "", admission.Create, &metav1.CreateOptions{}, false, nil)),
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			builder := newTestAuthorizationConditionsEvaluatorBuilder(t, celconfig.RuntimeCELCostBudget, authorizerfactory.NewAlwaysDenyAuthorizer(), tc.namespaces...)
			_, err := builder.EvaluatorForData(t.Context(), tc.data)
			if (err != nil) != tc.wantErr {
				t.Errorf("got err=%v, wantErr=%v", err, tc.wantErr)
			}
		})
	}
}

func TestAuthorizationConditionsEvaluator_AggregateCostBudget(t *testing.T) {
	cond := celCond("c", "object.spec.nodeName == 'node1'")

	// Measure the cost of a single evaluation, so the budget below fits exactly one.
	measuring := newTestAuthorizationConditionsEvaluatorBuilder(t, celconfig.RuntimeCELCostBudget, authorizerfactory.NewAlwaysDenyAuthorizer(), defaultNamespace)
	evaluator, err := measuring.EvaluatorForData(t.Context(), versioned(t, podCreateAttributes()))
	if err != nil {
		t.Fatalf("EvaluatorForData failed: %v", err)
	}
	if got := evalResultString(evaluator.EvaluateCondition(t.Context(), cond)); got != "True" {
		t.Fatalf("got %s, want True", got)
	}
	cost := celconfig.RuntimeCELCostBudget - evaluator.(*celAuthorizationConditionsEvaluator).remainingBudget
	if cost <= 0 {
		t.Fatalf("expected a positive evaluation cost, got %d", cost)
	}

	builder := newTestAuthorizationConditionsEvaluatorBuilder(t, cost+cost/2, authorizerfactory.NewAlwaysDenyAuthorizer(), defaultNamespace)
	newEvaluator := func() authorizer.BuiltinConditionsEvaluator {
		e, err := builder.EvaluatorForData(t.Context(), versioned(t, podCreateAttributes()))
		if err != nil {
			t.Fatalf("EvaluatorForData failed: %v", err)
		}
		return e
	}

	evaluator = newEvaluator()
	if got := evalResultString(evaluator.EvaluateCondition(t.Context(), cond)); got != "True" {
		t.Fatalf("first condition: got %s, want True", got)
	}
	// The second and third conditions exceed the shared budget; the third is not evaluated at all.
	for _, name := range []string{"second", "third"} {
		res := evaluator.EvaluateCondition(t.Context(), cond)
		if !res.IsError() || !errors.Is(res.Error(), apiservercel.ErrOutOfBudget) {
			t.Errorf("%s condition: got %s (err=%v), want out of budget error", name, evalResultString(res), res.Error())
		}
	}

	// The budget is per evaluator, not per builder.
	if got := evalResultString(newEvaluator().EvaluateCondition(t.Context(), cond)); got != "True" {
		t.Errorf("fresh evaluator: got %s, want True", got)
	}
}

func TestAuthorizationConditionsEvaluatorBuilder_Recognizes(t *testing.T) {
	builder := newTestAuthorizationConditionsEvaluatorBuilder(t, celconfig.RuntimeCELCostBudget, authorizerfactory.NewAlwaysDenyAuthorizer())
	tests := []struct {
		name      string
		condition authorizer.Condition
		want      bool
	}{
		{name: "CEL condition", condition: celCond("c", "true"), want: true},
		{name: "empty CEL condition", condition: celCond("c", "")},
		{name: "other type", condition: authorizer.GenericCondition{ID: "c", Type: "example.com/opaque", Condition: "true"}},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if got := builder.Recognizes(tc.condition); got != tc.want {
				t.Errorf("got %v, want %v", got, tc.want)
			}
		})
	}
}
