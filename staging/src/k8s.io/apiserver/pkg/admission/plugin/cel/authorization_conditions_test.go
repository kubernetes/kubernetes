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
	"testing"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apiserver/pkg/admission"
	celconfig "k8s.io/apiserver/pkg/apis/cel"
	"k8s.io/apiserver/pkg/authorization/authorizer"
)

var (
	podGVK = schema.GroupVersionKind{Group: "", Version: "v1", Kind: "Pod"}
	podGVR = schema.GroupVersionResource{Group: "", Version: "v1", Resource: "pods"}
)

func newTestAuthorizationConditionsEvaluator() *AuthorizationConditionsEvaluator {
	return NewAuthorizationConditionsEvaluator(nil, nil, nil, celconfig.RuntimeCELCostBudget)
}

func newTestAuthorizationConditionsEvaluatorWithBudget(budget int64) *AuthorizationConditionsEvaluator {
	return NewAuthorizationConditionsEvaluator(nil, nil, nil, budget)
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
		Type:      ConditionTypeAuthorizationCEL,
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
		attrs      admission.Attributes // nil → AdmissionControl is nil
		costBudget int64
		wantResult string // "True", "False", "Error", "Unevaluatable"
	}{
		// --- Skip / Unevaluatable cases ---
		{
			name:       "nil condition returns Unevaluatable",
			condition:  nil,
			attrs:      podCreateAttributes(),
			wantResult: "Unevaluatable",
		},
		{
			name:       "nil AdmissionControl returns Unevaluatable",
			condition:  celCond("c", "true"),
			attrs:      nil,
			wantResult: "Unevaluatable",
		},
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
				Type:      ConditionTypeAuthorizationCEL,
				Condition: "",
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

			evaluator := newTestAuthorizationConditionsEvaluatorWithBudget(budget)
			data := authorizer.ConditionsData{AdmissionControl: tc.attrs}
			result := evaluator.EvaluateCondition(t.Context(), tc.condition, data)

			if got := evalResultString(result); got != tc.wantResult {
				t.Errorf("got result %s, want %s (err=%v)", got, tc.wantResult, result.Error())
			}
		})
	}
}
