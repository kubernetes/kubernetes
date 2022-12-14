/*
Copyright 2022 The Kubernetes Authors.

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

package validatingadmissionpolicy

import (
	"strings"
	"testing"

	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"

	"github.com/stretchr/testify/require"

	v1 "k8s.io/api/admissionregistration/v1"
	"k8s.io/api/admissionregistration/v1alpha1"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apiserver/pkg/admission"
)

func TestCompile(t *testing.T) {
	cases := []struct {
		name             string
		policy           *v1alpha1.ValidatingAdmissionPolicy
		errorExpressions map[string]string
	}{
		{
			name: "invalid syntax",
			policy: &v1alpha1.ValidatingAdmissionPolicy{
				ObjectMeta: metav1.ObjectMeta{
					Name: "foo",
				},
				Spec: v1alpha1.ValidatingAdmissionPolicySpec{
					FailurePolicy: func() *v1alpha1.FailurePolicyType {
						r := v1alpha1.FailurePolicyType("Fail")
						return &r
					}(),
					ParamKind: &v1alpha1.ParamKind{
						APIVersion: "rules.example.com/v1",
						Kind:       "ReplicaLimit",
					},
					Validations: []v1alpha1.Validation{
						{
							Expression: "1 < 'asdf'",
						},
						{
							Expression: "1 < 2",
						},
					},
					MatchConstraints: &v1alpha1.MatchResources{
						MatchPolicy: func() *v1alpha1.MatchPolicyType {
							r := v1alpha1.MatchPolicyType("Exact")
							return &r
						}(),
						ResourceRules: []v1alpha1.NamedRuleWithOperations{
							{
								RuleWithOperations: v1alpha1.RuleWithOperations{
									Operations: []v1.OperationType{"CREATE"},
									Rule: v1.Rule{
										APIGroups:   []string{"a"},
										APIVersions: []string{"a"},
										Resources:   []string{"a"},
									},
								},
							},
						},
						ObjectSelector: &metav1.LabelSelector{
							MatchLabels: map[string]string{"a": "b"},
						},
						NamespaceSelector: &metav1.LabelSelector{
							MatchLabels: map[string]string{"a": "b"},
						},
					},
				},
			},
			errorExpressions: map[string]string{
				"1 < 'asdf'": "found no matching overload for '_<_' applied to '(int, string)",
			},
		},
		{
			name: "valid syntax",
			policy: &v1alpha1.ValidatingAdmissionPolicy{
				ObjectMeta: metav1.ObjectMeta{
					Name: "foo",
				},
				Spec: v1alpha1.ValidatingAdmissionPolicySpec{
					FailurePolicy: func() *v1alpha1.FailurePolicyType {
						r := v1alpha1.FailurePolicyType("Fail")
						return &r
					}(),
					Validations: []v1alpha1.Validation{
						{
							Expression: "1 < 2",
						},
						{
							Expression: "object.spec.string.matches('[0-9]+')",
						},
						{
							Expression: "request.kind.group == 'example.com' && request.kind.version == 'v1' && request.kind.kind == 'Fake'",
						},
					},
					MatchConstraints: &v1alpha1.MatchResources{
						MatchPolicy: func() *v1alpha1.MatchPolicyType {
							r := v1alpha1.MatchPolicyType("Exact")
							return &r
						}(),
						ResourceRules: []v1alpha1.NamedRuleWithOperations{
							{
								RuleWithOperations: v1alpha1.RuleWithOperations{
									Operations: []v1.OperationType{"CREATE"},
									Rule: v1.Rule{
										APIGroups:   []string{"a"},
										APIVersions: []string{"a"},
										Resources:   []string{"a"},
									},
								},
							},
						},
						ObjectSelector: &metav1.LabelSelector{
							MatchLabels: map[string]string{"a": "b"},
						},
						NamespaceSelector: &metav1.LabelSelector{
							MatchLabels: map[string]string{"a": "b"},
						},
					},
				},
			},
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			var c CELValidatorCompiler
			validator := c.Compile(tc.policy)
			if validator == nil {
				t.Fatalf("unexpected nil validator")
			}
			validations := tc.policy.Spec.Validations
			CompilationResults := validator.(*CELValidator).compilationResults
			require.Equal(t, len(validations), len(CompilationResults))

			meets := make([]bool, len(validations))
			for expr, expectErr := range tc.errorExpressions {
				for i, result := range CompilationResults {
					if validations[i].Expression == expr {
						if result.Error == nil {
							t.Errorf("Expect expression '%s' to contain error '%v' but got no error", expr, expectErr)
						} else if !strings.Contains(result.Error.Error(), expectErr) {
							t.Errorf("Expected validation '%s' error to contain '%v' but got: %v", expr, expectErr, result.Error)
						}
						meets[i] = true
					}
				}
			}
			for i, meet := range meets {
				if !meet && CompilationResults[i].Error != nil {
					t.Errorf("Unexpected err '%v' for expression '%s'", CompilationResults[i].Error, validations[i].Expression)
				}
			}
		})
	}
}

func getValidPolicy(validations []v1alpha1.Validation, params *v1alpha1.ParamKind, fp *v1alpha1.FailurePolicyType) *v1alpha1.ValidatingAdmissionPolicy {
	if fp == nil {
		fp = func() *v1alpha1.FailurePolicyType {
			r := v1alpha1.FailurePolicyType("Fail")
			return &r
		}()
	}
	return &v1alpha1.ValidatingAdmissionPolicy{
		ObjectMeta: metav1.ObjectMeta{
			Name: "foo",
		},
		Spec: v1alpha1.ValidatingAdmissionPolicySpec{
			FailurePolicy: fp,
			Validations:   validations,
			ParamKind:     params,
			MatchConstraints: &v1alpha1.MatchResources{
				MatchPolicy: func() *v1alpha1.MatchPolicyType {
					r := v1alpha1.MatchPolicyType("Exact")
					return &r
				}(),
				ResourceRules: []v1alpha1.NamedRuleWithOperations{
					{
						RuleWithOperations: v1alpha1.RuleWithOperations{
							Operations: []v1.OperationType{"CREATE"},
							Rule: v1.Rule{
								APIGroups:   []string{"a"},
								APIVersions: []string{"a"},
								Resources:   []string{"a"},
							},
						},
					},
				},
				ObjectSelector: &metav1.LabelSelector{
					MatchLabels: map[string]string{"a": "b"},
				},
				NamespaceSelector: &metav1.LabelSelector{
					MatchLabels: map[string]string{"a": "b"},
				},
			},
		},
	}
}

func generatedDecision(k policyDecisionAction, m string, r metav1.StatusReason) policyDecision {
	return policyDecision{action: k, message: m, reason: r}
}

func TestValidate(t *testing.T) {
	// we fake the paramKind in ValidatingAdmissionPolicy for testing since the params is directly passed from cel admission
	// Inside validator.go, we only check if paramKind exists
	hasParamKind := &v1alpha1.ParamKind{
		APIVersion: "v1",
		Kind:       "ConfigMap",
	}
	ignorePolicy := func() *v1alpha1.FailurePolicyType {
		r := v1alpha1.FailurePolicyType("Ignore")
		return &r
	}()
	forbiddenReason := func() *metav1.StatusReason {
		r := metav1.StatusReasonForbidden
		return &r
	}()

	configMapParams := &corev1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name: "foo",
		},
		Data: map[string]string{
			"fakeString": "fake",
		},
	}
	crdParams := &unstructured.Unstructured{
		Object: map[string]interface{}{
			"spec": map[string]interface{}{
				"testSize": 10,
			},
		},
	}
	podObject := corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name: "foo",
		},
		Spec: corev1.PodSpec{
			NodeName: "testnode",
		},
	}

	var nilUnstructured *unstructured.Unstructured

	cases := []struct {
		name            string
		policy          *v1alpha1.ValidatingAdmissionPolicy
		attributes      admission.Attributes
		params          runtime.Object
		policyDecisions []policyDecision
	}{
		{
			name: "valid syntax for object",
			policy: getValidPolicy([]v1alpha1.Validation{
				{
					Expression: "has(object.subsets) && object.subsets.size() < 2",
				},
			}, nil, nil),
			attributes: newValidAttribute(nil, false),
			policyDecisions: []policyDecision{
				generatedDecision(actionAdmit, "", ""),
			},
		},
		{
			name: "valid syntax for metadata",
			policy: getValidPolicy([]v1alpha1.Validation{
				{
					Expression: "object.metadata.name == 'endpoints1'",
				},
			}, nil, nil),
			attributes: newValidAttribute(nil, false),
			policyDecisions: []policyDecision{
				generatedDecision(actionAdmit, "", ""),
			},
		},
		{
			name: "valid syntax for oldObject",
			policy: getValidPolicy([]v1alpha1.Validation{
				{
					Expression: "oldObject == null",
				},
				{
					Expression: "object != null",
				},
			}, nil, nil),
			attributes: newValidAttribute(nil, false),
			policyDecisions: []policyDecision{
				generatedDecision(actionAdmit, "", ""),
				generatedDecision(actionAdmit, "", ""),
			},
		},
		{
			name: "valid syntax for request",
			policy: getValidPolicy([]v1alpha1.Validation{
				{Expression: "request.operation == 'CREATE'"},
			}, nil, nil),
			attributes: newValidAttribute(nil, false),
			policyDecisions: []policyDecision{
				generatedDecision(actionAdmit, "", ""),
			},
		},
		{
			name: "valid syntax for configMap",
			policy: getValidPolicy([]v1alpha1.Validation{
				{Expression: "request.namespace != params.data.fakeString"},
			}, hasParamKind, nil),
			attributes: newValidAttribute(nil, false),
			params:     configMapParams,
			policyDecisions: []policyDecision{
				generatedDecision(actionAdmit, "", ""),
			},
		},
		{
			name: "test failure policy with Ignore",
			policy: getValidPolicy([]v1alpha1.Validation{
				{Expression: "object.subsets.size() > 2"},
			}, hasParamKind, ignorePolicy),
			attributes: newValidAttribute(nil, false),
			params: &corev1.ConfigMap{
				ObjectMeta: metav1.ObjectMeta{
					Name: "foo",
				},
				Data: map[string]string{
					"fakeString": "fake",
				},
			},
			policyDecisions: []policyDecision{
				generatedDecision(actionDeny, "failed expression: object.subsets.size() > 2", metav1.StatusReasonInvalid),
			},
		},
		{
			name: "test failure policy with multiple validations",
			policy: getValidPolicy([]v1alpha1.Validation{
				{
					Expression: "has(object.subsets)",
				},
				{
					Expression: "object.subsets.size() > 2",
				},
			}, hasParamKind, ignorePolicy),
			attributes: newValidAttribute(nil, false),
			params:     configMapParams,
			policyDecisions: []policyDecision{
				generatedDecision(actionAdmit, "", ""),
				generatedDecision(actionDeny, "failed expression: object.subsets.size() > 2", metav1.StatusReasonInvalid),
			},
		},
		{
			name: "test failure policy with multiple failed validations",
			policy: getValidPolicy([]v1alpha1.Validation{
				{
					Expression: "oldObject != null",
				},
				{
					Expression: "object.subsets.size() > 2",
				},
			}, hasParamKind, nil),
			attributes: newValidAttribute(nil, false),
			params:     configMapParams,
			policyDecisions: []policyDecision{
				generatedDecision(actionDeny, "failed expression: oldObject != null", metav1.StatusReasonInvalid),
				generatedDecision(actionDeny, "failed expression: object.subsets.size() > 2", metav1.StatusReasonInvalid),
			},
		},
		{
			name: "test Object nul in delete",
			policy: getValidPolicy([]v1alpha1.Validation{
				{
					Expression: "oldObject != null",
				},
				{
					Expression: "object == null",
				},
			}, hasParamKind, nil),
			attributes: newValidAttribute(nil, true),
			params:     configMapParams,
			policyDecisions: []policyDecision{
				generatedDecision(actionAdmit, "", ""),
				generatedDecision(actionAdmit, "", ""),
			},
		},
		{
			name: "test reason for failed validation",
			policy: getValidPolicy([]v1alpha1.Validation{
				{
					Expression: "oldObject == null",
					Reason:     forbiddenReason,
				},
			}, hasParamKind, nil),
			attributes: newValidAttribute(nil, true),
			params:     configMapParams,
			policyDecisions: []policyDecision{
				generatedDecision(actionDeny, "failed expression: oldObject == null", metav1.StatusReasonForbidden),
			},
		},
		{
			name: "test message for failed validation",
			policy: getValidPolicy([]v1alpha1.Validation{
				{
					Expression: "oldObject == null",
					Reason:     forbiddenReason,
					Message:    "old object should be present",
				},
			}, hasParamKind, nil),
			attributes: newValidAttribute(nil, true),
			params:     configMapParams,
			policyDecisions: []policyDecision{
				generatedDecision(actionDeny, "old object should be present", metav1.StatusReasonForbidden),
			},
		},
		{
			name: "test runtime error",
			policy: getValidPolicy([]v1alpha1.Validation{
				{
					Expression: "oldObject.x == 100",
				},
			}, hasParamKind, nil),
			attributes: newValidAttribute(nil, true),
			params:     configMapParams,
			policyDecisions: []policyDecision{
				generatedDecision(actionDeny, "resulted in error", ""),
			},
		},
		{
			name: "test against crd param",
			policy: getValidPolicy([]v1alpha1.Validation{
				{
					Expression: "object.subsets.size() < params.spec.testSize",
				},
			}, hasParamKind, nil),
			attributes: newValidAttribute(nil, false),
			params:     crdParams,
			policyDecisions: []policyDecision{
				generatedDecision(actionAdmit, "", ""),
			},
		},
		{
			name: "test compile failure with FailurePolicy Fail",
			policy: getValidPolicy([]v1alpha1.Validation{
				{
					Expression: "fail to compile test",
				},
				{
					Expression: "object.subsets.size() > params.spec.testSize",
				},
			}, hasParamKind, nil),
			attributes: newValidAttribute(nil, false),
			params:     crdParams,
			policyDecisions: []policyDecision{
				generatedDecision(actionDeny, "compilation error: compilation failed: ERROR: <input>:1:6: Syntax error:", ""),
				generatedDecision(actionDeny, "failed expression: object.subsets.size() > params.spec.testSize", metav1.StatusReasonInvalid),
			},
		},
		{
			name: "test compile failure with FailurePolicy Ignore",
			policy: getValidPolicy([]v1alpha1.Validation{
				{
					Expression: "fail to compile test",
				},
				{
					Expression: "object.subsets.size() > params.spec.testSize",
				},
			}, hasParamKind, ignorePolicy),
			attributes: newValidAttribute(nil, false),
			params:     crdParams,
			policyDecisions: []policyDecision{
				generatedDecision(actionAdmit, "compilation error: compilation failed: ERROR:", ""),
				generatedDecision(actionDeny, "failed expression: object.subsets.size() > params.spec.testSize", metav1.StatusReasonInvalid),
			},
		},
		{
			name: "test pod",
			policy: getValidPolicy([]v1alpha1.Validation{
				{
					Expression: "object.spec.nodeName == 'testnode'",
				},
			}, nil, nil),
			attributes: newValidAttribute(&podObject, false),
			params:     crdParams,
			policyDecisions: []policyDecision{
				generatedDecision(actionAdmit, "", ""),
			},
		},
		{
			name: "test deny paramKind without paramRef",
			policy: getValidPolicy([]v1alpha1.Validation{
				{
					Expression: "params != null",
					Reason:     forbiddenReason,
					Message:    "params as required",
				},
			}, hasParamKind, nil),
			attributes: newValidAttribute(nil, true),
			// Simulate a interface holding a nil pointer, since this is how param is passed to Validate
			// if paramRef is unset on a binding
			params: runtime.Object(nilUnstructured),
			policyDecisions: []policyDecision{
				generatedDecision(actionDeny, "params as required", metav1.StatusReasonForbidden),
			},
		},
		{
			name: "test allow paramKind without paramRef",
			policy: getValidPolicy([]v1alpha1.Validation{
				{
					Expression: "params == null",
					Reason:     forbiddenReason,
				},
			}, hasParamKind, nil),
			attributes: newValidAttribute(nil, true),
			// Simulate a interface holding a nil pointer, since this is how param is passed to Validate
			// if paramRef is unset on a binding
			params: runtime.Object(nilUnstructured),
			policyDecisions: []policyDecision{
				generatedDecision(actionAdmit, "", ""),
			},
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			c := CELValidatorCompiler{}
			validator := c.Compile(tc.policy)
			if validator == nil {
				t.Fatalf("unexpected nil validator")
			}
			validations := tc.policy.Spec.Validations
			CompilationResults := validator.(*CELValidator).compilationResults
			require.Equal(t, len(validations), len(CompilationResults))

			policyResults, err := validator.Validate(tc.attributes, newObjectInterfacesForTest(), tc.params, tc.attributes.GetKind())
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			require.Equal(t, len(policyResults), len(tc.policyDecisions))
			for i, policyDecision := range tc.policyDecisions {
				if policyDecision.action != policyResults[i].action {
					t.Errorf("Expected policy decision kind '%v' but got '%v'", policyDecision.action, policyResults[i].action)
				}
				if !strings.Contains(policyResults[i].message, policyDecision.message) {
					t.Errorf("Expected policy decision message contains '%v' but got '%v'", policyDecision.message, policyResults[i].message)
				}
				if policyDecision.reason != policyResults[i].reason {
					t.Errorf("Expected policy decision reason '%v' but got '%v'", policyDecision.reason, policyResults[i].reason)
				}
			}
		})
	}
}

// newObjectInterfacesForTest returns an ObjectInterfaces appropriate for test cases in this file.
func newObjectInterfacesForTest() admission.ObjectInterfaces {
	scheme := runtime.NewScheme()
	corev1.AddToScheme(scheme)
	return admission.NewObjectInterfacesFromScheme(scheme)
}

func newValidAttribute(object runtime.Object, isDelete bool) admission.Attributes {
	var oldObject runtime.Object
	if !isDelete {
		if object == nil {
			object = &corev1.Endpoints{
				ObjectMeta: metav1.ObjectMeta{
					Name: "endpoints1",
				},
				Subsets: []corev1.EndpointSubset{
					{
						Addresses: []corev1.EndpointAddress{{IP: "127.0.0.0"}},
					},
				},
			}
		}
	} else {
		object = nil
		oldObject = &corev1.Endpoints{
			Subsets: []corev1.EndpointSubset{
				{
					Addresses: []corev1.EndpointAddress{{IP: "127.0.0.0"}},
				},
			},
		}
	}
	return admission.NewAttributesRecord(object, oldObject, schema.GroupVersionKind{}, "default", "foo", schema.GroupVersionResource{}, "", admission.Create, &metav1.CreateOptions{}, false, nil)

}
