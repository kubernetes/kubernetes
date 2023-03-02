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
	"errors"
	"strings"
	"testing"

	"github.com/stretchr/testify/require"

	celtypes "github.com/google/cel-go/common/types"

	admissionv1 "k8s.io/api/admission/v1"
	v1 "k8s.io/api/admissionregistration/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apiserver/pkg/admission"
	"k8s.io/apiserver/pkg/admission/plugin/cel"
	"k8s.io/apiserver/pkg/admission/plugin/webhook/generic"
)

var _ cel.Filter = &fakeCelFilter{}

type fakeCelFilter struct {
	evaluations []cel.EvaluationResult
	throwError  bool
}

func (f *fakeCelFilter) ForInput(versionedAttr *generic.VersionedAttributes, versionedParams runtime.Object, request *admissionv1.AdmissionRequest) ([]cel.EvaluationResult, error) {
	if f.throwError {
		return nil, errors.New("test error")
	}
	return f.evaluations, nil
}

func (f *fakeCelFilter) CompilationErrors() []error {
	return []error{}
}

func TestValidate(t *testing.T) {
	ignore := v1.Ignore
	fail := v1.Fail

	forbiddenReason := metav1.StatusReasonForbidden
	unauthorizedReason := metav1.StatusReasonUnauthorized

	fakeAttr := admission.NewAttributesRecord(nil, nil, schema.GroupVersionKind{}, "default", "foo", schema.GroupVersionResource{}, "", admission.Create, nil, false, nil)
	fakeVersionedAttr, _ := generic.NewVersionedAttributes(fakeAttr, schema.GroupVersionKind{}, nil)

	cases := []struct {
		name           string
		failPolicy     *v1.FailurePolicyType
		evaluations    []cel.EvaluationResult
		policyDecision []PolicyDecision
		throwError     bool
	}{
		{
			name: "test pass",
			evaluations: []cel.EvaluationResult{
				{
					EvalResult:         celtypes.True,
					ExpressionAccessor: &ValidationCondition{},
				},
			},
			policyDecision: []PolicyDecision{
				{
					Action: ActionAdmit,
				},
			},
		},
		{
			name: "test multiple pass",
			evaluations: []cel.EvaluationResult{
				{
					EvalResult:         celtypes.True,
					ExpressionAccessor: &ValidationCondition{},
				},
				{
					EvalResult:         celtypes.True,
					ExpressionAccessor: &ValidationCondition{},
				},
			},
			policyDecision: []PolicyDecision{
				{
					Action: ActionAdmit,
				},
				{
					Action: ActionAdmit,
				},
			},
		},
		{
			name: "test error with failurepolicy ignore",
			evaluations: []cel.EvaluationResult{
				{
					Error:              errors.New(""),
					ExpressionAccessor: &ValidationCondition{},
				},
			},
			policyDecision: []PolicyDecision{
				{
					Action: ActionAdmit,
				},
			},
			failPolicy: &ignore,
		},
		{
			name: "test error with failurepolicy nil",
			evaluations: []cel.EvaluationResult{
				{
					Error:              errors.New(""),
					ExpressionAccessor: &ValidationCondition{},
				},
			},
			policyDecision: []PolicyDecision{
				{
					Action: ActionDeny,
				},
			},
		},
		{
			name: "test fail with failurepolicy fail",
			evaluations: []cel.EvaluationResult{
				{
					Error:              errors.New(""),
					ExpressionAccessor: &ValidationCondition{},
				},
			},
			policyDecision: []PolicyDecision{
				{
					Action: ActionDeny,
				},
			},
			failPolicy: &fail,
		},
		{
			name: "test fail with failurepolicy ignore with multiple validations",
			evaluations: []cel.EvaluationResult{
				{
					EvalResult:         celtypes.True,
					ExpressionAccessor: &ValidationCondition{},
				},
				{
					Error:              errors.New(""),
					ExpressionAccessor: &ValidationCondition{},
				},
			},
			policyDecision: []PolicyDecision{
				{
					Action: ActionAdmit,
				},
				{
					Action: ActionAdmit,
				},
			},
			failPolicy: &ignore,
		},
		{
			name: "test fail with failurepolicy nil with multiple validations",
			evaluations: []cel.EvaluationResult{
				{
					EvalResult:         celtypes.True,
					ExpressionAccessor: &ValidationCondition{},
				},
				{
					Error:              errors.New(""),
					ExpressionAccessor: &ValidationCondition{},
				},
			},
			policyDecision: []PolicyDecision{
				{
					Action: ActionAdmit,
				},
				{
					Action: ActionDeny,
				},
			},
		},
		{
			name: "test fail with failurepolicy fail with multiple validations",
			evaluations: []cel.EvaluationResult{
				{
					EvalResult:         celtypes.True,
					ExpressionAccessor: &ValidationCondition{},
				},
				{
					Error:              errors.New(""),
					ExpressionAccessor: &ValidationCondition{},
				},
			},
			policyDecision: []PolicyDecision{
				{
					Action: ActionAdmit,
				},
				{
					Action: ActionDeny,
				},
			},
			failPolicy: &fail,
		},
		{
			name: "test fail with failurepolicy ignore with multiple failed validations",
			evaluations: []cel.EvaluationResult{
				{
					Error:              errors.New(""),
					ExpressionAccessor: &ValidationCondition{},
				},
				{
					Error:              errors.New(""),
					ExpressionAccessor: &ValidationCondition{},
				},
			},
			policyDecision: []PolicyDecision{
				{
					Action: ActionAdmit,
				},
				{
					Action: ActionAdmit,
				},
			},
			failPolicy: &ignore,
		},
		{
			name: "test fail with failurepolicy nil with multiple failed validations",
			evaluations: []cel.EvaluationResult{
				{
					Error:              errors.New(""),
					ExpressionAccessor: &ValidationCondition{},
				},
				{
					Error:              errors.New(""),
					ExpressionAccessor: &ValidationCondition{},
				},
			},
			policyDecision: []PolicyDecision{
				{
					Action: ActionDeny,
				},
				{
					Action: ActionDeny,
				},
			},
		},
		{
			name: "test fail with failurepolicy fail with multiple failed validations",
			evaluations: []cel.EvaluationResult{
				{
					Error:              errors.New(""),
					ExpressionAccessor: &ValidationCondition{},
				},
				{
					Error:              errors.New(""),
					ExpressionAccessor: &ValidationCondition{},
				},
			},
			policyDecision: []PolicyDecision{
				{
					Action: ActionDeny,
				},
				{
					Action: ActionDeny,
				},
			},
			failPolicy: &fail,
		},
		{
			name: "test reason for fail no reason set",
			evaluations: []cel.EvaluationResult{
				{
					EvalResult: celtypes.False,
					ExpressionAccessor: &ValidationCondition{
						Expression: "this.expression == unit.test",
					},
				},
			},
			policyDecision: []PolicyDecision{
				{
					Action:  ActionDeny,
					Reason:  metav1.StatusReasonInvalid,
					Message: "failed expression: this.expression == unit.test",
				},
			},
			failPolicy: &fail,
		},
		{
			name: "test reason for fail reason set",
			evaluations: []cel.EvaluationResult{
				{
					EvalResult: celtypes.False,
					ExpressionAccessor: &ValidationCondition{
						Reason:     &forbiddenReason,
						Expression: "this.expression == unit.test",
					},
				},
			},
			policyDecision: []PolicyDecision{
				{
					Action:  ActionDeny,
					Reason:  metav1.StatusReasonForbidden,
					Message: "failed expression: this.expression == unit.test",
				},
			},
			failPolicy: &fail,
		},
		{
			name: "test reason for failed validations multiple validations",
			evaluations: []cel.EvaluationResult{
				{
					EvalResult: celtypes.False,
					ExpressionAccessor: &ValidationCondition{
						Reason:     &forbiddenReason,
						Expression: "this.expression == unit.test",
					},
				},
				{
					EvalResult: celtypes.False,
					ExpressionAccessor: &ValidationCondition{
						Reason:     &unauthorizedReason,
						Expression: "this.expression.2 == unit.test.2",
					},
				},
			},
			policyDecision: []PolicyDecision{
				{
					Action:  ActionDeny,
					Reason:  metav1.StatusReasonForbidden,
					Message: "failed expression: this.expression == unit.test",
				},
				{
					Action:  ActionDeny,
					Reason:  metav1.StatusReasonUnauthorized,
					Message: "failed expression: this.expression.2 == unit.test.2",
				},
			},
			failPolicy: &fail,
		},
		{
			name: "test message for failed validations",
			evaluations: []cel.EvaluationResult{
				{
					EvalResult: celtypes.False,
					ExpressionAccessor: &ValidationCondition{
						Reason:     &forbiddenReason,
						Expression: "this.expression == unit.test",
						Message:    "test",
					},
				},
			},
			policyDecision: []PolicyDecision{
				{
					Action:  ActionDeny,
					Reason:  metav1.StatusReasonForbidden,
					Message: "test",
				},
			},
			failPolicy: &fail,
		},
		{
			name: "test message for failed validations multiple validations",
			evaluations: []cel.EvaluationResult{
				{
					EvalResult: celtypes.False,
					ExpressionAccessor: &ValidationCondition{
						Reason:     &forbiddenReason,
						Expression: "this.expression == unit.test",
						Message:    "test1",
					},
				},
				{
					EvalResult: celtypes.False,
					ExpressionAccessor: &ValidationCondition{
						Reason:     &forbiddenReason,
						Expression: "this.expression == unit.test",
						Message:    "test2",
					},
				},
			},
			policyDecision: []PolicyDecision{
				{
					Action:  ActionDeny,
					Reason:  metav1.StatusReasonForbidden,
					Message: "test1",
				},
				{
					Action:  ActionDeny,
					Reason:  metav1.StatusReasonForbidden,
					Message: "test2",
				},
			},
			failPolicy: &fail,
		},
		{
			name: "test filter error",
			evaluations: []cel.EvaluationResult{
				{
					EvalResult: celtypes.False,
					ExpressionAccessor: &ValidationCondition{
						Reason:     &forbiddenReason,
						Expression: "this.expression == unit.test",
						Message:    "test1",
					},
				},
			},
			policyDecision: []PolicyDecision{
				{
					Action:  ActionDeny,
					Message: "test error",
				},
			},
			failPolicy: &fail,
			throwError: true,
		},
		{
			name: "test filter error multiple evaluations",
			evaluations: []cel.EvaluationResult{
				{
					EvalResult: celtypes.False,
					ExpressionAccessor: &ValidationCondition{
						Reason:     &forbiddenReason,
						Expression: "this.expression == unit.test",
						Message:    "test1",
					},
				},
				{
					EvalResult: celtypes.False,
					ExpressionAccessor: &ValidationCondition{
						Reason:     &forbiddenReason,
						Expression: "this.expression == unit.test",
						Message:    "test2",
					},
				},
			},
			policyDecision: []PolicyDecision{
				{
					Action:  ActionDeny,
					Message: "test error",
				},
			},
			failPolicy: &fail,
			throwError: true,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			v := validator{
				failPolicy: tc.failPolicy,
				filter: &fakeCelFilter{
					evaluations: tc.evaluations,
					throwError:  tc.throwError,
				},
			}
			policyResults := v.Validate(fakeVersionedAttr, nil)

			require.Equal(t, len(policyResults), len(tc.policyDecision))

			for i, policyDecision := range tc.policyDecision {
				if policyDecision.Action != policyResults[i].Action {
					t.Errorf("Expected policy decision kind '%v' but got '%v'", policyDecision.Action, policyResults[i].Action)
				}
				if !strings.Contains(policyResults[i].Message, policyDecision.Message) {
					t.Errorf("Expected policy decision message contains '%v' but got '%v'", policyDecision.Message, policyResults[i].Message)
				}
				if policyDecision.Reason != policyResults[i].Reason {
					t.Errorf("Expected policy decision reason '%v' but got '%v'", policyDecision.Reason, policyResults[i].Reason)
				}
			}
		})
	}
}
