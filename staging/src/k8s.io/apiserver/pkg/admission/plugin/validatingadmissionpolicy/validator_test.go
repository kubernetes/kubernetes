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
	"context"
	"errors"
	"fmt"
	"strings"
	"testing"

	"github.com/stretchr/testify/require"

	celtypes "github.com/google/cel-go/common/types"

	admissionv1 "k8s.io/api/admission/v1"
	v1 "k8s.io/api/admissionregistration/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apiserver/pkg/admission"
	"k8s.io/apiserver/pkg/admission/plugin/cel"
	celconfig "k8s.io/apiserver/pkg/apis/cel"
)

var _ cel.Filter = &fakeCelFilter{}

type fakeCelFilter struct {
	evaluations []cel.EvaluationResult
	throwError  bool
}

func (f *fakeCelFilter) ForInput(context.Context, *admission.VersionedAttributes, *admissionv1.AdmissionRequest, cel.OptionalVariableBindings, int64) ([]cel.EvaluationResult, error) {
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
	fakeVersionedAttr, _ := admission.NewVersionedAttributes(fakeAttr, schema.GroupVersionKind{}, nil)

	cases := []struct {
		name             string
		failPolicy       *v1.FailurePolicyType
		evaluations      []cel.EvaluationResult
		auditEvaluations []cel.EvaluationResult
		policyDecision   []PolicyDecision
		auditAnnotations []PolicyAuditAnnotation
		throwError       bool
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
		{
			name: "test empty validations with non-empty audit annotations",
			auditEvaluations: []cel.EvaluationResult{
				{
					EvalResult: celtypes.String("string value"),
					ExpressionAccessor: &AuditAnnotationCondition{
						ValueExpression: "'string value'",
					},
				},
			},
			failPolicy: &fail,
			auditAnnotations: []PolicyAuditAnnotation{
				{
					Action: AuditAnnotationActionPublish,
					Value:  "string value",
				},
			},
		},
		{
			name: "test non-empty validations with non-empty audit annotations",
			evaluations: []cel.EvaluationResult{
				{
					EvalResult: celtypes.True,
					ExpressionAccessor: &ValidationCondition{
						Reason:     &forbiddenReason,
						Expression: "this.expression == unit.test",
						Message:    "test1",
					},
				},
			},
			auditEvaluations: []cel.EvaluationResult{
				{
					EvalResult: celtypes.String("string value"),
					ExpressionAccessor: &AuditAnnotationCondition{
						ValueExpression: "'string value'",
					},
				},
			},
			policyDecision: []PolicyDecision{
				{
					Action: ActionAdmit,
				},
			},
			auditAnnotations: []PolicyAuditAnnotation{
				{
					Action: AuditAnnotationActionPublish,
					Value:  "string value",
				},
			},
			failPolicy: &fail,
		},
		{
			name: "test audit annotations with null return",
			auditEvaluations: []cel.EvaluationResult{
				{
					EvalResult: celtypes.NullValue,
					ExpressionAccessor: &AuditAnnotationCondition{
						ValueExpression: "null",
					},
				},
				{
					EvalResult: celtypes.String("string value"),
					ExpressionAccessor: &AuditAnnotationCondition{
						ValueExpression: "'string value'",
					},
				},
			},
			auditAnnotations: []PolicyAuditAnnotation{
				{
					Action: AuditAnnotationActionExclude,
				},
				{
					Action: AuditAnnotationActionPublish,
					Value:  "string value",
				},
			},
			failPolicy: &fail,
		},
		{
			name: "test audit annotations with failPolicy=fail",
			auditEvaluations: []cel.EvaluationResult{
				{
					Error: fmt.Errorf("valueExpression ''this is not valid CEL' resulted in error: <nil>"),
					ExpressionAccessor: &AuditAnnotationCondition{
						ValueExpression: "'this is not valid CEL",
					},
				},
			},
			auditAnnotations: []PolicyAuditAnnotation{
				{
					Action: AuditAnnotationActionError,
					Error:  "valueExpression ''this is not valid CEL' resulted in error: <nil>",
				},
			},
			failPolicy: &fail,
		},
		{
			name: "test audit annotations with failPolicy=ignore",
			auditEvaluations: []cel.EvaluationResult{
				{
					Error: fmt.Errorf("valueExpression ''this is not valid CEL' resulted in error: <nil>"),
					ExpressionAccessor: &AuditAnnotationCondition{
						ValueExpression: "'this is not valid CEL",
					},
				},
			},
			auditAnnotations: []PolicyAuditAnnotation{
				{
					Action: AuditAnnotationActionExclude, // TODO: is this right?
					Error:  "valueExpression ''this is not valid CEL' resulted in error: <nil>",
				},
			},
			failPolicy: &ignore,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			v := validator{
				failPolicy: tc.failPolicy,
				validationFilter: &fakeCelFilter{
					evaluations: tc.evaluations,
					throwError:  tc.throwError,
				},
				auditAnnotationFilter: &fakeCelFilter{
					evaluations: tc.auditEvaluations,
					throwError:  tc.throwError,
				},
			}
			ctx := context.TODO()
			validateResult := v.Validate(ctx, fakeVersionedAttr, nil, celconfig.RuntimeCELCostBudget)

			require.Equal(t, len(validateResult.Decisions), len(tc.policyDecision))

			for i, policyDecision := range tc.policyDecision {
				if policyDecision.Action != validateResult.Decisions[i].Action {
					t.Errorf("Expected policy decision kind '%v' but got '%v'", policyDecision.Action, validateResult.Decisions[i].Action)
				}
				if !strings.Contains(validateResult.Decisions[i].Message, policyDecision.Message) {
					t.Errorf("Expected policy decision message contains '%v' but got '%v'", policyDecision.Message, validateResult.Decisions[i].Message)
				}
				if policyDecision.Reason != validateResult.Decisions[i].Reason {
					t.Errorf("Expected policy decision reason '%v' but got '%v'", policyDecision.Reason, validateResult.Decisions[i].Reason)
				}
			}
			require.Equal(t, len(tc.auditEvaluations), len(validateResult.AuditAnnotations))

			for i, auditAnnotation := range tc.auditAnnotations {
				actual := validateResult.AuditAnnotations[i]
				if auditAnnotation.Action != actual.Action {
					t.Errorf("Expected policy audit annotation action '%v' but got '%v'", auditAnnotation.Action, actual.Action)
				}
				if auditAnnotation.Error != actual.Error {
					t.Errorf("Expected audit annotation error '%v' but got '%v'", auditAnnotation.Error, actual.Error)
				}
				if auditAnnotation.Value != actual.Value {
					t.Errorf("Expected policy audit annotation value '%v' but got '%v'", auditAnnotation.Value, actual.Value)
				}
			}
		})
	}
}

func TestContextCanceled(t *testing.T) {
	fail := v1.Fail

	fakeAttr := admission.NewAttributesRecord(nil, nil, schema.GroupVersionKind{}, "default", "foo", schema.GroupVersionResource{}, "", admission.Create, nil, false, nil)
	fakeVersionedAttr, _ := admission.NewVersionedAttributes(fakeAttr, schema.GroupVersionKind{}, nil)
	fc := cel.NewFilterCompiler()
	f := fc.Compile([]cel.ExpressionAccessor{&ValidationCondition{Expression: "[1,2,3,4,5,6,7,8,9,10].map(x, [1,2,3,4,5,6,7,8,9,10].map(y, x*y)) == []"}}, cel.OptionalVariableDeclarations{HasParams: false, HasAuthorizer: false}, celconfig.PerCallLimit)
	v := validator{
		failPolicy:       &fail,
		validationFilter: f,
		auditAnnotationFilter: &fakeCelFilter{
			evaluations: nil,
			throwError:  false,
		},
	}
	ctx, cancel := context.WithCancel(context.TODO())
	cancel()
	validationResult := v.Validate(ctx, fakeVersionedAttr, nil, celconfig.RuntimeCELCostBudget)
	if len(validationResult.Decisions) != 1 || !strings.Contains(validationResult.Decisions[0].Message, "operation interrupted") {
		t.Errorf("Expected 'operation interrupted' but got %v", validationResult.Decisions)
	}
}
