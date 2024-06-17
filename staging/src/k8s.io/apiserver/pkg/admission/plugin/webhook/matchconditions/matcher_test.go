/*
Copyright 2023 The Kubernetes Authors.

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

package matchconditions

import (
	"context"
	"errors"
	"strings"
	"testing"

	api "k8s.io/api/core/v1"

	v1 "k8s.io/api/admissionregistration/v1"

	celtypes "github.com/google/cel-go/common/types"
	"github.com/stretchr/testify/require"

	admissionv1 "k8s.io/api/admission/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apiserver/pkg/admission"
	"k8s.io/apiserver/pkg/admission/plugin/cel"
)

var _ cel.Filter = &fakeCelFilter{}

type fakeCelFilter struct {
	evaluations []cel.EvaluationResult
	throwError  bool
}

func (f *fakeCelFilter) ForInput(context.Context, *admission.VersionedAttributes, *admissionv1.AdmissionRequest, cel.OptionalVariableBindings, *api.Namespace, int64) ([]cel.EvaluationResult, int64, error) {
	if f.throwError {
		return nil, 0, errors.New("test error")
	}
	return f.evaluations, 0, nil
}

func (f *fakeCelFilter) CompilationErrors() []error {
	return []error{}
}

func TestMatch(t *testing.T) {
	fakeAttr := admission.NewAttributesRecord(nil, nil, schema.GroupVersionKind{}, "default", "foo", schema.GroupVersionResource{}, "", admission.Create, nil, false, nil)
	fakeVersionedAttr, _ := admission.NewVersionedAttributes(fakeAttr, schema.GroupVersionKind{}, nil)
	fail := v1.Fail
	ignore := v1.Ignore

	cases := []struct {
		name         string
		evaluations  []cel.EvaluationResult
		throwError   bool
		shouldMatch  bool
		returnedName string
		failPolicy   *v1.FailurePolicyType
		expectError  string
	}{
		{
			name: "test single matches",
			evaluations: []cel.EvaluationResult{
				{
					EvalResult:         celtypes.True,
					ExpressionAccessor: &MatchCondition{},
				},
			},
			shouldMatch: true,
		},
		{
			name: "test multiple match",
			evaluations: []cel.EvaluationResult{
				{
					EvalResult:         celtypes.True,
					ExpressionAccessor: &MatchCondition{},
				},
				{
					EvalResult:         celtypes.True,
					ExpressionAccessor: &MatchCondition{},
				},
			},
			shouldMatch: true,
		},
		{
			name:        "test empty evals",
			evaluations: []cel.EvaluationResult{},
			shouldMatch: true,
		},
		{
			name: "test single no match",
			evaluations: []cel.EvaluationResult{
				{
					EvalResult: celtypes.False,
					ExpressionAccessor: &MatchCondition{
						Name: "test1",
					},
				},
			},
			shouldMatch:  false,
			returnedName: "test1",
		},
		{
			name: "test multiple no match",
			evaluations: []cel.EvaluationResult{
				{
					EvalResult: celtypes.False,
					ExpressionAccessor: &MatchCondition{
						Name: "test1",
					},
				},
				{
					EvalResult: celtypes.False,
					ExpressionAccessor: &MatchCondition{
						Name: "test2",
					},
				},
			},
			shouldMatch:  false,
			returnedName: "test1",
		},
		{
			name: "test mixed with no match first",
			evaluations: []cel.EvaluationResult{
				{
					EvalResult: celtypes.False,
					ExpressionAccessor: &MatchCondition{
						Name: "test1",
					},
				},
				{
					EvalResult: celtypes.True,
					ExpressionAccessor: &MatchCondition{
						Name: "test2",
					},
				},
			},
			shouldMatch:  false,
			returnedName: "test1",
		},
		{
			name: "test mixed with no match last",
			evaluations: []cel.EvaluationResult{
				{
					EvalResult: celtypes.True,
					ExpressionAccessor: &MatchCondition{
						Name: "test2",
					},
				},
				{
					EvalResult: celtypes.False,
					ExpressionAccessor: &MatchCondition{
						Name: "test1",
					},
				},
			},
			shouldMatch:  false,
			returnedName: "test1",
		},
		{
			name: "test mixed with no match middle",
			evaluations: []cel.EvaluationResult{
				{
					EvalResult: celtypes.True,
					ExpressionAccessor: &MatchCondition{
						Name: "test2",
					},
				},
				{
					EvalResult: celtypes.False,
					ExpressionAccessor: &MatchCondition{
						Name: "test1",
					},
				},
				{
					EvalResult: celtypes.True,
					ExpressionAccessor: &MatchCondition{
						Name: "test2",
					},
				},
			},
			shouldMatch:  false,
			returnedName: "test1",
		},
		{
			name: "test error, no fail policy",
			evaluations: []cel.EvaluationResult{
				{
					EvalResult:         celtypes.True,
					ExpressionAccessor: &MatchCondition{},
				},
			},
			shouldMatch: true,
			throwError:  true,
			expectError: "test error",
		},
		{
			name: "test error, fail policy fail",
			evaluations: []cel.EvaluationResult{
				{
					EvalResult:         celtypes.True,
					ExpressionAccessor: &MatchCondition{},
				},
			},
			failPolicy:  &fail,
			shouldMatch: true,
			throwError:  true,
			expectError: "test error",
		},
		{
			name: "test error, fail policy ignore",
			evaluations: []cel.EvaluationResult{
				{
					EvalResult:         celtypes.True,
					ExpressionAccessor: &MatchCondition{},
				},
			},
			failPolicy:  &ignore,
			shouldMatch: false,
			throwError:  true,
		},
		{
			name: "test mix of true, errors and false",
			evaluations: []cel.EvaluationResult{
				{
					EvalResult:         celtypes.True,
					ExpressionAccessor: &MatchCondition{},
				},
				{
					Error:              errors.New("test error"),
					ExpressionAccessor: &MatchCondition{},
				},
				{
					EvalResult:         celtypes.False,
					ExpressionAccessor: &MatchCondition{},
				},
				{
					EvalResult:         celtypes.True,
					ExpressionAccessor: &MatchCondition{},
				},
				{
					Error:              errors.New("test error"),
					ExpressionAccessor: &MatchCondition{},
				},
			},
			shouldMatch: false,
			throwError:  false,
		},
		{
			name: "test mix of true, errors and fail policy not set",
			evaluations: []cel.EvaluationResult{
				{
					EvalResult:         celtypes.True,
					ExpressionAccessor: &MatchCondition{},
				},
				{
					Error:              errors.New("test error"),
					ExpressionAccessor: &MatchCondition{},
				},
				{
					EvalResult:         celtypes.True,
					ExpressionAccessor: &MatchCondition{},
				},
				{
					Error:              errors.New("test error"),
					ExpressionAccessor: &MatchCondition{},
				},
			},
			shouldMatch: false,
			throwError:  false,
			expectError: "test error",
		},
		{
			name: "test mix of true, errors and fail policy fail",
			evaluations: []cel.EvaluationResult{
				{
					EvalResult:         celtypes.True,
					ExpressionAccessor: &MatchCondition{},
				},
				{
					Error:              errors.New("test error"),
					ExpressionAccessor: &MatchCondition{},
				},
				{
					EvalResult:         celtypes.True,
					ExpressionAccessor: &MatchCondition{},
				},
				{
					Error:              errors.New("test error"),
					ExpressionAccessor: &MatchCondition{},
				},
			},
			failPolicy:  &fail,
			shouldMatch: false,
			throwError:  false,
			expectError: "test error",
		},
		{
			name: "test mix of true, errors and fail policy ignore",
			evaluations: []cel.EvaluationResult{
				{
					EvalResult:         celtypes.True,
					ExpressionAccessor: &MatchCondition{},
				},
				{
					Error:              errors.New("test error"),
					ExpressionAccessor: &MatchCondition{},
				},
				{
					EvalResult:         celtypes.True,
					ExpressionAccessor: &MatchCondition{},
				},
				{
					Error:              errors.New("test error"),
					ExpressionAccessor: &MatchCondition{},
				},
			},
			failPolicy:  &ignore,
			shouldMatch: false,
			throwError:  false,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			m := NewMatcher(&fakeCelFilter{
				evaluations: tc.evaluations,
				throwError:  tc.throwError,
			}, tc.failPolicy, "webhook", "test", "testhook")
			ctx := context.TODO()
			matchResult := m.Match(ctx, fakeVersionedAttr, nil, nil)

			if matchResult.Error != nil {
				if len(tc.expectError) == 0 {
					t.Fatal(matchResult.Error)
				}
				if !strings.Contains(matchResult.Error.Error(), tc.expectError) {
					t.Fatalf("expected error containing %q, got %s", tc.expectError, matchResult.Error.Error())
				}
				return
			} else if len(tc.expectError) > 0 {
				t.Fatal("expected error but did not get one")
			}
			if len(tc.expectError) > 0 && matchResult.Error == nil {
				t.Errorf("expected error thrown when filter errors")
			}

			require.Equal(t, tc.shouldMatch, matchResult.Matches)
			require.Equal(t, tc.returnedName, matchResult.FailedConditionName)
		})
	}
}
