/*
Copyright 2021 The Kubernetes Authors.

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
	"fmt"
	"strings"
	"testing"

	apiextensions "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	"k8s.io/apiextensions-apiserver/pkg/apiserver/schema"
)

type validationMatcher interface {
	matches(cr CompilationResult) bool
	String() string
}

type allMatcher []validationMatcher

func matchesAll(matchers ...validationMatcher) validationMatcher {
	return allMatcher(matchers)
}

func (m allMatcher) matches(cr CompilationResult) bool {
	for _, each := range m {
		if !each.matches(cr) {
			return false
		}
	}
	return true
}

func (m allMatcher) String() string {
	if len(m) == 0 {
		return "any result"
	}
	var b strings.Builder
	for i, each := range m {
		b.WriteString(each.String())
		if i < len(m)-1 {
			b.WriteString(" and ")
		}
	}
	return b.String()
}

type fnMatcher struct {
	fn  func(CompilationResult) bool
	msg string
}

func (m fnMatcher) matches(cr CompilationResult) bool {
	return m.fn(cr)
}

func (m fnMatcher) String() string {
	return m.msg
}

type errorMatcher struct {
	errorType ErrorType
	contains  string
}

func invalidError(contains string) validationMatcher {
	return errorMatcher{errorType: ErrorTypeInvalid, contains: contains}
}

func (v errorMatcher) matches(cr CompilationResult) bool {
	return cr.Error != nil && cr.Error.Type == v.errorType && strings.Contains(cr.Error.Error(), v.contains)
}

func (v errorMatcher) String() string {
	return fmt.Sprintf("has error of type %q containing string %q", v.errorType, v.contains)
}

type noErrorMatcher struct{}

func noError() validationMatcher {
	return noErrorMatcher{}
}

func (noErrorMatcher) matches(cr CompilationResult) bool {
	return cr.Error == nil
}

func (noErrorMatcher) String() string {
	return "no error"
}

type transitionRuleMatcher bool

func transitionRule(t bool) validationMatcher {
	return transitionRuleMatcher(t)
}

func (v transitionRuleMatcher) matches(cr CompilationResult) bool {
	return cr.TransitionRule == bool(v)
}

func (v transitionRuleMatcher) String() string {
	if v {
		return "is a transition rule"
	}
	return "is not a transition rule"
}

func TestCelCompilation(t *testing.T) {
	cases := []struct {
		name            string
		input           schema.Structural
		expectedResults []validationMatcher
	}{
		{
			name: "valid object",
			input: schema.Structural{
				Generic: schema.Generic{
					Type: "object",
				},
				Properties: map[string]schema.Structural{
					"minReplicas": {
						Generic: schema.Generic{
							Type: "integer",
						},
					},
					"maxReplicas": {
						Generic: schema.Generic{
							Type: "integer",
						},
					},
				},
				Extensions: schema.Extensions{
					XValidations: apiextensions.ValidationRules{
						{
							Rule:    "self.minReplicas < self.maxReplicas",
							Message: "minReplicas should be smaller than maxReplicas",
						},
					},
				},
			},
			expectedResults: []validationMatcher{
				noError(),
			},
		},
		{
			name: "valid for string",
			input: schema.Structural{
				Generic: schema.Generic{
					Type: "string",
				},
				Extensions: schema.Extensions{
					XValidations: apiextensions.ValidationRules{
						{
							Rule:    "self.startsWith('s')",
							Message: "scoped field should start with 's'",
						},
					},
				},
			},
			expectedResults: []validationMatcher{
				noError(),
			},
		},
		{
			name: "valid for byte",
			input: schema.Structural{
				Generic: schema.Generic{
					Type: "string",
				},
				ValueValidation: &schema.ValueValidation{
					Format: "byte",
				},
				Extensions: schema.Extensions{
					XValidations: apiextensions.ValidationRules{
						{
							Rule:    "string(self).endsWith('s')",
							Message: "scoped field should end with 's'",
						},
					},
				},
			},
			expectedResults: []validationMatcher{
				noError(),
			},
		},
		{
			name: "valid for boolean",
			input: schema.Structural{
				Generic: schema.Generic{
					Type: "boolean",
				},
				Extensions: schema.Extensions{
					XValidations: apiextensions.ValidationRules{
						{
							Rule:    "self == true",
							Message: "scoped field should be true",
						},
					},
				},
			},
			expectedResults: []validationMatcher{
				noError(),
			},
		},
		{
			name: "valid for integer",
			input: schema.Structural{
				Generic: schema.Generic{
					Type: "integer",
				},
				Extensions: schema.Extensions{
					XValidations: apiextensions.ValidationRules{
						{
							Rule:    "self > 0",
							Message: "scoped field should be greater than 0",
						},
					},
				},
			},
			expectedResults: []validationMatcher{
				noError(),
			},
		},
		{
			name: "valid for number",
			input: schema.Structural{
				Generic: schema.Generic{
					Type: "number",
				},
				Extensions: schema.Extensions{
					XValidations: apiextensions.ValidationRules{
						{
							Rule:    "self > 1.0",
							Message: "scoped field should be greater than 1.0",
						},
					},
				},
			},
			expectedResults: []validationMatcher{
				noError(),
			},
		},
		{
			name: "valid nested object of object",
			input: schema.Structural{
				Generic: schema.Generic{
					Type: "object",
				},
				Properties: map[string]schema.Structural{
					"nestedObj": {
						Generic: schema.Generic{
							Type: "object",
						},
						Properties: map[string]schema.Structural{
							"val": {
								Generic: schema.Generic{
									Type: "integer",
								},
								ValueValidation: &schema.ValueValidation{
									Format: "int64",
								},
							},
						},
					},
				},
				Extensions: schema.Extensions{
					XValidations: apiextensions.ValidationRules{
						{
							Rule:    "self.nestedObj.val == 10",
							Message: "val should be equal to 10",
						},
					},
				},
			},
			expectedResults: []validationMatcher{
				noError(),
			},
		},
		{
			name: "valid nested object of array",
			input: schema.Structural{
				Generic: schema.Generic{
					Type: "object",
				},
				Properties: map[string]schema.Structural{
					"nestedObj": {
						Generic: schema.Generic{
							Type: "array",
						},
						Items: &schema.Structural{
							Generic: schema.Generic{
								Type: "array",
							},
							Items: &schema.Structural{
								Generic: schema.Generic{
									Type: "string",
								},
							},
						},
					},
				},
				Extensions: schema.Extensions{
					XValidations: apiextensions.ValidationRules{
						{
							Rule:    "size(self.nestedObj[0]) == 10",
							Message: "size of first element in nestedObj should be equal to 10",
						},
					},
				},
			},
			expectedResults: []validationMatcher{
				noError(),
			},
		},
		{
			name: "valid nested array of array",
			input: schema.Structural{
				Generic: schema.Generic{
					Type: "array",
				},
				Items: &schema.Structural{
					Generic: schema.Generic{
						Type: "array",
					},
					Items: &schema.Structural{
						Generic: schema.Generic{
							Type: "array",
						},
						Items: &schema.Structural{
							Generic: schema.Generic{
								Type: "string",
							},
						},
					},
				},
				Extensions: schema.Extensions{
					XValidations: apiextensions.ValidationRules{
						{
							Rule:    "size(self[0][0]) == 10",
							Message: "size of items under items of scoped field should be equal to 10",
						},
					},
				},
			},
			expectedResults: []validationMatcher{
				noError(),
			},
		},
		{
			name: "valid nested array of object",
			input: schema.Structural{
				Generic: schema.Generic{
					Type: "array",
				},
				Items: &schema.Structural{
					Generic: schema.Generic{
						Type: "object",
					},
					Properties: map[string]schema.Structural{
						"nestedObj": {
							Generic: schema.Generic{
								Type: "object",
							},
							Properties: map[string]schema.Structural{
								"val": {
									Generic: schema.Generic{
										Type: "integer",
									},
									ValueValidation: &schema.ValueValidation{
										Format: "int64",
									},
								},
							},
						},
					},
				},
				Extensions: schema.Extensions{
					XValidations: apiextensions.ValidationRules{
						{
							Rule:    "self[0].nestedObj.val == 10",
							Message: "val under nestedObj under properties under items should be equal to 10",
						},
					},
				},
			},
			expectedResults: []validationMatcher{
				noError(),
			},
		},
		{
			name: "valid map",
			input: schema.Structural{
				Generic: schema.Generic{
					Type: "object",
					AdditionalProperties: &schema.StructuralOrBool{
						Bool: true,
						Structural: &schema.Structural{
							Generic: schema.Generic{
								Type:     "boolean",
								Nullable: false,
							},
						},
					},
				},
				Extensions: schema.Extensions{
					XValidations: apiextensions.ValidationRules{
						{
							Rule:    "size(self) > 0",
							Message: "size of scoped field should be greater than 0",
						},
					},
				},
			},
			expectedResults: []validationMatcher{
				noError(),
			},
		},
		{
			name: "invalid checking for number",
			input: schema.Structural{
				Generic: schema.Generic{
					Type: "number",
				},
				Extensions: schema.Extensions{
					XValidations: apiextensions.ValidationRules{
						{
							Rule:    "size(self) == 10",
							Message: "size of scoped field should be equal to 10",
						},
					},
				},
			},
			expectedResults: []validationMatcher{
				invalidError("compilation failed"),
			},
		},
		{
			name: "compilation failure",
			input: schema.Structural{
				Generic: schema.Generic{
					Type: "integer",
				},
				Extensions: schema.Extensions{
					XValidations: apiextensions.ValidationRules{
						{
							Rule:    "size(self) == 10",
							Message: "size of scoped field should be equal to 10",
						},
					},
				},
			},
			expectedResults: []validationMatcher{
				invalidError("compilation failed"),
			},
		},
		{
			name: "valid for escaping",
			input: schema.Structural{
				Generic: schema.Generic{
					Type: "object",
				},
				Properties: map[string]schema.Structural{
					"namespace": {
						Generic: schema.Generic{
							Type: "array",
						},
						Items: &schema.Structural{
							Generic: schema.Generic{
								Type: "array",
							},
							Items: &schema.Structural{
								Generic: schema.Generic{
									Type: "string",
								},
							},
						},
					},
					"if": {
						Generic: schema.Generic{
							Type: "integer",
						},
					},
					"self": {
						Generic: schema.Generic{
							Type: "integer",
						},
					},
				},
				Extensions: schema.Extensions{
					XValidations: apiextensions.ValidationRules{
						{
							Rule:    "size(self.__namespace__[0]) == 10",
							Message: "size of first element in nestedObj should be equal to 10",
						},
						{
							Rule: "self.__if__ == 10",
						},
						{
							Rule: "self.self == 10",
						},
					},
				},
			},
			expectedResults: []validationMatcher{
				noError(),
				noError(),
				noError(),
			},
		},
		{
			name: "invalid for escaping",
			input: schema.Structural{
				Generic: schema.Generic{
					Type: "object",
				},
				Properties: map[string]schema.Structural{
					"namespace": {
						Generic: schema.Generic{
							Type: "array",
						},
						Items: &schema.Structural{
							Generic: schema.Generic{
								Type: "array",
							},
							Items: &schema.Structural{
								Generic: schema.Generic{
									Type: "string",
								},
							},
						},
					},
					"if": {
						Generic: schema.Generic{
							Type: "integer",
						},
					},
					"self": {
						Generic: schema.Generic{
							Type: "integer",
						},
					},
				},
				Extensions: schema.Extensions{
					XValidations: apiextensions.ValidationRules{
						{
							Rule:    "size(self.namespace[0]) == 10",
							Message: "size of first element in nestedObj should be equal to 10",
						},
						{
							Rule: "self.if == 10",
						},
						{
							Rule: "self == 10",
						},
					},
				},
			},
			expectedResults: []validationMatcher{
				invalidError("undefined field 'namespace'"),
				invalidError("undefined field 'if'"),
				invalidError("found no matching overload"),
			},
		},
		{
			name: "transition rule identified",
			input: schema.Structural{
				Generic: schema.Generic{
					Type: "integer",
				},
				Extensions: schema.Extensions{
					XValidations: apiextensions.ValidationRules{
						{Rule: "self > 0"},
						{Rule: "self >= oldSelf"},
					},
				},
			},
			expectedResults: []validationMatcher{
				matchesAll(noError(), transitionRule(false)),
				matchesAll(noError(), transitionRule(true)),
			},
		},
		{
			name: "whitespace-only rule",
			input: schema.Structural{
				Generic: schema.Generic{
					Type: "object",
				},
				Extensions: schema.Extensions{
					XValidations: apiextensions.ValidationRules{
						{Rule: " \t"},
					},
				},
			},
			expectedResults: []validationMatcher{
				matchesAll(
					noError(),
					fnMatcher{
						msg: "program is nil",
						fn: func(cr CompilationResult) bool {
							return cr.Program == nil
						},
					}),
			},
		},
		{
			name: "expression must evaluate to bool",
			input: schema.Structural{
				Generic: schema.Generic{
					Type: "object",
				},
				Extensions: schema.Extensions{
					XValidations: apiextensions.ValidationRules{
						{Rule: "42"},
					},
				},
			},
			expectedResults: []validationMatcher{
				invalidError("must evaluate to a bool"),
			},
		},
	}

	for _, tt := range cases {
		t.Run(tt.name, func(t *testing.T) {
			compilationResults, err := Compile(&tt.input, false, PerCallLimit)
			if err != nil {
				t.Errorf("Expected no error, but got: %v", err)
			}

			if len(compilationResults) != len(tt.input.XValidations) {
				t.Fatalf("compilation did not produce one result per rule")
			}

			if len(compilationResults) != len(tt.expectedResults) {
				t.Fatalf("one test expectation per rule is required")
			}

			for i, expectedResult := range tt.expectedResults {
				if !expectedResult.matches(compilationResults[i]) {
					t.Errorf("result %d does not match expectation: %v", i+1, expectedResult)
				}
			}
		})
	}
}
