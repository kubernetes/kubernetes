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
	"math"
	"strings"
	"testing"

	celgo "github.com/google/cel-go/cel"
	"github.com/google/cel-go/common/types"
	"github.com/google/cel-go/common/types/ref"

	apiextensions "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	"k8s.io/apiextensions-apiserver/pkg/apiserver/schema"
	"k8s.io/apiextensions-apiserver/pkg/apiserver/schema/cel/model"
	apiextensionsfeatures "k8s.io/apiextensions-apiserver/pkg/features"
	"k8s.io/apimachinery/pkg/util/version"
	celconfig "k8s.io/apiserver/pkg/apis/cel"
	"k8s.io/apiserver/pkg/cel"
	"k8s.io/apiserver/pkg/cel/environment"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/utils/ptr"
)

const (
	costLimit = 100000000
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
	errorType cel.ErrorType
	contains  string
}

func invalidError(contains string) validationMatcher {
	return errorMatcher{errorType: cel.ErrorTypeInvalid, contains: contains}
}

func (v errorMatcher) matches(cr CompilationResult) bool {
	return cr.Error != nil && cr.Error.Type == v.errorType && strings.Contains(cr.Error.Error(), v.contains)
}

func (v errorMatcher) String() string {
	return fmt.Sprintf("has error of type %q containing string %q", v.errorType, v.contains)
}

type messageExpressionErrorMatcher struct {
	contains string
}

func messageExpressionError(contains string) validationMatcher {
	return messageExpressionErrorMatcher{contains: contains}
}

func (m messageExpressionErrorMatcher) matches(cr CompilationResult) bool {
	return cr.MessageExpressionError != nil && cr.MessageExpressionError.Type == cel.ErrorTypeInvalid && strings.Contains(cr.MessageExpressionError.Error(), m.contains)
}

func (m messageExpressionErrorMatcher) String() string {
	return fmt.Sprintf("has messageExpression error containing string %q", m.contains)
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
	return cr.UsesOldSelf == bool(v)
}

func (v transitionRuleMatcher) String() string {
	if v {
		return "is a transition rule"
	}
	return "is not a transition rule"
}

func TestCelCompilation(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, apiextensionsfeatures.CRDValidationRatcheting, true)
	cases := []struct {
		name            string
		input           schema.Structural
		expectedResults []validationMatcher
		unmodified      bool
	}{
		{
			name: "optional primitive transition rule type checking",
			input: schema.Structural{
				Generic: schema.Generic{
					Type: "integer",
				},
				ValidationExtensions: schema.ValidationExtensions{
					XValidations: apiextensions.ValidationRules{
						{Rule: "self >= oldSelf.value()", OptionalOldSelf: ptr.To(true)},
						{Rule: "self >= oldSelf.orValue(1)", OptionalOldSelf: ptr.To(true)},
						{Rule: "oldSelf.hasValue() ? self >= oldSelf.value() : true", OptionalOldSelf: ptr.To(true)},
						{Rule: "self >= oldSelf", OptionalOldSelf: ptr.To(true)},
						{Rule: "self >= oldSelf.orValue('')", OptionalOldSelf: ptr.To(true)},
					},
				},
			},
			expectedResults: []validationMatcher{
				matchesAll(noError(), transitionRule(true)),
				matchesAll(noError(), transitionRule(true)),
				matchesAll(noError(), transitionRule(true)),
				matchesAll(invalidError("optional")),
				matchesAll(invalidError("orValue")),
			},
		},
		{
			name: "optional complex transition rule type checking",
			input: schema.Structural{
				Generic: schema.Generic{
					Type: "object",
				},
				Properties: map[string]schema.Structural{
					"i": {Generic: schema.Generic{Type: "integer"}},
					"b": {Generic: schema.Generic{Type: "boolean"}},
					"s": {Generic: schema.Generic{Type: "string"}},
					"a": {
						Generic: schema.Generic{Type: "array"},
						Items:   &schema.Structural{Generic: schema.Generic{Type: "integer"}},
					},
					"o": {
						Generic: schema.Generic{Type: "object"},
						Properties: map[string]schema.Structural{
							"i": {Generic: schema.Generic{Type: "integer"}},
							"b": {Generic: schema.Generic{Type: "boolean"}},
							"s": {Generic: schema.Generic{Type: "string"}},
							"a": {
								Generic: schema.Generic{Type: "array"},
								Items:   &schema.Structural{Generic: schema.Generic{Type: "integer"}},
							},
							"o": {
								Generic: schema.Generic{Type: "object"},
							},
						},
					},
				},
				ValidationExtensions: schema.ValidationExtensions{
					XValidations: apiextensions.ValidationRules{
						{Rule: "self.i >= oldSelf.i.value()", OptionalOldSelf: ptr.To(true)},
						{Rule: "self.s == oldSelf.s.value()", OptionalOldSelf: ptr.To(true)},
						{Rule: "self.b == oldSelf.b.value()", OptionalOldSelf: ptr.To(true)},
						{Rule: "self.o == oldSelf.o.value()", OptionalOldSelf: ptr.To(true)},
						{Rule: "self.o.i >= oldSelf.o.i.value()", OptionalOldSelf: ptr.To(true)},
						{Rule: "self.o.s == oldSelf.o.s.value()", OptionalOldSelf: ptr.To(true)},
						{Rule: "self.o.b == oldSelf.o.b.value()", OptionalOldSelf: ptr.To(true)},
						{Rule: "self.o.o == oldSelf.o.o.value()", OptionalOldSelf: ptr.To(true)},
						{Rule: "self.o.i >= oldSelf.o.i.orValue(1)", OptionalOldSelf: ptr.To(true)},
						{Rule: "oldSelf.hasValue() ? self.o.i >= oldSelf.o.i.value() : true", OptionalOldSelf: ptr.To(true)},
						{Rule: "self.o.i >= oldSelf.o.i", OptionalOldSelf: ptr.To(true)},
						{Rule: "self.o.i >= oldSelf.o.s.orValue(0)", OptionalOldSelf: ptr.To(true)},
					},
				},
			},
			expectedResults: []validationMatcher{
				matchesAll(noError(), transitionRule(true)),
				matchesAll(noError(), transitionRule(true)),
				matchesAll(noError(), transitionRule(true)),
				matchesAll(noError(), transitionRule(true)),
				matchesAll(noError(), transitionRule(true)),
				matchesAll(noError(), transitionRule(true)),
				matchesAll(noError(), transitionRule(true)),
				matchesAll(noError(), transitionRule(true)),
				matchesAll(noError(), transitionRule(true)),
				matchesAll(noError(), transitionRule(true)),
				matchesAll(invalidError("optional")),
				matchesAll(invalidError("orValue")),
			},
		},
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
				ValidationExtensions: schema.ValidationExtensions{
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
				ValidationExtensions: schema.ValidationExtensions{
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
				ValidationExtensions: schema.ValidationExtensions{
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
				ValidationExtensions: schema.ValidationExtensions{
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
				ValidationExtensions: schema.ValidationExtensions{
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
				ValidationExtensions: schema.ValidationExtensions{
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
				ValidationExtensions: schema.ValidationExtensions{
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
				ValidationExtensions: schema.ValidationExtensions{
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
				ValidationExtensions: schema.ValidationExtensions{
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
				ValidationExtensions: schema.ValidationExtensions{
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
				},
				AdditionalProperties: &schema.StructuralOrBool{
					Bool: true,
					Structural: &schema.Structural{
						Generic: schema.Generic{
							Type:     "boolean",
							Nullable: false,
						},
					},
				},
				ValidationExtensions: schema.ValidationExtensions{
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
				ValidationExtensions: schema.ValidationExtensions{
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
				ValidationExtensions: schema.ValidationExtensions{
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
				ValidationExtensions: schema.ValidationExtensions{
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
				ValidationExtensions: schema.ValidationExtensions{
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
				noError(),
				noError(),
				invalidError("found no matching overload"),
			},
		},
		{
			name: "transition rule identified",
			input: schema.Structural{
				Generic: schema.Generic{
					Type: "integer",
				},
				ValidationExtensions: schema.ValidationExtensions{
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
				ValidationExtensions: schema.ValidationExtensions{
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
				ValidationExtensions: schema.ValidationExtensions{
					XValidations: apiextensions.ValidationRules{
						{Rule: "42"},
					},
				},
			},
			expectedResults: []validationMatcher{
				invalidError("must evaluate to a bool"),
			},
		},
		{
			name: "messageExpression inclusion",
			input: schema.Structural{
				Generic: schema.Generic{
					Type: "string",
				},
				ValidationExtensions: schema.ValidationExtensions{
					XValidations: apiextensions.ValidationRules{
						{
							Rule:              "self.startsWith('s')",
							MessageExpression: `"scoped field should start with 's'"`,
						},
					},
				},
			},
			expectedResults: []validationMatcher{
				noError(),
			},
		},
		{
			name: "messageExpression must evaluate to a string",
			input: schema.Structural{
				Generic: schema.Generic{
					Type: "integer",
				},
				ValidationExtensions: schema.ValidationExtensions{
					XValidations: apiextensions.ValidationRules{
						{
							Rule:              "self == 5",
							MessageExpression: `42`,
						},
					},
				},
			},
			expectedResults: []validationMatcher{
				messageExpressionError("must evaluate to a string"),
			},
		},
		{
			name: "messageExpression syntax error",
			input: schema.Structural{
				Generic: schema.Generic{
					Type: "number",
				},
				ValidationExtensions: schema.ValidationExtensions{
					XValidations: apiextensions.ValidationRules{
						{
							Rule:              "self < 32.0",
							MessageExpression: `"abc`,
						},
					},
				},
			},
			expectedResults: []validationMatcher{
				messageExpressionError("messageExpression compilation failed"),
			},
		},
		{
			name: "unmodified expression may use CEL environment features planned to be added in future releases",
			input: schema.Structural{
				Generic: schema.Generic{
					Type: "object",
				},
				ValidationExtensions: schema.ValidationExtensions{
					XValidations: apiextensions.ValidationRules{
						{Rule: "fakeFunction('abc') == 'ABC'"},
					},
				},
			},
			unmodified: true,
			expectedResults: []validationMatcher{
				noError(),
			},
		},
		{
			name: "modified expressions may not use CEL environment features planned to be added in future releases",
			input: schema.Structural{
				Generic: schema.Generic{
					Type: "object",
				},
				ValidationExtensions: schema.ValidationExtensions{
					XValidations: apiextensions.ValidationRules{
						{Rule: "fakeFunction('abc') == 'ABC'"},
					},
				},
			},
			unmodified: false,
			expectedResults: []validationMatcher{
				invalidError("undeclared reference to 'fakeFunction'"),
			},
		},
	}

	for _, tt := range cases {
		t.Run(tt.name, func(t *testing.T) {
			env, err := environment.MustBaseEnvSet(environment.DefaultCompatibilityVersion(), true).Extend(
				environment.VersionedOptions{
					IntroducedVersion: version.MajorMinor(1, 999),
					EnvOptions:        []celgo.EnvOption{celgo.Lib(&fakeLib{})},
				})
			if err != nil {
				t.Fatal(err)
			}
			loader := NewExpressionsEnvLoader()
			if tt.unmodified {
				loader = StoredExpressionsEnvLoader()
			}
			compilationResults, err := Compile(&tt.input, model.SchemaDeclType(&tt.input, false), celconfig.PerCallLimit, env, loader)
			if err != nil {
				t.Fatalf("Expected no error, but got: %v", err)
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

// take a single rule type in (string/number/map/etc.) and return appropriate values for
// Type, Format, and XIntOrString
func parseRuleType(ruleType string) (string, string, bool) {
	if ruleType == "duration" || ruleType == "date" || ruleType == "date-time" {
		return "string", ruleType, false
	}
	if ruleType == "int-or-string" {
		return "", "", true
	}
	return ruleType, "", false
}

func genArrayWithRule(arrayType, rule string) func(maxItems *int64) *schema.Structural {
	passedType, passedFormat, xIntString := parseRuleType(arrayType)
	return func(maxItems *int64) *schema.Structural {
		return &schema.Structural{
			Generic: schema.Generic{
				Type: "array",
			},
			Items: &schema.Structural{
				Generic: schema.Generic{
					Type: passedType,
				},
				ValueValidation: &schema.ValueValidation{
					Format: passedFormat,
				},
				Extensions: schema.Extensions{
					XIntOrString: xIntString,
				},
			},
			ValueValidation: &schema.ValueValidation{
				MaxItems: maxItems,
			},
			ValidationExtensions: schema.ValidationExtensions{
				XValidations: apiextensions.ValidationRules{
					{
						Rule: rule,
					},
				},
			},
		}
	}
}

func genArrayOfArraysWithRule(arrayType, rule string) func(maxItems *int64) *schema.Structural {
	return func(maxItems *int64) *schema.Structural {
		return &schema.Structural{
			Generic: schema.Generic{
				Type: "array",
			},
			Items: &schema.Structural{
				Generic: schema.Generic{
					Type: "array",
				},
				Items: &schema.Structural{
					Generic: schema.Generic{
						Type: arrayType,
					},
				},
			},
			ValueValidation: &schema.ValueValidation{
				MaxItems: maxItems,
			},
			ValidationExtensions: schema.ValidationExtensions{
				XValidations: apiextensions.ValidationRules{
					{
						Rule: rule,
					},
				},
			},
		}
	}
}

func genObjectArrayWithRule(rule string) func(maxItems *int64) *schema.Structural {
	return func(maxItems *int64) *schema.Structural {
		return &schema.Structural{
			Generic: schema.Generic{
				Type: "array",
			},
			Items: &schema.Structural{
				Generic: schema.Generic{
					Type: "object",
				},
				Properties: map[string]schema.Structural{
					"required": {
						Generic: schema.Generic{
							Type: "string",
						},
					},
					"optional": {
						Generic: schema.Generic{
							Type: "string",
						},
					},
				},
				ValueValidation: &schema.ValueValidation{
					Required: []string{"required"},
				},
			},
			ValueValidation: &schema.ValueValidation{
				MaxItems: maxItems,
			},
			ValidationExtensions: schema.ValidationExtensions{
				XValidations: apiextensions.ValidationRules{
					{
						Rule: rule,
					},
				},
			},
		}
	}
}

func getMapArrayWithRule(mapType, rule string) func(maxItems *int64) *schema.Structural {
	return func(maxItems *int64) *schema.Structural {
		return &schema.Structural{
			Generic: schema.Generic{
				Type: "array",
			},
			Items: &schema.Structural{
				Generic: schema.Generic{
					Type: "object",
				},
				AdditionalProperties: &schema.StructuralOrBool{Structural: &schema.Structural{
					Generic: schema.Generic{
						Type: mapType,
					},
				}},
			},
			ValueValidation: &schema.ValueValidation{
				MaxItems: maxItems,
			},
			ValidationExtensions: schema.ValidationExtensions{
				XValidations: apiextensions.ValidationRules{
					{
						Rule: rule,
					},
				},
			},
		}
	}
}

func genMapWithRule(mapType, rule string) func(maxProperties *int64) *schema.Structural {
	passedType, passedFormat, xIntString := parseRuleType(mapType)
	return func(maxProperties *int64) *schema.Structural {
		return &schema.Structural{
			Generic: schema.Generic{
				Type: "object",
			},
			AdditionalProperties: &schema.StructuralOrBool{Structural: &schema.Structural{
				Generic: schema.Generic{
					Type: passedType,
				},
				ValueValidation: &schema.ValueValidation{
					Format: passedFormat,
				},
				Extensions: schema.Extensions{
					XIntOrString: xIntString,
				},
			}},
			ValueValidation: &schema.ValueValidation{
				MaxProperties: maxProperties,
			},
			ValidationExtensions: schema.ValidationExtensions{
				XValidations: apiextensions.ValidationRules{
					{
						Rule: rule,
					},
				},
			},
		}
	}
}

func genStringWithRule(rule string) func(maxLength *int64) *schema.Structural {
	return func(maxLength *int64) *schema.Structural {
		return &schema.Structural{
			Generic: schema.Generic{
				Type: "string",
			},
			ValueValidation: &schema.ValueValidation{
				MaxLength: maxLength,
			},
			ValidationExtensions: schema.ValidationExtensions{
				XValidations: apiextensions.ValidationRules{
					{
						Rule: rule,
					},
				},
			},
		}
	}
}

// genEnumWithRuleAndValues creates a function that accepts an optional maxLength
// with given validation rule and a set of enum values, following the convention of existing tests.
// The test has two checks, first with maxLength unset to check if maxLength can be concluded from enums,
// second with maxLength set to ensure it takes precedence.
func genEnumWithRuleAndValues(rule string, values ...string) func(maxLength *int64) *schema.Structural {
	enums := make([]schema.JSON, 0, len(values))
	for _, v := range values {
		enums = append(enums, schema.JSON{Object: v})
	}
	return func(maxLength *int64) *schema.Structural {
		return &schema.Structural{
			Generic: schema.Generic{
				Type: "string",
			},
			ValueValidation: &schema.ValueValidation{
				MaxLength: maxLength,
				Enum:      enums,
			},
			ValidationExtensions: schema.ValidationExtensions{
				XValidations: apiextensions.ValidationRules{
					{
						Rule: rule,
					},
				},
			},
		}
	}
}

func genBytesWithRule(rule string) func(maxLength *int64) *schema.Structural {
	return func(maxLength *int64) *schema.Structural {
		return &schema.Structural{
			Generic: schema.Generic{
				Type: "string",
			},
			ValueValidation: &schema.ValueValidation{
				MaxLength: maxLength,
				Format:    "byte",
			},
			ValidationExtensions: schema.ValidationExtensions{
				XValidations: apiextensions.ValidationRules{
					{
						Rule: rule,
					},
				},
			},
		}
	}
}

func genNestedSpecWithRule(rule string) func(maxLength *int64) *schema.Structural {
	return func(maxLength *int64) *schema.Structural {
		return &schema.Structural{
			Generic: schema.Generic{
				Type: "object",
			},
			AdditionalProperties: &schema.StructuralOrBool{Structural: &schema.Structural{
				Generic: schema.Generic{
					Type: "string",
				},
				ValueValidation: &schema.ValueValidation{
					MaxLength: maxLength,
				},
			}},
			ValueValidation: &schema.ValueValidation{
				MaxProperties: maxLength,
			},
			ValidationExtensions: schema.ValidationExtensions{
				XValidations: apiextensions.ValidationRules{
					{
						Rule: rule,
					},
				},
			},
		}
	}
}

func genAllMaxNestedSpecWithRootRule(rule string) func(maxLength *int64) *schema.Structural {
	return func(maxLength *int64) *schema.Structural {
		return &schema.Structural{
			Generic: schema.Generic{
				Type: "array",
			},
			Items: &schema.Structural{
				Generic: schema.Generic{
					Type: "object",
				},
				AdditionalProperties: &schema.StructuralOrBool{Structural: &schema.Structural{
					Generic: schema.Generic{
						Type: "object",
					},
					ValueValidation: &schema.ValueValidation{
						Required:      []string{"required"},
						MaxProperties: maxLength,
					},
					Properties: map[string]schema.Structural{
						"required": {
							Generic: schema.Generic{
								Type: "string",
							},
						},
						"optional": {
							Generic: schema.Generic{
								Type: "string",
							},
						},
					},
				}},
				ValueValidation: &schema.ValueValidation{
					MaxProperties: maxLength,
				},
			},
			ValueValidation: &schema.ValueValidation{
				MaxItems: maxLength,
			},
			ValidationExtensions: schema.ValidationExtensions{
				XValidations: apiextensions.ValidationRules{
					{
						Rule: rule,
					},
				},
			},
		}
	}
}

func genOneMaxNestedSpecWithRootRule(rule string) func(maxLength *int64) *schema.Structural {
	return func(maxLength *int64) *schema.Structural {
		return &schema.Structural{
			Generic: schema.Generic{
				Type: "array",
			},
			Items: &schema.Structural{
				Generic: schema.Generic{
					Type: "object",
				},
				AdditionalProperties: &schema.StructuralOrBool{Structural: &schema.Structural{
					Generic: schema.Generic{
						Type: "object",
					},
					ValueValidation: &schema.ValueValidation{
						Required: []string{"required"},
					},
					Properties: map[string]schema.Structural{
						"required": {
							Generic: schema.Generic{
								Type: "string",
							},
						},
						"optional": {
							Generic: schema.Generic{
								Type: "string",
							},
						},
					},
				}},
				ValueValidation: &schema.ValueValidation{
					MaxProperties: maxLength,
				},
			},
			ValidationExtensions: schema.ValidationExtensions{
				XValidations: apiextensions.ValidationRules{
					{
						Rule: rule,
					},
				},
			},
		}
	}
}

func genObjectForMap() *schema.Structural {
	return &schema.Structural{
		Generic: schema.Generic{
			Type: "object",
		},
		Properties: map[string]schema.Structural{
			"required": {
				Generic: schema.Generic{
					Type: "string",
				},
			},
			"optional": {
				Generic: schema.Generic{
					Type: "string",
				},
			},
		},
		ValueValidation: &schema.ValueValidation{
			Required: []string{"required"},
		},
	}
}

func genArrayForMap() *schema.Structural {
	return &schema.Structural{
		Generic: schema.Generic{
			Type: "array",
		},
		Items: &schema.Structural{
			Generic: schema.Generic{
				Type: "number",
			},
		},
	}
}

func genMapForMap() *schema.Structural {
	return &schema.Structural{
		Generic: schema.Generic{
			Type: "object",
		},
		AdditionalProperties: &schema.StructuralOrBool{Structural: &schema.Structural{
			Generic: schema.Generic{
				Type: "number",
			},
		}},
	}
}

func genMapWithCustomItemRule(item *schema.Structural, rule string) func(maxProperties *int64) *schema.Structural {
	return func(maxProperties *int64) *schema.Structural {
		return &schema.Structural{
			Generic: schema.Generic{
				Type: "object",
			},
			AdditionalProperties: &schema.StructuralOrBool{Structural: item},
			ValueValidation: &schema.ValueValidation{
				MaxProperties: maxProperties,
			},
			ValidationExtensions: schema.ValidationExtensions{
				XValidations: apiextensions.ValidationRules{
					{
						Rule: rule,
					},
				},
			},
		}
	}
}

// schemaChecker checks the cost of the validation rule declared in the provided schema (it requires there be exactly one rule)
// and checks that the resulting equals the expectedCost if expectedCost is non-zero, and that the resulting cost is >= expectedCostExceedsLimit
// if expectedCostExceedsLimit is non-zero. Typically, only expectedCost or expectedCostExceedsLimit is non-zero, not both.
func schemaChecker(schema *schema.Structural, expectedCost uint64, expectedCostExceedsLimit uint64, t *testing.T) func(t *testing.T) {
	return func(t *testing.T) {
		compilationResults, err := Compile(schema, model.SchemaDeclType(schema, false), celconfig.PerCallLimit, environment.MustBaseEnvSet(environment.DefaultCompatibilityVersion(), true), NewExpressionsEnvLoader())
		if err != nil {
			t.Fatalf("Expected no error, got: %v", err)
		}
		if len(compilationResults) != 1 {
			t.Fatalf("Expected one rule, got: %d", len(compilationResults))
		}
		result := compilationResults[0]
		if result.Error != nil {
			t.Errorf("Expected no compile-time error, got: %v", result.Error)
		}
		if expectedCost > 0 {
			if result.MaxCost != expectedCost {
				t.Errorf("Wrong cost (expected %d, got %d)", expectedCost, result.MaxCost)
			}
		}
		if expectedCostExceedsLimit > 0 {
			if result.MaxCost < expectedCostExceedsLimit {
				t.Errorf("Cost did not exceed limit as expected (expected more than %d, got %d)", expectedCostExceedsLimit, result.MaxCost)
			}
		}
	}
}

func TestCostEstimation(t *testing.T) {
	cases := []struct {
		name            string
		schemaGenerator func(maxLength *int64) *schema.Structural
		setMaxElements  int64

		// calc costs expectations are checked against the generated schema without any max element limits set
		expectedCalcCost           uint64
		expectCalcCostExceedsLimit uint64

		// calc costs expectations are checked against the generated schema with max element limits set
		expectedSetCost             uint64
		expectedSetCostExceedsLimit uint64
	}{
		{
			name:             "number array with all",
			schemaGenerator:  genArrayWithRule("number", "self.all(x, true)"),
			expectedCalcCost: 4718591,
			setMaxElements:   10,
			expectedSetCost:  32,
		},
		{
			name:             "string array with all",
			schemaGenerator:  genArrayWithRule("string", "self.all(x, true)"),
			expectedCalcCost: 3145727,
			setMaxElements:   20,
			expectedSetCost:  62,
		},
		{
			name:             "boolean array with all",
			schemaGenerator:  genArrayWithRule("boolean", "self.all(x, true)"),
			expectedCalcCost: 1887437,
			setMaxElements:   5,
			expectedSetCost:  17,
		},
		// all array-of-array tests should have the same expected cost along the same expression,
		// since arrays-of-arrays are serialized the same in minimized form regardless of item type
		// of the subarray ([[], [], ...])
		{
			name:             "array of number arrays with all",
			schemaGenerator:  genArrayOfArraysWithRule("number", "self.all(x, true)"),
			expectedCalcCost: 3145727,
			setMaxElements:   100,
			expectedSetCost:  302,
		},
		{
			name:             "array of objects with all",
			schemaGenerator:  genObjectArrayWithRule("self.all(x, true)"),
			expectedCalcCost: 555128,
			setMaxElements:   50,
			expectedSetCost:  152,
		},
		{
			name:             "map of numbers with all",
			schemaGenerator:  genMapWithRule("number", "self.all(x, true)"),
			expectedCalcCost: 1348169,
			setMaxElements:   10,
			expectedSetCost:  32,
		},
		{
			name:             "map of numbers with has",
			schemaGenerator:  genMapWithRule("number", "has(self.x)"),
			expectedCalcCost: 0,
			setMaxElements:   100,
			expectedSetCost:  0,
		},
		{
			name:             "map of strings with all",
			schemaGenerator:  genMapWithRule("string", "self.all(x, true)"),
			expectedCalcCost: 1179647,
			setMaxElements:   3,
			expectedSetCost:  11,
		},
		{
			name:             "map of strings with has",
			schemaGenerator:  genMapWithRule("string", "has(self.x)"),
			expectedCalcCost: 0,
			setMaxElements:   550,
			expectedSetCost:  0,
		},
		{
			name:             "map of booleans with all",
			schemaGenerator:  genMapWithRule("boolean", "self.all(x, true)"),
			expectedCalcCost: 943718,
			setMaxElements:   100,
			expectedSetCost:  302,
		},
		{
			name:             "map of booleans with has",
			schemaGenerator:  genMapWithRule("boolean", "has(self.x)"),
			expectedCalcCost: 0,
			setMaxElements:   1024,
			expectedSetCost:  0,
		},
		{
			name:             "string with contains",
			schemaGenerator:  genStringWithRule("self.contains('test')"),
			expectedCalcCost: 314574,
			setMaxElements:   10,
			expectedSetCost:  5,
		},
		{
			name:             "string with startsWith",
			schemaGenerator:  genStringWithRule("self.startsWith('test')"),
			expectedCalcCost: 2,
			setMaxElements:   15,
			expectedSetCost:  2,
		},
		{
			name:             "string with endsWith",
			schemaGenerator:  genStringWithRule("self.endsWith('test')"),
			expectedCalcCost: 2,
			setMaxElements:   30,
			expectedSetCost:  2,
		},
		{
			name:             "concat string",
			schemaGenerator:  genStringWithRule(`size(self + "hello") > size("hello")`),
			expectedCalcCost: 314578,
			setMaxElements:   4,
			expectedSetCost:  7,
		},
		{
			name:             "index of array with numbers",
			schemaGenerator:  genArrayWithRule("number", "self[1] == 0.0"),
			expectedCalcCost: 2,
			setMaxElements:   5000,
			expectedSetCost:  2,
		},
		{
			name:             "index of array with strings",
			schemaGenerator:  genArrayWithRule("string", "self[1] == self[1]"),
			expectedCalcCost: 314577,
			setMaxElements:   8,
			expectedSetCost:  314577,
		},
		{
			name:                       "O(n^2) loop with numbers",
			schemaGenerator:            genArrayWithRule("number", "self.all(x, self.all(y, true))"),
			expectCalcCostExceedsLimit: costLimit,
			setMaxElements:             10,
			expectedSetCost:            352,
		},
		{
			name:                       "O(n^3) loop with numbers",
			schemaGenerator:            genArrayWithRule("number", "self.all(x, self.all(y, self.all(z, true)))"),
			expectCalcCostExceedsLimit: costLimit,
			setMaxElements:             10,
			expectedSetCost:            3552,
		},
		{
			name:             "regex matches simple",
			schemaGenerator:  genStringWithRule(`self.matches("x")`),
			expectedCalcCost: 314574,
			setMaxElements:   50,
			expectedSetCost:  22,
		},
		{
			name:             "regex matches empty string",
			schemaGenerator:  genStringWithRule(`"".matches("(((((((((())))))))))[0-9]")`),
			expectedCalcCost: 7,
			setMaxElements:   10,
			expectedSetCost:  7,
		},
		{
			name:             "regex matches empty regex",
			schemaGenerator:  genStringWithRule(`self.matches("")`),
			expectedCalcCost: 1,
			setMaxElements:   100,
			expectedSetCost:  1,
		},
		{
			name:             "map of strings with value length",
			schemaGenerator:  genNestedSpecWithRule("self.all(x, x.contains(self[x]))"),
			expectedCalcCost: 2752507,
			setMaxElements:   10,
			expectedSetCost:  72,
		},
		{
			name:             "set array maxLength to zero",
			schemaGenerator:  genArrayWithRule("number", "self[3] == 0.0"),
			expectedCalcCost: 2,
			setMaxElements:   0,
			expectedSetCost:  2,
		},
		{
			name:             "set map maxLength to zero",
			schemaGenerator:  genMapWithRule("number", `self["x"] == 0.0`),
			expectedCalcCost: 2,
			setMaxElements:   0,
			expectedSetCost:  2,
		},
		{
			name:             "set string maxLength to zero",
			schemaGenerator:  genStringWithRule(`self == "x"`),
			expectedCalcCost: 2,
			setMaxElements:   0,
			expectedSetCost:  1,
		},
		{
			name:             "set bytes maxLength to zero",
			schemaGenerator:  genBytesWithRule(`self == b"x"`),
			expectedCalcCost: 2,
			setMaxElements:   0,
			expectedSetCost:  1,
		},
		{
			name:             "set maxLength greater than estimated maxLength",
			schemaGenerator:  genArrayWithRule("number", "self.all(x, x == 0.0)"),
			expectedCalcCost: 6291454,
			setMaxElements:   3 * 1024 * 2048,
			expectedSetCost:  25165826,
		},
		{
			name:             "nested types with root rule with all supporting maxLength",
			schemaGenerator:  genAllMaxNestedSpecWithRootRule(`self.all(x, x["y"].required == "z")`),
			expectedCalcCost: 7340027,
			setMaxElements:   10,
			expectedSetCost:  72,
		},
		{
			name:             "nested types with root rule with one supporting maxLength",
			schemaGenerator:  genOneMaxNestedSpecWithRootRule(`self.all(x, x["y"].required == "z")`),
			expectedCalcCost: 7340027,
			setMaxElements:   10,
			expectedSetCost:  7340027,
		},
		{
			name:             "int-or-string array with all",
			schemaGenerator:  genArrayWithRule("int-or-string", "self.all(x, true)"),
			expectedCalcCost: 4718591,
			setMaxElements:   10,
			expectedSetCost:  32,
		},
		{
			name:             "index of array with int-or-strings",
			schemaGenerator:  genArrayWithRule("int-or-string", "self[0] == 5"),
			expectedCalcCost: 3,
			setMaxElements:   10,
			expectedSetCost:  3,
		},
		{
			name:             "index of array with booleans",
			schemaGenerator:  genArrayWithRule("boolean", "self[0] == false"),
			expectedCalcCost: 2,
			setMaxElements:   25,
			expectedSetCost:  2,
		},
		{
			name:             "index of array of objects",
			schemaGenerator:  genObjectArrayWithRule("self[0] == null"),
			expectedCalcCost: 2,
			setMaxElements:   422,
			expectedSetCost:  2,
		},
		{
			name:             "index of array of array of numnbers",
			schemaGenerator:  genArrayOfArraysWithRule("number", "self[0][0] == -1.0"),
			expectedCalcCost: 3,
			setMaxElements:   51,
			expectedSetCost:  3,
		},
		{
			name:             "array of number maps with all",
			schemaGenerator:  getMapArrayWithRule("number", `self.all(x, x.y == 25.2)`),
			expectedCalcCost: 6291452,
			setMaxElements:   12,
			expectedSetCost:  74,
		},
		{
			name:             "index of array of number maps",
			schemaGenerator:  getMapArrayWithRule("number", `self[0].x > 2.0`),
			expectedCalcCost: 4,
			setMaxElements:   3000,
			expectedSetCost:  4,
		},
		{
			name:             "duration array with all",
			schemaGenerator:  genArrayWithRule("duration", "self.all(x, true)"),
			expectedCalcCost: 2359295,
			setMaxElements:   5,
			expectedSetCost:  17,
		},
		{
			name:             "index of duration array",
			schemaGenerator:  genArrayWithRule("duration", "self[0].getHours() == 2"),
			expectedCalcCost: 4,
			setMaxElements:   525,
			expectedSetCost:  4,
		},
		{
			name:             "date array with all",
			schemaGenerator:  genArrayWithRule("date", "self.all(x, true)"),
			expectedCalcCost: 725936,
			setMaxElements:   15,
			expectedSetCost:  47,
		},
		{
			name:             "index of date array",
			schemaGenerator:  genArrayWithRule("date", "self[2].getDayOfMonth() == 13"),
			expectedCalcCost: 4,
			setMaxElements:   42,
			expectedSetCost:  4,
		},
		{
			name:             "date-time array with all",
			schemaGenerator:  genArrayWithRule("date-time", "self.all(x, true)"),
			expectedCalcCost: 428963,
			setMaxElements:   25,
			expectedSetCost:  77,
		},
		{
			name:             "index of date-time array",
			schemaGenerator:  genArrayWithRule("date-time", "self[2].getMinutes() == 45"),
			expectedCalcCost: 4,
			setMaxElements:   99,
			expectedSetCost:  4,
		},
		{
			name:             "map of int-or-strings with all",
			schemaGenerator:  genMapWithRule("int-or-string", "self.all(x, true)"),
			expectedCalcCost: 1348169,
			setMaxElements:   15,
			expectedSetCost:  47,
		},
		{
			name:             "map of int-or-strings with has",
			schemaGenerator:  genMapWithRule("int-or-string", "has(self.x)"),
			expectedCalcCost: 0,
			setMaxElements:   5000,
			expectedSetCost:  0,
		},
		{
			name:             "map of objects with all",
			schemaGenerator:  genMapWithCustomItemRule(genObjectForMap(), "self.all(x, true)"),
			expectedCalcCost: 428963,
			setMaxElements:   20,
			expectedSetCost:  62,
		},
		{
			name:             "map of objects with has",
			schemaGenerator:  genMapWithCustomItemRule(genObjectForMap(), "has(self.x)"),
			expectedCalcCost: 0,
			setMaxElements:   9001,
			expectedSetCost:  0,
		},
		{
			name:             "map of number maps with all",
			schemaGenerator:  genMapWithCustomItemRule(genMapForMap(), "self.all(x, true)"),
			expectedCalcCost: 1179647,
			setMaxElements:   10,
			expectedSetCost:  32,
		},
		{
			name:             "map of number maps with has",
			schemaGenerator:  genMapWithCustomItemRule(genMapForMap(), "has(self.x)"),
			expectedCalcCost: 0,
			setMaxElements:   101,
			expectedSetCost:  0,
		},
		{
			name:             "map of number arrays with all",
			schemaGenerator:  genMapWithCustomItemRule(genArrayForMap(), "self.all(x, true)"),
			expectedCalcCost: 1179647,
			setMaxElements:   25,
			expectedSetCost:  77,
		},
		{
			name:             "map of number arrays with has",
			schemaGenerator:  genMapWithCustomItemRule(genArrayForMap(), "has(self.x)"),
			expectedCalcCost: 0,
			setMaxElements:   40000,
			expectedSetCost:  0,
		},
		{
			name:             "map of durations with all",
			schemaGenerator:  genMapWithRule("duration", "self.all(x, true)"),
			expectedCalcCost: 1048577,
			setMaxElements:   5,
			expectedSetCost:  17,
		},
		{
			name:             "map of durations with has",
			schemaGenerator:  genMapWithRule("duration", "has(self.x)"),
			expectedCalcCost: 0,
			setMaxElements:   256,
			expectedSetCost:  0,
		},
		{
			name:             "map of dates with all",
			schemaGenerator:  genMapWithRule("date", "self.all(x, true)"),
			expectedCalcCost: 524288,
			setMaxElements:   10,
			expectedSetCost:  32,
		},
		{
			name:             "map of dates with has",
			schemaGenerator:  genMapWithRule("date", "has(self.x)"),
			expectedCalcCost: 0,
			setMaxElements:   65536,
			expectedSetCost:  0,
		},
		{
			name:             "map of date-times with all",
			schemaGenerator:  genMapWithRule("date-time", "self.all(x, true)"),
			expectedCalcCost: 349526,
			setMaxElements:   25,
			expectedSetCost:  77,
		},
		{
			name:             "map of date-times with has",
			schemaGenerator:  genMapWithRule("date-time", "has(self.x)"),
			expectedCalcCost: 0,
			setMaxElements:   490,
			expectedSetCost:  0,
		},
		// Ensure library functions are integrated with size estimates by testing the interesting cases.
		{
			name:             "extended library regex find",
			schemaGenerator:  genStringWithRule("self.find('[0-9]+') == ''"),
			expectedCalcCost: 629147,
			setMaxElements:   10,
			expectedSetCost:  11,
		},
		{
			name: "extended library join",
			schemaGenerator: func(max *int64) *schema.Structural {
				strType := withMaxLength(primitiveType("string", ""), max)
				array := withMaxItems(arrayType("atomic", nil, &strType), max)
				array = withRule(array, "self.join(' ') == 'aa bb'")
				return &array
			},
			expectedCalcCost: 329853068905,
			setMaxElements:   10,
			expectedSetCost:  43,
		},
		{
			name: "extended library isSorted",
			schemaGenerator: func(max *int64) *schema.Structural {
				strType := withMaxLength(primitiveType("string", ""), max)
				array := withMaxItems(arrayType("atomic", nil, &strType), max)
				array = withRule(array, "self.isSorted() == true")
				return &array
			},
			expectedCalcCost: 329854432052,
			setMaxElements:   10,
			expectedSetCost:  52,
		},
		{
			name: "extended library replace",
			schemaGenerator: func(max *int64) *schema.Structural {
				strType := withMaxLength(primitiveType("string", ""), max)
				beforeLen := int64(2)
				afterLen := int64(4)
				objType := objectType(map[string]schema.Structural{
					"str":    strType,
					"before": withMaxLength(primitiveType("string", ""), &beforeLen),
					"after":  withMaxLength(primitiveType("string", ""), &afterLen),
				})
				objType = withRule(objType, "self.str.replace(self.before, self.after) == 'does not matter'")
				return &objType
			},
			expectedCalcCost: 629154, // cost is based on the result size of the replace() call
			setMaxElements:   4,
			expectedSetCost:  12,
		},
		{
			name: "extended library split",
			schemaGenerator: func(max *int64) *schema.Structural {
				strType := withMaxLength(primitiveType("string", ""), max)
				objType := objectType(map[string]schema.Structural{
					"str":       strType,
					"separator": strType,
				})
				objType = withRule(objType, "self.str.split(self.separator) == []")
				return &objType
			},
			expectedCalcCost: 629160,
			setMaxElements:   10,
			expectedSetCost:  22,
		},
		{
			name: "extended library lowerAscii",
			schemaGenerator: func(max *int64) *schema.Structural {
				strType := withMaxLength(primitiveType("string", ""), max)
				strType = withRule(strType, "self.lowerAscii() == 'lower!'")
				return &strType
			},
			expectedCalcCost: 314575,
			setMaxElements:   10,
			expectedSetCost:  6,
		},
		{
			name:             "check cost of size call",
			schemaGenerator:  genMapWithRule("integer", "oldSelf.size() == self.size()"),
			expectedCalcCost: 5,
			setMaxElements:   10,
			expectedSetCost:  5,
		},
		{
			name:             "check cost of timestamp comparison",
			schemaGenerator:  genMapWithRule("date-time", `self["a"] == self["b"]`),
			expectedCalcCost: 8,
			setMaxElements:   7,
			expectedSetCost:  8,
		},
		{
			name:             "check cost of duration comparison",
			schemaGenerator:  genMapWithRule("duration", `self["c"] == self["d"]`),
			expectedCalcCost: 8,
			setMaxElements:   42,
			expectedSetCost:  8,
		},
		{
			name:             "enums with maxLength equals to the longest possible value",
			schemaGenerator:  genEnumWithRuleAndValues("self.contains('A')", "A", "B", "C", "LongValue"),
			expectedCalcCost: 2,
			setMaxElements:   1000,
			expectedSetCost:  401,
		},
	}
	for _, testCase := range cases {
		t.Run(testCase.name, func(t *testing.T) {
			// dynamic maxLength case
			schema := testCase.schemaGenerator(nil)
			t.Run("calc maxLength", schemaChecker(schema, testCase.expectedCalcCost, testCase.expectCalcCostExceedsLimit, t))
			// static maxLength case
			setSchema := testCase.schemaGenerator(&testCase.setMaxElements)
			t.Run("set maxLength", schemaChecker(setSchema, testCase.expectedSetCost, testCase.expectedSetCostExceedsLimit, t))
		})
	}
}

func BenchmarkCompile(b *testing.B) {
	env := environment.MustBaseEnvSet(environment.DefaultCompatibilityVersion(), true) // prepare the environment
	s := genArrayWithRule("number", "true")(nil)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := Compile(s, model.SchemaDeclType(s, false), math.MaxInt64, env, NewExpressionsEnvLoader())
		if err != nil {
			b.Fatal(err)
		}
	}
}

type fakeLib struct{}

var testLibraryDecls = map[string][]celgo.FunctionOpt{
	"fakeFunction": {
		celgo.Overload("fakeFunction", []*celgo.Type{celgo.StringType}, celgo.StringType,
			celgo.UnaryBinding(fakeFunction))},
}

func (*fakeLib) CompileOptions() []celgo.EnvOption {
	options := make([]celgo.EnvOption, 0, len(testLibraryDecls))
	for name, overloads := range testLibraryDecls {
		options = append(options, celgo.Function(name, overloads...))
	}
	return options
}

func (*fakeLib) ProgramOptions() []celgo.ProgramOption {
	return []celgo.ProgramOption{}
}

func fakeFunction(arg1 ref.Val) ref.Val {
	arg, ok := arg1.Value().(string)
	if !ok {
		return types.MaybeNoSuchOverloadErr(arg1)
	}

	return types.String(strings.ToUpper(arg))
}
