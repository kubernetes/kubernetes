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
	"strings"
	"testing"

	"k8s.io/apiextensions-apiserver/pkg/apis/apiextensions"
	"k8s.io/apiextensions-apiserver/pkg/apiserver/schema"
)

type validationMatch struct {
	errorType ErrorType
	contains  string
}

func invalidError(contains string) validationMatch {
	return validationMatch{errorType: ErrorTypeInvalid, contains: contains}
}

func (v validationMatch) matches(err *Error) bool {
	return err.Type == v.errorType && strings.Contains(err.Error(), v.contains)
}

func TestCelCompilation(t *testing.T) {
	cases := []struct {
		name           string
		input          schema.Structural
		expectedErrors []validationMatch
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
			expectedErrors: []validationMatch{
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
			expectedErrors: []validationMatch{
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
			expectedErrors: []validationMatch{
				invalidError("undefined field 'namespace'"),
				invalidError("undefined field 'if'"),
				invalidError("found no matching overload"),
			},
		},
	}

	for _, tt := range cases {
		t.Run(tt.name, func(t *testing.T) {
			compilationResults, err := Compile(&tt.input, false)
			if err != nil {
				t.Errorf("Expected no error, but got: %v", err)
			}

			seenErrs := make([]bool, len(compilationResults))

			for _, expectedError := range tt.expectedErrors {
				found := false
				for i, result := range compilationResults {
					if expectedError.matches(result.Error) && !seenErrs[i] {
						found = true
						seenErrs[i] = true
						break
					}
				}

				if !found {
					t.Errorf("expected error: %v", expectedError)
				}
			}

			for i, seen := range seenErrs {
				if !seen && compilationResults[i].Error != nil {
					t.Errorf("unexpected error: %v", compilationResults[i].Error)
				}
			}
		})
	}
}
