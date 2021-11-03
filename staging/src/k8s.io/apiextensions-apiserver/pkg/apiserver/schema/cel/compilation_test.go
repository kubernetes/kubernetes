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
	"k8s.io/apiextensions-apiserver/pkg/apis/apiextensions"
	"k8s.io/apiextensions-apiserver/pkg/apiserver/schema"
	"strings"
	"testing"
)

func TestCelCompilation(t *testing.T) {
	cases := []struct {
		name               string
		input              schema.Structural
		wantError          bool
		checkErrorMessage  bool
		expectedErrMessage string
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
							Rule:    "minReplicas < maxReplicas",
							Message: "minReplicas should be smaller than maxReplicas",
						},
					},
				},
			},
			wantError:         false,
			checkErrorMessage: false,
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
			wantError:         false,
			checkErrorMessage: false,
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
			wantError:         false,
			checkErrorMessage: false,
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
			wantError:         false,
			checkErrorMessage: false,
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
			wantError:         false,
			checkErrorMessage: false,
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
			wantError:         false,
			checkErrorMessage: false,
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
							Rule:    "nestedObj.val == 10",
							Message: "val should be equal to 10",
						},
					},
				},
			},
			wantError:         false,
			checkErrorMessage: false,
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
			wantError:         false,
			checkErrorMessage: false,
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
			wantError:         false,
			checkErrorMessage: false,
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
			wantError:         false,
			checkErrorMessage: false,
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
			wantError:         false,
			checkErrorMessage: false,
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
			wantError:          true,
			checkErrorMessage:  true,
			expectedErrMessage: "compilation failed",
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
			wantError:          true,
			checkErrorMessage:  true,
			expectedErrMessage: "compilation failed",
		},
		{
			name: "valid is not specified",
			input: schema.Structural{
				Generic: schema.Generic{
					Type: "integer",
				},
				Extensions: schema.Extensions{
					XValidations: apiextensions.ValidationRules{
						{
							Message: "size of scoped field should be equal to 10",
						},
					},
				},
			},
			wantError:          true,
			checkErrorMessage:  true,
			expectedErrMessage: "rule is not specified",
		},
		{
			name: "valid is not specified",
			input: schema.Structural{
				Generic: schema.Generic{
					Type: "integer",
				},
				Extensions: schema.Extensions{
					XValidations: apiextensions.ValidationRules{
						{
							Message: "size of scoped field should be equal to 10",
						},
					},
				},
			},
			wantError:          true,
			checkErrorMessage:  true,
			expectedErrMessage: "rule is not specified",
		},
	}

	for _, tt := range cases {
		t.Run(tt.name, func(t *testing.T) {
			compilationResults, err := Compile(&tt.input)
			if err != nil {
				t.Errorf("Expected no error, but got: %v", err)
			}
			var allErrors []Error
			var pass = false
			for _, compilationResult := range compilationResults {
				if compilationResult.Error.Type != "" {
					allErrors = append(allErrors, compilationResult.Error)
					if strings.Contains(compilationResult.Error.Detail, tt.expectedErrMessage) {
						pass = true
					}
				}
			}
			if tt.checkErrorMessage && !pass {
				t.Errorf("Expected error massage contains: %v, but got error: %v", tt.expectedErrMessage, allErrors)
			}
			if !tt.wantError && len(allErrors) > 0 {
				t.Errorf("Expected no error, but got: %v", allErrors)
			} else if tt.wantError && len(allErrors) == 0 {
				t.Error("Expected error, but got none")
			}
		})
	}
}
