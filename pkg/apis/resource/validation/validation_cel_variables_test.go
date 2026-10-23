/*
Copyright 2026 The Kubernetes Authors.

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

package validation

import (
	"strings"
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/validation/field"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/kubernetes/pkg/apis/resource"
	"k8s.io/kubernetes/pkg/features"
)

func TestValidateDeviceClassCELVariables(t *testing.T) {
	goodName := "foo"

	scenarios := map[string]struct {
		variablesEnabled bool
		class            *resource.DeviceClass
		wantFailures     field.ErrorList
	}{
		"variables-disabled": {
			variablesEnabled: false,
			wantFailures: field.ErrorList{
				field.Forbidden(field.NewPath("spec", "selectors").Index(0).Child("cel", "variables"), "feature DRAVariablesInCEL is disabled"),
			},
			class: func() *resource.DeviceClass {
				return &resource.DeviceClass{
					ObjectMeta: metav1.ObjectMeta{Name: goodName},
					Spec: resource.DeviceClassSpec{
						Selectors: []resource.DeviceSelector{{
							CEL: &resource.CELDeviceSelector{
								Expression: "variables.x",
								Variables: []resource.CELSelectorVariable{{
									Name:       "x",
									Expression: "true",
								}},
							},
						}},
					},
				}
			}(),
		},
		"valid-variables": {
			variablesEnabled: true,
			class: func() *resource.DeviceClass {
				return &resource.DeviceClass{
					ObjectMeta: metav1.ObjectMeta{Name: goodName},
					Spec: resource.DeviceClassSpec{
						Selectors: []resource.DeviceSelector{{
							CEL: &resource.CELDeviceSelector{
								Expression: "variables.gpu && variables.hasName",
								Variables: []resource.CELSelectorVariable{
									{Name: "gpu", Expression: `device.driver == "gpu.example.com"`},
									{Name: "hasName", Expression: `device.attributes["dra.example.com"].name != ""`},
								},
							},
						}},
					},
				}
			}(),
		},
		"duplicate-name": {
			variablesEnabled: true,
			wantFailures: field.ErrorList{
				field.Duplicate(field.NewPath("spec", "selectors").Index(0).Child("cel", "variables").Index(1).Child("name"), "x"),
			},
			class: func() *resource.DeviceClass {
				return &resource.DeviceClass{
					ObjectMeta: metav1.ObjectMeta{Name: goodName},
					Spec: resource.DeviceClassSpec{
						Selectors: []resource.DeviceSelector{{
							CEL: &resource.CELDeviceSelector{
								Expression: "variables.x",
								Variables: []resource.CELSelectorVariable{
									{Name: "x", Expression: "true"},
									{Name: "x", Expression: "false"},
								},
							},
						}},
					},
				}
			}(),
		},
		"invalid-identifier": {
			variablesEnabled: true,
			wantFailures: field.ErrorList{
				field.Invalid(field.NewPath("spec", "selectors").Index(0).Child("cel", "variables").Index(0).Child("name"), "1bad", "must be a valid CEL identifier"),
			},
			class: func() *resource.DeviceClass {
				return &resource.DeviceClass{
					ObjectMeta: metav1.ObjectMeta{Name: goodName},
					Spec: resource.DeviceClassSpec{
						Selectors: []resource.DeviceSelector{{
							CEL: &resource.CELDeviceSelector{
								Expression: "true",
								Variables: []resource.CELSelectorVariable{
									{Name: "1bad", Expression: "true"},
								},
							},
						}},
					},
				}
			}(),
		},
		"reserved-identifier": {
			variablesEnabled: true,
			wantFailures: field.ErrorList{
				field.Invalid(field.NewPath("spec", "selectors").Index(0).Child("cel", "variables").Index(0).Child("name"), "true", "must be a valid CEL identifier"),
			},
			class: func() *resource.DeviceClass {
				return &resource.DeviceClass{
					ObjectMeta: metav1.ObjectMeta{Name: goodName},
					Spec: resource.DeviceClassSpec{
						Selectors: []resource.DeviceSelector{{
							CEL: &resource.CELDeviceSelector{
								Expression: "true",
								Variables: []resource.CELSelectorVariable{
									{Name: "true", Expression: "1 == 1"},
								},
							},
						}},
					},
				}
			}(),
		},
		"empty-name": {
			variablesEnabled: true,
			wantFailures: field.ErrorList{
				field.Required(field.NewPath("spec", "selectors").Index(0).Child("cel", "variables").Index(0).Child("name"), ""),
			},
			class: func() *resource.DeviceClass {
				return &resource.DeviceClass{
					ObjectMeta: metav1.ObjectMeta{Name: goodName},
					Spec: resource.DeviceClassSpec{
						Selectors: []resource.DeviceSelector{{
							CEL: &resource.CELDeviceSelector{
								Expression: "true",
								Variables: []resource.CELSelectorVariable{
									{Name: "", Expression: "true"},
								},
							},
						}},
					},
				}
			}(),
		},
		"empty-expression": {
			variablesEnabled: true,
			wantFailures: field.ErrorList{
				field.Required(field.NewPath("spec", "selectors").Index(0).Child("cel", "variables").Index(0).Child("expression"), ""),
			},
			class: func() *resource.DeviceClass {
				return &resource.DeviceClass{
					ObjectMeta: metav1.ObjectMeta{Name: goodName},
					Spec: resource.DeviceClassSpec{
						Selectors: []resource.DeviceSelector{{
							CEL: &resource.CELDeviceSelector{
								Expression: "true",
								Variables: []resource.CELSelectorVariable{
									{Name: "x", Expression: ""},
								},
							},
						}},
					},
				}
			}(),
		},
		"too-many-variables": {
			variablesEnabled: true,
			wantFailures: field.ErrorList{
				field.TooMany(field.NewPath("spec", "selectors").Index(0).Child("cel", "variables"), resource.CELSelectorVariablesMaxSize+1, resource.CELSelectorVariablesMaxSize),
			},
			class: func() *resource.DeviceClass {
				vars := make([]resource.CELSelectorVariable, 0, resource.CELSelectorVariablesMaxSize+1)
				for i := 0; i < resource.CELSelectorVariablesMaxSize+1; i++ {
					vars = append(vars, resource.CELSelectorVariable{
						Name:       "v" + itoa(i),
						Expression: "true",
					})
				}
				return &resource.DeviceClass{
					ObjectMeta: metav1.ObjectMeta{Name: goodName},
					Spec: resource.DeviceClassSpec{
						Selectors: []resource.DeviceSelector{{
							CEL: &resource.CELDeviceSelector{
								Expression: "true",
								Variables:  vars,
							},
						}},
					},
				}
			}(),
		},
		"name-too-long": {
			variablesEnabled: true,
			wantFailures: field.ErrorList{
				field.TooLong(field.NewPath("spec", "selectors").Index(0).Child("cel", "variables").Index(0).Child("name"), "" /*unused*/, resource.CELSelectorVariableNameMaxLength),
			},
			class: func() *resource.DeviceClass {
				return &resource.DeviceClass{
					ObjectMeta: metav1.ObjectMeta{Name: goodName},
					Spec: resource.DeviceClassSpec{
						Selectors: []resource.DeviceSelector{{
							CEL: &resource.CELDeviceSelector{
								Expression: "true",
								Variables: []resource.CELSelectorVariable{{
									Name:       "a" + strings.Repeat("b", resource.CELSelectorVariableNameMaxLength),
									Expression: "true",
								}},
							},
						}},
					},
				}
			}(),
		},
		"forward-reference-rejected": {
			variablesEnabled: true,
			wantFailures: field.ErrorList{
				field.Invalid(field.NewPath("spec", "selectors").Index(0).Child("cel", "expression"), "variables.earlier", "variable \"earlier\": compilation failed: ERROR: <input>:1:10: undefined field 'later'\n | variables.later\n | .........^"),
			},
			class: func() *resource.DeviceClass {
				return &resource.DeviceClass{
					ObjectMeta: metav1.ObjectMeta{Name: goodName},
					Spec: resource.DeviceClassSpec{
						Selectors: []resource.DeviceSelector{{
							CEL: &resource.CELDeviceSelector{
								Expression: "variables.earlier",
								Variables: []resource.CELSelectorVariable{
									{Name: "earlier", Expression: "variables.later"},
									{Name: "later", Expression: "true"},
								},
							},
						}},
					},
				}
			}(),
		},
		"undeclared-variable-in-expression": {
			variablesEnabled: true,
			wantFailures: field.ErrorList{
				field.Invalid(field.NewPath("spec", "selectors").Index(0).Child("cel", "expression"), "variables.missing", "compilation failed: ERROR: <input>:1:10: undefined field 'missing'\n | variables.missing\n | .........^"),
			},
			class: func() *resource.DeviceClass {
				return &resource.DeviceClass{
					ObjectMeta: metav1.ObjectMeta{Name: goodName},
					Spec: resource.DeviceClassSpec{
						Selectors: []resource.DeviceSelector{{
							CEL: &resource.CELDeviceSelector{
								Expression: "variables.missing",
								Variables: []resource.CELSelectorVariable{
									{Name: "x", Expression: "true"},
								},
							},
						}},
					},
				}
			}(),
		},
	}

	for name, scenario := range scenarios {
		t.Run(name, func(t *testing.T) {
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.DRAVariablesInCEL, scenario.variablesEnabled)
			errs := ValidateDeviceClass(scenario.class)
			assertFailures(t, scenario.wantFailures, errs)
		})
	}
}

func itoa(i int) string {
	if i == 0 {
		return "0"
	}
	var b []byte
	for n := i; n > 0; n /= 10 {
		b = append([]byte{byte('0' + n%10)}, b...)
	}
	return string(b)
}
