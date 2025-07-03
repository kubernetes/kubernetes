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

package validation

import (
	"strings"
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/kubernetes/pkg/apis/resource"
	"k8s.io/utils/ptr"
)

func testClass(name string) *resource.DeviceClass {
	return &resource.DeviceClass{
		ObjectMeta: metav1.ObjectMeta{
			Name: name,
		},
	}
}

func TestValidateClass(t *testing.T) {
	goodName := "foo"
	now := metav1.Now()
	badName := "!@#$%^"
	badValue := "spaces not allowed"

	scenarios := map[string]struct {
		class        *resource.DeviceClass
		wantFailures field.ErrorList
	}{
		"good-class": {
			class: testClass(goodName),
		},
		"missing-name": {
			wantFailures: field.ErrorList{field.Required(field.NewPath("metadata", "name"), "name or generateName is required")},
			class:        testClass(""),
		},
		"bad-name": {
			wantFailures: field.ErrorList{field.Invalid(field.NewPath("metadata", "name"), badName, "a lowercase RFC 1123 subdomain must consist of lower case alphanumeric characters, '-' or '.', and must start and end with an alphanumeric character (e.g. 'example.com', regex used for validation is '[a-z0-9]([-a-z0-9]*[a-z0-9])?(\\.[a-z0-9]([-a-z0-9]*[a-z0-9])?)*')")},
			class:        testClass(badName),
		},
		"generate-name": {
			class: func() *resource.DeviceClass {
				class := testClass(goodName)
				class.GenerateName = "pvc-"
				return class
			}(),
		},
		"uid": {
			class: func() *resource.DeviceClass {
				class := testClass(goodName)
				class.UID = "ac051fac-2ead-46d9-b8b4-4e0fbeb7455d"
				return class
			}(),
		},
		"resource-version": {
			class: func() *resource.DeviceClass {
				class := testClass(goodName)
				class.ResourceVersion = "1"
				return class
			}(),
		},
		"generation": {
			class: func() *resource.DeviceClass {
				class := testClass(goodName)
				class.Generation = 100
				return class
			}(),
		},
		"creation-timestamp": {
			class: func() *resource.DeviceClass {
				class := testClass(goodName)
				class.CreationTimestamp = now
				return class
			}(),
		},
		"deletion-grace-period-seconds": {
			class: func() *resource.DeviceClass {
				class := testClass(goodName)
				class.DeletionGracePeriodSeconds = ptr.To(int64(10))
				return class
			}(),
		},
		"owner-references": {
			class: func() *resource.DeviceClass {
				class := testClass(goodName)
				class.OwnerReferences = []metav1.OwnerReference{
					{
						APIVersion: "v1",
						Kind:       "pod",
						Name:       "foo",
						UID:        "ac051fac-2ead-46d9-b8b4-4e0fbeb7455d",
					},
				}
				return class
			}(),
		},
		"finalizers": {
			class: func() *resource.DeviceClass {
				class := testClass(goodName)
				class.Finalizers = []string{
					"example.com/foo",
				}
				return class
			}(),
		},
		"managed-fields": {
			class: func() *resource.DeviceClass {
				class := testClass(goodName)
				class.ManagedFields = []metav1.ManagedFieldsEntry{
					{
						FieldsType: "FieldsV1",
						Operation:  "Apply",
						APIVersion: "apps/v1",
						Manager:    "foo",
					},
				}
				return class
			}(),
		},
		"good-labels": {
			class: func() *resource.DeviceClass {
				class := testClass(goodName)
				class.Labels = map[string]string{
					"apps.kubernetes.io/name": "test",
				}
				return class
			}(),
		},
		"bad-labels": {
			wantFailures: field.ErrorList{field.Invalid(field.NewPath("metadata", "labels"), badValue, "a valid label must be an empty string or consist of alphanumeric characters, '-', '_' or '.', and must start and end with an alphanumeric character (e.g. 'MyValue',  or 'my_value',  or '12345', regex used for validation is '(([A-Za-z0-9][-A-Za-z0-9_.]*)?[A-Za-z0-9])?')")},
			class: func() *resource.DeviceClass {
				class := testClass(goodName)
				class.Labels = map[string]string{
					"hello-world": badValue,
				}
				return class
			}(),
		},
		"good-annotations": {
			class: func() *resource.DeviceClass {
				class := testClass(goodName)
				class.Annotations = map[string]string{
					"foo": "bar",
				}
				return class
			}(),
		},
		"bad-annotations": {
			wantFailures: field.ErrorList{field.Invalid(field.NewPath("metadata", "annotations"), badName, "name part must consist of alphanumeric characters, '-', '_' or '.', and must start and end with an alphanumeric character (e.g. 'MyName',  or 'my.name',  or '123-abc', regex used for validation is '([A-Za-z0-9][-A-Za-z0-9_.]*)?[A-Za-z0-9]')")},
			class: func() *resource.DeviceClass {
				class := testClass(goodName)
				class.Annotations = map[string]string{
					badName: "hello world",
				}
				return class
			}(),
		},
		"selectors": {
			wantFailures: field.ErrorList{
				field.Required(field.NewPath("spec", "selectors").Index(1).Child("cel"), ""),
				field.Invalid(field.NewPath("spec", "selectors").Index(2).Child("cel", "expression"), "noSuchVar", "compilation failed: ERROR: <input>:1:1: undeclared reference to 'noSuchVar' (in container '')\n | noSuchVar\n | ^"),
			},
			class: func() *resource.DeviceClass {
				class := testClass(goodName)
				validSelector := resource.DeviceSelector{
					CEL: &resource.CELDeviceSelector{
						Expression: "true",
					},
				}
				class.Spec.Selectors = []resource.DeviceSelector{
					validSelector,
					{
						/* Missing CEL. */
					},
					{
						CEL: &resource.CELDeviceSelector{
							Expression: "noSuchVar",
						},
					},
				}
				for i := len(class.Spec.Selectors); i < resource.DeviceSelectorsMaxSize; i++ {
					class.Spec.Selectors = append(class.Spec.Selectors, validSelector)
				}
				return class
			}(),
		},
		"too-many-selectors": {
			wantFailures: field.ErrorList{
				field.TooMany(field.NewPath("spec", "selectors"), resource.DeviceSelectorsMaxSize+1, resource.DeviceSelectorsMaxSize),
			},
			class: func() *resource.DeviceClass {
				class := testClass(goodName)
				validSelector := resource.DeviceSelector{
					CEL: &resource.CELDeviceSelector{
						Expression: "true",
					},
				}
				for i := 0; i < resource.DeviceSelectorsMaxSize+1; i++ {
					class.Spec.Selectors = append(class.Spec.Selectors, validSelector)
				}
				return class
			}(),
		},
		"configuration": {
			wantFailures: field.ErrorList{
				field.Required(field.NewPath("spec", "config").Index(1).Child("opaque", "driver"), ""),
				field.Invalid(field.NewPath("spec", "config").Index(1).Child("opaque", "driver"), "", "a lowercase RFC 1123 subdomain must consist of lower case alphanumeric characters, '-' or '.', and must start and end with an alphanumeric character (e.g. 'example.com', regex used for validation is '[a-z0-9]([-a-z0-9]*[a-z0-9])?(\\.[a-z0-9]([-a-z0-9]*[a-z0-9])?)*')"),
				field.Required(field.NewPath("spec", "config").Index(1).Child("opaque", "parameters"), ""),
				field.Invalid(field.NewPath("spec", "config").Index(2).Child("opaque", "parameters"), "<value omitted>", "error parsing data as JSON: invalid character 'x' looking for beginning of value"),
				field.Invalid(field.NewPath("spec", "config").Index(3).Child("opaque", "parameters"), "<value omitted>", "must be a valid JSON object"),
				field.Required(field.NewPath("spec", "config").Index(4).Child("opaque", "parameters"), ""),
				field.Required(field.NewPath("spec", "config").Index(5).Child("opaque"), ""),
				field.TooLong(field.NewPath("spec", "config").Index(7).Child("opaque", "parameters"), "" /* unused */, resource.OpaqueParametersMaxLength),
			},
			class: func() *resource.DeviceClass {
				class := testClass(goodName)
				validConfig := resource.DeviceClassConfiguration{
					DeviceConfiguration: resource.DeviceConfiguration{
						Opaque: &resource.OpaqueDeviceConfiguration{
							Driver:     goodName,
							Parameters: runtime.RawExtension{Raw: []byte(`{"foo":42}`)},
						},
					},
				}
				class.Spec.Config = []resource.DeviceClassConfiguration{
					validConfig,
					{
						DeviceConfiguration: resource.DeviceConfiguration{
							Opaque: &resource.OpaqueDeviceConfiguration{ /* Bad, both fields are required! */ },
						},
					},
					{
						DeviceConfiguration: resource.DeviceConfiguration{
							Opaque: &resource.OpaqueDeviceConfiguration{
								Driver:     goodName,
								Parameters: runtime.RawExtension{Raw: []byte(`xxx`)}, /* Bad, not JSON. */
							},
						},
					},
					{
						DeviceConfiguration: resource.DeviceConfiguration{
							Opaque: &resource.OpaqueDeviceConfiguration{
								Driver:     goodName,
								Parameters: runtime.RawExtension{Raw: []byte(`"hello-world"`)}, /* Bad, not object. */
							},
						},
					},
					{
						DeviceConfiguration: resource.DeviceConfiguration{
							Opaque: &resource.OpaqueDeviceConfiguration{
								Driver:     goodName,
								Parameters: runtime.RawExtension{Raw: []byte(`null`)}, /* Bad, nil object. */
							},
						},
					},
					{
						DeviceConfiguration: resource.DeviceConfiguration{ /* Bad, empty. */ },
					},
					{
						DeviceConfiguration: resource.DeviceConfiguration{
							Opaque: &resource.OpaqueDeviceConfiguration{
								Driver:     goodName,
								Parameters: runtime.RawExtension{Raw: []byte(`{"str": "` + strings.Repeat("x", resource.OpaqueParametersMaxLength-9-2) + `"}`)},
							},
						},
					},
					{
						DeviceConfiguration: resource.DeviceConfiguration{
							Opaque: &resource.OpaqueDeviceConfiguration{
								Driver:     goodName,
								Parameters: runtime.RawExtension{Raw: []byte(`{"str": "` + strings.Repeat("x", resource.OpaqueParametersMaxLength-9-2+1 /* too large by one */) + `"}`)},
							},
						},
					},
				}
				for i := len(class.Spec.Config); i < resource.DeviceConfigMaxSize; i++ {
					class.Spec.Config = append(class.Spec.Config, validConfig)
				}
				return class
			}(),
		},
		"too-many-configs": {
			wantFailures: field.ErrorList{
				field.TooMany(field.NewPath("spec", "config"), resource.DeviceConfigMaxSize+1, resource.DeviceConfigMaxSize),
			},
			class: func() *resource.DeviceClass {
				class := testClass(goodName)
				validConfig := resource.DeviceClassConfiguration{
					DeviceConfiguration: resource.DeviceConfiguration{
						Opaque: &resource.OpaqueDeviceConfiguration{
							Driver:     goodName,
							Parameters: runtime.RawExtension{Raw: []byte(`{"foo":42}`)},
						},
					},
				}
				for i := len(class.Spec.Config); i < resource.DeviceConfigMaxSize+1; i++ {
					class.Spec.Config = append(class.Spec.Config, validConfig)
				}
				return class
			}(),
		},
	}

	for name, scenario := range scenarios {
		t.Run(name, func(t *testing.T) {
			errs := ValidateDeviceClass(scenario.class)
			assertFailures(t, scenario.wantFailures, errs)
		})
	}
}

func TestValidateClassUpdate(t *testing.T) {
	validClass := testClass(goodName)

	scenarios := map[string]struct {
		oldClass     *resource.DeviceClass
		update       func(class *resource.DeviceClass) *resource.DeviceClass
		wantFailures field.ErrorList
	}{
		"valid-no-op-update": {
			oldClass: validClass,
			update:   func(class *resource.DeviceClass) *resource.DeviceClass { return class },
		},
		"valid-config-large": {
			oldClass: validClass,
			update: func(class *resource.DeviceClass) *resource.DeviceClass {
				class.Spec.Config = []resource.DeviceClassConfiguration{{
					DeviceConfiguration: resource.DeviceConfiguration{
						Opaque: &resource.OpaqueDeviceConfiguration{
							Driver:     goodName,
							Parameters: runtime.RawExtension{Raw: []byte(`{"str": "` + strings.Repeat("x", resource.OpaqueParametersMaxLength-9-2) + `"}`)},
						},
					},
				}}
				return class
			},
		},
		"invalid-config-too-large": {
			wantFailures: field.ErrorList{
				field.TooLong(field.NewPath("spec", "config").Index(0).Child("opaque", "parameters"), "" /* unused */, resource.OpaqueParametersMaxLength),
			},
			oldClass: validClass,
			update: func(class *resource.DeviceClass) *resource.DeviceClass {
				class.Spec.Config = []resource.DeviceClassConfiguration{{
					DeviceConfiguration: resource.DeviceConfiguration{
						Opaque: &resource.OpaqueDeviceConfiguration{
							Driver:     goodName,
							Parameters: runtime.RawExtension{Raw: []byte(`{"str": "` + strings.Repeat("x", resource.OpaqueParametersMaxLength-9-2+1 /* too large by one */) + `"}`)},
						},
					},
				}}
				return class
			},
		},
		"too-large-config-valid-if-stored": {
			oldClass: func() *resource.DeviceClass {
				class := validClass.DeepCopy()
				class.Spec.Config = []resource.DeviceClassConfiguration{{
					DeviceConfiguration: resource.DeviceConfiguration{
						Opaque: &resource.OpaqueDeviceConfiguration{
							Driver:     goodName,
							Parameters: runtime.RawExtension{Raw: []byte(`{"str": "` + strings.Repeat("x", resource.OpaqueParametersMaxLength-9-2+1 /* too large by one */) + `"}`)},
						},
					},
				}}
				return class
			}(),
			update: func(class *resource.DeviceClass) *resource.DeviceClass {
				// No changes -> remains valid.
				return class
			},
		},
	}

	for name, scenario := range scenarios {
		t.Run(name, func(t *testing.T) {
			scenario.oldClass.ResourceVersion = "1"
			errs := ValidateDeviceClassUpdate(scenario.update(scenario.oldClass.DeepCopy()), scenario.oldClass)
			assertFailures(t, scenario.wantFailures, errs)
		})
	}
}
