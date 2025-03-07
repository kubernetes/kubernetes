/*
Copyright 2025 The Kubernetes Authors.

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
	resourceapi "k8s.io/kubernetes/pkg/apis/resource"
	"k8s.io/utils/ptr"
)

func testDeviceTaint(name string, spec resourceapi.DeviceTaintSpec) *resourceapi.DeviceTaint {
	return &resourceapi.DeviceTaint{
		ObjectMeta: metav1.ObjectMeta{
			Name: name,
		},
		Spec: *spec.DeepCopy(),
	}
}

var validPatchSpec = resourceapi.DeviceTaintSpec{
	Filter: &resourceapi.DeviceTaintFilter{
		DeviceClassName: ptr.To(goodName),
		Driver:          ptr.To("test.example.com"),
		Pool:            ptr.To(goodName),
		Device:          ptr.To(goodName),
	},
	Taint: resourceapi.DeviceTaintAtom{
		Key:    "example.com/taint",
		Value:  "tainted",
		Effect: resourceapi.DeviceTaintEffectNoSchedule,
	},
}

func TestValidateDeviceTaint(t *testing.T) {
	goodName := "foo"
	now := metav1.Now()
	badName := "!@#$%^"
	badValue := "spaces not allowed"

	scenarios := map[string]struct {
		patch        *resourceapi.DeviceTaint
		wantFailures field.ErrorList
	}{
		"good-patch": {
			patch: testDeviceTaint(goodName, validPatchSpec),
		},
		"missing-name": {
			wantFailures: field.ErrorList{field.Required(field.NewPath("metadata", "name"), "name or generateName is required")},
			patch:        testDeviceTaint("", validPatchSpec),
		},
		"bad-name": {
			wantFailures: field.ErrorList{field.Invalid(field.NewPath("metadata", "name"), badName, "a lowercase RFC 1123 subdomain must consist of lower case alphanumeric characters, '-' or '.', and must start and end with an alphanumeric character (e.g. 'example.com', regex used for validation is '[a-z0-9]([-a-z0-9]*[a-z0-9])?(\\.[a-z0-9]([-a-z0-9]*[a-z0-9])?)*')")},
			patch:        testDeviceTaint(badName, validPatchSpec),
		},
		"generate-name": {
			patch: func() *resourceapi.DeviceTaint {
				patch := testDeviceTaint(goodName, validPatchSpec)
				patch.GenerateName = "pvc-"
				return patch
			}(),
		},
		"uid": {
			patch: func() *resourceapi.DeviceTaint {
				patch := testDeviceTaint(goodName, validPatchSpec)
				patch.UID = "ac051fac-2ead-46d9-b8b4-4e0fbeb7455d"
				return patch
			}(),
		},
		"resource-version": {
			patch: func() *resourceapi.DeviceTaint {
				patch := testDeviceTaint(goodName, validPatchSpec)
				patch.ResourceVersion = "1"
				return patch
			}(),
		},
		"generation": {
			patch: func() *resourceapi.DeviceTaint {
				patch := testDeviceTaint(goodName, validPatchSpec)
				patch.Generation = 100
				return patch
			}(),
		},
		"creation-timestamp": {
			patch: func() *resourceapi.DeviceTaint {
				patch := testDeviceTaint(goodName, validPatchSpec)
				patch.CreationTimestamp = now
				return patch
			}(),
		},
		"deletion-grace-period-seconds": {
			patch: func() *resourceapi.DeviceTaint {
				patch := testDeviceTaint(goodName, validPatchSpec)
				patch.DeletionGracePeriodSeconds = ptr.To(int64(10))
				return patch
			}(),
		},
		"owner-references": {
			patch: func() *resourceapi.DeviceTaint {
				patch := testDeviceTaint(goodName, validPatchSpec)
				patch.OwnerReferences = []metav1.OwnerReference{
					{
						APIVersion: "v1",
						Kind:       "pod",
						Name:       "foo",
						UID:        "ac051fac-2ead-46d9-b8b4-4e0fbeb7455d",
					},
				}
				return patch
			}(),
		},
		"finalizers": {
			patch: func() *resourceapi.DeviceTaint {
				patch := testDeviceTaint(goodName, validPatchSpec)
				patch.Finalizers = []string{
					"example.com/foo",
				}
				return patch
			}(),
		},
		"managed-fields": {
			patch: func() *resourceapi.DeviceTaint {
				patch := testDeviceTaint(goodName, validPatchSpec)
				patch.ManagedFields = []metav1.ManagedFieldsEntry{
					{
						FieldsType: "FieldsV1",
						Operation:  "Apply",
						APIVersion: "apps/v1",
						Manager:    "foo",
					},
				}
				return patch
			}(),
		},
		"good-labels": {
			patch: func() *resourceapi.DeviceTaint {
				patch := testDeviceTaint(goodName, validPatchSpec)
				patch.Labels = map[string]string{
					"apps.kubernetes.io/name": "test",
				}
				return patch
			}(),
		},
		"bad-labels": {
			wantFailures: field.ErrorList{field.Invalid(field.NewPath("metadata", "labels"), badValue, "a valid label must be an empty string or consist of alphanumeric characters, '-', '_' or '.', and must start and end with an alphanumeric character (e.g. 'MyValue',  or 'my_value',  or '12345', regex used for validation is '(([A-Za-z0-9][-A-Za-z0-9_.]*)?[A-Za-z0-9])?')")},
			patch: func() *resourceapi.DeviceTaint {
				patch := testDeviceTaint(goodName, validPatchSpec)
				patch.Labels = map[string]string{
					"hello-world": badValue,
				}
				return patch
			}(),
		},
		"good-annotations": {
			patch: func() *resourceapi.DeviceTaint {
				patch := testDeviceTaint(goodName, validPatchSpec)
				patch.Annotations = map[string]string{
					"foo": "bar",
				}
				return patch
			}(),
		},
		"bad-annotations": {
			wantFailures: field.ErrorList{field.Invalid(field.NewPath("metadata", "annotations"), badName, "name part must consist of alphanumeric characters, '-', '_' or '.', and must start and end with an alphanumeric character (e.g. 'MyName',  or 'my.name',  or '123-abc', regex used for validation is '([A-Za-z0-9][-A-Za-z0-9_.]*)?[A-Za-z0-9]')")},
			patch: func() *resourceapi.DeviceTaint {
				patch := testDeviceTaint(goodName, validPatchSpec)
				patch.Annotations = map[string]string{
					badName: "hello world",
				}
				return patch
			}(),
		},
		"bad-class": {
			wantFailures: field.ErrorList{field.Invalid(field.NewPath("spec", "filter", "deviceClassName"), badName, "a lowercase RFC 1123 subdomain must consist of lower case alphanumeric characters, '-' or '.', and must start and end with an alphanumeric character (e.g. 'example.com', regex used for validation is '[a-z0-9]([-a-z0-9]*[a-z0-9])?(\\.[a-z0-9]([-a-z0-9]*[a-z0-9])?)*')")},
			patch: func() *resourceapi.DeviceTaint {
				patch := testDeviceTaint(goodName, validPatchSpec)
				patch.Spec.Filter.DeviceClassName = ptr.To(badName)
				return patch
			}(),
		},
		"bad-driver": {
			wantFailures: field.ErrorList{field.Invalid(field.NewPath("spec", "filter", "driver"), badName, "a lowercase RFC 1123 subdomain must consist of lower case alphanumeric characters, '-' or '.', and must start and end with an alphanumeric character (e.g. 'example.com', regex used for validation is '[a-z0-9]([-a-z0-9]*[a-z0-9])?(\\.[a-z0-9]([-a-z0-9]*[a-z0-9])?)*')")},
			patch: func() *resourceapi.DeviceTaint {
				patch := testDeviceTaint(goodName, validPatchSpec)
				patch.Spec.Filter.Driver = ptr.To(badName)
				return patch
			}(),
		},
		"bad-pool": {
			wantFailures: field.ErrorList{field.Invalid(field.NewPath("spec", "filter", "pool"), badName, "a lowercase RFC 1123 subdomain must consist of lower case alphanumeric characters, '-' or '.', and must start and end with an alphanumeric character (e.g. 'example.com', regex used for validation is '[a-z0-9]([-a-z0-9]*[a-z0-9])?(\\.[a-z0-9]([-a-z0-9]*[a-z0-9])?)*')")},
			patch: func() *resourceapi.DeviceTaint {
				patch := testDeviceTaint(goodName, validPatchSpec)
				patch.Spec.Filter.Pool = ptr.To(badName)
				return patch
			}(),
		},
		"bad-device": {
			wantFailures: field.ErrorList{field.Invalid(field.NewPath("spec", "filter", "device"), badName, "a lowercase RFC 1123 label must consist of lower case alphanumeric characters or '-', and must start and end with an alphanumeric character (e.g. 'my-name',  or '123-abc', regex used for validation is '[a-z0-9]([-a-z0-9]*[a-z0-9])?')")},
			patch: func() *resourceapi.DeviceTaint {
				patch := testDeviceTaint(goodName, validPatchSpec)
				patch.Spec.Filter.Device = ptr.To(badName)
				return patch
			}(),
		},
		"CEL-compile-errors": {
			wantFailures: field.ErrorList{
				field.Invalid(field.NewPath("spec", "filter", "selectors").Index(1).Child("cel", "expression"), `device.attributes[true].someBoolean`, "compilation failed: ERROR: <input>:1:18: found no matching overload for '_[_]' applied to '(map(string, map(string, any)), bool)'\n | device.attributes[true].someBoolean\n | .................^"),
			},
			patch: func() *resourceapi.DeviceTaint {
				patch := testDeviceTaint(goodName, validPatchSpec)
				patch.Spec.Filter.Selectors = []resourceapi.DeviceSelector{
					{
						// Good selector.
						CEL: &resourceapi.CELDeviceSelector{
							Expression: `device.driver == "dra.example.com"`,
						},
					},
					{
						// Bad selector.
						CEL: &resourceapi.CELDeviceSelector{
							Expression: `device.attributes[true].someBoolean`,
						},
					},
				}
				return patch
			}(),
		},
		"CEL-length": {
			wantFailures: field.ErrorList{
				field.TooLong(field.NewPath("spec", "filter", "selectors").Index(1).Child("cel", "expression"), "" /*unused*/, resourceapi.CELSelectorExpressionMaxLength),
			},
			patch: func() *resourceapi.DeviceTaint {
				patch := testDeviceTaint(goodName, validPatchSpec)
				expression := `device.driver == ""`
				patch.Spec.Filter.Selectors = []resourceapi.DeviceSelector{
					{
						// Good selector.
						CEL: &resourceapi.CELDeviceSelector{
							Expression: strings.ReplaceAll(expression, `""`, `"`+strings.Repeat("x", resourceapi.CELSelectorExpressionMaxLength-len(expression))+`"`),
						},
					},
					{
						// Too long by one selector.
						CEL: &resourceapi.CELDeviceSelector{
							Expression: strings.ReplaceAll(expression, `""`, `"`+strings.Repeat("x", resourceapi.CELSelectorExpressionMaxLength-len(expression)+1)+`"`),
						},
					},
				}
				return patch
			}(),
		},
		"CEL-cost": {
			wantFailures: field.ErrorList{
				field.Forbidden(field.NewPath("spec", "filter", "selectors").Index(0).Child("cel", "expression"), "too complex, exceeds cost limit"),
			},
			patch: func() *resourceapi.DeviceTaint {
				claim := testDeviceTaint(goodName, validPatchSpec)
				claim.Spec.Filter.Selectors = []resourceapi.DeviceSelector{
					{
						CEL: &resourceapi.CELDeviceSelector{
							// From https://github.com/kubernetes/kubernetes/blob/50fc400f178d2078d0ca46aee955ee26375fc437/test/integration/apiserver/cel/validatingadmissionpolicy_test.go#L2150.
							Expression: `[1, 2, 3, 4, 5, 6, 7, 8, 9, 10].all(x, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10].all(y, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10].all(z, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10].all(z2, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10].all(z3, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10].all(z4, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10].all(z5, int('1'.find('[0-9]*')) < 100)))))))`,
						},
					},
				}
				return claim
			}(),
		},
		// TODO: taint validation
	}

	for name, scenario := range scenarios {
		t.Run(name, func(t *testing.T) {
			errs := ValidateDeviceTaint(scenario.patch)
			assertFailures(t, scenario.wantFailures, errs)
		})
	}
}

func TestValidateDeviceTaintUpdate(t *testing.T) {
	name := "valid"
	validPatch := testDeviceTaint(name, validPatchSpec)

	scenarios := map[string]struct {
		oldPatch     *resourceapi.DeviceTaint
		update       func(patch *resourceapi.DeviceTaint) *resourceapi.DeviceTaint
		wantFailures field.ErrorList
	}{
		"valid-no-op-update": {
			oldPatch: validPatch,
			update:   func(patch *resourceapi.DeviceTaint) *resourceapi.DeviceTaint { return patch },
		},
		"invalid-name-update": {
			wantFailures: field.ErrorList{field.Invalid(field.NewPath("metadata", "name"), name+"-update", "field is immutable")},
			oldPatch:     validPatch,
			update: func(patch *resourceapi.DeviceTaint) *resourceapi.DeviceTaint {
				patch.Name += "-update"
				return patch
			},
		},
	}

	for name, scenario := range scenarios {
		t.Run(name, func(t *testing.T) {
			scenario.oldPatch.ResourceVersion = "1"
			errs := ValidateDeviceTaintUpdate(scenario.update(scenario.oldPatch.DeepCopy()), scenario.oldPatch)
			assertFailures(t, scenario.wantFailures, errs)
		})
	}
}
