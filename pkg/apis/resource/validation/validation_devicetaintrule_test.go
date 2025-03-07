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

func testDeviceTaintRule(name string, spec resourceapi.DeviceTaintRuleSpec) *resourceapi.DeviceTaintRule {
	return &resourceapi.DeviceTaintRule{
		ObjectMeta: metav1.ObjectMeta{
			Name: name,
		},
		Spec: *spec.DeepCopy(),
	}
}

var validDeviceTaintRuleSpec = resourceapi.DeviceTaintRuleSpec{
	DeviceSelector: &resourceapi.DeviceTaintSelector{
		DeviceClassName: ptr.To(goodName),
		Driver:          ptr.To("test.example.com"),
		Pool:            ptr.To(goodName),
		Device:          ptr.To(goodName),
	},
	Taint: resourceapi.DeviceTaint{
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
		taintRule    *resourceapi.DeviceTaintRule
		wantFailures field.ErrorList
	}{
		"good": {
			taintRule: testDeviceTaintRule(goodName, validDeviceTaintRuleSpec),
		},
		"missing-name": {
			wantFailures: field.ErrorList{field.Required(field.NewPath("metadata", "name"), "name or generateName is required")},
			taintRule:    testDeviceTaintRule("", validDeviceTaintRuleSpec),
		},
		"bad-name": {
			wantFailures: field.ErrorList{field.Invalid(field.NewPath("metadata", "name"), badName, "a lowercase RFC 1123 subdomain must consist of lower case alphanumeric characters, '-' or '.', and must start and end with an alphanumeric character (e.g. 'example.com', regex used for validation is '[a-z0-9]([-a-z0-9]*[a-z0-9])?(\\.[a-z0-9]([-a-z0-9]*[a-z0-9])?)*')")},
			taintRule:    testDeviceTaintRule(badName, validDeviceTaintRuleSpec),
		},
		"generate-name": {
			taintRule: func() *resourceapi.DeviceTaintRule {
				taintRule := testDeviceTaintRule(goodName, validDeviceTaintRuleSpec)
				taintRule.GenerateName = "pvc-"
				return taintRule
			}(),
		},
		"uid": {
			taintRule: func() *resourceapi.DeviceTaintRule {
				taintRule := testDeviceTaintRule(goodName, validDeviceTaintRuleSpec)
				taintRule.UID = "ac051fac-2ead-46d9-b8b4-4e0fbeb7455d"
				return taintRule
			}(),
		},
		"resource-version": {
			taintRule: func() *resourceapi.DeviceTaintRule {
				taintRule := testDeviceTaintRule(goodName, validDeviceTaintRuleSpec)
				taintRule.ResourceVersion = "1"
				return taintRule
			}(),
		},
		"generation": {
			taintRule: func() *resourceapi.DeviceTaintRule {
				taintRule := testDeviceTaintRule(goodName, validDeviceTaintRuleSpec)
				taintRule.Generation = 100
				return taintRule
			}(),
		},
		"creation-timestamp": {
			taintRule: func() *resourceapi.DeviceTaintRule {
				taintRule := testDeviceTaintRule(goodName, validDeviceTaintRuleSpec)
				taintRule.CreationTimestamp = now
				return taintRule
			}(),
		},
		"deletion-grace-period-seconds": {
			taintRule: func() *resourceapi.DeviceTaintRule {
				taintRule := testDeviceTaintRule(goodName, validDeviceTaintRuleSpec)
				taintRule.DeletionGracePeriodSeconds = ptr.To(int64(10))
				return taintRule
			}(),
		},
		"owner-references": {
			taintRule: func() *resourceapi.DeviceTaintRule {
				taintRule := testDeviceTaintRule(goodName, validDeviceTaintRuleSpec)
				taintRule.OwnerReferences = []metav1.OwnerReference{
					{
						APIVersion: "v1",
						Kind:       "pod",
						Name:       "foo",
						UID:        "ac051fac-2ead-46d9-b8b4-4e0fbeb7455d",
					},
				}
				return taintRule
			}(),
		},
		"finalizers": {
			taintRule: func() *resourceapi.DeviceTaintRule {
				taintRule := testDeviceTaintRule(goodName, validDeviceTaintRuleSpec)
				taintRule.Finalizers = []string{
					"example.com/foo",
				}
				return taintRule
			}(),
		},
		"managed-fields": {
			taintRule: func() *resourceapi.DeviceTaintRule {
				taintRule := testDeviceTaintRule(goodName, validDeviceTaintRuleSpec)
				taintRule.ManagedFields = []metav1.ManagedFieldsEntry{
					{
						FieldsType: "FieldsV1",
						Operation:  "Apply",
						APIVersion: "apps/v1",
						Manager:    "foo",
					},
				}
				return taintRule
			}(),
		},
		"good-labels": {
			taintRule: func() *resourceapi.DeviceTaintRule {
				taintRule := testDeviceTaintRule(goodName, validDeviceTaintRuleSpec)
				taintRule.Labels = map[string]string{
					"apps.kubernetes.io/name": "test",
				}
				return taintRule
			}(),
		},
		"bad-labels": {
			wantFailures: field.ErrorList{field.Invalid(field.NewPath("metadata", "labels"), badValue, "a valid label must be an empty string or consist of alphanumeric characters, '-', '_' or '.', and must start and end with an alphanumeric character (e.g. 'MyValue',  or 'my_value',  or '12345', regex used for validation is '(([A-Za-z0-9][-A-Za-z0-9_.]*)?[A-Za-z0-9])?')")},
			taintRule: func() *resourceapi.DeviceTaintRule {
				taintRule := testDeviceTaintRule(goodName, validDeviceTaintRuleSpec)
				taintRule.Labels = map[string]string{
					"hello-world": badValue,
				}
				return taintRule
			}(),
		},
		"good-annotations": {
			taintRule: func() *resourceapi.DeviceTaintRule {
				taintRule := testDeviceTaintRule(goodName, validDeviceTaintRuleSpec)
				taintRule.Annotations = map[string]string{
					"foo": "bar",
				}
				return taintRule
			}(),
		},
		"bad-annotations": {
			wantFailures: field.ErrorList{field.Invalid(field.NewPath("metadata", "annotations"), badName, "name part must consist of alphanumeric characters, '-', '_' or '.', and must start and end with an alphanumeric character (e.g. 'MyName',  or 'my.name',  or '123-abc', regex used for validation is '([A-Za-z0-9][-A-Za-z0-9_.]*)?[A-Za-z0-9]')")},
			taintRule: func() *resourceapi.DeviceTaintRule {
				taintRule := testDeviceTaintRule(goodName, validDeviceTaintRuleSpec)
				taintRule.Annotations = map[string]string{
					badName: "hello world",
				}
				return taintRule
			}(),
		},
		"bad-class": {
			wantFailures: field.ErrorList{field.Invalid(field.NewPath("spec", "deviceSelector", "deviceClassName"), badName, "a lowercase RFC 1123 subdomain must consist of lower case alphanumeric characters, '-' or '.', and must start and end with an alphanumeric character (e.g. 'example.com', regex used for validation is '[a-z0-9]([-a-z0-9]*[a-z0-9])?(\\.[a-z0-9]([-a-z0-9]*[a-z0-9])?)*')")},
			taintRule: func() *resourceapi.DeviceTaintRule {
				taintRule := testDeviceTaintRule(goodName, validDeviceTaintRuleSpec)
				taintRule.Spec.DeviceSelector.DeviceClassName = ptr.To(badName)
				return taintRule
			}(),
		},
		"bad-driver": {
			wantFailures: field.ErrorList{field.Invalid(field.NewPath("spec", "deviceSelector", "driver"), badName, "a lowercase RFC 1123 subdomain must consist of lower case alphanumeric characters, '-' or '.', and must start and end with an alphanumeric character (e.g. 'example.com', regex used for validation is '[a-z0-9]([-a-z0-9]*[a-z0-9])?(\\.[a-z0-9]([-a-z0-9]*[a-z0-9])?)*')")},
			taintRule: func() *resourceapi.DeviceTaintRule {
				taintRule := testDeviceTaintRule(goodName, validDeviceTaintRuleSpec)
				taintRule.Spec.DeviceSelector.Driver = ptr.To(badName)
				return taintRule
			}(),
		},
		"bad-pool": {
			wantFailures: field.ErrorList{field.Invalid(field.NewPath("spec", "deviceSelector", "pool"), badName, "a lowercase RFC 1123 subdomain must consist of lower case alphanumeric characters, '-' or '.', and must start and end with an alphanumeric character (e.g. 'example.com', regex used for validation is '[a-z0-9]([-a-z0-9]*[a-z0-9])?(\\.[a-z0-9]([-a-z0-9]*[a-z0-9])?)*')")},
			taintRule: func() *resourceapi.DeviceTaintRule {
				taintRule := testDeviceTaintRule(goodName, validDeviceTaintRuleSpec)
				taintRule.Spec.DeviceSelector.Pool = ptr.To(badName)
				return taintRule
			}(),
		},
		"bad-device": {
			wantFailures: field.ErrorList{field.Invalid(field.NewPath("spec", "deviceSelector", "device"), badName, "a lowercase RFC 1123 label must consist of lower case alphanumeric characters or '-', and must start and end with an alphanumeric character (e.g. 'my-name',  or '123-abc', regex used for validation is '[a-z0-9]([-a-z0-9]*[a-z0-9])?')")},
			taintRule: func() *resourceapi.DeviceTaintRule {
				taintRule := testDeviceTaintRule(goodName, validDeviceTaintRuleSpec)
				taintRule.Spec.DeviceSelector.Device = ptr.To(badName)
				return taintRule
			}(),
		},
		"CEL-compile-errors": {
			wantFailures: field.ErrorList{
				field.Invalid(field.NewPath("spec", "deviceSelector", "selectors").Index(1).Child("cel", "expression"), `device.attributes[true].someBoolean`, "compilation failed: ERROR: <input>:1:18: found no matching overload for '_[_]' applied to '(map(string, map(string, any)), bool)'\n | device.attributes[true].someBoolean\n | .................^"),
			},
			taintRule: func() *resourceapi.DeviceTaintRule {
				taintRule := testDeviceTaintRule(goodName, validDeviceTaintRuleSpec)
				taintRule.Spec.DeviceSelector.Selectors = []resourceapi.DeviceSelector{
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
				return taintRule
			}(),
		},
		"CEL-length": {
			wantFailures: field.ErrorList{
				field.TooLong(field.NewPath("spec", "deviceSelector", "selectors").Index(1).Child("cel", "expression"), "" /*unused*/, resourceapi.CELSelectorExpressionMaxLength),
			},
			taintRule: func() *resourceapi.DeviceTaintRule {
				taintRule := testDeviceTaintRule(goodName, validDeviceTaintRuleSpec)
				expression := `device.driver == ""`
				taintRule.Spec.DeviceSelector.Selectors = []resourceapi.DeviceSelector{
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
				return taintRule
			}(),
		},
		"CEL-cost": {
			wantFailures: field.ErrorList{
				field.Forbidden(field.NewPath("spec", "deviceSelector", "selectors").Index(0).Child("cel", "expression"), "too complex, exceeds cost limit"),
			},
			taintRule: func() *resourceapi.DeviceTaintRule {
				claim := testDeviceTaintRule(goodName, validDeviceTaintRuleSpec)
				claim.Spec.DeviceSelector.Selectors = []resourceapi.DeviceSelector{
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
		"valid-taint": {
			taintRule: func() *resourceapi.DeviceTaintRule {
				claim := testDeviceTaintRule(goodName, validDeviceTaintRuleSpec)
				claim.Spec.Taint = resourceapi.DeviceTaint{
					Key:    goodName,
					Value:  goodName,
					Effect: resourceapi.DeviceTaintEffectNoExecute,
				}
				return claim
			}(),
		},
		"invalid-taint": {
			wantFailures: field.ErrorList{
				field.Required(field.NewPath("spec", "taint", "effect"), ""),
			},
			taintRule: func() *resourceapi.DeviceTaintRule {
				claim := testDeviceTaintRule(goodName, validDeviceTaintRuleSpec)
				claim.Spec.Taint = resourceapi.DeviceTaint{
					// Minimal test. Full coverage of validateDeviceTaint is in ResourceSlice test.
					Key:   goodName,
					Value: goodName,
				}
				return claim
			}(),
		},
	}

	for name, scenario := range scenarios {
		t.Run(name, func(t *testing.T) {
			errs := ValidateDeviceTaintRule(scenario.taintRule)
			assertFailures(t, scenario.wantFailures, errs)
		})
	}
}

func TestValidateDeviceTaintUpdate(t *testing.T) {
	name := "valid"
	validTaintRule := testDeviceTaintRule(name, validDeviceTaintRuleSpec)

	scenarios := map[string]struct {
		old          *resourceapi.DeviceTaintRule
		update       func(patch *resourceapi.DeviceTaintRule) *resourceapi.DeviceTaintRule
		wantFailures field.ErrorList
	}{
		"valid-no-op-update": {
			old:    validTaintRule,
			update: func(taintRule *resourceapi.DeviceTaintRule) *resourceapi.DeviceTaintRule { return taintRule },
		},
		"invalid-name-update": {
			wantFailures: field.ErrorList{field.Invalid(field.NewPath("metadata", "name"), name+"-update", "field is immutable")},
			old:          validTaintRule,
			update: func(taintRule *resourceapi.DeviceTaintRule) *resourceapi.DeviceTaintRule {
				taintRule.Name += "-update"
				return taintRule
			},
		},
	}

	for name, scenario := range scenarios {
		t.Run(name, func(t *testing.T) {
			scenario.old.ResourceVersion = "1"
			errs := ValidateDeviceTaintRuleUpdate(scenario.update(scenario.old.DeepCopy()), scenario.old)
			assertFailures(t, scenario.wantFailures, errs)
		})
	}
}
