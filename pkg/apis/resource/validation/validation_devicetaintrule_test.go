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
		Driver: ptr.To("test.example.com"),
		Pool:   ptr.To(goodName),
		Device: ptr.To(goodName),
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
		// Minimal tests for DeviceTaint. Full coverage of validateDeviceTaint is in ResourceSlice test.
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
		"required-taint": {
			wantFailures: field.ErrorList{
				field.Required(field.NewPath("spec", "taint", "effect"), "").MarkCoveredByDeclarative(),
			},
			taintRule: func() *resourceapi.DeviceTaintRule {
				claim := testDeviceTaintRule(goodName, validDeviceTaintRuleSpec)
				claim.Spec.Taint = resourceapi.DeviceTaint{
					Key:   goodName,
					Value: goodName,
				}
				return claim
			}(),
		},
		"invalid-taint": {
			wantFailures: field.ErrorList{
				field.NotSupported(field.NewPath("spec", "taint", "effect"), resourceapi.DeviceTaintEffect("some-other-effect"), []resourceapi.DeviceTaintEffect{resourceapi.DeviceTaintEffectNoExecute, resourceapi.DeviceTaintEffectNoSchedule, resourceapi.DeviceTaintEffectNone}).MarkCoveredByDeclarative(),
			},
			taintRule: func() *resourceapi.DeviceTaintRule {
				claim := testDeviceTaintRule(goodName, validDeviceTaintRuleSpec)
				claim.Spec.Taint = resourceapi.DeviceTaint{
					Effect: "some-other-effect",
					Key:    goodName,
					Value:  goodName,
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
	invalidTaintEffectRule := validTaintRule.DeepCopy()
	invalidTaintEffectRule.Spec.Taint.Effect = "some-other-effect"

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
		"valid-existing-unknown-effect": {
			old: invalidTaintEffectRule,
			update: func(taintRule *resourceapi.DeviceTaintRule) *resourceapi.DeviceTaintRule {
				taintRule.Labels = map[string]string{"a": "b"}
				return taintRule
			},
		},
		"invalid-new-unknown-effect": {
			wantFailures: field.ErrorList{field.NotSupported(field.NewPath("spec", "taint", "effect"), resourceapi.DeviceTaintEffect("some-other-effect"), []resourceapi.DeviceTaintEffect{resourceapi.DeviceTaintEffectNoExecute, resourceapi.DeviceTaintEffectNoSchedule, resourceapi.DeviceTaintEffectNone})}.MarkCoveredByDeclarative(),
			old:          validTaintRule,
			update: func(taintRule *resourceapi.DeviceTaintRule) *resourceapi.DeviceTaintRule {
				taintRule.Spec.Taint.Effect = "some-other-effect"
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
