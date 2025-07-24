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
	"testing"

	apiresource "k8s.io/apimachinery/pkg/api/resource"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/kubernetes/pkg/apis/resource"
	"k8s.io/utils/ptr"
)

func testDeviceCapacity(value apiresource.Quantity, policy *resource.CapacityRequestPolicy) resource.DeviceCapacity {
	return resource.DeviceCapacity{
		Value:         value,
		RequestPolicy: policy,
	}
}

func testCapacityRequestPolicy(defaultValue *apiresource.Quantity,
	validValues []apiresource.Quantity,
	validRange *resource.CapacityRequestPolicyRange) *resource.CapacityRequestPolicy {
	return &resource.CapacityRequestPolicy{
		Default:     defaultValue,
		ValidValues: validValues,
		ValidRange:  validRange,
	}
}

func testValidRange(min *apiresource.Quantity, max, step *apiresource.Quantity) *resource.CapacityRequestPolicyRange {
	return &resource.CapacityRequestPolicyRange{
		Min:  min,
		Max:  max,
		Step: step,
	}
}

func TestValidateDeviceCapacity(t *testing.T) {
	oneKIAbbreviated := apiresource.MustParse("1Ki")
	oneKIUnabbreviated := apiresource.MustParse("1024")

	one := apiresource.MustParse("1Gi")
	two := apiresource.MustParse("2Gi")
	maxCapacity := apiresource.MustParse("10Gi")
	overCapacity := apiresource.MustParse("20Gi")

	capacityField := field.NewPath("spec", "devices", "capacity")
	policyField := capacityField.Child("requestPolicy")
	validValuesField := policyField.Child("validValues")
	validRangeField := policyField.Child("validRange")

	scenarios := map[string]struct {
		capacity     resource.DeviceCapacity
		wantFailures field.ErrorList
	}{
		"no-policy": {
			capacity: testDeviceCapacity(one, nil),
		},
		"policy-without-default": {
			capacity: testDeviceCapacity(one, testCapacityRequestPolicy(nil, nil, nil)),
		},
		"policy-with-valid-values-without-default": {
			capacity: testDeviceCapacity(one, testCapacityRequestPolicy(nil, []apiresource.Quantity{one}, nil)),
			wantFailures: field.ErrorList{
				field.Required(policyField.Child("default"), "required when validValues is defined"),
			},
		},
		"policy-with-valid-range-without-default": {
			capacity: testDeviceCapacity(one, testCapacityRequestPolicy(nil, nil, testValidRange(ptr.To(one), nil, nil))),
			wantFailures: field.ErrorList{
				field.Required(policyField.Child("default"), "required when validRange is defined"),
			},
		},
		"policy-with-valid-range-without-min": {
			capacity: testDeviceCapacity(one, testCapacityRequestPolicy(&one, nil, testValidRange(nil, nil, nil))),
			wantFailures: field.ErrorList{
				field.Required(validRangeField.Child("min"), "required when validRange is defined"),
			},
		},
		"valid-simple-range": {
			capacity: testDeviceCapacity(maxCapacity, testCapacityRequestPolicy(&one, nil, testValidRange(ptr.To(one), nil, nil))),
		},
		"valid-range-with-max": {
			capacity: testDeviceCapacity(maxCapacity, testCapacityRequestPolicy(&one, nil, testValidRange(ptr.To(one), ptr.To(maxCapacity), nil))),
		},
		"valid-range-with-step": {
			capacity: testDeviceCapacity(maxCapacity, testCapacityRequestPolicy(&one, nil, testValidRange(ptr.To(one), nil, ptr.To(one)))),
		},
		"valid-range-with-max-and-step": {
			capacity: testDeviceCapacity(maxCapacity, testCapacityRequestPolicy(&one, nil, testValidRange(ptr.To(one), ptr.To(maxCapacity), ptr.To(one)))),
		},
		"valid-single-option": {
			capacity: testDeviceCapacity(maxCapacity, testCapacityRequestPolicy(&one, []apiresource.Quantity{one}, nil)),
		},
		"valid-two-options": {
			capacity: testDeviceCapacity(maxCapacity, testCapacityRequestPolicy(&one, []apiresource.Quantity{one, maxCapacity}, nil)),
		},
		"default-without-policy": {
			capacity: testDeviceCapacity(maxCapacity, testCapacityRequestPolicy(&one, nil, nil)),
		},
		"more-than-one-policy": {
			capacity: testDeviceCapacity(maxCapacity, testCapacityRequestPolicy(&one,
				[]apiresource.Quantity{one}, testValidRange(ptr.To(one), nil, nil))),
			wantFailures: field.ErrorList{
				field.Forbidden(policyField, `exactly one policy can be specified, cannot specify "validValues" and "validRange" at the same time`),
			},
		},
		"invalid-options": {
			capacity: testDeviceCapacity(maxCapacity, testCapacityRequestPolicy(&one, []apiresource.Quantity{overCapacity}, nil)),
			wantFailures: field.ErrorList{
				field.Invalid(validValuesField.Index(0), "20Gi", "option is larger than capacity value: 10Gi"),
				field.Invalid(validValuesField, "1Gi", "default value is not valid according to the requestPolicy"),
			},
		},
		"invalid-options-duplicate": {
			capacity: testDeviceCapacity(maxCapacity, testCapacityRequestPolicy(&one, []apiresource.Quantity{one, one}, nil)),
			wantFailures: field.ErrorList{
				field.Duplicate(validValuesField.Index(1), "1073741824"), // 1Gi
			},
		},
		"invalid-options-duplicate-normalized": {
			capacity: testDeviceCapacity(maxCapacity, testCapacityRequestPolicy(&oneKIAbbreviated, []apiresource.Quantity{oneKIAbbreviated, oneKIUnabbreviated}, nil)),
			wantFailures: field.ErrorList{
				field.Duplicate(validValuesField.Index(1), "1024"),
			},
		},
		"invalid-options-unsort": {
			capacity: testDeviceCapacity(maxCapacity, testCapacityRequestPolicy(&one, []apiresource.Quantity{two, one}, nil)),
			wantFailures: field.ErrorList{
				field.Invalid(validValuesField.Index(1), one.String(), "values must be sorted in ascending order"),
			},
		},
		"invalid-range-large-min-small-max": {
			capacity: testDeviceCapacity(maxCapacity, testCapacityRequestPolicy(&two, nil, testValidRange(ptr.To(overCapacity), ptr.To(one), nil))),
			wantFailures: field.ErrorList{
				field.Invalid(validRangeField.Child("min"), "20Gi", "min is larger than capacity value: 10Gi"),
				field.Invalid(validRangeField.Child("min"), "2Gi", "default is less than min: 20Gi"),
				field.Invalid(validRangeField.Child("max"), "20Gi", "min is larger than max: 1Gi"),
				field.Invalid(validRangeField.Child("max"), "2Gi", "default is more than max: 1Gi"),
			},
		},
		"invalid-range-large-max": {
			capacity: testDeviceCapacity(maxCapacity, testCapacityRequestPolicy(&one, nil, testValidRange(ptr.To(one), ptr.To(overCapacity), nil))),
			wantFailures: field.ErrorList{
				field.Invalid(validRangeField.Child("max"), "20Gi", "max is larger than capacity value: 10Gi"),
			},
		},
		"invalid-range-multiple-of-step": {
			capacity: testDeviceCapacity(maxCapacity, testCapacityRequestPolicy(&two, nil, testValidRange(ptr.To(one), ptr.To(maxCapacity), ptr.To(two)))),
			wantFailures: field.ErrorList{
				field.Invalid(validRangeField.Child("step"), "2Gi", "value is not a multiple of a given step (2Gi) from (1Gi)"),
				field.Invalid(validRangeField.Child("step"), "10Gi", "value is not a multiple of a given step (2Gi) from (1Gi)"),
			},
		},
		"invalid-range-large-step": {
			capacity: testDeviceCapacity(maxCapacity, testCapacityRequestPolicy(&one, nil, testValidRange(ptr.To(one), nil, ptr.To(maxCapacity)))),
			wantFailures: field.ErrorList{
				field.Invalid(validRangeField.Child("step"), "10Gi", "one step 11Gi is larger than capacity value: 10Gi"),
			},
		},
	}
	for name, scenario := range scenarios {
		t.Run(name, func(t *testing.T) {
			errs := validateMultiAllocatableDeviceCapacity(scenario.capacity, capacityField)
			assertFailures(t, scenario.wantFailures, errs)
		})
	}
}
