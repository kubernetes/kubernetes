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
	validRange *resource.CapacityRequestPolicyRange,
	zeroConsumption *bool) *resource.CapacityRequestPolicy {
	return &resource.CapacityRequestPolicy{
		Default:         defaultValue,
		ValidValues:     validValues,
		ValidRange:      validRange,
		ZeroConsumption: zeroConsumption,
	}
}

func testValidRange(min apiresource.Quantity, max, step *apiresource.Quantity) *resource.CapacityRequestPolicyRange {
	return &resource.CapacityRequestPolicyRange{
		Min:  ptr.To(min),
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
			capacity: testDeviceCapacity(one, testCapacityRequestPolicy(nil, nil, nil, nil)),
		},
		"policy-with-valid-values-without-default": {
			capacity: testDeviceCapacity(one, testCapacityRequestPolicy(nil, []apiresource.Quantity{one}, nil, nil)),
			wantFailures: field.ErrorList{
				field.Required(policyField.Child("default"), "required when validValues is specified"),
			},
		},
		"policy-with-valid-range-without-default": {
			capacity: testDeviceCapacity(one, testCapacityRequestPolicy(nil, nil, testValidRange(one, nil, nil), nil)),
			wantFailures: field.ErrorList{
				field.Required(policyField.Child("default"), "required when validRange is defined"),
			},
		},
		"valid-simple-range": {
			capacity: testDeviceCapacity(maxCapacity, testCapacityRequestPolicy(&one, nil, testValidRange(one, nil, nil), nil)),
		},
		"valid-range-with-maximum": {
			capacity: testDeviceCapacity(maxCapacity, testCapacityRequestPolicy(&one, nil, testValidRange(one, ptr.To(maxCapacity), nil), nil)),
		},
		"valid-range-with-step": {
			capacity: testDeviceCapacity(maxCapacity, testCapacityRequestPolicy(&one, nil, testValidRange(one, nil, ptr.To(one)), nil)),
		},
		"valid-range-with-maximum-and-step": {
			capacity: testDeviceCapacity(maxCapacity, testCapacityRequestPolicy(&one, nil, testValidRange(one, ptr.To(maxCapacity), ptr.To(one)), nil)),
		},
		"valid-single-option": {
			capacity: testDeviceCapacity(maxCapacity, testCapacityRequestPolicy(&one, []apiresource.Quantity{one}, nil, nil)),
		},
		"valid-two-options": {
			capacity: testDeviceCapacity(maxCapacity, testCapacityRequestPolicy(&one, []apiresource.Quantity{one, maxCapacity}, nil, nil)),
		},
		"valid-zero-consumption": {
			capacity: testDeviceCapacity(maxCapacity, testCapacityRequestPolicy(nil, nil, nil, ptr.To(true))),
		},
		"default-without-policy": {
			capacity: testDeviceCapacity(maxCapacity, testCapacityRequestPolicy(&one, nil, nil, nil)),
		},
		"valid-false-zero-consumption": { // same as nil
			capacity: testDeviceCapacity(maxCapacity, testCapacityRequestPolicy(&one, nil, nil, ptr.To(false))),
		},
		"more-than-one-policy": {
			capacity: testDeviceCapacity(maxCapacity, testCapacityRequestPolicy(&one,
				[]apiresource.Quantity{one}, testValidRange(one, nil, nil), nil)),
			wantFailures: field.ErrorList{
				field.Forbidden(policyField, `exactly one policy can be specified, cannot specify any of "zeroConsumption", "validValues" and "validRange" at the same time`),
			},
		},
		"invalid-options": {
			capacity: testDeviceCapacity(maxCapacity, testCapacityRequestPolicy(&one, []apiresource.Quantity{overCapacity}, nil, nil)),
			wantFailures: field.ErrorList{
				field.Invalid(validValuesField.Index(0), "20Gi", "option is larger than capacity value: 10Gi"),
				field.Invalid(validValuesField, "1Gi", "default value is not valid according to the requestPolicy"),
			},
		},
		"invalid-options-duplicate": {
			capacity: testDeviceCapacity(maxCapacity, testCapacityRequestPolicy(&one, []apiresource.Quantity{one, one}, nil, nil)),
			wantFailures: field.ErrorList{
				field.Duplicate(validValuesField.Index(1), "1073741824"), // 1Gi
			},
		},
		"invalid-options-duplicate-normalized": {
			capacity: testDeviceCapacity(maxCapacity, testCapacityRequestPolicy(&oneKIAbbreviated, []apiresource.Quantity{oneKIAbbreviated, oneKIUnabbreviated}, nil, nil)),
			wantFailures: field.ErrorList{
				field.Duplicate(validValuesField.Index(1), "1024"),
			},
		},
		"invalid-options-unsort": {
			capacity: testDeviceCapacity(maxCapacity, testCapacityRequestPolicy(&one, []apiresource.Quantity{two, one}, nil, nil)),
			wantFailures: field.ErrorList{
				field.Invalid(validValuesField.Index(1), one.String(), "values must be sorted in ascending order"),
			},
		},
		"invalid-range-large-minimum-small-maximum": {
			capacity: testDeviceCapacity(maxCapacity, testCapacityRequestPolicy(&two, nil, testValidRange(overCapacity, ptr.To(one), nil), nil)),
			wantFailures: field.ErrorList{
				field.Invalid(validRangeField.Child("minimum"), "20Gi", "minimum is larger than capacity value: 10Gi"),
				field.Invalid(validRangeField.Child("minimum"), "2Gi", "default is less than minimum: 20Gi"),
				field.Invalid(validRangeField.Child("maximum"), "20Gi", "minimum is larger than maximum: 1Gi"),
				field.Invalid(validRangeField.Child("maximum"), "2Gi", "default is more than maximum: 1Gi"),
			},
		},
		"invalid-range-large-maximum": {
			capacity: testDeviceCapacity(maxCapacity, testCapacityRequestPolicy(&one, nil, testValidRange(one, ptr.To(overCapacity), nil), nil)),
			wantFailures: field.ErrorList{
				field.Invalid(validRangeField.Child("maximum"), "20Gi", "maximum is larger than capacity value: 10Gi"),
			},
		},
		"invalid-range-multiple-of-step": {
			capacity: testDeviceCapacity(maxCapacity, testCapacityRequestPolicy(&two, nil, testValidRange(one, ptr.To(maxCapacity), ptr.To(two)), nil)),
			wantFailures: field.ErrorList{
				field.Invalid(validRangeField.Child("step"), "2Gi", "value is not a multiple of a given step (2Gi) from (1Gi)"),
				field.Invalid(validRangeField.Child("step"), "10Gi", "value is not a multiple of a given step (2Gi) from (1Gi)"),
			},
		},
		"invalid-range-large-step": {
			capacity: testDeviceCapacity(maxCapacity, testCapacityRequestPolicy(&one, nil, testValidRange(one, nil, ptr.To(maxCapacity)), nil)),
			wantFailures: field.ErrorList{
				field.Invalid(validRangeField.Child("step"), "10Gi", "one step 11Gi is larger than capacity value: 10Gi"),
			},
		},
		"forbidden-zero-consumption-with-default": {
			capacity: testDeviceCapacity(maxCapacity, testCapacityRequestPolicy(&one, nil, nil, ptr.To(true))),
			wantFailures: field.ErrorList{
				field.Forbidden(policyField.Child("default"), "default must not be defined when zeroConsumption=true"),
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
