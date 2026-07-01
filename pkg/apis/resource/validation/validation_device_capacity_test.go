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
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/kubernetes/pkg/apis/resource"
	"k8s.io/kubernetes/pkg/features"
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

	hundredMilli := apiresource.MustParse("100m")
	twoHundredMilli := apiresource.MustParse("200m")
	oneUnit := apiresource.MustParse("1")

	one := apiresource.MustParse("1Gi")
	two := apiresource.MustParse("2Gi")
	maxCapacity := apiresource.MustParse("10Gi")
	overCapacity := apiresource.MustParse("20Gi")

	capacityField := field.NewPath("spec", "devices", "capacity")
	policyField := capacityField.Child("requestPolicy")
	validValuesField := policyField.Child("validValues")
	validRangeField := policyField.Child("validRange")

	scenarios := map[string]struct {
		capacity                    resource.DeviceCapacity
		oldPolicy                   *resource.CapacityRequestPolicy
		fractionalCapacityRangeGate bool
		wantFailures                field.ErrorList
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
		"valid-fractional-values": {
			// 100m (0.1), 200m (0.2), 1 are ascending, all ≤ maxCapacity; AsDec normalises them distinctly
			capacity:                    testDeviceCapacity(maxCapacity, testCapacityRequestPolicy(&hundredMilli, []apiresource.Quantity{hundredMilli, twoHundredMilli, oneUnit}, nil)),
			fractionalCapacityRangeGate: true,
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
		"valid-range-fractional-step": {
			// min=0.2, step=0.1, max=1, default=0.2, capacity=1: 0.2 = min+0*0.1, 1.0 = min+8*0.1
			capacity: testDeviceCapacity(
				apiresource.MustParse("1"),
				testCapacityRequestPolicy(
					ptr.To(apiresource.MustParse("200m")),
					nil,
					testValidRange(
						ptr.To(apiresource.MustParse("200m")),
						ptr.To(apiresource.MustParse("1")),
						ptr.To(apiresource.MustParse("100m")),
					),
				),
			),
			fractionalCapacityRangeGate: true,
		},
		"invalid-range-fractional-step-not-aligned": {
			// default=0.25 is not a multiple of 0.1 from 0.2; gate must be on for milli-arithmetic
			capacity: testDeviceCapacity(
				apiresource.MustParse("1"),
				testCapacityRequestPolicy(
					ptr.To(apiresource.MustParse("250m")),
					nil,
					testValidRange(
						ptr.To(apiresource.MustParse("200m")),
						ptr.To(apiresource.MustParse("1")),
						ptr.To(apiresource.MustParse("100m")),
					),
				),
			),
			fractionalCapacityRangeGate: true,
			wantFailures: field.ErrorList{
				field.Invalid(validRangeField.Child("step"), "250m", "value is not a multiple of a given step (100m) from (200m)"),
			},
		},
		"invalid-range-large-step": {
			capacity: testDeviceCapacity(maxCapacity, testCapacityRequestPolicy(&one, nil, testValidRange(ptr.To(one), nil, ptr.To(maxCapacity)))),
			wantFailures: field.ErrorList{
				field.Invalid(validRangeField.Child("step"), "10Gi", "one step 11Gi is larger than capacity value: 10Gi"),
			},
		},
		"fractional-range-gate-disabled-valid": {
			// gate off, fractional values present — no overflow error, no milli-arithmetic alignment
			capacity: testDeviceCapacity(
				apiresource.MustParse("1"),
				testCapacityRequestPolicy(
					ptr.To(apiresource.MustParse("200m")),
					nil,
					testValidRange(
						ptr.To(apiresource.MustParse("200m")),
						ptr.To(apiresource.MustParse("1")),
						ptr.To(apiresource.MustParse("100m")),
					),
				),
			),
			// fractionalCapacityRangeGate: false (default)
		},
		"fractional-range-gate-disabled-misaligned": {
			// gate off, fractional values present — hasFractional stays false, integer arithmetic
			// treats all sub-integer values as 1, so no misalignment is detected
			capacity: testDeviceCapacity(
				apiresource.MustParse("1"),
				testCapacityRequestPolicy(
					ptr.To(apiresource.MustParse("250m")),
					nil,
					testValidRange(
						ptr.To(apiresource.MustParse("200m")),
						ptr.To(apiresource.MustParse("1")),
						ptr.To(apiresource.MustParse("100m")),
					),
				),
			),
			// fractionalCapacityRangeGate: false (default) — no errors expected
		},
		"identical-range-update-skips-validation": {
			// when oldRange == newRange the function returns nil immediately,
			// even if the range would otherwise fail (e.g. 10P > MaxMilliValue with gate on)
			capacity: testDeviceCapacity(
				apiresource.MustParse("100P"),
				testCapacityRequestPolicy(
					ptr.To(apiresource.MustParse("10P")),
					nil,
					testValidRange(ptr.To(apiresource.MustParse("10P")), nil, ptr.To(apiresource.MustParse("100m"))),
				),
			),
			oldPolicy: testCapacityRequestPolicy(
				ptr.To(apiresource.MustParse("10P")),
				nil,
				testValidRange(ptr.To(apiresource.MustParse("10P")), nil, ptr.To(apiresource.MustParse("100m"))),
			),
			fractionalCapacityRangeGate: true,
			// no wantFailures: identical range short-circuits all checks
		},
		"fractional-range-overflow-min": {
			// min exceeds MaxMilliValue — rejected when gate is on and range is fractional
			capacity: testDeviceCapacity(
				apiresource.MustParse("100P"),
				testCapacityRequestPolicy(
					ptr.To(apiresource.MustParse("20P")),
					nil,
					testValidRange(ptr.To(apiresource.MustParse("10P")), nil, ptr.To(apiresource.MustParse("100m"))),
				),
			),
			fractionalCapacityRangeGate: true,
			wantFailures: field.ErrorList{
				field.Invalid(validRangeField.Child("default"), "20P", "value cannot be represented as a milli value"),
				field.Invalid(validRangeField.Child("min"), "10P", "value cannot be represented as a milli value"),
			},
		},
		"fractional-too-fine-precision": {
			capacity: testDeviceCapacity(
				apiresource.MustParse("1"),
				testCapacityRequestPolicy(
					ptr.To(apiresource.MustParse("20u")),
					nil,
					testValidRange(ptr.To(apiresource.MustParse("10u")), nil, nil),
				),
			),
			fractionalCapacityRangeGate: true,
			wantFailures: field.ErrorList{
				field.Invalid(validRangeField.Child("default"), "20u", "value cannot be represented as a milli value"),
				field.Invalid(validRangeField.Child("min"), "10u", "value cannot be represented as a milli value"),
			},
		},
		"fractional-range-stored-object-exemption": {
			// min=10P exceeds MaxMilliValue and would normally trigger the overflow guard,
			// but the old stored range already had the same fractional step — so the overflow
			// check is skipped to avoid breaking updates of objects that pre-date the gate.
			capacity: testDeviceCapacity(
				apiresource.MustParse("100P"),
				testCapacityRequestPolicy(
					ptr.To(apiresource.MustParse("10P")),
					nil,
					testValidRange(ptr.To(apiresource.MustParse("10P")), nil, ptr.To(apiresource.MustParse("100m"))),
				),
			),
			oldPolicy: testCapacityRequestPolicy(
				ptr.To(apiresource.MustParse("10P")),
				nil,
				testValidRange(ptr.To(apiresource.MustParse("10P")), nil, ptr.To(apiresource.MustParse("100m"))),
			),
			fractionalCapacityRangeGate: true,
		},
	}
	for name, scenario := range scenarios {
		t.Run(name, func(t *testing.T) {
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.DRAFractionalCapacityRange, scenario.fractionalCapacityRangeGate)
			errs := validateMultiAllocatableDeviceCapacity(scenario.capacity, scenario.oldPolicy, capacityField)
			assertFailures(t, scenario.wantFailures, errs)
		})
	}
}
