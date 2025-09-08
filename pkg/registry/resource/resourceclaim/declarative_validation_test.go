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
package resourceclaim

import (
	"testing"

	v1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/validation/field"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/kubernetes/fake"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	apitesting "k8s.io/kubernetes/pkg/api/testing"
	"k8s.io/kubernetes/pkg/apis/resource"
	"k8s.io/kubernetes/pkg/features"
)

var apiVersions = []string{"v1beta1", "v1beta2", "v1"} // "v1alpha3" is excluded because it doesn't have ResourceClaim

func TestDeclarativeValidate(t *testing.T) {
	for _, apiVersion := range apiVersions {
		t.Run(apiVersion, func(t *testing.T) {
			testDeclarativeValidate(t, apiVersion)
		})
	}
}

func testDeclarativeValidate(t *testing.T, apiVersion string) {
	ctx := genericapirequest.WithRequestInfo(genericapirequest.NewDefaultContext(), &genericapirequest.RequestInfo{
		APIGroup:   "resource.k8s.io",
		APIVersion: apiVersion,
		Resource:   "resourceclaims",
	})
	fakeClient := fake.NewSimpleClientset()
	mockNSClient := fakeClient.CoreV1().Namespaces()
	Strategy := NewStrategy(mockNSClient)
	testCases := map[string]struct {
		input        resource.ResourceClaim
		expectedErrs field.ErrorList
	}{
		"valid": {
			input: mkValidResourceClaim(),
		},
		// TODO: Add more test cases
	}
	for k, tc := range testCases {
		t.Run(k, func(t *testing.T) {
			var declarativeTakeoverErrs field.ErrorList
			var imperativeErrs field.ErrorList
			for _, gateVal := range []bool{true, false} {
				featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.DeclarativeValidation, gateVal)
				featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.DeclarativeValidationTakeover, gateVal)

				errs := Strategy.Validate(ctx, &tc.input)
				if gateVal {
					declarativeTakeoverErrs = errs
				} else {
					imperativeErrs = errs
				}
			}
			equivalenceMatcher := field.ErrorMatcher{}.ByType().ByField().ByOrigin()
			equivalenceMatcher.Test(t, imperativeErrs, declarativeTakeoverErrs)

			apitesting.VerifyVersionedValidationEquivalence(t, &tc.input, nil)
		})
	}
}

func TestDeclarativeValidateUpdate(t *testing.T) {
	for _, apiVersion := range apiVersions {
		t.Run(apiVersion, func(t *testing.T) {
			testDeclarativeValidateUpdate(t, apiVersion)
		})
	}
}

func testDeclarativeValidateUpdate(t *testing.T, apiVersion string) {
	ctx := genericapirequest.WithRequestInfo(genericapirequest.NewDefaultContext(), &genericapirequest.RequestInfo{
		APIGroup:   "resource.k8s.io",
		APIVersion: apiVersion,
		Resource:   "resourceclaims",
	})
	fakeClient := fake.NewSimpleClientset()
	mockNSClient := fakeClient.CoreV1().Namespaces()
	Strategy := NewStrategy(mockNSClient)
	validClaim := mkValidResourceClaim()
	testCases := map[string]struct {
		update       resource.ResourceClaim
		old          resource.ResourceClaim
		expectedErrs field.ErrorList
	}{
		"valid": {
			update: validClaim,
			old:    validClaim,
		},
		// TODO: Add more test cases
	}
	for k, tc := range testCases {
		t.Run(k, func(t *testing.T) {
			var declarativeTakeoverErrs field.ErrorList
			var imperativeErrs field.ErrorList
			for _, gateVal := range []bool{true, false} {
				featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.DeclarativeValidation, gateVal)
				featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.DeclarativeValidationTakeover, gateVal)

				errs := Strategy.ValidateUpdate(ctx, &tc.update, &tc.old)
				if gateVal {
					declarativeTakeoverErrs = errs
				} else {
					imperativeErrs = errs
				}
			}
			equivalenceMatcher := field.ErrorMatcher{}.ByType().ByField().ByOrigin()
			equivalenceMatcher.Test(t, imperativeErrs, declarativeTakeoverErrs)

			apitesting.VerifyVersionedValidationEquivalence(t, &tc.update, &tc.old)
		})
	}
}

func mkValidResourceClaim() resource.ResourceClaim {
	return resource.ResourceClaim{
		ObjectMeta: v1.ObjectMeta{
			Name:      "valid-claim",
			Namespace: "default",
		},
		Spec: resource.ResourceClaimSpec{
			Devices: resource.DeviceClaim{
				Requests: []resource.DeviceRequest{
					{
						Name: "req-0",
						Exactly: &resource.ExactDeviceRequest{
							DeviceClassName: "class",
							AllocationMode:  resource.DeviceAllocationModeAll,
						},
					},
				},
			},
		},
	}
}
