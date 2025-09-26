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
	"strings"
	"testing"

	v1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/validation/field"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/client-go/kubernetes/fake"
	apitesting "k8s.io/kubernetes/pkg/api/testing"
	"k8s.io/kubernetes/pkg/apis/resource"
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
	fakeClient := fake.NewClientset()
	mockNSClient := fakeClient.CoreV1().Namespaces()
	Strategy := NewStrategy(mockNSClient)
	testCases := map[string]struct {
		input        resource.ResourceClaim
		expectedErrs field.ErrorList
	}{
		"valid": {
			input: mkValidResourceClaim(),
		},
		"valid opaque driver, lowercase": {
			input: mkDeviceConfig(mkValidResourceClaim(), "dra.example.com"),
		},
		"valid opaque driver, mixed case": {
			input: mkDeviceConfig(mkValidResourceClaim(), "DRA.Example.COM"),
		},
		"valid opaque driver, max length": {
			input: mkDeviceConfig(mkValidResourceClaim(), strings.Repeat("a", 63)),
		},
		"invalid opaque driver, empty": {
			input: mkDeviceConfig(mkValidResourceClaim(), ""),
			expectedErrs: field.ErrorList{
				field.Required(field.NewPath("spec", "devices", "config").Index(0).Child("opaque", "driver"), ""),
			},
		},
		"invalid opaque driver, too long": {
			input: mkDeviceConfig(mkValidResourceClaim(), strings.Repeat("a", 64)),
			expectedErrs: field.ErrorList{
				field.TooLong(field.NewPath("spec", "devices", "config").Index(0).Child("opaque", "driver"), "", 63),
			},
		},
		"invalid opaque driver, invalid character": {
			input: mkDeviceConfig(mkValidResourceClaim(), "dra_example.com"),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("spec", "devices", "config").Index(0).Child("opaque", "driver"), "dra_example.com", "").WithOrigin("format=k8s-long-name-caseless"),
			},
		},
		"invalid opaque driver, invalid DNS name (leading dot)": {
			input: mkDeviceConfig(mkValidResourceClaim(), ".example.com"),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("spec", "devices", "config").Index(0).Child("opaque", "driver"), ".example.com", "").WithOrigin("format=k8s-long-name-caseless"),
			},
		},
		// TODO: Add more test cases
	}
	for k, tc := range testCases {
		t.Run(k, func(t *testing.T) {
			apitesting.VerifyValidationEquivalence(t, ctx, &tc.input, Strategy.Validate, tc.expectedErrs)
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
	fakeClient := fake.NewClientset()
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
			tc.old.ResourceVersion = "1"
			tc.update.ResourceVersion = "2"
			apitesting.VerifyUpdateValidationEquivalence(t, ctx, &tc.update, &tc.old, Strategy.ValidateUpdate, tc.expectedErrs)
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

func mkDeviceConfig(claim resource.ResourceClaim, driverName string) resource.ResourceClaim {
	claim.Spec.Devices.Config = []resource.DeviceClaimConfiguration{
		{
			Requests: []string{"req-0"},
			DeviceConfiguration: resource.DeviceConfiguration{
				Opaque: &resource.OpaqueDeviceConfiguration{
					Driver:     driverName,
					Parameters: runtime.RawExtension{Raw: []byte(`{"key":"value"}`)},
				},
			},
		},
	}
	return claim
}
