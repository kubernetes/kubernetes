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

func TestValidateStatusUpdateForDeclarative(t *testing.T) {
	fakeClient := fake.NewClientset()
	mockNSClient := fakeClient.CoreV1().Namespaces()
	Strategy := NewStrategy(mockNSClient)
	strategy := NewStatusStrategy(Strategy)

	ctx := genericapirequest.WithRequestInfo(genericapirequest.NewDefaultContext(), &genericapirequest.RequestInfo{
		APIGroup:    "resource.k8s.io",
		APIVersion:  "v1",
		Subresource: "status",
	})
	poolPath := field.NewPath("status", "allocation", "devices", "results").Index(0).Child("pool")
	testCases := map[string]struct {
		old          resource.ResourceClaim
		update       resource.ResourceClaim
		expectedErrs field.ErrorList
	}{
		"valid pool name": {
			old:    mkValidResourceClaim(),
			update: tweakStatusDeviceRequestAllocationResultPool(mkResourceClaimWithStatus(), "dra.example.com/pool-a"),
		},
		"valid pool name, max length": {
			old:    mkValidResourceClaim(),
			update: tweakStatusDeviceRequestAllocationResultPool(mkResourceClaimWithStatus(), strings.Repeat("a", 63)+"."+strings.Repeat("b", 63)+"."+strings.Repeat("c", 63)+"."+strings.Repeat("d", 55)),
		},
		"invalid pool name, required": {
			old:    mkValidResourceClaim(),
			update: tweakStatusDeviceRequestAllocationResultPool(mkResourceClaimWithStatus(), ""),
			expectedErrs: field.ErrorList{
				field.Required(poolPath, ""),
			},
		},
		"invalid pool name, too long": {
			old:    mkValidResourceClaim(),
			update: tweakStatusDeviceRequestAllocationResultPool(mkResourceClaimWithStatus(), strings.Repeat("a", 253)+"/"+strings.Repeat("a", 253)),
			expectedErrs: field.ErrorList{
				field.TooLong(poolPath, "", 253).WithOrigin("format=k8s-resource-pool-name"),
			},
		},
		"invalid pool name, format": {
			old:    mkValidResourceClaim(),
			update: tweakStatusDeviceRequestAllocationResultPool(mkResourceClaimWithStatus(), "a/Not_Valid"),
			expectedErrs: field.ErrorList{
				field.Invalid(poolPath, "Not_Valid", "").WithOrigin("format=k8s-resource-pool-name"),
			},
		},
		"invalid pool name, leading slash": {
			old:    mkValidResourceClaim(),
			update: tweakStatusDeviceRequestAllocationResultPool(mkResourceClaimWithStatus(), "/a"),
			expectedErrs: field.ErrorList{
				field.Invalid(poolPath, "", "").WithOrigin("format=k8s-resource-pool-name"),
			},
		},
		"invalid pool name, trailing slash": {
			old:    mkValidResourceClaim(),
			update: tweakStatusDeviceRequestAllocationResultPool(mkResourceClaimWithStatus(), "a/"),
			expectedErrs: field.ErrorList{
				field.Invalid(poolPath, "", "").WithOrigin("format=k8s-resource-pool-name"),
			},
		},
		"invalid pool name, double slash": {
			old:    mkValidResourceClaim(),
			update: tweakStatusDeviceRequestAllocationResultPool(mkResourceClaimWithStatus(), "a//b"),
			expectedErrs: field.ErrorList{
				field.Invalid(poolPath, "", "").WithOrigin("format=k8s-resource-pool-name"),
			},
		},
	}
	for k, tc := range testCases {
		t.Run(k, func(t *testing.T) {
			apitesting.VerifyUpdateValidationEquivalence(t, ctx, &tc.update, &tc.old, strategy.ValidateUpdate, tc.expectedErrs)
		})
	}
}

func mkValidResourceClaim() resource.ResourceClaim {
	return resource.ResourceClaim{
		ObjectMeta: v1.ObjectMeta{
			Name:            "valid-claim",
			Namespace:       "default",
			ResourceVersion: "0",
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

func mkResourceClaimWithStatus() resource.ResourceClaim {
	return resource.ResourceClaim{
		ObjectMeta: v1.ObjectMeta{
			Name:            "valid-claim",
			Namespace:       "default",
			ResourceVersion: "1",
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
		Status: resource.ResourceClaimStatus{
			Allocation: &resource.AllocationResult{
				Devices: resource.DeviceAllocationResult{
					Results: []resource.DeviceRequestAllocationResult{
						{
							Request: "req-0",
							Driver:  "dra.example.com",
							Pool:    "pool-0",
							Device:  "device-0",
						},
					},
				},
			},
		},
	}
}

func tweakStatusDeviceRequestAllocationResultPool(obj resource.ResourceClaim, pool string) resource.ResourceClaim {
	for i := range obj.Status.Allocation.Devices.Results {
		obj.Status.Allocation.Devices.Results[i].Pool = pool
	}
	return obj
}
