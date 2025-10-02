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
	"fmt"
	"strings"
	"testing"

	"github.com/google/uuid"
	v1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/validation/field"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/client-go/kubernetes/fake"
	apitesting "k8s.io/kubernetes/pkg/api/testing"
	"k8s.io/kubernetes/pkg/apis/resource"
	pointer "k8s.io/utils/ptr"
)

var apiVersions = []string{"v1beta1", "v1beta2", "v1"} // "v1alpha3" is excluded because it doesn't have ResourceClaim

const (
	validUUID  = "550e8400-e29b-41d4-a716-446655440000"
	validUUID1 = "550e8400-e29b-41d4-a716-446655440001"
)

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
		"valid requests, max allowed": {
			input: mkValidResourceClaim(tweakDevicesConfigs(32)),
		},
		"valid constraints, max allowed": {
			input: mkValidResourceClaim(tweakDevicesConstraints(32)),
		},
		"valid config, max allowed": {
			input: mkValidResourceClaim(tweakDevicesConfigs(32)),
		},
		"invalid requests, too many": {
			input: mkValidResourceClaim(tweakDevicesRequests(33)),
			expectedErrs: field.ErrorList{
				field.TooMany(field.NewPath("spec", "devices", "requests"), 33, 32).WithOrigin("maxItems"),
			},
		},
		"invalid constraints, too many": {
			input: mkValidResourceClaim(tweakDevicesConstraints(33)),
			expectedErrs: field.ErrorList{
				field.TooMany(field.NewPath("spec", "devices", "constraints"), 33, 32).WithOrigin("maxItems"),
			},
		},
		"invalid config, too many": {
			input: mkValidResourceClaim(tweakDevicesConfigs(33)),
			expectedErrs: field.ErrorList{
				field.TooMany(field.NewPath("spec", "devices", "config"), 33, 32).WithOrigin("maxItems"),
			},
		},
		"invalid firstAvailable, too many": {
			input: mkValidResourceClaim(tweakFirstAvailable(9)),
			expectedErrs: field.ErrorList{
				field.TooMany(field.NewPath("spec", "devices", "requests").Index(0).Child("firstAvailable"), 9, 8).WithOrigin("maxItems"),
			},
		},
		"invalid selectors, too many": {
			input: mkValidResourceClaim(tweakExactlySelectors(33)),
			expectedErrs: field.ErrorList{
				field.TooMany(field.NewPath("spec", "devices", "requests").Index(0).Child("exactly", "selectors"), 33, 32).WithOrigin("maxItems").MarkCoveredByDeclarative(),
			},
		},
		"invalid subrequest selectors, too many": {
			input: mkValidResourceClaim(tweakSubRequestSelectors(33)),
			expectedErrs: field.ErrorList{
				field.TooMany(field.NewPath("spec", "devices", "requests").Index(0).Child("firstAvailable").Index(0).Child("selectors"), 33, 32).WithOrigin("maxItems"),
			},
		},
		"invalid constraint requests, too many": {
			input: mkValidResourceClaim(tweakConstraintRequests(33)),
			expectedErrs: field.ErrorList{
				field.TooMany(field.NewPath("spec", "devices", "requests"), 33, 32).WithOrigin("maxItems"),
				field.TooMany(field.NewPath("spec", "devices", "constraints").Index(0).Child("requests"), 33, 32).WithOrigin("maxItems"),
			},
		},
		"invalid config requests, too many": {
			input: mkValidResourceClaim(tweakConfigRequests(33)),
			expectedErrs: field.ErrorList{
				field.TooMany(field.NewPath("spec", "devices", "requests"), 33, 32).WithOrigin("maxItems"),
				field.TooMany(field.NewPath("spec", "devices", "config").Index(0).Child("requests"), 33, 32).WithOrigin("maxItems"),
			},
		},
		"valid firstAvailable, max allowed": {
			input: mkValidResourceClaim(tweakFirstAvailable(8)),
		},
		"valid selectors, max allowed": {
			input: mkValidResourceClaim(tweakExactlySelectors(32)),
		},
		"valid subrequest selectors, max allowed": {
			input: mkValidResourceClaim(tweakSubRequestSelectors(32)),
		},
		"valid constraint requests, max allowed": {
			input: mkValidResourceClaim(tweakConstraintRequests(32)),
		},
		"valid config requests, max allowed": {
			input: mkValidResourceClaim(tweakConfigRequests(32)),
		},
		// TODO: Add more test cases
	}
	for k, tc := range testCases {
		t.Run(k, func(t *testing.T) {
			apitesting.VerifyValidationEquivalence(t, ctx, &tc.input, Strategy.Validate, tc.expectedErrs, apitesting.WithNormalizationRules(resourceClaimNormalizationRules...))
		})
	}
}

func tweakDevicesConfigs(items int) func(*resource.ResourceClaim) {
	return func(rc *resource.ResourceClaim) {
		for i := 0; i < items; i++ {
			rc.Spec.Devices.Config = append(rc.Spec.Devices.Config, mkDeviceClaimConfiguration())
		}
	}
}

func tweakDevicesConstraints(items int) func(*resource.ResourceClaim) {
	return func(rc *resource.ResourceClaim) {
		for i := 0; i < items; i++ {
			rc.Spec.Devices.Constraints = append(rc.Spec.Devices.Constraints, mkDeviceConstraint())
		}
	}
}

func tweakDevicesRequests(items int) func(*resource.ResourceClaim) {
	return func(rc *resource.ResourceClaim) {
		// The first request already exists in the valid template
		for i := 1; i < items; i++ {
			rc.Spec.Devices.Requests = append(rc.Spec.Devices.Requests, mkDeviceRequest(fmt.Sprintf("req-%d", i)))
		}
	}
}

func tweakExactlySelectors(items int) func(*resource.ResourceClaim) {
	return func(rc *resource.ResourceClaim) {
		for i := 0; i < items; i++ {
			rc.Spec.Devices.Requests[0].Exactly.Selectors = append(rc.Spec.Devices.Requests[0].Exactly.Selectors,
				resource.DeviceSelector{
					CEL: &resource.CELDeviceSelector{
						Expression: fmt.Sprintf("device.driver == \"test.driver.io%d\"", i),
					},
				},
			)
		}
	}
}

func tweakSubRequestSelectors(items int) func(*resource.ResourceClaim) {
	return func(rc *resource.ResourceClaim) {
		rc.Spec.Devices.Requests[0].Exactly = nil
		rc.Spec.Devices.Requests[0].FirstAvailable = []resource.DeviceSubRequest{
			{
				Name:            "sub-0",
				DeviceClassName: "class",
				AllocationMode:  resource.DeviceAllocationModeAll,
			},
		}
		for i := 0; i < items; i++ {
			rc.Spec.Devices.Requests[0].FirstAvailable[0].Selectors = append(rc.Spec.Devices.Requests[0].FirstAvailable[0].Selectors,
				resource.DeviceSelector{
					CEL: &resource.CELDeviceSelector{
						Expression: fmt.Sprintf("device.driver == \"test.driver.io%d\"", i),
					},
				},
			)
		}
	}
}

func tweakConstraintRequests(count int) func(*resource.ResourceClaim) {
	return func(rc *resource.ResourceClaim) {
		tweakDevicesRequests(count)(rc)
		if len(rc.Spec.Devices.Constraints) == 0 {
			rc.Spec.Devices.Constraints = append(rc.Spec.Devices.Constraints, mkDeviceConstraint())
		}
		rc.Spec.Devices.Constraints[0].Requests = []string{}
		for i := 0; i < count; i++ {
			rc.Spec.Devices.Constraints[0].Requests = append(rc.Spec.Devices.Constraints[0].Requests, fmt.Sprintf("req-%d", i))
		}
	}
}

func tweakConfigRequests(count int) func(*resource.ResourceClaim) {
	return func(rc *resource.ResourceClaim) {
		tweakDevicesRequests(count)(rc)
		if len(rc.Spec.Devices.Config) == 0 {
			rc.Spec.Devices.Config = append(rc.Spec.Devices.Config, mkDeviceClaimConfiguration())
		}
		rc.Spec.Devices.Config[0].Requests = []string{}
		for i := 0; i < count; i++ {
			rc.Spec.Devices.Config[0].Requests = append(rc.Spec.Devices.Config[0].Requests, fmt.Sprintf("req-%d", i))
		}
	}
}

func tweakFirstAvailable(items int) func(*resource.ResourceClaim) {
	return func(rc *resource.ResourceClaim) {
		rc.Spec.Devices.Requests[0].Exactly = nil
		for i := 0; i < items; i++ {
			rc.Spec.Devices.Requests[0].FirstAvailable = append(rc.Spec.Devices.Requests[0].FirstAvailable,
				resource.DeviceSubRequest{
					Name:            fmt.Sprintf("sub-%d", i),
					DeviceClassName: "class",
					AllocationMode:  resource.DeviceAllocationModeAll,
				},
			)
		}
	}
}

func mkDeviceClaimConfiguration() resource.DeviceClaimConfiguration {
	return resource.DeviceClaimConfiguration{
		Requests: []string{"req-0"},
		DeviceConfiguration: resource.DeviceConfiguration{
			Opaque: &resource.OpaqueDeviceConfiguration{
				Driver: "dra.example.com",
				Parameters: runtime.RawExtension{
					Raw: []byte(`{"kind": "foo", "apiVersion": "dra.example.com/v1"}`),
				}},
		},
	}
}

func mkDeviceConstraint() resource.DeviceConstraint {
	return resource.DeviceConstraint{
		Requests:       []string{"req-0"},
		MatchAttribute: pointer.To(resource.FullyQualifiedName("foo/bar")),
	}
}

func mkDeviceRequest(name string) resource.DeviceRequest {
	return resource.DeviceRequest{
		Name: name,
		Exactly: &resource.ExactDeviceRequest{
			DeviceClassName: "class",
			AllocationMode:  resource.DeviceAllocationModeAll,
		},
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
		"spec immutable: modify request class name": {
			update: mkValidResourceClaim(tweakSpecChangeClassName("another-class")),
			old:    validClaim,
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("spec"), "field is immutable", "").WithOrigin("immutable"),
			},
		},
		"spec immutable: add request": {
			update: mkValidResourceClaim(tweakSpecAddRequest(mkDeviceRequest("req-1"))),
			old:    validClaim,
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("spec"), "field is immutable", "").WithOrigin("immutable"),
			},
		},
		"spec immutable: remove request": {
			update: mkValidResourceClaim(tweakSpecRemoveRequest(0)),
			old:    validClaim,
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("spec"), "field is immutable", "").WithOrigin("immutable"),
			},
		},
		"spec immutable: add constraint": {
			update: mkValidResourceClaim(tweakSpecAddConstraint(mkDeviceConstraint())),
			old:    validClaim,
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("spec"), "field is immutable", "").WithOrigin("immutable"),
			},
		},
		// TODO: Add more test cases
	}
	for k, tc := range testCases {
		t.Run(k, func(t *testing.T) {
			tc.old.ResourceVersion = "1"
			tc.update.ResourceVersion = "2"
			apitesting.VerifyUpdateValidationEquivalence(t, ctx, &tc.update, &tc.old, Strategy.ValidateUpdate, tc.expectedErrs, apitesting.WithNormalizationRules(resourceClaimNormalizationRules...))
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
		Resource:    "resourceclaims",
		Subresource: "status",
	})
	poolPath := field.NewPath("status", "allocation", "devices", "results").Index(0).Child("pool")
	testCases := map[string]struct {
		old          resource.ResourceClaim
		update       resource.ResourceClaim
		expectedErrs field.ErrorList
	}{
		// .Status.Allocation.Devices.Results[%d].Pool
		"valid pool name": {
			old:    mkValidResourceClaim(),
			update: mkResourceClaimWithStatus(tweakStatusDeviceRequestAllocationResultPool("dra.example.com/pool-a")),
		},
		"valid pool name, max length": {
			old:    mkValidResourceClaim(),
			update: mkResourceClaimWithStatus(tweakStatusDeviceRequestAllocationResultPool(strings.Repeat("a", 63) + "." + strings.Repeat("b", 63) + "." + strings.Repeat("c", 63) + "." + strings.Repeat("d", 55))),
		},
		"invalid pool name, required": {
			old:    mkValidResourceClaim(),
			update: mkResourceClaimWithStatus(tweakStatusDeviceRequestAllocationResultPool("")),
			expectedErrs: field.ErrorList{
				field.Required(poolPath, ""),
			},
		},
		"invalid pool name, too long": {
			old:    mkValidResourceClaim(),
			update: mkResourceClaimWithStatus(tweakStatusDeviceRequestAllocationResultPool(strings.Repeat("a", 253) + "/" + strings.Repeat("a", 253))),
			expectedErrs: field.ErrorList{
				field.TooLong(poolPath, "", 253).WithOrigin("format=k8s-resource-pool-name"),
			},
		},
		"invalid pool name, format": {
			old:    mkValidResourceClaim(),
			update: mkResourceClaimWithStatus(tweakStatusDeviceRequestAllocationResultPool("a/Not_Valid")),
			expectedErrs: field.ErrorList{
				field.Invalid(poolPath, "Not_Valid", "").WithOrigin("format=k8s-resource-pool-name"),
			},
		},
		"invalid pool name, leading slash": {
			old:    mkValidResourceClaim(),
			update: mkResourceClaimWithStatus(tweakStatusDeviceRequestAllocationResultPool("/a")),
			expectedErrs: field.ErrorList{
				field.Invalid(poolPath, "", "").WithOrigin("format=k8s-resource-pool-name"),
			},
		},
		"invalid pool name, trailing slash": {
			old:    mkValidResourceClaim(),
			update: mkResourceClaimWithStatus(tweakStatusDeviceRequestAllocationResultPool("a/")),
			expectedErrs: field.ErrorList{
				field.Invalid(poolPath, "", "").WithOrigin("format=k8s-resource-pool-name"),
			},
		},
		"invalid pool name, double slash": {
			old:    mkValidResourceClaim(),
			update: mkResourceClaimWithStatus(tweakStatusDeviceRequestAllocationResultPool("a//b")),
			expectedErrs: field.ErrorList{
				field.Invalid(poolPath, "", "").WithOrigin("format=k8s-resource-pool-name"),
			},
		},
		// .Status.Allocation.Devices.Results[%d].ShareID
		"valid status.Allocation.Devices.Results[].ShareID": {
			old:    mkValidResourceClaim(),
			update: mkResourceClaimWithStatus(tweakStatusDeviceRequestAllocationResultShareID(validUUID)),
		},
		"invalid status.Allocation.Devices.Results[].ShareID": {
			old:    mkValidResourceClaim(),
			update: mkResourceClaimWithStatus(tweakStatusDeviceRequestAllocationResultShareID("invalid-uid")),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("status", "allocation", "devices", "results").Index(0).Child("shareID"), "invalid-uid", "").WithOrigin("format=k8s-uuid"),
			},
		},
		"invalid uppercase status.Allocation.Devices.Results[].ShareID ": {
			old:    mkValidResourceClaim(),
			update: mkResourceClaimWithStatus(tweakStatusDeviceRequestAllocationResultShareID("123e4567-E89b-12d3-A456-426614174000")),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("status", "allocation", "devices", "results").Index(0).Child("shareID"), "invalid-uid", "").WithOrigin("format=k8s-uuid"),
			},
		},
		// .Status.Devices[%d].ShareID
		"valid status.Devices[].ShareID": {
			old: mkValidResourceClaim(),
			update: mkResourceClaimWithStatus(
				tweakStatusDevices(standardAllocatedDeviceStatus()),
				tweakStatusDeviceRequestAllocationResultShareID(validUUID),
				tweakStatusAllocatedDeviceStatusShareID(validUUID),
			),
		},
		"invalid status.Devices[].ShareID": {
			old: mkValidResourceClaim(),
			update: mkResourceClaimWithStatus(
				tweakStatusDevices(standardAllocatedDeviceStatus()),
				tweakStatusDeviceRequestAllocationResultShareID("invalid-uid"),
				tweakStatusAllocatedDeviceStatusShareID("invalid-uid"),
			),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("status", "allocation", "devices", "results").Index(0).Child("shareID"), "invalid-uid", "must be a lowercase UUID in 8-4-4-4-12 format").WithOrigin("format=k8s-uuid"),
				field.Invalid(field.NewPath("status", "devices").Index(0).Child("shareID"), "invalid-uid", "must be a lowercase UUID in 8-4-4-4-12 format").WithOrigin("format=k8s-uuid"),
			},
		},
		"invalid upper case status.Devices[].ShareID": {
			old: mkValidResourceClaim(),
			update: mkResourceClaimWithStatus(
				tweakStatusDevices(standardAllocatedDeviceStatus()),
				tweakStatusDeviceRequestAllocationResultShareID("123e4567-E89b-12d3-A456-426614174000"),
				tweakStatusAllocatedDeviceStatusShareID("123e4567-E89b-12d3-A456-426614174000"),
			),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("status", "allocation", "devices", "results").Index(0).Child("shareID"), "invalid-uid", "must be a lowercase UUID in 8-4-4-4-12 format").WithOrigin("format=k8s-uuid"),
				field.Invalid(field.NewPath("status", "devices").Index(0).Child("shareID"), "invalid-uid", "must be a lowercase UUID in 8-4-4-4-12 format").WithOrigin("format=k8s-uuid"),
			},
		},
		// UID in status.ReservedFor
		"duplicate uid in status.ReservedFor": {
			old: mkValidResourceClaim(),
			update: mkResourceClaimWithStatus(
				tweakStatusReservedFor(
					resourceClaimReference(validUUID),
					resourceClaimReference(uuid.New().String()),
					resourceClaimReference(validUUID),
				)),
			expectedErrs: field.ErrorList{
				field.Duplicate(field.NewPath("status", "reservedFor").Index(2), ""),
			},
		},
		"multiple- duplicate uid in status.ReservedFor": {
			old: mkValidResourceClaim(),
			update: mkResourceClaimWithStatus(tweakStatusReservedFor(
				resourceClaimReference(validUUID),
				resourceClaimReference(uuid.New().String()),
				resourceClaimReference(validUUID),
				resourceClaimReference(validUUID1),
				resourceClaimReference(validUUID1),
				resourceClaimReference(uuid.New().String()),
			)),
			expectedErrs: field.ErrorList{
				field.Duplicate(field.NewPath("status", "reservedFor").Index(2), ""),
				field.Duplicate(field.NewPath("status", "reservedFor").Index(4), ""),
			},
		},
	}
	for k, tc := range testCases {
		t.Run(k, func(t *testing.T) {
			tc.old.ObjectMeta.ResourceVersion = "1"
			tc.update.ObjectMeta.ResourceVersion = "1"
			apitesting.VerifyUpdateValidationEquivalence(t, ctx, &tc.update, &tc.old, strategy.ValidateUpdate, tc.expectedErrs, apitesting.WithSubResources("status"))
		})
	}
}

func mkValidResourceClaim(tweaks ...func(rc *resource.ResourceClaim)) resource.ResourceClaim {
	rc := resource.ResourceClaim{
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

	for _, tweak := range tweaks {
		tweak(&rc)
	}
	return rc
}

func mkResourceClaimWithStatus(tweaks ...func(rc *resource.ResourceClaim)) resource.ResourceClaim {
	rc := resource.ResourceClaim{
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
	for _, tweak := range tweaks {
		tweak(&rc)
	}
	return rc
}

func tweakStatusDevices(devices ...resource.AllocatedDeviceStatus) func(rc *resource.ResourceClaim) {
	return func(rc *resource.ResourceClaim) {
		rc.Status.Devices = devices
	}
}

func tweakStatusDeviceRequestAllocationResultPool(pool string) func(rc *resource.ResourceClaim) {
	return func(rc *resource.ResourceClaim) {
		for i := range rc.Status.Allocation.Devices.Results {
			rc.Status.Allocation.Devices.Results[i].Pool = pool
		}
	}
}

func tweakStatusDeviceRequestAllocationResultShareID(shareID types.UID) func(rc *resource.ResourceClaim) {
	return func(rc *resource.ResourceClaim) {
		for i := range rc.Status.Allocation.Devices.Results {
			rc.Status.Allocation.Devices.Results[i].ShareID = &shareID
		}
	}
}

func tweakSpecChangeClassName(deviceClassName string) func(rc *resource.ResourceClaim) {
	return func(rc *resource.ResourceClaim) {
		if len(rc.Spec.Devices.Requests) > 0 && rc.Spec.Devices.Requests[0].Exactly != nil {
			rc.Spec.Devices.Requests[0].Exactly.DeviceClassName = deviceClassName
		}
	}
}

func tweakStatusAllocatedDeviceStatusShareID(shareID string) func(rc *resource.ResourceClaim) {
	return func(rc *resource.ResourceClaim) {
		for i := range rc.Status.Devices {
			rc.Status.Devices[i].ShareID = &shareID
		}
	}
}

func tweakSpecAddRequest(req resource.DeviceRequest) func(rc *resource.ResourceClaim) {
	return func(rc *resource.ResourceClaim) {
		rc.Spec.Devices.Requests = append(rc.Spec.Devices.Requests, req)
	}
}

func tweakSpecRemoveRequest(index int) func(rc *resource.ResourceClaim) {
	return func(rc *resource.ResourceClaim) {
		if index >= 0 && index < len(rc.Spec.Devices.Requests) {
			rc.Spec.Devices.Requests = append(rc.Spec.Devices.Requests[:index], rc.Spec.Devices.Requests[index+1:]...)
		}
	}
}

func standardAllocatedDeviceStatus() resource.AllocatedDeviceStatus {
	return resource.AllocatedDeviceStatus{
		Driver: "dra.example.com",
		Pool:   "pool-0",
		Device: "device-0",
	}
}

func resourceClaimReference(uid string) resource.ResourceClaimConsumerReference {
	return resource.ResourceClaimConsumerReference{
		UID:      types.UID(uid),
		Resource: "Pod",
		Name:     "pod-name",
	}
}

func tweakStatusReservedFor(refs ...resource.ResourceClaimConsumerReference) func(rc *resource.ResourceClaim) {
	return func(rc *resource.ResourceClaim) {
		rc.Status.ReservedFor = refs
	}
}
func tweakSpecAddConstraint(c resource.DeviceConstraint) func(rc *resource.ResourceClaim) {
	return func(rc *resource.ResourceClaim) {
		rc.Spec.Devices.Constraints = append(rc.Spec.Devices.Constraints, c)
	}
}
