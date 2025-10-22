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

	opaqueDriverPath := field.NewPath("spec", "devices", "config").Index(0).Child("opaque", "driver")

	// TODO: As we accumulate more and more test cases, consider breaking this
	// up into smaller tests for maintainability.
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
		"invalid requests, duplicate name": {
			input: mkValidResourceClaim(tweakAddDeviceRequest(mkDeviceRequest("req-0"))),
			expectedErrs: field.ErrorList{
				field.Duplicate(field.NewPath("spec", "devices", "requests").Index(1), "req-0"),
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
		"invalid firstAvailable, duplicate name": {
			input: mkValidResourceClaim(tweakDuplicateFirstAvailableName("sub-0")),
			expectedErrs: field.ErrorList{
				field.Duplicate(field.NewPath("spec", "devices", "requests").Index(0).Child("firstAvailable").Index(1), "sub-0"),
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
		"invalid constraint requests, duplicate name": {
			input: mkValidResourceClaim(tweakDuplicateConstraintRequest("req-0")),
			expectedErrs: field.ErrorList{
				field.Duplicate(field.NewPath("spec", "devices", "constraints").Index(0).Child("requests").Index(1), "req-0"),
			},
		},
		"invalid config requests, duplicate name": {
			input: mkValidResourceClaim(tweakDuplicateConfigRequest("req-0")),
			expectedErrs: field.ErrorList{
				field.Duplicate(field.NewPath("spec", "devices", "config").Index(0).Child("requests").Index(1), "req-0"),
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
		"valid opaque driver, lowercase": {
			input: mkValidResourceClaim(tweakDeviceConfigWithDriver("dra.example.com")),
		},
		"valid opaque driver, mixed case": {
			input: mkValidResourceClaim(tweakDeviceConfigWithDriver("DRA.Example.COM")),
		},
		"valid opaque driver, max length": {
			input: mkValidResourceClaim(tweakDeviceConfigWithDriver(strings.Repeat("a", 63))),
		},
		"invalid opaque driver, empty": {
			input: mkValidResourceClaim(tweakDeviceConfigWithDriver("")),
			expectedErrs: field.ErrorList{
				field.Required(opaqueDriverPath, ""),
			},
		},
		"invalid opaque driver, too long": {
			input: mkValidResourceClaim(tweakDeviceConfigWithDriver(strings.Repeat("a", 64))),
			expectedErrs: field.ErrorList{
				field.TooLong(opaqueDriverPath, "", 63),
			},
		},
		"invalid opaque driver, invalid character": {
			input: mkValidResourceClaim(tweakDeviceConfigWithDriver("dra_example.com")),
			expectedErrs: field.ErrorList{
				field.Invalid(opaqueDriverPath, "dra_example.com", "").WithOrigin("format=k8s-long-name-caseless"),
			},
		},
		"invalid opaque driver, invalid DNS name (leading dot)": {
			input: mkValidResourceClaim(tweakDeviceConfigWithDriver(".example.com")),
			expectedErrs: field.ErrorList{
				field.Invalid(opaqueDriverPath, ".example.com", "").WithOrigin("format=k8s-long-name-caseless"),
			},
		},
		// spec.Devices.Requests[%d].Exactly.Tolerations.Key
		"valid Exactly.Tolerations.Key": {
			input: mkValidResourceClaim(tweakExactlyTolerations([]resource.DeviceToleration{
				{Key: "valid-key", Operator: resource.DeviceTolerationOpExists},
			})),
		},
		"valid Exactly.Tolerations.Key empty": {
			input: mkValidResourceClaim(tweakExactlyTolerations([]resource.DeviceToleration{
				{Key: "", Operator: resource.DeviceTolerationOpExists},
			})),
		},
		"invalid Exactly.Tolerations.Key": {
			input: mkValidResourceClaim(tweakExactlyTolerations([]resource.DeviceToleration{
				{Key: "invalid_key!", Operator: resource.DeviceTolerationOpExists},
			})),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("spec", "devices", "requests").Index(0).Child("exactly", "tolerations").Index(0).Child("key"), "invalid_key!", "").WithOrigin("format=k8s-label-key"),
			},
		},
		"invalid  Exactly.Tolerations.Key - multiple slashes": {
			input: mkValidResourceClaim(tweakExactlyTolerations([]resource.DeviceToleration{
				{Key: "a/b/c", Operator: resource.DeviceTolerationOpExists},
			})),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("spec", "devices", "requests").Index(0).Child("exactly", "tolerations").Index(0).Child("key"), "a/b/c", "").WithOrigin("format=k8s-label-key"),
			},
		},
		// spec.Devices.Requests[%d].FirsAvailable[%d].Tolerations.Key
		"valid FirstAvailable.Tolerations.Key": {
			input: mkValidResourceClaim(tweakFirstAvailable(1), tweakFirstAvailableTolerations([]resource.DeviceToleration{
				{Key: "valid-key", Operator: resource.DeviceTolerationOpExists},
			})),
		},
		"valid FirstAvailable.Tolerations.Key empty": {
			input: mkValidResourceClaim(tweakFirstAvailable(1), tweakFirstAvailableTolerations([]resource.DeviceToleration{
				{Key: "", Operator: resource.DeviceTolerationOpExists},
			})),
		},
		"invalid FirstAvailable.Tolerations.Key": {
			input: mkValidResourceClaim(tweakFirstAvailable(1), tweakFirstAvailableTolerations([]resource.DeviceToleration{
				{Key: "invalid_key!", Operator: resource.DeviceTolerationOpExists},
			})),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("spec", "devices", "requests").Index(0).Child("firstAvailable").Index(0).Child("tolerations").Index(0).Child("key"), "invalid_key!", "").WithOrigin("format=k8s-label-key"),
			},
		},
		"invalid FirstAvailable.Tolerations.Key - multiple slashes": {
			input: mkValidResourceClaim(tweakFirstAvailable(1), tweakFirstAvailableTolerations([]resource.DeviceToleration{
				{Key: "a/b/c", Operator: resource.DeviceTolerationOpExists},
			})),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("spec", "devices", "requests").Index(0).Child("firstAvailable").Index(0).Child("tolerations").Index(0).Child("key"), "a/b/c", "").WithOrigin("format=k8s-label-key"),
			},
		},
		// TODO: Add more test cases
		"valid DeviceAllocationMode - All": {
			input: mkValidResourceClaim(tweakAllocationMode(resource.DeviceAllocationModeAll, 0)),
		},
		"invalid DeviceAllocationMode - Exactly": {
			input: mkValidResourceClaim(tweakAllocationMode("InvalidMode", 1)),
			expectedErrs: field.ErrorList{
				field.NotSupported(
					field.NewPath("spec", "devices", "requests").Index(0).Child("exactly", "allocationMode"),
					resource.DeviceAllocationMode("InvalidMode"),
					[]string{"All", "ExactCount"},
				).WithOrigin("enum"),
			},
		},
		"valid DeviceAllocationMode - FirstAvailable": {
			input: mkValidResourceClaim(tweakFirstAvailableAllocationMode(resource.DeviceAllocationModeAll, 0)),
		},
		"invalid DeviceAllocationMode - FirstAvailable": {
			input: mkValidResourceClaim(tweakFirstAvailableAllocationMode("InvalidMode", 1)),
			expectedErrs: field.ErrorList{
				field.NotSupported(
					field.NewPath("spec", "devices", "requests").Index(0).Child("firstAvailable").Index(0).Child("allocationMode"),
					resource.DeviceAllocationMode("InvalidMode"),
					[]string{"All", "ExactCount"},
				).WithOrigin("enum"),
			},
		},
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

func tweakDuplicateFirstAvailableName(name string) func(*resource.ResourceClaim) {
	return func(rc *resource.ResourceClaim) {
		rc.Spec.Devices.Requests[0].Exactly = nil
		rc.Spec.Devices.Requests[0].FirstAvailable = []resource.DeviceSubRequest{
			{
				Name:            name,
				DeviceClassName: "class",
				AllocationMode:  resource.DeviceAllocationModeAll,
			},
			{
				Name:            name,
				DeviceClassName: "class",
				AllocationMode:  resource.DeviceAllocationModeAll,
			},
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

func tweakDuplicateConstraintRequest(name string) func(*resource.ResourceClaim) {
	return func(rc *resource.ResourceClaim) {
		if len(rc.Spec.Devices.Constraints) == 0 {
			rc.Spec.Devices.Constraints = append(rc.Spec.Devices.Constraints, mkDeviceConstraint())
		}
		rc.Spec.Devices.Constraints[0].Requests = append(rc.Spec.Devices.Constraints[0].Requests, name)
	}
}

func tweakDuplicateConfigRequest(name string) func(*resource.ResourceClaim) {
	return func(rc *resource.ResourceClaim) {
		if len(rc.Spec.Devices.Config) == 0 {
			rc.Spec.Devices.Config = append(rc.Spec.Devices.Config, mkDeviceClaimConfiguration())
		}
		rc.Spec.Devices.Config[0].Requests = append(rc.Spec.Devices.Config[0].Requests, name)
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

func tweakAllocationMode(mode resource.DeviceAllocationMode, count int64) func(*resource.ResourceClaim) {
	return func(rc *resource.ResourceClaim) {
		if len(rc.Spec.Devices.Requests) > 0 && rc.Spec.Devices.Requests[0].Exactly != nil {
			rc.Spec.Devices.Requests[0].Exactly.AllocationMode = mode
			rc.Spec.Devices.Requests[0].Exactly.Count = count
		}
	}
}

func tweakFirstAvailableAllocationMode(mode resource.DeviceAllocationMode, count int64) func(*resource.ResourceClaim) {
	return func(rc *resource.ResourceClaim) {
		if len(rc.Spec.Devices.Requests) > 0 {
			// Clear Exactly and set FirstAvailable
			rc.Spec.Devices.Requests[0].Exactly = nil
			rc.Spec.Devices.Requests[0].FirstAvailable = []resource.DeviceSubRequest{
				{
					Name:            "sub-0",
					DeviceClassName: "class",
					AllocationMode:  mode,
					Count:           count,
				},
			}
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
			AllocationMode:  resource.DeviceAllocationModeExactCount,
			Count:           1,
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
	// TODO: As we accumulate more and more test cases, consider breaking this
	// up into smaller tests for maintainability.
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
			update: mkValidResourceClaim(tweakAddDeviceRequest(mkDeviceRequest("req-1"))),
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
		"spec immutable: short-circuits other errors (e.g. TooMany)": {
			update: mkValidResourceClaim(tweakDevicesRequests(33)),
			old:    mkValidResourceClaim(),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("spec"), "field is immutable", "").WithOrigin("immutable"),
			},
		},
		"spec immutable: add Exactly.Tolerations": {
			update: mkValidResourceClaim(tweakExactlyTolerations([]resource.DeviceToleration{
				{Key: "valid-key", Operator: resource.DeviceTolerationOpExists},
			})),
			old: validClaim,
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("spec"), "field is immutable", "").WithOrigin("immutable"),
			},
		},
		"spec immutable: change Exactly.Tolerations.Key": {
			update: mkValidResourceClaim(tweakExactlyTolerations([]resource.DeviceToleration{
				{Key: "another-valid-key", Operator: resource.DeviceTolerationOpExists},
			})),
			old: mkValidResourceClaim(tweakExactlyTolerations([]resource.DeviceToleration{
				{Key: "valid-key", Operator: resource.DeviceTolerationOpExists},
			})),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("spec"), "field is immutable", "").WithOrigin("immutable"),
			},
		},
		// spec.Devices.Requests[%d].FirsAvailable[%d].Tolerations.Key
		"spec immutable: add FirstAvailable.Tolerations": {
			update: mkValidResourceClaim(tweakFirstAvailable(1), tweakFirstAvailableTolerations([]resource.DeviceToleration{
				{Key: "valid-key", Operator: resource.DeviceTolerationOpExists},
			})),
			old: mkValidResourceClaim(tweakFirstAvailable(1)),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("spec"), "field is immutable", "").WithOrigin("immutable"),
			},
		},
		"spec immutable: change FirstAvailable.Tolerations.Key": {
			update: mkValidResourceClaim(tweakFirstAvailable(1), tweakFirstAvailableTolerations([]resource.DeviceToleration{
				{Key: "another-valid-key", Operator: resource.DeviceTolerationOpExists},
			})),
			old: mkValidResourceClaim(tweakFirstAvailable(1), tweakFirstAvailableTolerations([]resource.DeviceToleration{
				{Key: "valid-key", Operator: resource.DeviceTolerationOpExists},
			})),
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
	for _, apiVersion := range apiVersions {
		t.Run(apiVersion, func(t *testing.T) {
			testValidateStatusUpdateForDeclarative(t, apiVersion)
		})
	}
}

func testValidateStatusUpdateForDeclarative(t *testing.T, apiVersion string) {
	fakeClient := fake.NewClientset()
	mockNSClient := fakeClient.CoreV1().Namespaces()
	Strategy := NewStrategy(mockNSClient)
	strategy := NewStatusStrategy(Strategy)

	ctx := genericapirequest.WithRequestInfo(genericapirequest.NewDefaultContext(), &genericapirequest.RequestInfo{
		APIGroup:    "resource.k8s.io",
		APIVersion:  apiVersion,
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
		"invalid status.ReservedFor, too many": {
			old: mkValidResourceClaim(),
			update: mkResourceClaimWithStatus(
				tweakStatusReservedFor(generateResourceClaimReferences(257)...),
			),
			expectedErrs: field.ErrorList{
				field.TooMany(field.NewPath("status", "reservedFor"), 257, 256).WithOrigin("maxItems"),
			},
		},
		"valid status.ReservedFor, max items": {
			old: mkValidResourceClaim(),
			update: mkResourceClaimWithStatus(
				tweakStatusReservedFor(generateResourceClaimReferences(256)...),
			),
		},
		"valid status.allocation.devices.results, max items": {
			old: mkValidResourceClaim(),
			update: mkResourceClaimWithStatus(
				tweakStatusAllocationDevicesResults(32),
			),
		},
		"valid status.allocation unchanged": {
			old:    mkResourceClaimWithStatus(),
			update: mkResourceClaimWithStatus(),
		},
		"valid status.allocation set from nil": {
			old:    mkValidResourceClaim(),
			update: mkResourceClaimWithStatus(),
		},
		"valid status.allocation cleared (Unset is allowed)": {
			old:    mkResourceClaimWithStatus(),
			update: mkValidResourceClaim(),
		},
		"invalid status.allocation changed device (NoModify)": {
			old:    mkResourceClaimWithStatus(),
			update: tweakStatusAllocationDevice(mkResourceClaimWithStatus(), "device-different"),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("status", "allocation"), nil, "field is immutable").WithOrigin("update"),
			},
		},
		"invalid status.allocation changed driver (NoModify)": {
			old:    mkResourceClaimWithStatus(),
			update: tweakStatusAllocationDriver(mkResourceClaimWithStatus(), "different.example.com"),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("status", "allocation"), nil, "field is immutable").WithOrigin("update"),
			},
		},
		"invalid status.allocation changed pool (NoModify)": {
			old:    mkResourceClaimWithStatus(),
			update: tweakStatusAllocationPool(mkResourceClaimWithStatus(), "different-pool"),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("status", "allocation"), nil, "field is immutable").WithOrigin("update"),
			},
		},
		"invalid status.allocation added result (NoModify)": {
			old:    mkResourceClaimWithStatus(),
			update: addStatusAllocationResult(mkResourceClaimWithStatus()),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("status", "allocation"), nil, "field is immutable").WithOrigin("update"),
			},
		},
		"invalid status.allocation removed result (NoModify)": {
			old:    addStatusAllocationResult(mkResourceClaimWithStatus()),
			update: mkResourceClaimWithStatus(),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("status", "allocation"), nil, "field is immutable").WithOrigin("update"),
			},
		},
		"invalid status.allocation.devices.results, too many": {
			old: mkValidResourceClaim(),
			update: mkResourceClaimWithStatus(
				tweakStatusAllocationDevicesResults(33),
			),
			expectedErrs: field.ErrorList{
				field.TooMany(field.NewPath("status", "allocation", "devices", "results"), 33, 32).WithOrigin("maxItems"),
			},
		},
		"valid status.allocation.devices.config, max items": {
			old: mkValidResourceClaim(),
			update: mkResourceClaimWithStatus(
				tweakStatusAllocationDevicesConfig(64),
			),
		},
		"invalid status.allocation.devices.config, too many": {
			old: mkValidResourceClaim(),
			update: mkResourceClaimWithStatus(
				tweakStatusAllocationDevicesConfig(65),
			),
			expectedErrs: field.ErrorList{
				field.TooMany(field.NewPath("status", "allocation", "devices", "config"), 33, 32).WithOrigin("maxItems"),
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
					mkDeviceRequest("req-0"),
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
	rc := mkValidResourceClaim()
	rc.Status = resource.ResourceClaimStatus{
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

func tweakAddDeviceRequest(req resource.DeviceRequest) func(rc *resource.ResourceClaim) {
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

func generateResourceClaimReferences(count int) []resource.ResourceClaimConsumerReference {
	refs := make([]resource.ResourceClaimConsumerReference, count)
	for i := 0; i < count; i++ {
		refs[i] = resource.ResourceClaimConsumerReference{
			Resource: "pods",
			Name:     fmt.Sprintf("pod-%d", i),
			UID:      types.UID(uuid.New().String()),
		}
	}
	return refs
}

func tweakStatusReservedFor(refs ...resource.ResourceClaimConsumerReference) func(rc *resource.ResourceClaim) {
	return func(rc *resource.ResourceClaim) {
		rc.Status.ReservedFor = refs
	}
}

func tweakStatusAllocationDevicesResults(count int) func(rc *resource.ResourceClaim) {
	return func(rc *resource.ResourceClaim) {
		rc.Status.Allocation.Devices.Results = []resource.DeviceRequestAllocationResult{}
		for i := 0; i < count; i++ {
			rc.Status.Allocation.Devices.Results = append(rc.Status.Allocation.Devices.Results, resource.DeviceRequestAllocationResult{
				Request: "req-0",
				Driver:  "dra.example.com",
				Pool:    fmt.Sprintf("pool-%d", i),
				Device:  fmt.Sprintf("device-%d", i),
			})
		}
	}
}

func tweakStatusAllocationDevicesConfig(count int) func(rc *resource.ResourceClaim) {
	return func(rc *resource.ResourceClaim) {
		if rc.Status.Allocation == nil {
			return
		}
		rc.Status.Allocation.Devices.Config = []resource.DeviceAllocationConfiguration{}
		for i := 0; i < count; i++ {
			rc.Status.Allocation.Devices.Config = append(rc.Status.Allocation.Devices.Config, resource.DeviceAllocationConfiguration{
				Source:   resource.AllocationConfigSourceClaim,
				Requests: []string{"req-0"},
				DeviceConfiguration: resource.DeviceConfiguration{
					Opaque: &resource.OpaqueDeviceConfiguration{
						Driver: "dra.example.com",
						Parameters: runtime.RawExtension{
							Raw: []byte(fmt.Sprintf(`{"item": %d}`, i)),
						},
					},
				},
			})
		}
	}
}

func tweakSpecAddConstraint(c resource.DeviceConstraint) func(rc *resource.ResourceClaim) {
	return func(rc *resource.ResourceClaim) {
		rc.Spec.Devices.Constraints = append(rc.Spec.Devices.Constraints, c)
	}
}

func tweakDeviceConfigWithDriver(driverName string) func(rc *resource.ResourceClaim) {
	return func(rc *resource.ResourceClaim) {
		rc.Spec.Devices.Config = []resource.DeviceClaimConfiguration{
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
	}
}

func tweakExactlyTolerations(tolerations []resource.DeviceToleration) func(rc *resource.ResourceClaim) {
	return func(rc *resource.ResourceClaim) {
		for i := range rc.Spec.Devices.Requests {
			rc.Spec.Devices.Requests[i].Exactly.Tolerations = tolerations
		}
	}
}

func tweakFirstAvailableTolerations(tolerations []resource.DeviceToleration) func(rc *resource.ResourceClaim) {
	return func(rc *resource.ResourceClaim) {
		for i := range rc.Spec.Devices.Requests {
			for j := range rc.Spec.Devices.Requests[i].FirstAvailable {
				rc.Spec.Devices.Requests[i].FirstAvailable[j].Tolerations = tolerations
			}
		}
	}
}

func tweakStatusAllocationDevice(obj resource.ResourceClaim, device string) resource.ResourceClaim {
	if obj.Status.Allocation != nil && len(obj.Status.Allocation.Devices.Results) > 0 {
		obj.Status.Allocation.Devices.Results[0].Device = device
	}
	return obj
}

func tweakStatusAllocationDriver(obj resource.ResourceClaim, driver string) resource.ResourceClaim {
	if obj.Status.Allocation != nil && len(obj.Status.Allocation.Devices.Results) > 0 {
		obj.Status.Allocation.Devices.Results[0].Driver = driver
	}
	return obj
}

func tweakStatusAllocationPool(obj resource.ResourceClaim, pool string) resource.ResourceClaim {
	if obj.Status.Allocation != nil && len(obj.Status.Allocation.Devices.Results) > 0 {
		obj.Status.Allocation.Devices.Results[0].Pool = pool
	}
	return obj
}

func addStatusAllocationResult(obj resource.ResourceClaim) resource.ResourceClaim {
	if obj.Status.Allocation != nil {
		obj.Status.Allocation.Devices.Results = append(obj.Status.Allocation.Devices.Results,
			resource.DeviceRequestAllocationResult{
				Request: "req-0",
				Driver:  "another.example.com",
				Pool:    "pool-1",
				Device:  "device-1",
			})
	}
	return obj
}
