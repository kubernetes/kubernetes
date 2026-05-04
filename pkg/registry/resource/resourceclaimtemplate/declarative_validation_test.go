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

package resourceclaimtemplate

import (
	"fmt"
	"strings"
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/validation/field"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/client-go/kubernetes/fake"
	apitesting "k8s.io/kubernetes/pkg/api/testing"
	"k8s.io/kubernetes/pkg/apis/resource"
	"k8s.io/kubernetes/pkg/apis/resource/validation"
	pointer "k8s.io/utils/ptr"
)

var apiVersions = []string{"v1beta1", "v1beta2", "v1"}

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
		Resource:   "resourceclaimtemplates",
	})
	fakeClient := fake.NewClientset()
	nsClient := fakeClient.CoreV1().Namespaces()
	Strategy := NewStrategy(nsClient)

	opaqueDriverPath := field.NewPath("spec", "spec", "devices", "config").Index(0).Child("opaque", "driver")

	testCases := map[string]struct {
		input        resource.ResourceClaimTemplate
		expectedErrs field.ErrorList
	}{
		"valid": {
			input: mkValidResourceClaimTemplate(),
		},
		"valid requests, max allowed": {
			input: mkValidResourceClaimTemplate(tweakDevicesRequests(32)),
		},
		"valid constraints, max allowed": {
			input: mkValidResourceClaimTemplate(tweakDevicesConstraints(32)),
		},
		"valid config, max allowed": {
			input: mkValidResourceClaimTemplate(tweakDevicesConfigs(32)),
		},
		"invalid requests, too many": {
			input: mkValidResourceClaimTemplate(tweakDevicesRequests(33)),
			expectedErrs: field.ErrorList{
				field.TooMany(field.NewPath("spec", "spec", "devices", "requests"), 33, 32).WithOrigin("maxItems").MarkAlpha(),
			},
		},
		"invalid requests, duplicate name": {
			input: mkValidResourceClaimTemplate(tweakAddDeviceRequest(mkDeviceRequest("req-0"))),
			expectedErrs: field.ErrorList{
				field.Duplicate(field.NewPath("spec", "spec", "devices", "requests").Index(1), "req-0").MarkAlpha(),
			},
		},
		"invalid constraints, too many": {
			input: mkValidResourceClaimTemplate(tweakDevicesConstraints(33)),
			expectedErrs: field.ErrorList{
				field.TooMany(field.NewPath("spec", "spec", "devices", "constraints"), 33, 32).WithOrigin("maxItems").MarkAlpha(),
			},
		},
		"invalid config, too many": {
			input: mkValidResourceClaimTemplate(tweakDevicesConfigs(33)),
			expectedErrs: field.ErrorList{
				field.TooMany(field.NewPath("spec", "spec", "devices", "config"), 33, 32).WithOrigin("maxItems").MarkAlpha(),
			},
		},
		"invalid firstAvailable, too many": {
			input: mkValidResourceClaimTemplate(tweakFirstAvailable(9)),
			expectedErrs: field.ErrorList{
				field.TooMany(field.NewPath("spec", "spec", "devices", "requests").Index(0).Child("firstAvailable"), 9, 8).WithOrigin("maxItems").MarkAlpha(),
			},
		},
		"invalid firstAvailable, duplicate name": {
			input: mkValidResourceClaimTemplate(tweakDuplicateFirstAvailableName("sub-0")),
			expectedErrs: field.ErrorList{
				field.Duplicate(field.NewPath("spec", "spec", "devices", "requests").Index(0).Child("firstAvailable").Index(1), "sub-0").MarkAlpha(),
			},
		},
		"invalid selectors, too many": {
			input: mkValidResourceClaimTemplate(tweakExactlySelectors(33)),
			expectedErrs: field.ErrorList{
				field.TooMany(field.NewPath("spec", "spec", "devices", "requests").Index(0).Child("exactly", "selectors"), 33, 32).WithOrigin("maxItems").MarkCoveredByDeclarative().MarkAlpha(),
			},
		},
		"invalid subrequest selectors, too many": {
			input: mkValidResourceClaimTemplate(tweakSubRequestSelectors(33)),
			expectedErrs: field.ErrorList{
				field.TooMany(field.NewPath("spec", "spec", "devices", "requests").Index(0).Child("firstAvailable").Index(0).Child("selectors"), 33, 32).WithOrigin("maxItems").MarkAlpha(),
			},
		},
		"invalid constraint requests, too many": {
			input: mkValidResourceClaimTemplate(tweakConstraintRequests(33)),
			expectedErrs: field.ErrorList{
				field.TooMany(field.NewPath("spec", "spec", "devices", "requests"), 33, 32).WithOrigin("maxItems").MarkAlpha(),
				field.TooMany(field.NewPath("spec", "spec", "devices", "constraints").Index(0).Child("requests"), 33, 32).WithOrigin("maxItems").MarkAlpha(),
			},
		},
		"invalid config requests, too many": {
			input: mkValidResourceClaimTemplate(tweakConfigRequests(33)),
			expectedErrs: field.ErrorList{
				field.TooMany(field.NewPath("spec", "spec", "devices", "requests"), 33, 32).WithOrigin("maxItems").MarkAlpha(),
				field.TooMany(field.NewPath("spec", "spec", "devices", "config").Index(0).Child("requests"), 33, 32).WithOrigin("maxItems").MarkAlpha(),
			},
		},
		"invalid constraint requests, duplicate name": {
			input: mkValidResourceClaimTemplate(tweakDuplicateConstraintRequest("req-0")),
			expectedErrs: field.ErrorList{
				field.Duplicate(field.NewPath("spec", "spec", "devices", "constraints").Index(0).Child("requests").Index(1), "req-0").MarkAlpha(),
			},
		},
		"invalid config requests, duplicate name": {
			input: mkValidResourceClaimTemplate(tweakDuplicateConfigRequest("req-0")),
			expectedErrs: field.ErrorList{
				field.Duplicate(field.NewPath("spec", "spec", "devices", "config").Index(0).Child("requests").Index(1), "req-0").MarkAlpha(),
			},
		},
		"valid opaque driver, lowercase": {
			input: mkValidResourceClaimTemplate(tweakDeviceConfigWithDriver("dra.example.com")),
		},
		"valid opaque driver, mixed case": {
			input: mkValidResourceClaimTemplate(tweakDeviceConfigWithDriver("DRA.Example.COM")),
		},
		"valid opaque driver, max length": {
			input: mkValidResourceClaimTemplate(tweakDeviceConfigWithDriver(strings.Repeat("a", 63))),
		},
		"invalid opaque driver, empty": {
			input: mkValidResourceClaimTemplate(tweakDeviceConfigWithDriver("")),
			expectedErrs: field.ErrorList{
				field.Required(opaqueDriverPath, "").MarkAlpha(),
			},
		},
		"invalid opaque driver, too long - 64 characters": {
			input: mkValidResourceClaimTemplate(tweakDeviceConfigWithDriver(strings.Repeat("a", 64))),
			expectedErrs: field.ErrorList{
				field.TooLong(opaqueDriverPath, "", 63).WithOrigin("maxLength").MarkAlpha(),
			},
		},
		"invalid opaque driver, too long - 255 characters": {
			input: mkValidResourceClaimTemplate(tweakDeviceConfigWithDriver(strings.Repeat("a", 255))),
			expectedErrs: field.ErrorList{
				field.TooLong(opaqueDriverPath, "", 63).WithOrigin("maxLength").MarkAlpha(),
				field.Invalid(opaqueDriverPath, "", "").WithOrigin("format=k8s-long-name-caseless").MarkAlpha(),
			},
		},
		"invalid opaque driver, invalid character": {
			input: mkValidResourceClaimTemplate(tweakDeviceConfigWithDriver("dra_example.com")),
			expectedErrs: field.ErrorList{
				field.Invalid(opaqueDriverPath, "dra_example.com", "").WithOrigin("format=k8s-long-name-caseless").MarkAlpha(),
			},
		},
		"invalid opaque driver, invalid DNS name (leading dot)": {
			input: mkValidResourceClaimTemplate(tweakDeviceConfigWithDriver(".example.com")),
			expectedErrs: field.ErrorList{
				field.Invalid(opaqueDriverPath, ".example.com", "").WithOrigin("format=k8s-long-name-caseless").MarkAlpha(),
			},
		},
		"valid Exactly.Tolerations.Key": {
			input: mkValidResourceClaimTemplate(tweakExactlyTolerations([]resource.DeviceToleration{
				{Key: "valid-key", Operator: resource.DeviceTolerationOpExists, Effect: resource.DeviceTaintEffectNoSchedule},
			})),
		},
		"valid Exactly.Tolerations.Key empty": {
			input: mkValidResourceClaimTemplate(tweakExactlyTolerations([]resource.DeviceToleration{
				{Key: "", Operator: resource.DeviceTolerationOpExists, Effect: resource.DeviceTaintEffectNoSchedule},
			})),
		},
		"invalid Exactly.Tolerations.Key": {
			input: mkValidResourceClaimTemplate(tweakExactlyTolerations([]resource.DeviceToleration{
				{Key: "invalid_key!", Operator: resource.DeviceTolerationOpExists, Effect: resource.DeviceTaintEffectNoSchedule},
			})),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("spec", "spec", "devices", "requests").Index(0).Child("exactly", "tolerations").Index(0).Child("key"), "invalid_key!", "").WithOrigin("format=k8s-label-key").MarkAlpha(),
			},
		},
		"invalid Exactly.Tolerations.Key - multiple slashes": {
			input: mkValidResourceClaimTemplate(tweakExactlyTolerations([]resource.DeviceToleration{
				{Key: "a/b/c", Operator: resource.DeviceTolerationOpExists, Effect: resource.DeviceTaintEffectNoSchedule},
			})),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("spec", "spec", "devices", "requests").Index(0).Child("exactly", "tolerations").Index(0).Child("key"), "a/b/c", "").WithOrigin("format=k8s-label-key").MarkAlpha(),
			},
		},
		"valid FirstAvailable.Tolerations.Key": {
			input: mkValidResourceClaimTemplate(tweakFirstAvailable(1), tweakFirstAvailableTolerations([]resource.DeviceToleration{
				{Key: "valid-key", Operator: resource.DeviceTolerationOpExists, Effect: resource.DeviceTaintEffectNoSchedule},
			})),
		},
		"valid FirstAvailable.Tolerations.Key empty": {
			input: mkValidResourceClaimTemplate(tweakFirstAvailable(1), tweakFirstAvailableTolerations([]resource.DeviceToleration{
				{Key: "", Operator: resource.DeviceTolerationOpExists, Effect: resource.DeviceTaintEffectNoSchedule},
			})),
		},
		"invalid FirstAvailable.Tolerations.Key": {
			input: mkValidResourceClaimTemplate(tweakFirstAvailable(1), tweakFirstAvailableTolerations([]resource.DeviceToleration{
				{Key: "invalid_key!", Operator: resource.DeviceTolerationOpExists, Effect: resource.DeviceTaintEffectNoSchedule},
			})),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("spec", "spec", "devices", "requests").Index(0).Child("firstAvailable").Index(0).Child("tolerations").Index(0).Child("key"), "invalid_key!", "").WithOrigin("format=k8s-label-key").MarkAlpha(),
			},
		},
		"invalid FirstAvailable.Tolerations.Key - multiple slashes": {
			input: mkValidResourceClaimTemplate(tweakFirstAvailable(1), tweakFirstAvailableTolerations([]resource.DeviceToleration{
				{Key: "a/b/c", Operator: resource.DeviceTolerationOpExists, Effect: resource.DeviceTaintEffectNoSchedule},
			})),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("spec", "spec", "devices", "requests").Index(0).Child("firstAvailable").Index(0).Child("tolerations").Index(0).Child("key"), "a/b/c", "").WithOrigin("format=k8s-label-key").MarkAlpha(),
			},
		},
		"valid DeviceAllocationMode - All": {
			input: mkValidResourceClaimTemplate(tweakExactlyAllocationMode(resource.DeviceAllocationModeAll, 0)),
		},
		"invalid DeviceAllocationMode - Exactly": {
			input: mkValidResourceClaimTemplate(tweakExactlyAllocationMode("InvalidMode", 1)),
			expectedErrs: field.ErrorList{
				field.NotSupported(
					field.NewPath("spec", "spec", "devices", "requests").Index(0).Child("exactly", "allocationMode"),
					resource.DeviceAllocationMode("InvalidMode"),
					[]string{"All", "ExactCount"},
				).MarkAlpha(),
			},
		},
		"valid DeviceAllocationMode - FirstAvailable": {
			input: mkValidResourceClaimTemplate(tweakFirstAvailableAllocationMode(resource.DeviceAllocationModeAll, 0)),
		},
		"invalid DeviceAllocationMode - FirstAvailable": {
			input: mkValidResourceClaimTemplate(tweakFirstAvailableAllocationMode("InvalidMode", 1)),
			expectedErrs: field.ErrorList{
				field.NotSupported(
					field.NewPath("spec", "spec", "devices", "requests").Index(0).Child("firstAvailable").Index(0).Child("allocationMode"),
					resource.DeviceAllocationMode("InvalidMode"),
					[]string{"All", "ExactCount"},
				).MarkAlpha(),
			},
		},
		"valid firstAvailable class name": {
			input: mkValidResourceClaimTemplate(tweakFirstAvailableDeviceClassName("class")),
		},
		"invalid firstAvailable class name - invalid characters": {
			input: mkValidResourceClaimTemplate(tweakFirstAvailableDeviceClassName("Class&")),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("spec", "spec", "devices", "requests").Index(0).Child("firstAvailable").Index(0).Child("deviceClassName"), "Class&", "").WithOrigin("format=k8s-long-name").MarkAlpha(),
			},
		},
		"invalid firstAvailable class name - empty": {
			input: mkValidResourceClaimTemplate(tweakFirstAvailableDeviceClassName("")),
			expectedErrs: field.ErrorList{
				field.Required(field.NewPath("spec", "spec", "devices", "requests").Index(0).Child("firstAvailable").Index(0).Child("deviceClassName"), "").MarkAlpha(),
			},
		},
		"valid DeviceTolerationOperator/Effect - Exactly": {
			input: mkValidResourceClaimTemplate(tweakExactlyTolerations([]resource.DeviceToleration{
				{
					Key:      "key",
					Operator: resource.DeviceTolerationOpEqual,
					Value:    "value",
					Effect:   resource.DeviceTaintEffectNoSchedule,
				},
			})),
		},
		"invalid DeviceTolerationOperator empty - Exactly": {
			input: mkValidResourceClaimTemplate(
				tweakExactlyTolerations([]resource.DeviceToleration{{Key: "key", Value: "value", Effect: resource.DeviceTaintEffectNoSchedule, Operator: ""}}),
			),
			expectedErrs: field.ErrorList{
				field.Required(field.NewPath("spec", "spec", "devices", "requests").Index(0).Child("exactly", "tolerations").Index(0).Child("operator"), "").MarkAlpha(),
			},
		},
		"invalid DeviceTolerationOperator - Exactly": {
			input: mkValidResourceClaimTemplate(tweakExactlyTolerations([]resource.DeviceToleration{
				{
					Key:      "key",
					Operator: "InvalidOp",
					Value:    "value",
					Effect:   resource.DeviceTaintEffectNoSchedule,
				},
			})),
			expectedErrs: field.ErrorList{
				field.NotSupported(
					field.NewPath("spec", "spec", "devices", "requests").Index(0).Child("exactly", "tolerations").Index(0).Child("operator"),
					resource.DeviceTolerationOperator("InvalidOp"),
					[]string{"Equal", "Exists"},
				).MarkAlpha(),
			},
		},
		"invalid DeviceTaintEffect - Exactly": {
			input: mkValidResourceClaimTemplate(tweakExactlyTolerations([]resource.DeviceToleration{
				{
					Key:      "key",
					Operator: resource.DeviceTolerationOpEqual,
					Value:    "value",
					Effect:   "InvalidEffect",
				},
			})),
			expectedErrs: field.ErrorList{
				field.NotSupported(
					field.NewPath("spec", "spec", "devices", "requests").Index(0).Child("exactly", "tolerations").Index(0).Child("effect"),
					resource.DeviceTaintEffect("InvalidEffect"),
					[]string{"NoExecute", "NoSchedule"},
				).MarkAlpha(),
			},
		},
		"valid DeviceTolerationOperator/Effect - FirstAvailable": {
			input: mkValidResourceClaimTemplate(tweakFirstAvailable(1), tweakFirstAvailableTolerations([]resource.DeviceToleration{
				{
					Key:      "key",
					Operator: resource.DeviceTolerationOpEqual,
					Value:    "value",
					Effect:   resource.DeviceTaintEffectNoSchedule,
				},
			})),
		},
		"invalid DeviceTolerationOperator empty - FirstAvailable": {
			input: mkValidResourceClaimTemplate(
				tweakFirstAvailable(1),
				tweakFirstAvailableTolerations([]resource.DeviceToleration{{Key: "key", Value: "value", Effect: resource.DeviceTaintEffectNoSchedule, Operator: ""}}),
			),
			expectedErrs: field.ErrorList{
				field.Required(field.NewPath("spec", "spec", "devices", "requests").Index(0).Child("firstAvailable").Index(0).Child("tolerations").Index(0).Child("operator"), "").MarkAlpha(),
			},
		},
		"invalid DeviceTolerationOperator - FirstAvailable": {
			input: mkValidResourceClaimTemplate(tweakFirstAvailable(1), tweakFirstAvailableTolerations([]resource.DeviceToleration{
				{
					Key:      "key",
					Operator: "InvalidOp",
					Value:    "value",
					Effect:   resource.DeviceTaintEffectNoSchedule,
				},
			})),
			expectedErrs: field.ErrorList{
				field.NotSupported(
					field.NewPath("spec", "spec", "devices", "requests").Index(0).Child("firstAvailable").Index(0).Child("tolerations").Index(0).Child("operator"),
					resource.DeviceTolerationOperator("InvalidOp"),
					[]string{"Equal", "Exists"},
				).MarkAlpha(),
			},
		},
		"invalid DeviceTaintEffect - FirstAvailable": {
			input: mkValidResourceClaimTemplate(tweakFirstAvailable(1), tweakFirstAvailableTolerations([]resource.DeviceToleration{
				{
					Key:      "key",
					Operator: resource.DeviceTolerationOpEqual,
					Value:    "value",
					Effect:   "InvalidEffect",
				},
			})),
			expectedErrs: field.ErrorList{
				field.NotSupported(
					field.NewPath("spec", "spec", "devices", "requests").Index(0).Child("firstAvailable").Index(0).Child("tolerations").Index(0).Child("effect"),
					resource.DeviceTaintEffect("InvalidEffect"),
					[]string{"NoExecute", "NoSchedule"},
				).MarkAlpha(),
			},
		},
		"invalid match attribute": {
			input: mkValidResourceClaimTemplate(tweakMatchAttribute("invalid!")),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("spec", "spec", "devices", "constraints").Index(0).Child("matchAttribute"), "invalid!", "").WithOrigin("format=k8s-resource-fully-qualified-name").MarkAlpha(),
			},
		},
		"match attribute without domain": {
			input: mkValidResourceClaimTemplate(tweakMatchAttribute("nodomain")),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("spec", "spec", "devices", "constraints").Index(0).Child("matchAttribute"), "nodomain", "a fully qualified name must be a domain and a name separated by a slash").WithOrigin("format=k8s-resource-fully-qualified-name").MarkAlpha(),
			},
		},
		"match attribute empty": {
			input: mkValidResourceClaimTemplate(tweakMatchAttribute("")),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("spec", "spec", "devices", "constraints").Index(0).Child("matchAttribute"), "", "").WithOrigin("format=k8s-resource-fully-qualified-name").MarkAlpha(),
			},
		},
		// TODO: Add more test cases
	}

	for name, tc := range testCases {
		t.Run(name, func(t *testing.T) {
			apitesting.VerifyValidationEquivalence(t, ctx, &tc.input, Strategy, tc.expectedErrs, apitesting.WithNormalizationRules(validation.ResourceNormalizationRules...))
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
		Resource:   "resourceclaimtemplates",
	})
	fakeClient := fake.NewClientset()
	nsClient := fakeClient.CoreV1().Namespaces()
	Strategy := NewStrategy(nsClient)

	testCases := map[string]struct {
		old          resource.ResourceClaimTemplate
		update       resource.ResourceClaimTemplate
		expectedErrs field.ErrorList
	}{
		"valid update (no spec change)": {
			old:    mkValidResourceClaimTemplate(),
			update: mkValidResourceClaimTemplate(),
		},
		// TODO: Add more test cases
	}

	for name, tc := range testCases {
		t.Run(name, func(t *testing.T) {
			tc.old.ResourceVersion = "1"
			tc.update.ResourceVersion = "2"
			apitesting.VerifyUpdateValidationEquivalence(t, ctx, &tc.update, &tc.old, Strategy, tc.expectedErrs, apitesting.WithNormalizationRules(validation.ResourceNormalizationRules...))
		})
	}
}

// --- Builders & tweaks ---
//
// These mirror the ones in pkg/registry/resource/resourceclaim/declarative_validation_test.go,
// adapted to operate on ResourceClaimTemplate.Spec.Spec.Devices instead of
// ResourceClaim.Spec.Devices. The structural shape is identical.

func mkValidResourceClaimTemplate(tweaks ...func(rct *resource.ResourceClaimTemplate)) resource.ResourceClaimTemplate {
	rct := resource.ResourceClaimTemplate{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "valid-claim-template",
			Namespace: "default",
		},
		Spec: resource.ResourceClaimTemplateSpec{
			Spec: resource.ResourceClaimSpec{
				Devices: resource.DeviceClaim{
					Requests: []resource.DeviceRequest{
						mkDeviceRequest("req-0"),
					},
				},
			},
		},
	}
	for _, tweak := range tweaks {
		tweak(&rct)
	}
	return rct
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

func tweakAddDeviceRequest(req resource.DeviceRequest) func(*resource.ResourceClaimTemplate) {
	return func(rct *resource.ResourceClaimTemplate) {
		rct.Spec.Spec.Devices.Requests = append(rct.Spec.Spec.Devices.Requests, req)
	}
}

func tweakDevicesRequests(items int) func(*resource.ResourceClaimTemplate) {
	return func(rct *resource.ResourceClaimTemplate) {
		// The first request already exists in the valid template
		for i := 1; i < items; i++ {
			rct.Spec.Spec.Devices.Requests = append(rct.Spec.Spec.Devices.Requests, mkDeviceRequest(fmt.Sprintf("req-%d", i)))
		}
	}
}

func tweakDevicesConstraints(items int) func(*resource.ResourceClaimTemplate) {
	return func(rct *resource.ResourceClaimTemplate) {
		for range items {
			rct.Spec.Spec.Devices.Constraints = append(rct.Spec.Spec.Devices.Constraints, mkDeviceConstraint())
		}
	}
}

func tweakDevicesConfigs(items int) func(*resource.ResourceClaimTemplate) {
	return func(rct *resource.ResourceClaimTemplate) {
		for range items {
			rct.Spec.Spec.Devices.Config = append(rct.Spec.Spec.Devices.Config, mkDeviceClaimConfiguration())
		}
	}
}

func tweakFirstAvailable(items int) func(*resource.ResourceClaimTemplate) {
	return func(rct *resource.ResourceClaimTemplate) {
		rct.Spec.Spec.Devices.Requests[0].Exactly = nil
		for i := range items {
			rct.Spec.Spec.Devices.Requests[0].FirstAvailable = append(rct.Spec.Spec.Devices.Requests[0].FirstAvailable,
				resource.DeviceSubRequest{
					Name:            fmt.Sprintf("sub-%d", i),
					DeviceClassName: "class",
					AllocationMode:  resource.DeviceAllocationModeAll,
				},
			)
		}
	}
}

func tweakDuplicateFirstAvailableName(name string) func(*resource.ResourceClaimTemplate) {
	return func(rct *resource.ResourceClaimTemplate) {
		rct.Spec.Spec.Devices.Requests[0].Exactly = nil
		rct.Spec.Spec.Devices.Requests[0].FirstAvailable = []resource.DeviceSubRequest{
			{Name: name, DeviceClassName: "class", AllocationMode: resource.DeviceAllocationModeAll},
			{Name: name, DeviceClassName: "class", AllocationMode: resource.DeviceAllocationModeAll},
		}
	}
}

func tweakExactlySelectors(items int) func(*resource.ResourceClaimTemplate) {
	return func(rct *resource.ResourceClaimTemplate) {
		for i := range items {
			rct.Spec.Spec.Devices.Requests[0].Exactly.Selectors = append(rct.Spec.Spec.Devices.Requests[0].Exactly.Selectors,
				resource.DeviceSelector{
					CEL: &resource.CELDeviceSelector{
						Expression: fmt.Sprintf("device.driver == \"test.driver.io%d\"", i),
					},
				},
			)
		}
	}
}

func tweakSubRequestSelectors(items int) func(*resource.ResourceClaimTemplate) {
	return func(rct *resource.ResourceClaimTemplate) {
		rct.Spec.Spec.Devices.Requests[0].Exactly = nil
		rct.Spec.Spec.Devices.Requests[0].FirstAvailable = []resource.DeviceSubRequest{
			{Name: "sub-0", DeviceClassName: "class", AllocationMode: resource.DeviceAllocationModeAll},
		}
		for i := range items {
			rct.Spec.Spec.Devices.Requests[0].FirstAvailable[0].Selectors = append(rct.Spec.Spec.Devices.Requests[0].FirstAvailable[0].Selectors,
				resource.DeviceSelector{
					CEL: &resource.CELDeviceSelector{
						Expression: fmt.Sprintf("device.driver == \"test.driver.io%d\"", i),
					},
				},
			)
		}
	}
}

func tweakConstraintRequests(count int) func(*resource.ResourceClaimTemplate) {
	return func(rct *resource.ResourceClaimTemplate) {
		tweakDevicesRequests(count)(rct)
		if len(rct.Spec.Spec.Devices.Constraints) == 0 {
			rct.Spec.Spec.Devices.Constraints = append(rct.Spec.Spec.Devices.Constraints, mkDeviceConstraint())
		}
		rct.Spec.Spec.Devices.Constraints[0].Requests = []string{}
		for i := range count {
			rct.Spec.Spec.Devices.Constraints[0].Requests = append(rct.Spec.Spec.Devices.Constraints[0].Requests, fmt.Sprintf("req-%d", i))
		}
	}
}

func tweakConfigRequests(count int) func(*resource.ResourceClaimTemplate) {
	return func(rct *resource.ResourceClaimTemplate) {
		tweakDevicesRequests(count)(rct)
		if len(rct.Spec.Spec.Devices.Config) == 0 {
			rct.Spec.Spec.Devices.Config = append(rct.Spec.Spec.Devices.Config, mkDeviceClaimConfiguration())
		}
		rct.Spec.Spec.Devices.Config[0].Requests = []string{}
		for i := range count {
			rct.Spec.Spec.Devices.Config[0].Requests = append(rct.Spec.Spec.Devices.Config[0].Requests, fmt.Sprintf("req-%d", i))
		}
	}
}

func tweakDuplicateConstraintRequest(name string) func(*resource.ResourceClaimTemplate) {
	return func(rct *resource.ResourceClaimTemplate) {
		if len(rct.Spec.Spec.Devices.Constraints) == 0 {
			rct.Spec.Spec.Devices.Constraints = append(rct.Spec.Spec.Devices.Constraints, mkDeviceConstraint())
		}
		rct.Spec.Spec.Devices.Constraints[0].Requests = append(rct.Spec.Spec.Devices.Constraints[0].Requests, name)
	}
}

func tweakDuplicateConfigRequest(name string) func(*resource.ResourceClaimTemplate) {
	return func(rct *resource.ResourceClaimTemplate) {
		if len(rct.Spec.Spec.Devices.Config) == 0 {
			rct.Spec.Spec.Devices.Config = append(rct.Spec.Spec.Devices.Config, mkDeviceClaimConfiguration())
		}
		rct.Spec.Spec.Devices.Config[0].Requests = append(rct.Spec.Spec.Devices.Config[0].Requests, name)
	}
}

func tweakExactlyAllocationMode(mode resource.DeviceAllocationMode, count int64) func(*resource.ResourceClaimTemplate) {
	return func(rct *resource.ResourceClaimTemplate) {
		if len(rct.Spec.Spec.Devices.Requests) > 0 && rct.Spec.Spec.Devices.Requests[0].Exactly != nil {
			rct.Spec.Spec.Devices.Requests[0].Exactly.AllocationMode = mode
			rct.Spec.Spec.Devices.Requests[0].Exactly.Count = count
		}
	}
}

func tweakFirstAvailableAllocationMode(mode resource.DeviceAllocationMode, count int64) func(*resource.ResourceClaimTemplate) {
	return func(rct *resource.ResourceClaimTemplate) {
		if len(rct.Spec.Spec.Devices.Requests) > 0 {
			rct.Spec.Spec.Devices.Requests[0].Exactly = nil
			rct.Spec.Spec.Devices.Requests[0].FirstAvailable = []resource.DeviceSubRequest{
				{Name: "sub-0", DeviceClassName: "class", AllocationMode: mode, Count: count},
			}
		}
	}
}

func tweakFirstAvailableDeviceClassName(name string) func(*resource.ResourceClaimTemplate) {
	return func(rct *resource.ResourceClaimTemplate) {
		rct.Spec.Spec.Devices.Requests[0].Exactly = nil
		rct.Spec.Spec.Devices.Requests[0].FirstAvailable = []resource.DeviceSubRequest{
			{Name: "sub-0", DeviceClassName: name, AllocationMode: resource.DeviceAllocationModeAll},
		}
	}
}

func tweakDeviceConfigWithDriver(driverName string) func(*resource.ResourceClaimTemplate) {
	return func(rct *resource.ResourceClaimTemplate) {
		rct.Spec.Spec.Devices.Config = []resource.DeviceClaimConfiguration{
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

func tweakExactlyTolerations(tolerations []resource.DeviceToleration) func(*resource.ResourceClaimTemplate) {
	return func(rct *resource.ResourceClaimTemplate) {
		for i := range rct.Spec.Spec.Devices.Requests {
			if rct.Spec.Spec.Devices.Requests[i].Exactly != nil {
				rct.Spec.Spec.Devices.Requests[i].Exactly.Tolerations = tolerations
			}
		}
	}
}

func tweakFirstAvailableTolerations(tolerations []resource.DeviceToleration) func(*resource.ResourceClaimTemplate) {
	return func(rct *resource.ResourceClaimTemplate) {
		for i := range rct.Spec.Spec.Devices.Requests {
			for j := range rct.Spec.Spec.Devices.Requests[i].FirstAvailable {
				rct.Spec.Spec.Devices.Requests[i].FirstAvailable[j].Tolerations = tolerations
			}
		}
	}
}

func tweakMatchAttribute(val string) func(*resource.ResourceClaimTemplate) {
	return func(rct *resource.ResourceClaimTemplate) {
		fullyQualifiedName := resource.FullyQualifiedName(val)
		rct.Spec.Spec.Devices.Constraints = []resource.DeviceConstraint{
			{MatchAttribute: &fullyQualifiedName},
		}
	}
}
