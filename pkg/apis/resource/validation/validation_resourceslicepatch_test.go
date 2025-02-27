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
	"fmt"
	"strings"
	"testing"

	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/validation/field"
	resourceapi "k8s.io/kubernetes/pkg/apis/resource"
	"k8s.io/utils/ptr"
)

func testResourceSlicePatch(name string, spec resourceapi.ResourceSlicePatchSpec) *resourceapi.ResourceSlicePatch {
	return &resourceapi.ResourceSlicePatch{
		ObjectMeta: metav1.ObjectMeta{
			Name: name,
		},
		Spec: *spec.DeepCopy(),
	}
}

var validPatchSpec = resourceapi.ResourceSlicePatchSpec{
	Devices: resourceapi.DevicePatch{
		Filter: &resourceapi.DevicePatchFilter{
			DeviceClassName: ptr.To(goodName),
			Driver:          ptr.To("test.example.com"),
			Pool:            ptr.To(goodName),
			Device:          ptr.To(goodName),
		},
		Priority: ptr.To[int32](100),
		Attributes: map[resourceapi.FullyQualifiedName]resourceapi.NullableDeviceAttribute{
			"test.example.com/int":     {DeviceAttribute: resourceapi.DeviceAttribute{IntValue: ptr.To(int64(42))}},
			"test.example.com/string":  {DeviceAttribute: resourceapi.DeviceAttribute{StringValue: ptr.To("hello world")}},
			"test.example.com/version": {DeviceAttribute: resourceapi.DeviceAttribute{VersionValue: ptr.To("1.2.3")}},
			"test.example.com/bool":    {DeviceAttribute: resourceapi.DeviceAttribute{BoolValue: ptr.To(true)}},
			"test.example.com/null":    {NullValue: &resourceapi.NullValue{}},
		},
		Capacity: map[resourceapi.FullyQualifiedName]resourceapi.DeviceCapacity{
			"test.example.com/memory": {Value: resource.MustParse("1Gi")},
		},
	},
}

func TestValidateResourceSlicePatch(t *testing.T) {
	goodName := "foo"
	now := metav1.Now()
	badName := "!@#$%^"
	badValue := "spaces not allowed"
	goodDomain := "test.example.com"
	goodQualifiedName := goodDomain + "/" + goodName

	scenarios := map[string]struct {
		patch        *resourceapi.ResourceSlicePatch
		wantFailures field.ErrorList
	}{
		"good-patch": {
			patch: testResourceSlicePatch(goodName, validPatchSpec),
		},
		"missing-name": {
			wantFailures: field.ErrorList{field.Required(field.NewPath("metadata", "name"), "name or generateName is required")},
			patch:        testResourceSlicePatch("", validPatchSpec),
		},
		"bad-name": {
			wantFailures: field.ErrorList{field.Invalid(field.NewPath("metadata", "name"), badName, "a lowercase RFC 1123 subdomain must consist of lower case alphanumeric characters, '-' or '.', and must start and end with an alphanumeric character (e.g. 'example.com', regex used for validation is '[a-z0-9]([-a-z0-9]*[a-z0-9])?(\\.[a-z0-9]([-a-z0-9]*[a-z0-9])?)*')")},
			patch:        testResourceSlicePatch(badName, validPatchSpec),
		},
		"generate-name": {
			patch: func() *resourceapi.ResourceSlicePatch {
				patch := testResourceSlicePatch(goodName, validPatchSpec)
				patch.GenerateName = "pvc-"
				return patch
			}(),
		},
		"uid": {
			patch: func() *resourceapi.ResourceSlicePatch {
				patch := testResourceSlicePatch(goodName, validPatchSpec)
				patch.UID = "ac051fac-2ead-46d9-b8b4-4e0fbeb7455d"
				return patch
			}(),
		},
		"resource-version": {
			patch: func() *resourceapi.ResourceSlicePatch {
				patch := testResourceSlicePatch(goodName, validPatchSpec)
				patch.ResourceVersion = "1"
				return patch
			}(),
		},
		"generation": {
			patch: func() *resourceapi.ResourceSlicePatch {
				patch := testResourceSlicePatch(goodName, validPatchSpec)
				patch.Generation = 100
				return patch
			}(),
		},
		"creation-timestamp": {
			patch: func() *resourceapi.ResourceSlicePatch {
				patch := testResourceSlicePatch(goodName, validPatchSpec)
				patch.CreationTimestamp = now
				return patch
			}(),
		},
		"deletion-grace-period-seconds": {
			patch: func() *resourceapi.ResourceSlicePatch {
				patch := testResourceSlicePatch(goodName, validPatchSpec)
				patch.DeletionGracePeriodSeconds = ptr.To(int64(10))
				return patch
			}(),
		},
		"owner-references": {
			patch: func() *resourceapi.ResourceSlicePatch {
				patch := testResourceSlicePatch(goodName, validPatchSpec)
				patch.OwnerReferences = []metav1.OwnerReference{
					{
						APIVersion: "v1",
						Kind:       "pod",
						Name:       "foo",
						UID:        "ac051fac-2ead-46d9-b8b4-4e0fbeb7455d",
					},
				}
				return patch
			}(),
		},
		"finalizers": {
			patch: func() *resourceapi.ResourceSlicePatch {
				patch := testResourceSlicePatch(goodName, validPatchSpec)
				patch.Finalizers = []string{
					"example.com/foo",
				}
				return patch
			}(),
		},
		"managed-fields": {
			patch: func() *resourceapi.ResourceSlicePatch {
				patch := testResourceSlicePatch(goodName, validPatchSpec)
				patch.ManagedFields = []metav1.ManagedFieldsEntry{
					{
						FieldsType: "FieldsV1",
						Operation:  "Apply",
						APIVersion: "apps/v1",
						Manager:    "foo",
					},
				}
				return patch
			}(),
		},
		"good-labels": {
			patch: func() *resourceapi.ResourceSlicePatch {
				patch := testResourceSlicePatch(goodName, validPatchSpec)
				patch.Labels = map[string]string{
					"apps.kubernetes.io/name": "test",
				}
				return patch
			}(),
		},
		"bad-labels": {
			wantFailures: field.ErrorList{field.Invalid(field.NewPath("metadata", "labels"), badValue, "a valid label must be an empty string or consist of alphanumeric characters, '-', '_' or '.', and must start and end with an alphanumeric character (e.g. 'MyValue',  or 'my_value',  or '12345', regex used for validation is '(([A-Za-z0-9][-A-Za-z0-9_.]*)?[A-Za-z0-9])?')")},
			patch: func() *resourceapi.ResourceSlicePatch {
				patch := testResourceSlicePatch(goodName, validPatchSpec)
				patch.Labels = map[string]string{
					"hello-world": badValue,
				}
				return patch
			}(),
		},
		"good-annotations": {
			patch: func() *resourceapi.ResourceSlicePatch {
				patch := testResourceSlicePatch(goodName, validPatchSpec)
				patch.Annotations = map[string]string{
					"foo": "bar",
				}
				return patch
			}(),
		},
		"bad-annotations": {
			wantFailures: field.ErrorList{field.Invalid(field.NewPath("metadata", "annotations"), badName, "name part must consist of alphanumeric characters, '-', '_' or '.', and must start and end with an alphanumeric character (e.g. 'MyName',  or 'my.name',  or '123-abc', regex used for validation is '([A-Za-z0-9][-A-Za-z0-9_.]*)?[A-Za-z0-9]')")},
			patch: func() *resourceapi.ResourceSlicePatch {
				patch := testResourceSlicePatch(goodName, validPatchSpec)
				patch.Annotations = map[string]string{
					badName: "hello world",
				}
				return patch
			}(),
		},
		"bad-class": {
			wantFailures: field.ErrorList{field.Invalid(field.NewPath("spec", "devices", "filter", "deviceClassName"), badName, "a lowercase RFC 1123 subdomain must consist of lower case alphanumeric characters, '-' or '.', and must start and end with an alphanumeric character (e.g. 'example.com', regex used for validation is '[a-z0-9]([-a-z0-9]*[a-z0-9])?(\\.[a-z0-9]([-a-z0-9]*[a-z0-9])?)*')")},
			patch: func() *resourceapi.ResourceSlicePatch {
				patch := testResourceSlicePatch(goodName, validPatchSpec)
				patch.Spec.Devices.Filter.DeviceClassName = ptr.To(badName)
				return patch
			}(),
		},
		"bad-driver": {
			wantFailures: field.ErrorList{field.Invalid(field.NewPath("spec", "devices", "filter", "driver"), badName, "a lowercase RFC 1123 subdomain must consist of lower case alphanumeric characters, '-' or '.', and must start and end with an alphanumeric character (e.g. 'example.com', regex used for validation is '[a-z0-9]([-a-z0-9]*[a-z0-9])?(\\.[a-z0-9]([-a-z0-9]*[a-z0-9])?)*')")},
			patch: func() *resourceapi.ResourceSlicePatch {
				patch := testResourceSlicePatch(goodName, validPatchSpec)
				patch.Spec.Devices.Filter.Driver = ptr.To(badName)
				return patch
			}(),
		},
		"bad-pool": {
			wantFailures: field.ErrorList{field.Invalid(field.NewPath("spec", "devices", "filter", "pool"), badName, "a lowercase RFC 1123 subdomain must consist of lower case alphanumeric characters, '-' or '.', and must start and end with an alphanumeric character (e.g. 'example.com', regex used for validation is '[a-z0-9]([-a-z0-9]*[a-z0-9])?(\\.[a-z0-9]([-a-z0-9]*[a-z0-9])?)*')")},
			patch: func() *resourceapi.ResourceSlicePatch {
				patch := testResourceSlicePatch(goodName, validPatchSpec)
				patch.Spec.Devices.Filter.Pool = ptr.To(badName)
				return patch
			}(),
		},
		"bad-device": {
			wantFailures: field.ErrorList{field.Invalid(field.NewPath("spec", "devices", "filter", "device"), badName, "a lowercase RFC 1123 label must consist of lower case alphanumeric characters or '-', and must start and end with an alphanumeric character (e.g. 'my-name',  or '123-abc', regex used for validation is '[a-z0-9]([-a-z0-9]*[a-z0-9])?')")},
			patch: func() *resourceapi.ResourceSlicePatch {
				patch := testResourceSlicePatch(goodName, validPatchSpec)
				patch.Spec.Devices.Filter.Device = ptr.To(badName)
				return patch
			}(),
		},
		"CEL-compile-errors": {
			wantFailures: field.ErrorList{
				field.Invalid(field.NewPath("spec", "devices", "filter", "selectors").Index(1).Child("cel", "expression"), `device.attributes[true].someBoolean`, "compilation failed: ERROR: <input>:1:18: found no matching overload for '_[_]' applied to '(map(string, map(string, any)), bool)'\n | device.attributes[true].someBoolean\n | .................^"),
			},
			patch: func() *resourceapi.ResourceSlicePatch {
				patch := testResourceSlicePatch(goodName, validPatchSpec)
				patch.Spec.Devices.Filter.Selectors = []resourceapi.DeviceSelector{
					{
						// Good selector.
						CEL: &resourceapi.CELDeviceSelector{
							Expression: `device.driver == "dra.example.com"`,
						},
					},
					{
						// Bad selector.
						CEL: &resourceapi.CELDeviceSelector{
							Expression: `device.attributes[true].someBoolean`,
						},
					},
				}
				return patch
			}(),
		},
		"CEL-length": {
			wantFailures: field.ErrorList{
				field.TooLong(field.NewPath("spec", "devices", "filter", "selectors").Index(1).Child("cel", "expression"), "" /*unused*/, resourceapi.CELSelectorExpressionMaxLength),
			},
			patch: func() *resourceapi.ResourceSlicePatch {
				patch := testResourceSlicePatch(goodName, validPatchSpec)
				expression := `device.driver == ""`
				patch.Spec.Devices.Filter.Selectors = []resourceapi.DeviceSelector{
					{
						// Good selector.
						CEL: &resourceapi.CELDeviceSelector{
							Expression: strings.ReplaceAll(expression, `""`, `"`+strings.Repeat("x", resourceapi.CELSelectorExpressionMaxLength-len(expression))+`"`),
						},
					},
					{
						// Too long by one selector.
						CEL: &resourceapi.CELDeviceSelector{
							Expression: strings.ReplaceAll(expression, `""`, `"`+strings.Repeat("x", resourceapi.CELSelectorExpressionMaxLength-len(expression)+1)+`"`),
						},
					},
				}
				return patch
			}(),
		},
		"CEL-cost": {
			wantFailures: field.ErrorList{
				field.Forbidden(field.NewPath("spec", "devices", "filter", "selectors").Index(0).Child("cel", "expression"), "too complex, exceeds cost limit"),
			},
			patch: func() *resourceapi.ResourceSlicePatch {
				claim := testResourceSlicePatch(goodName, validPatchSpec)
				claim.Spec.Devices.Filter.Selectors = []resourceapi.DeviceSelector{
					{
						CEL: &resourceapi.CELDeviceSelector{
							// From https://github.com/kubernetes/kubernetes/blob/50fc400f178d2078d0ca46aee955ee26375fc437/test/integration/apiserver/cel/validatingadmissionpolicy_test.go#L2150.
							Expression: `[1, 2, 3, 4, 5, 6, 7, 8, 9, 10].all(x, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10].all(y, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10].all(z, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10].all(z2, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10].all(z3, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10].all(z4, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10].all(z5, int('1'.find('[0-9]*')) < 100)))))))`,
						},
					},
				}
				return claim
			}(),
		},
		"bad-attribute-missing-domain": {
			wantFailures: field.ErrorList{
				field.Invalid(field.NewPath("spec", "devices", "attributes").Key(goodName), resourceapi.FullyQualifiedName(goodName), "must include a domain"),
			},
			patch: func() *resourceapi.ResourceSlicePatch {
				patch := testResourceSlicePatch(goodName, validPatchSpec)
				patch.Spec.Devices.Attributes = map[resourceapi.FullyQualifiedName]resourceapi.NullableDeviceAttribute{
					resourceapi.FullyQualifiedName(goodName): {DeviceAttribute: resourceapi.DeviceAttribute{StringValue: ptr.To("x")}},
				}
				return patch
			}(),
		},
		"bad-attribute-zero-values": {
			wantFailures: field.ErrorList{
				field.Required(field.NewPath("spec", "devices", "attributes").Key(goodQualifiedName), "exactly one value must be specified"),
			},
			patch: func() *resourceapi.ResourceSlicePatch {
				patch := testResourceSlicePatch(goodName, validPatchSpec)
				patch.Spec.Devices.Attributes = map[resourceapi.FullyQualifiedName]resourceapi.NullableDeviceAttribute{
					resourceapi.FullyQualifiedName(goodQualifiedName): {},
				}
				return patch
			}(),
		},
		"bad-attribute-two-values": {
			wantFailures: field.ErrorList{
				field.Invalid(
					field.NewPath("spec", "devices", "attributes").Key(goodQualifiedName),
					resourceapi.NullableDeviceAttribute{DeviceAttribute: resourceapi.DeviceAttribute{StringValue: ptr.To("x"), VersionValue: ptr.To("1.2.3")}},
					"exactly one value must be specified",
				),
			},
			patch: func() *resourceapi.ResourceSlicePatch {
				patch := testResourceSlicePatch(goodName, validPatchSpec)
				patch.Spec.Devices.Attributes = map[resourceapi.FullyQualifiedName]resourceapi.NullableDeviceAttribute{
					resourceapi.FullyQualifiedName(goodQualifiedName): {DeviceAttribute: resourceapi.DeviceAttribute{StringValue: ptr.To("x"), VersionValue: ptr.To("1.2.3")}},
				}
				return patch
			}(),
		},
		"bad-attribute-string-too-long": {
			wantFailures: field.ErrorList{
				field.TooLongMaxLength(field.NewPath("spec", "devices", "attributes").Key(goodQualifiedName).Child("string"), strings.Repeat("x", resourceapi.DeviceAttributeMaxValueLength+1), resourceapi.DeviceAttributeMaxValueLength),
			},
			patch: func() *resourceapi.ResourceSlicePatch {
				patch := testResourceSlicePatch(goodName, validPatchSpec)
				patch.Spec.Devices.Attributes = map[resourceapi.FullyQualifiedName]resourceapi.NullableDeviceAttribute{
					resourceapi.FullyQualifiedName(goodQualifiedName): {DeviceAttribute: resourceapi.DeviceAttribute{StringValue: ptr.To(strings.Repeat("x", resourceapi.DeviceAttributeMaxValueLength+1))}},
				}
				return patch
			}(),
		},
		"bad-attribute-version-too-long": {
			wantFailures: field.ErrorList{
				field.Invalid(
					field.NewPath("spec", "devices", "attributes").Key(goodQualifiedName).Child("version"),
					strings.Repeat("x", resourceapi.DeviceAttributeMaxValueLength+1),
					"must be a string compatible with semver.org spec 2.0.0",
				),
				field.TooLongMaxLength(
					field.NewPath("spec", "devices", "attributes").Key(goodQualifiedName).Child("version"),
					strings.Repeat("x", resourceapi.DeviceAttributeMaxValueLength+1),
					resourceapi.DeviceAttributeMaxValueLength,
				),
			},
			patch: func() *resourceapi.ResourceSlicePatch {
				patch := testResourceSlicePatch(goodName, validPatchSpec)
				patch.Spec.Devices.Attributes = map[resourceapi.FullyQualifiedName]resourceapi.NullableDeviceAttribute{
					resourceapi.FullyQualifiedName(goodQualifiedName): {DeviceAttribute: resourceapi.DeviceAttribute{VersionValue: ptr.To(strings.Repeat("x", resourceapi.DeviceAttributeMaxValueLength+1))}},
				}
				return patch
			}(),
		},
		"good-attribute-name-max-length": {
			patch: func() *resourceapi.ResourceSlicePatch {
				patch := testResourceSlicePatch(goodName, validPatchSpec)
				key := resourceapi.FullyQualifiedName(strings.Repeat("x", resourceapi.DeviceMaxDomainLength) + "/" + strings.Repeat("y", resourceapi.DeviceMaxIDLength))
				patch.Spec.Devices.Attributes = map[resourceapi.FullyQualifiedName]resourceapi.NullableDeviceAttribute{
					key: {DeviceAttribute: resourceapi.DeviceAttribute{StringValue: ptr.To("z")}},
				}
				return patch
			}(),
		},
		"bad-attribute-c-identifier": {
			wantFailures: field.ErrorList{
				field.TooLongMaxLength(
					field.NewPath("spec", "devices", "attributes").Key(goodDomain+"/"+strings.Repeat(".", resourceapi.DeviceMaxIDLength+1)),
					strings.Repeat(".", resourceapi.DeviceMaxIDLength+1),
					resourceapi.DeviceMaxIDLength,
				),
				field.TypeInvalid(
					field.NewPath("spec", "devices", "attributes").Key(goodDomain+"/"+strings.Repeat(".", resourceapi.DeviceMaxIDLength+1)),
					strings.Repeat(".", resourceapi.DeviceMaxIDLength+1),
					"a valid C identifier must start with alphabetic character or '_', followed by a string of alphanumeric characters or '_' (e.g. 'my_name',  or 'MY_NAME',  or 'MyName', regex used for validation is '[A-Za-z_][A-Za-z0-9_]*')",
				),
			},
			patch: func() *resourceapi.ResourceSlicePatch {
				patch := testResourceSlicePatch(goodName, validPatchSpec)
				patch.Spec.Devices.Attributes = map[resourceapi.FullyQualifiedName]resourceapi.NullableDeviceAttribute{
					resourceapi.FullyQualifiedName(goodDomain + "/" + strings.Repeat(".", resourceapi.DeviceMaxIDLength+1)): {DeviceAttribute: resourceapi.DeviceAttribute{StringValue: ptr.To("y")}},
				}
				return patch
			}(),
		},
		"bad-attribute-domain": {
			wantFailures: field.ErrorList{
				field.TooLong(
					field.NewPath("spec", "devices", "attributes").Key(strings.Repeat("_", resourceapi.DeviceMaxDomainLength+1)+"/y"),
					strings.Repeat("_", resourceapi.DeviceMaxDomainLength+1),
					resourceapi.DeviceMaxDomainLength,
				),
				field.Invalid(
					field.NewPath("spec", "devices", "attributes").Key(strings.Repeat("_", resourceapi.DeviceMaxDomainLength+1)+"/y"),
					strings.Repeat("_", resourceapi.DeviceMaxDomainLength+1),
					"a lowercase RFC 1123 subdomain must consist of lower case alphanumeric characters, '-' or '.', and must start and end with an alphanumeric character (e.g. 'example.com', regex used for validation is '[a-z0-9]([-a-z0-9]*[a-z0-9])?(\\.[a-z0-9]([-a-z0-9]*[a-z0-9])?)*')",
				),
			},
			patch: func() *resourceapi.ResourceSlicePatch {
				slice := testResourceSlicePatch(goodName, validPatchSpec)
				slice.Spec.Devices.Attributes = map[resourceapi.FullyQualifiedName]resourceapi.NullableDeviceAttribute{
					resourceapi.FullyQualifiedName(strings.Repeat("_", resourceapi.DeviceMaxDomainLength+1) + "/y"): {DeviceAttribute: resourceapi.DeviceAttribute{StringValue: ptr.To("z")}},
				}
				return slice
			}(),
		},
		"bad-key-too-long": {
			wantFailures: field.ErrorList{
				field.TooLong(
					field.NewPath("spec", "devices", "attributes").Key("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx...xxxxxxxxxxxx/yyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy"),
					strings.Repeat("x", resourceapi.DeviceMaxDomainLength+1),
					resourceapi.DeviceMaxDomainLength,
				),
				field.TooLongMaxLength(
					field.NewPath("spec", "devices", "attributes").Key("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx...xxxxxxxxxxxx/yyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy"),
					strings.Repeat("y", resourceapi.DeviceMaxIDLength+1),
					resourceapi.DeviceMaxIDLength,
				),
			},
			patch: func() *resourceapi.ResourceSlicePatch {
				slice := testResourceSlicePatch(goodName, validPatchSpec)
				slice.Spec.Devices.Attributes = map[resourceapi.FullyQualifiedName]resourceapi.NullableDeviceAttribute{
					resourceapi.FullyQualifiedName(strings.Repeat("x", resourceapi.DeviceMaxDomainLength+1) + "/" + strings.Repeat("y", resourceapi.DeviceMaxIDLength+1)): {DeviceAttribute: resourceapi.DeviceAttribute{StringValue: ptr.To("z")}},
				}
				return slice
			}(),
		},
		"bad-attribute-empty-domain-and-c-identifier": {
			wantFailures: field.ErrorList{
				field.Required(field.NewPath("spec", "devices", "attributes").Key("/"), "the domain must not be empty"),
				field.Required(field.NewPath("spec", "devices", "attributes").Key("/"), "the name must not be empty"),
			},
			patch: func() *resourceapi.ResourceSlicePatch {
				patch := testResourceSlicePatch(goodName, validPatchSpec)
				patch.Spec.Devices.Attributes = map[resourceapi.FullyQualifiedName]resourceapi.NullableDeviceAttribute{
					resourceapi.FullyQualifiedName("/"): {DeviceAttribute: resourceapi.DeviceAttribute{StringValue: ptr.To("z")}},
				}
				return patch
			}(),
		},
		"combined-attributes-and-capacity-length-max-attributes": {
			patch: func() *resourceapi.ResourceSlicePatch {
				patch := testResourceSlicePatch(goodName, validPatchSpec)
				patch.Spec.Devices.Attributes = map[resourceapi.FullyQualifiedName]resourceapi.NullableDeviceAttribute{}
				patch.Spec.Devices.Capacity = map[resourceapi.FullyQualifiedName]resourceapi.DeviceCapacity{}
				for i := 0; i < resourceapi.ResourceSliceMaxAttributesAndCapacitiesPerDevice; i++ {
					patch.Spec.Devices.Attributes[resourceapi.FullyQualifiedName(fmt.Sprintf("%s/attr_%d", goodDomain, i))] = resourceapi.NullableDeviceAttribute{DeviceAttribute: resourceapi.DeviceAttribute{StringValue: ptr.To("x")}}
				}
				return patch
			}(),
		},
		"combined-attributes-and-capacity-length-max-capacities": {
			patch: func() *resourceapi.ResourceSlicePatch {
				patch := testResourceSlicePatch(goodName, validPatchSpec)
				patch.Spec.Devices.Attributes = map[resourceapi.FullyQualifiedName]resourceapi.NullableDeviceAttribute{}
				patch.Spec.Devices.Capacity = map[resourceapi.FullyQualifiedName]resourceapi.DeviceCapacity{}
				quantity := resource.MustParse("1Gi")
				capacity := resourceapi.DeviceCapacity{Value: quantity}
				for i := 0; i < resourceapi.ResourceSliceMaxAttributesAndCapacitiesPerDevice; i++ {
					patch.Spec.Devices.Capacity[resourceapi.FullyQualifiedName(fmt.Sprintf("%s/cap_%d", goodDomain, i))] = capacity
				}
				return patch
			}(),
		},
		"combined-attributes-and-capacity-length-too-many": {
			wantFailures: field.ErrorList{
				field.Invalid(
					field.NewPath("spec", "devices"),
					resourceapi.ResourceSliceMaxAttributesAndCapacitiesPerDevice+1,
					fmt.Sprintf("the total number of attributes and capacities must not exceed %d", resourceapi.ResourceSliceMaxAttributesAndCapacitiesPerDevice),
				),
			},
			patch: func() *resourceapi.ResourceSlicePatch {
				patch := testResourceSlicePatch(goodName, validPatchSpec)
				patch.Spec.Devices.Attributes = map[resourceapi.FullyQualifiedName]resourceapi.NullableDeviceAttribute{}
				patch.Spec.Devices.Capacity = map[resourceapi.FullyQualifiedName]resourceapi.DeviceCapacity{}
				for i := 0; i < resourceapi.ResourceSliceMaxAttributesAndCapacitiesPerDevice; i++ {
					key := resourceapi.FullyQualifiedName(fmt.Sprintf("%s/attr_%d", goodDomain, i))
					value := resourceapi.NullableDeviceAttribute{DeviceAttribute: resourceapi.DeviceAttribute{StringValue: ptr.To("x")}}
					patch.Spec.Devices.Attributes[key] = value
				}
				quantity := resource.MustParse("1Gi")
				capacity := resourceapi.DeviceCapacity{Value: quantity}
				// Too large together by one.
				patch.Spec.Devices.Capacity[resourceapi.FullyQualifiedName(fmt.Sprintf("%s/cap", goodDomain))] = capacity
				return patch
			}(),
		},
	}

	for name, scenario := range scenarios {
		t.Run(name, func(t *testing.T) {
			errs := ValidateResourceSlicePatch(scenario.patch)
			assertFailures(t, scenario.wantFailures, errs)
		})
	}
}

func TestValidateResourceSlicePatchUpdate(t *testing.T) {
	name := "valid"
	validPatch := testResourceSlicePatch(name, validPatchSpec)

	scenarios := map[string]struct {
		oldPatch     *resourceapi.ResourceSlicePatch
		update       func(patch *resourceapi.ResourceSlicePatch) *resourceapi.ResourceSlicePatch
		wantFailures field.ErrorList
	}{
		"valid-no-op-update": {
			oldPatch: validPatch,
			update:   func(patch *resourceapi.ResourceSlicePatch) *resourceapi.ResourceSlicePatch { return patch },
		},
		"invalid-name-update": {
			wantFailures: field.ErrorList{field.Invalid(field.NewPath("metadata", "name"), name+"-update", "field is immutable")},
			oldPatch:     validPatch,
			update: func(patch *resourceapi.ResourceSlicePatch) *resourceapi.ResourceSlicePatch {
				patch.Name += "-update"
				return patch
			},
		},
	}

	for name, scenario := range scenarios {
		t.Run(name, func(t *testing.T) {
			scenario.oldPatch.ResourceVersion = "1"
			errs := ValidateResourceSlicePatchUpdate(scenario.update(scenario.oldPatch.DeepCopy()), scenario.oldPatch)
			assertFailures(t, scenario.wantFailures, errs)
		})
	}
}
