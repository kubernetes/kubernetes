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
	"fmt"
	"strings"
	"testing"

	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/kubernetes/pkg/apis/core"
	resourceapi "k8s.io/kubernetes/pkg/apis/resource"
	"k8s.io/utils/ptr"
)

func testAttributes() map[resourceapi.QualifiedName]resourceapi.DeviceAttribute {
	return map[resourceapi.QualifiedName]resourceapi.DeviceAttribute{
		"int":     {IntValue: ptr.To(int64(42))},
		"string":  {StringValue: ptr.To("hello world")},
		"version": {VersionValue: ptr.To("1.2.3")},
		"bool":    {BoolValue: ptr.To(true)},
	}
}

func testCapacity() map[resourceapi.QualifiedName]resourceapi.DeviceCapacity {
	return map[resourceapi.QualifiedName]resourceapi.DeviceCapacity{
		"memory": {Value: resource.MustParse("1Gi")},
	}
}

func testResourceSlice(name, nodeName, driverName string, numDevices int) *resourceapi.ResourceSlice {
	slice := &resourceapi.ResourceSlice{
		ObjectMeta: metav1.ObjectMeta{
			Name: name,
		},
		Spec: resourceapi.ResourceSliceSpec{
			NodeName: nodeName,
			Driver:   driverName,
			Pool: resourceapi.ResourcePool{
				Name:               nodeName,
				ResourceSliceCount: 1,
			},
		},
	}
	for i := 0; i < numDevices; i++ {
		device := resourceapi.Device{
			Name:       fmt.Sprintf("device-%d", i),
			Attributes: testAttributes(),
			Capacity:   testCapacity(),
		}
		slice.Spec.Devices = append(slice.Spec.Devices, device)
	}
	return slice
}

func TestValidateResourceSlice(t *testing.T) {
	goodName := "foo"
	badName := "!@#$%^"
	driverName := "test.example.com"
	now := metav1.Now()
	badValue := "spaces not allowed"

	scenarios := map[string]struct {
		slice        *resourceapi.ResourceSlice
		wantFailures field.ErrorList
	}{
		"good": {
			slice: testResourceSlice(goodName, goodName, driverName, resourceapi.ResourceSliceMaxDevices),
		},
		"too-large": {
			wantFailures: field.ErrorList{field.TooMany(field.NewPath("spec", "devices"), resourceapi.ResourceSliceMaxDevices+1, resourceapi.ResourceSliceMaxDevices)},
			slice:        testResourceSlice(goodName, goodName, goodName, resourceapi.ResourceSliceMaxDevices+1),
		},
		"missing-name": {
			wantFailures: field.ErrorList{field.Required(field.NewPath("metadata", "name"), "name or generateName is required")},
			slice:        testResourceSlice("", goodName, driverName, 1),
		},
		"bad-name": {
			wantFailures: field.ErrorList{field.Invalid(field.NewPath("metadata", "name"), badName, "a lowercase RFC 1123 subdomain must consist of lower case alphanumeric characters, '-' or '.', and must start and end with an alphanumeric character (e.g. 'example.com', regex used for validation is '[a-z0-9]([-a-z0-9]*[a-z0-9])?(\\.[a-z0-9]([-a-z0-9]*[a-z0-9])?)*')")},
			slice:        testResourceSlice(badName, goodName, driverName, 1),
		},
		"generate-name": {
			slice: func() *resourceapi.ResourceSlice {
				slice := testResourceSlice(goodName, goodName, driverName, 1)
				slice.GenerateName = "prefix-"
				return slice
			}(),
		},
		"uid": {
			slice: func() *resourceapi.ResourceSlice {
				slice := testResourceSlice(goodName, goodName, driverName, 1)
				slice.UID = "ac051fac-2ead-46d9-b8b4-4e0fbeb7455d"
				return slice
			}(),
		},
		"resource-version": {
			slice: func() *resourceapi.ResourceSlice {
				slice := testResourceSlice(goodName, goodName, driverName, 1)
				slice.ResourceVersion = "1"
				return slice
			}(),
		},
		"generation": {
			slice: func() *resourceapi.ResourceSlice {
				slice := testResourceSlice(goodName, goodName, driverName, 1)
				slice.Generation = 100
				return slice
			}(),
		},
		"creation-timestamp": {
			slice: func() *resourceapi.ResourceSlice {
				slice := testResourceSlice(goodName, goodName, driverName, 1)
				slice.CreationTimestamp = now
				return slice
			}(),
		},
		"deletion-grace-period-seconds": {
			slice: func() *resourceapi.ResourceSlice {
				slice := testResourceSlice(goodName, goodName, driverName, 1)
				slice.DeletionGracePeriodSeconds = ptr.To[int64](10)
				return slice
			}(),
		},
		"owner-references": {
			slice: func() *resourceapi.ResourceSlice {
				slice := testResourceSlice(goodName, goodName, driverName, 1)
				slice.OwnerReferences = []metav1.OwnerReference{
					{
						APIVersion: "v1",
						Kind:       "pod",
						Name:       "foo",
						UID:        "ac051fac-2ead-46d9-b8b4-4e0fbeb7455d",
					},
				}
				return slice
			}(),
		},
		"finalizers": {
			slice: func() *resourceapi.ResourceSlice {
				slice := testResourceSlice(goodName, goodName, driverName, 1)
				slice.Finalizers = []string{
					"example.com/foo",
				}
				return slice
			}(),
		},
		"managed-fields": {
			slice: func() *resourceapi.ResourceSlice {
				slice := testResourceSlice(goodName, goodName, driverName, 1)
				slice.ManagedFields = []metav1.ManagedFieldsEntry{
					{
						FieldsType: "FieldsV1",
						Operation:  "Apply",
						APIVersion: "apps/v1",
						Manager:    "foo",
					},
				}
				return slice
			}(),
		},
		"good-labels": {
			slice: func() *resourceapi.ResourceSlice {
				slice := testResourceSlice(goodName, goodName, driverName, 1)
				slice.Labels = map[string]string{
					"apps.kubernetes.io/name": "test",
				}
				return slice
			}(),
		},
		"bad-labels": {
			wantFailures: field.ErrorList{field.Invalid(field.NewPath("metadata", "labels"), badValue, "a valid label must be an empty string or consist of alphanumeric characters, '-', '_' or '.', and must start and end with an alphanumeric character (e.g. 'MyValue',  or 'my_value',  or '12345', regex used for validation is '(([A-Za-z0-9][-A-Za-z0-9_.]*)?[A-Za-z0-9])?')")},
			slice: func() *resourceapi.ResourceSlice {
				slice := testResourceSlice(goodName, goodName, driverName, 1)
				slice.Labels = map[string]string{
					"hello-world": badValue,
				}
				return slice
			}(),
		},
		"good-annotations": {
			slice: func() *resourceapi.ResourceSlice {
				slice := testResourceSlice(goodName, goodName, driverName, 1)
				slice.Annotations = map[string]string{
					"foo": "bar",
				}
				return slice
			}(),
		},
		"bad-annotations": {
			wantFailures: field.ErrorList{field.Invalid(field.NewPath("metadata", "annotations"), badName, "name part must consist of alphanumeric characters, '-', '_' or '.', and must start and end with an alphanumeric character (e.g. 'MyName',  or 'my.name',  or '123-abc', regex used for validation is '([A-Za-z0-9][-A-Za-z0-9_.]*)?[A-Za-z0-9]')")},
			slice: func() *resourceapi.ResourceSlice {
				slice := testResourceSlice(goodName, goodName, driverName, 1)
				slice.Annotations = map[string]string{
					badName: "hello world",
				}
				return slice
			}(),
		},
		"bad-nodename": {
			wantFailures: field.ErrorList{
				field.Invalid(field.NewPath("spec", "pool", "name"), badName, "a lowercase RFC 1123 subdomain must consist of lower case alphanumeric characters, '-' or '.', and must start and end with an alphanumeric character (e.g. 'example.com', regex used for validation is '[a-z0-9]([-a-z0-9]*[a-z0-9])?(\\.[a-z0-9]([-a-z0-9]*[a-z0-9])?)*')"),
				field.Invalid(field.NewPath("spec", "nodeName"), badName, "a lowercase RFC 1123 subdomain must consist of lower case alphanumeric characters, '-' or '.', and must start and end with an alphanumeric character (e.g. 'example.com', regex used for validation is '[a-z0-9]([-a-z0-9]*[a-z0-9])?(\\.[a-z0-9]([-a-z0-9]*[a-z0-9])?)*')"),
			},
			slice: testResourceSlice(goodName, badName, driverName, 1),
		},
		"bad-multi-pool-name": {
			wantFailures: field.ErrorList{
				field.Invalid(field.NewPath("spec", "pool", "name"), badName, "a lowercase RFC 1123 subdomain must consist of lower case alphanumeric characters, '-' or '.', and must start and end with an alphanumeric character (e.g. 'example.com', regex used for validation is '[a-z0-9]([-a-z0-9]*[a-z0-9])?(\\.[a-z0-9]([-a-z0-9]*[a-z0-9])?)*')"),
				field.Invalid(field.NewPath("spec", "pool", "name"), badName, "a lowercase RFC 1123 subdomain must consist of lower case alphanumeric characters, '-' or '.', and must start and end with an alphanumeric character (e.g. 'example.com', regex used for validation is '[a-z0-9]([-a-z0-9]*[a-z0-9])?(\\.[a-z0-9]([-a-z0-9]*[a-z0-9])?)*')"),
				field.Invalid(field.NewPath("spec", "nodeName"), badName+"/"+badName, "a lowercase RFC 1123 subdomain must consist of lower case alphanumeric characters, '-' or '.', and must start and end with an alphanumeric character (e.g. 'example.com', regex used for validation is '[a-z0-9]([-a-z0-9]*[a-z0-9])?(\\.[a-z0-9]([-a-z0-9]*[a-z0-9])?)*')"),
			},
			slice: testResourceSlice(goodName, badName+"/"+badName, driverName, 1),
		},
		"good-pool-name": {
			slice: func() *resourceapi.ResourceSlice {
				slice := testResourceSlice(goodName, goodName, driverName, 1)
				slice.Spec.Pool.Name = strings.Repeat("x", resourceapi.PoolNameMaxLength)
				return slice
			}(),
		},
		"bad-pool": {
			wantFailures: field.ErrorList{
				field.TooLongMaxLength(field.NewPath("spec", "pool", "name"), strings.Repeat("x/", resourceapi.PoolNameMaxLength/2)+"xy", resourceapi.PoolNameMaxLength),
				field.Invalid(field.NewPath("spec", "pool", "resourceSliceCount"), int64(0), "must be greater than zero"),
				field.Invalid(field.NewPath("spec", "pool", "generation"), int64(-1), "must be greater than or equal to zero"),
			},
			slice: func() *resourceapi.ResourceSlice {
				slice := testResourceSlice(goodName, goodName, driverName, 1)
				slice.Spec.Pool.Name = strings.Repeat("x/", resourceapi.PoolNameMaxLength/2) + "xy"
				slice.Spec.Pool.ResourceSliceCount = 0
				slice.Spec.Pool.Generation = -1
				return slice
			}(),
		},
		"missing-pool-name": {
			wantFailures: field.ErrorList{
				field.Required(field.NewPath("spec", "pool", "name"), ""),
			},
			slice: func() *resourceapi.ResourceSlice {
				slice := testResourceSlice(goodName, goodName, driverName, 1)
				slice.Spec.Pool.Name = ""
				return slice
			}(),
		},
		"bad-empty-node-selector": {
			wantFailures: field.ErrorList{
				field.Required(field.NewPath("spec", "nodeSelector", "nodeSelectorTerms"), "must have at least one node selector term"),                             // From core validation.
				field.Invalid(field.NewPath("spec", "nodeSelector", "nodeSelectorTerms"), []core.NodeSelectorTerm(nil), "must have exactly one node selector term"), // From DRA validation.
			},
			slice: func() *resourceapi.ResourceSlice {
				slice := testResourceSlice(goodName, goodName, driverName, 1)
				slice.Spec.NodeName = ""
				slice.Spec.NodeSelector = &core.NodeSelector{}
				return slice
			}(),
		},
		"bad-node-selection": {
			wantFailures: field.ErrorList{field.Invalid(field.NewPath("spec"), nil, "exactly one of `nodeName`, `nodeSelector`, `allNodes`, or `perDeviceNodeSelection` is required")},
			slice: func() *resourceapi.ResourceSlice {
				slice := testResourceSlice(goodName, goodName, driverName, 1)
				slice.Spec.NodeName = "worker"
				slice.Spec.NodeSelector = &core.NodeSelector{
					NodeSelectorTerms: []core.NodeSelectorTerm{{MatchFields: []core.NodeSelectorRequirement{{Key: "metadata.name", Operator: core.NodeSelectorOpIn, Values: []string{"worker"}}}}},
				}
				return slice
			}(),
		},
		"bad-node-selection-all-nodes": {
			wantFailures: field.ErrorList{field.Invalid(field.NewPath("spec"), nil, "exactly one of `nodeName`, `nodeSelector`, `allNodes`, or `perDeviceNodeSelection` is required")},
			slice: func() *resourceapi.ResourceSlice {
				slice := testResourceSlice(goodName, goodName, driverName, 1)
				slice.Spec.NodeName = "worker"
				slice.Spec.AllNodes = true
				return slice
			}(),
		},
		"empty-node-selection": {
			wantFailures: field.ErrorList{field.Required(field.NewPath("spec"), "exactly one of `nodeName`, `nodeSelector`, `allNodes`, or `perDeviceNodeSelection` is required")},
			slice: func() *resourceapi.ResourceSlice {
				slice := testResourceSlice(goodName, goodName, driverName, 1)
				slice.Spec.NodeName = ""
				return slice
			}(),
		},
		"bad-drivername": {
			wantFailures: field.ErrorList{field.Invalid(field.NewPath("spec", "driver"), badName, "a lowercase RFC 1123 subdomain must consist of lower case alphanumeric characters, '-' or '.', and must start and end with an alphanumeric character (e.g. 'example.com', regex used for validation is '[a-z0-9]([-a-z0-9]*[a-z0-9])?(\\.[a-z0-9]([-a-z0-9]*[a-z0-9])?)*')")},
			slice:        testResourceSlice(goodName, goodName, badName, 1),
		},
		"bad-devices": {
			wantFailures: field.ErrorList{
				field.Invalid(field.NewPath("spec", "devices").Index(1).Child("name"), badName, "a lowercase RFC 1123 label must consist of lower case alphanumeric characters or '-', and must start and end with an alphanumeric character (e.g. 'my-name',  or '123-abc', regex used for validation is '[a-z0-9]([-a-z0-9]*[a-z0-9])?')"),
			},
			slice: func() *resourceapi.ResourceSlice {
				slice := testResourceSlice(goodName, goodName, goodName, 3)
				slice.Spec.Devices[1].Name = badName
				return slice
			}(),
		},
		"combined-attributes-and-capacity-length": {
			wantFailures: field.ErrorList{
				field.Invalid(field.NewPath("spec", "devices").Index(2), resourceapi.ResourceSliceMaxAttributesAndCapacitiesPerDevice+1, fmt.Sprintf("the total number of attributes and capacities must not exceed %d", resourceapi.ResourceSliceMaxAttributesAndCapacitiesPerDevice)),
			},
			slice: func() *resourceapi.ResourceSlice {
				slice := testResourceSlice(goodName, goodName, goodName, 3)
				slice.Spec.Devices[0].Attributes = map[resourceapi.QualifiedName]resourceapi.DeviceAttribute{}
				slice.Spec.Devices[0].Capacity = map[resourceapi.QualifiedName]resourceapi.DeviceCapacity{}
				for i := 0; i < resourceapi.ResourceSliceMaxAttributesAndCapacitiesPerDevice; i++ {
					slice.Spec.Devices[0].Attributes[resourceapi.QualifiedName(fmt.Sprintf("attr_%d", i))] = resourceapi.DeviceAttribute{StringValue: ptr.To("x")}
				}
				slice.Spec.Devices[1].Attributes = map[resourceapi.QualifiedName]resourceapi.DeviceAttribute{}
				slice.Spec.Devices[1].Capacity = map[resourceapi.QualifiedName]resourceapi.DeviceCapacity{}
				quantity := resource.MustParse("1Gi")
				capacity := resourceapi.DeviceCapacity{Value: quantity}
				for i := 0; i < resourceapi.ResourceSliceMaxAttributesAndCapacitiesPerDevice; i++ {
					slice.Spec.Devices[1].Capacity[resourceapi.QualifiedName(fmt.Sprintf("cap_%d", i))] = capacity
				}
				// Too large together by one.
				slice.Spec.Devices[2].Attributes = slice.Spec.Devices[0].Attributes
				slice.Spec.Devices[2].Capacity = map[resourceapi.QualifiedName]resourceapi.DeviceCapacity{
					"cap": capacity,
				}
				return slice
			}(),
		},
		"invalid-node-selector-label-value": {
			wantFailures: field.ErrorList{field.Invalid(field.NewPath("spec", "nodeSelector", "nodeSelectorTerms").Index(0).Child("matchExpressions").Index(0).Child("values").Index(0), "-1", "a valid label must be an empty string or consist of alphanumeric characters, '-', '_' or '.', and must start and end with an alphanumeric character (e.g. 'MyValue',  or 'my_value',  or '12345', regex used for validation is '(([A-Za-z0-9][-A-Za-z0-9_.]*)?[A-Za-z0-9])?')")},
			slice: func() *resourceapi.ResourceSlice {
				slice := testResourceSlice(goodName, goodName, goodName, 3)
				slice.Spec.NodeName = ""
				slice.Spec.NodeSelector = &core.NodeSelector{
					NodeSelectorTerms: []core.NodeSelectorTerm{{
						MatchExpressions: []core.NodeSelectorRequirement{{
							Key:      "foo",
							Operator: core.NodeSelectorOpIn,
							Values:   []string{"-1"},
						}},
					}},
				}
				return slice
			}(),
		},
		"too-many-mixins": {
			wantFailures: field.ErrorList{field.Invalid(field.NewPath("spec", "mixins"), 129, "the total number `device`, `deviceCapacityConsumption`, and `capacityPool` mixins must not exceed 128")},
			slice: func() *resourceapi.ResourceSlice {
				slice := testResourceSlice(goodName, goodName, goodName, 1)
				slice.Spec.Mixins = &resourceapi.ResourceSliceMixins{
					Device:                    make([]resourceapi.DeviceMixin, 0),
					DeviceCapacityConsumption: make([]resourceapi.DeviceCapacityConsumptionMixin, 0),
					CapacityPool:              make([]resourceapi.CapacityPoolMixin, 0),
				}
				for i := 0; i < (resourceapi.ResourceSliceMaxMixins/3)+1; i++ {
					slice.Spec.Mixins.Device = append(slice.Spec.Mixins.Device, resourceapi.DeviceMixin{
						Name: fmt.Sprintf("device-mixin-%d", i),
						Capacity: map[resourceapi.QualifiedName]resourceapi.DeviceCapacity{
							resourceapi.QualifiedName("memory"): {
								Value: resource.MustParse("50Gi"),
							},
						},
					})
				}
				for i := 0; i < (resourceapi.ResourceSliceMaxMixins/3)+1; i++ {
					slice.Spec.Mixins.DeviceCapacityConsumption = append(slice.Spec.Mixins.DeviceCapacityConsumption, resourceapi.DeviceCapacityConsumptionMixin{
						Name: fmt.Sprintf("device-capacity-consumption-mixin-%d", i),
						Capacity: map[resourceapi.QualifiedName]resourceapi.DeviceCapacity{
							resourceapi.QualifiedName("memory"): {
								Value: resource.MustParse("50Gi"),
							},
						},
					})
				}
				for i := 0; i < (resourceapi.ResourceSliceMaxMixins/3)+1; i++ {
					slice.Spec.Mixins.CapacityPool = append(slice.Spec.Mixins.CapacityPool, resourceapi.CapacityPoolMixin{
						Name: fmt.Sprintf("capacity-pool-mixin-%d", i),
						Capacity: map[resourceapi.QualifiedName]resourceapi.DeviceCapacity{
							resourceapi.QualifiedName("memory"): {
								Value: resource.MustParse("50Gi"),
							},
						},
					})
				}
				return slice
			}(),
		},
		"invalid-device-mixin-name": {
			wantFailures: field.ErrorList{
				field.Invalid(field.NewPath("spec", "mixins", "device").Index(0).Child("name"), badName, "a lowercase RFC 1123 label must consist of lower case alphanumeric characters or '-', and must start and end with an alphanumeric character (e.g. 'my-name',  or '123-abc', regex used for validation is '[a-z0-9]([-a-z0-9]*[a-z0-9])?')"),
			},
			slice: func() *resourceapi.ResourceSlice {
				slice := testResourceSlice(goodName, goodName, goodName, 1)
				slice.Spec.Mixins = &resourceapi.ResourceSliceMixins{
					Device: []resourceapi.DeviceMixin{
						{
							Name: badName,
							Capacity: map[resourceapi.QualifiedName]resourceapi.DeviceCapacity{
								resourceapi.QualifiedName("memory"): {
									Value: resource.MustParse("80Gi"),
								},
							},
						},
					},
				}
				return slice
			}(),
		},
		"invalid-device-no-capacity-or-attributes": {
			wantFailures: field.ErrorList{
				field.Required(field.NewPath("spec", "mixins", "device").Index(0), "`attributes` and `capacity` can not both be empty"),
			},
			slice: func() *resourceapi.ResourceSlice {
				slice := testResourceSlice(goodName, goodName, goodName, 1)
				slice.Spec.Mixins = &resourceapi.ResourceSliceMixins{
					Device: []resourceapi.DeviceMixin{
						{
							Name: goodName,
						},
					},
				}
				return slice
			}(),
		},
		"device-mixin-combined-attributes-and-capacity-length": {
			wantFailures: field.ErrorList{
				field.Invalid(field.NewPath("spec", "mixins", "device").Index(2), resourceapi.ResourceSliceMaxAttributesAndCapacitiesPerDevice+1, fmt.Sprintf("the total number of attributes and capacities must not exceed %d", resourceapi.ResourceSliceMaxAttributesAndCapacitiesPerDevice)),
			},
			slice: func() *resourceapi.ResourceSlice {
				slice := testResourceSlice(goodName, goodName, goodName, 1)
				slice.Spec.Mixins = &resourceapi.ResourceSliceMixins{
					Device: []resourceapi.DeviceMixin{
						{
							Name: "device-mixin-1",
						},
						{
							Name: "device-mixin-2",
						},
						{
							Name: "device-mixin-3",
						},
					},
				}

				slice.Spec.Mixins.Device[0].Attributes = map[resourceapi.QualifiedName]resourceapi.DeviceAttribute{}
				slice.Spec.Mixins.Device[0].Capacity = map[resourceapi.QualifiedName]resourceapi.DeviceCapacity{}
				for i := 0; i < resourceapi.ResourceSliceMaxAttributesAndCapacitiesPerDevice; i++ {
					slice.Spec.Mixins.Device[0].Attributes[resourceapi.QualifiedName(fmt.Sprintf("attr_%d", i))] = resourceapi.DeviceAttribute{StringValue: ptr.To("x")}
				}
				slice.Spec.Mixins.Device[1].Attributes = map[resourceapi.QualifiedName]resourceapi.DeviceAttribute{}
				slice.Spec.Mixins.Device[1].Capacity = map[resourceapi.QualifiedName]resourceapi.DeviceCapacity{}
				quantity := resource.MustParse("1Gi")
				capacity := resourceapi.DeviceCapacity{Value: quantity}
				for i := 0; i < resourceapi.ResourceSliceMaxAttributesAndCapacitiesPerDevice; i++ {
					slice.Spec.Mixins.Device[1].Capacity[resourceapi.QualifiedName(fmt.Sprintf("cap_%d", i))] = capacity
				}
				// Too large together by one.
				slice.Spec.Mixins.Device[2].Attributes = slice.Spec.Mixins.Device[0].Attributes
				slice.Spec.Mixins.Device[2].Capacity = map[resourceapi.QualifiedName]resourceapi.DeviceCapacity{
					"cap": capacity,
				}
				return slice
			}(),
		},
		"invalid-device-capacity-consumption-mixin-name": {
			wantFailures: field.ErrorList{
				field.Invalid(field.NewPath("spec", "mixins", "deviceCapacityConsumption").Index(0).Child("name"), badName, "a lowercase RFC 1123 label must consist of lower case alphanumeric characters or '-', and must start and end with an alphanumeric character (e.g. 'my-name',  or '123-abc', regex used for validation is '[a-z0-9]([-a-z0-9]*[a-z0-9])?')"),
			},
			slice: func() *resourceapi.ResourceSlice {
				slice := testResourceSlice(goodName, goodName, goodName, 1)
				slice.Spec.Mixins = &resourceapi.ResourceSliceMixins{
					DeviceCapacityConsumption: []resourceapi.DeviceCapacityConsumptionMixin{
						{
							Name: badName,
							Capacity: map[resourceapi.QualifiedName]resourceapi.DeviceCapacity{
								resourceapi.QualifiedName("memory"): {
									Value: resource.MustParse("40Gi"),
								},
							},
						},
					},
				}
				return slice
			}(),
		},
		"device-capacity-consumption-mixin-without-capacity": {
			wantFailures: field.ErrorList{
				field.Required(field.NewPath("spec", "mixins", "deviceCapacityConsumption").Index(0).Child("capacity"), "`capacity` can not be empty"),
			},
			slice: func() *resourceapi.ResourceSlice {
				slice := testResourceSlice(goodName, goodName, goodName, 1)
				slice.Spec.Mixins = &resourceapi.ResourceSliceMixins{
					DeviceCapacityConsumption: []resourceapi.DeviceCapacityConsumptionMixin{
						{
							Name: goodName,
						},
					},
				}
				return slice
			}(),
		},
		"device-capacity-consumption-mixin-too-many-capacity": {
			wantFailures: field.ErrorList{
				field.TooMany(field.NewPath("spec", "mixins", "deviceCapacityConsumption").Index(0).Child("capacity"), resourceapi.ResourceSliceMaxAttributesAndCapacities+1, resourceapi.ResourceSliceMaxAttributesAndCapacities),
			},
			slice: func() *resourceapi.ResourceSlice {
				slice := testResourceSlice(goodName, goodName, goodName, 1)
				slice.Spec.Mixins = &resourceapi.ResourceSliceMixins{
					DeviceCapacityConsumption: []resourceapi.DeviceCapacityConsumptionMixin{
						{
							Name:     goodName,
							Capacity: make(map[resourceapi.QualifiedName]resourceapi.DeviceCapacity),
						},
					},
				}
				for i := 0; i < resourceapi.ResourceSliceMaxAttributesAndCapacities+1; i++ {
					key := fmt.Sprintf("key%d", i)
					slice.Spec.Mixins.DeviceCapacityConsumption[0].Capacity[resourceapi.QualifiedName(key)] = resourceapi.DeviceCapacity{
						Value: resource.MustParse("40Gi"),
					}
				}
				return slice
			}(),
		},
		"invalid-capacity-pool-mixin-name": {
			wantFailures: field.ErrorList{
				field.Invalid(field.NewPath("spec", "mixins", "capacityPool").Index(0).Child("name"), badName, "a lowercase RFC 1123 label must consist of lower case alphanumeric characters or '-', and must start and end with an alphanumeric character (e.g. 'my-name',  or '123-abc', regex used for validation is '[a-z0-9]([-a-z0-9]*[a-z0-9])?')"),
			},
			slice: func() *resourceapi.ResourceSlice {
				slice := testResourceSlice(goodName, goodName, goodName, 1)
				slice.Spec.Mixins = &resourceapi.ResourceSliceMixins{
					CapacityPool: []resourceapi.CapacityPoolMixin{
						{
							Name: badName,
							Capacity: map[resourceapi.QualifiedName]resourceapi.DeviceCapacity{
								resourceapi.QualifiedName("memory"): {
									Value: resource.MustParse("40Gi"),
								},
							},
						},
					},
				}
				return slice
			}(),
		},
		"capacity-pool-mixin-without-capacity": {
			wantFailures: field.ErrorList{
				field.Required(field.NewPath("spec", "mixins", "capacityPool").Index(0).Child("capacity"), "`capacity` can not be empty"),
			},
			slice: func() *resourceapi.ResourceSlice {
				slice := testResourceSlice(goodName, goodName, goodName, 1)
				slice.Spec.Mixins = &resourceapi.ResourceSliceMixins{
					CapacityPool: []resourceapi.CapacityPoolMixin{
						{
							Name: goodName,
						},
					},
				}
				return slice
			}(),
		},
		"capacity-pool-mixin-too-many-capacity": {
			wantFailures: field.ErrorList{
				field.TooMany(field.NewPath("spec", "mixins", "capacityPool").Index(0).Child("capacity"), resourceapi.ResourceSliceMaxAttributesAndCapacities+1, resourceapi.ResourceSliceMaxAttributesAndCapacities),
			},
			slice: func() *resourceapi.ResourceSlice {
				slice := testResourceSlice(goodName, goodName, goodName, 1)
				slice.Spec.Mixins = &resourceapi.ResourceSliceMixins{
					CapacityPool: []resourceapi.CapacityPoolMixin{
						{
							Name:     goodName,
							Capacity: make(map[resourceapi.QualifiedName]resourceapi.DeviceCapacity),
						},
					},
				}
				for i := 0; i < resourceapi.ResourceSliceMaxAttributesAndCapacities+1; i++ {
					key := fmt.Sprintf("key%d", i)
					slice.Spec.Mixins.CapacityPool[0].Capacity[resourceapi.QualifiedName(key)] = resourceapi.DeviceCapacity{
						Value: resource.MustParse("40Gi"),
					}
				}
				return slice
			}(),
		},
		"invalid-capacity-pool-name": {
			wantFailures: field.ErrorList{
				field.Invalid(field.NewPath("spec", "capacityPools").Index(0).Child("name"), badName, "a lowercase RFC 1123 label must consist of lower case alphanumeric characters or '-', and must start and end with an alphanumeric character (e.g. 'my-name',  or '123-abc', regex used for validation is '[a-z0-9]([-a-z0-9]*[a-z0-9])?')"),
			},
			slice: func() *resourceapi.ResourceSlice {
				slice := testResourceSlice(goodName, goodName, goodName, 1)
				slice.Spec.CapacityPools = []resourceapi.CapacityPool{
					{
						Name: badName,
						Capacity: map[resourceapi.QualifiedName]resourceapi.DeviceCapacity{
							resourceapi.QualifiedName("memory"): {
								Value: resource.MustParse("40Gi"),
							},
						},
					},
				}
				return slice
			}(),
		},
		"invalid-capacity-pool-include-ref": {
			wantFailures: field.ErrorList{
				field.Invalid(field.NewPath("spec", "capacityPools").Index(0).Child("includes").Index(0), "does-not-exist", "must reference a capacity pool mixin defined in the ResourceSlice"),
			},
			slice: func() *resourceapi.ResourceSlice {
				slice := testResourceSlice(goodName, goodName, goodName, 1)
				slice.Spec.CapacityPools = []resourceapi.CapacityPool{
					{
						Name: goodName,
						Includes: []resourceapi.CapacityPoolMixinRef{
							{
								Name: "does-not-exist",
							},
						},
					},
				}
				return slice
			}(),
		},
		"too-many-capacity-pool-include-refs": {
			wantFailures: field.ErrorList{
				field.TooMany(field.NewPath("spec", "capacityPools").Index(0).Child("includes"),
					resourceapi.ResourceSliceMaxCapacityPoolMixinRefs+1, resourceapi.ResourceSliceMaxCapacityPoolMixinRefs),
			},
			slice: func() *resourceapi.ResourceSlice {
				slice := testResourceSlice(goodName, goodName, goodName, 1)
				slice.Spec.Mixins = &resourceapi.ResourceSliceMixins{
					CapacityPool: make([]resourceapi.CapacityPoolMixin, 0),
				}
				slice.Spec.CapacityPools = []resourceapi.CapacityPool{
					{
						Name:     goodName,
						Includes: make([]resourceapi.CapacityPoolMixinRef, 0),
					},
				}
				for i := 0; i < resourceapi.ResourceSliceMaxCapacityPoolMixinRefs+1; i++ {
					name := fmt.Sprintf("capacity-pool-mixin-%d", i)
					slice.Spec.Mixins.CapacityPool = append(slice.Spec.Mixins.CapacityPool, resourceapi.CapacityPoolMixin{
						Name: name,
						Capacity: map[resourceapi.QualifiedName]resourceapi.DeviceCapacity{
							resourceapi.QualifiedName("memory"): {
								Value: resource.MustParse("50Gi"),
							},
						},
					})
					slice.Spec.CapacityPools[0].Includes = append(slice.Spec.CapacityPools[0].Includes, resourceapi.CapacityPoolMixinRef{
						Name: name,
					})
				}
				return slice
			}(),
		},
		"valid-capacity-pool-include-ref": {
			slice: func() *resourceapi.ResourceSlice {
				slice := testResourceSlice(goodName, goodName, goodName, 1)
				slice.Spec.Mixins = &resourceapi.ResourceSliceMixins{
					CapacityPool: []resourceapi.CapacityPoolMixin{
						{
							Name: "capacity-pool-mixin",
							Capacity: map[resourceapi.QualifiedName]resourceapi.DeviceCapacity{
								resourceapi.QualifiedName("memory"): {
									Value: resource.MustParse("50Gi"),
								},
							},
						},
					},
				}
				slice.Spec.CapacityPools = []resourceapi.CapacityPool{
					{
						Name: goodName,
						Includes: []resourceapi.CapacityPoolMixinRef{
							{
								Name: "capacity-pool-mixin",
							},
						},
					},
				}
				return slice
			}(),
		},
		"capacity-pool-too-many-capacity": {
			wantFailures: field.ErrorList{
				field.TooMany(field.NewPath("spec", "capacityPools").Index(0).Child("capacity"), resourceapi.ResourceSliceMaxAttributesAndCapacities+1, resourceapi.ResourceSliceMaxAttributesAndCapacities),
			},
			slice: func() *resourceapi.ResourceSlice {
				slice := testResourceSlice(goodName, goodName, goodName, 1)
				slice.Spec.CapacityPools = []resourceapi.CapacityPool{
					{
						Name:     goodName,
						Capacity: make(map[resourceapi.QualifiedName]resourceapi.DeviceCapacity),
					},
				}
				for i := 0; i < resourceapi.ResourceSliceMaxAttributesAndCapacities+1; i++ {
					key := fmt.Sprintf("key%d", i)
					slice.Spec.CapacityPools[0].Capacity[resourceapi.QualifiedName(key)] = resourceapi.DeviceCapacity{
						Value: resource.MustParse("40Gi"),
					}
				}
				return slice
			}(),
		},
		"invalid-device-include-reference": {
			wantFailures: field.ErrorList{
				field.Invalid(field.NewPath("spec", "devices").Index(0).Child("includes").Index(0), "does-not-exist", "must reference a device mixin defined in the ResourceSlice"),
			},
			slice: func() *resourceapi.ResourceSlice {
				slice := testResourceSlice(goodName, goodName, goodName, 1)
				slice.Spec.Devices[0].Includes = []resourceapi.DeviceMixinRef{
					{
						Name: "does-not-exist",
					},
				}
				return slice
			}(),
		},
		"too-many-device-include-refs": {
			wantFailures: field.ErrorList{
				field.TooMany(field.NewPath("spec", "devices").Index(0).Child("includes"),
					resourceapi.ResourceSliceMaxDeviceMixinRefs+1, resourceapi.ResourceSliceMaxDeviceMixinRefs),
			},
			slice: func() *resourceapi.ResourceSlice {
				slice := testResourceSlice(goodName, goodName, goodName, 1)
				slice.Spec.Mixins = &resourceapi.ResourceSliceMixins{
					Device: make([]resourceapi.DeviceMixin, 0),
				}
				for i := 0; i < resourceapi.ResourceSliceMaxCapacityPoolMixinRefs+1; i++ {
					name := fmt.Sprintf("device-mixin-%d", i)
					slice.Spec.Mixins.Device = append(slice.Spec.Mixins.Device, resourceapi.DeviceMixin{
						Name: name,
						Capacity: map[resourceapi.QualifiedName]resourceapi.DeviceCapacity{
							resourceapi.QualifiedName("memory"): {
								Value: resource.MustParse("50Gi"),
							},
						},
					})
					slice.Spec.Devices[0].Includes = append(slice.Spec.Devices[0].Includes, resourceapi.DeviceMixinRef{
						Name: name,
					})
				}
				return slice
			}(),
		},
		"valid-device-include-reference": {
			slice: func() *resourceapi.ResourceSlice {
				slice := testResourceSlice(goodName, goodName, goodName, 1)
				slice.Spec.Mixins = &resourceapi.ResourceSliceMixins{
					Device: []resourceapi.DeviceMixin{
						{
							Name: "device-mixin",
							Capacity: map[resourceapi.QualifiedName]resourceapi.DeviceCapacity{
								resourceapi.QualifiedName("memory"): {
									Value: resource.MustParse("50Gi"),
								},
							},
						},
					},
				}
				slice.Spec.Devices[0].Includes = []resourceapi.DeviceMixinRef{
					{
						Name: "device-mixin",
					},
				}
				return slice
			}(),
		},
		"valid-device-capacity-consumption": {
			slice: func() *resourceapi.ResourceSlice {
				slice := testResourceSlice(goodName, goodName, goodName, 1)
				slice.Spec.CapacityPools = []resourceapi.CapacityPool{
					{
						Name: "capacity-pool",
						Capacity: map[resourceapi.QualifiedName]resourceapi.DeviceCapacity{
							resourceapi.QualifiedName("memory"): {
								Value: resource.MustParse("50Gi"),
							},
						},
					},
				}
				slice.Spec.Mixins = &resourceapi.ResourceSliceMixins{
					DeviceCapacityConsumption: []resourceapi.DeviceCapacityConsumptionMixin{
						{
							Name: "device-capacity-consumption-mixin",
							Capacity: map[resourceapi.QualifiedName]resourceapi.DeviceCapacity{
								resourceapi.QualifiedName("memory"): {
									Value: resource.MustParse("50Gi"),
								},
							},
						},
					},
				}
				slice.Spec.Devices[0].ConsumesCapacity = []resourceapi.DeviceCapacityConsumption{
					{
						CapacityPool: "capacity-pool",
						Includes: []resourceapi.DeviceCapacityConsumptionMixinRef{
							{
								Name: "device-capacity-consumption-mixin",
							},
						},
					},
				}
				return slice
			}(),
		},
		"duplicate-capacity-pool-in-device-capacity-consumption": {
			wantFailures: field.ErrorList{
				field.Duplicate(field.NewPath("spec", "devices").Index(0).Child("consumesCapacity").Index(1).Child("capacityPool"), "capacity-pool"),
			},
			slice: func() *resourceapi.ResourceSlice {
				slice := testResourceSlice(goodName, goodName, goodName, 1)
				slice.Spec.CapacityPools = []resourceapi.CapacityPool{
					{
						Name: "capacity-pool",
						Capacity: map[resourceapi.QualifiedName]resourceapi.DeviceCapacity{
							resourceapi.QualifiedName("memory"): {
								Value: resource.MustParse("50Gi"),
							},
						},
					},
				}
				slice.Spec.Devices[0].ConsumesCapacity = []resourceapi.DeviceCapacityConsumption{
					{
						CapacityPool: "capacity-pool",
						Capacity: map[resourceapi.QualifiedName]resourceapi.DeviceCapacity{
							resourceapi.QualifiedName("memory"): {
								Value: resource.MustParse("50Gi"),
							},
						},
					},
					{
						CapacityPool: "capacity-pool",
						Capacity: map[resourceapi.QualifiedName]resourceapi.DeviceCapacity{
							resourceapi.QualifiedName("memory"): {
								Value: resource.MustParse("50Gi"),
							},
						},
					},
				}
				return slice
			}(),
		},
		"invalid-capacity-pool-in-device-capacity-consumption": {
			wantFailures: field.ErrorList{
				field.Invalid(field.NewPath("spec", "devices").Index(0).Child("consumesCapacity").Index(0).Child("capacityPool"),
					"does-not-exist", "must reference a capacity pool defined in the ResourceSlice"),
			},
			slice: func() *resourceapi.ResourceSlice {
				slice := testResourceSlice(goodName, goodName, goodName, 1)
				slice.Spec.Mixins = &resourceapi.ResourceSliceMixins{
					DeviceCapacityConsumption: []resourceapi.DeviceCapacityConsumptionMixin{
						{
							Name: "device-capacity-consumption-mixin",
							Capacity: map[resourceapi.QualifiedName]resourceapi.DeviceCapacity{
								resourceapi.QualifiedName("memory"): {
									Value: resource.MustParse("50Gi"),
								},
							},
						},
					},
				}
				slice.Spec.Devices[0].ConsumesCapacity = []resourceapi.DeviceCapacityConsumption{
					{
						CapacityPool: "does-not-exist",
						Includes: []resourceapi.DeviceCapacityConsumptionMixinRef{
							{
								Name: "device-capacity-consumption-mixin",
							},
						},
					},
				}
				return slice
			}(),
		},
		"invalid-includes-in-device-capacity-consumption": {
			wantFailures: field.ErrorList{
				field.Invalid(field.NewPath("spec", "devices").Index(0).Child("consumesCapacity").Index(0).Child("includes").Index(0),
					"does-not-exist", "must reference a device capacity consumption mixin defined in the ResourceSlice"),
			},
			slice: func() *resourceapi.ResourceSlice {
				slice := testResourceSlice(goodName, goodName, goodName, 1)
				slice.Spec.CapacityPools = []resourceapi.CapacityPool{
					{
						Name: "capacity-pool",
						Capacity: map[resourceapi.QualifiedName]resourceapi.DeviceCapacity{
							resourceapi.QualifiedName("memory"): {
								Value: resource.MustParse("50Gi"),
							},
						},
					},
				}
				slice.Spec.Devices[0].ConsumesCapacity = []resourceapi.DeviceCapacityConsumption{
					{
						CapacityPool: "capacity-pool",
						Includes: []resourceapi.DeviceCapacityConsumptionMixinRef{
							{
								Name: "does-not-exist",
							},
						},
					},
				}
				return slice
			}(),
		},
		"too-many-includes-in-device-capacity-consumption": {
			wantFailures: field.ErrorList{
				field.TooMany(field.NewPath("spec", "devices").Index(0).Child("consumesCapacity").Index(0).Child("includes"),
					resourceapi.ResourceSliceMaxDeviceCapacityConsumptionMixinRefs+1, resourceapi.ResourceSliceMaxDeviceCapacityConsumptionMixinRefs),
			},
			slice: func() *resourceapi.ResourceSlice {
				slice := testResourceSlice(goodName, goodName, goodName, 1)
				slice.Spec.CapacityPools = []resourceapi.CapacityPool{
					{
						Name: "capacity-pool",
						Capacity: map[resourceapi.QualifiedName]resourceapi.DeviceCapacity{
							resourceapi.QualifiedName("memory"): {
								Value: resource.MustParse("50Gi"),
							},
						},
					},
				}
				slice.Spec.Mixins = &resourceapi.ResourceSliceMixins{}
				slice.Spec.Devices[0].ConsumesCapacity = []resourceapi.DeviceCapacityConsumption{
					{
						CapacityPool: "capacity-pool",
					},
				}
				for i := 0; i < resourceapi.ResourceSliceMaxDeviceCapacityConsumptionMixinRefs+1; i++ {
					name := fmt.Sprintf("device-consumes-capacity-mixin-%d", i)
					slice.Spec.Mixins.DeviceCapacityConsumption = append(slice.Spec.Mixins.DeviceCapacityConsumption, resourceapi.DeviceCapacityConsumptionMixin{
						Name: name,
						Capacity: map[resourceapi.QualifiedName]resourceapi.DeviceCapacity{
							resourceapi.QualifiedName("memory"): {
								Value: resource.MustParse("50Gi"),
							},
						},
					})
					slice.Spec.Devices[0].ConsumesCapacity[0].Includes = append(slice.Spec.Devices[0].ConsumesCapacity[0].Includes, resourceapi.DeviceCapacityConsumptionMixinRef{
						Name: name,
					})
				}
				return slice
			}(),
		},
		"device-capacity-consumption-too-many-capacity": {
			wantFailures: field.ErrorList{
				field.TooMany(field.NewPath("spec", "devices").Index(0).Child("consumesCapacity").Index(0).Child("capacity"), resourceapi.ResourceSliceMaxAttributesAndCapacities+1, resourceapi.ResourceSliceMaxAttributesAndCapacities),
			},
			slice: func() *resourceapi.ResourceSlice {
				slice := testResourceSlice(goodName, goodName, goodName, 1)
				slice.Spec.CapacityPools = []resourceapi.CapacityPool{
					{
						Name: "capacity-pool",
						Capacity: map[resourceapi.QualifiedName]resourceapi.DeviceCapacity{
							resourceapi.QualifiedName("memory"): {
								Value: resource.MustParse("50Gi"),
							},
						},
					},
				}
				slice.Spec.Devices[0].ConsumesCapacity = []resourceapi.DeviceCapacityConsumption{
					{
						CapacityPool: "capacity-pool",
						Capacity:     make(map[resourceapi.QualifiedName]resourceapi.DeviceCapacity),
					},
				}

				for i := 0; i < resourceapi.ResourceSliceMaxAttributesAndCapacities+1; i++ {
					key := fmt.Sprintf("key%d", i)
					slice.Spec.Devices[0].ConsumesCapacity[0].Capacity[resourceapi.QualifiedName(key)] = resourceapi.DeviceCapacity{
						Value: resource.MustParse("40Gi"),
					}
				}
				return slice
			}(),
		},
		"device-capacity-consumption-must-have-include-or-capacity": {
			wantFailures: field.ErrorList{
				field.Invalid(field.NewPath("spec", "devices").Index(0).Child("consumesCapacity").Index(0), nil, "at least one of `includes` or `capacity` must be specified"),
			},
			slice: func() *resourceapi.ResourceSlice {
				slice := testResourceSlice(goodName, goodName, goodName, 1)
				slice.Spec.CapacityPools = []resourceapi.CapacityPool{
					{
						Name: "capacity-pool",
						Capacity: map[resourceapi.QualifiedName]resourceapi.DeviceCapacity{
							resourceapi.QualifiedName("memory"): {
								Value: resource.MustParse("50Gi"),
							},
						},
					},
				}
				slice.Spec.Devices[0].ConsumesCapacity = []resourceapi.DeviceCapacityConsumption{
					{
						CapacityPool: "capacity-pool",
					},
				}
				return slice
			}(),
		},
		"per-device-node-selection-and-device-without-node-selector": {
			wantFailures: field.ErrorList{
				field.Required(field.NewPath("spec", "devices").Index(1), "exactly one of `nodeName`, `nodeSelector`, or `allNodes` is required when `perDeviceNodeSelection` is set in the ResourceSlice spec"),
			},
			slice: func() *resourceapi.ResourceSlice {
				slice := testResourceSlice(goodName, goodName, goodName, 2)
				slice.Spec.NodeName = ""
				slice.Spec.PerDeviceNodeSelection = true
				slice.Spec.Devices[0].NodeName = "node"
				return slice
			}(),
		},
		"not-per-device-node-selection-and-device-node-selection": {
			wantFailures: field.ErrorList{
				field.Invalid(field.NewPath("spec", "devices").Index(0), nil, "`nodeName`, `nodeSelector` and `allNodes` can only be set if `perDeviceNodeSelection` is set in the ResourceSlice spec"),
				field.Invalid(field.NewPath("spec", "devices").Index(1), nil, "`nodeName`, `nodeSelector` and `allNodes` can only be set if `perDeviceNodeSelection` is set in the ResourceSlice spec"),
				field.Invalid(field.NewPath("spec", "devices").Index(2), nil, "`nodeName`, `nodeSelector` and `allNodes` can only be set if `perDeviceNodeSelection` is set in the ResourceSlice spec"),
			},
			slice: func() *resourceapi.ResourceSlice {
				slice := testResourceSlice(goodName, goodName, goodName, 3)
				slice.Spec.Devices[0].NodeName = "node"
				slice.Spec.Devices[1].NodeSelector = &core.NodeSelector{
					NodeSelectorTerms: []core.NodeSelectorTerm{{
						MatchExpressions: []core.NodeSelectorRequirement{{
							Key:      "foo",
							Operator: core.NodeSelectorOpIn,
							Values:   []string{"-1"},
						}},
					}},
				}
				slice.Spec.Devices[2].AllNodes = true
				return slice
			}(),
		},
		"invalid-device-node-selector": {
			wantFailures: field.ErrorList{
				field.Invalid(field.NewPath("spec", "devices").Index(0).Child("nodeSelector", "nodeSelectorTerms").Index(0).Child("matchExpressions").Index(0).Child("values").Index(0), "-1", "a valid label must be an empty string or consist of alphanumeric characters, '-', '_' or '.', and must start and end with an alphanumeric character (e.g. 'MyValue',  or 'my_value',  or '12345', regex used for validation is '(([A-Za-z0-9][-A-Za-z0-9_.]*)?[A-Za-z0-9])?')"),
			},
			slice: func() *resourceapi.ResourceSlice {
				slice := testResourceSlice(goodName, goodName, goodName, 1)
				slice.Spec.NodeName = ""
				slice.Spec.PerDeviceNodeSelection = true
				slice.Spec.Devices[0].NodeSelector = &core.NodeSelector{
					NodeSelectorTerms: []core.NodeSelectorTerm{{
						MatchExpressions: []core.NodeSelectorRequirement{{
							Key:      "foo",
							Operator: core.NodeSelectorOpIn,
							Values:   []string{"-1"},
						}},
					}},
				}
				return slice
			}(),
		},
	}

	for name, scenario := range scenarios {
		t.Run(name, func(t *testing.T) {
			errs := ValidateResourceSlice(scenario.slice)
			assertFailures(t, scenario.wantFailures, errs)
		})
	}
}

func TestResourceSliceCapacityAndAttributesFields(t *testing.T) {
	testCases := map[string]struct {
		attributes           map[resourceapi.QualifiedName]resourceapi.DeviceAttribute
		capacity             map[resourceapi.QualifiedName]resourceapi.DeviceCapacity
		expectedFailuresFunc func(basePath *field.Path, includeAttributes bool) field.ErrorList
		attributeTest        bool
	}{
		"bad-name": {
			capacity: map[resourceapi.QualifiedName]resourceapi.DeviceCapacity{
				resourceapi.QualifiedName(badName): {},
			},
			attributes: map[resourceapi.QualifiedName]resourceapi.DeviceAttribute{
				resourceapi.QualifiedName(badName): {},
			},
			expectedFailuresFunc: func(basePath *field.Path, includeAttributes bool) field.ErrorList {
				var errs field.ErrorList
				if includeAttributes {
					errs = append(errs, field.ErrorList{
						field.TypeInvalid(basePath.Index(0).Child("attributes").Key(badName), badName, "a valid C identifier must start with alphabetic character or '_', followed by a string of alphanumeric characters or '_' (e.g. 'my_name',  or 'MY_NAME',  or 'MyName', regex used for validation is '[A-Za-z_][A-Za-z0-9_]*')"),
						field.Required(basePath.Index(0).Child("attributes").Key(badName), "exactly one value must be specified"),
					}...)
				}
				errs = append(errs, field.ErrorList{
					field.TypeInvalid(basePath.Index(0).Child("capacity").Key(badName), badName, "a valid C identifier must start with alphabetic character or '_', followed by a string of alphanumeric characters or '_' (e.g. 'my_name',  or 'MY_NAME',  or 'MyName', regex used for validation is '[A-Za-z_][A-Za-z0-9_]*')"),
				}...)
				return errs
			},
		},
		"good-names": {
			capacity: map[resourceapi.QualifiedName]resourceapi.DeviceCapacity{
				resourceapi.QualifiedName(strings.Repeat("x", resourceapi.DeviceMaxIDLength)): {
					Value: resource.MustParse("40Gi"),
				},
				resourceapi.QualifiedName(strings.Repeat("x", resourceapi.DeviceMaxDomainLength) + "/" + strings.Repeat("y", resourceapi.DeviceMaxIDLength)): {
					Value: resource.MustParse("20Gi"),
				},
			},
			attributes: map[resourceapi.QualifiedName]resourceapi.DeviceAttribute{
				resourceapi.QualifiedName(strings.Repeat("x", resourceapi.DeviceMaxIDLength)):                                                                {StringValue: ptr.To("y")},
				resourceapi.QualifiedName(strings.Repeat("x", resourceapi.DeviceMaxDomainLength) + "/" + strings.Repeat("y", resourceapi.DeviceMaxIDLength)): {StringValue: ptr.To("z")},
			},
		},
		"bad-c-identifier": {
			capacity: map[resourceapi.QualifiedName]resourceapi.DeviceCapacity{
				resourceapi.QualifiedName(strings.Repeat(".", resourceapi.DeviceMaxIDLength+1)): {
					Value: resource.MustParse("40Gi"),
				},
			},
			attributes: map[resourceapi.QualifiedName]resourceapi.DeviceAttribute{
				resourceapi.QualifiedName(strings.Repeat(".", resourceapi.DeviceMaxIDLength+1)): {StringValue: ptr.To("y")},
			},
			expectedFailuresFunc: func(basePath *field.Path, includeAttributes bool) field.ErrorList {
				var errs field.ErrorList
				if includeAttributes {
					errs = append(errs, field.ErrorList{
						field.TooLongMaxLength(basePath.Index(0).Child("attributes").Key(strings.Repeat(".", resourceapi.DeviceMaxIDLength+1)), strings.Repeat(".", resourceapi.DeviceMaxIDLength+1), resourceapi.DeviceMaxIDLength),
						field.TypeInvalid(basePath.Index(0).Child("attributes").Key(strings.Repeat(".", resourceapi.DeviceMaxIDLength+1)), strings.Repeat(".", resourceapi.DeviceMaxIDLength+1), "a valid C identifier must start with alphabetic character or '_', followed by a string of alphanumeric characters or '_' (e.g. 'my_name',  or 'MY_NAME',  or 'MyName', regex used for validation is '[A-Za-z_][A-Za-z0-9_]*')"),
					}...)
				}
				errs = append(errs, field.ErrorList{
					field.TooLongMaxLength(basePath.Index(0).Child("capacity").Key(strings.Repeat(".", resourceapi.DeviceMaxIDLength+1)), strings.Repeat(".", resourceapi.DeviceMaxIDLength+1), resourceapi.DeviceMaxIDLength),
					field.TypeInvalid(basePath.Index(0).Child("capacity").Key(strings.Repeat(".", resourceapi.DeviceMaxIDLength+1)), strings.Repeat(".", resourceapi.DeviceMaxIDLength+1), "a valid C identifier must start with alphabetic character or '_', followed by a string of alphanumeric characters or '_' (e.g. 'my_name',  or 'MY_NAME',  or 'MyName', regex used for validation is '[A-Za-z_][A-Za-z0-9_]*')"),
				}...)
				return errs
			},
		},
		"bad-domain": {
			capacity: map[resourceapi.QualifiedName]resourceapi.DeviceCapacity{
				resourceapi.QualifiedName(strings.Repeat("_", resourceapi.DeviceMaxDomainLength+1) + "/y"): {
					Value: resource.MustParse("50Gi"),
				},
			},
			attributes: map[resourceapi.QualifiedName]resourceapi.DeviceAttribute{
				resourceapi.QualifiedName(strings.Repeat("_", resourceapi.DeviceMaxDomainLength+1) + "/y"): {StringValue: ptr.To("z")},
			},
			expectedFailuresFunc: func(basePath *field.Path, includeAttributes bool) field.ErrorList {
				var errs field.ErrorList
				if includeAttributes {
					errs = append(errs, field.ErrorList{
						field.TooLong(basePath.Index(0).Child("attributes").Key(strings.Repeat("_", resourceapi.DeviceMaxDomainLength+1)+"/y"), strings.Repeat("_", resourceapi.DeviceMaxDomainLength+1), resourceapi.DeviceMaxDomainLength),
						field.Invalid(basePath.Index(0).Child("attributes").Key(strings.Repeat("_", resourceapi.DeviceMaxDomainLength+1)+"/y"), strings.Repeat("_", resourceapi.DeviceMaxDomainLength+1), "a lowercase RFC 1123 subdomain must consist of lower case alphanumeric characters, '-' or '.', and must start and end with an alphanumeric character (e.g. 'example.com', regex used for validation is '[a-z0-9]([-a-z0-9]*[a-z0-9])?(\\.[a-z0-9]([-a-z0-9]*[a-z0-9])?)*')"),
					}...)
				}
				errs = append(errs, field.ErrorList{
					field.TooLong(basePath.Index(0).Child("capacity").Key(strings.Repeat("_", resourceapi.DeviceMaxDomainLength+1)+"/y"), strings.Repeat("_", resourceapi.DeviceMaxDomainLength+1), resourceapi.DeviceMaxDomainLength),
					field.Invalid(basePath.Index(0).Child("capacity").Key(strings.Repeat("_", resourceapi.DeviceMaxDomainLength+1)+"/y"), strings.Repeat("_", resourceapi.DeviceMaxDomainLength+1), "a lowercase RFC 1123 subdomain must consist of lower case alphanumeric characters, '-' or '.', and must start and end with an alphanumeric character (e.g. 'example.com', regex used for validation is '[a-z0-9]([-a-z0-9]*[a-z0-9])?(\\.[a-z0-9]([-a-z0-9]*[a-z0-9])?)*')"),
				}...)
				return errs
			},
		},
		"bad-key-too-long": {
			capacity: map[resourceapi.QualifiedName]resourceapi.DeviceCapacity{
				resourceapi.QualifiedName(strings.Repeat("x", resourceapi.DeviceMaxDomainLength+1) + "/" + strings.Repeat("y", resourceapi.DeviceMaxIDLength+1)): {
					Value: resource.MustParse("10Gi"),
				},
			},
			attributes: map[resourceapi.QualifiedName]resourceapi.DeviceAttribute{
				resourceapi.QualifiedName(strings.Repeat("x", resourceapi.DeviceMaxDomainLength+1) + "/" + strings.Repeat("y", resourceapi.DeviceMaxIDLength+1)): {StringValue: ptr.To("z")},
			},
			expectedFailuresFunc: func(basePath *field.Path, includeAttributes bool) field.ErrorList {
				var errs field.ErrorList
				if includeAttributes {
					errs = append(errs, field.ErrorList{
						field.TooLong(basePath.Index(0).Child("attributes").Key("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx...xxxxxxxxxxxx/yyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy"), strings.Repeat("x", resourceapi.DeviceMaxDomainLength+1), resourceapi.DeviceMaxDomainLength),
						field.TooLongMaxLength(basePath.Index(0).Child("attributes").Key("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx...xxxxxxxxxxxx/yyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy"), strings.Repeat("y", resourceapi.DeviceMaxIDLength+1), resourceapi.DeviceMaxIDLength),
					}...)
				}
				errs = append(errs, field.ErrorList{
					field.TooLong(basePath.Index(0).Child("capacity").Key("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx...xxxxxxxxxxxx/yyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy"), strings.Repeat("x", resourceapi.DeviceMaxDomainLength+1), resourceapi.DeviceMaxDomainLength),
					field.TooLongMaxLength(basePath.Index(0).Child("capacity").Key("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx...xxxxxxxxxxxx/yyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy"), strings.Repeat("y", resourceapi.DeviceMaxIDLength+1), resourceapi.DeviceMaxIDLength),
				}...)
				return errs
			},
		},
		"bad-empty-domain-and-c-identifier": {
			capacity: map[resourceapi.QualifiedName]resourceapi.DeviceCapacity{
				resourceapi.QualifiedName("/"): {
					Value: resource.MustParse("30Gi"),
				},
			},
			attributes: map[resourceapi.QualifiedName]resourceapi.DeviceAttribute{
				resourceapi.QualifiedName("/"): {StringValue: ptr.To("z")},
			},
			expectedFailuresFunc: func(basePath *field.Path, includeAttributes bool) field.ErrorList {
				var errs field.ErrorList
				if includeAttributes {
					errs = append(errs, field.ErrorList{
						field.Required(basePath.Index(0).Child("attributes").Key("/"), "the domain must not be empty"),
						field.Required(basePath.Index(0).Child("attributes").Key("/"), "the name must not be empty"),
					}...)
				}
				errs = append(errs, field.ErrorList{
					field.Required(basePath.Index(0).Child("capacity").Key("/"), "the domain must not be empty"),
					field.Required(basePath.Index(0).Child("capacity").Key("/"), "the name must not be empty"),
				}...)
				return errs
			},
		},
		"multiple-attribute-values": {
			attributes: map[resourceapi.QualifiedName]resourceapi.DeviceAttribute{
				resourceapi.QualifiedName("attribute1"): {StringValue: ptr.To("x"), VersionValue: ptr.To("1.2.3")},
			},
			expectedFailuresFunc: func(basePath *field.Path, includeAttributes bool) field.ErrorList {
				var errs field.ErrorList
				if includeAttributes {
					errs = append(errs, field.ErrorList{
						field.Invalid(basePath.Index(0).Child("attributes").Key("attribute1"), resourceapi.DeviceAttribute{StringValue: ptr.To("x"), VersionValue: ptr.To("1.2.3")}, "exactly one value must be specified"),
					}...)
				}
				return errs
			},
			attributeTest: true,
		},
		"version-attribute-must-be-semver": {
			attributes: map[resourceapi.QualifiedName]resourceapi.DeviceAttribute{
				resourceapi.QualifiedName("attribute2"): {VersionValue: ptr.To(strings.Repeat("x", resourceapi.DeviceAttributeMaxValueLength+1))},
			},
			expectedFailuresFunc: func(basePath *field.Path, includeAttributes bool) field.ErrorList {
				var errs field.ErrorList
				if includeAttributes {
					errs = append(errs, field.ErrorList{
						field.Invalid(basePath.Index(0).Child("attributes").Key("attribute2").Child("version"), strings.Repeat("x", resourceapi.DeviceAttributeMaxValueLength+1), "must be a string compatible with semver.org spec 2.0.0"),
						field.TooLongMaxLength(basePath.Index(0).Child("attributes").Key("attribute2").Child("version"), strings.Repeat("x", resourceapi.DeviceAttributeMaxValueLength+1), resourceapi.DeviceAttributeMaxValueLength),
					}...)
				}
				return errs
			},
			attributeTest: true,
		},
		"string-attribute-value-too-long": {
			attributes: map[resourceapi.QualifiedName]resourceapi.DeviceAttribute{
				resourceapi.QualifiedName("attribute3"): {StringValue: ptr.To(strings.Repeat("x", resourceapi.DeviceAttributeMaxValueLength+1))},
			},
			expectedFailuresFunc: func(basePath *field.Path, includeAttributes bool) field.ErrorList {
				var errs field.ErrorList
				if includeAttributes {
					errs = append(errs, field.ErrorList{
						field.TooLongMaxLength(basePath.Index(0).Child("attributes").Key("attribute3").Child("string"), strings.Repeat("x", resourceapi.DeviceAttributeMaxValueLength+1), resourceapi.DeviceAttributeMaxValueLength),
					}...)
				}
				return errs
			},
			attributeTest: true,
		},
	}

	fieldConfigs := map[string]struct {
		sliceCreateFunc func(capacity map[resourceapi.QualifiedName]resourceapi.DeviceCapacity, attributes map[resourceapi.QualifiedName]resourceapi.DeviceAttribute) *resourceapi.ResourceSlice
		hasAttributes   bool
		basePath        *field.Path
	}{
		"device": {
			sliceCreateFunc: func(capacity map[resourceapi.QualifiedName]resourceapi.DeviceCapacity, attributes map[resourceapi.QualifiedName]resourceapi.DeviceAttribute) *resourceapi.ResourceSlice {
				slice := testResourceSlice(goodName, goodName, goodName, 1)
				slice.Spec.Devices[0].Attributes = attributes
				slice.Spec.Devices[0].Capacity = capacity
				return slice
			},
			hasAttributes: true,
			basePath:      field.NewPath("spec", "devices"),
		},
		"device-mixin": {
			sliceCreateFunc: func(capacity map[resourceapi.QualifiedName]resourceapi.DeviceCapacity, attributes map[resourceapi.QualifiedName]resourceapi.DeviceAttribute) *resourceapi.ResourceSlice {
				slice := testResourceSlice(goodName, goodName, goodName, 1)
				slice.Spec.Mixins = &resourceapi.ResourceSliceMixins{
					Device: []resourceapi.DeviceMixin{
						{
							Name:       "device-mixin-1",
							Attributes: attributes,
							Capacity:   capacity,
						},
					},
				}
				return slice
			},
			hasAttributes: true,
			basePath:      field.NewPath("spec", "mixins", "device"),
		},
		"device-consumption-mixin": {
			sliceCreateFunc: func(capacity map[resourceapi.QualifiedName]resourceapi.DeviceCapacity, attributes map[resourceapi.QualifiedName]resourceapi.DeviceAttribute) *resourceapi.ResourceSlice {
				slice := testResourceSlice(goodName, goodName, goodName, 1)
				slice.Spec.Mixins = &resourceapi.ResourceSliceMixins{
					DeviceCapacityConsumption: []resourceapi.DeviceCapacityConsumptionMixin{
						{
							Name:     "device-consumption-mixin-1",
							Capacity: capacity,
						},
					},
				}
				return slice
			},
			hasAttributes: false,
			basePath:      field.NewPath("spec", "mixins", "deviceCapacityConsumption"),
		},
		"capacity-pool-mixin": {
			sliceCreateFunc: func(capacity map[resourceapi.QualifiedName]resourceapi.DeviceCapacity, attributes map[resourceapi.QualifiedName]resourceapi.DeviceAttribute) *resourceapi.ResourceSlice {
				slice := testResourceSlice(goodName, goodName, goodName, 1)
				slice.Spec.Mixins = &resourceapi.ResourceSliceMixins{
					CapacityPool: []resourceapi.CapacityPoolMixin{
						{
							Name:     "capacity-pool-mixin",
							Capacity: capacity,
						},
					},
				}
				return slice
			},
			hasAttributes: false,
			basePath:      field.NewPath("spec", "mixins", "capacityPool"),
		},
		"capacity-pool": {
			sliceCreateFunc: func(capacity map[resourceapi.QualifiedName]resourceapi.DeviceCapacity, attributes map[resourceapi.QualifiedName]resourceapi.DeviceAttribute) *resourceapi.ResourceSlice {
				slice := testResourceSlice(goodName, goodName, goodName, 1)
				slice.Spec.CapacityPools = []resourceapi.CapacityPool{
					{
						Name:     "capacity-pool",
						Capacity: capacity,
					},
				}
				return slice
			},
			hasAttributes: false,
			basePath:      field.NewPath("spec", "capacityPools"),
		},
		"device-capacity-consumption": {
			sliceCreateFunc: func(capacity map[resourceapi.QualifiedName]resourceapi.DeviceCapacity, attributes map[resourceapi.QualifiedName]resourceapi.DeviceAttribute) *resourceapi.ResourceSlice {
				slice := testResourceSlice(goodName, goodName, goodName, 1)
				slice.Spec.CapacityPools = []resourceapi.CapacityPool{
					{
						Name: "capacity-pool",
						Capacity: map[resourceapi.QualifiedName]resourceapi.DeviceCapacity{
							resourceapi.QualifiedName("memory"): {
								Value: resource.MustParse("50Gi"),
							},
						},
					},
				}
				slice.Spec.Devices[0].ConsumesCapacity = []resourceapi.DeviceCapacityConsumption{
					{
						CapacityPool: "capacity-pool",
						Capacity:     capacity,
					},
				}
				return slice
			},
			hasAttributes: false,
			basePath:      field.NewPath("spec", "devices").Index(0).Child("consumesCapacity"),
		},
	}

	for fieldName, fieldConfig := range fieldConfigs {
		for name, tc := range testCases {
			t.Run(fmt.Sprintf("%s/%s", fieldName, name), func(t *testing.T) {
				// Don't run the attributes tests for fieldConfigs that doesn't
				// have the attributes field.
				if tc.attributeTest && !fieldConfig.hasAttributes {
					t.Skip()
				}
				slice := fieldConfig.sliceCreateFunc(tc.capacity, tc.attributes)
				errs := ValidateResourceSlice(slice)
				var expectedFailures field.ErrorList
				if tc.expectedFailuresFunc != nil {
					expectedFailures = tc.expectedFailuresFunc(fieldConfig.basePath, fieldConfig.hasAttributes)
				}
				assertFailures(t, expectedFailures, errs)
			})
		}
	}
}

func TestValidateResourceSliceUpdate(t *testing.T) {
	name := "valid"
	validResourceSlice := testResourceSlice(name, name, name, 1)

	scenarios := map[string]struct {
		oldResourceSlice *resourceapi.ResourceSlice
		update           func(slice *resourceapi.ResourceSlice) *resourceapi.ResourceSlice
		wantFailures     field.ErrorList
	}{
		"valid-no-op-update": {
			oldResourceSlice: validResourceSlice,
			update:           func(slice *resourceapi.ResourceSlice) *resourceapi.ResourceSlice { return slice },
		},
		"invalid-name-update": {
			oldResourceSlice: validResourceSlice,
			update: func(slice *resourceapi.ResourceSlice) *resourceapi.ResourceSlice {
				slice.Name += "-update"
				return slice
			},
			wantFailures: field.ErrorList{field.Invalid(field.NewPath("metadata", "name"), name+"-update", "field is immutable")},
		},
		"invalid-update-nodename": {
			wantFailures:     field.ErrorList{field.Invalid(field.NewPath("spec", "nodeName"), name+"-updated", "field is immutable")},
			oldResourceSlice: validResourceSlice,
			update: func(slice *resourceapi.ResourceSlice) *resourceapi.ResourceSlice {
				slice.Spec.NodeName += "-updated"
				return slice
			},
		},
		"invalid-update-drivername": {
			wantFailures:     field.ErrorList{field.Invalid(field.NewPath("spec", "driver"), name+"-updated", "field is immutable")},
			oldResourceSlice: validResourceSlice,
			update: func(slice *resourceapi.ResourceSlice) *resourceapi.ResourceSlice {
				slice.Spec.Driver += "-updated"
				return slice
			},
		},
		"invalid-update-pool": {
			wantFailures:     field.ErrorList{field.Invalid(field.NewPath("spec", "pool", "name"), validResourceSlice.Spec.Pool.Name+"-updated", "field is immutable")},
			oldResourceSlice: validResourceSlice,
			update: func(slice *resourceapi.ResourceSlice) *resourceapi.ResourceSlice {
				slice.Spec.Pool.Name += "-updated"
				return slice
			},
		},
		"invalid-update-to-invalid-nodeselector-label-value": {
			wantFailures: field.ErrorList{field.Invalid(field.NewPath("spec", "nodeSelector", "nodeSelectorTerms").Index(0).Child("matchExpressions").Index(0).Child("values").Index(0), "-1", "a valid label must be an empty string or consist of alphanumeric characters, '-', '_' or '.', and must start and end with an alphanumeric character (e.g. 'MyValue',  or 'my_value',  or '12345', regex used for validation is '(([A-Za-z0-9][-A-Za-z0-9_.]*)?[A-Za-z0-9])?')")},
			oldResourceSlice: func() *resourceapi.ResourceSlice {
				slice := validResourceSlice.DeepCopy()
				slice.Spec.NodeName = ""
				slice.Spec.NodeSelector = &core.NodeSelector{
					NodeSelectorTerms: []core.NodeSelectorTerm{{
						MatchExpressions: []core.NodeSelectorRequirement{{
							Key:      "foo",
							Operator: core.NodeSelectorOpIn,
							Values:   []string{"bar"},
						}},
					}},
				}
				return slice
			}(),
			update: func(slice *resourceapi.ResourceSlice) *resourceapi.ResourceSlice {
				slice.Spec.NodeSelector = &core.NodeSelector{
					NodeSelectorTerms: []core.NodeSelectorTerm{{
						MatchExpressions: []core.NodeSelectorRequirement{{
							Key:      "foo",
							Operator: core.NodeSelectorOpIn,
							Values:   []string{"-1"},
						}},
					}},
				}
				return slice
			},
		},
	}

	for name, scenario := range scenarios {
		t.Run(name, func(t *testing.T) {
			scenario.oldResourceSlice.ResourceVersion = "1"
			errs := ValidateResourceSliceUpdate(scenario.update(scenario.oldResourceSlice.DeepCopy()), scenario.oldResourceSlice)
			assertFailures(t, scenario.wantFailures, errs)
		})
	}
}
