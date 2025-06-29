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

	_ "k8s.io/kubernetes/pkg/apis/resource/install"
)

func testAttributes() map[resourceapi.QualifiedName]resourceapi.DeviceAttribute {
	return map[resourceapi.QualifiedName]resourceapi.DeviceAttribute{
		"int":     {IntValue: ptr.To(int64(42))},
		"string":  {StringValue: ptr.To("hello world")},
		"version": {VersionValue: ptr.To("1.2.3")},
		"bool":    {BoolValue: ptr.To(true)},
	}
}

func testStringAttributes(namePrefix string, count int) map[resourceapi.QualifiedName]resourceapi.DeviceAttribute {
	attributes := make(map[resourceapi.QualifiedName]resourceapi.DeviceAttribute)
	for i := 0; i < count; i++ {
		attributes[resourceapi.QualifiedName(fmt.Sprintf("%s%d", namePrefix, i))] = resourceapi.DeviceAttribute{
			StringValue: ptr.To("hello world"),
		}
	}
	return attributes
}

func testCapacity() map[resourceapi.QualifiedName]resourceapi.DeviceCapacity {
	return map[resourceapi.QualifiedName]resourceapi.DeviceCapacity{
		"memory": {Value: resource.MustParse("1Gi")},
	}
}

func testCapacities(namePrefix string, count int) map[resourceapi.QualifiedName]resourceapi.DeviceCapacity {
	capacities := make(map[resourceapi.QualifiedName]resourceapi.DeviceCapacity)
	for i := 0; i < count; i++ {
		capacities[resourceapi.QualifiedName(fmt.Sprintf("%s%d", namePrefix, i))] = resourceapi.DeviceCapacity{
			Value: resource.MustParse("1Gi"),
		}
	}
	return capacities
}

func testCounter() map[string]resourceapi.Counter {
	return map[string]resourceapi.Counter{
		"memory": {Value: resource.MustParse("1Gi")},
	}
}

func testCounters(count int) map[string]resourceapi.Counter {
	counters := make(map[string]resourceapi.Counter)
	for i := 0; i < count; i++ {
		counters[fmt.Sprintf("counter-%d", i)] = resourceapi.Counter{
			Value: resource.MustParse("1Gi"),
		}
	}
	return counters
}

func testResourceSlice(name, nodeName, driverName string, numDevices int) *resourceapi.ResourceSlice {
	slice := &resourceapi.ResourceSlice{
		ObjectMeta: metav1.ObjectMeta{
			Name: name,
		},
		Spec: resourceapi.ResourceSliceSpec{
			NodeName: &nodeName,
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
				slice.Spec.NodeName = nil
				slice.Spec.NodeSelector = &core.NodeSelector{}
				return slice
			}(),
		},
		"bad-node-selection": {
			wantFailures: field.ErrorList{field.Invalid(field.NewPath("spec"), "{`nodeName`, `nodeSelector`}", "exactly one of `nodeName`, `nodeSelector`, `allNodes`, `perDeviceNodeSelection` is required, but multiple fields are set")},
			slice: func() *resourceapi.ResourceSlice {
				slice := testResourceSlice(goodName, goodName, driverName, 1)
				slice.Spec.NodeName = ptr.To("worker")
				slice.Spec.NodeSelector = &core.NodeSelector{
					NodeSelectorTerms: []core.NodeSelectorTerm{{MatchFields: []core.NodeSelectorRequirement{{Key: "metadata.name", Operator: core.NodeSelectorOpIn, Values: []string{"worker"}}}}},
				}
				return slice
			}(),
		},
		"bad-node-selection-all-nodes": {
			wantFailures: field.ErrorList{field.Invalid(field.NewPath("spec"), "{`nodeName`, `allNodes`}", "exactly one of `nodeName`, `nodeSelector`, `allNodes`, `perDeviceNodeSelection` is required, but multiple fields are set")},
			slice: func() *resourceapi.ResourceSlice {
				slice := testResourceSlice(goodName, goodName, driverName, 1)
				slice.Spec.NodeName = ptr.To("worker")
				slice.Spec.AllNodes = ptr.To(true)
				return slice
			}(),
		},
		"empty-node-selection": {
			wantFailures: field.ErrorList{field.Required(field.NewPath("spec"), "exactly one of `nodeName`, `nodeSelector`, `allNodes`, `perDeviceNodeSelection` is required")},
			slice: func() *resourceapi.ResourceSlice {
				slice := testResourceSlice(goodName, goodName, driverName, 1)
				slice.Spec.NodeName = nil
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
		"combined-attributes-capacity-length": {
			wantFailures: field.ErrorList{
				field.Invalid(field.NewPath("spec", "devices").Index(3), resourceapi.ResourceSliceMaxAttributesAndCapacitiesPerDeviceAfterMixins+1, fmt.Sprintf("the total number of attributes and capacities in a device after mixins have been applied must not exceed %d", resourceapi.ResourceSliceMaxAttributesAndCapacitiesPerDeviceAfterMixins)),
			},
			slice: func() *resourceapi.ResourceSlice {
				slice := testResourceSlice(goodName, goodName, goodName, 5)
				slice.Spec.Devices[0].Attributes = map[resourceapi.QualifiedName]resourceapi.DeviceAttribute{}
				slice.Spec.Devices[0].Capacity = map[resourceapi.QualifiedName]resourceapi.DeviceCapacity{}
				for i := 0; i < resourceapi.ResourceSliceMaxAttributesAndCapacitiesPerDeviceAfterMixins; i++ {
					slice.Spec.Devices[0].Attributes[resourceapi.QualifiedName(fmt.Sprintf("attr_%d", i))] = resourceapi.DeviceAttribute{StringValue: ptr.To("x")}
				}
				slice.Spec.Devices[1].Attributes = map[resourceapi.QualifiedName]resourceapi.DeviceAttribute{}
				slice.Spec.Devices[1].Capacity = map[resourceapi.QualifiedName]resourceapi.DeviceCapacity{}
				quantity := resource.MustParse("1Gi")
				capacity := resourceapi.DeviceCapacity{Value: quantity}
				for i := 0; i < resourceapi.ResourceSliceMaxAttributesAndCapacitiesPerDeviceAfterMixins; i++ {
					slice.Spec.Devices[1].Capacity[resourceapi.QualifiedName(fmt.Sprintf("cap_%d", i))] = capacity
				}

				// Too large together by one.
				slice.Spec.Devices[3].Attributes = slice.Spec.Devices[0].Attributes
				slice.Spec.Devices[3].Capacity = map[resourceapi.QualifiedName]resourceapi.DeviceCapacity{
					"cap": capacity,
				}
				return slice
			}(),
		},
		"combined-attributes-capacity-length-after-mixins-applied": {
			wantFailures: field.ErrorList{
				field.Invalid(field.NewPath("spec", "devices").Index(1), resourceapi.ResourceSliceMaxAttributesAndCapacitiesPerDeviceAfterMixins+5, fmt.Sprintf("the total number of attributes and capacities in a device after mixins have been applied must not exceed %d", resourceapi.ResourceSliceMaxAttributesAndCapacitiesPerDeviceAfterMixins)),
				field.Invalid(field.NewPath("spec", "devices").Index(3), resourceapi.ResourceSliceMaxAttributesAndCapacitiesPerDeviceAfterMixins+5, fmt.Sprintf("the total number of attributes and capacities in a device after mixins have been applied must not exceed %d", resourceapi.ResourceSliceMaxAttributesAndCapacitiesPerDeviceAfterMixins)),
				field.Invalid(field.NewPath("spec", "devices").Index(5), resourceapi.ResourceSliceMaxAttributesAndCapacitiesPerDeviceAfterMixins+10, fmt.Sprintf("the total number of attributes and capacities in a device after mixins have been applied must not exceed %d", resourceapi.ResourceSliceMaxAttributesAndCapacitiesPerDeviceAfterMixins)),
			},
			slice: func() *resourceapi.ResourceSlice {
				slice := testResourceSlice(goodName, goodName, goodName, 6)
				slice.Spec.Mixins = &resourceapi.ResourceSliceMixins{
					Device: []resourceapi.DeviceMixin{
						{
							Name:       "device-mixin-only-attr",
							Attributes: testStringAttributes("mixin_attr_", 5),
						},
						{
							Name:     "device-mixin-only-cap",
							Capacity: testCapacities("mixin_cap_", 5),
						},
					},
				}
				// Device has max number of attributes and attributes from mixin are not new ones.
				slice.Spec.Devices[0].Includes = createDeviceMixinRefs(slice.Spec.Mixins.Device[0])
				slice.Spec.Devices[0].Attributes = testStringAttributes("attr_", resourceapi.ResourceSliceMaxAttributesAndCapacitiesPerDeviceAfterMixins-5)
				slice.Spec.Devices[0].Capacity = map[resourceapi.QualifiedName]resourceapi.DeviceCapacity{}
				for qname, attr := range slice.Spec.Mixins.Device[0].Attributes {
					slice.Spec.Devices[0].Attributes[qname] = attr
				}
				// Device has max number of attributes and attributes from mixin are new ones that
				// brings the number of the limit.
				slice.Spec.Devices[1].Includes = createDeviceMixinRefs(slice.Spec.Mixins.Device[0])
				slice.Spec.Devices[1].Attributes = testStringAttributes("attr_", resourceapi.ResourceSliceMaxAttributesAndCapacitiesPerDeviceAfterMixins)
				slice.Spec.Devices[1].Capacity = map[resourceapi.QualifiedName]resourceapi.DeviceCapacity{}

				// Device has max number of capacities and capacities from mixin are not new ones.
				slice.Spec.Devices[2].Includes = createDeviceMixinRefs(slice.Spec.Mixins.Device[1])
				slice.Spec.Devices[2].Capacity = testCapacities("cap_", resourceapi.ResourceSliceMaxAttributesAndCapacitiesPerDeviceAfterMixins-5)
				slice.Spec.Devices[2].Attributes = map[resourceapi.QualifiedName]resourceapi.DeviceAttribute{}
				for qname, cap := range slice.Spec.Mixins.Device[1].Capacity {
					slice.Spec.Devices[2].Capacity[qname] = cap
				}
				// Device has max number of capacities and capacities from mixin are new ones that
				// brings the number of the limit.
				slice.Spec.Devices[3].Includes = createDeviceMixinRefs(slice.Spec.Mixins.Device[1])
				slice.Spec.Devices[3].Capacity = testCapacities("cap_", resourceapi.ResourceSliceMaxAttributesAndCapacitiesPerDeviceAfterMixins)
				slice.Spec.Devices[3].Attributes = map[resourceapi.QualifiedName]resourceapi.DeviceAttribute{}
				// Device has max number of attributes and capacities and those from the mixins are not new ones.
				slice.Spec.Devices[4].Includes = createDeviceMixinRefs(slice.Spec.Mixins.Device...)
				slice.Spec.Devices[4].Attributes = testStringAttributes("attr_", resourceapi.ResourceSliceMaxAttributesAndCapacitiesPerDeviceAfterMixins/2-5)
				for qname, attr := range slice.Spec.Mixins.Device[0].Attributes {
					slice.Spec.Devices[4].Attributes[qname] = attr
				}
				slice.Spec.Devices[4].Capacity = testCapacities("cap_", resourceapi.ResourceSliceMaxAttributesAndCapacitiesPerDeviceAfterMixins/2-5)
				for qname, cap := range slice.Spec.Mixins.Device[1].Capacity {
					slice.Spec.Devices[4].Capacity[qname] = cap
				}
				// Device has max number of attributes and capacities and the ones from the mixin brings the
				// count above the limit
				slice.Spec.Devices[5].Includes = createDeviceMixinRefs(slice.Spec.Mixins.Device...)
				slice.Spec.Devices[5].Attributes = testStringAttributes("attr_", resourceapi.ResourceSliceMaxAttributesAndCapacitiesPerDeviceAfterMixins/2)
				slice.Spec.Devices[5].Capacity = testCapacities("cap_", resourceapi.ResourceSliceMaxAttributesAndCapacitiesPerDeviceAfterMixins/2)
				return slice
			}(),
		},
		"max-attributes-capacity-in-resourceslice": {
			wantFailures: field.ErrorList{
				field.Invalid(field.NewPath("spec"), resourceapi.ResourceSliceMaxAttributesAndCapacitiesPerResourceSlice+1, fmt.Sprintf("the total number of attributes and capacities in devices and mixins must not exceed %d", resourceapi.ResourceSliceMaxAttributesAndCapacitiesPerResourceSlice)),
			},
			slice: func() *resourceapi.ResourceSlice {
				slice := testResourceSlice(goodName, goodName, goodName, resourceapi.ResourceSliceMaxDevices)
				for i := 0; i < resourceapi.ResourceSliceMaxDevices; i++ {
					slice.Spec.Devices[i].Attributes = map[resourceapi.QualifiedName]resourceapi.DeviceAttribute{}
					slice.Spec.Devices[i].Capacity = map[resourceapi.QualifiedName]resourceapi.DeviceCapacity{}
					for j := 0; j < resourceapi.ResourceSliceMaxAttributesAndCapacitiesPerDeviceAfterMixins/2; j++ {
						slice.Spec.Devices[i].Attributes[resourceapi.QualifiedName(fmt.Sprintf("attr_%d", j))] = resourceapi.DeviceAttribute{StringValue: ptr.To("x")}
					}
					for j := 0; j < resourceapi.ResourceSliceMaxAttributesAndCapacitiesPerDeviceAfterMixins/2; j++ {
						slice.Spec.Devices[i].Capacity[resourceapi.QualifiedName(fmt.Sprintf("cap_%d", j))] = resourceapi.DeviceCapacity{Value: resource.MustParse("1Gi")}
					}
				}
				slice.Spec.Mixins = &resourceapi.ResourceSliceMixins{
					Device: []resourceapi.DeviceMixin{
						{
							Name:     "device-mixin",
							Capacity: testCapacity(),
						},
					},
				}
				return slice
			}(),
		},
		"invalid-node-selecor-label-value": {
			wantFailures: field.ErrorList{field.Invalid(field.NewPath("spec", "nodeSelector", "nodeSelectorTerms").Index(0).Child("matchExpressions").Index(0).Child("values").Index(0), "-1", "a valid label must be an empty string or consist of alphanumeric characters, '-', '_' or '.', and must start and end with an alphanumeric character (e.g. 'MyValue',  or 'my_value',  or '12345', regex used for validation is '(([A-Za-z0-9][-A-Za-z0-9_.]*)?[A-Za-z0-9])?')")},
			slice: func() *resourceapi.ResourceSlice {
				slice := testResourceSlice(goodName, goodName, goodName, 3)
				slice.Spec.NodeName = nil
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
		"taints": {
			wantFailures: func() field.ErrorList {
				fldPath := field.NewPath("spec", "devices").Index(0).Child("taints")
				return field.ErrorList{
					field.Invalid(fldPath.Index(2).Child("key"), "", "name part must be non-empty"),
					field.Invalid(fldPath.Index(2).Child("key"), "", "name part must consist of alphanumeric characters, '-', '_' or '.', and must start and end with an alphanumeric character (e.g. 'MyName',  or 'my.name',  or '123-abc', regex used for validation is '([A-Za-z0-9][-A-Za-z0-9_.]*)?[A-Za-z0-9]')"),
					field.Required(fldPath.Index(2).Child("effect"), ""),

					field.Invalid(fldPath.Index(3).Child("key"), badName, "name part must consist of alphanumeric characters, '-', '_' or '.', and must start and end with an alphanumeric character (e.g. 'MyName',  or 'my.name',  or '123-abc', regex used for validation is '([A-Za-z0-9][-A-Za-z0-9_.]*)?[A-Za-z0-9]')"),
					field.Invalid(fldPath.Index(3).Child("value"), badName, "a valid label must be an empty string or consist of alphanumeric characters, '-', '_' or '.', and must start and end with an alphanumeric character (e.g. 'MyValue',  or 'my_value',  or '12345', regex used for validation is '(([A-Za-z0-9][-A-Za-z0-9_.]*)?[A-Za-z0-9])?')"),
					field.NotSupported(fldPath.Index(3).Child("effect"), resourceapi.DeviceTaintEffect("some-other-op"), []resourceapi.DeviceTaintEffect{resourceapi.DeviceTaintEffectNoExecute, resourceapi.DeviceTaintEffectNoSchedule}),
				}
			}(),
			slice: func() *resourceapi.ResourceSlice {
				slice := testResourceSlice(goodName, goodName, goodName, 3)
				slice.Spec.Devices[0].Taints = []resourceapi.DeviceTaint{
					{
						// Minimal valid taint.
						Key:    "example.com/taint",
						Effect: resourceapi.DeviceTaintEffectNoExecute,
					},
					{
						// Full valid taint, other key and effect.
						Key:       "taint",
						Value:     "tainted",
						Effect:    resourceapi.DeviceTaintEffectNoSchedule,
						TimeAdded: ptr.To(metav1.Now()),
					},
					{
						// Invalid, all empty!
					},
					{
						// Invalid strings.
						Key:    badName,
						Value:  badName,
						Effect: "some-other-op",
					},
				}
				return slice
			}(),
		},
		"too-many-taints": {
			wantFailures: field.ErrorList{
				field.TooMany(field.NewPath("spec", "devices").Index(0).Child("taints"), resourceapi.DeviceTaintsMaxLength+1, resourceapi.DeviceTaintsMaxLength),
			},
			slice: func() *resourceapi.ResourceSlice {
				slice := testResourceSlice(goodName, goodName, driverName, 1)
				for i := 0; i < resourceapi.DeviceTaintsMaxLength+1; i++ {
					slice.Spec.Devices[0].Taints = append(slice.Spec.Devices[0].Taints, resourceapi.DeviceTaint{
						Key:    "example.com/taint",
						Effect: resourceapi.DeviceTaintEffectNoExecute,
					})
				}
				return slice
			}(),
		},
		"bad-PerDeviceNodeSelection": {
			wantFailures: field.ErrorList{
				field.Invalid(field.NewPath("spec"), "{`nodeName`, `perDeviceNodeSelection`}", "exactly one of `nodeName`, `nodeSelector`, `allNodes`, `perDeviceNodeSelection` is required, but multiple fields are set"),
				field.Required(field.NewPath("spec", "devices").Index(0), "exactly one of `nodeName`, `nodeSelector`, or `allNodes` is required when `perDeviceNodeSelection` is set to true in the ResourceSlice spec"),
			},
			slice: func() *resourceapi.ResourceSlice {
				slice := testResourceSlice(goodName, goodName, driverName, 1)
				slice.Spec.NodeName = ptr.To("worker")
				slice.Spec.PerDeviceNodeSelection = func() *bool {
					r := true
					return &r
				}()
				return slice
			}(),
		},
		"invalid-false-PerDeviceNodeSelection": {
			wantFailures: field.ErrorList{
				field.Invalid(field.NewPath("spec", "perDeviceNodeSelection"), false, "must be either unset or set to true"),
			},
			slice: func() *resourceapi.ResourceSlice {
				slice := testResourceSlice(goodName, goodName, driverName, 1)
				slice.Spec.NodeName = ptr.To("worker")
				slice.Spec.PerDeviceNodeSelection = func() *bool {
					r := false
					return &r
				}()
				return slice
			}(),
		},
		"invalid-false-AllNodes": {
			wantFailures: field.ErrorList{
				field.Invalid(field.NewPath("spec", "allNodes"), false, "must be either unset or set to true"),
			},
			slice: func() *resourceapi.ResourceSlice {
				slice := testResourceSlice(goodName, goodName, driverName, 1)
				slice.Spec.NodeName = ptr.To("worker")
				slice.Spec.AllNodes = func() *bool {
					r := false
					return &r
				}()
				return slice
			}(),
		},
		"invalid-empty-NodeName": {
			wantFailures: field.ErrorList{
				field.Invalid(field.NewPath("spec", "nodeName"), "", "must be either unset or set to a non-empty string"),
			},
			slice: func() *resourceapi.ResourceSlice {
				slice := testResourceSlice(goodName, goodName, driverName, 1)
				slice.Spec.NodeName = ptr.To("")
				slice.Spec.AllNodes = func() *bool {
					r := true
					return &r
				}()
				return slice
			}(),
		},
		"invalid-node-selector-in-basicdevice": {
			wantFailures: field.ErrorList{
				field.Invalid(field.NewPath("spec", "devices").Index(0).Child("nodeName"), "", "must not be empty"),
				field.Invalid(field.NewPath("spec", "devices").Index(0).Child("allNodes"), false, "must be either unset or set to true"),
				field.Required(field.NewPath("spec", "devices").Index(0), "exactly one of `nodeName`, `nodeSelector`, or `allNodes` is required when `perDeviceNodeSelection` is set to true in the ResourceSlice spec"),
			},
			slice: func() *resourceapi.ResourceSlice {
				slice := testResourceSlice(goodName, goodName, driverName, 1)
				slice.Spec.PerDeviceNodeSelection = func() *bool {
					r := true
					return &r
				}()
				slice.Spec.NodeName = nil
				slice.Spec.Devices[0].NodeName = func() *string {
					r := ""
					return &r
				}()
				slice.Spec.Devices[0].AllNodes = func() *bool {
					r := false
					return &r
				}()
				return slice
			}(),
		},
		"bad-node-selector-in-basicdevice": {
			wantFailures: field.ErrorList{
				field.Invalid(field.NewPath("spec", "devices").Index(0), "{`nodeName`, `allNodes`}", "exactly one of `nodeName`, `nodeSelector`, or `allNodes` is required when `perDeviceNodeSelection` is set to true in the ResourceSlice spec"),
			},
			slice: func() *resourceapi.ResourceSlice {
				slice := testResourceSlice(goodName, goodName, driverName, 1)
				slice.Spec.PerDeviceNodeSelection = func() *bool {
					r := true
					return &r
				}()
				slice.Spec.NodeName = nil
				slice.Spec.Devices[0].NodeName = func() *string {
					r := "worker"
					return &r
				}()
				slice.Spec.Devices[0].AllNodes = func() *bool {
					r := true
					return &r
				}()
				return slice
			}(),
		},
		"bad-name-shared-counters": {
			wantFailures: field.ErrorList{
				field.Invalid(field.NewPath("spec", "sharedCounters").Index(0).Child("name"), badName, "a lowercase RFC 1123 label must consist of lower case alphanumeric characters or '-', and must start and end with an alphanumeric character (e.g. 'my-name',  or '123-abc', regex used for validation is '[a-z0-9]([-a-z0-9]*[a-z0-9])?')"),
			},
			slice: func() *resourceapi.ResourceSlice {
				slice := testResourceSlice(goodName, goodName, driverName, 1)
				slice.Spec.SharedCounters = []resourceapi.CounterSet{
					{
						Name:     badName,
						Counters: testCounter(),
					},
				}
				return slice
			}(),
		},
		"shared-counters-no-counters-or-includes": {
			wantFailures: field.ErrorList{
				field.Required(field.NewPath("spec", "sharedCounters").Index(0), "at least one of `counters` or `includes` must be specified"),
			},
			slice: func() *resourceapi.ResourceSlice {
				slice := testResourceSlice(goodName, goodName, driverName, 1)
				slice.Spec.SharedCounters = []resourceapi.CounterSet{
					{
						Name: goodName,
					},
				}
				return slice
			}(),
		},
		"bad-countername-shared-counters": {
			wantFailures: field.ErrorList{
				field.Invalid(field.NewPath("spec", "sharedCounters").Index(0).Child("counters").Key(badName), badName, "a lowercase RFC 1123 label must consist of lower case alphanumeric characters or '-', and must start and end with an alphanumeric character (e.g. 'my-name',  or '123-abc', regex used for validation is '[a-z0-9]([-a-z0-9]*[a-z0-9])?')"),
			},
			slice: func() *resourceapi.ResourceSlice {
				slice := testResourceSlice(goodName, goodName, driverName, 1)
				slice.Spec.SharedCounters = []resourceapi.CounterSet{
					{
						Name: goodName,
						Counters: map[string]resourceapi.Counter{
							badName: {Value: resource.MustParse("1Gi")},
						},
					},
				}
				return slice
			}(),
		},
		"missing-name-shared-counters": {
			wantFailures: field.ErrorList{
				field.Required(field.NewPath("spec", "sharedCounters").Index(0).Child("name"), ""),
			},
			slice: func() *resourceapi.ResourceSlice {
				slice := testResourceSlice(goodName, goodName, driverName, 1)
				slice.Spec.SharedCounters = []resourceapi.CounterSet{
					{
						Counters: testCounter(),
					},
				}
				return slice
			}(),
		},
		"duplicate-shared-counters": {
			wantFailures: field.ErrorList{
				field.Duplicate(field.NewPath("spec", "sharedCounters").Index(1).Child("name"), goodName),
			},
			slice: func() *resourceapi.ResourceSlice {
				slice := testResourceSlice(goodName, goodName, driverName, 1)
				slice.Spec.SharedCounters = []resourceapi.CounterSet{
					{
						Name:     goodName,
						Counters: testCounter(),
					},
					{
						Name:     goodName,
						Counters: testCounter(),
					},
				}
				return slice
			}(),
		},
		"too-large-shared-counters": {
			wantFailures: field.ErrorList{
				field.Invalid(field.NewPath("spec"), resourceapi.ResourceSliceMaxCountersPerResourceSlice+1, fmt.Sprintf("the total number of counters in shared counters and counter set mixins must not exceed %d", resourceapi.ResourceSliceMaxCountersPerResourceSlice)),
			},
			slice: func() *resourceapi.ResourceSlice {
				slice := testResourceSlice(goodName, goodName, driverName, 1)
				slice.Spec.SharedCounters = createSharedCounters(resourceapi.ResourceSliceMaxCountersPerResourceSlice + 1)
				return slice
			}(),
		},
		"too-large-shared-counters-with-mixins": {
			wantFailures: field.ErrorList{
				field.Invalid(field.NewPath("spec"), resourceapi.ResourceSliceMaxCountersPerResourceSlice+1, fmt.Sprintf("the total number of counters in shared counters and counter set mixins must not exceed %d", resourceapi.ResourceSliceMaxCountersPerResourceSlice)),
			},
			slice: func() *resourceapi.ResourceSlice {
				slice := testResourceSlice(goodName, goodName, driverName, 1)
				slice.Spec.Mixins = &resourceapi.ResourceSliceMixins{
					CounterSet: createCounterSetMixins(resourceapi.ResourceSliceMaxCountersPerResourceSlice/2 + 1),
				}
				slice.Spec.SharedCounters = createSharedCounters(resourceapi.ResourceSliceMaxCountersPerResourceSlice / 2)
				return slice
			}(),
		},
		"missing-counterset-consumes-counter": {
			wantFailures: field.ErrorList{
				field.Required(field.NewPath("spec", "devices").Index(0).Child("consumesCounters").Index(0).Child("counterSet"), ""),
				field.Invalid(field.NewPath("spec", "devices").Index(0).Child("consumesCounters").Index(0).Child("counterSet"), "", "must reference a counterSet defined in the ResourceSlice sharedCounters"),
			},
			slice: func() *resourceapi.ResourceSlice {
				slice := testResourceSlice(goodName, goodName, driverName, 1)
				slice.Spec.SharedCounters = createSharedCounters(1)
				slice.Spec.Devices[0].ConsumesCounters = []resourceapi.DeviceCounterConsumption{
					{
						Counters: testCounter(),
					},
				}
				return slice
			}(),
		},
		"missing-counter-consumes-counter-or-includes": {
			wantFailures: field.ErrorList{
				field.Required(field.NewPath("spec", "devices").Index(0).Child("consumesCounters").Index(0), "at least one of `counters` or `includes` must be specified"),
			},
			slice: func() *resourceapi.ResourceSlice {
				slice := testResourceSlice(goodName, goodName, driverName, 1)
				slice.Spec.SharedCounters = createSharedCounters(1)
				slice.Spec.Devices[0].ConsumesCounters = []resourceapi.DeviceCounterConsumption{
					{
						CounterSet: "counterset-0",
					},
				}
				return slice
			}(),
		},
		"wrong-counterref-consumes-counter": {
			wantFailures: field.ErrorList{
				field.Invalid(field.NewPath("spec", "devices").Index(0).Child("consumesCounters").Index(0).Child("counters"), "fake", "must reference a counter defined in the ResourceSlice sharedCounters"),
			},
			slice: func() *resourceapi.ResourceSlice {
				slice := testResourceSlice(goodName, goodName, driverName, 1)
				slice.Spec.SharedCounters = createSharedCounters(1)
				slice.Spec.Devices[0].ConsumesCounters = []resourceapi.DeviceCounterConsumption{
					{
						Counters: map[string]resourceapi.Counter{
							"fake": {Value: resource.MustParse("1Gi")},
						},
						CounterSet: "counterset-0",
					},
				}
				return slice
			}(),
		},
		"wrong-sharedcounterref-consumes-counter": {
			wantFailures: field.ErrorList{
				field.Invalid(field.NewPath("spec", "devices").Index(0).Child("consumesCounters").Index(0).Child("counterSet"), "fake", "must reference a counterSet defined in the ResourceSlice sharedCounters"),
			},
			slice: func() *resourceapi.ResourceSlice {
				slice := testResourceSlice(goodName, goodName, driverName, 1)
				slice.Spec.SharedCounters = createSharedCounters(1)
				slice.Spec.Devices[0].ConsumesCounters = []resourceapi.DeviceCounterConsumption{
					{
						Counters:   testCounter(),
						CounterSet: "fake",
					},
				}
				return slice
			}(),
		},
		"counterref-consumes-counter-from-mixin": {
			slice: func() *resourceapi.ResourceSlice {
				slice := testResourceSlice(goodName, goodName, driverName, 1)
				slice.Spec.Mixins = &resourceapi.ResourceSliceMixins{
					CounterSet: []resourceapi.CounterSetMixin{
						{
							Name: "counterset-mixin",
							Counters: map[string]resourceapi.Counter{
								"from-counter-set-mixin": {Value: resource.MustParse("1Gi")},
							},
						},
					},
				}
				slice.Spec.SharedCounters = []resourceapi.CounterSet{
					{
						Name:     "counterset-0",
						Includes: []string{"counterset-mixin"},
					},
				}
				slice.Spec.Devices[0].ConsumesCounters = []resourceapi.DeviceCounterConsumption{
					{
						Counters: map[string]resourceapi.Counter{
							"from-counter-set-mixin": {Value: resource.MustParse("1Gi")},
						},
						CounterSet: slice.Spec.SharedCounters[0].Name,
					},
				}
				return slice
			}(),
		},
		"too-many-consumed-counters-in-slice": {
			wantFailures: field.ErrorList{
				field.Invalid(field.NewPath("spec"), resourceapi.ResourceSliceMaxConsumedCountersPerResourceSlice+1, fmt.Sprintf("the total number of consumed counters in devices and mixins must not exceed %d", resourceapi.ResourceSliceMaxConsumedCountersPerResourceSlice)),
			},
			slice: func() *resourceapi.ResourceSlice {
				slice := testResourceSlice(goodName, goodName, driverName, resourceapi.ResourceSliceMaxDevices)
				slice.Spec.SharedCounters = createSharedCounters(16)
				for i := 0; i < resourceapi.ResourceSliceMaxDevices; i++ {
					slice.Spec.Devices[i].ConsumesCounters = createConsumesCountersFromCounterSets(slice.Spec.SharedCounters...)
				}
				slice.Spec.SharedCounters = append(slice.Spec.SharedCounters, resourceapi.CounterSet{
					Name:     "last-counterset",
					Counters: testCounter(),
				})
				slice.Spec.Devices[0].ConsumesCounters = append(slice.Spec.Devices[0].ConsumesCounters, resourceapi.DeviceCounterConsumption{
					CounterSet: "last-counterset",
					Counters:   testCounter(),
				})
				return slice
			}(),
		},
		"too-many-consumed-counters-in-slice-with-mixins": {
			wantFailures: field.ErrorList{
				field.Invalid(field.NewPath("spec"), resourceapi.ResourceSliceMaxConsumedCountersPerResourceSlice+16, fmt.Sprintf("the total number of consumed counters in devices and mixins must not exceed %d", resourceapi.ResourceSliceMaxConsumedCountersPerResourceSlice)),
			},
			slice: func() *resourceapi.ResourceSlice {
				slice := testResourceSlice(goodName, goodName, driverName, 1)
				slice.Spec.SharedCounters = []resourceapi.CounterSet{
					{
						Name:     "counterset",
						Counters: testCounters(16),
					},
				}
				slice.Spec.Mixins = &resourceapi.ResourceSliceMixins{}
				for i := 0; i < resourceapi.ResourceSliceMaxDevices; i++ {
					dcc := createDeviceCounterConsumptionMixinsFromCounterSets(slice.Spec.SharedCounters[0])[0]
					dcc.Name = fmt.Sprintf("counterset-mixin-%d", i)
					slice.Spec.Mixins.DeviceCounterConsumption = append(slice.Spec.Mixins.DeviceCounterConsumption, dcc)
				}
				slice.Spec.Devices[0].ConsumesCounters = createConsumesCountersFromCounterSets(slice.Spec.SharedCounters[0])
				return slice
			}(),
		},
		"empty-device-mixin-name": {
			wantFailures: field.ErrorList{field.Required(field.NewPath("spec", "mixins", "device").Index(0).Child("name"), "")},
			slice: func() *resourceapi.ResourceSlice {
				slice := testResourceSlice(goodName, goodName, driverName, 1)
				slice.Spec.Mixins = &resourceapi.ResourceSliceMixins{
					Device: createDeviceMixins(1),
				}
				slice.Spec.Mixins.Device[0].Name = ""
				return slice
			}(),
		},
		"bad-device-mixin-name": {
			wantFailures: field.ErrorList{field.Invalid(field.NewPath("spec", "mixins", "device").Index(0).Child("name"), badName, "a lowercase RFC 1123 label must consist of lower case alphanumeric characters or '-', and must start and end with an alphanumeric character (e.g. 'my-name',  or '123-abc', regex used for validation is '[a-z0-9]([-a-z0-9]*[a-z0-9])?')")},
			slice: func() *resourceapi.ResourceSlice {
				slice := testResourceSlice(goodName, goodName, driverName, 1)
				slice.Spec.Mixins = &resourceapi.ResourceSliceMixins{
					Device: createDeviceMixins(1),
				}
				slice.Spec.Mixins.Device[0].Name = badName
				return slice
			}(),
		},
		"too-many-device-mixin-refs": {
			wantFailures: field.ErrorList{
				field.TooMany(field.NewPath("spec", "devices").Index(0).Child("includes"), resourceapi.ResourceSliceMaxIncludes+1, resourceapi.ResourceSliceMaxIncludes),
			},
			slice: func() *resourceapi.ResourceSlice {
				slice := testResourceSlice(goodName, goodName, driverName, 1)
				slice.Spec.Mixins = &resourceapi.ResourceSliceMixins{
					Device: createDeviceMixins(resourceapi.ResourceSliceMaxIncludes + 1),
				}
				slice.Spec.Devices[0].Includes = createDeviceMixinRefs(slice.Spec.Mixins.Device...)
				return slice
			}(),
		},
		"unknown-device-mixin-ref": {
			wantFailures: field.ErrorList{
				field.Invalid(field.NewPath("spec", "devices").Index(0).Child("includes").Index(0), "does-not-exist", "must reference a device mixin defined in the ResourceSlice"),
			},
			slice: func() *resourceapi.ResourceSlice {
				slice := testResourceSlice(goodName, goodName, driverName, 1)
				slice.Spec.Devices[0].Includes = []string{"does-not-exist"}
				return slice
			}(),
		},
		"empty-counter-set-mixin-name": {
			wantFailures: field.ErrorList{field.Required(field.NewPath("spec", "mixins", "counterSet").Index(0).Child("name"), "")},
			slice: func() *resourceapi.ResourceSlice {
				slice := testResourceSlice(goodName, goodName, driverName, 1)
				slice.Spec.Mixins = &resourceapi.ResourceSliceMixins{
					CounterSet: createCounterSetMixins(1),
				}
				slice.Spec.Mixins.CounterSet[0].Name = ""
				return slice
			}(),
		},
		"bad-counter-set-mixin-name": {
			wantFailures: field.ErrorList{field.Invalid(field.NewPath("spec", "mixins", "counterSet").Index(0).Child("name"), badName, "a lowercase RFC 1123 label must consist of lower case alphanumeric characters or '-', and must start and end with an alphanumeric character (e.g. 'my-name',  or '123-abc', regex used for validation is '[a-z0-9]([-a-z0-9]*[a-z0-9])?')")},
			slice: func() *resourceapi.ResourceSlice {
				slice := testResourceSlice(goodName, goodName, driverName, 1)
				slice.Spec.Mixins = &resourceapi.ResourceSliceMixins{
					CounterSet: createCounterSetMixins(1),
				}
				slice.Spec.Mixins.CounterSet[0].Name = badName
				return slice
			}(),
		},
		"too-many-counter-set-mixin-refs": {
			wantFailures: field.ErrorList{
				field.TooMany(field.NewPath("spec", "sharedCounters").Index(0).Child("includes"), resourceapi.ResourceSliceMaxIncludes+1, resourceapi.ResourceSliceMaxIncludes),
			},
			slice: func() *resourceapi.ResourceSlice {
				slice := testResourceSlice(goodName, goodName, driverName, 1)
				slice.Spec.Mixins = &resourceapi.ResourceSliceMixins{
					CounterSet: createCounterSetMixins(resourceapi.ResourceSliceMaxIncludes + 1),
				}
				slice.Spec.SharedCounters = []resourceapi.CounterSet{
					{
						Name:     "counterset-0",
						Counters: testCounter(),
						Includes: createCounterSetMixinRefs(slice.Spec.Mixins.CounterSet...),
					},
				}
				return slice
			}(),
		},
		"unknown-counter-set-mixin-ref": {
			wantFailures: field.ErrorList{
				field.Invalid(field.NewPath("spec", "sharedCounters").Index(0).Child("includes").Index(0), "does-not-exist", "must reference a counter set mixin defined in the ResourceSlice"),
			},
			slice: func() *resourceapi.ResourceSlice {
				slice := testResourceSlice(goodName, goodName, driverName, 1)
				slice.Spec.SharedCounters = []resourceapi.CounterSet{
					{
						Name:     "counterset",
						Counters: testCounter(),
						Includes: []string{"does-not-exist"},
					},
				}
				return slice
			}(),
		},
		"empty-device-counter-consumption-mixin-name": {
			wantFailures: field.ErrorList{field.Required(field.NewPath("spec", "mixins", "deviceCounterConsumption").Index(0).Child("name"), "")},
			slice: func() *resourceapi.ResourceSlice {
				slice := testResourceSlice(goodName, goodName, driverName, 1)
				slice.Spec.Mixins = &resourceapi.ResourceSliceMixins{
					DeviceCounterConsumption: createDeviceCounterConsumptionMixins(1),
				}
				slice.Spec.Mixins.DeviceCounterConsumption[0].Name = ""
				return slice
			}(),
		},
		"bad-device-counter-consumption-mixin-name": {
			wantFailures: field.ErrorList{field.Invalid(field.NewPath("spec", "mixins", "deviceCounterConsumption").Index(0).Child("name"), badName, "a lowercase RFC 1123 label must consist of lower case alphanumeric characters or '-', and must start and end with an alphanumeric character (e.g. 'my-name',  or '123-abc', regex used for validation is '[a-z0-9]([-a-z0-9]*[a-z0-9])?')")},
			slice: func() *resourceapi.ResourceSlice {
				slice := testResourceSlice(goodName, goodName, driverName, 1)
				slice.Spec.Mixins = &resourceapi.ResourceSliceMixins{
					DeviceCounterConsumption: createDeviceCounterConsumptionMixins(1),
				}
				slice.Spec.Mixins.DeviceCounterConsumption[0].Name = badName
				return slice
			}(),
		},
		"too-many-device-counter-consumption-mixin-refs": {
			wantFailures: field.ErrorList{
				field.TooMany(field.NewPath("spec", "devices").Index(0).Child("consumesCounters").Index(0).Child("includes"), resourceapi.ResourceSliceMaxIncludes+1, resourceapi.ResourceSliceMaxIncludes),
			},
			slice: func() *resourceapi.ResourceSlice {
				slice := testResourceSlice(goodName, goodName, driverName, 1)
				slice.Spec.SharedCounters = createSharedCounters(1)
				slice.Spec.Mixins = &resourceapi.ResourceSliceMixins{
					DeviceCounterConsumption: createDeviceCounterConsumptionMixins(resourceapi.ResourceSliceMaxIncludes + 1),
				}
				slice.Spec.Devices[0].ConsumesCounters = createConsumesCountersFromCounterSets(slice.Spec.SharedCounters...)
				slice.Spec.Devices[0].ConsumesCounters[0].Includes = createDeviceCounterConsumptionMixinRefs(slice.Spec.Mixins.DeviceCounterConsumption...)
				return slice
			}(),
		},
		"unknown-device-counter-consumption-mixin-ref": {
			wantFailures: field.ErrorList{
				field.Invalid(field.NewPath("spec", "devices").Index(0).Child("consumesCounters").Index(0).Child("includes").Index(0), "does-not-exist", "must reference a device counter consumption mixin defined in the ResourceSlice"),
			},
			slice: func() *resourceapi.ResourceSlice {
				slice := testResourceSlice(goodName, goodName, driverName, 1)
				slice.Spec.SharedCounters = createSharedCounters(1)
				slice.Spec.Devices[0].ConsumesCounters = createConsumesCountersFromCounterSets(slice.Spec.SharedCounters...)
				slice.Spec.Devices[0].ConsumesCounters[0].Includes = []string{"does-not-exist"}
				return slice
			}(),
		},
		"device-counter-consumption-mixin-ref-unknown-counters": {
			wantFailures: field.ErrorList{
				field.Invalid(field.NewPath("spec", "devices").Index(0).Child("consumesCounters").Index(0).Child("includes").Index(0), "", "mixin references counters does-not-exist, does-not-exist-2 that do not exist in counter set counterset-0"),
			},
			slice: func() *resourceapi.ResourceSlice {
				slice := testResourceSlice(goodName, goodName, driverName, 1)
				slice.Spec.Mixins = &resourceapi.ResourceSliceMixins{
					DeviceCounterConsumption: createDeviceCounterConsumptionMixins(1),
				}
				slice.Spec.Mixins.DeviceCounterConsumption[0].Counters = map[string]resourceapi.Counter{
					"does-not-exist":   {Value: resource.MustParse("1Gi")},
					"does-not-exist-2": {Value: resource.MustParse("1Gi")},
				}
				slice.Spec.SharedCounters = createSharedCounters(1)
				slice.Spec.Devices[0].ConsumesCounters = []resourceapi.DeviceCounterConsumption{
					{
						CounterSet: slice.Spec.SharedCounters[0].Name,
						Includes:   createDeviceCounterConsumptionMixinRefs(slice.Spec.Mixins.DeviceCounterConsumption...),
					},
				}
				return slice
			}(),
		},
		"device-counter-consumption-multiple-mixin-refs-unknown-counter": {
			wantFailures: field.ErrorList{
				field.Invalid(field.NewPath("spec", "devices").Index(0).Child("consumesCounters").Index(0).Child("includes").Index(0), "", "mixin references counters does-not-exist that do not exist in counter set counterset-0"),
				field.Invalid(field.NewPath("spec", "devices").Index(0).Child("consumesCounters").Index(0).Child("includes").Index(1), "", "mixin references counters does-not-exist that do not exist in counter set counterset-0"),
			},
			slice: func() *resourceapi.ResourceSlice {
				slice := testResourceSlice(goodName, goodName, driverName, 1)
				slice.Spec.Mixins = &resourceapi.ResourceSliceMixins{
					DeviceCounterConsumption: createDeviceCounterConsumptionMixins(2),
				}
				slice.Spec.Mixins.DeviceCounterConsumption[0].Counters = map[string]resourceapi.Counter{
					"does-not-exist": {Value: resource.MustParse("1Gi")},
				}
				slice.Spec.Mixins.DeviceCounterConsumption[1].Counters = map[string]resourceapi.Counter{
					"does-not-exist": {Value: resource.MustParse("1Gi")},
				}
				slice.Spec.SharedCounters = createSharedCounters(1)
				slice.Spec.Devices[0].ConsumesCounters = []resourceapi.DeviceCounterConsumption{
					{
						CounterSet: slice.Spec.SharedCounters[0].Name,
						Includes:   createDeviceCounterConsumptionMixinRefs(slice.Spec.Mixins.DeviceCounterConsumption...),
					},
				}
				return slice
			}(),
		},
		"device-counter-consumption-through-mixin-from-counterset-mixin": {
			slice: func() *resourceapi.ResourceSlice {
				slice := testResourceSlice(goodName, goodName, driverName, 1)
				slice.Spec.Mixins = &resourceapi.ResourceSliceMixins{
					CounterSet:               createCounterSetMixins(1),
					DeviceCounterConsumption: createDeviceCounterConsumptionMixins(1),
				}
				slice.Spec.SharedCounters = []resourceapi.CounterSet{
					{
						Name:     "counterset-0",
						Includes: createCounterSetMixinRefs(slice.Spec.Mixins.CounterSet...),
					},
				}
				slice.Spec.Devices[0].ConsumesCounters = []resourceapi.DeviceCounterConsumption{
					{
						CounterSet: slice.Spec.SharedCounters[0].Name,
						Includes:   createDeviceCounterConsumptionMixinRefs(slice.Spec.Mixins.DeviceCounterConsumption...),
					},
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
		expectedFailuresFunc func(basePath *field.Path) field.ErrorList
	}{
		"bad-name": {
			capacity: map[resourceapi.QualifiedName]resourceapi.DeviceCapacity{
				resourceapi.QualifiedName(badName): {},
			},
			attributes: map[resourceapi.QualifiedName]resourceapi.DeviceAttribute{
				resourceapi.QualifiedName(badName): {},
			},
			expectedFailuresFunc: func(basePath *field.Path) field.ErrorList {
				return field.ErrorList{
					field.Invalid(basePath.Index(0).Child("attributes").Key(badName), badName, "a valid C identifier must start with alphabetic character or '_', followed by a string of alphanumeric characters or '_' (e.g. 'my_name',  or 'MY_NAME',  or 'MyName', regex used for validation is '[A-Za-z_][A-Za-z0-9_]*')"),
					field.Required(basePath.Index(0).Child("attributes").Key(badName), "exactly one value must be specified"),
					field.Invalid(basePath.Index(0).Child("capacity").Key(badName), badName, "a valid C identifier must start with alphabetic character or '_', followed by a string of alphanumeric characters or '_' (e.g. 'my_name',  or 'MY_NAME',  or 'MyName', regex used for validation is '[A-Za-z_][A-Za-z0-9_]*')"),
				}
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
			expectedFailuresFunc: func(basePath *field.Path) field.ErrorList {
				return field.ErrorList{
					field.TooLongMaxLength(basePath.Index(0).Child("attributes").Key(strings.Repeat(".", resourceapi.DeviceMaxIDLength+1)), strings.Repeat(".", resourceapi.DeviceMaxIDLength+1), resourceapi.DeviceMaxIDLength),
					field.Invalid(basePath.Index(0).Child("attributes").Key(strings.Repeat(".", resourceapi.DeviceMaxIDLength+1)), strings.Repeat(".", resourceapi.DeviceMaxIDLength+1), "a valid C identifier must start with alphabetic character or '_', followed by a string of alphanumeric characters or '_' (e.g. 'my_name',  or 'MY_NAME',  or 'MyName', regex used for validation is '[A-Za-z_][A-Za-z0-9_]*')"),
					field.TooLongMaxLength(basePath.Index(0).Child("capacity").Key(strings.Repeat(".", resourceapi.DeviceMaxIDLength+1)), strings.Repeat(".", resourceapi.DeviceMaxIDLength+1), resourceapi.DeviceMaxIDLength),
					field.Invalid(basePath.Index(0).Child("capacity").Key(strings.Repeat(".", resourceapi.DeviceMaxIDLength+1)), strings.Repeat(".", resourceapi.DeviceMaxIDLength+1), "a valid C identifier must start with alphabetic character or '_', followed by a string of alphanumeric characters or '_' (e.g. 'my_name',  or 'MY_NAME',  or 'MyName', regex used for validation is '[A-Za-z_][A-Za-z0-9_]*')"),
				}
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
			expectedFailuresFunc: func(basePath *field.Path) field.ErrorList {
				return field.ErrorList{
					field.TooLong(basePath.Index(0).Child("attributes").Key(strings.Repeat("_", resourceapi.DeviceMaxDomainLength+1)+"/y"), strings.Repeat("_", resourceapi.DeviceMaxDomainLength+1), resourceapi.DeviceMaxDomainLength),
					field.Invalid(basePath.Index(0).Child("attributes").Key(strings.Repeat("_", resourceapi.DeviceMaxDomainLength+1)+"/y"), strings.Repeat("_", resourceapi.DeviceMaxDomainLength+1), "a lowercase RFC 1123 subdomain must consist of lower case alphanumeric characters, '-' or '.', and must start and end with an alphanumeric character (e.g. 'example.com', regex used for validation is '[a-z0-9]([-a-z0-9]*[a-z0-9])?(\\.[a-z0-9]([-a-z0-9]*[a-z0-9])?)*')"),
					field.TooLong(basePath.Index(0).Child("capacity").Key(strings.Repeat("_", resourceapi.DeviceMaxDomainLength+1)+"/y"), strings.Repeat("_", resourceapi.DeviceMaxDomainLength+1), resourceapi.DeviceMaxDomainLength),
					field.Invalid(basePath.Index(0).Child("capacity").Key(strings.Repeat("_", resourceapi.DeviceMaxDomainLength+1)+"/y"), strings.Repeat("_", resourceapi.DeviceMaxDomainLength+1), "a lowercase RFC 1123 subdomain must consist of lower case alphanumeric characters, '-' or '.', and must start and end with an alphanumeric character (e.g. 'example.com', regex used for validation is '[a-z0-9]([-a-z0-9]*[a-z0-9])?(\\.[a-z0-9]([-a-z0-9]*[a-z0-9])?)*')"),
				}
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
			expectedFailuresFunc: func(basePath *field.Path) field.ErrorList {
				return field.ErrorList{
					field.TooLong(basePath.Index(0).Child("attributes").Key("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx...xxxxxxxxxxxx/yyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy"), strings.Repeat("x", resourceapi.DeviceMaxDomainLength+1), resourceapi.DeviceMaxDomainLength),
					field.TooLongMaxLength(basePath.Index(0).Child("attributes").Key("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx...xxxxxxxxxxxx/yyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy"), strings.Repeat("y", resourceapi.DeviceMaxIDLength+1), resourceapi.DeviceMaxIDLength),
					field.TooLong(basePath.Index(0).Child("capacity").Key("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx...xxxxxxxxxxxx/yyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy"), strings.Repeat("x", resourceapi.DeviceMaxDomainLength+1), resourceapi.DeviceMaxDomainLength),
					field.TooLongMaxLength(basePath.Index(0).Child("capacity").Key("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx...xxxxxxxxxxxx/yyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy"), strings.Repeat("y", resourceapi.DeviceMaxIDLength+1), resourceapi.DeviceMaxIDLength),
				}
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
			expectedFailuresFunc: func(basePath *field.Path) field.ErrorList {
				return field.ErrorList{
					field.Required(basePath.Index(0).Child("attributes").Key("/"), "the domain must not be empty"),
					field.Required(basePath.Index(0).Child("attributes").Key("/"), "the name must not be empty"),
					field.Required(basePath.Index(0).Child("capacity").Key("/"), "the domain must not be empty"),
					field.Required(basePath.Index(0).Child("capacity").Key("/"), "the name must not be empty"),
				}
			},
		},
		"multiple-attribute-values": {
			attributes: map[resourceapi.QualifiedName]resourceapi.DeviceAttribute{
				resourceapi.QualifiedName("attribute1"): {StringValue: ptr.To("x"), VersionValue: ptr.To("1.2.3")},
			},
			expectedFailuresFunc: func(basePath *field.Path) field.ErrorList {
				return field.ErrorList{
					field.Invalid(basePath.Index(0).Child("attributes").Key("attribute1"), resourceapi.DeviceAttribute{StringValue: ptr.To("x"), VersionValue: ptr.To("1.2.3")}, "exactly one value must be specified"),
				}
			},
		},
		"version-attribute-must-be-semver": {
			attributes: map[resourceapi.QualifiedName]resourceapi.DeviceAttribute{
				resourceapi.QualifiedName("attribute2"): {VersionValue: ptr.To(strings.Repeat("x", resourceapi.DeviceAttributeMaxValueLength+1))},
			},
			expectedFailuresFunc: func(basePath *field.Path) field.ErrorList {
				return field.ErrorList{
					field.Invalid(basePath.Index(0).Child("attributes").Key("attribute2").Child("version"), strings.Repeat("x", resourceapi.DeviceAttributeMaxValueLength+1), "must be a string compatible with semver.org spec 2.0.0"),
					field.TooLongMaxLength(basePath.Index(0).Child("attributes").Key("attribute2").Child("version"), strings.Repeat("x", resourceapi.DeviceAttributeMaxValueLength+1), resourceapi.DeviceAttributeMaxValueLength),
				}
			},
		},
		"string-attribute-value-too-long": {
			attributes: map[resourceapi.QualifiedName]resourceapi.DeviceAttribute{
				resourceapi.QualifiedName("attribute3"): {StringValue: ptr.To(strings.Repeat("x", resourceapi.DeviceAttributeMaxValueLength+1))},
			},
			expectedFailuresFunc: func(basePath *field.Path) field.ErrorList {
				return field.ErrorList{
					field.TooLongMaxLength(basePath.Index(0).Child("attributes").Key("attribute3").Child("string"), strings.Repeat("x", resourceapi.DeviceAttributeMaxValueLength+1), resourceapi.DeviceAttributeMaxValueLength),
				}
			},
		},
	}

	fieldConfigs := map[string]struct {
		sliceCreateFunc func(capacity map[resourceapi.QualifiedName]resourceapi.DeviceCapacity, attributes map[resourceapi.QualifiedName]resourceapi.DeviceAttribute) *resourceapi.ResourceSlice
		basePath        *field.Path
	}{
		"device": {
			sliceCreateFunc: func(capacity map[resourceapi.QualifiedName]resourceapi.DeviceCapacity, attributes map[resourceapi.QualifiedName]resourceapi.DeviceAttribute) *resourceapi.ResourceSlice {
				slice := testResourceSlice(goodName, goodName, goodName, 1)
				slice.Spec.Devices[0].Attributes = attributes
				slice.Spec.Devices[0].Capacity = capacity
				return slice
			},
			basePath: field.NewPath("spec", "devices"),
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
			basePath: field.NewPath("spec", "mixins", "device"),
		},
	}

	for fieldName, fieldConfig := range fieldConfigs {
		for name, tc := range testCases {
			t.Run(fmt.Sprintf("%s/%s", fieldName, name), func(t *testing.T) {
				slice := fieldConfig.sliceCreateFunc(tc.capacity, tc.attributes)
				errs := ValidateResourceSlice(slice)
				var expectedFailures field.ErrorList
				if tc.expectedFailuresFunc != nil {
					expectedFailures = tc.expectedFailuresFunc(fieldConfig.basePath)
				}
				assertFailures(t, expectedFailures, errs)
			})
		}
	}
}

func TestResourceSliceCounterFields(t *testing.T) {
	testCases := map[string]struct {
		counters             map[string]resourceapi.Counter
		expectedFailuresFunc func(basePath *field.Path) field.ErrorList
	}{
		"bad-name": {
			counters: map[string]resourceapi.Counter{
				badName: {},
			},
			expectedFailuresFunc: func(basePath *field.Path) field.ErrorList {
				return field.ErrorList{
					field.Invalid(basePath.Index(0).Child("counters").Key(badName), badName, "a lowercase RFC 1123 label must consist of lower case alphanumeric characters or '-', and must start and end with an alphanumeric character (e.g. 'my-name',  or '123-abc', regex used for validation is '[a-z0-9]([-a-z0-9]*[a-z0-9])?')"),
				}
			},
		},
	}

	fieldConfigs := map[string]struct {
		sliceCreateFunc func(counters map[string]resourceapi.Counter) *resourceapi.ResourceSlice
		basePath        *field.Path
	}{
		"counter-set": {
			sliceCreateFunc: func(counters map[string]resourceapi.Counter) *resourceapi.ResourceSlice {
				slice := testResourceSlice(goodName, goodName, goodName, 1)
				slice.Spec.SharedCounters = []resourceapi.CounterSet{
					{
						Name:     goodName,
						Counters: counters,
					},
				}
				return slice
			},
			basePath: field.NewPath("spec", "sharedCounters"),
		},
		"counter-set-mixin": {
			sliceCreateFunc: func(counters map[string]resourceapi.Counter) *resourceapi.ResourceSlice {
				slice := testResourceSlice(goodName, goodName, goodName, 1)
				slice.Spec.Mixins = &resourceapi.ResourceSliceMixins{
					CounterSet: []resourceapi.CounterSetMixin{
						{
							Name:     goodName,
							Counters: counters,
						},
					},
				}
				return slice
			},
			basePath: field.NewPath("spec", "mixins", "counterSet"),
		},
		"device-counter-consumption-mixin": {
			sliceCreateFunc: func(counters map[string]resourceapi.Counter) *resourceapi.ResourceSlice {
				slice := testResourceSlice(goodName, goodName, goodName, 1)
				slice.Spec.Mixins = &resourceapi.ResourceSliceMixins{
					DeviceCounterConsumption: []resourceapi.DeviceCounterConsumptionMixin{
						{
							Name:     goodName,
							Counters: counters,
						},
					},
				}
				return slice
			},
			basePath: field.NewPath("spec", "mixins", "deviceCounterConsumption"),
		},
	}

	for fieldName, fieldConfig := range fieldConfigs {
		for name, tc := range testCases {
			t.Run(fmt.Sprintf("%s/%s", fieldName, name), func(t *testing.T) {
				slice := fieldConfig.sliceCreateFunc(tc.counters)
				errs := ValidateResourceSlice(slice)
				var expectedFailures field.ErrorList
				if tc.expectedFailuresFunc != nil {
					expectedFailures = tc.expectedFailuresFunc(fieldConfig.basePath)
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
			wantFailures:     field.ErrorList{field.Invalid(field.NewPath("spec", "nodeName"), ptr.To(name+"-updated"), "field is immutable")},
			oldResourceSlice: validResourceSlice,
			update: func(slice *resourceapi.ResourceSlice) *resourceapi.ResourceSlice {
				slice.Spec.NodeName = ptr.To(*slice.Spec.NodeName + "-updated")
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
				slice.Spec.NodeName = nil
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

func createSharedCounters(count int) []resourceapi.CounterSet {
	sharedCounters := make([]resourceapi.CounterSet, count)
	for i := 0; i < count; i++ {
		sharedCounters[i] = resourceapi.CounterSet{
			Name:     fmt.Sprintf("counterset-%d", i),
			Counters: testCounter(),
		}
	}
	return sharedCounters
}

func createConsumesCountersFromCounterSets(counterSets ...resourceapi.CounterSet) []resourceapi.DeviceCounterConsumption {
	var deviceCounterConsumptions []resourceapi.DeviceCounterConsumption
	for _, counterSet := range counterSets {
		counters := make(map[string]resourceapi.Counter)
		for name, counter := range counterSet.Counters {
			counters[name] = counter
		}
		deviceCounterConsumptions = append(deviceCounterConsumptions, resourceapi.DeviceCounterConsumption{
			CounterSet: counterSet.Name,
			Counters:   counters,
		})
	}
	return deviceCounterConsumptions
}

func createDeviceMixins(count int) []resourceapi.DeviceMixin {
	deviceMixins := make([]resourceapi.DeviceMixin, count)
	for i := 0; i < count; i++ {
		deviceMixins[i] = resourceapi.DeviceMixin{
			Name:       fmt.Sprintf("device-mixin-%d", i),
			Attributes: testAttributes(),
			Capacity:   testCapacity(),
		}
	}
	return deviceMixins
}

func createDeviceMixinRefs(mixins ...resourceapi.DeviceMixin) []string {
	deviceMixinRefs := make([]string, len(mixins))
	for i, mixin := range mixins {
		deviceMixinRefs[i] = mixin.Name
	}
	return deviceMixinRefs
}

func createCounterSetMixins(count int) []resourceapi.CounterSetMixin {
	counterSetMixins := make([]resourceapi.CounterSetMixin, count)
	for i := 0; i < count; i++ {
		counterSetMixins[i] = resourceapi.CounterSetMixin{
			Name:     fmt.Sprintf("coumter-set-mixin-%d", i),
			Counters: testCounter(),
		}
	}
	return counterSetMixins
}

func createCounterSetMixinRefs(mixins ...resourceapi.CounterSetMixin) []string {
	counterSetMixinRefs := make([]string, len(mixins))
	for i, mixin := range mixins {
		counterSetMixinRefs[i] = mixin.Name
	}
	return counterSetMixinRefs
}

func createDeviceCounterConsumptionMixins(count int) []resourceapi.DeviceCounterConsumptionMixin {
	deviceCounterConsumptionMixins := make([]resourceapi.DeviceCounterConsumptionMixin, count)
	for i := 0; i < count; i++ {
		deviceCounterConsumptionMixins[i] = resourceapi.DeviceCounterConsumptionMixin{
			Name:     fmt.Sprintf("device-counter-consumption-mixin-%d", i),
			Counters: testCounter(),
		}
	}
	return deviceCounterConsumptionMixins
}

func createDeviceCounterConsumptionMixinsFromCounterSets(counterSets ...resourceapi.CounterSet) []resourceapi.DeviceCounterConsumptionMixin {
	var deviceCounterConsumptionMixins []resourceapi.DeviceCounterConsumptionMixin
	for i, counterSet := range counterSets {
		counters := make(map[string]resourceapi.Counter)
		for name, counter := range counterSet.Counters {
			counters[name] = counter
		}
		deviceCounterConsumptionMixins = append(deviceCounterConsumptionMixins, resourceapi.DeviceCounterConsumptionMixin{
			Name:     fmt.Sprintf("device-counter-consumption-mixin-%d", i),
			Counters: counters,
		})
	}
	return deviceCounterConsumptionMixins
}

func createDeviceCounterConsumptionMixinRefs(mixins ...resourceapi.DeviceCounterConsumptionMixin) []string {
	deviceCounterConsumptionMixinRefs := make([]string, len(mixins))
	for i, mixin := range mixins {
		deviceCounterConsumptionMixinRefs[i] = mixin.Name
	}
	return deviceCounterConsumptionMixinRefs
}
