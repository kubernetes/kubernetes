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

package resourceslice

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	v1 "k8s.io/api/core/v1"
	k8sresource "k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/apimachinery/pkg/util/version"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	apiserverfeatures "k8s.io/apiserver/pkg/features"
	"k8s.io/apiserver/pkg/registry/rest"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/kubernetes/pkg/apis/resource"
	_ "k8s.io/kubernetes/pkg/apis/resource/install"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/utils/ptr"
)

var slice = &resource.ResourceSlice{
	ObjectMeta: metav1.ObjectMeta{
		Name: "valid-resource-slice",
	},
	Spec: resource.ResourceSliceSpec{
		NodeName: ptr.To("valid-node-name"),
		Driver:   "testdriver.example.com",
		Pool: resource.ResourcePool{
			Name:               "valid-pool-name",
			ResourceSliceCount: 1,
		},
		Devices: []resource.Device{{
			Name: "device-0",
		}},
	},
}

var sliceWithDeviceTaints = func() *resource.ResourceSlice {
	slice := slice.DeepCopy()
	slice.Spec.Devices[0].Taints = []resource.DeviceTaint{{
		Key:    "example.com/tainted",
		Effect: resource.DeviceTaintEffectNoSchedule,
	}}
	return slice
}()

var sliceWithBindingConditions = func() *resource.ResourceSlice {
	slice := slice.DeepCopy()
	slice.Spec.Devices[0].BindingConditions = []string{"cond1"}
	slice.Spec.Devices[0].BindingFailureConditions = []string{"fail1"}
	slice.Spec.Devices[0].BindsToNode = ptr.To(true)
	return slice
}()

var sliceWithPartitionableDevicesPerDeviceNodeSelection = &resource.ResourceSlice{
	ObjectMeta: metav1.ObjectMeta{
		Name: "valid-resource-slice",
	},
	Spec: resource.ResourceSliceSpec{
		PerDeviceNodeSelection: func() *bool {
			r := true
			return &r
		}(),
		Driver: "testdriver.example.com",
		Pool: resource.ResourcePool{
			Name:               "valid-pool-name",
			ResourceSliceCount: 1,
			Generation:         1,
		},
		Devices: []resource.Device{
			{
				Name: "device",
				NodeName: func() *string {
					r := "valid-node-name"
					return &r
				}(),
			},
		},
	},
}

var sliceWithPartitionableDevicesConsumesCounters = &resource.ResourceSlice{
	ObjectMeta: metav1.ObjectMeta{
		Name: "valid-resource-slice",
	},
	Spec: resource.ResourceSliceSpec{
		NodeName: ptr.To("valid-node-name"),
		Driver:   "testdriver.example.com",
		Pool: resource.ResourcePool{
			Name:               "valid-pool-name",
			ResourceSliceCount: 1,
			Generation:         1,
		},
		Devices: []resource.Device{
			{
				Name: "device",
				ConsumesCounters: []resource.DeviceCounterConsumption{
					{
						CounterSet: "pool-1",
						Counters: map[string]resource.CounterRequirement{
							"memory": {
								Value: new(k8sresource.MustParse("40Gi")),
							},
						},
					},
				},
				Attributes: map[resource.QualifiedName]resource.DeviceAttribute{
					resource.QualifiedName("version"): {
						StringValue: func() *string {
							v := "v1"
							return &v
						}(),
					},
				},
				Capacity: map[resource.QualifiedName]resource.DeviceCapacity{
					resource.QualifiedName("memory"): {
						Value: k8sresource.MustParse("40Gi"),
					},
				},
			},
		},
	},
}

var sliceWithPartitionableDevicesSharedCounters = &resource.ResourceSlice{
	ObjectMeta: metav1.ObjectMeta{
		Name: "valid-resource-slice",
	},
	Spec: resource.ResourceSliceSpec{
		NodeName: ptr.To("valid-node-name"),
		Driver:   "testdriver.example.com",
		Pool: resource.ResourcePool{
			Name:               "valid-pool-name",
			ResourceSliceCount: 1,
			Generation:         1,
		},
		SharedCounters: []resource.CounterSet{
			{
				Name: "pool-1",
				Counters: map[string]resource.Counter{
					"memory": {
						Value: k8sresource.MustParse("40Gi"),
					},
				},
			},
		},
	},
}

var sliceWithSharedConsumableSharedCounters = func() *resource.ResourceSlice {
	obj := sliceWithPartitionableDevicesSharedCounters.DeepCopy()
	defaultVal := k8sresource.MustParse("1Gi")
	minVal := k8sresource.MustParse("1Gi")
	obj.Spec.SharedCounters[0].Counters["memory"] = resource.Counter{
		Value: k8sresource.MustParse("40Gi"),
		ConsumptionPolicy: &resource.CapacityRequestPolicy{
			Default: &defaultVal,
			ValidRange: &resource.CapacityRequestPolicyRange{
				Min: &minVal,
			},
		},
	}
	return obj
}()

var sliceWithCapacity = func() *resource.ResourceSlice {
	obj := slice.DeepCopy()
	obj.Spec.Devices[0].Capacity = map[resource.QualifiedName]resource.DeviceCapacity{
		"memory": {
			Value: k8sresource.MustParse("40Gi"),
		},
	}
	return obj
}()

var sliceWithCompatibilityGroups = func() *resource.ResourceSlice {
	obj := sliceWithPartitionableDevicesConsumesCounters.DeepCopy()
	obj.Spec.Devices[0].ConsumesCounters[0].CompatibilityGroups = []string{"mig"}
	return obj
}()

var sliceWithSharedConsumableConsumesCounters = func() *resource.ResourceSlice {
	obj := sliceWithPartitionableDevicesConsumesCounters.DeepCopy()
	obj.Spec.Devices[0].ConsumesCounters[0].Counters["memory"] = resource.CounterRequirement{
		ValueFrom: &resource.CounterValueFrom{
			CapacityName: "dra.example.com/memory",
		},
	}
	return obj
}()

var sliceWithConsumableCapacity = func() *resource.ResourceSlice {
	obj := sliceWithCapacity.DeepCopy()
	obj.Spec.Devices[0].AllowMultipleAllocations = ptr.To(true)
	obj.Spec.Devices[0].Capacity["memory"] =
		resource.DeviceCapacity{
			Value: k8sresource.MustParse("40Gi"),
			RequestPolicy: &resource.CapacityRequestPolicy{
				Default: ptr.To(k8sresource.MustParse("1Gi")),
				ValidRange: &resource.CapacityRequestPolicyRange{
					Min: ptr.To(k8sresource.MustParse("1Gi")),
				},
			},
		}
	return obj
}()

var sliceWithNodeAllocatableResources = func() *resource.ResourceSlice {
	obj := slice.DeepCopy()
	instanceQuantity := k8sresource.MustParse("1")
	obj.Spec.Devices[0].NodeAllocatableResources = map[v1.ResourceName]resource.NodeAllocatableResource{
		v1.ResourceCPU: {
			Mapping: &resource.NodeAllocatableMapping{
				DeviceMultiplier: &instanceQuantity,
			},
		},
	}
	return obj
}()

var sliceWithListTypeAttributes = func() *resource.ResourceSlice {
	obj := slice.DeepCopy()
	obj.Spec.Devices[0].Attributes = map[resource.QualifiedName]resource.DeviceAttribute{
		resource.QualifiedName("list_attribute"): {
			StringValues: []string{"value1", "value2"},
		},
	}
	return obj
}()

// The attribute is only permitted on a slice whose devices consume counters,
// and each of them must carry it as a string. The bare "version" key qualifies
// with the driver's own domain.
var sliceWithPartitionTypeAttribute = func() *resource.ResourceSlice {
	obj := sliceWithPartitionableDevicesConsumesCounters.DeepCopy()
	obj.Spec.PartitionTypeAttribute = ptr.To(resource.FullyQualifiedName("testdriver.example.com/version"))
	return obj
}()

var sliceWithSkipNodeOperations = func() *resource.ResourceSlice {
	obj := slice.DeepCopy()
	obj.Spec.SkipNodeOperations = []resource.SkipNodeOperation{resource.SkipNodeOperationAll}
	return obj
}()

func TestResourceSliceStrategy(t *testing.T) {
	if Strategy.NamespaceScoped() {
		t.Errorf("ResourceSlice must not be namespace scoped")
	}
	if Strategy.AllowCreateOnUpdate(context.Background()) {
		t.Errorf("ResourceSlice should not allow create on update")
	}
}

func TestResourceSliceStrategyCreate(t *testing.T) {
	ctx := genericapirequest.NewDefaultContext()
	testCases := map[string]struct {
		obj                     *resource.ResourceSlice
		featureOverrides        featuregatetesting.FeatureOverrides
		emulatedVersion         string
		expectedValidationError bool
		expectObj               *resource.ResourceSlice
	}{
		"simple": {
			obj: slice,
			expectObj: func() *resource.ResourceSlice {
				obj := slice.DeepCopy()
				obj.ObjectMeta.Generation = 1
				return obj
			}(),
		},
		"validation error": {
			obj: func() *resource.ResourceSlice {
				obj := slice.DeepCopy()
				obj.Name = "%#@$%$"
				return obj
			}(),
			expectedValidationError: true,
		},
		"drop-fields-device-taints": {
			obj:              sliceWithDeviceTaints,
			featureOverrides: featuregatetesting.FeatureOverrides{features.DRADeviceTaints: false},
			emulatedVersion:  "1.35",
			expectObj: func() *resource.ResourceSlice {
				obj := slice.DeepCopy()
				obj.Generation = 1
				return obj
			}(),
		},
		"keep-fields-device-taints": {
			obj:              sliceWithDeviceTaints,
			featureOverrides: featuregatetesting.FeatureOverrides{features.DRADeviceTaints: true},
			expectObj: func() *resource.ResourceSlice {
				obj := sliceWithDeviceTaints.DeepCopy()
				obj.ObjectMeta.Generation = 1
				return obj
			}(),
		},
		"drop-fields-partitionable-devices-with-consumes-counters": {
			obj: func() *resource.ResourceSlice {
				obj := sliceWithPartitionableDevicesConsumesCounters.DeepCopy()
				return obj
			}(),
			featureOverrides: featuregatetesting.FeatureOverrides{features.DRAPartitionableDevices: false},
			emulatedVersion:  "1.37",
			expectObj: func() *resource.ResourceSlice {
				obj := sliceWithPartitionableDevicesConsumesCounters.DeepCopy()
				obj.ObjectMeta.Generation = 1
				obj.Spec.Devices[0].ConsumesCounters = nil
				return obj
			}(),
		},
		// This should return a validation error since the slice will not
		// have a node selector after the perDeviceNodeSelection field got
		// dropped.
		"drop-fields-partitionable-devices-with-per-device-node-selection": {
			obj:                     sliceWithPartitionableDevicesPerDeviceNodeSelection,
			featureOverrides:        featuregatetesting.FeatureOverrides{features.DRAPartitionableDevices: false},
			emulatedVersion:         "1.37",
			expectedValidationError: true,
		},
		"drop-fields-partitionable-devices-with-shared-counters": {
			obj: func() *resource.ResourceSlice {
				obj := sliceWithPartitionableDevicesSharedCounters.DeepCopy()
				return obj
			}(),
			featureOverrides: featuregatetesting.FeatureOverrides{features.DRAPartitionableDevices: false},
			emulatedVersion:  "1.37",
			expectObj: func() *resource.ResourceSlice {
				obj := sliceWithPartitionableDevicesSharedCounters.DeepCopy()
				obj.ObjectMeta.Generation = 1
				obj.Spec.SharedCounters = nil
				return obj
			}(),
		},
		"keep-fields-partitionable-devices-with-consumes-counters": {
			obj:              sliceWithPartitionableDevicesConsumesCounters,
			featureOverrides: featuregatetesting.FeatureOverrides{features.DRAPartitionableDevices: true},
			expectObj: func() *resource.ResourceSlice {
				obj := sliceWithPartitionableDevicesConsumesCounters.DeepCopy()
				obj.Generation = 1
				return obj
			}(),
		},
		"keep-fields-partitionable-devices-with-per-device-node-selection": {
			obj:              sliceWithPartitionableDevicesPerDeviceNodeSelection,
			featureOverrides: featuregatetesting.FeatureOverrides{features.DRAPartitionableDevices: true},
			expectObj: func() *resource.ResourceSlice {
				obj := sliceWithPartitionableDevicesPerDeviceNodeSelection.DeepCopy()
				obj.Generation = 1
				return obj
			}(),
		},
		"keep-fields-partitionable-devices-with-shared-counters": {
			obj:              sliceWithPartitionableDevicesSharedCounters,
			featureOverrides: featuregatetesting.FeatureOverrides{features.DRAPartitionableDevices: true},
			expectObj: func() *resource.ResourceSlice {
				obj := sliceWithPartitionableDevicesSharedCounters.DeepCopy()
				obj.Generation = 1
				return obj
			}(),
		},
		"drop-fields-shared-consumable-capacity-shared-counters": {
			obj:              sliceWithSharedConsumableSharedCounters,
			featureOverrides: featuregatetesting.FeatureOverrides{features.DRAPartitionableDevices: true, features.DRASharedConsumableCapacity: false},
			expectObj: func() *resource.ResourceSlice {
				obj := sliceWithPartitionableDevicesSharedCounters.DeepCopy()
				obj.Generation = 1
				return obj
			}(),
		},
		"keep-fields-shared-consumable-capacity-shared-counters": {
			obj:              sliceWithSharedConsumableSharedCounters,
			featureOverrides: featuregatetesting.FeatureOverrides{features.DRAPartitionableDevices: true, features.DRAConsumableCapacity: true, features.DRASharedConsumableCapacity: true},
			expectObj: func() *resource.ResourceSlice {
				obj := sliceWithSharedConsumableSharedCounters.DeepCopy()
				obj.Generation = 1
				return obj
			}(),
		},
		"drop-fields-shared-consumable-capacity-consumes-counters": {
			obj:              sliceWithSharedConsumableConsumesCounters,
			featureOverrides: featuregatetesting.FeatureOverrides{features.DRAPartitionableDevices: true, features.DRASharedConsumableCapacity: false},
			expectObj: func() *resource.ResourceSlice {
				obj := sliceWithSharedConsumableConsumesCounters.DeepCopy()
				obj.Spec.Devices[0].ConsumesCounters[0].Counters["memory"] = resource.CounterRequirement{}
				obj.Generation = 1
				return obj
			}(),
		},
		"keep-fields-shared-consumable-capacity-consumes-counters": {
			obj:              sliceWithSharedConsumableConsumesCounters,
			featureOverrides: featuregatetesting.FeatureOverrides{features.DRAPartitionableDevices: true, features.DRAConsumableCapacity: true, features.DRASharedConsumableCapacity: true},
			expectObj: func() *resource.ResourceSlice {
				obj := sliceWithSharedConsumableConsumesCounters.DeepCopy()
				obj.Generation = 1
				return obj
			}(),
		},
		"drop-fields-binding-conditions": {
			obj:              sliceWithBindingConditions,
			emulatedVersion:  "1.35",
			featureOverrides: featuregatetesting.FeatureOverrides{features.DRADeviceBindingConditions: false, features.DRAResourceClaimDeviceStatus: false},
			expectObj: func() *resource.ResourceSlice {
				obj := slice.DeepCopy()
				obj.Generation = 1
				return obj
			}(),
		},
		"drop-fields-binding-conditions-with-binding-conditions": {
			obj:              sliceWithBindingConditions,
			featureOverrides: featuregatetesting.FeatureOverrides{features.DRADeviceBindingConditions: false, features.DRAResourceClaimDeviceStatus: true},
			emulatedVersion:  "1.37",
			expectObj: func() *resource.ResourceSlice {
				obj := slice.DeepCopy()
				obj.Generation = 1
				return obj
			}(),
		},
		"keep-fields-binding-conditions": {
			obj:              sliceWithBindingConditions,
			featureOverrides: featuregatetesting.FeatureOverrides{features.DRADeviceBindingConditions: true, features.DRAResourceClaimDeviceStatus: true},
			expectObj: func() *resource.ResourceSlice {
				obj := sliceWithBindingConditions.DeepCopy()
				obj.Generation = 1
				return obj
			}(),
		},
		"keep-fields-consumable-capacity": {
			obj:              sliceWithConsumableCapacity,
			featureOverrides: featuregatetesting.FeatureOverrides{features.DRAConsumableCapacity: true},
			expectObj: func() *resource.ResourceSlice {
				obj := sliceWithConsumableCapacity.DeepCopy()
				obj.Generation = 1
				return obj
			}(),
		},
		"drop-fields-consumable-capacity-disabled-feature": {
			obj:              sliceWithConsumableCapacity,
			featureOverrides: featuregatetesting.FeatureOverrides{features.DRAConsumableCapacity: false},
			expectObj: func() *resource.ResourceSlice {
				obj := sliceWithCapacity.DeepCopy()
				obj.Generation = 1
				return obj
			}(),
		},
		"keep-fields-node-allocatable-dra-claims": {
			obj:              sliceWithNodeAllocatableResources,
			featureOverrides: featuregatetesting.FeatureOverrides{features.DRANodeAllocatableResources: true},
			expectObj: func() *resource.ResourceSlice {
				obj := sliceWithNodeAllocatableResources.DeepCopy()
				obj.Generation = 1
				return obj
			}(),
		},
		"drop-fields-node-allocatable-dra-claims-disabled-feature": {
			obj:              sliceWithNodeAllocatableResources,
			featureOverrides: featuregatetesting.FeatureOverrides{features.DRANodeAllocatableResources: false},
			expectObj: func() *resource.ResourceSlice {
				obj := slice.DeepCopy()
				obj.Generation = 1
				return obj
			}(),
		},
		"keep-fields-list-type-attributes": {
			obj:              sliceWithListTypeAttributes,
			featureOverrides: featuregatetesting.FeatureOverrides{features.DRAListTypeAttributes: true},
			expectObj: func() *resource.ResourceSlice {
				obj := sliceWithListTypeAttributes.DeepCopy()
				obj.Generation = 1
				return obj
			}(),
		},
		"keep-fields-list-type-attributes-by-default": {
			obj: sliceWithListTypeAttributes,
			expectObj: func() *resource.ResourceSlice {
				obj := sliceWithListTypeAttributes.DeepCopy()
				obj.Generation = 1
				return obj
			}(),
		},
		"drop-fields-list-type-attributes-1.37-default": {
			obj:                     sliceWithListTypeAttributes,
			emulatedVersion:         "1.37",
			expectedValidationError: true,
		},
		"drop-fields-list-type-attributes": {
			obj:                     sliceWithListTypeAttributes,
			featureOverrides:        featuregatetesting.FeatureOverrides{features.DRAListTypeAttributes: false},
			expectedValidationError: true,
		},
		"keep-fields-partition-type-attribute": {
			obj:              sliceWithPartitionTypeAttribute,
			featureOverrides: featuregatetesting.FeatureOverrides{features.DRAPartitionableDevices: true, features.DRAResourcePoolStatus: true, features.DRAPartitionableDevicesType: true},
			expectObj: func() *resource.ResourceSlice {
				obj := sliceWithPartitionTypeAttribute.DeepCopy()
				obj.Generation = 1
				return obj
			}(),
		},
		"drop-fields-partition-type-attribute": {
			obj:              sliceWithPartitionTypeAttribute,
			featureOverrides: featuregatetesting.FeatureOverrides{features.DRAPartitionableDevices: true, features.DRAResourcePoolStatus: true, features.DRAPartitionableDevicesType: false},
			expectObj: func() *resource.ResourceSlice {
				obj := sliceWithPartitionableDevicesConsumesCounters.DeepCopy()
				obj.Generation = 1
				return obj
			}(),
		},
		"drop-fields-optional-node-operations": {
			obj:              sliceWithSkipNodeOperations,
			featureOverrides: featuregatetesting.FeatureOverrides{features.DRAOptionalNodeOperations: false},
			expectObj: func() *resource.ResourceSlice {
				obj := slice.DeepCopy()
				obj.Generation = 1
				return obj
			}(),
		},
		"keep-fields-optional-node-operations": {
			obj:              sliceWithSkipNodeOperations,
			featureOverrides: featuregatetesting.FeatureOverrides{features.DRAOptionalNodeOperations: true},
			expectObj: func() *resource.ResourceSlice {
				obj := sliceWithSkipNodeOperations.DeepCopy()
				obj.Generation = 1
				return obj
			}(),
		},
		"keep-fields-compatibility-groups": {
			obj:              sliceWithCompatibilityGroups,
			featureOverrides: featuregatetesting.FeatureOverrides{features.DRAPartitionableDevices: true, features.DRADeviceCompatibilityGroups: true},
			expectObj: func() *resource.ResourceSlice {
				obj := sliceWithCompatibilityGroups.DeepCopy()
				obj.Generation = 1
				return obj
			}(),
		},
		"drop-fields-compatibility-groups-disabled-feature": {
			obj:              sliceWithCompatibilityGroups,
			featureOverrides: featuregatetesting.FeatureOverrides{features.DRAPartitionableDevices: true, features.DRADeviceCompatibilityGroups: false},
			expectObj: func() *resource.ResourceSlice {
				obj := sliceWithPartitionableDevicesConsumesCounters.DeepCopy()
				obj.Generation = 1
				return obj
			}(),
		},
	}

	for name, tc := range testCases {
		t.Run(name, func(t *testing.T) {
			if tc.emulatedVersion != "" {
				featuregatetesting.SetFeatureGateEmulationVersionDuringTest(t, utilfeature.DefaultFeatureGate, version.MustParse(tc.emulatedVersion))
			}
			featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, tc.featureOverrides)

			obj := tc.obj.DeepCopy()

			Strategy.PrepareForCreate(ctx, obj)
			if errs := Strategy.Validate(ctx, obj); len(errs) != 0 {
				if !tc.expectedValidationError {
					t.Fatalf("unexpected validation errors: %q", errs)
				}
				return
			}
			if warnings := Strategy.WarningsOnCreate(ctx, obj); len(warnings) != 0 {
				t.Fatalf("unexpected warnings: %q", warnings)
			}
			Strategy.Canonicalize(obj)
			assert.Equal(t, tc.expectObj, obj)
		})
	}
}

func TestResourceSliceStrategyUpdate(t *testing.T) {
	ctx := genericapirequest.NewDefaultContext()

	testcases := map[string]struct {
		oldObj                *resource.ResourceSlice
		newObj                *resource.ResourceSlice
		featureOverrides      featuregatetesting.FeatureOverrides
		emulatedVersion       string
		expectValidationError bool
		expectObj             *resource.ResourceSlice
	}{
		"no-changes-okay": {
			oldObj: slice,
			newObj: func() *resource.ResourceSlice {
				obj := slice.DeepCopy()
				obj.ResourceVersion = "4"
				return obj
			}(),
			expectObj: func() *resource.ResourceSlice {
				obj := slice.DeepCopy()
				obj.ResourceVersion = "4"
				return obj
			}(),
		},
		"name-change-not-allowed": {
			oldObj: slice,
			newObj: func() *resource.ResourceSlice {
				obj := slice.DeepCopy()
				obj.Name = "valid-slice-2"
				obj.ResourceVersion = "4"
				return obj
			}(),
			expectValidationError: true,
		},
		"drop-fields-device-taints": {
			oldObj: slice,
			newObj: func() *resource.ResourceSlice {
				obj := slice.DeepCopy()
				obj.ResourceVersion = "4"
				return obj
			}(),
			featureOverrides: featuregatetesting.FeatureOverrides{features.DRADeviceTaints: false},
			emulatedVersion:  "1.35",
			expectObj: func() *resource.ResourceSlice {
				obj := slice.DeepCopy()
				obj.ResourceVersion = "4"
				return obj
			}(),
		},
		"keep-fields-device-taints": {
			oldObj: slice,
			newObj: func() *resource.ResourceSlice {
				obj := sliceWithDeviceTaints.DeepCopy()
				obj.ResourceVersion = "4"
				return obj
			}(),
			featureOverrides: featuregatetesting.FeatureOverrides{features.DRADeviceTaints: true},
			expectObj: func() *resource.ResourceSlice {
				obj := sliceWithDeviceTaints.DeepCopy()
				obj.ResourceVersion = "4"
				obj.Generation = 1
				return obj
			}(),
		},
		"keep-existing-fields-device-taints": {
			oldObj: sliceWithDeviceTaints,
			newObj: func() *resource.ResourceSlice {
				obj := sliceWithDeviceTaints.DeepCopy()
				obj.ResourceVersion = "4"
				return obj
			}(),
			featureOverrides: featuregatetesting.FeatureOverrides{features.DRADeviceTaints: true},
			expectObj: func() *resource.ResourceSlice {
				obj := sliceWithDeviceTaints.DeepCopy()
				obj.ResourceVersion = "4"
				return obj
			}(),
		},
		"keep-existing-fields-device-taints-disabled-feature": {
			oldObj: sliceWithDeviceTaints,
			newObj: func() *resource.ResourceSlice {
				obj := sliceWithDeviceTaints.DeepCopy()
				obj.ResourceVersion = "4"
				return obj
			}(),
			featureOverrides: featuregatetesting.FeatureOverrides{features.DRADeviceTaints: false},
			emulatedVersion:  "1.35",
			expectObj: func() *resource.ResourceSlice {
				obj := sliceWithDeviceTaints.DeepCopy()
				obj.ResourceVersion = "4"
				return obj
			}(),
		},
		"keep-existing-fields-partition-type-attribute-disabled-feature": {
			oldObj: sliceWithPartitionTypeAttribute,
			newObj: func() *resource.ResourceSlice {
				obj := sliceWithPartitionTypeAttribute.DeepCopy()
				obj.ResourceVersion = "4"
				return obj
			}(),
			featureOverrides: featuregatetesting.FeatureOverrides{features.DRAPartitionableDevices: true, features.DRAResourcePoolStatus: true, features.DRAPartitionableDevicesType: false},
			expectObj: func() *resource.ResourceSlice {
				obj := sliceWithPartitionTypeAttribute.DeepCopy()
				obj.ResourceVersion = "4"
				return obj
			}(),
		},
		"drop-fields-partitionable-devices-with-consumes-counters": {
			oldObj: slice,
			newObj: func() *resource.ResourceSlice {
				obj := sliceWithPartitionableDevicesConsumesCounters.DeepCopy()
				obj.ResourceVersion = "4"
				return obj
			}(),
			featureOverrides: featuregatetesting.FeatureOverrides{features.DRAPartitionableDevices: false},
			emulatedVersion:  "1.37",
			expectObj: func() *resource.ResourceSlice {
				obj := sliceWithPartitionableDevicesConsumesCounters.DeepCopy()
				obj.ResourceVersion = "4"
				obj.Generation = 1
				obj.Spec.Devices[0].ConsumesCounters = nil
				return obj
			}(),
		},
		"drop-fields-partitionable-devices-with-per-device-node-selection": {
			oldObj: slice,
			newObj: func() *resource.ResourceSlice {
				obj := sliceWithPartitionableDevicesPerDeviceNodeSelection.DeepCopy()
				obj.ResourceVersion = "4"
				return obj
			}(),
			featureOverrides:      featuregatetesting.FeatureOverrides{features.DRAPartitionableDevices: false},
			emulatedVersion:       "1.37",
			expectValidationError: true,
		},
		"drop-fields-partitionable-devices-with-shared-counters": {
			oldObj: slice,
			newObj: func() *resource.ResourceSlice {
				obj := sliceWithPartitionableDevicesSharedCounters.DeepCopy()
				obj.ResourceVersion = "4"
				return obj
			}(),
			featureOverrides: featuregatetesting.FeatureOverrides{features.DRAPartitionableDevices: false},
			emulatedVersion:  "1.37",
			expectObj: func() *resource.ResourceSlice {
				obj := sliceWithPartitionableDevicesSharedCounters.DeepCopy()
				obj.ResourceVersion = "4"
				obj.Generation = 1
				obj.Spec.SharedCounters = nil
				return obj
			}(),
		},
		"keep-fields-partitionable-devices-with-consumes-counters": {
			oldObj: slice,
			newObj: func() *resource.ResourceSlice {
				obj := sliceWithPartitionableDevicesConsumesCounters.DeepCopy()
				obj.ResourceVersion = "4"
				return obj
			}(),
			featureOverrides: featuregatetesting.FeatureOverrides{features.DRAPartitionableDevices: true},
			expectObj: func() *resource.ResourceSlice {
				obj := sliceWithPartitionableDevicesConsumesCounters.DeepCopy()
				obj.ResourceVersion = "4"
				obj.Generation = 1
				return obj
			}(),
		},
		"keep-fields-partitionable-devices-with-per-device-node-selection": {
			oldObj: slice,
			newObj: func() *resource.ResourceSlice {
				obj := sliceWithPartitionableDevicesPerDeviceNodeSelection.DeepCopy()
				obj.ResourceVersion = "4"
				return obj
			}(),
			featureOverrides:      featuregatetesting.FeatureOverrides{features.DRAPartitionableDevices: true},
			expectValidationError: true, // Spec.NodeName is immutable.
		},
		"keep-fields-partitionable-devices-with-shared-counters": {
			oldObj: slice,
			newObj: func() *resource.ResourceSlice {
				obj := sliceWithPartitionableDevicesSharedCounters.DeepCopy()
				obj.ResourceVersion = "4"
				return obj
			}(),
			featureOverrides: featuregatetesting.FeatureOverrides{features.DRAPartitionableDevices: true},
			expectObj: func() *resource.ResourceSlice {
				obj := sliceWithPartitionableDevicesSharedCounters.DeepCopy()
				obj.ResourceVersion = "4"
				obj.Generation = 1
				return obj
			}(),
		},
		"keep-existing-fields-partitionable-devices-with-consumes-counters": {
			oldObj: sliceWithPartitionableDevicesConsumesCounters,
			newObj: func() *resource.ResourceSlice {
				obj := sliceWithPartitionableDevicesConsumesCounters.DeepCopy()
				obj.ResourceVersion = "4"
				return obj
			}(),
			featureOverrides: featuregatetesting.FeatureOverrides{features.DRAPartitionableDevices: true},
			expectObj: func() *resource.ResourceSlice {
				obj := sliceWithPartitionableDevicesConsumesCounters.DeepCopy()
				obj.ResourceVersion = "4"
				return obj
			}(),
		},
		"keep-existing-fields-partitionable-devices-with-per-device-node-selection": {
			oldObj: sliceWithPartitionableDevicesPerDeviceNodeSelection,
			newObj: func() *resource.ResourceSlice {
				obj := sliceWithPartitionableDevicesPerDeviceNodeSelection.DeepCopy()
				obj.ResourceVersion = "4"
				return obj
			}(),
			featureOverrides: featuregatetesting.FeatureOverrides{features.DRAPartitionableDevices: true},
			expectObj: func() *resource.ResourceSlice {
				obj := sliceWithPartitionableDevicesPerDeviceNodeSelection.DeepCopy()
				obj.ResourceVersion = "4"
				return obj
			}(),
		},
		"keep-existing-fields-partitionable-devices-with-shared-counters": {
			oldObj: sliceWithPartitionableDevicesSharedCounters,
			newObj: func() *resource.ResourceSlice {
				obj := sliceWithPartitionableDevicesSharedCounters.DeepCopy()
				obj.ResourceVersion = "4"
				return obj
			}(),
			featureOverrides: featuregatetesting.FeatureOverrides{features.DRAPartitionableDevices: true},
			expectObj: func() *resource.ResourceSlice {
				obj := sliceWithPartitionableDevicesSharedCounters.DeepCopy()
				obj.ResourceVersion = "4"
				return obj
			}(),
		},
		"keep-existing-fields-partitionable-devices-consumes-counters-disabled-feature": {
			oldObj: sliceWithPartitionableDevicesConsumesCounters,
			newObj: func() *resource.ResourceSlice {
				obj := sliceWithPartitionableDevicesConsumesCounters.DeepCopy()
				obj.ResourceVersion = "4"
				return obj
			}(),
			featureOverrides: featuregatetesting.FeatureOverrides{features.DRAPartitionableDevices: false},
			emulatedVersion:  "1.37",
			expectObj: func() *resource.ResourceSlice {
				obj := sliceWithPartitionableDevicesConsumesCounters.DeepCopy()
				obj.ResourceVersion = "4"
				return obj
			}(),
		},
		"keep-existing-fields-partitionable-devices-per-device-node-selection-disabled-feature": {
			oldObj: sliceWithPartitionableDevicesPerDeviceNodeSelection,
			newObj: func() *resource.ResourceSlice {
				obj := sliceWithPartitionableDevicesPerDeviceNodeSelection.DeepCopy()
				obj.ResourceVersion = "4"
				return obj
			}(),
			featureOverrides: featuregatetesting.FeatureOverrides{features.DRAPartitionableDevices: false},
			emulatedVersion:  "1.37",
			expectObj: func() *resource.ResourceSlice {
				obj := sliceWithPartitionableDevicesPerDeviceNodeSelection.DeepCopy()
				obj.ResourceVersion = "4"
				return obj
			}(),
		},
		"keep-existing-fields-partitionable-devices-shared-counters-disabled-feature": {
			oldObj: sliceWithPartitionableDevicesSharedCounters,
			newObj: func() *resource.ResourceSlice {
				obj := sliceWithPartitionableDevicesSharedCounters.DeepCopy()
				obj.ResourceVersion = "4"
				return obj
			}(),
			featureOverrides: featuregatetesting.FeatureOverrides{features.DRAPartitionableDevices: false},
			emulatedVersion:  "1.37",
			expectObj: func() *resource.ResourceSlice {
				obj := sliceWithPartitionableDevicesSharedCounters.DeepCopy()
				obj.ResourceVersion = "4"
				return obj
			}(),
		},
		"drop-fields-shared-consumable-capacity-shared-counters-disabled-feature": {
			oldObj: sliceWithPartitionableDevicesSharedCounters,
			newObj: func() *resource.ResourceSlice {
				obj := sliceWithSharedConsumableSharedCounters.DeepCopy()
				obj.ResourceVersion = "4"
				return obj
			}(),
			featureOverrides: featuregatetesting.FeatureOverrides{features.DRAPartitionableDevices: true, features.DRASharedConsumableCapacity: false},
			expectObj: func() *resource.ResourceSlice {
				obj := sliceWithPartitionableDevicesSharedCounters.DeepCopy()
				obj.ResourceVersion = "4"
				obj.Generation = 1
				return obj
			}(),
		},
		"keep-fields-shared-consumable-capacity-shared-counters": {
			oldObj: sliceWithPartitionableDevicesSharedCounters,
			newObj: func() *resource.ResourceSlice {
				obj := sliceWithSharedConsumableSharedCounters.DeepCopy()
				obj.ResourceVersion = "4"
				return obj
			}(),
			featureOverrides: featuregatetesting.FeatureOverrides{features.DRAPartitionableDevices: true, features.DRAConsumableCapacity: true, features.DRASharedConsumableCapacity: true},
			expectObj: func() *resource.ResourceSlice {
				obj := sliceWithSharedConsumableSharedCounters.DeepCopy()
				obj.ResourceVersion = "4"
				obj.Generation = 1
				return obj
			}(),
		},
		"keep-existing-fields-shared-consumable-capacity-shared-counters-disabled-feature": {
			oldObj: sliceWithSharedConsumableSharedCounters,
			newObj: func() *resource.ResourceSlice {
				obj := sliceWithSharedConsumableSharedCounters.DeepCopy()
				obj.ResourceVersion = "4"
				return obj
			}(),
			featureOverrides: featuregatetesting.FeatureOverrides{features.DRAPartitionableDevices: true, features.DRASharedConsumableCapacity: false},
			expectObj: func() *resource.ResourceSlice {
				obj := sliceWithSharedConsumableSharedCounters.DeepCopy()
				obj.ResourceVersion = "4"
				return obj
			}(),
		},
		"drop-fields-shared-consumable-capacity-consumes-counters-disabled-feature": {
			oldObj: sliceWithPartitionableDevicesConsumesCounters,
			newObj: func() *resource.ResourceSlice {
				obj := sliceWithSharedConsumableConsumesCounters.DeepCopy()
				obj.ResourceVersion = "4"
				return obj
			}(),
			featureOverrides: featuregatetesting.FeatureOverrides{features.DRAPartitionableDevices: true, features.DRASharedConsumableCapacity: false},
			expectObj: func() *resource.ResourceSlice {
				obj := sliceWithSharedConsumableConsumesCounters.DeepCopy()
				obj.Spec.Devices[0].ConsumesCounters[0].Counters["memory"] = resource.CounterRequirement{}
				obj.ResourceVersion = "4"
				obj.Generation = 1
				return obj
			}(),
		},
		"keep-fields-shared-consumable-capacity-consumes-counters": {
			oldObj: sliceWithPartitionableDevicesConsumesCounters,
			newObj: func() *resource.ResourceSlice {
				obj := sliceWithSharedConsumableConsumesCounters.DeepCopy()
				obj.ResourceVersion = "4"
				return obj
			}(),
			featureOverrides: featuregatetesting.FeatureOverrides{features.DRAPartitionableDevices: true, features.DRAConsumableCapacity: true, features.DRASharedConsumableCapacity: true},
			expectObj: func() *resource.ResourceSlice {
				obj := sliceWithSharedConsumableConsumesCounters.DeepCopy()
				obj.ResourceVersion = "4"
				obj.Generation = 1
				return obj
			}(),
		},
		"keep-existing-fields-shared-consumable-capacity-consumes-counters-disabled-feature": {
			oldObj: sliceWithSharedConsumableConsumesCounters,
			newObj: func() *resource.ResourceSlice {
				obj := sliceWithSharedConsumableConsumesCounters.DeepCopy()
				obj.ResourceVersion = "4"
				return obj
			}(),
			featureOverrides: featuregatetesting.FeatureOverrides{features.DRAPartitionableDevices: true, features.DRASharedConsumableCapacity: false},
			expectObj: func() *resource.ResourceSlice {
				obj := sliceWithSharedConsumableConsumesCounters.DeepCopy()
				obj.ResourceVersion = "4"
				return obj
			}(),
		},
		"drop-fields-binding-conditions": {
			oldObj: slice,
			newObj: func() *resource.ResourceSlice {
				obj := sliceWithBindingConditions.DeepCopy()
				obj.ResourceVersion = "4"
				return obj
			}(),
			expectObj: func() *resource.ResourceSlice {
				obj := slice.DeepCopy()
				obj.ResourceVersion = "4"
				obj.Generation = 1
				return obj
			}(),
			emulatedVersion:  "1.35",
			featureOverrides: featuregatetesting.FeatureOverrides{features.DRADeviceBindingConditions: false, features.DRAResourceClaimDeviceStatus: false},
		},
		"drop-fields-binding-conditions-with-binding-conditions": {
			oldObj: slice,
			newObj: func() *resource.ResourceSlice {
				obj := sliceWithBindingConditions.DeepCopy()
				obj.ResourceVersion = "4"
				return obj
			}(),
			expectObj: func() *resource.ResourceSlice {
				obj := slice.DeepCopy()
				obj.ResourceVersion = "4"
				obj.Generation = 1
				return obj
			}(),
			featureOverrides: featuregatetesting.FeatureOverrides{features.DRADeviceBindingConditions: false, features.DRAResourceClaimDeviceStatus: true},
			emulatedVersion:  "1.37",
		},
		"keep-fields-binding-conditions": {
			oldObj: slice,
			newObj: func() *resource.ResourceSlice {
				obj := sliceWithBindingConditions.DeepCopy()
				obj.ResourceVersion = "4"
				return obj
			}(),
			expectObj: func() *resource.ResourceSlice {
				obj := sliceWithBindingConditions.DeepCopy()
				obj.ResourceVersion = "4"
				obj.Generation = 1
				return obj
			}(),
			featureOverrides: featuregatetesting.FeatureOverrides{features.DRADeviceBindingConditions: true, features.DRAResourceClaimDeviceStatus: true},
		},
		"keep-existing-fields-binding-conditions-without-featuregate-enabled": {
			oldObj: sliceWithBindingConditions,
			newObj: func() *resource.ResourceSlice {
				obj := sliceWithBindingConditions.DeepCopy()
				obj.ResourceVersion = "4"
				obj.Generation = 1
				return obj
			}(),
			expectObj: func() *resource.ResourceSlice {
				obj := sliceWithBindingConditions.DeepCopy()
				obj.ResourceVersion = "4"
				obj.Generation = 1
				return obj
			}(),
			featureOverrides: featuregatetesting.FeatureOverrides{features.DRADeviceBindingConditions: false, features.DRAResourceClaimDeviceStatus: true},
			emulatedVersion:  "1.37",
		},
		"keep-consumable-capacity": {
			oldObj: slice,
			newObj: func() *resource.ResourceSlice {
				obj := sliceWithConsumableCapacity.DeepCopy()
				obj.ResourceVersion = "4"
				return obj
			}(),
			featureOverrides: featuregatetesting.FeatureOverrides{features.DRAConsumableCapacity: true},
			expectObj: func() *resource.ResourceSlice {
				obj := sliceWithConsumableCapacity.DeepCopy()
				obj.ResourceVersion = "4"
				obj.Generation = 1
				return obj
			}(),
		},
		"keep-existing-fields-consumable-capacity-disabled-feature": {
			oldObj: sliceWithConsumableCapacity,
			newObj: func() *resource.ResourceSlice {
				obj := sliceWithConsumableCapacity.DeepCopy()
				obj.ResourceVersion = "4"
				return obj
			}(),
			featureOverrides: featuregatetesting.FeatureOverrides{features.DRAConsumableCapacity: false},
			expectObj: func() *resource.ResourceSlice {
				obj := sliceWithConsumableCapacity.DeepCopy()
				obj.ResourceVersion = "4"
				return obj
			}(),
		},
		"drop-fields-consumable-capacity-disabled-feature": {
			oldObj: sliceWithCapacity,
			newObj: func() *resource.ResourceSlice {
				obj := sliceWithConsumableCapacity.DeepCopy()
				obj.ResourceVersion = "4"
				return obj
			}(),
			featureOverrides: featuregatetesting.FeatureOverrides{features.DRAConsumableCapacity: false},
			expectObj: func() *resource.ResourceSlice {
				obj := sliceWithCapacity.DeepCopy()
				obj.ResourceVersion = "4"
				obj.Generation = 1
				return obj
			}(),
		},
		"drop-list-type-attributes": {
			oldObj: slice,
			newObj: func() *resource.ResourceSlice {
				obj := sliceWithListTypeAttributes.DeepCopy()
				obj.ResourceVersion = "4"
				return obj
			}(),
			featureOverrides:      featuregatetesting.FeatureOverrides{features.DRAListTypeAttributes: false},
			expectValidationError: true,
		},
		"keep-fields-list-type-attributes": {
			oldObj: slice.DeepCopy(),
			newObj: func() *resource.ResourceSlice {
				obj := sliceWithListTypeAttributes.DeepCopy()
				obj.ResourceVersion = "4"
				return obj
			}(),
			featureOverrides: featuregatetesting.FeatureOverrides{features.DRAListTypeAttributes: true},
			expectObj: func() *resource.ResourceSlice {
				obj := sliceWithListTypeAttributes.DeepCopy()
				obj.ResourceVersion = "4"
				obj.Generation = 1
				return obj
			}(),
		},
		"keep-new-list-type-attributes-by-default": {
			oldObj: slice,
			newObj: func() *resource.ResourceSlice {
				obj := sliceWithListTypeAttributes.DeepCopy()
				obj.ResourceVersion = "4"
				return obj
			}(),
			expectObj: func() *resource.ResourceSlice {
				obj := sliceWithListTypeAttributes.DeepCopy()
				obj.ResourceVersion = "4"
				obj.Generation = 1
				return obj
			}(),
		},
		"keep-existing-fields-list-type-attributes": {
			oldObj: sliceWithListTypeAttributes,
			newObj: func() *resource.ResourceSlice {
				obj := sliceWithListTypeAttributes.DeepCopy()
				obj.ResourceVersion = "4"
				return obj
			}(),
			featureOverrides: featuregatetesting.FeatureOverrides{features.DRAListTypeAttributes: true},
			expectObj: func() *resource.ResourceSlice {
				obj := sliceWithListTypeAttributes.DeepCopy()
				obj.ResourceVersion = "4"
				return obj
			}(),
		},
		"keep-existing-list-type-attributes-without-featuregate-enabled": {
			oldObj: sliceWithListTypeAttributes,
			newObj: func() *resource.ResourceSlice {
				obj := sliceWithListTypeAttributes.DeepCopy()
				obj.ResourceVersion = "4"
				return obj
			}(),
			featureOverrides: featuregatetesting.FeatureOverrides{features.DRAListTypeAttributes: false},
			expectObj: func() *resource.ResourceSlice {
				obj := sliceWithListTypeAttributes.DeepCopy()
				obj.ResourceVersion = "4"
				return obj
			}(),
		},
		"keep-existing-fields-node-allocatable-dra-claims": {
			oldObj: sliceWithNodeAllocatableResources,
			newObj: func() *resource.ResourceSlice {
				obj := sliceWithNodeAllocatableResources.DeepCopy()
				obj.ResourceVersion = "4"
				return obj
			}(),
			featureOverrides: featuregatetesting.FeatureOverrides{features.DRANodeAllocatableResources: true},
			expectObj: func() *resource.ResourceSlice {
				obj := sliceWithNodeAllocatableResources.DeepCopy()
				obj.ResourceVersion = "4"
				return obj
			}(),
		},
		"drop-fields-node-allocatable-dra-claims-disabled-feature": {
			oldObj: slice,
			newObj: func() *resource.ResourceSlice {
				obj := sliceWithNodeAllocatableResources.DeepCopy()
				obj.ResourceVersion = "4"
				return obj
			}(),
			featureOverrides: featuregatetesting.FeatureOverrides{features.DRANodeAllocatableResources: false},
			expectObj: func() *resource.ResourceSlice {
				obj := slice.DeepCopy()
				obj.ResourceVersion = "4"
				obj.Generation = 1
				return obj
			}(),
		},
		"keep-existing-fields-node-allocatable-dra-claims-disabled-feature": {
			oldObj: sliceWithNodeAllocatableResources,
			newObj: func() *resource.ResourceSlice {
				obj := sliceWithNodeAllocatableResources.DeepCopy()
				obj.ResourceVersion = "4"
				return obj
			}(),
			featureOverrides: featuregatetesting.FeatureOverrides{features.DRANodeAllocatableResources: false},
			expectObj: func() *resource.ResourceSlice {
				obj := sliceWithNodeAllocatableResources.DeepCopy()
				obj.ResourceVersion = "4"
				return obj
			}(),
		},
		"drop-fields-optional-node-operations": {
			oldObj: slice,
			newObj: func() *resource.ResourceSlice {
				obj := sliceWithSkipNodeOperations.DeepCopy()
				obj.ResourceVersion = "4"
				return obj
			}(),
			featureOverrides: featuregatetesting.FeatureOverrides{features.DRAOptionalNodeOperations: false},
			expectObj: func() *resource.ResourceSlice {
				obj := slice.DeepCopy()
				obj.ResourceVersion = "4"
				obj.Generation = 1
				return obj
			}(),
		},
		"keep-fields-optional-node-operations": {
			oldObj: slice,
			newObj: func() *resource.ResourceSlice {
				obj := sliceWithSkipNodeOperations.DeepCopy()
				obj.ResourceVersion = "4"
				return obj
			}(),
			featureOverrides: featuregatetesting.FeatureOverrides{features.DRAOptionalNodeOperations: true},
			expectObj: func() *resource.ResourceSlice {
				obj := sliceWithSkipNodeOperations.DeepCopy()
				obj.ResourceVersion = "4"
				obj.Generation = 1
				return obj
			}(),
		},
		"keep-existing-optional-node-operations-without-featuregate-enabled": {
			oldObj: sliceWithSkipNodeOperations,
			newObj: func() *resource.ResourceSlice {
				obj := sliceWithSkipNodeOperations.DeepCopy()
				obj.ResourceVersion = "4"
				return obj
			}(),
			featureOverrides: featuregatetesting.FeatureOverrides{features.DRAOptionalNodeOperations: false},
			expectObj: func() *resource.ResourceSlice {
				obj := sliceWithSkipNodeOperations.DeepCopy()
				obj.ResourceVersion = "4"
				return obj
			}(),
		},
		"drop-fields-compatibility-groups": {
			oldObj: sliceWithPartitionableDevicesConsumesCounters,
			newObj: func() *resource.ResourceSlice {
				obj := sliceWithCompatibilityGroups.DeepCopy()
				obj.ResourceVersion = "4"
				return obj
			}(),
			featureOverrides: featuregatetesting.FeatureOverrides{features.DRAPartitionableDevices: true, features.DRADeviceCompatibilityGroups: false},
			expectObj: func() *resource.ResourceSlice {
				obj := sliceWithPartitionableDevicesConsumesCounters.DeepCopy()
				obj.ResourceVersion = "4"
				// The generation is bumped for the incoming spec change even
				// though dropping the disabled field makes the stored spec
				// identical again, like in the partitionable drop case above.
				obj.Generation = 1
				return obj
			}(),
		},
		"keep-fields-compatibility-groups": {
			oldObj: sliceWithPartitionableDevicesConsumesCounters,
			newObj: func() *resource.ResourceSlice {
				obj := sliceWithCompatibilityGroups.DeepCopy()
				obj.ResourceVersion = "4"
				return obj
			}(),
			featureOverrides: featuregatetesting.FeatureOverrides{features.DRAPartitionableDevices: true, features.DRADeviceCompatibilityGroups: true},
			expectObj: func() *resource.ResourceSlice {
				obj := sliceWithCompatibilityGroups.DeepCopy()
				obj.ResourceVersion = "4"
				obj.Generation = 1
				return obj
			}(),
		},
		"keep-existing-fields-compatibility-groups": {
			oldObj: sliceWithCompatibilityGroups,
			newObj: func() *resource.ResourceSlice {
				obj := sliceWithCompatibilityGroups.DeepCopy()
				obj.ResourceVersion = "4"
				return obj
			}(),
			featureOverrides: featuregatetesting.FeatureOverrides{features.DRAPartitionableDevices: true, features.DRADeviceCompatibilityGroups: true},
			expectObj: func() *resource.ResourceSlice {
				obj := sliceWithCompatibilityGroups.DeepCopy()
				obj.ResourceVersion = "4"
				return obj
			}(),
		},
		"keep-existing-fields-compatibility-groups-disabled-feature": {
			oldObj: sliceWithCompatibilityGroups,
			newObj: func() *resource.ResourceSlice {
				obj := sliceWithCompatibilityGroups.DeepCopy()
				obj.ResourceVersion = "4"
				return obj
			}(),
			featureOverrides: featuregatetesting.FeatureOverrides{features.DRAPartitionableDevices: true, features.DRADeviceCompatibilityGroups: false},
			expectObj: func() *resource.ResourceSlice {
				obj := sliceWithCompatibilityGroups.DeepCopy()
				obj.ResourceVersion = "4"
				return obj
			}(),
		},
	}

	for name, tc := range testcases {
		t.Run(name, func(t *testing.T) {
			if tc.emulatedVersion != "" {
				featuregatetesting.SetFeatureGateEmulationVersionDuringTest(t, utilfeature.DefaultFeatureGate, version.MustParse(tc.emulatedVersion))
			}
			featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, tc.featureOverrides)

			oldObj := tc.oldObj.DeepCopy()
			newObj := tc.newObj.DeepCopy()

			Strategy.PrepareForUpdate(ctx, newObj, oldObj)
			if errs := Strategy.ValidateUpdate(ctx, newObj, oldObj); len(errs) != 0 {
				if !tc.expectValidationError {
					t.Fatalf("unexpected validation errors: %q", errs)
				}
				return
			} else if tc.expectValidationError {
				t.Fatal("expected validation error(s), got none")
			}
			if warnings := Strategy.WarningsOnUpdate(ctx, newObj, oldObj); len(warnings) != 0 {
				t.Fatalf("unexpected warnings: %q", warnings)
			}
			Strategy.Canonicalize(newObj)

			expectObj := tc.expectObj.DeepCopy()
			assert.Equal(t, expectObj, newObj)

		})
	}
}

func TestWarningsOnCreate(t *testing.T) {
	ctx := genericapirequest.NewDefaultContext()

	testCases := map[string]struct {
		obj                 *resource.ResourceSlice
		wantWarningMessages []string
	}{
		"valid driver": {
			obj:                 slice,
			wantWarningMessages: []string{},
		},
		"uppercase driver warning": {
			obj: func() *resource.ResourceSlice {
				obj := slice.DeepCopy()
				obj.Spec.Driver = "Foo.COM"
				return obj
			}(),
			wantWarningMessages: []string{
				`spec.driver: driver names should be lowercase; "Foo.COM" contains uppercase characters`,
			},
		},
	}

	for name, tc := range testCases {
		t.Run(name, func(t *testing.T) {
			warnings := Strategy.WarningsOnCreate(ctx, tc.obj)
			if warnings == nil {
				warnings = []string{}
			}
			require.Equal(t, tc.wantWarningMessages, warnings)
		})
	}
}

func TestWarningsOnUpdate(t *testing.T) {
	ctx := genericapirequest.NewDefaultContext()

	testCases := map[string]struct {
		newObj              *resource.ResourceSlice
		oldObj              *resource.ResourceSlice
		wantWarningMessages []string
	}{
		"valid driver update": {
			newObj:              slice.DeepCopy(),
			oldObj:              slice.DeepCopy(),
			wantWarningMessages: []string{},
		},
		"uppercase driver warning on update": {
			newObj: func() *resource.ResourceSlice {
				obj := slice.DeepCopy()
				obj.Spec.Driver = "Foo.COM"
				return obj
			}(),
			oldObj: slice.DeepCopy(),
			wantWarningMessages: []string{
				`spec.driver: driver names should be lowercase; "Foo.COM" contains uppercase characters`,
			},
		},
	}

	for name, tc := range testCases {
		t.Run(name, func(t *testing.T) {
			warnings := Strategy.WarningsOnUpdate(ctx, tc.newObj, tc.oldObj)
			if warnings == nil {
				warnings = []string{}
			}
			require.Equal(t, tc.wantWarningMessages, warnings)
		})
	}
}

func TestMatch(t *testing.T) {
	testCases := map[string]struct {
		in            *resource.ResourceSlice
		fieldSelector fields.Selector
		expectMatch   bool
	}{
		"match-metadata-name": {
			in: &resource.ResourceSlice{
				ObjectMeta: metav1.ObjectMeta{Name: "foo"},
			},
			fieldSelector: fields.ParseSelectorOrDie("metadata.name=foo"),
			expectMatch:   true,
		},
		"not-match-metadata-name": {
			in: &resource.ResourceSlice{
				ObjectMeta: metav1.ObjectMeta{Name: "foo"},
			},
			fieldSelector: fields.ParseSelectorOrDie("metadata.name=bar"),
			expectMatch:   false,
		},
		"match-spec-driver": {
			in: &resource.ResourceSlice{
				Spec: resource.ResourceSliceSpec{
					Driver: "foo",
				},
			},
			fieldSelector: fields.ParseSelectorOrDie("spec.driver=foo"),
			expectMatch:   true,
		},
		"not-match-spec-driver": {
			in: &resource.ResourceSlice{
				Spec: resource.ResourceSliceSpec{
					Driver: "foo",
				},
			},
			fieldSelector: fields.ParseSelectorOrDie("spec.driver=bar"),
			expectMatch:   false,
		},
		"match-spec-node-name": {
			in: &resource.ResourceSlice{
				Spec: resource.ResourceSliceSpec{
					NodeName: new("foo"),
				},
			},
			fieldSelector: fields.ParseSelectorOrDie("spec.nodeName=foo"),
			expectMatch:   true,
		},
		"not-match-spec-node-name": {
			in: &resource.ResourceSlice{
				Spec: resource.ResourceSliceSpec{
					NodeName: new("foo"),
				},
			},
			fieldSelector: fields.ParseSelectorOrDie("spec.nodeName=bar"),
			expectMatch:   false,
		},
		"match-spec-pool-name": {
			in: &resource.ResourceSlice{
				Spec: resource.ResourceSliceSpec{
					Pool: resource.ResourcePool{
						Name: "foo",
					},
				},
			},
			fieldSelector: fields.ParseSelectorOrDie("spec.pool.name=foo"),
			expectMatch:   true,
		},
		"not-match-spec-pool-name": {
			in: &resource.ResourceSlice{
				Spec: resource.ResourceSliceSpec{
					Pool: resource.ResourcePool{
						Name: "foo",
					},
				},
			},
			fieldSelector: fields.ParseSelectorOrDie("spec.pool.name=bar"),
			expectMatch:   false,
		},
		"not-match-spec-pool-generation": {
			in: &resource.ResourceSlice{
				Spec: resource.ResourceSliceSpec{
					Pool: resource.ResourcePool{
						Generation: 1,
					},
				},
			},
			fieldSelector: fields.ParseSelectorOrDie("spec.pool.generation=1"),
			expectMatch:   false,
		},
	}

	for name, testCase := range testCases {
		t.Run(name, func(t *testing.T) {
			m := Match(labels.Everything(), testCase.fieldSelector)
			result, err := m.Matches(testCase.in)
			require.NoError(t, err)
			assert.Equal(t, testCase.expectMatch, result, "selector: %s, resourceSlice: %+v", testCase.fieldSelector.String(), testCase.in)
		})
	}
}

// TestCounterRequirementValidation checks counter validation and updates to existing invalid counters when feature gates change.
func TestCounterRequirementValidation(t *testing.T) {
	zero := k8sresource.MustParse("0")
	one := k8sresource.MustParse("1")
	negative := k8sresource.MustParse("-1")
	none := resource.CounterRequirement{}
	fixedZero := resource.CounterRequirement{Value: &zero}
	fixed := resource.CounterRequirement{Value: &one}
	from := resource.CounterRequirement{ValueFrom: &resource.CounterValueFrom{CapacityName: "driver.example.com/bandwidth"}}
	both := resource.CounterRequirement{Value: &zero, ValueFrom: from.ValueFrom}
	makeSlice := func(requirement resource.CounterRequirement) *resource.ResourceSlice {
		obj := slice.DeepCopy()
		obj.UID = "counter-validation"
		obj.Spec.Devices[0].ConsumesCounters = []resource.DeviceCounterConsumption{{
			CounterSet: "set", Counters: map[string]resource.CounterRequirement{"memory": requirement},
		}}
		return obj
	}
	forms := []struct {
		name            string
		requirement     resource.CounterRequirement
		wantOn, wantOff bool
	}{
		{"neither", none, false, false},
		{"explicit-zero", fixedZero, true, true},
		{"positive", fixed, true, true},
		{"negative", resource.CounterRequirement{Value: &negative}, false, false},
		{"valueFrom", from, true, false},
		{"unqualified-capacity-name", resource.CounterRequirement{ValueFrom: &resource.CounterValueFrom{CapacityName: "bandwidth"}}, true, false},
		{"empty-capacity-name", resource.CounterRequirement{ValueFrom: &resource.CounterValueFrom{}}, false, false},
		{"invalid-capacity-name", resource.CounterRequirement{ValueFrom: &resource.CounterValueFrom{CapacityName: "bad/name/extra"}}, false, false},
		{"both", both, false, true},
		{"both-with-negative", resource.CounterRequirement{Value: &negative, ValueFrom: from.ValueFrom}, false, false},
	}
	type scenario struct {
		name     string
		obj, old *resource.ResourceSlice
		allowed  bool
	}
	for name, config := range map[string]struct {
		apiVersion   string
		beta, shared bool
	}{
		"v1-beta=false-shared=false":      {"v1", false, false},
		"v1-beta=false-shared=true":       {"v1", false, true},
		"v1-beta=true-shared=false":       {"v1", true, false},
		"v1-beta=true-shared=true":        {"v1", true, true},
		"v1beta1-beta=false-shared=false": {"v1beta1", false, false},
		"v1beta1-beta=false-shared=true":  {"v1beta1", false, true},
		"v1beta1-beta=true-shared=false":  {"v1beta1", true, false},
		"v1beta1-beta=true-shared=true":   {"v1beta1", true, true},
		"v1beta2-beta=false-shared=false": {"v1beta2", false, false},
		"v1beta2-beta=false-shared=true":  {"v1beta2", false, true},
		"v1beta2-beta=true-shared=false":  {"v1beta2", true, false},
		"v1beta2-beta=true-shared=true":   {"v1beta2", true, true},
	} {
		t.Run(name, func(t *testing.T) {
			version, beta, shared := config.apiVersion, config.beta, config.shared
			devicePath := field.NewPath("spec", "devices").Index(0)
			if version == "v1beta1" {
				devicePath = devicePath.Child("basic")
			}
			counterPath := devicePath.Child("consumesCounters").Index(0).Child("counters").Key("memory")
			featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
				apiserverfeatures.DeclarativeValidationBeta: beta,
				features.DRAPartitionableDevices:            true,
				features.DRAConsumableCapacity:              true,
				features.DRASharedConsumableCapacity:        shared,
			})
			var scenarios []scenario
			for _, form := range forms {
				allowed := form.wantOff
				if shared {
					allowed = form.wantOn
				}
				scenarios = append(scenarios,
					scenario{"create-" + form.name, makeSlice(form.requirement), nil, allowed},
					scenario{"update-" + form.name, makeSlice(form.requirement), makeSlice(fixed), allowed},
				)
			}
			for _, original := range []struct {
				name        string
				requirement resource.CounterRequirement
			}{
				{"neither", none}, {"both", both}, {"empty-capacity-name", resource.CounterRequirement{ValueFrom: &resource.CounterValueFrom{}}},
			} {
				old := makeSlice(original.requirement)
				unchanged := old.DeepCopy()
				unchanged.Labels = map[string]string{"test": "unchanged-devices"}
				changed := old.DeepCopy()
				changed.Spec.Devices[0].Name = "renamed-device"
				repaired := makeSlice(fixedZero)
				scenarios = append(scenarios,
					scenario{"stored-invalid-" + original.name + "-metadata-only", unchanged, old, true},
					scenario{"stored-invalid-" + original.name + "-changed-devices", changed, old, false},
					scenario{"stored-invalid-" + original.name + "-repair-to-zero", repaired, old, true},
				)
			}
			oldFrom := makeSlice(from)
			changedFrom := oldFrom.DeepCopy()
			changedFrom.Spec.Devices[0].Name = "renamed-device"
			scenarios = append(scenarios, scenario{"existing-valueFrom-changed-device", changedFrom, oldFrom, true})
			for _, tc := range scenarios {
				t.Run(tc.name, func(t *testing.T) {
					obj := tc.obj.DeepCopy()
					var old *resource.ResourceSlice
					verb := "create"
					if tc.old != nil {
						old = tc.old.DeepCopy()
						old.ResourceVersion, obj.ResourceVersion = "1", "1"
						verb = "update"
					}
					ctx := genericapirequest.WithRequestInfo(context.Background(), &genericapirequest.RequestInfo{
						APIGroup: "resource.k8s.io", APIVersion: version, Resource: "resourceslices", Verb: verb, IsResourceRequest: true,
					})
					var errs field.ErrorList
					if old == nil {
						Strategy.PrepareForCreate(ctx, obj)
						errs = rest.ValidateCreate(ctx, obj, Strategy)
					} else {
						Strategy.PrepareForUpdate(ctx, obj, old)
						errs = rest.ValidateUpdate(ctx, obj, old, Strategy)
					}
					if (len(errs) == 0) != tc.allowed {
						t.Fatalf("admission allowed=%t, want %t: %v", len(errs) == 0, tc.allowed, errs)
					}
					if shared && (tc.name == "create-neither" || tc.name == "create-both") {
						if len(errs) != 1 || errs[0].Type != field.ErrorTypeInvalid || errs[0].Origin != "union" || errs[0].Field != counterPath.String() || errs[0].IsAlpha() || errs[0].IsBeta() {
							t.Fatalf("expected one authoritative union error at %s, got %v", counterPath, errs)
						}
					}
					if shared && (tc.name == "create-empty-capacity-name" || tc.name == "update-empty-capacity-name") {
						capacityNamePath := counterPath.Child("valueFrom", "capacityName")
						if len(errs) != 1 || errs[0].Type != field.ErrorTypeRequired || errs[0].Field != capacityNamePath.String() || errs[0].IsAlpha() || errs[0].IsBeta() {
							t.Fatalf("expected one authoritative required error at %s, got %v", capacityNamePath, errs)
						}
					}
				})
			}
		})
	}
}

// TestCounterValueValidation checks quantity limits and which updates to existing negative counter values are allowed.
func TestCounterValueValidation(t *testing.T) {
	for name, config := range map[string]struct {
		apiVersion   string
		beta, shared bool
	}{
		"shared=false-v1-beta=false":      {"v1", false, false},
		"shared=true-v1-beta=false":       {"v1", false, true},
		"shared=false-v1-beta=true":       {"v1", true, false},
		"shared=true-v1-beta=true":        {"v1", true, true},
		"shared=false-v1beta1-beta=false": {"v1beta1", false, false},
		"shared=true-v1beta1-beta=false":  {"v1beta1", false, true},
		"shared=false-v1beta1-beta=true":  {"v1beta1", true, false},
		"shared=true-v1beta1-beta=true":   {"v1beta1", true, true},
		"shared=false-v1beta2-beta=false": {"v1beta2", false, false},
		"shared=true-v1beta2-beta=false":  {"v1beta2", false, true},
		"shared=false-v1beta2-beta=true":  {"v1beta2", true, false},
		"shared=true-v1beta2-beta=true":   {"v1beta2", true, true},
	} {
		t.Run(name, func(t *testing.T) {
			apiVersion, beta, shared := config.apiVersion, config.beta, config.shared
			makeSlice := func(value string) *resource.ResourceSlice {
				obj := slice.DeepCopy()
				obj.UID = "counter-values"
				quantity := k8sresource.MustParse(value)
				if shared {
					obj.Spec.Devices = nil
					obj.Spec.SharedCounters = []resource.CounterSet{{
						Name: "set", Counters: map[string]resource.Counter{"memory": {Value: quantity}},
					}}
				} else {
					obj.Spec.Devices[0].ConsumesCounters = []resource.DeviceCounterConsumption{{
						CounterSet: "set", Counters: map[string]resource.CounterRequirement{"memory": {Value: &quantity}},
					}}
				}
				return obj
			}
			valuePath := field.NewPath("spec", "sharedCounters").Index(0).Child("counters").Key("memory").Child("value")
			if !shared {
				valuePath = field.NewPath("spec", "devices").Index(0)
				if apiVersion == "v1beta1" {
					valuePath = valuePath.Child("basic")
				}
				valuePath = valuePath.Child("consumesCounters").Index(0).Child("counters").Key("memory").Child("value")
			}
			featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
				apiserverfeatures.DeclarativeValidationBeta: beta,
				features.DRAPartitionableDevices:            true,
				features.DRAConsumableCapacity:              true,
				features.DRASharedConsumableCapacity:        true,
			})
			type scenario struct {
				name         string
				obj, old     *resource.ResourceSlice
				wantNegative bool
			}
			var scenarios []scenario
			for _, value := range []struct {
				quantity string
				negative bool
			}{
				{"0", false}, {"1", false}, {"0.001", false},
				{"9223372036854775807", false}, {"10000000000000000000", false},
				{"-1", true}, {"-500m", true}, {"-10000000000000000000", true},
			} {
				scenarios = append(scenarios,
					scenario{"create-" + value.quantity, makeSlice(value.quantity), nil, value.negative},
					scenario{"update-" + value.quantity, makeSlice(value.quantity), makeSlice("1"), value.negative},
				)
			}
			old := makeSlice("-1")
			unchanged := old.DeepCopy()
			unchanged.Labels = map[string]string{"test": "unchanged-counter"}
			generation := old.DeepCopy()
			generation.Spec.Pool.Generation++
			renamed := old.DeepCopy()
			sibling := old.DeepCopy()
			if shared {
				renamed.Spec.SharedCounters[0].Name = "renamed-set"
				sibling.Spec.SharedCounters[0].Counters["other"] = resource.Counter{Value: k8sresource.MustParse("1")}
			} else {
				renamed.Spec.Devices[0].Name = "renamed-device"
				sibling.Spec.Devices[0].ConsumesCounters[0].Counters["other"] = resource.CounterRequirement{Value: new(k8sresource.MustParse("1"))}
			}
			scenarios = append(scenarios,
				scenario{"stored-negative-label-only", unchanged, old, false},
				scenario{"stored-negative-pool-generation", generation, old, false},
				scenario{"stored-negative-renamed-list-item", renamed, old, true},
				scenario{"stored-negative-sibling-change", sibling, old, !shared},
				scenario{"stored-negative-changed-value", makeSlice("-2"), old, true},
				scenario{"stored-negative-repair-to-zero", makeSlice("0"), old, false},
			)
			for _, tc := range scenarios {
				t.Run(tc.name, func(t *testing.T) {
					obj := tc.obj.DeepCopy()
					verb := "create"
					if tc.old != nil {
						verb = "update"
					}
					ctx := genericapirequest.WithRequestInfo(context.Background(), &genericapirequest.RequestInfo{
						APIGroup: "resource.k8s.io", APIVersion: apiVersion, Resource: "resourceslices", Verb: verb, IsResourceRequest: true,
					})
					var errs field.ErrorList
					if tc.old == nil {
						Strategy.PrepareForCreate(ctx, obj)
						errs = rest.ValidateCreate(ctx, obj, Strategy)
					} else {
						old := tc.old.DeepCopy()
						old.ResourceVersion, obj.ResourceVersion = "1", "1"
						Strategy.PrepareForUpdate(ctx, obj, old)
						errs = rest.ValidateUpdate(ctx, obj, old, Strategy)
					}
					if !tc.wantNegative {
						if len(errs) != 0 {
							t.Fatalf("expected admission to succeed, got %v", errs)
						}
					} else if len(errs) != 1 || errs[0].Type != field.ErrorTypeInvalid || errs[0].Origin != "minimum" || errs[0].Field != valuePath.String() || errs[0].IsAlpha() || errs[0].IsBeta() {
						t.Fatalf("expected one standard minimum error at %s, got %v", valuePath, errs)
					}
				})
			}
		})
	}
}

// TestSharedCounterPolicyUpdates checks which updates to existing policies are allowed after fractional capacity support is disabled.
func TestSharedCounterPolicyUpdates(t *testing.T) {
	for name, config := range map[string]struct {
		apiVersion       string
		beta, fractional bool
	}{
		"v1-beta=false/fractional=false":      {"v1", false, false},
		"v1-beta=false/fractional=true":       {"v1", false, true},
		"v1-beta=true/fractional=false":       {"v1", true, false},
		"v1-beta=true/fractional=true":        {"v1", true, true},
		"v1beta1-beta=false/fractional=false": {"v1beta1", false, false},
		"v1beta1-beta=false/fractional=true":  {"v1beta1", false, true},
		"v1beta1-beta=true/fractional=false":  {"v1beta1", true, false},
		"v1beta1-beta=true/fractional=true":   {"v1beta1", true, true},
		"v1beta2-beta=false/fractional=false": {"v1beta2", false, false},
		"v1beta2-beta=false/fractional=true":  {"v1beta2", false, true},
		"v1beta2-beta=true/fractional=false":  {"v1beta2", true, false},
		"v1beta2-beta=true/fractional=true":   {"v1beta2", true, true},
	} {
		t.Run(name, func(t *testing.T) {
			apiVersion, beta, fractional := config.apiVersion, config.beta, config.fractional
			featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
				apiserverfeatures.DeclarativeValidationBeta: beta,
				features.DRAPartitionableDevices:            true,
				features.DRAConsumableCapacity:              true,
				features.DRASharedConsumableCapacity:        true,
				features.DRAFractionalCapacityRange:         true,
			})
			old := sliceWithPartitionableDevicesSharedCounters.DeepCopy()
			old.Spec.SharedCounters[0].Counters["memory"] = resource.Counter{
				Value: k8sresource.MustParse("1"),
				ConsumptionPolicy: &resource.CapacityRequestPolicy{
					Default: new(k8sresource.MustParse("500m")), ValidValues: []k8sresource.Quantity{k8sresource.MustParse("500m"), k8sresource.MustParse("1")},
				},
			}
			createCtx := genericapirequest.WithRequestInfo(context.Background(), &genericapirequest.RequestInfo{
				APIGroup: "resource.k8s.io", APIVersion: apiVersion, Resource: "resourceslices", Verb: "create", IsResourceRequest: true,
			})
			Strategy.PrepareForCreate(createCtx, old)
			if errs := rest.ValidateCreate(createCtx, old, Strategy); len(errs) != 0 {
				t.Fatalf("expected fractional policy to be valid before gate change, got %v", errs)
			}
			old.ResourceVersion = "1"
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.DRAFractionalCapacityRange, fractional)
			for _, change := range []string{"label-only", "pool-generation", "sibling-counter", "policy-change", "repair-policy"} {
				t.Run(change, func(t *testing.T) {
					obj := old.DeepCopy()
					counter := obj.Spec.SharedCounters[0].Counters["memory"]
					switch change {
					case "label-only":
						obj.Labels = map[string]string{"test": "unchanged-policy"}
					case "pool-generation":
						obj.Spec.Pool.Generation++
					case "sibling-counter":
						obj.Spec.SharedCounters[0].Counters["other"] = resource.Counter{Value: k8sresource.MustParse("1")}
					case "policy-change":
						counter.ConsumptionPolicy.Default = new(k8sresource.MustParse("1"))
					case "repair-policy":
						counter.ConsumptionPolicy.Default = new(k8sresource.MustParse("1"))
						counter.ConsumptionPolicy.ValidValues = []k8sresource.Quantity{k8sresource.MustParse("1")}
					}
					obj.Spec.SharedCounters[0].Counters["memory"] = counter
					ctx := genericapirequest.WithRequestInfo(context.Background(), &genericapirequest.RequestInfo{
						APIGroup: "resource.k8s.io", APIVersion: apiVersion, Resource: "resourceslices", Verb: "update", IsResourceRequest: true,
					})
					Strategy.PrepareForUpdate(ctx, obj, old)
					errs := rest.ValidateUpdate(ctx, obj, old, Strategy)
					if !fractional && (change == "sibling-counter" || change == "policy-change") {
						if len(errs) != 1 || errs[0].Type != field.ErrorTypeDuplicate || errs[0].Field != "spec.sharedCounters[0].counters[memory].consumptionPolicy.validValues[1]" {
							t.Fatalf("expected policy revalidation to reject rounded duplicate values, got %v", errs)
						}
					} else if len(errs) != 0 {
						t.Fatalf("expected policy update to succeed, got %v", errs)
					}
					if !fractional && (obj.Spec.SharedCounters[0].Counters["memory"].ConsumptionPolicy == nil) {
						t.Fatal("stored consumption policy was dropped after the fractional gate changed")
					}
				})
			}
		})
	}
}
