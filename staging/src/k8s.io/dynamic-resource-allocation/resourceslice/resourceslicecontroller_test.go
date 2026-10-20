/*
Copyright 2024 The Kubernetes Authors.

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
	"errors"
	"fmt"
	"math"
	"sort"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	v1 "k8s.io/api/core/v1"
	resourceapi "k8s.io/api/resource/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/client-go/kubernetes/fake"
	k8stesting "k8s.io/client-go/testing"
	"k8s.io/dynamic-resource-allocation/internal/workqueue"
	"k8s.io/klog/v2"
	"k8s.io/klog/v2/ktesting"
	"k8s.io/utils/ptr"
)

func init() {
	klog.InitFlags(nil)
}

// TestControllerSyncPool verifies that syncPool produces the right ResourceSlices.
// Update vs. Create API calls are checked by bumping the ResourceVersion in
// updates.
func TestControllerSyncPool(t *testing.T) {

	var (
		ownerName      = "owner"
		nodeUID        = types.UID("node-uid")
		driverName     = "driver"
		poolName       = "pool"
		deviceName     = "device"
		deviceName1    = "device-1"
		deviceName2    = "device-2"
		deviceName3    = "device-3"
		deviceName4    = "device-4"
		generateName1  = encodeIndex(0, resourceSliceIndexMinLength) + "-" + driverName + "-" + ownerName + "-"
		generateName2  = encodeIndex(1, resourceSliceIndexMinLength) + "-" + driverName + "-" + ownerName + "-"
		generateName3  = encodeIndex(2, resourceSliceIndexMinLength) + "-" + driverName + "-" + ownerName + "-"
		generateName4  = encodeIndex(3, resourceSliceIndexMinLength) + "-" + driverName + "-" + ownerName + "-"
		resourceSlice1 = generateName1 + "0"
		resourceSlice2 = generateName2 + "1"
		resourceSlice3 = generateName3 + "2"
		attrs          = map[resourceapi.QualifiedName]resourceapi.DeviceAttribute{
			"new-attribute": {StringValue: ptr.To("value")},
		}
		nodeSelector = &v1.NodeSelector{
			NodeSelectorTerms: []v1.NodeSelectorTerm{{
				MatchFields: []v1.NodeSelectorRequirement{{
					Key:    "name",
					Values: []string{"node-a"},
				}},
			}},
		}
		otherNodeSelector = &v1.NodeSelector{
			NodeSelectorTerms: []v1.NodeSelectorTerm{{
				MatchFields: []v1.NodeSelectorRequirement{{
					Key:    "name",
					Values: []string{"node-b"},
				}},
			}},
		}
		timeAdded                 = metav1.Time{Time: time.Now().Round(time.Second)}
		timeAddedLater            = metav1.Time{Time: timeAdded.Add(time.Minute)}
		timeAddedEarlier          = metav1.Time{Time: timeAdded.Add(-time.Minute)}
		timeAddedEarlierSubSecond = metav1.Time{Time: timeAddedEarlier.Add(100 * time.Millisecond)}
	)

	testCases := map[string]struct {
		features  features
		syncDelay *time.Duration
		// nodeUID is empty if not a node-local.
		nodeUID types.UID
		// noOwner completely disables setting an owner.
		noOwner bool
		// reconcilePoolWithName limits reconciliation to a single pool (issue #137011).
		// When set, NodeName is not set on slices even for Node owner.
		reconcilePoolWithName string
		// initialObjects is a list of initial resource slices to be used in the test.
		initialObjects         []runtime.Object
		initialOtherObjects    []runtime.Object
		inputDriverResources   *DriverResources
		expectedResourceSlices []resourceapi.ResourceSlice
		expectedStats          Stats
		expectedErrors         []string
	}{
		"create-slice": {
			nodeUID:        nodeUID,
			initialObjects: []runtime.Object{},
			inputDriverResources: &DriverResources{
				Pools: map[string]Pool{
					poolName: {
						Slices: []Slice{{Devices: []resourceapi.Device{}}},
					},
				},
			},
			expectedStats: Stats{
				NumCreates: 1,
			},
			expectedResourceSlices: []resourceapi.ResourceSlice{
				*MakeResourceSlice().Name(resourceSlice1).GenerateName(generateName1).
					ResourceVersion("1").
					NodeOwnerReferences(ownerName, string(nodeUID)).NodeName(ownerName).
					Driver(driverName).Devices([]resourceapi.Device{}).
					Pool(resourceapi.ResourcePool{Name: poolName, Generation: 1, ResourceSliceCount: 1}).Obj(),
			},
		},
		"invalid-other-pool-blocks-create-slice": {
			nodeUID:        nodeUID,
			initialObjects: []runtime.Object{},
			inputDriverResources: &DriverResources{
				Pools: map[string]Pool{
					poolName:     {Slices: []Slice{{Devices: []resourceapi.Device{}}}},
					"other-pool": {Slices: []Slice{{Devices: []resourceapi.Device{newDevice(deviceName), newDevice(deviceName)}}}},
				},
			},
			expectedErrors: []string{`pool validation failed: found duplicate device "device" in pool "other-pool"`},
		},
		"create-slice-with-skip-node-operations": {
			nodeUID:        nodeUID,
			initialObjects: []runtime.Object{},
			inputDriverResources: &DriverResources{
				Pools: map[string]Pool{
					poolName: {
						Slices: []Slice{{
							Devices:            []resourceapi.Device{},
							SkipNodeOperations: []resourceapi.SkipNodeOperation{resourceapi.SkipNodeOperationAll},
						}},
					},
				},
			},
			expectedStats: Stats{
				NumCreates: 1,
			},
			expectedResourceSlices: []resourceapi.ResourceSlice{
				*MakeResourceSlice().Name(resourceSlice1).GenerateName(generateName1).
					ResourceVersion("1").
					NodeOwnerReferences(ownerName, string(nodeUID)).NodeName(ownerName).
					Driver(driverName).Devices([]resourceapi.Device{}).
					SkipNodeOperations(resourceapi.SkipNodeOperationAll).
					Pool(resourceapi.ResourcePool{Name: poolName, Generation: 1, ResourceSliceCount: 1}).Obj(),
			},
		},
		"keep-slice-unchanged": {
			nodeUID: nodeUID,
			initialObjects: []runtime.Object{
				MakeResourceSlice().Name(resourceSlice1).GenerateName(generateName1).
					NodeOwnerReferences(ownerName, string(nodeUID)).NodeName(ownerName).
					Driver(driverName).Devices([]resourceapi.Device{newDevice(deviceName)}).
					Pool(resourceapi.ResourcePool{Name: poolName, Generation: 1, ResourceSliceCount: 1}).Obj(),
			},
			inputDriverResources: &DriverResources{
				Pools: map[string]Pool{
					poolName: {
						Generation: 1,
						Slices:     []Slice{{Devices: []resourceapi.Device{newDevice(deviceName)}}},
					},
				},
			},
			expectedResourceSlices: []resourceapi.ResourceSlice{
				*MakeResourceSlice().Name(resourceSlice1).GenerateName(generateName1).
					NodeOwnerReferences(ownerName, string(nodeUID)).NodeName(ownerName).
					Driver(driverName).Devices([]resourceapi.Device{newDevice(deviceName)}).
					Pool(resourceapi.ResourcePool{Name: poolName, Generation: 1, ResourceSliceCount: 1}).Obj(),
			},
		},
		"keep-taint-unchanged": {
			nodeUID: nodeUID,
			initialObjects: []runtime.Object{
				MakeResourceSlice().Name(resourceSlice1).GenerateName(generateName1).
					NodeOwnerReferences(ownerName, string(nodeUID)).NodeName(ownerName).
					Driver(driverName).
					Devices([]resourceapi.Device{
						newDevice(
							deviceName,
							resourceapi.DeviceTaint{
								Effect:    resourceapi.DeviceTaintEffectNoExecute,
								TimeAdded: &timeAdded,
							},
						)}).
					Pool(resourceapi.ResourcePool{Name: poolName, Generation: 1, ResourceSliceCount: 1}).
					Obj(),
			},
			inputDriverResources: &DriverResources{
				Pools: map[string]Pool{
					poolName: {
						Generation: 1,
						Slices: []Slice{{Devices: []resourceapi.Device{
							newDevice(
								deviceName,
								resourceapi.DeviceTaint{
									Effect: resourceapi.DeviceTaintEffectNoExecute,
									// No time added here! No need to update the slice.
								},
							),
						}}},
					},
				},
			},
			expectedResourceSlices: []resourceapi.ResourceSlice{
				*MakeResourceSlice().Name(resourceSlice1).GenerateName(generateName1).
					NodeOwnerReferences(ownerName, string(nodeUID)).NodeName(ownerName).
					Driver(driverName).
					Devices([]resourceapi.Device{
						newDevice(
							deviceName,
							resourceapi.DeviceTaint{
								Effect:    resourceapi.DeviceTaintEffectNoExecute,
								TimeAdded: &timeAdded,
							},
						)}).
					Pool(resourceapi.ResourcePool{Name: poolName, Generation: 1, ResourceSliceCount: 1}).
					Obj(),
			},
		},
		// Two taints whose TimeAdded has to survive an update triggered by
		// something else. With one taint the previous case passes either way.
		"keep-time-added-of-every-taint": {
			nodeUID: nodeUID,
			initialObjects: []runtime.Object{
				MakeResourceSlice().Name(resourceSlice1).GenerateName(generateName1).
					NodeOwnerReferences(ownerName, string(nodeUID)).NodeName(ownerName).
					Driver(driverName).
					Devices([]resourceapi.Device{
						newDevice(
							deviceName,
							resourceapi.DeviceTaint{
								Key:       "taint-a",
								Effect:    resourceapi.DeviceTaintEffectNoExecute,
								TimeAdded: &timeAdded,
							},
							resourceapi.DeviceTaint{
								Key:       "taint-b",
								Effect:    resourceapi.DeviceTaintEffectNoExecute,
								TimeAdded: &timeAddedLater,
							},
						)}).
					Pool(resourceapi.ResourcePool{Name: poolName, Generation: 1, ResourceSliceCount: 1}).
					Obj(),
			},
			inputDriverResources: &DriverResources{
				Pools: map[string]Pool{
					poolName: {
						Generation: 1,
						Slices: []Slice{{Devices: []resourceapi.Device{
							newDevice(
								deviceName,
								// The attribute is what makes the controller update the
								// slice. A change in TimeAdded alone would not.
								attrs,
								resourceapi.DeviceTaint{
									Key:    "taint-a",
									Effect: resourceapi.DeviceTaintEffectNoExecute,
								},
								resourceapi.DeviceTaint{
									Key:    "taint-b",
									Effect: resourceapi.DeviceTaintEffectNoExecute,
								},
							),
						}}},
					},
				},
			},
			expectedStats: Stats{
				NumUpdates: 1,
			},
			expectedResourceSlices: []resourceapi.ResourceSlice{
				*MakeResourceSlice().Name(resourceSlice1).GenerateName(generateName1).
					ResourceVersion("1").
					NodeOwnerReferences(ownerName, string(nodeUID)).NodeName(ownerName).
					Driver(driverName).
					Devices([]resourceapi.Device{
						newDevice(
							deviceName,
							attrs,
							resourceapi.DeviceTaint{
								Key:       "taint-a",
								Effect:    resourceapi.DeviceTaintEffectNoExecute,
								TimeAdded: &timeAdded,
							},
							resourceapi.DeviceTaint{
								Key:       "taint-b",
								Effect:    resourceapi.DeviceTaintEffectNoExecute,
								TimeAdded: &timeAddedLater,
							},
						)}).
					Pool(resourceapi.ResourcePool{Name: poolName, Generation: 1, ResourceSliceCount: 1}).
					Obj(),
			},
		},
		"add-taints": {
			nodeUID: nodeUID,
			initialObjects: []runtime.Object{
				MakeResourceSlice().Name(resourceSlice1).GenerateName(generateName1).
					NodeOwnerReferences(ownerName, string(nodeUID)).NodeName(ownerName).
					Driver(driverName).
					Devices([]resourceapi.Device{
						newDevice(
							deviceName,
							resourceapi.DeviceTaint{
								Effect:    resourceapi.DeviceTaintEffectNoExecute,
								TimeAdded: &timeAdded,
							},
						)}).
					Pool(resourceapi.ResourcePool{Name: poolName, Generation: 1, ResourceSliceCount: 1}).
					Obj(),
			},
			inputDriverResources: &DriverResources{
				Pools: map[string]Pool{
					poolName: {
						Generation: 1,
						Slices: []Slice{{Devices: []resourceapi.Device{
							newDevice(
								deviceName,
								[]resourceapi.DeviceTaint{
									{
										Effect: resourceapi.DeviceTaintEffectNoExecute,
										// No time added here! Time from existing slice must get copied during update.
									},
									{
										Key:    "example.com/tainted",
										Effect: resourceapi.DeviceTaintEffectNoSchedule,
										// No time added, will be set to timeAddedLater by reactor.
									},
									{
										Key:       "example.com/tainted2",
										Effect:    resourceapi.DeviceTaintEffectNoExecute,
										TimeAdded: &timeAddedEarlierSubSecond, // Gets rounded, both by controller and apiserver roundtripping.
									},
								},
							),
						}}},
					},
				},
			},
			expectedStats: Stats{
				NumUpdates: 1,
			},
			expectedResourceSlices: []resourceapi.ResourceSlice{
				*MakeResourceSlice().Name(resourceSlice1).GenerateName(generateName1).
					ResourceVersion("1").
					NodeOwnerReferences(ownerName, string(nodeUID)).NodeName(ownerName).
					Driver(driverName).
					Devices([]resourceapi.Device{
						newDevice(
							deviceName,
							[]resourceapi.DeviceTaint{
								{
									Effect:    resourceapi.DeviceTaintEffectNoExecute,
									TimeAdded: &timeAdded,
								},
								{
									Key:       "example.com/tainted",
									Effect:    resourceapi.DeviceTaintEffectNoSchedule,
									TimeAdded: &timeAddedLater,
								},
								{
									Key:       "example.com/tainted2",
									Effect:    resourceapi.DeviceTaintEffectNoExecute,
									TimeAdded: &timeAddedEarlier,
								},
							},
						)}).
					Pool(resourceapi.ResourcePool{Name: poolName, Generation: 1, ResourceSliceCount: 1}).
					Obj(),
			},
		},
		"drop-taints": {
			features: features{disableDeviceTaints: true},
			nodeUID:  nodeUID,
			initialObjects: []runtime.Object{
				MakeResourceSlice().Name(resourceSlice1).GenerateName(generateName1).
					NodeOwnerReferences(ownerName, string(nodeUID)).NodeName(ownerName).
					Driver(driverName).
					Devices([]resourceapi.Device{newDevice(deviceName)}).
					Pool(resourceapi.ResourcePool{Name: poolName, Generation: 1, ResourceSliceCount: 1}).
					Obj(),
			},
			inputDriverResources: &DriverResources{
				Pools: map[string]Pool{
					poolName: {
						Generation: 1,
						Slices: []Slice{{Devices: []resourceapi.Device{
							newDevice(
								deviceName,
								resourceapi.DeviceTaint{
									Effect: resourceapi.DeviceTaintEffectNoExecute,
								},
							),
						}}},
					},
				},
			},
			expectedStats: Stats{
				NumUpdates: 1,
			},
			expectedResourceSlices: []resourceapi.ResourceSlice{
				*MakeResourceSlice().Name(resourceSlice1).GenerateName(generateName1).
					ResourceVersion("1").
					NodeOwnerReferences(ownerName, string(nodeUID)).NodeName(ownerName).
					Driver(driverName).
					Devices([]resourceapi.Device{newDevice(deviceName)}).
					Pool(resourceapi.ResourcePool{Name: poolName, Generation: 1, ResourceSliceCount: 1}).
					Obj(),
			},
			expectedErrors: []string{`update ResourceSlice: pool "pool", slice #0: some fields were dropped by the apiserver, probably because these features are disabled: DRADeviceTaints`},
		},
		"drop-consumable-capacity-field": {
			features: features{disableConsumableCapacity: true},
			nodeUID:  nodeUID,
			initialObjects: []runtime.Object{
				MakeResourceSlice().Name(resourceSlice1).GenerateName(generateName1).
					NodeOwnerReferences(ownerName, string(nodeUID)).NodeName(ownerName).
					Driver(driverName).
					Devices([]resourceapi.Device{newDevice(deviceName)}).
					Pool(resourceapi.ResourcePool{Name: poolName, Generation: 1, ResourceSliceCount: 1}).
					Obj(),
			},
			inputDriverResources: &DriverResources{
				Pools: map[string]Pool{
					poolName: {
						Generation: 1,
						Slices: []Slice{{Devices: []resourceapi.Device{
							newDevice(
								deviceName,
								allowMultipleAllocationsField(true),
							),
						}}},
					},
				},
			},
			expectedStats: Stats{
				NumUpdates: 1,
			},
			expectedResourceSlices: []resourceapi.ResourceSlice{
				*MakeResourceSlice().Name(resourceSlice1).GenerateName(generateName1).
					ResourceVersion("1").
					NodeOwnerReferences(ownerName, string(nodeUID)).NodeName(ownerName).
					Driver(driverName).
					Devices([]resourceapi.Device{newDevice(deviceName)}).
					Pool(resourceapi.ResourcePool{Name: poolName, Generation: 1, ResourceSliceCount: 1}).
					Obj(),
			},
			expectedErrors: []string{`update ResourceSlice: pool "pool", slice #0: some fields were dropped by the apiserver, probably because these features are disabled: DRAConsumableCapacity`},
		},
		"drop-optional-node-operations-field": {
			features: features{disableOptionalNodeOperations: true},
			nodeUID:  nodeUID,
			initialObjects: []runtime.Object{
				MakeResourceSlice().Name(resourceSlice1).GenerateName(generateName1).
					NodeOwnerReferences(ownerName, string(nodeUID)).NodeName(ownerName).
					Driver(driverName).
					Devices([]resourceapi.Device{newDevice(deviceName)}).
					Pool(resourceapi.ResourcePool{Name: poolName, Generation: 1, ResourceSliceCount: 1}).
					Obj(),
			},
			inputDriverResources: &DriverResources{
				Pools: map[string]Pool{
					poolName: {
						Generation: 1,
						Slices: []Slice{{
							Devices:            []resourceapi.Device{newDevice(deviceName)},
							SkipNodeOperations: []resourceapi.SkipNodeOperation{resourceapi.SkipNodeOperationAll},
						}},
					},
				},
			},
			expectedStats: Stats{
				NumUpdates: 1,
			},
			expectedResourceSlices: []resourceapi.ResourceSlice{
				*MakeResourceSlice().Name(resourceSlice1).GenerateName(generateName1).
					ResourceVersion("1").
					NodeOwnerReferences(ownerName, string(nodeUID)).NodeName(ownerName).
					Driver(driverName).
					Devices([]resourceapi.Device{newDevice(deviceName)}).
					Pool(resourceapi.ResourcePool{Name: poolName, Generation: 1, ResourceSliceCount: 1}).
					Obj(),
			},
			expectedErrors: []string{`update ResourceSlice: pool "pool", slice #0: some fields were dropped by the apiserver, probably because these features are disabled: DRAOptionalNodeOperations`},
		},
		"drop-compatibility-groups-field": {
			features: features{disableCompatibilityGroups: true},
			inputDriverResources: &DriverResources{
				Pools: map[string]Pool{
					poolName: {
						Generation: 1,
						Slices: []Slice{
							{
								PerDeviceNodeSelection: new(true),
								SharedCounters: []resourceapi.CounterSet{{
									Name: "gpu-0",
									Counters: map[string]resourceapi.Counter{
										"mem": {Value: resource.MustParse("1")},
									},
								}},
							},
							{
								PerDeviceNodeSelection: new(true),
								Devices: []resourceapi.Device{
									newDevice(
										deviceName,
										nodeNameField(ownerName),
										[]resourceapi.DeviceCounterConsumption{{
											CounterSet: "gpu-0",
											Counters: map[string]resourceapi.Counter{
												"mem": {Value: resource.MustParse("1")},
											},
											CompatibilityGroups: []string{"group1", "group2"},
										}},
									),
								},
							},
						},
					},
				},
			},
			expectedStats: Stats{
				NumCreates: 2,
			},
			expectedResourceSlices: []resourceapi.ResourceSlice{
				*MakeResourceSlice().Name(resourceSlice1).GenerateName(generateName1).
					ResourceVersion("1").
					AppOwnerReferences(ownerName).
					AllNodes(false).
					NodeName("").
					NodeSelector(nil).
					PerDeviceNodeSelection(true).
					SharedCounters([]resourceapi.CounterSet{{
						Name: "gpu-0",
						Counters: map[string]resourceapi.Counter{
							"mem": {Value: resource.MustParse("1")},
						},
					}}).
					Driver(driverName).
					Pool(resourceapi.ResourcePool{Name: poolName, Generation: 1, ResourceSliceCount: 2}).
					Obj(),
				*MakeResourceSlice().Name(resourceSlice2).GenerateName(generateName2).
					ResourceVersion("1").
					AppOwnerReferences(ownerName).
					AllNodes(false).
					NodeName("").
					NodeSelector(nil).
					PerDeviceNodeSelection(true).
					Driver(driverName).
					Devices([]resourceapi.Device{
						newDevice(
							deviceName,
							nodeNameField(ownerName),
							resourceapi.DeviceCounterConsumption{
								CounterSet: "gpu-0",
								Counters: map[string]resourceapi.Counter{
									"mem": {Value: resource.MustParse("1")},
								},
							},
						),
					}).
					Pool(resourceapi.ResourcePool{Name: poolName, Generation: 1, ResourceSliceCount: 2}).
					Obj(),
			},
			expectedErrors: []string{`create ResourceSlice: pool "pool", slice #1: some fields were dropped by the apiserver, probably because these features are disabled: DRADeviceCompatibilityGroups`},
		},
		"remove-pool": {
			nodeUID:   nodeUID,
			syncDelay: ptr.To(time.Duration(0)), // Ensure that the initial object causes an immediate sync of the pool.
			initialObjects: []runtime.Object{
				MakeResourceSlice().Name(resourceSlice1).UID(resourceSlice1).
					NodeOwnerReferences(ownerName, string(nodeUID)).NodeName(ownerName).
					Driver(driverName).Devices([]resourceapi.Device{}).
					Pool(resourceapi.ResourcePool{Name: poolName, Generation: 1, ResourceSliceCount: 1}).Obj(),
			},
			inputDriverResources: &DriverResources{},
			expectedStats: Stats{
				NumDeletes: 1,
			},
			expectedResourceSlices: nil,
		},
		"one-existing-and-one-desired-slice-should-be-updated-inplace": {
			nodeUID: nodeUID,
			initialObjects: []runtime.Object{
				// no devices
				MakeResourceSlice().Name(resourceSlice1).UID(resourceSlice1).
					NodeOwnerReferences(ownerName, string(nodeUID)).NodeName(ownerName).
					Driver(driverName).Devices([]resourceapi.Device{}).
					Pool(resourceapi.ResourcePool{Name: poolName, Generation: 1, ResourceSliceCount: 1}).Obj(),
			},
			inputDriverResources: &DriverResources{
				Pools: map[string]Pool{
					poolName: {
						Slices: []Slice{{Devices: []resourceapi.Device{newDevice(deviceName)}}}},
				},
			},
			expectedStats: Stats{
				NumUpdates: 1,
			},
			expectedResourceSlices: []resourceapi.ResourceSlice{
				*MakeResourceSlice().Name(resourceSlice1).UID(resourceSlice1).ResourceVersion("1").
					NodeOwnerReferences(ownerName, string(nodeUID)).NodeName(ownerName).
					Driver(driverName).Devices([]resourceapi.Device{{Name: deviceName}}).
					Pool(resourceapi.ResourcePool{Name: poolName, Generation: 1, ResourceSliceCount: 1}).Obj(),
			},
		},
		"update-slice-when-more-than-one-existing-or-desired-slice": {
			nodeUID: nodeUID,
			initialObjects: []runtime.Object{
				// No devices in first ResourceSlice.
				MakeResourceSlice().Name(resourceSlice1).GenerateName(generateName1).UID(resourceSlice1).
					NodeOwnerReferences(ownerName, string(nodeUID)).NodeName(ownerName).
					Driver(driverName).Devices([]resourceapi.Device{}).
					Pool(resourceapi.ResourcePool{Name: poolName, Generation: 1, ResourceSliceCount: 2}).Obj(),
				MakeResourceSlice().Name(resourceSlice2).GenerateName(generateName2).UID(resourceSlice2).
					NodeOwnerReferences(ownerName, string(nodeUID)).NodeName(ownerName).
					Driver(driverName).Devices([]resourceapi.Device{{Name: deviceName2}}).
					Pool(resourceapi.ResourcePool{Name: poolName, Generation: 1, ResourceSliceCount: 2}).Obj(),
			},
			inputDriverResources: &DriverResources{
				Pools: map[string]Pool{
					poolName: {
						Slices: []Slice{
							{Devices: []resourceapi.Device{{Name: deviceName1}}},
							{Devices: []resourceapi.Device{{Name: deviceName2}}},
						},
					},
				},
			},
			expectedStats: Stats{
				NumUpdates: 1,
			},
			expectedResourceSlices: []resourceapi.ResourceSlice{
				*MakeResourceSlice().Name(resourceSlice1).GenerateName(generateName1).UID(resourceSlice1).ResourceVersion("1").
					NodeOwnerReferences(ownerName, string(nodeUID)).NodeName(ownerName).
					Driver(driverName).Devices([]resourceapi.Device{newDevice(deviceName1)}).
					Pool(resourceapi.ResourcePool{Name: poolName, Generation: 1, ResourceSliceCount: 2}).Obj(),
				*MakeResourceSlice().Name(resourceSlice2).GenerateName(generateName2).UID(resourceSlice2).
					NodeOwnerReferences(ownerName, string(nodeUID)).NodeName(ownerName).
					Driver(driverName).Devices([]resourceapi.Device{newDevice(deviceName2)}).
					Pool(resourceapi.ResourcePool{Name: poolName, Generation: 1, ResourceSliceCount: 2}).Obj(),
			},
		},
		"delete-redundant-slice": {
			nodeUID: nodeUID,
			initialObjects: []runtime.Object{
				MakeResourceSlice().Name(resourceSlice1).UID(resourceSlice1).
					NodeOwnerReferences(ownerName, string(nodeUID)).NodeName(ownerName).
					Driver(driverName).Devices([]resourceapi.Device{newDevice(deviceName)}).
					Pool(resourceapi.ResourcePool{Name: poolName, Generation: 1, ResourceSliceCount: 1}).Obj(),
				MakeResourceSlice().Name(resourceSlice2).UID(resourceSlice2).
					NodeOwnerReferences(ownerName, string(nodeUID)).NodeName(ownerName).
					Driver(driverName).Devices([]resourceapi.Device{newDevice(deviceName)}).
					Pool(resourceapi.ResourcePool{Name: poolName, Generation: 1, ResourceSliceCount: 1}).Obj(),
			},
			inputDriverResources: &DriverResources{
				Pools: map[string]Pool{
					poolName: {
						Slices: []Slice{{Devices: []resourceapi.Device{newDevice(deviceName)}}}},
				},
			},
			expectedStats: Stats{
				NumDeletes: 1,
			},
			expectedResourceSlices: []resourceapi.ResourceSlice{
				*MakeResourceSlice().Name(resourceSlice1).UID(resourceSlice1).
					NodeOwnerReferences(ownerName, string(nodeUID)).NodeName(ownerName).
					Driver(driverName).Devices([]resourceapi.Device{newDevice(deviceName)}).
					Pool(resourceapi.ResourcePool{Name: poolName, Generation: 1, ResourceSliceCount: 1}).Obj(),
			},
		},
		"update-slice": {
			nodeUID: nodeUID,
			initialObjects: []runtime.Object{
				MakeResourceSlice().Name(resourceSlice1).UID(resourceSlice1).
					NodeOwnerReferences(ownerName, string(nodeUID)).NodeName(ownerName).
					Driver(driverName).Devices([]resourceapi.Device{newDevice(deviceName)}).
					Pool(resourceapi.ResourcePool{Name: poolName, Generation: 1, ResourceSliceCount: 1}).Obj(),
			},
			inputDriverResources: &DriverResources{
				Pools: map[string]Pool{
					poolName: {
						Slices: []Slice{{
							Devices: []resourceapi.Device{newDevice(deviceName, attrs)},
						}},
					},
				},
			},
			expectedStats: Stats{
				NumUpdates: 1,
			},
			expectedResourceSlices: []resourceapi.ResourceSlice{
				*MakeResourceSlice().Name(resourceSlice1).UID(resourceSlice1).ResourceVersion("1").
					NodeOwnerReferences(ownerName, string(nodeUID)).NodeName(ownerName).
					Driver(driverName).Devices([]resourceapi.Device{newDevice(deviceName, attrs)}).
					Pool(resourceapi.ResourcePool{Name: poolName, Generation: 1, ResourceSliceCount: 1}).Obj(),
			},
		},
		"update-slice-many-devices": {
			nodeUID: nodeUID,
			initialObjects: []runtime.Object{
				MakeResourceSlice().Name(resourceSlice1).UID(resourceSlice1).
					NodeOwnerReferences(ownerName, string(nodeUID)).NodeName(ownerName).
					Driver(driverName).Devices([]resourceapi.Device{newDevice(deviceName1), newDevice(deviceName2)}).
					Pool(resourceapi.ResourcePool{Name: poolName, Generation: 1, ResourceSliceCount: 1}).Obj(),
			},
			inputDriverResources: &DriverResources{
				Pools: map[string]Pool{
					poolName: {
						Slices: []Slice{{
							Devices: []resourceapi.Device{
								newDevice(deviceName1),
								newDevice(deviceName2, attrs),
							},
						}},
					},
				},
			},
			expectedStats: Stats{
				NumUpdates: 1,
			},
			expectedResourceSlices: []resourceapi.ResourceSlice{
				*MakeResourceSlice().Name(resourceSlice1).UID(resourceSlice1).ResourceVersion("1").
					NodeOwnerReferences(ownerName, string(nodeUID)).NodeName(ownerName).
					Driver(driverName).
					Devices([]resourceapi.Device{
						newDevice(deviceName1),
						newDevice(deviceName2, attrs),
					}).
					Pool(resourceapi.ResourcePool{Name: poolName, Generation: 1, ResourceSliceCount: 1}).Obj(),
			},
		},
		"multiple-resourceslices-existing-no-changes": {
			nodeUID: nodeUID,
			initialObjects: []runtime.Object{
				MakeResourceSlice().Name(resourceSlice1).UID(resourceSlice1).
					NodeOwnerReferences(ownerName, string(nodeUID)).NodeName(ownerName).
					Driver(driverName).Devices([]resourceapi.Device{newDevice(deviceName1)}).
					Pool(resourceapi.ResourcePool{Name: poolName, Generation: 1, ResourceSliceCount: 3}).Obj(),
				MakeResourceSlice().Name(resourceSlice2).UID(resourceSlice2).
					NodeOwnerReferences(ownerName, string(nodeUID)).NodeName(ownerName).
					Driver(driverName).Devices([]resourceapi.Device{newDevice(deviceName2)}).
					Pool(resourceapi.ResourcePool{Name: poolName, Generation: 1, ResourceSliceCount: 3}).Obj(),
				MakeResourceSlice().Name(resourceSlice3).UID(resourceSlice3).
					NodeOwnerReferences(ownerName, string(nodeUID)).NodeName(ownerName).
					Driver(driverName).Devices([]resourceapi.Device{newDevice(deviceName3)}).
					Pool(resourceapi.ResourcePool{Name: poolName, Generation: 1, ResourceSliceCount: 3}).Obj(),
			},
			inputDriverResources: &DriverResources{
				Pools: map[string]Pool{
					poolName: {
						Slices: []Slice{
							{Devices: []resourceapi.Device{newDevice(deviceName1)}},
							{Devices: []resourceapi.Device{newDevice(deviceName2)}},
							{Devices: []resourceapi.Device{newDevice(deviceName3)}},
						},
					},
				},
			},
			expectedResourceSlices: []resourceapi.ResourceSlice{
				*MakeResourceSlice().Name(resourceSlice1).UID(resourceSlice1).
					NodeOwnerReferences(ownerName, string(nodeUID)).NodeName(ownerName).
					Driver(driverName).Devices([]resourceapi.Device{newDevice(deviceName1)}).
					Pool(resourceapi.ResourcePool{Name: poolName, Generation: 1, ResourceSliceCount: 3}).Obj(),
				*MakeResourceSlice().Name(resourceSlice2).UID(resourceSlice2).
					NodeOwnerReferences(ownerName, string(nodeUID)).NodeName(ownerName).
					Driver(driverName).Devices([]resourceapi.Device{newDevice(deviceName2)}).
					Pool(resourceapi.ResourcePool{Name: poolName, Generation: 1, ResourceSliceCount: 3}).Obj(),
				*MakeResourceSlice().Name(resourceSlice3).UID(resourceSlice3).
					NodeOwnerReferences(ownerName, string(nodeUID)).NodeName(ownerName).
					Driver(driverName).Devices([]resourceapi.Device{newDevice(deviceName3)}).
					Pool(resourceapi.ResourcePool{Name: poolName, Generation: 1, ResourceSliceCount: 3}).Obj(),
			},
		},
		"multiple-resourceslices-existing-with-different-resource-pool-generation": {
			nodeUID: nodeUID,
			initialObjects: []runtime.Object{
				// no devices
				MakeResourceSlice().Name(resourceSlice1).UID(resourceSlice1).
					NodeOwnerReferences(ownerName, string(nodeUID)).NodeName(ownerName).
					Driver(driverName).Devices([]resourceapi.Device{}).
					Pool(resourceapi.ResourcePool{Name: poolName, Generation: 1, ResourceSliceCount: 1}).Obj(),
				// matching device
				MakeResourceSlice().Name(resourceSlice2).UID(resourceSlice2).
					NodeOwnerReferences(ownerName, string(nodeUID)).NodeName(ownerName).
					Driver(driverName).Devices([]resourceapi.Device{newDevice(deviceName)}).
					Pool(resourceapi.ResourcePool{Name: poolName, Generation: 2, ResourceSliceCount: 1}).Obj(),
				// no devices
				MakeResourceSlice().Name(resourceSlice3).UID(resourceSlice3).
					NodeOwnerReferences(ownerName, string(nodeUID)).NodeName(ownerName).
					Driver(driverName).Devices([]resourceapi.Device{}).
					Pool(resourceapi.ResourcePool{Name: poolName, Generation: 1, ResourceSliceCount: 1}).Obj(),
			},
			inputDriverResources: &DriverResources{
				Pools: map[string]Pool{
					poolName: {
						Slices: []Slice{{
							Devices: []resourceapi.Device{
								{
									Name: deviceName,
								},
							},
						}},
					},
				},
			},
			expectedStats: Stats{
				NumCreates: 1,
				NumDeletes: 3,
			},
			expectedResourceSlices: []resourceapi.ResourceSlice{
				*MakeResourceSlice().Name(resourceSlice1).GenerateName(generateName1).
					ResourceVersion("1").
					NodeOwnerReferences(ownerName, string(nodeUID)).NodeName(ownerName).
					Driver(driverName).Devices([]resourceapi.Device{newDevice(deviceName)}).
					Pool(resourceapi.ResourcePool{Name: poolName, Generation: 3, ResourceSliceCount: 1}).Obj(),
			},
		},
		"multiple-resourceslices-existing-one-changed": {
			nodeUID: nodeUID,
			initialObjects: []runtime.Object{
				MakeResourceSlice().Name(resourceSlice1).UID(resourceSlice1).
					NodeOwnerReferences(ownerName, string(nodeUID)).NodeName(ownerName).
					Driver(driverName).Devices([]resourceapi.Device{newDevice(deviceName1)}).
					Pool(resourceapi.ResourcePool{Name: poolName, Generation: 1, ResourceSliceCount: 3}).Obj(),
				MakeResourceSlice().Name(resourceSlice2).UID(resourceSlice2).
					NodeOwnerReferences(ownerName, string(nodeUID)).NodeName(ownerName).
					Driver(driverName).Devices([]resourceapi.Device{newDevice(deviceName2)}).
					Pool(resourceapi.ResourcePool{Name: poolName, Generation: 1, ResourceSliceCount: 3}).Obj(),
				MakeResourceSlice().Name(resourceSlice3).UID(resourceSlice3).
					NodeOwnerReferences(ownerName, string(nodeUID)).NodeName(ownerName).
					Driver(driverName).Devices([]resourceapi.Device{newDevice(deviceName3)}).
					Pool(resourceapi.ResourcePool{Name: poolName, Generation: 1, ResourceSliceCount: 3}).Obj(),
			},
			inputDriverResources: &DriverResources{
				Pools: map[string]Pool{
					poolName: {
						Slices: []Slice{
							{Devices: []resourceapi.Device{newDevice(deviceName1)}},
							{Devices: []resourceapi.Device{newDevice(deviceName2, attrs)}},
							{Devices: []resourceapi.Device{newDevice(deviceName3)}},
						},
					},
				},
			},
			// Generation not bumped, only one update.
			expectedStats: Stats{
				NumUpdates: 1,
			},
			expectedResourceSlices: []resourceapi.ResourceSlice{
				*MakeResourceSlice().Name(resourceSlice1).UID(resourceSlice1).
					NodeOwnerReferences(ownerName, string(nodeUID)).NodeName(ownerName).
					Driver(driverName).Devices([]resourceapi.Device{newDevice(deviceName1)}).
					Pool(resourceapi.ResourcePool{Name: poolName, Generation: 1, ResourceSliceCount: 3}).Obj(),
				*MakeResourceSlice().Name(resourceSlice2).UID(resourceSlice2).ResourceVersion("1").
					NodeOwnerReferences(ownerName, string(nodeUID)).NodeName(ownerName).
					Driver(driverName).Devices([]resourceapi.Device{newDevice(deviceName2, attrs)}).
					Pool(resourceapi.ResourcePool{Name: poolName, Generation: 1, ResourceSliceCount: 3}).Obj(),
				*MakeResourceSlice().Name(resourceSlice3).UID(resourceSlice3).
					NodeOwnerReferences(ownerName, string(nodeUID)).NodeName(ownerName).
					Driver(driverName).Devices([]resourceapi.Device{newDevice(deviceName3)}).
					Pool(resourceapi.ResourcePool{Name: poolName, Generation: 1, ResourceSliceCount: 3}).Obj(),
			},
		},
		"multiple-resourceslices-existing-two-changed": {
			nodeUID: nodeUID,
			initialObjects: []runtime.Object{
				MakeResourceSlice().Name(resourceSlice1).UID(resourceSlice1).
					NodeOwnerReferences(ownerName, string(nodeUID)).NodeName(ownerName).
					Driver(driverName).Devices([]resourceapi.Device{newDevice(deviceName1)}).
					Pool(resourceapi.ResourcePool{Name: poolName, Generation: 1, ResourceSliceCount: 3}).Obj(),
				MakeResourceSlice().Name(resourceSlice2).UID(resourceSlice2).
					NodeOwnerReferences(ownerName, string(nodeUID)).NodeName(ownerName).
					Driver(driverName).Devices([]resourceapi.Device{newDevice(deviceName2)}).
					Pool(resourceapi.ResourcePool{Name: poolName, Generation: 1, ResourceSliceCount: 3}).Obj(),
				MakeResourceSlice().Name(resourceSlice3).UID(resourceSlice3).
					NodeOwnerReferences(ownerName, string(nodeUID)).NodeName(ownerName).
					Driver(driverName).Devices([]resourceapi.Device{newDevice(deviceName3)}).
					Pool(resourceapi.ResourcePool{Name: poolName, Generation: 1, ResourceSliceCount: 3}).Obj(),
			},
			inputDriverResources: &DriverResources{
				Pools: map[string]Pool{
					poolName: {
						Slices: []Slice{
							{Devices: []resourceapi.Device{newDevice(deviceName1)}},
							{Devices: []resourceapi.Device{newDevice(deviceName2, attrs)}},
							{Devices: []resourceapi.Device{newDevice(deviceName3, attrs)}},
						},
					},
				},
			},
			// Generation bumped, all updated.
			expectedStats: Stats{
				NumUpdates: 3,
			},
			expectedResourceSlices: []resourceapi.ResourceSlice{
				*MakeResourceSlice().Name(resourceSlice1).UID(resourceSlice1).
					NodeOwnerReferences(ownerName, string(nodeUID)).NodeName(ownerName).ResourceVersion("1").
					Driver(driverName).Devices([]resourceapi.Device{newDevice(deviceName1)}).
					Pool(resourceapi.ResourcePool{Name: poolName, Generation: 2, ResourceSliceCount: 3}).Obj(),
				*MakeResourceSlice().Name(resourceSlice2).UID(resourceSlice2).ResourceVersion("1").
					NodeOwnerReferences(ownerName, string(nodeUID)).NodeName(ownerName).
					Driver(driverName).Devices([]resourceapi.Device{newDevice(deviceName2, attrs)}).
					Pool(resourceapi.ResourcePool{Name: poolName, Generation: 2, ResourceSliceCount: 3}).Obj(),
				*MakeResourceSlice().Name(resourceSlice3).UID(resourceSlice3).
					NodeOwnerReferences(ownerName, string(nodeUID)).NodeName(ownerName).ResourceVersion("1").
					Driver(driverName).Devices([]resourceapi.Device{newDevice(deviceName3, attrs)}).
					Pool(resourceapi.ResourcePool{Name: poolName, Generation: 2, ResourceSliceCount: 3}).Obj(),
			},
		},
		"multiple-resourceslices-existing-one-removed": {
			nodeUID: nodeUID,
			initialObjects: []runtime.Object{
				MakeResourceSlice().Name(resourceSlice1).UID(resourceSlice1).
					NodeOwnerReferences(ownerName, string(nodeUID)).NodeName(ownerName).
					Driver(driverName).Devices([]resourceapi.Device{newDevice(deviceName1)}).
					Pool(resourceapi.ResourcePool{Name: poolName, Generation: 1, ResourceSliceCount: 3}).Obj(),
				MakeResourceSlice().Name(resourceSlice2).UID(resourceSlice2).
					NodeOwnerReferences(ownerName, string(nodeUID)).NodeName(ownerName).
					Driver(driverName).Devices([]resourceapi.Device{newDevice(deviceName2)}).
					Pool(resourceapi.ResourcePool{Name: poolName, Generation: 1, ResourceSliceCount: 3}).Obj(),
				MakeResourceSlice().Name(resourceSlice3).UID(resourceSlice3).
					NodeOwnerReferences(ownerName, string(nodeUID)).NodeName(ownerName).
					Driver(driverName).Devices([]resourceapi.Device{newDevice(deviceName3)}).
					Pool(resourceapi.ResourcePool{Name: poolName, Generation: 1, ResourceSliceCount: 3}).Obj(),
			},
			inputDriverResources: &DriverResources{
				Pools: map[string]Pool{
					poolName: {
						Slices: []Slice{
							{Devices: []resourceapi.Device{newDevice(deviceName1)}},
							{Devices: []resourceapi.Device{newDevice(deviceName2)}},
						},
					},
				},
			},
			// Generation bumped, two updated, one removed.
			expectedStats: Stats{
				NumUpdates: 2,
				NumDeletes: 1,
			},
			expectedResourceSlices: []resourceapi.ResourceSlice{
				*MakeResourceSlice().Name(resourceSlice1).UID(resourceSlice1).
					NodeOwnerReferences(ownerName, string(nodeUID)).NodeName(ownerName).ResourceVersion("1").
					Driver(driverName).Devices([]resourceapi.Device{newDevice(deviceName1)}).
					Pool(resourceapi.ResourcePool{Name: poolName, Generation: 2, ResourceSliceCount: 2}).Obj(),
				*MakeResourceSlice().Name(resourceSlice2).UID(resourceSlice2).ResourceVersion("1").
					NodeOwnerReferences(ownerName, string(nodeUID)).NodeName(ownerName).
					Driver(driverName).Devices([]resourceapi.Device{newDevice(deviceName2)}).
					Pool(resourceapi.ResourcePool{Name: poolName, Generation: 2, ResourceSliceCount: 2}).Obj(),
			},
		},
		"multiple-resourceslices-existing-one-added": {
			nodeUID: nodeUID,
			initialObjects: []runtime.Object{
				MakeResourceSlice().Name(resourceSlice1).UID(resourceSlice1).
					NodeOwnerReferences(ownerName, string(nodeUID)).NodeName(ownerName).
					Driver(driverName).Devices([]resourceapi.Device{newDevice(deviceName1)}).
					Pool(resourceapi.ResourcePool{Name: poolName, Generation: 1, ResourceSliceCount: 3}).Obj(),
				MakeResourceSlice().Name(resourceSlice2).UID(resourceSlice2).
					NodeOwnerReferences(ownerName, string(nodeUID)).NodeName(ownerName).
					Driver(driverName).Devices([]resourceapi.Device{newDevice(deviceName2)}).
					Pool(resourceapi.ResourcePool{Name: poolName, Generation: 1, ResourceSliceCount: 3}).Obj(),
				MakeResourceSlice().Name(resourceSlice3).UID(resourceSlice3).
					NodeOwnerReferences(ownerName, string(nodeUID)).NodeName(ownerName).
					Driver(driverName).Devices([]resourceapi.Device{newDevice(deviceName3)}).
					Pool(resourceapi.ResourcePool{Name: poolName, Generation: 1, ResourceSliceCount: 3}).Obj(),
			},
			inputDriverResources: &DriverResources{
				Pools: map[string]Pool{
					poolName: {
						Slices: []Slice{
							{Devices: []resourceapi.Device{newDevice(deviceName1)}},
							{Devices: []resourceapi.Device{newDevice(deviceName2)}},
							{Devices: []resourceapi.Device{newDevice(deviceName3)}},
							{Devices: []resourceapi.Device{newDevice(deviceName4)}},
						},
					},
				},
			},
			expectedStats: Stats{
				NumUpdates: 3,
				NumCreates: 1,
			},
			expectedResourceSlices: []resourceapi.ResourceSlice{
				*MakeResourceSlice().Name(resourceSlice1).UID(resourceSlice1).ResourceVersion("1").
					NodeOwnerReferences(ownerName, string(nodeUID)).NodeName(ownerName).
					Driver(driverName).Devices([]resourceapi.Device{newDevice(deviceName1)}).
					Pool(resourceapi.ResourcePool{Name: poolName, Generation: 2, ResourceSliceCount: 4}).Obj(),
				*MakeResourceSlice().Name(resourceSlice2).UID(resourceSlice2).ResourceVersion("1").
					NodeOwnerReferences(ownerName, string(nodeUID)).NodeName(ownerName).
					Driver(driverName).Devices([]resourceapi.Device{newDevice(deviceName2)}).
					Pool(resourceapi.ResourcePool{Name: poolName, Generation: 2, ResourceSliceCount: 4}).Obj(),
				*MakeResourceSlice().Name(resourceSlice3).UID(resourceSlice3).ResourceVersion("1").
					NodeOwnerReferences(ownerName, string(nodeUID)).NodeName(ownerName).
					Driver(driverName).Devices([]resourceapi.Device{newDevice(deviceName3)}).
					Pool(resourceapi.ResourcePool{Name: poolName, Generation: 2, ResourceSliceCount: 4}).Obj(),
				*MakeResourceSlice().Name(generateName4+"0").GenerateName(generateName4).
					ResourceVersion("1").
					NodeOwnerReferences(ownerName, string(nodeUID)).NodeName(ownerName).
					Driver(driverName).Devices([]resourceapi.Device{newDevice(deviceName4)}).
					Pool(resourceapi.ResourcePool{Name: poolName, Generation: 2, ResourceSliceCount: 4}).Obj(),
			},
		},
		"add-one-network-device-all-nodes": {
			initialObjects: []runtime.Object{},
			noOwner:        true,
			inputDriverResources: &DriverResources{
				Pools: map[string]Pool{
					poolName: {
						Slices:   []Slice{{Devices: []resourceapi.Device{newDevice(deviceName)}}},
						AllNodes: true,
					},
				},
			},
			expectedStats: Stats{
				NumCreates: 1,
			},
			expectedResourceSlices: []resourceapi.ResourceSlice{
				*MakeResourceSlice().Name(encodeIndex(0, resourceSliceIndexMinLength) + "-" + driverName + "-0").
					GenerateName(encodeIndex(0, resourceSliceIndexMinLength) + "-" + driverName + "-").
					ResourceVersion("1").
					AllNodes(true).
					NodeName("").
					NodeSelector(nil).
					PerDeviceNodeSelection(false).
					Driver(driverName).Devices([]resourceapi.Device{newDevice(deviceName)}).
					Pool(resourceapi.ResourcePool{Name: poolName, Generation: 1, ResourceSliceCount: 1}).Obj(),
			},
		},
		"add-one-network-device-some-nodes": {
			initialObjects: []runtime.Object{},
			inputDriverResources: &DriverResources{
				Pools: map[string]Pool{
					poolName: {
						NodeSelector: nodeSelector,
						Slices:       []Slice{{Devices: []resourceapi.Device{newDevice(deviceName)}}},
					},
				},
			},
			expectedStats: Stats{
				NumCreates: 1,
			},
			expectedResourceSlices: []resourceapi.ResourceSlice{
				*MakeResourceSlice().Name(generateName1 + "0").GenerateName(generateName1).
					ResourceVersion("1").
					AppOwnerReferences(ownerName).NodeSelector(nodeSelector).
					Driver(driverName).Devices([]resourceapi.Device{newDevice(deviceName)}).
					Pool(resourceapi.ResourcePool{Name: poolName, Generation: 1, ResourceSliceCount: 1}).Obj(),
			},
		},
		"reconcile-with-pool-name-create-slice-no-node-name": {
			nodeUID:               nodeUID,
			reconcilePoolWithName: poolName,
			initialObjects:        []runtime.Object{},
			inputDriverResources: &DriverResources{
				Pools: map[string]Pool{
					poolName: {
						Slices:   []Slice{{Devices: []resourceapi.Device{}}},
						AllNodes: true,
					},
				},
			},
			expectedStats: Stats{NumCreates: 1},
			expectedResourceSlices: []resourceapi.ResourceSlice{
				*MakeResourceSlice().Name(resourceSlice1).GenerateName(generateName1).
					ResourceVersion("1").
					NodeOwnerReferences(ownerName, string(nodeUID)).
					AllNodes(true).
					NodeName("").
					NodeSelector(nil).
					PerDeviceNodeSelection(false).
					Driver(driverName).Devices([]resourceapi.Device{}).
					Pool(resourceapi.ResourcePool{Name: poolName, Generation: 1, ResourceSliceCount: 1}).Obj(),
			},
		},
		"reconcile-with-pool-name-with-node-selector": {
			nodeUID:               nodeUID,
			reconcilePoolWithName: poolName,
			initialObjects:        []runtime.Object{},
			inputDriverResources: &DriverResources{
				Pools: map[string]Pool{
					poolName: {
						NodeSelector: nodeSelector,
						Slices:       []Slice{{Devices: []resourceapi.Device{newDevice(deviceName)}}},
					},
				},
			},
			expectedStats: Stats{NumCreates: 1},
			expectedResourceSlices: []resourceapi.ResourceSlice{
				*MakeResourceSlice().Name(resourceSlice1).GenerateName(generateName1).
					ResourceVersion("1").
					NodeOwnerReferences(ownerName, string(nodeUID)).
					NodeSelector(nodeSelector).
					Driver(driverName).Devices([]resourceapi.Device{newDevice(deviceName)}).
					Pool(resourceapi.ResourcePool{Name: poolName, Generation: 1, ResourceSliceCount: 1}).Obj(),
			},
		},
		"reconcile-with-pool-name-with-perdevice-node-selector": {
			nodeUID:               nodeUID,
			reconcilePoolWithName: poolName,
			initialObjects:        []runtime.Object{},
			inputDriverResources: &DriverResources{
				Pools: map[string]Pool{
					poolName: {
						Slices: []Slice{{Devices: []resourceapi.Device{newDevice(deviceName, nodeSelector)}, PerDeviceNodeSelection: ptr.To(true)}},
					},
				},
			},
			expectedStats: Stats{NumCreates: 1},
			expectedResourceSlices: []resourceapi.ResourceSlice{
				*MakeResourceSlice().Name(resourceSlice1).GenerateName(generateName1).
					ResourceVersion("1").
					NodeOwnerReferences(ownerName, string(nodeUID)).
					PerDeviceNodeSelection(true).
					Driver(driverName).Devices([]resourceapi.Device{newDevice(deviceName, nodeSelector)}).
					Pool(resourceapi.ResourcePool{Name: poolName, Generation: 1, ResourceSliceCount: 1}).Obj(),
			},
		},
		"reconcile-with-pool-name-other-pools": {
			nodeUID:               nodeUID,
			reconcilePoolWithName: poolName,
			initialObjects:        []runtime.Object{},
			inputDriverResources: &DriverResources{
				Pools: map[string]Pool{
					poolName: {
						Slices:   []Slice{{Devices: []resourceapi.Device{}}},
						AllNodes: true,
					},
					"other-pool-1": {Slices: []Slice{{Devices: []resourceapi.Device{}}}},
					"other-pool-2": {Slices: []Slice{{Devices: []resourceapi.Device{newDevice(deviceName), newDevice(deviceName)}}}},
				},
			},
			expectedStats: Stats{NumCreates: 1},
			expectedResourceSlices: []resourceapi.ResourceSlice{
				*MakeResourceSlice().Name(resourceSlice1).GenerateName(generateName1).
					ResourceVersion("1").
					NodeOwnerReferences(ownerName, string(nodeUID)).
					AllNodes(true).
					NodeName("").
					NodeSelector(nil).
					PerDeviceNodeSelection(false).
					Driver(driverName).Devices([]resourceapi.Device{}).
					Pool(resourceapi.ResourcePool{Name: poolName, Generation: 1, ResourceSliceCount: 1}).Obj(),
			},
			expectedErrors: []string{
				`pool validation failed: found pool "other-pool-1", but ReconcilePoolWithName only allows pool "pool"`,
				`pool validation failed: found pool "other-pool-2", but ReconcilePoolWithName only allows pool "pool"`,
			},
		},
		"reconcile-with-pool-name-wrong-pool": {
			nodeUID:               nodeUID,
			reconcilePoolWithName: poolName,
			syncDelay:             new(time.Duration(0)), // Ensure that the initial object causes an immediate sync of the pool.
			initialObjects: []runtime.Object{
				MakeResourceSlice().Name(resourceSlice1).UID(resourceSlice1).
					NodeOwnerReferences(ownerName, string(nodeUID)).
					Driver(driverName).Devices([]resourceapi.Device{}).
					Pool(resourceapi.ResourcePool{Name: poolName, Generation: 1, ResourceSliceCount: 1}).Obj(),
			},
			inputDriverResources: &DriverResources{
				Pools: map[string]Pool{
					"other-pool": {Slices: []Slice{{Devices: []resourceapi.Device{}}}},
				},
			},
			expectedStats:  Stats{NumDeletes: 1},
			expectedErrors: []string{`pool validation failed: found pool "other-pool", but ReconcilePoolWithName only allows pool "pool"`},
		},
		"reconcile-with-pool-name-invalid-pool": {
			nodeUID:               nodeUID,
			reconcilePoolWithName: poolName,
			inputDriverResources: &DriverResources{
				Pools: map[string]Pool{
					poolName: {Slices: []Slice{{Devices: []resourceapi.Device{newDevice(deviceName), newDevice(deviceName)}}}},
				},
			},
			expectedErrors: []string{`pool validation failed: found duplicate device "device" in pool "pool"`},
		},
		"reconcile-with-pool-name-no-pools": {
			nodeUID:               nodeUID,
			reconcilePoolWithName: poolName,
			syncDelay:             new(time.Duration(0)), // Ensure that the initial object causes an immediate sync of the pool.
			initialObjects: []runtime.Object{
				MakeResourceSlice().Name(resourceSlice1).UID(resourceSlice1).
					NodeOwnerReferences(ownerName, string(nodeUID)).
					Driver(driverName).Devices([]resourceapi.Device{}).
					Pool(resourceapi.ResourcePool{Name: poolName, Generation: 1, ResourceSliceCount: 1}).Obj(),
			},
			inputDriverResources: &DriverResources{Pools: map[string]Pool{}},
			expectedStats:        Stats{NumDeletes: 1},
		},
		"update-node-selector": {
			initialObjects: []runtime.Object{
				MakeResourceSlice().Name(resourceSlice1).UID(resourceSlice1).
					AppOwnerReferences(ownerName).NodeSelector(nodeSelector).
					Driver(driverName).Devices([]resourceapi.Device{newDevice(deviceName)}).
					Pool(resourceapi.ResourcePool{Name: poolName, Generation: 1, ResourceSliceCount: 1}).Obj(),
			},
			inputDriverResources: &DriverResources{
				Pools: map[string]Pool{
					poolName: {
						NodeSelector: otherNodeSelector,
						Slices:       []Slice{{Devices: []resourceapi.Device{newDevice(deviceName)}}},
					},
				},
			},
			expectedStats: Stats{
				NumUpdates: 1,
			},
			expectedResourceSlices: []resourceapi.ResourceSlice{
				*MakeResourceSlice().Name(resourceSlice1).UID(resourceSlice1).ResourceVersion("1").
					AppOwnerReferences(ownerName).NodeSelector(otherNodeSelector).
					Driver(driverName).Devices([]resourceapi.Device{newDevice(deviceName)}).
					Pool(resourceapi.ResourcePool{Name: poolName, Generation: 1, ResourceSliceCount: 1}).Obj(),
			},
		},
		"create-partitionable-devices": {
			inputDriverResources: &DriverResources{
				Pools: map[string]Pool{
					poolName: {
						Generation: 1,
						Slices: []Slice{
							{
								PerDeviceNodeSelection: ptr.To(true),
								SharedCounters: []resourceapi.CounterSet{{
									Name: "gpu-0",
									Counters: map[string]resourceapi.Counter{
										"mem": {Value: resource.MustParse("1")},
									},
								}},
							},
							{
								PerDeviceNodeSelection: ptr.To(true),
								Devices: []resourceapi.Device{
									newDevice(
										deviceName,
										nodeNameField(ownerName),
										[]resourceapi.DeviceCounterConsumption{{
											CounterSet: "gpu-0",
											Counters: map[string]resourceapi.Counter{
												"mem": {Value: resource.MustParse("1")},
											},
										}},
									),
								},
							},
						},
					},
				},
			},
			expectedStats: Stats{
				NumCreates: 2,
			},
			expectedResourceSlices: []resourceapi.ResourceSlice{
				*MakeResourceSlice().Name(resourceSlice1).GenerateName(generateName1).
					ResourceVersion("1").
					AppOwnerReferences(ownerName).
					AllNodes(false).
					NodeName("").
					NodeSelector(nil).
					PerDeviceNodeSelection(true).
					SharedCounters([]resourceapi.CounterSet{{
						Name: "gpu-0",
						Counters: map[string]resourceapi.Counter{
							"mem": {Value: resource.MustParse("1")},
						},
					}}).
					Driver(driverName).
					Pool(resourceapi.ResourcePool{Name: poolName, Generation: 1, ResourceSliceCount: 2}).
					Obj(),
				*MakeResourceSlice().Name(resourceSlice2).GenerateName(generateName2).
					ResourceVersion("1").
					AppOwnerReferences(ownerName).
					AllNodes(false).
					NodeName("").
					NodeSelector(nil).
					PerDeviceNodeSelection(true).
					Driver(driverName).
					Devices([]resourceapi.Device{
						newDevice(
							deviceName,
							nodeNameField(ownerName),
							resourceapi.DeviceCounterConsumption{
								CounterSet: "gpu-0",
								Counters: map[string]resourceapi.Counter{
									"mem": {Value: resource.MustParse("1")},
								},
							},
						),
					}).
					Pool(resourceapi.ResourcePool{Name: poolName, Generation: 1, ResourceSliceCount: 2}).
					Obj(),
			},
		},
		"drop-partitionable-devices": {
			features: features{disablePartitionableDevices: true},
			inputDriverResources: &DriverResources{
				Pools: map[string]Pool{
					poolName: {
						Generation: 1,
						Slices: []Slice{
							{
								PerDeviceNodeSelection: ptr.To(true),
								SharedCounters: []resourceapi.CounterSet{{
									Name: "gpu-0",
									Counters: map[string]resourceapi.Counter{
										"mem": {Value: resource.MustParse("1")},
									},
								}},
							},
							{
								PerDeviceNodeSelection: ptr.To(true),
								Devices: []resourceapi.Device{
									newDevice(
										deviceName,
										nodeNameField(ownerName),
										resourceapi.DeviceCounterConsumption{
											CounterSet: "gpu-0",
											Counters: map[string]resourceapi.Counter{
												"mem": {Value: resource.MustParse("1")},
											},
										},
									),
								},
							},
						},
					},
				},
			},
			expectedStats: Stats{
				NumCreates: 2,
			},
			expectedResourceSlices: []resourceapi.ResourceSlice{
				*MakeResourceSlice().Name(resourceSlice1).GenerateName(generateName1).
					ResourceVersion("1").
					AppOwnerReferences(ownerName).
					AllNodes(false).
					NodeName("").
					NodeSelector(nil).
					PerDeviceNodeSelection(false). // Should be dropped.
					Driver(driverName).
					Pool(resourceapi.ResourcePool{Name: poolName, Generation: 1, ResourceSliceCount: 2}).
					Obj(),
				*MakeResourceSlice().Name(resourceSlice2).GenerateName(generateName2).
					ResourceVersion("1").
					AppOwnerReferences(ownerName).
					AllNodes(false).
					NodeName("").
					NodeSelector(nil).
					PerDeviceNodeSelection(false). // Should be dropped.
					Driver(driverName).
					Devices([]resourceapi.Device{newDevice(deviceName)}).
					Pool(resourceapi.ResourcePool{Name: poolName, Generation: 1, ResourceSliceCount: 2}).
					Obj(),
			},
			expectedErrors: []string{
				`create ResourceSlice: pool "pool", slice #0: some fields were dropped by the apiserver, probably because these features are disabled: DRAPartitionableDevices`,
				`create ResourceSlice: pool "pool", slice #1: some fields were dropped by the apiserver, probably because these features are disabled: DRAPartitionableDevices`,
			},
		},
		"create-device-with-binding-condition": {
			nodeUID: nodeUID,
			inputDriverResources: &DriverResources{
				Pools: map[string]Pool{
					poolName: {
						Generation: 1,
						Slices: []Slice{{
							Devices: func() []resourceapi.Device {
								d := newDevice(deviceName)
								d.BindingConditions = []string{"condition1", "condition2"}
								d.BindingFailureConditions = []string{"failure-condition1"}
								d.BindsToNode = ptr.To(true)
								return []resourceapi.Device{d}
							}(),
						}},
					},
				},
			},
			expectedStats: Stats{
				NumCreates: 1,
			},
			expectedResourceSlices: []resourceapi.ResourceSlice{
				*MakeResourceSlice().Name(resourceSlice1).GenerateName(generateName1).
					ResourceVersion("1").
					NodeOwnerReferences(ownerName, string(nodeUID)).NodeName(ownerName).
					Driver(driverName).
					Devices(func() []resourceapi.Device {
						d := newDevice(deviceName)
						d.BindingConditions = []string{"condition1", "condition2"}
						d.BindingFailureConditions = []string{"failure-condition1"}
						d.BindsToNode = ptr.To(true)
						return []resourceapi.Device{d}
					}()).
					Pool(resourceapi.ResourcePool{Name: poolName, Generation: 1, ResourceSliceCount: 1}).
					Obj(),
			},
		},
		"drop-device-with-binding-condition": {
			features: features{disableBindingConditions: true},
			nodeUID:  nodeUID,
			inputDriverResources: &DriverResources{
				Pools: map[string]Pool{
					poolName: {
						Generation: 1,
						Slices: []Slice{{
							Devices: func() []resourceapi.Device {
								d := newDevice(deviceName)
								d.BindingConditions = []string{"condition1", "condition2"}
								d.BindingFailureConditions = []string{"failure-condition1"}
								d.BindsToNode = ptr.To(true)
								return []resourceapi.Device{d}
							}(),
						}},
					},
				},
			},
			expectedStats: Stats{
				NumCreates: 1,
			},
			expectedResourceSlices: []resourceapi.ResourceSlice{
				*MakeResourceSlice().Name(resourceSlice1).GenerateName(generateName1).
					ResourceVersion("1").
					NodeOwnerReferences(ownerName, string(nodeUID)).NodeName(ownerName).
					Driver(driverName).
					Devices([]resourceapi.Device{newDevice(deviceName)}).
					Pool(resourceapi.ResourcePool{Name: poolName, Generation: 1, ResourceSliceCount: 1}).
					Obj(),
			},
			expectedErrors: []string{`create ResourceSlice: pool "pool", slice #0: some fields were dropped by the apiserver, probably because these features are disabled: DRADeviceBindingConditions`},
		},
		"detect-resource-pools-with-duplicate-counter-sets": {
			nodeUID: nodeUID,
			inputDriverResources: &DriverResources{
				Pools: map[string]Pool{
					poolName: {
						Generation: 1,
						Slices: []Slice{
							{
								SharedCounters: []resourceapi.CounterSet{
									{
										Name: "counterset",
									},
								},
							},
							{
								SharedCounters: []resourceapi.CounterSet{
									{
										Name: "counterset",
									},
								},
							},
						},
					},
				},
			},
			expectedStats: Stats{
				NumCreates: 0,
			},
			expectedErrors: []string{`pool validation failed: found duplicate counter set "counterset" in pool "pool"`},
		},
		"detect-duplicate-devices": {
			nodeUID: nodeUID,
			inputDriverResources: &DriverResources{
				Pools: map[string]Pool{
					poolName: {
						Generation: 1,
						Slices: []Slice{
							{
								Devices: []resourceapi.Device{
									{
										Name: deviceName,
									},
								},
							},
							{
								Devices: []resourceapi.Device{
									{
										Name: deviceName,
									},
								},
							},
						},
					},
				},
			},
			expectedStats: Stats{
				NumCreates: 0,
			},
			expectedErrors: []string{`pool validation failed: found duplicate device "device" in pool "pool"`},
		},
		"detect-device-referencing-unknown-counter-set": {
			nodeUID: nodeUID,
			inputDriverResources: &DriverResources{
				Pools: map[string]Pool{
					poolName: {
						Generation: 1,
						Slices: []Slice{
							{
								Devices: []resourceapi.Device{
									{
										Name: deviceName,
										ConsumesCounters: []resourceapi.DeviceCounterConsumption{
											{
												CounterSet: "counterset",
											},
										},
									},
								},
							},
						},
					},
				},
			},
			expectedStats: Stats{
				NumCreates: 0,
			},
			expectedErrors: []string{`pool validation failed: counter set "counterset" referenced by device "device" not found`},
		},
		"detect-device-referencing-unknown-counter-in-counter-set": {
			nodeUID: nodeUID,
			inputDriverResources: &DriverResources{
				Pools: map[string]Pool{
					poolName: {
						Generation: 1,
						Slices: []Slice{
							{
								SharedCounters: []resourceapi.CounterSet{
									{
										Name: "counterset",
										Counters: map[string]resourceapi.Counter{
											"memory": {
												Value: resource.MustParse("8Gi"),
											},
										},
									},
								},
							},
							{
								Devices: []resourceapi.Device{
									{
										Name: "device",
										ConsumesCounters: []resourceapi.DeviceCounterConsumption{
											{
												CounterSet: "counterset",
												Counters: map[string]resourceapi.Counter{
													"cpu": {
														Value: resource.MustParse("4"),
													},
												},
											},
										},
									},
								},
							},
						},
					},
				},
			},
			expectedStats: Stats{
				NumCreates: 0,
			},
			expectedErrors: []string{`pool validation failed: counter "cpu" referenced by device "device" not found in counter set "counterset"`},
		},
		"detect-device-attribute-qualified-with-driver-name": {
			nodeUID: nodeUID,
			inputDriverResources: &DriverResources{
				Pools: map[string]Pool{
					poolName: {
						Generation: 1,
						Slices: []Slice{
							{
								Devices: []resourceapi.Device{
									{
										Name: deviceName,
										Attributes: map[resourceapi.QualifiedName]resourceapi.DeviceAttribute{
											resourceapi.QualifiedName(driverName + "/foo"): {StringValue: new("bar")},
										},
									},
								},
							},
						},
					},
				},
			},
			expectedStats: Stats{
				NumCreates: 0,
			},
			expectedErrors: []string{`pool validation failed: pool "pool": device "device": attribute "driver/foo": use unqualified "foo", the driver name is added implicitly`},
		},
		"migration-from-random-naming-to-index-based-naming": {
			nodeUID: nodeUID,
			initialObjects: []runtime.Object{
				MakeResourceSlice().Name("random-name1").
					NodeOwnerReferences(ownerName, string(nodeUID)).NodeName(ownerName).
					Driver(driverName).Devices([]resourceapi.Device{newDevice(deviceName1)}).
					Pool(resourceapi.ResourcePool{Name: poolName, Generation: 1, ResourceSliceCount: 3}).Obj(),
				MakeResourceSlice().Name("random-name2").
					NodeOwnerReferences(ownerName, string(nodeUID)).NodeName(ownerName).
					Driver(driverName).Devices([]resourceapi.Device{newDevice(deviceName2)}).
					Pool(resourceapi.ResourcePool{Name: poolName, Generation: 1, ResourceSliceCount: 3}).Obj(),
				MakeResourceSlice().Name("random-name3").
					NodeOwnerReferences(ownerName, string(nodeUID)).NodeName(ownerName).
					Driver(driverName).Devices([]resourceapi.Device{newDevice(deviceName3)}).
					Pool(resourceapi.ResourcePool{Name: poolName, Generation: 1, ResourceSliceCount: 3}).Obj(),
			},
			inputDriverResources: &DriverResources{
				Pools: map[string]Pool{
					poolName: {
						Slices: []Slice{
							{Devices: []resourceapi.Device{newDevice(deviceName1)}},
							{Devices: []resourceapi.Device{newDevice(deviceName2)}},
							{Devices: []resourceapi.Device{newDevice(deviceName3)}},
						},
					},
				},
			},
			expectedStats: Stats{
				NumCreates: 3,
				NumDeletes: 3,
			},
			expectedResourceSlices: []resourceapi.ResourceSlice{
				*MakeResourceSlice().Name(resourceSlice1).GenerateName(generateName1).
					ResourceVersion("1").
					NodeOwnerReferences(ownerName, string(nodeUID)).NodeName(ownerName).
					Driver(driverName).Devices([]resourceapi.Device{newDevice(deviceName1)}).
					Pool(resourceapi.ResourcePool{Name: poolName, Generation: 2, ResourceSliceCount: 3}).Obj(),
				*MakeResourceSlice().Name(resourceSlice2).GenerateName(generateName2).
					ResourceVersion("1").
					NodeOwnerReferences(ownerName, string(nodeUID)).NodeName(ownerName).
					Driver(driverName).Devices([]resourceapi.Device{newDevice(deviceName2)}).
					Pool(resourceapi.ResourcePool{Name: poolName, Generation: 2, ResourceSliceCount: 3}).Obj(),
				*MakeResourceSlice().Name(resourceSlice3).GenerateName(generateName3).
					ResourceVersion("1").
					NodeOwnerReferences(ownerName, string(nodeUID)).NodeName(ownerName).
					Driver(driverName).Devices([]resourceapi.Device{newDevice(deviceName3)}).
					Pool(resourceapi.ResourcePool{Name: poolName, Generation: 2, ResourceSliceCount: 3}).Obj(),
			},
		},
		"migration-mixed-naming": {
			nodeUID: nodeUID,
			initialObjects: []runtime.Object{
				MakeResourceSlice().Name("random-name1").
					NodeOwnerReferences(ownerName, string(nodeUID)).NodeName(ownerName).
					Driver(driverName).Devices([]resourceapi.Device{newDevice(deviceName1)}).
					Pool(resourceapi.ResourcePool{Name: poolName, Generation: 1, ResourceSliceCount: 3}).Obj(),
				MakeResourceSlice().Name(resourceSlice2).GenerateName(generateName2).UID(resourceSlice2).
					NodeOwnerReferences(ownerName, string(nodeUID)).NodeName(ownerName).
					Driver(driverName).Devices([]resourceapi.Device{newDevice(deviceName2)}).
					Pool(resourceapi.ResourcePool{Name: poolName, Generation: 1, ResourceSliceCount: 3}).Obj(),
				MakeResourceSlice().Name("random-name3").
					NodeOwnerReferences(ownerName, string(nodeUID)).NodeName(ownerName).
					Driver(driverName).Devices([]resourceapi.Device{newDevice(deviceName3)}).
					Pool(resourceapi.ResourcePool{Name: poolName, Generation: 1, ResourceSliceCount: 3}).Obj(),
			},
			inputDriverResources: &DriverResources{
				Pools: map[string]Pool{
					poolName: {
						Slices: []Slice{
							{Devices: []resourceapi.Device{newDevice(deviceName1)}},
							{Devices: []resourceapi.Device{newDevice(deviceName2)}},
							{Devices: []resourceapi.Device{newDevice(deviceName3)}},
						},
					},
				},
			},
			expectedStats: Stats{
				NumCreates: 2, // For index 0 and 2
				NumDeletes: 2, // For random-name1 and random-name3
				NumUpdates: 1, // resourceSlice2 is updated to Generation 2
			},
			expectedResourceSlices: []resourceapi.ResourceSlice{
				*MakeResourceSlice().Name(generateName1+"0").GenerateName(generateName1).
					ResourceVersion("1").
					NodeOwnerReferences(ownerName, string(nodeUID)).NodeName(ownerName).
					Driver(driverName).Devices([]resourceapi.Device{newDevice(deviceName1)}).
					Pool(resourceapi.ResourcePool{Name: poolName, Generation: 2, ResourceSliceCount: 3}).Obj(),
				*MakeResourceSlice().Name(resourceSlice2).GenerateName(generateName2).UID(resourceSlice2).ResourceVersion("1").
					NodeOwnerReferences(ownerName, string(nodeUID)).NodeName(ownerName).
					Driver(driverName).Devices([]resourceapi.Device{newDevice(deviceName2)}).
					Pool(resourceapi.ResourcePool{Name: poolName, Generation: 2, ResourceSliceCount: 3}).Obj(),
				*MakeResourceSlice().Name(generateName3+"1").GenerateName(generateName3).
					ResourceVersion("1").
					NodeOwnerReferences(ownerName, string(nodeUID)).NodeName(ownerName).
					Driver(driverName).Devices([]resourceapi.Device{newDevice(deviceName3)}).
					Pool(resourceapi.ResourcePool{Name: poolName, Generation: 2, ResourceSliceCount: 3}).Obj(),
			},
		},
		"drop-node-allocatable-resources-field": {
			features: features{disableNodeAllocatableResources: true},
			nodeUID:  nodeUID,
			initialObjects: []runtime.Object{
				MakeResourceSlice().Name(resourceSlice1).GenerateName(generateName1).
					NodeOwnerReferences(ownerName, string(nodeUID)).NodeName(ownerName).
					Driver(driverName).
					Devices([]resourceapi.Device{newDevice(deviceName)}).
					Pool(resourceapi.ResourcePool{Name: poolName, Generation: 1, ResourceSliceCount: 1}).
					Obj(),
			},
			inputDriverResources: &DriverResources{
				Pools: map[string]Pool{
					poolName: {
						Generation: 1,
						Slices: []Slice{{
							Devices: []resourceapi.Device{{
								Name: deviceName,
								NodeAllocatableResources: map[v1.ResourceName]resourceapi.NodeAllocatableResource{
									v1.ResourceCPU: {Mapping: &resourceapi.NodeAllocatableMapping{DeviceMultiplier: new(resource.MustParse("1"))}},
								},
							}},
						}},
					},
				},
			},
			expectedStats: Stats{
				NumUpdates: 1,
			},
			expectedResourceSlices: []resourceapi.ResourceSlice{
				*MakeResourceSlice().Name(resourceSlice1).GenerateName(generateName1).
					ResourceVersion("1").
					NodeOwnerReferences(ownerName, string(nodeUID)).NodeName(ownerName).
					Driver(driverName).
					Devices([]resourceapi.Device{newDevice(deviceName)}).
					Pool(resourceapi.ResourcePool{Name: poolName, Generation: 1, ResourceSliceCount: 1}).
					Obj(),
			},
			expectedErrors: []string{`update ResourceSlice: pool "pool", slice #0: some fields were dropped by the apiserver, probably because these features are disabled: DRANodeAllocatableResources`},
		},
	}
	for name, test := range testCases {
		t.Run(name, func(t *testing.T) {
			_, ctx := ktesting.NewTestContext(t)
			inputObjects := make([]runtime.Object, 0, len(test.initialObjects)+len(test.initialOtherObjects))
			for _, initialOtherObject := range test.initialOtherObjects {
				inputObjects = append(inputObjects, initialOtherObject.DeepCopyObject())
			}
			for _, initialObject := range test.initialObjects {
				if _, ok := initialObject.(*resourceapi.ResourceSlice); !ok {
					t.Fatalf("test.initialObjects have to be of type *resourceapi.ResourceSlice")
				}
				inputObjects = append(inputObjects, initialObject.DeepCopyObject())
			}
			kubeClient := createTestClient(test.features, timeAddedLater, inputObjects...)
			var queue workqueue.Mock[string]
			owner := &Owner{
				APIVersion: "v1",
				Kind:       "Node",
				Name:       ownerName,
				UID:        test.nodeUID,
			}
			if test.nodeUID == "" {
				owner = &Owner{
					APIVersion: "apps/v1",
					Kind:       "Something",
					Name:       ownerName,
				}
			}
			if test.noOwner {
				owner = nil
			}
			var controllerErrors []error
			ctrl, err := newController(ctx, Options{
				DriverName: driverName,
				KubeClient: kubeClient,
				Owner:      owner,
				Resources:  test.inputDriverResources,
				Queue:      &queue,
				SyncDelay:  test.syncDelay,
				ErrorHandler: func(ctx context.Context, err error, msg string) {
					controllerErrors = append(controllerErrors, fmt.Errorf("%s: %w", msg, err))
				},
				ReconcilePoolWithName: test.reconcilePoolWithName,
			})
			defer ctrl.Stop()
			require.NoError(t, err, "unexpected controller creation error")

			// Process work items in the queue until the queue is empty.
			// Processing races with informers adding new work items,
			// but the desired state should already be reached in the
			// first iteration, so all following iterations should be nops.
			ctrl.run(ctx)

			// Check ResourceSlices
			resourceSlices, err := kubeClient.ResourceV1().ResourceSlices().List(ctx, metav1.ListOptions{})
			require.NoError(t, err, "list resource slices")

			sortResourceSlices(test.expectedResourceSlices)
			sortResourceSlices(resourceSlices.Items)
			assert.Equal(t, test.expectedResourceSlices, resourceSlices.Items)
			assert.Equal(t, test.expectedStats, ctrl.GetStats())

			// The informer might have added a work item before or after ctrl.run returned,
			// therefore we cannot compare the `Later` field. It's either defaultMutationCacheTTL
			// (last AddAfter call was after a create) or defaultSyncDelay (last AddAfter was
			// from informer event handler).
			actualState := queue.State()
			actualState.Later = nil
			// If we let the event handler schedule syncs immediately, then that also races
			// and then Ready cannot be compared either.
			if test.syncDelay != nil && *test.syncDelay == 0 {
				actualState.Ready = nil
			}
			var expectState workqueue.MockState[string]
			assert.Equal(t, expectState, actualState)

			// Sync all pools again. Nothing changed, so the statistics should remain the same.
			if test.inputDriverResources != nil {
				for poolName := range test.inputDriverResources.Pools {
					queue.Add(poolName)
				}
				ctrl.run(ctx)
				assert.Equal(t, test.expectedStats, ctrl.GetStats(), "statistics after re-sync")
			}

			// Dedup the list of errors since we synced the pool twice.
			var dedupedControllerErrors []error
			errorMsgs := sets.New[string]()
			for _, err := range controllerErrors {
				errorMsg := err.Error()
				if errorMsgs.Has(errorMsg) {
					continue
				}
				errorMsgs.Insert(errorMsg)
				dedupedControllerErrors = append(dedupedControllerErrors, err)
			}

			ctrl.Stop()
			switch {
			case len(test.expectedErrors) != 0 && len(dedupedControllerErrors) == 0:
				t.Errorf("expected errors, got none: %s", joinErrors(test.expectedErrors))
			case len(test.expectedErrors) == 0 && len(dedupedControllerErrors) > 0:
				t.Errorf("expected no error, got:\n  %s", joinErrors(formatErrors(controllerErrors)))
			case len(test.expectedErrors) != len(dedupedControllerErrors):
				t.Errorf("expected %d errors, got %d:\n  %s", len(test.expectedErrors), len(dedupedControllerErrors), joinErrors(formatErrors(dedupedControllerErrors)))
			default:
				expectedErrorsSet := sets.New(test.expectedErrors...)
				actualErrorsSet := sets.New(errsToStrings(dedupedControllerErrors)...)
				if !expectedErrorsSet.Equal(actualErrorsSet) {
					t.Errorf("expected errors:\n  %s\ngot:\n  %s", joinErrors(test.expectedErrors), joinErrors(formatErrors(dedupedControllerErrors)))
				}
			}
		})
	}
}

// TestControllerUpdateDeleteRecreate reproduces the bug where a ResourceSlice
// that was updated by the controller (leaving a mutation in the MutationCache)
// and then deleted on the apiserver was not being recreated.
//
// The root cause was that the MutationCache kept the stale updated object, so
// the controller believed the slice still existed and skipped the create.
// The fix calls OnDelete in the informer's delete event handler, which clears
// the mutation before the work queue item is processed.
func TestControllerUpdateDeleteRecreate(t *testing.T) {
	_, ctx := ktesting.NewTestContext(t)

	const (
		driverName = "driver"
		poolName   = "pool"
	)
	generateName := encodeIndex(0, resourceSliceIndexMinLength) + "-" + driverName + "-"
	sliceName := generateName + "0"

	initialSlice := MakeResourceSlice().
		Name(sliceName).
		UID("original-uid").
		GenerateName(generateName).
		ResourceVersion("1").
		Driver(driverName).
		AllNodes(true).
		Devices([]resourceapi.Device{newDevice("old-device")}).
		Pool(resourceapi.ResourcePool{Name: poolName, Generation: 1, ResourceSliceCount: 1}).
		Obj()

	kubeClient := createTestClient(features{}, metav1.Time{}, initialSlice)
	syncDelay := time.Duration(0)

	ctrl, err := StartController(ctx, Options{
		DriverName: driverName,
		KubeClient: kubeClient,
		Resources: &DriverResources{
			Pools: map[string]Pool{
				poolName: {
					AllNodes:   true,
					Generation: 1,
					Slices:     []Slice{{Devices: []resourceapi.Device{newDevice("new-device")}}},
				},
			},
		},
		SyncDelay: &syncDelay,
	})
	require.NoError(t, err)
	defer ctrl.Stop()

	// Wait for the controller to update the initial slice to reflect the
	// desired "new-device" state.
	require.Eventually(t, func() bool {
		return ctrl.GetStats().NumUpdates >= 1
	}, 5*time.Second, 10*time.Millisecond, "controller should update the initial slice")

	// Simulate what happens when the DRA driver restarts and kubelet deletes
	// the slice before the driver can reclaim it.  The informer's delete
	// handler calls OnDelete, which clears the stale mutation from the cache
	// before the work-queue item is processed.  Without that call the
	// MutationCache would serve the stale updated copy and syncPool would
	// skip the necessary create.
	require.NoError(t, kubeClient.ResourceV1().ResourceSlices().Delete(ctx, sliceName, metav1.DeleteOptions{}))

	require.Eventually(t, func() bool {
		return ctrl.GetStats().NumCreates >= 1
	}, 5*time.Second, 10*time.Millisecond, "controller should recreate the deleted slice")
}

// TestControllerUpdateErrorHandlerCanReplaceResources verifies that an error
// handler can replace invalid resources by calling Update.
func TestControllerUpdateErrorHandlerCanReplaceResources(t *testing.T) {
	_, ctx := ktesting.NewTestContext(t)
	type ctxKey struct{}
	ctx = context.WithValue(ctx, ctxKey{}, "controller")
	const poolName = "pool"
	valid := &DriverResources{
		Pools: map[string]Pool{
			poolName: {AllNodes: true, Slices: []Slice{{Devices: []resourceapi.Device{}}}},
		},
	}
	invalid := &DriverResources{
		Pools: map[string]Pool{
			"other-pool": {AllNodes: true, Slices: []Slice{{Devices: []resourceapi.Device{}}}},
		},
	}
	mixed := &DriverResources{
		Pools: map[string]Pool{
			poolName:     valid.Pools[poolName],
			"other-pool": invalid.Pools["other-pool"],
		},
	}

	var ctrl *Controller
	var queue workqueue.Mock[string]
	var controllerErrors []error
	// No data race: ctrl.run calls the handler on this goroutine.
	calling := true
	ctrl, err := newController(ctx, Options{
		DriverName:            "driver",
		KubeClient:            createTestClient(features{}, metav1.Time{}),
		Resources:             invalid,
		Queue:                 &queue,
		ReconcilePoolWithName: poolName,
		ErrorHandler: func(ctx context.Context, err error, _ string) {
			controllerErrors = append(controllerErrors, err)
			// kubeletplugin.Helper holds its mutex while it creates or updates
			// the controller, so republishing from HandleError would deadlock.
			if calling {
				t.Error("error handler called synchronously")
				return
			}
			// Fail instead of deadlocking if the lock is held.
			if !ctrl.mutex.TryLock() {
				t.Error("error handler called while holding the lock")
				return
			}
			ctrl.mutex.Unlock()
			// Contextual logging needs the controller's context.
			assert.Equal(t, "controller", ctx.Value(ctxKey{}))
			// Replace only once, so that a regression fails below instead of looping.
			if len(controllerErrors) == 1 {
				ctrl.Update(valid)
			}
		},
	})
	calling = false
	require.NoError(t, err)
	defer ctrl.Stop()

	ctrl.run(ctx)
	require.Len(t, controllerErrors, 1)
	require.ErrorContains(t, controllerErrors[0], `found pool "other-pool"`)
	require.Equal(t, Stats{NumCreates: 1}, ctrl.GetStats())

	calling = true
	ctrl.Update(mixed)
	calling = false
	ctrl.run(ctx)
	require.Len(t, controllerErrors, 2)
	require.ErrorContains(t, controllerErrors[1], `found pool "other-pool"`)
	assert.Equal(t, Stats{NumCreates: 1}, ctrl.GetStats())
}

func TestControllerPoolNameFieldSelector(t *testing.T) {
	const (
		driverName = "test-driver"
		poolName   = "pool-a"
	)

	testCases := map[string]struct {
		rejectList                  bool
		rejectWatch                 bool
		initialObjects              []runtime.Object
		expectedListSelectors       []bool
		expectedWatchSelectors      []bool
		checkListFallbackFiltering  bool
		checkWatchFallbackFiltering bool
	}{
		"field-selector-supported": {
			expectedListSelectors:  []bool{true},
			expectedWatchSelectors: []bool{true},
		},
		"field-selector-unsupported-for-list": {
			rejectList: true,
			initialObjects: []runtime.Object{
				resourceSliceForPool(poolName, "matching"),
				resourceSliceForPool("other-pool", "other"),
			},
			expectedListSelectors:      []bool{true, false},
			expectedWatchSelectors:     []bool{false},
			checkListFallbackFiltering: true,
		},
		"field-selector-unsupported-for-watch": {
			rejectWatch:                 true,
			expectedListSelectors:       []bool{true},
			expectedWatchSelectors:      []bool{true, false},
			checkWatchFallbackFiltering: true,
		},
	}

	badRequestError := apierrors.NewBadRequest(fmt.Sprintf("field label not supported for %s: %s", resourceapi.SchemeGroupVersion.WithKind("ResourceSlice"), resourceapi.ResourceSliceSelectorPoolName))

	for name, testCase := range testCases {
		t.Run(name, func(t *testing.T) {
			ctx := t.Context()
			kubeClient := fake.NewSimpleClientset(testCase.initialObjects...)
			kubeClient.PrependReactor("list", "resourceslices", func(action k8stesting.Action) (handled bool, ret runtime.Object, err error) {
				if testCase.rejectList && hasPoolNameFieldSelector(action, poolName) {
					return true, nil, badRequestError
				}
				return false, nil, nil
			})
			kubeClient.PrependWatchReactor("resourceslices", func(action k8stesting.Action) (handled bool, ret watch.Interface, err error) {
				if testCase.rejectWatch && hasPoolNameFieldSelector(action, poolName) {
					return true, nil, badRequestError
				}
				return false, nil, nil
			})

			ctrl, err := newController(ctx, Options{
				DriverName:            driverName,
				KubeClient:            kubeClient,
				ReconcilePoolWithName: poolName,
				SyncDelay:             new(time.Hour),
			})
			require.NoError(t, err)
			t.Cleanup(ctrl.Stop)

			require.Eventually(t, func() bool {
				return len(resourceSliceSelectorRequests(kubeClient.Actions(), "watch", poolName)) == len(testCase.expectedWatchSelectors)
			}, 5*time.Second, 10*time.Millisecond, "wait for ResourceSlice watch requests")
			assert.Equal(t, testCase.expectedListSelectors, resourceSliceSelectorRequests(kubeClient.Actions(), "list", poolName))
			assert.Equal(t, testCase.expectedWatchSelectors, resourceSliceSelectorRequests(kubeClient.Actions(), "watch", poolName))
			assert.Equal(t, !testCase.rejectList && !testCase.rejectWatch, ctrl.usePoolNameFieldSelector.Load())

			if testCase.checkListFallbackFiltering {
				matching, err := ctrl.sliceStore.ByIndex(poolNameIndex, poolName)
				require.NoError(t, err)
				assert.Len(t, matching, 1)
				other, err := ctrl.sliceStore.ByIndex(poolNameIndex, "other-pool")
				require.NoError(t, err)
				assert.Empty(t, other)
			}

			if testCase.checkWatchFallbackFiltering {
				_, err := kubeClient.ResourceV1().ResourceSlices().Create(ctx, resourceSliceForPool(poolName, "matching"), metav1.CreateOptions{})
				require.NoError(t, err)
				_, err = kubeClient.ResourceV1().ResourceSlices().Create(ctx, resourceSliceForPool("other-pool", "other"), metav1.CreateOptions{})
				require.NoError(t, err)
				require.Eventually(t, func() bool {
					matching, err := ctrl.sliceStore.ByIndex(poolNameIndex, poolName)
					if err != nil || len(matching) != 1 {
						return false
					}
					other, err := ctrl.sliceStore.ByIndex(poolNameIndex, "other-pool")
					return err == nil && len(other) == 0
				}, 5*time.Second, 10*time.Millisecond, "wait for the fallback watcher to filter ResourceSlices")
			}
		})
	}
}

func hasPoolNameFieldSelector(action k8stesting.Action, poolName string) bool {
	var fieldSelector fields.Selector
	switch action := action.(type) {
	case k8stesting.ListAction:
		fieldSelector = action.GetListRestrictions().Fields
	case k8stesting.WatchAction:
		fieldSelector = action.GetWatchRestrictions().Fields
	default:
		return false
	}

	value, found := fieldSelector.RequiresExactMatch(resourceapi.ResourceSliceSelectorPoolName)
	return found && value == poolName
}

func resourceSliceSelectorRequests(actions []k8stesting.Action, verb, poolName string) []bool {
	var requests []bool
	for _, action := range actions {
		if action.Matches(verb, "resourceslices") {
			requests = append(requests, hasPoolNameFieldSelector(action, poolName))
		}
	}
	return requests
}

func joinErrors(errors []string) string {
	return strings.Join(errors, "\n  ")
}

func errsToStrings(errs []error) []string {
	var strings []string
	for _, err := range errs {
		strings = append(strings, err.Error())
	}
	return strings
}

func formatError(err error) string {
	var droppedFields *DroppedFieldsError
	if errors.As(err, &droppedFields) {
		return fmt.Sprintf("%v\n%s", err, cmp.Diff(droppedFields.DesiredSlice.Spec, droppedFields.ActualSlice.Spec))
	} else {
		return err.Error()
	}
}

func formatErrors(errs []error) []string {
	var errMsgs []string
	for _, err := range errs {
		errMsgs = append(errMsgs, formatError(err))
	}
	return errMsgs
}

func sortResourceSlices(slices []resourceapi.ResourceSlice) {
	sort.Slice(slices, func(i, j int) bool {
		if len(slices[i].Name) == 0 && len(slices[j].Name) == 0 {
			return slices[i].ObjectMeta.GenerateName < slices[j].ObjectMeta.GenerateName
		}
		return slices[i].Name < slices[j].Name
	})
}

type features struct {
	disableBindingConditions        bool
	disableDeviceTaints             bool
	disablePartitionableDevices     bool
	disableConsumableCapacity       bool
	disableOptionalNodeOperations   bool
	disableCompatibilityGroups      bool
	disableNodeAllocatableResources bool
}

func createTestClient(features features, timeAdded metav1.Time, objects ...runtime.Object) *fake.Clientset {
	fakeClient := fake.NewSimpleClientset(objects...)
	fakeClient.PrependReactor("create", "resourceslices", createResourceSliceCreateReactor(features, timeAdded))
	fakeClient.PrependReactor("update", "resourceslices", createResourceSliceUpdateReactor(features, timeAdded))
	return fakeClient
}

// createResourceSliceCreateReactor returns a function which
// implements the logic required for the GenerateName field to work when using
// the fake client. Add it with client.PrependReactor to your fake client.
func createResourceSliceCreateReactor(features features, timeAdded metav1.Time) func(action k8stesting.Action) (handled bool, ret runtime.Object, err error) {
	nameCounter := 0
	var mutex sync.Mutex
	return func(action k8stesting.Action) (handled bool, ret runtime.Object, err error) {
		mutex.Lock()
		defer mutex.Unlock()
		resourceslice := action.(k8stesting.CreateAction).GetObject().(*resourceapi.ResourceSlice)
		// Assign initial ResourceVersion "1" to newly created slices.
		// Avoid using "0" because "0" and "" are treated the same by client-go's MutationCache.
		resourceslice.ResourceVersion = "1"
		if resourceslice.Name == "" && resourceslice.GenerateName != "" {
			resourceslice.Name = fmt.Sprintf("%s%d", resourceslice.GenerateName, nameCounter)
		}
		nameCounter++
		dropDisabledFields(features, resourceslice)
		addTimeAdded(timeAdded, resourceslice)
		return false, nil, nil
	}
}

// resourceSliceUpdateReactor implements the ResourceVersion bump for a fake client.
func createResourceSliceUpdateReactor(features features, timeAdded metav1.Time) func(action k8stesting.Action) (handled bool, ret runtime.Object, err error) {
	return func(action k8stesting.Action) (handled bool, ret runtime.Object, err error) {
		resourceslice := action.(k8stesting.UpdateAction).GetObject().(*resourceapi.ResourceSlice)
		// If ResourceVersion is not set, assume the object started with version "0".
		// This can happen when initial test objects don't have ResourceVersion assigned.
		rev := 0
		if resourceslice.ResourceVersion != "" {
			oldRev, err := strconv.Atoi(resourceslice.ResourceVersion)
			if err != nil {
				return false, nil, fmt.Errorf("ResourceVersion %q should have been an int: %w", resourceslice.ResourceVersion, err)
			}
			rev = oldRev
		}
		rev++
		resourceslice.ResourceVersion = fmt.Sprintf("%d", rev)
		dropDisabledFields(features, resourceslice)
		addTimeAdded(timeAdded, resourceslice)
		return false, nil, nil
	}
}

func dropDisabledFields(features features, resourceslice *resourceapi.ResourceSlice) {
	if features.disableDeviceTaints {
		for i := range resourceslice.Spec.Devices {
			resourceslice.Spec.Devices[i].Taints = nil
		}
	}
	if features.disablePartitionableDevices {
		resourceslice.Spec.PerDeviceNodeSelection = nil
		resourceslice.Spec.SharedCounters = nil
		for i := range resourceslice.Spec.Devices {
			resourceslice.Spec.Devices[i].NodeName = nil
			resourceslice.Spec.Devices[i].NodeSelector = nil
			resourceslice.Spec.Devices[i].ConsumesCounters = nil
		}
	}
	if features.disableBindingConditions {
		for i := range resourceslice.Spec.Devices {
			resourceslice.Spec.Devices[i].BindingConditions = nil
			resourceslice.Spec.Devices[i].BindingFailureConditions = nil
			resourceslice.Spec.Devices[i].BindsToNode = nil
		}
	}
	if features.disableConsumableCapacity {
		for i := range resourceslice.Spec.Devices {
			resourceslice.Spec.Devices[i].AllowMultipleAllocations = nil
		}
	}
	if features.disableOptionalNodeOperations {
		resourceslice.Spec.SkipNodeOperations = nil
	}
	if features.disableCompatibilityGroups {
		for i := range resourceslice.Spec.Devices {
			for j := range resourceslice.Spec.Devices[i].ConsumesCounters {
				resourceslice.Spec.Devices[i].ConsumesCounters[j].CompatibilityGroups = nil
			}
		}
	}
	if features.disableNodeAllocatableResources {
		for i := range resourceslice.Spec.Devices {
			resourceslice.Spec.Devices[i].NodeAllocatableResources = nil
		}
	}
}

func addTimeAdded(timeAdded metav1.Time, resourceslice *resourceapi.ResourceSlice) {
	for i := range resourceslice.Spec.Devices {
		for e := range resourceslice.Spec.Devices[i].Taints {
			taint := &resourceslice.Spec.Devices[i].Taints[e]
			if taint.TimeAdded == nil {
				taint.TimeAdded = &timeAdded
			}
		}
	}
}

// ResourceSliceWrapper wraps a ResourceSlice.
type ResourceSliceWrapper struct {
	resourceapi.ResourceSlice
}

// MakeResourceSlice creates a wrapper for a ResourceSlice.
func MakeResourceSlice() *ResourceSliceWrapper {
	return &ResourceSliceWrapper{
		resourceapi.ResourceSlice{
			ObjectMeta: metav1.ObjectMeta{},
			Spec: resourceapi.ResourceSliceSpec{
				Pool: resourceapi.ResourcePool{},
			},
		},
	}
}

// Obj returns the inner ResourceSlice.
func (r *ResourceSliceWrapper) Obj() *resourceapi.ResourceSlice {
	return &r.ResourceSlice
}

// Name sets the value of ResourceSlice.ObjectMeta.Name
func (r *ResourceSliceWrapper) Name(name string) *ResourceSliceWrapper {
	r.ObjectMeta.Name = name
	return r
}

// GenerateName sets the value of ResourceSlice.ObjectMeta.GenerateName
func (r *ResourceSliceWrapper) GenerateName(generateName string) *ResourceSliceWrapper {
	r.ObjectMeta.GenerateName = generateName
	return r
}

// UID sets the value of ResourceSlice.ObjectMeta.UID
func (r *ResourceSliceWrapper) UID(uid string) *ResourceSliceWrapper {
	r.ObjectMeta.UID = types.UID(uid)
	return r
}

// ResourceVersion sets the value of ResourceSlice.ObjectMeta.ResourceVersion
func (r *ResourceSliceWrapper) ResourceVersion(rev string) *ResourceSliceWrapper {
	r.ObjectMeta.ResourceVersion = rev
	return r
}

// NodeOwnerReferences sets the value of ResourceSlice.ObjectMeta.NodeOwnerReferences
// to a v1.Node
func (r *ResourceSliceWrapper) NodeOwnerReferences(nodeName, nodeUID string) *ResourceSliceWrapper {
	r.ObjectMeta.OwnerReferences = []metav1.OwnerReference{
		{
			APIVersion: "v1",
			Kind:       "Node",
			Name:       nodeName,
			UID:        types.UID(nodeUID),
			Controller: ptr.To(true),
		},
	}
	return r
}

// AppOwnerReferences sets the value of ResourceSlice.ObjectMeta.NodeOwnerReferences
// to some fictional app controller resource
func (r *ResourceSliceWrapper) AppOwnerReferences(appName string) *ResourceSliceWrapper {
	r.ObjectMeta.OwnerReferences = []metav1.OwnerReference{
		{
			APIVersion: "apps/v1",
			Kind:       "Something",
			Name:       appName,
			Controller: ptr.To(true),
		},
	}
	return r
}

// Driver sets the value of ResourceSlice.Spec.Driver
func (r *ResourceSliceWrapper) Driver(driver string) *ResourceSliceWrapper {
	r.Spec.Driver = driver
	return r
}

// Pool sets the value of ResourceSlice.Spec.Pool
func (r *ResourceSliceWrapper) Pool(pool resourceapi.ResourcePool) *ResourceSliceWrapper {
	r.Spec.Pool = pool
	return r
}

// NodeName sets the value of ResourceSlice.Spec.NodeName
func (r *ResourceSliceWrapper) NodeName(nodeName string) *ResourceSliceWrapper {
	r.Spec.NodeName = refIfNotZero(nodeName)
	return r
}

// NodeSelector sets the value of ResourceSlice.Spec.NodeSelector
func (r *ResourceSliceWrapper) NodeSelector(nodeSelector *v1.NodeSelector) *ResourceSliceWrapper {
	r.Spec.NodeSelector = nodeSelector
	return r
}

// AllNodes sets the value of ResourceSlice.Spec.AllNodes
func (r *ResourceSliceWrapper) AllNodes(allNodes bool) *ResourceSliceWrapper {
	r.Spec.AllNodes = refIfNotZero(allNodes)
	return r
}

// Devices sets the value of ResourceSlice.Spec.Devices
func (r *ResourceSliceWrapper) Devices(devices []resourceapi.Device) *ResourceSliceWrapper {
	r.Spec.Devices = devices
	return r
}

// PerDeviceNodeSelection sets ResourceSlice.Spec.PerDeviceNodeSelection.
func (r *ResourceSliceWrapper) PerDeviceNodeSelection(perDeviceNodeSelection bool) *ResourceSliceWrapper {
	if perDeviceNodeSelection {
		r.Spec.PerDeviceNodeSelection = ptr.To(true)
	} else {
		r.Spec.PerDeviceNodeSelection = nil
	}
	return r
}

// SharedCounters sets ResourceSlice.Spec.SharedCounters.
func (r *ResourceSliceWrapper) SharedCounters(counters []resourceapi.CounterSet) *ResourceSliceWrapper {
	r.Spec.SharedCounters = counters
	return r
}

// SkipNodeOperations sets ResourceSlice.Spec.SkipNodeOperations.
func (r *ResourceSliceWrapper) SkipNodeOperations(skip ...resourceapi.SkipNodeOperation) *ResourceSliceWrapper {
	r.Spec.SkipNodeOperations = skip
	return r
}

type nodeNameField string
type allowMultipleAllocationsField bool

func newDevice(name string, fields ...any) resourceapi.Device {
	device := resourceapi.Device{
		Name: name,
	}
	for _, field := range fields {
		switch f := field.(type) {
		case map[resourceapi.QualifiedName]resourceapi.DeviceAttribute:
			device.Attributes = f
		case resourceapi.DeviceTaint:
			device.Taints = append(device.Taints, f)
		case []resourceapi.DeviceTaint:
			device.Taints = append(device.Taints, f...)
		case resourceapi.DeviceCounterConsumption:
			device.ConsumesCounters = append(device.ConsumesCounters, f)
		case []resourceapi.DeviceCounterConsumption:
			device.ConsumesCounters = append(device.ConsumesCounters, f...)
		case nodeNameField:
			device.NodeName = ptr.To(string(f))
		case allowMultipleAllocationsField:
			device.AllowMultipleAllocations = ptr.To(bool(f))
		case *v1.NodeSelector:
			device.NodeSelector = f
		default:
			panic(fmt.Sprintf("unsupported resourceapi.Device field type %T", field))
		}
	}
	return device
}

func TestGetIndexLength(t *testing.T) {
	testCases := []struct {
		numSlices int
		expected  int
	}{
		{-1, 5},
		{0, 5},
		{1, 5},
		{16, 5},
		{0x10000, 5},
		{0x100000, 5},
		{0x100001, 6},
		{0x1000000, 6},
	}

	for _, tc := range testCases {
		t.Run(fmt.Sprintf("%d", tc.numSlices), func(t *testing.T) {
			assert.Equal(t, tc.expected, getIndexLength(tc.numSlices))
		})
	}
}

func TestEncodeIndex(t *testing.T) {
	testCases := []struct {
		index          int
		expectedLength int
		expected       string
	}{
		{0, 5, "00000"},
		{1, 5, "00001"},
		{9, 5, "00009"},
		{10, 5, "0000a"},
		{26, 5, "0001a"},
		{27, 5, "0001b"},
		{35, 5, "00023"},
		{36, 5, "00024"},
		{100000, 5, "186a0"},
		{0, 6, "000000"},
		{1048576, 6, "100000"},
	}

	for _, tc := range testCases {
		t.Run(fmt.Sprintf("%d", tc.index), func(t *testing.T) {
			assert.Equal(t, tc.expected, encodeIndex(tc.index, tc.expectedLength))
		})
	}
}

func TestDecodeIndex(t *testing.T) {
	testCases := []struct {
		name           string
		expectedLength int
		expected       int
		expectError    bool
		expectedError  string
	}{
		{name: "00000-", expectedLength: 5, expected: 0},
		{name: "00001-", expectedLength: 5, expected: 1},
		{name: "00009-", expectedLength: 5, expected: 9},
		{name: "0000a-", expectedLength: 5, expected: 10},
		{name: "0001a-", expectedLength: 5, expected: 26},
		{name: "0001b-", expectedLength: 5, expected: 27},
		{name: "00023-", expectedLength: 5, expected: 35},
		{name: "00024-", expectedLength: 5, expected: 36},
		{name: "186a0-", expectedLength: 5, expected: 100000},
		{name: "100000-", expectedLength: 6, expected: 1048576},
		// Large value that fits in int64 but might exceed int32
		{
			name:           "80000000-",
			expectedLength: 8,
			expected:       getExpectedForLargeInt(int64(math.MaxInt32) + 1),
			expectError:    getExpectedForLargeInt(int64(math.MaxInt32)+1) == -1,
			expectedError:  "failed to parse index",
		},
		// Overflow Cases
		{name: "8000000000000000-", expectedLength: 16, expectError: true, expectedError: "failed to parse index"}, // Absolute maximum possible string (overflows int64)
		// With suffix (as created by GenerateName)
		{name: "00000-driver-owner-suffix", expectedLength: 5, expected: 0},
		{name: "00001-driver-owner-suffix", expectedLength: 5, expected: 1},
		// Invalid
		{name: "00000", expectedLength: 5, expectError: true, expectedError: "invalid index length or missing separator"},  // Missing separator
		{name: "0000g-", expectedLength: 5, expectError: true, expectedError: "failed to parse index"},                     // 'g' is not base16
		{name: "-0001-", expectedLength: 5, expectError: true, expectedError: "invalid index length or missing separator"}, // Negative hex (dash is separator, so strings.Index is 0)
		// Wrong length
		{name: "00000-", expectedLength: 6, expectError: true, expectedError: "invalid index length or missing separator"},
		{name: "100000-", expectedLength: 5, expectError: true, expectedError: "invalid index length or missing separator"},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			actual, err := decodeIndex(tc.name, tc.expectedLength)
			if tc.expectError {
				require.Error(t, err)
				if tc.expectedError != "" {
					assert.Contains(t, err.Error(), tc.expectedError)
				}
			} else {
				require.NoError(t, err)
				assert.Equal(t, tc.expected, actual)
			}
		})
	}
}

// Helper to handle platform-dependent test expectations
func getExpectedForLargeInt(val int64) int {
	if val > int64(math.MaxInt) {
		return -1
	}
	return int(val)
}

func deepCopyDevices(devices []resourceapi.Device) []resourceapi.Device {
	out := make([]resourceapi.Device, len(devices))
	for i := range devices {
		out[i] = *devices[i].DeepCopy()
	}
	return out
}

// Use synchronous syncPool calls so negative assertions have an
// exact completion barrier, rather than a quiet-period timeout.
func TestSharingAffinityPublisher(t *testing.T) {
	t.Run("deepcopy-isolation", func(t *testing.T) {
		for _, tc := range []struct {
			name    string
			entries []resourceapi.SharingAffinityExtractor
		}{
			{"nil", nil},
			{"empty", []resourceapi.SharingAffinityExtractor{}},
			{"populated", []resourceapi.SharingAffinityExtractor{sharingAffinityEntry("subnet", `"blue"`)}},
		} {
			t.Run(tc.name, func(t *testing.T) {
				source := Slice{SharingAffinity: tc.entries}
				copied := source.DeepCopy()
				if (copied.SharingAffinity == nil) != (tc.entries == nil) {
					t.Fatal("deepcopy changed nil/empty representation")
				}
				want := append([]resourceapi.SharingAffinityExtractor(nil), tc.entries...)
				if len(source.SharingAffinity) > 0 {
					source.SharingAffinity[0].Name = "changed"
					source.SharingAffinity[0].Expression = `"changed"`
				}
				source.SharingAffinity = append(source.SharingAffinity, sharingAffinityEntry("extra", `"x"`))
				if len(copied.SharingAffinity) != len(want) {
					t.Fatalf("copy length changed: %d, want %d", len(copied.SharingAffinity), len(want))
				}
				for i := range want {
					if copied.SharingAffinity[i] != want[i] {
						t.Errorf("copy aliases input: got %#v, want %#v", copied.SharingAffinity[i], want[i])
					}
				}
			})
		}
	})
	for _, useUpdate := range []bool{false, true} {
		t.Run(fmt.Sprintf("caller-mutation-after-update=%t", useUpdate), func(t *testing.T) {
			pool := sharingAffinityPublisherPool()
			resources := &DriverResources{Pools: map[string]Pool{"pool": pool}}
			h := newSharingAffinityHarness(t, resources, nil, "", false)
			if useUpdate {
				h.ctrl.Update(resources)
			}
			pool.Slices[0].SharingAffinity[0].Name = "changed"
			pool.Slices[0].SharingAffinity[0].Expression = `"changed"`
			pool.Slices[0].SharingAffinity = append(pool.Slices[0].SharingAffinity, sharingAffinityEntry("extra", `"x"`))
			delete(resources.Pools, "pool")
			h.sync(t)
			h.noErrors(t)
			h.expectPool(t, sharingAffinityPublisherPool(), 1)
		})
	}
	for _, tc := range []struct {
		name       string
		changes    int
		wantStats  Stats
		generation int64
	}{
		{"equal-input-no-churn", 0, Stats{}, 7},
		{"update-preserves-fields", 1, Stats{NumUpdates: 1}, 7},
		{"declaration-change-two-slices-bumps-generation", 2, Stats{NumUpdates: 4}, 8},
	} {
		t.Run(tc.name, func(t *testing.T) {
			pool := sharingAffinityPublisherPool()
			old := sharingAffinityStoredSlices(pool, 7)
			for i := 0; i < tc.changes; i++ {
				pool.Slices[i].SharingAffinity[0].Expression = `"red"`
			}
			h := newSharingAffinityHarness(t, &DriverResources{Pools: map[string]Pool{"pool": pool}}, old, "", false)
			h.sync(t)
			h.noErrors(t)
			if got := h.ctrl.GetStats(); got != tc.wantStats {
				t.Errorf("stats = %+v, want %+v", got, tc.wantStats)
			}
			h.expectPool(t, pool, tc.generation)
			for i, stored := range h.list(t) {
				if stored.UID != old[i].UID || stored.Labels["preserved"] != "yes" {
					t.Errorf("update replaced object or lost unrelated metadata: %s", stored.Name)
				}
				if tc.changes == 1 && i != 0 && stored.ResourceVersion != old[i].ResourceVersion {
					t.Errorf("unchanged sibling %s was updated", stored.Name)
				}
			}
			h.client.ClearActions()
			stats := h.ctrl.GetStats()
			h.sync(t)
			if got := h.ctrl.GetStats(); got != stats {
				t.Errorf("equal-input resync churned: %+v -> %+v", stats, got)
			}
			h.noWrites(t)
		})
	}
	t.Run("create-multi-slice", func(t *testing.T) {
		pool := sharingAffinityPublisherPool()
		h := newSharingAffinityHarness(t, &DriverResources{Pools: map[string]Pool{"pool": pool}}, nil, "", false)
		h.sync(t)
		h.noErrors(t)
		h.expectPool(t, pool, 1)
		if got := h.ctrl.GetStats(); got != (Stats{NumCreates: 4}) {
			t.Errorf("stats = %+v, want four creates", got)
		}
	})
	t.Run("no-affinity-compat", func(t *testing.T) {
		pool := sharingAffinityPublisherPool()
		pool.Slices = pool.Slices[2:]
		h := newSharingAffinityHarness(t, &DriverResources{Pools: map[string]Pool{"pool": pool}}, nil, "", false)
		h.sync(t)
		h.noErrors(t)
		h.expectPool(t, pool, 1)
		h.client.ClearActions()
		h.sync(t)
		h.noWrites(t)
	})
	t.Run("nil-empty-no-churn", func(t *testing.T) {
		for _, emptyDesired := range []bool{false, true} {
			pool := sharingAffinityPool(sharingAffinityEntry("subnet", `"blue"`))
			pool.Slices[0].SharingAffinity = nil
			old := sharingAffinityStoredSlices(pool, 1)
			if emptyDesired {
				pool.Slices[0].SharingAffinity = []resourceapi.SharingAffinityExtractor{}
			} else {
				old[0].Spec.SharingAffinity = []resourceapi.SharingAffinityExtractor{}
			}
			h := newSharingAffinityHarness(t, &DriverResources{Pools: map[string]Pool{"pool": pool}}, old, "", false)
			h.sync(t)
			h.noErrors(t)
			h.noWrites(t)
		}
	})
	for _, verb := range []string{"create", "update", "preserved"} {
		t.Run("dropped-on-"+verb, func(t *testing.T) {
			pool := sharingAffinityPool(sharingAffinityEntry("subnet", `"blue"`))
			var old []*resourceapi.ResourceSlice
			if verb != "create" {
				old = sharingAffinityStoredSlices(pool, 1)
				pool.Generation = 2
				if verb == "update" {
					old[0].Spec.SharingAffinity = nil
				}
			}
			h := newSharingAffinityHarness(t, &DriverResources{Pools: map[string]Pool{"pool": pool}}, old, "", false)
			if verb != "preserved" {
				// Modify only the server result, never the desired object used
				// by sliceStored's comparison.
				h.client.PrependReactor(verb, "resourceslices", func(action k8stesting.Action) (bool, runtime.Object, error) {
					var actual *resourceapi.ResourceSlice
					// CreateAction and UpdateAction have the same methods, so
					// a type switch would also treat updates as creates.
					switch action.GetVerb() {
					case "create":
						actual = action.(k8stesting.CreateAction).GetObject().(*resourceapi.ResourceSlice).DeepCopy()
						actual.Name = actual.GenerateName + "0"
						actual.ResourceVersion = "1"
					case "update":
						actual = action.(k8stesting.UpdateAction).GetObject().(*resourceapi.ResourceSlice).DeepCopy()
						actual.ResourceVersion = "11"
					default:
						t.Fatalf("unexpected action %T", action)
					}
					actual.Spec.SharingAffinity = nil
					gvr := resourceapi.SchemeGroupVersion.WithResource("resourceslices")
					var err error
					if verb == "create" {
						err = h.client.Tracker().Create(gvr, actual, "")
					} else {
						err = h.client.Tracker().Update(gvr, actual, "")
					}
					return true, actual, err
				})
			}
			h.sync(t)
			if verb == "preserved" {
				h.noErrors(t)
				h.expectPool(t, pool, 2)
				if got := h.ctrl.GetStats(); got != (Stats{NumUpdates: 1}) {
					t.Errorf("preservation was not exercised by an update: %+v", got)
				}
			} else {
				if len(h.errs) != 1 {
					t.Fatalf("reported errors = %v, want one dropped-field error", h.errs)
				}
				var dropped *DroppedFieldsError
				if !errors.As(h.errs[0], &dropped) {
					t.Fatalf("error = %v, want DroppedFieldsError", h.errs[0])
				}
				if diff := cmp.Diff([]string{"DRASharingAffinity"}, dropped.DisabledFeatures()); diff != "" {
					t.Errorf("disabled features (-want,+got): %s", diff)
				}
				if len(dropped.DesiredSlice.Spec.SharingAffinity) != 1 || len(dropped.ActualSlice.Spec.SharingAffinity) != 0 {
					t.Error("dropped error lost desired/actual declarations")
				}
				if dropped.PoolName != "pool" || dropped.SliceIndex != 0 {
					t.Errorf("incorrect drop attribution: pool %q slice %d", dropped.PoolName, dropped.SliceIndex)
				}
				if got := h.list(t); len(got) != 1 || len(got[0].Spec.SharingAffinity) != 0 {
					t.Fatalf("fake server did not store dropped result: %#v", got)
				}
			}
			h.client.ClearActions()
			count := len(h.errs)
			h.sync(t)
			h.noWrites(t)
			if len(h.errs) != count {
				t.Errorf("drop was reported again on unchanged resync: %v", h.errs)
			}
		})
	}
	t.Run("disabled-features-unit", func(t *testing.T) {
		for _, tc := range []struct {
			name             string
			affinity, taints bool
			want             []string
		}{
			{"affinity-only", true, false, []string{"DRASharingAffinity"}},
			{"affinity-and-taints", true, true, []string{"DRADeviceTaints", "DRASharingAffinity"}},
			{"taints-only", false, true, []string{"DRADeviceTaints"}},
			{"empty-desired", false, false, nil},
		} {
			t.Run(tc.name, func(t *testing.T) {
				desired := &resourceapi.ResourceSlice{}
				if tc.affinity {
					desired.Spec.SharingAffinity = []resourceapi.SharingAffinityExtractor{sharingAffinityEntry("subnet", `"blue"`)}
				}
				if tc.taints {
					desired.Spec.Devices = []resourceapi.Device{{Name: "device", Taints: []resourceapi.DeviceTaint{{Key: "example.com/taint", Effect: resourceapi.DeviceTaintEffectNoSchedule}}}}
				}
				actual := desired.DeepCopy()
				actual.Spec.SharingAffinity = nil
				for i := range actual.Spec.Devices {
					actual.Spec.Devices[i].Taints = nil
				}
				err := &DroppedFieldsError{DesiredSlice: desired, ActualSlice: actual}
				if diff := cmp.Diff(tc.want, err.DisabledFeatures()); diff != "" {
					t.Errorf("disabled features (-want,+got): %s", diff)
				}
			})
		}
	})
	// Run every preflight failure both before first publication and
	// over a valid pool. Prefix/suffix slices catch partial writes in either order.
	for _, tc := range sharingAffinityValidationCases(t) {
		if tc.wantText == "" && tc.wantExact == "" && tc.wantKind == nil {
			continue
		}
		t.Run("invalid-pool-no-writes/"+tc.name, func(t *testing.T) {
			for _, existing := range []bool{false, true} {
				for _, scoped := range []bool{false, true} {
					t.Run(fmt.Sprintf("existing=%t/scoped=%t", existing, scoped), func(t *testing.T) {
						pool := *tc.pool.DeepCopy()
						pool.AllNodes = true
						pool.Slices = append(pool.Slices, Slice{Devices: []resourceapi.Device{{Name: "unrelated"}}})
						var old []*resourceapi.ResourceSlice
						if existing {
							valid := *pool.DeepCopy()
							for i := range valid.Slices {
								valid.Slices[i] = Slice{Devices: []resourceapi.Device{{Name: fmt.Sprintf("old-%d", i)}}}
							}
							old = sharingAffinityStoredSlices(valid, 1)
							// This would be deleted before creates without preflight.
							obsolete := old[0].DeepCopy()
							obsolete.Name = "obsolete-with-invalid-prefix"
							old = append(old, obsolete)
						}
						scope := ""
						if scoped {
							scope = "pool"
						}
						h := newSharingAffinityHarness(t, &DriverResources{Pools: map[string]Pool{"pool": pool}}, old, scope, true)
						before := h.list(t)
						h.sync(t)
						h.noWrites(t)
						if len(h.errs) != 1 || h.messages[0] != "pool validation failed" {
							t.Fatalf("validation error not reported: messages %v, errors %v", h.messages, h.errs)
						}
						checkSharingAffinityValidationError(t, h.errs[0], tc)
						if diff := cmp.Diff(before, h.list(t)); diff != "" {
							t.Errorf("invalid pool changed stored objects (-before,+after): %s", diff)
						}
					})
				}
			}
		})
	}
	for _, tc := range sharingAffinityRatchetCases(t) {
		t.Run("ratchet-"+tc.name, func(t *testing.T) {
			for _, scope := range []string{"", "pool"} {
				t.Run("scope="+scope, func(t *testing.T) {
					h := newSharingAffinityHarness(t, &DriverResources{Pools: map[string]Pool{"pool": tc.pool}}, tc.old, scope, false)
					before := h.list(t)
					h.sync(t)
					if tc.wantText != "" || tc.wantKind != nil {
						h.noWrites(t)
						if len(h.errs) != 1 || h.messages[0] != "pool validation failed" {
							t.Fatalf("expected validation failure, got %v (%v)", h.errs, h.messages)
						}
						checkSharingAffinityValidationError(t, h.errs[0], tc)
						if diff := cmp.Diff(before, h.list(t)); diff != "" {
							t.Errorf("invalid ratcheted pool changed stored state: %s", diff)
						}
						return
					}
					h.noErrors(t)
					got := h.list(t)
					if len(got) != len(tc.pool.Slices) {
						t.Fatalf("stored %d slices, want %d", len(got), len(tc.pool.Slices))
					}
					for i := range got {
						if diff := cmp.Diff(tc.pool.Slices[i].SharingAffinity, got[i].Spec.SharingAffinity); diff != "" {
							t.Errorf("slice %d declarations (-want,+got): %s", i, diff)
						}
						if i < len(tc.old) && tc.old[i] != nil && got[i].UID != tc.old[i].UID {
							t.Errorf("ratcheted declaration replaced instead of updated: %s", got[i].Name)
						}
					}
					if h.ctrl.GetStats() == (Stats{}) {
						t.Error("fixture did not exercise a write")
					}
				})
			}
		})
	}
	for _, obsoleteReason := range []string{"lower-generation", "invalid-name", "out-of-range-index"} {
		t.Run("obsolete-generation-declaration/"+obsoleteReason, func(t *testing.T) {
			pool := sharingAffinityPool(sharingAffinityEntry("subnet", `"`+strings.Repeat("a", 10239)+`"`), sharingAffinityEntry("zone", `"west"`))
			old := sharingAffinityStoredSlices(pool, 2)
			switch obsoleteReason {
			case "lower-generation":
				old[0].Spec.Pool.Generation = 1
			case "invalid-name":
				old[0].Name = "not-an-index"
			case "out-of-range-index":
				old[0].Name = encodeIndex(2, getIndexLength(2)) + "-driver-old"
			}
			h := newSharingAffinityHarness(t, &DriverResources{Pools: map[string]Pool{"pool": pool}}, old, "", false)
			before := h.list(t)
			h.sync(t)
			h.noWrites(t)
			if len(h.errs) != 1 || !strings.Contains(h.errs[0].Error(), "expression length 10241") {
				t.Fatalf("obsolete declaration incorrectly granted ratchet: %v", h.errs)
			}
			if diff := cmp.Diff(before, h.list(t)); diff != "" {
				t.Errorf("stored objects changed: %s", diff)
			}
		})
	}
	t.Run("remove-last-declaration", func(t *testing.T) {
		for _, removeSlice := range []bool{false, true} {
			t.Run(fmt.Sprintf("remove-slice=%t", removeSlice), func(t *testing.T) {
				pool := Pool{AllNodes: true, Slices: []Slice{
					{Devices: []resourceapi.Device{{Name: "device"}}},
					{SharingAffinity: []resourceapi.SharingAffinityExtractor{sharingAffinityEntry("subnet", `"blue"`)}},
				}}
				old := sharingAffinityStoredSlices(pool, 1)
				if removeSlice {
					pool.Slices = pool.Slices[:1]
				} else {
					pool.Slices[1].SharingAffinity = nil
				}
				h := newSharingAffinityHarness(t, &DriverResources{Pools: map[string]Pool{"pool": pool}}, old, "", false)
				h.sync(t)
				h.noErrors(t)
				h.expectPool(t, pool, 1)
				want := Stats{NumUpdates: 1}
				if removeSlice {
					want.NumDeletes = 1
				}
				if got := h.ctrl.GetStats(); got != want {
					t.Errorf("stats = %+v, want %+v", got, want)
				}
			})
		}
	})
	t.Run("invalid-pool-blocks-other-pools", func(t *testing.T) {
		entry := sharingAffinityEntry("subnet", `"blue"`)
		h := newSharingAffinityHarness(t, &DriverResources{Pools: map[string]Pool{
			"pool": sharingAffinityPool(entry), "invalid": sharingAffinityPool(entry, entry),
		}}, nil, "", false)
		h.sync(t)
		h.noWrites(t)
		if len(h.errs) != 1 || !strings.Contains(h.errs[0].Error(), `pool "invalid"`) {
			t.Errorf("invalid sibling pool did not block publication: %v", h.errs)
		}
	})
}

func TestMatchExistingSlices(t *testing.T) {
	logger, _ := ktesting.NewTestContext(t)
	old := sharingAffinityStoredSlices(sharingAffinityPool(sharingAffinityEntry("a", `"a"`), sharingAffinityEntry("b", `"b"`)), 3)
	duplicate := old[0].DeepCopy()
	duplicate.Name += "-duplicate"
	invalid := old[0].DeepCopy()
	invalid.Name = "invalid"
	lower := old[0].DeepCopy()
	lower.Spec.Pool.Generation = 2
	for _, tc := range []struct {
		name       string
		slices     []*resourceapi.ResourceSlice
		desired    int
		generation int64
		current    map[int]*resourceapi.ResourceSlice
		obsolete   []*resourceapi.ResourceSlice
	}{
		{"empty", nil, 2, 0, map[int]*resourceapi.ResourceSlice{}, nil},
		{"index-match", old, 2, 3, map[int]*resourceapi.ResourceSlice{0: old[0], 1: old[1]}, nil},
		{"duplicate-index", []*resourceapi.ResourceSlice{old[0], duplicate}, 2, 3, map[int]*resourceapi.ResourceSlice{0: old[0]}, []*resourceapi.ResourceSlice{duplicate}},
		{"invalid-name", []*resourceapi.ResourceSlice{invalid}, 2, 3, map[int]*resourceapi.ResourceSlice{}, []*resourceapi.ResourceSlice{invalid}},
		{"lower-generation", []*resourceapi.ResourceSlice{lower, old[1]}, 2, 3, map[int]*resourceapi.ResourceSlice{1: old[1]}, []*resourceapi.ResourceSlice{lower}},
		{"out-of-range", old, 1, 3, map[int]*resourceapi.ResourceSlice{0: old[0]}, []*resourceapi.ResourceSlice{old[1]}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got := matchExistingSlices(logger, tc.slices, tc.desired)
			if got.generation != tc.generation {
				t.Errorf("generation = %d, want %d", got.generation, tc.generation)
			}
			if diff := cmp.Diff(tc.current, got.current); diff != "" {
				t.Errorf("matches (-want,+got): %s", diff)
			}
			if len(got.obsolete) != len(tc.obsolete) {
				t.Fatalf("obsolete count = %d, want %d", len(got.obsolete), len(tc.obsolete))
			}
			for i := range tc.obsolete {
				if got.obsolete[i] != tc.obsolete[i] {
					t.Errorf("obsolete[%d] = %s, want %s", i, got.obsolete[i].Name, tc.obsolete[i].Name)
				}
			}
		})
	}
}

func sharingAffinityPublisherPool() Pool {
	pool := sharingAffinityPool(sharingAffinityEntry("subnet", `"blue"`), sharingAffinityEntry("zone", `"west"`))
	pool.AllNodes = false
	pool.NodeSelector = &v1.NodeSelector{NodeSelectorTerms: []v1.NodeSelectorTerm{{
		MatchExpressions: []v1.NodeSelectorRequirement{{Key: "example.com/zone", Operator: v1.NodeSelectorOpIn, Values: []string{"west"}}},
	}}}
	pool.Slices = append(pool.Slices,
		Slice{
			Devices: []resourceapi.Device{{
				Name:                     "device",
				Attributes:               map[resourceapi.QualifiedName]resourceapi.DeviceAttribute{"model": {StringValue: ptr.To("nic")}},
				Capacity:                 map[resourceapi.QualifiedName]resourceapi.DeviceCapacity{"memory": {Value: resource.MustParse("1Gi")}},
				AllowMultipleAllocations: ptr.To(true),
			}},
			PartitionTypeAttribute: ptr.To(resourceapi.FullyQualifiedName("example.com/model")),
			SkipNodeOperations:     []resourceapi.SkipNodeOperation{resourceapi.SkipNodeOperationAll},
		},
		Slice{SharedCounters: []resourceapi.CounterSet{{
			Name: "counter", Counters: map[string]resourceapi.Counter{"memory": {Value: resource.MustParse("1Gi")}},
		}}},
	)
	return pool
}

func sharingAffinityStoredSlices(pool Pool, generation int64) []*resourceapi.ResourceSlice {
	var result []*resourceapi.ResourceSlice
	for i, slice := range pool.Slices {
		name := encodeIndex(i, getIndexLength(len(pool.Slices))) + "-driver.example.com-stored"
		result = append(result, &resourceapi.ResourceSlice{
			ObjectMeta: metav1.ObjectMeta{Name: name, UID: types.UID(name), ResourceVersion: "10", Labels: map[string]string{"preserved": "yes"}},
			Spec: resourceapi.ResourceSliceSpec{
				Driver:   "driver.example.com",
				Pool:     resourceapi.ResourcePool{Name: "pool", Generation: generation, ResourceSliceCount: int64(len(pool.Slices))},
				AllNodes: refIfNotZero(pool.AllNodes), NodeSelector: pool.NodeSelector,
				Devices: slice.Devices, SharedCounters: slice.SharedCounters, SharingAffinity: slice.SharingAffinity,
				PerDeviceNodeSelection: slice.PerDeviceNodeSelection, PartitionTypeAttribute: slice.PartitionTypeAttribute, SkipNodeOperations: slice.SkipNodeOperations,
			},
		})
		result[i] = result[i].DeepCopy()
	}
	return result
}

type sharingAffinityHarness struct {
	ctrl     *Controller
	client   *fake.Clientset
	ctx      context.Context
	errs     []error
	messages []string
}

func newSharingAffinityHarness(t *testing.T, resources *DriverResources, old []*resourceapi.ResourceSlice, scope string, disableQualifiedNames bool) *sharingAffinityHarness {
	t.Helper()
	_, ctx := ktesting.NewTestContext(t)
	var objects []runtime.Object
	for _, slice := range old {
		if slice != nil {
			objects = append(objects, slice.DeepCopy())
		}
	}
	h := &sharingAffinityHarness{ctx: ctx, client: createTestClient(features{}, metav1.Time{}, objects...)}
	var queue workqueue.Mock[string]
	ctrl, err := newController(ctx, Options{
		DriverName: "driver.example.com", KubeClient: h.client, Resources: resources, Queue: &queue,
		ReconcilePoolWithName: scope, ValidateQualifiedNames: ptr.To(!disableQualifiedNames),
		ErrorHandler: func(_ context.Context, err error, msg string) {
			h.errs = append(h.errs, err)
			h.messages = append(h.messages, msg)
		},
	})
	if err != nil {
		t.Fatalf("newController: %v", err)
	}
	h.ctrl = ctrl
	t.Cleanup(ctrl.Stop)
	h.client.ClearActions()
	return h
}

func (h *sharingAffinityHarness) sync(t *testing.T) {
	t.Helper()
	if err := h.ctrl.syncPool(h.ctx, "pool"); err != nil {
		t.Fatalf("syncPool: %v", err)
	}
}

func (h *sharingAffinityHarness) noErrors(t *testing.T) {
	t.Helper()
	if len(h.errs) != 0 {
		t.Fatalf("unexpected controller errors: %v (%v)", h.errs, h.messages)
	}
}

func (h *sharingAffinityHarness) noWrites(t *testing.T) {
	t.Helper()
	for _, action := range h.client.Actions() {
		if action.Matches("create", "resourceslices") || action.Matches("update", "resourceslices") || action.Matches("delete", "resourceslices") {
			t.Errorf("unexpected %s on ResourceSlices", action.GetVerb())
		}
	}
}

func (h *sharingAffinityHarness) list(t *testing.T) []resourceapi.ResourceSlice {
	t.Helper()
	list, err := h.client.ResourceV1().ResourceSlices().List(h.ctx, metav1.ListOptions{})
	if err != nil {
		t.Fatalf("list ResourceSlices: %v", err)
	}
	sortResourceSlices(list.Items)
	return list.Items
}

func (h *sharingAffinityHarness) expectPool(t *testing.T, pool Pool, generation int64) {
	t.Helper()
	got := h.list(t)
	want := sharingAffinityStoredSlices(pool, generation)
	if len(got) != len(want) {
		t.Fatalf("stored %d slices, want %d", len(got), len(want))
	}
	for i := range want {
		if diff := cmp.Diff(want[i].Spec, got[i].Spec); diff != "" {
			t.Errorf("slice %d spec (-want,+got): %s", i, diff)
		}
	}
}
func TestCopyTaintTimeAdded(t *testing.T) {
	t1 := metav1.Time{Time: time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC)}
	t2 := metav1.Time{Time: time.Date(2026, 2, 2, 0, 0, 0, 0, time.UTC)}
	t3 := metav1.Time{Time: time.Date(2026, 3, 3, 0, 0, 0, 0, time.UTC)}
	taint := func(key, value string, timeAdded *metav1.Time) resourceapi.DeviceTaint {
		return resourceapi.DeviceTaint{
			Key:       key,
			Value:     value,
			Effect:    resourceapi.DeviceTaintEffectNoExecute,
			TimeAdded: timeAdded,
		}
	}
	device := func(taints ...resourceapi.DeviceTaint) []resourceapi.Device {
		return []resourceapi.Device{newDevice("device-0", taints)}
	}

	for name, tc := range map[string]struct {
		from, to, want []resourceapi.Device
	}{
		"every-matching-taint": {
			from: device(taint("a", "value-a", &t1), taint("b", "value-b", &t2)),
			to:   device(taint("a", "value-a", nil), taint("b", "value-b", nil)),
			want: device(taint("a", "value-a", &t1), taint("b", "value-b", &t2)),
		},
		"matched-by-content-not-position": {
			from: device(taint("b", "value-b", &t2), taint("a", "value-a", &t1)),
			to:   device(taint("a", "value-a", nil), taint("b", "value-b", nil)),
			want: device(taint("a", "value-a", &t1), taint("b", "value-b", &t2)),
		},
		"explicit-time-wins": {
			from: device(taint("a", "value-a", &t1), taint("b", "value-b", &t2)),
			to:   device(taint("a", "value-a", &t3), taint("b", "value-b", nil)),
			want: device(taint("a", "value-a", &t3), taint("b", "value-b", &t2)),
		},
		"no-matching-device": {
			from: []resourceapi.Device{newDevice("other", taint("a", "value-a", &t1))},
			to:   device(taint("a", "value-a", nil)),
			want: device(taint("a", "value-a", nil)),
		},
		// A taint is the whole of key, value and effect, so an edit to any of
		// them is a different taint and starts its own clock.
		"changed-taint-does-not-inherit-the-old-time": {
			from: device(taint("a", "value-a", &t1), taint("b", "old", &t2)),
			to:   device(taint("a", "value-a", nil), taint("b", "new", nil), taint("c", "value-c", nil)),
			want: device(taint("a", "value-a", &t1), taint("b", "new", nil), taint("c", "value-c", nil)),
		},
		// from is in the other order, so a device read by position rather than
		// by name would take the wrong one's times.
		"devices-are-matched-by-name": {
			from: []resourceapi.Device{
				newDevice("device-1", taint("b", "value-b", &t2)),
				newDevice("device-0", taint("a", "value-a", &t1)),
			},
			to: []resourceapi.Device{
				newDevice("device-0", taint("a", "value-a", nil)),
				newDevice("device-1", taint("b", "value-b", nil)),
			},
			want: []resourceapi.Device{
				newDevice("device-0", taint("a", "value-a", &t1)),
				newDevice("device-1", taint("b", "value-b", &t2)),
			},
		},
	} {
		t.Run(name, func(t *testing.T) {
			// Deep copies: a shallow one keeps pointing at the same taints,
			// so a write into them would show up in the baseline as well.
			fromBefore := deepCopyDevices(tc.from)
			toBefore := deepCopyDevices(tc.to)

			got := copyTaintTimeAdded(tc.from, tc.to)

			assert.Equal(t, tc.want, got)
			// Both inputs are documented as read-only.
			assert.Equal(t, fromBefore, tc.from, "from was modified")
			assert.Equal(t, toBefore, tc.to, "to was modified")
		})
	}
}
