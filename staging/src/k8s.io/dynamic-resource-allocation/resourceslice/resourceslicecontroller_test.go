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
	resourceapi "k8s.io/api/resource/v1beta2"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
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
		resourceSlice1 = "resource-slice-1"
		resourceSlice2 = "resource-slice-2"
		resourceSlice3 = "resource-slice-3"
		generateName   = ownerName + "-" + driverName + "-"
		generatedName1 = generateName + "0"
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
		// initialObjects is a list of initial resource slices to be used in the test.
		initialObjects         []runtime.Object
		initialOtherObjects    []runtime.Object
		inputDriverResources   *DriverResources
		expectedResourceSlices []resourceapi.ResourceSlice
		expectedStats          Stats
		expectedError          string
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
				*MakeResourceSlice().Name(generatedName1).GenerateName(generateName).
					NodeOwnerReferences(ownerName, string(nodeUID)).NodeName(ownerName).
					Driver(driverName).Devices([]resourceapi.Device{}).
					Pool(resourceapi.ResourcePool{Name: poolName, Generation: 1, ResourceSliceCount: 1}).Obj(),
			},
		},
		"keep-slice-unchanged": {
			nodeUID: nodeUID,
			initialObjects: []runtime.Object{
				MakeResourceSlice().Name(generatedName1).GenerateName(generateName).
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
				*MakeResourceSlice().Name(generatedName1).GenerateName(generateName).
					NodeOwnerReferences(ownerName, string(nodeUID)).NodeName(ownerName).
					Driver(driverName).Devices([]resourceapi.Device{newDevice(deviceName)}).
					Pool(resourceapi.ResourcePool{Name: poolName, Generation: 1, ResourceSliceCount: 1}).Obj(),
			},
		},
		"keep-taint-unchanged": {
			nodeUID: nodeUID,
			initialObjects: []runtime.Object{
				MakeResourceSlice().Name(generatedName1).GenerateName(generateName).
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
				*MakeResourceSlice().Name(generatedName1).GenerateName(generateName).
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
		"add-taints": {
			nodeUID: nodeUID,
			initialObjects: []runtime.Object{
				MakeResourceSlice().Name(generatedName1).GenerateName(generateName).
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
				*MakeResourceSlice().Name(generatedName1).GenerateName(generateName).
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
				MakeResourceSlice().Name(generatedName1).GenerateName(generateName).
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
				*MakeResourceSlice().Name(generatedName1).GenerateName(generateName).
					ResourceVersion("1").
					NodeOwnerReferences(ownerName, string(nodeUID)).NodeName(ownerName).
					Driver(driverName).
					Devices([]resourceapi.Device{newDevice(deviceName)}).
					Pool(resourceapi.ResourcePool{Name: poolName, Generation: 1, ResourceSliceCount: 1}).
					Obj(),
			},
			expectedError: `update ResourceSlice: pool "pool", slice #0: some fields were dropped by the apiserver, probably because these features are disabled: DRADeviceTaints`,
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
		"delete-and-add-slice-when-more-than-one-existing-or-desired-slice": {
			nodeUID: nodeUID,
			initialObjects: []runtime.Object{
				// No devices in first ResourceSlice.
				MakeResourceSlice().Name(resourceSlice1).UID(resourceSlice1).
					NodeOwnerReferences(ownerName, string(nodeUID)).NodeName(ownerName).
					Driver(driverName).Devices([]resourceapi.Device{}).
					Pool(resourceapi.ResourcePool{Name: poolName, Generation: 1, ResourceSliceCount: 2}).Obj(),
				MakeResourceSlice().Name(resourceSlice2).UID(resourceSlice2).
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
				NumCreates: 1,
				NumUpdates: 1,
				NumDeletes: 1,
			},
			expectedResourceSlices: []resourceapi.ResourceSlice{
				*MakeResourceSlice().Name(generatedName1).GenerateName(generateName).
					NodeOwnerReferences(ownerName, string(nodeUID)).NodeName(ownerName).
					Driver(driverName).Devices([]resourceapi.Device{newDevice(deviceName1)}).
					Pool(resourceapi.ResourcePool{Name: poolName, Generation: 2, ResourceSliceCount: 2}).Obj(),
				*MakeResourceSlice().Name(resourceSlice2).UID(resourceSlice2).ResourceVersion("1").
					NodeOwnerReferences(ownerName, string(nodeUID)).NodeName(ownerName).
					Driver(driverName).Devices([]resourceapi.Device{newDevice(deviceName2)}).
					Pool(resourceapi.ResourcePool{Name: poolName, Generation: 2, ResourceSliceCount: 2}).Obj(),
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
				NumDeletes: 2,
			},
			expectedResourceSlices: []resourceapi.ResourceSlice{
				*MakeResourceSlice().Name(resourceSlice2).UID(resourceSlice2).
					NodeOwnerReferences(ownerName, string(nodeUID)).NodeName(ownerName).
					Driver(driverName).Devices([]resourceapi.Device{newDevice(deviceName)}).
					Pool(resourceapi.ResourcePool{Name: poolName, Generation: 2, ResourceSliceCount: 1}).Obj(),
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
				*MakeResourceSlice().GenerateName(generateName).Name(generatedName1).
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
						Slices: []Slice{{Devices: []resourceapi.Device{newDevice(deviceName)}}},
					},
				},
			},
			expectedStats: Stats{
				NumCreates: 1,
			},
			expectedResourceSlices: []resourceapi.ResourceSlice{
				*MakeResourceSlice().Name(driverName + "-0").GenerateName(driverName + "-").
					AllNodes(true).
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
				*MakeResourceSlice().Name(generatedName1).GenerateName(generateName).
					AppOwnerReferences(ownerName).NodeSelector(nodeSelector).
					Driver(driverName).Devices([]resourceapi.Device{newDevice(deviceName)}).
					Pool(resourceapi.ResourcePool{Name: poolName, Generation: 1, ResourceSliceCount: 1}).Obj(),
			},
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
		"create-partitionable-device": {
			nodeUID: nodeUID,
			inputDriverResources: &DriverResources{
				Pools: map[string]Pool{
					poolName: {
						Generation: 1,
						Slices: []Slice{{
							PerDeviceNodeSelection: ptr.To(true),
							SharedCounters: []resourceapi.CounterSet{{
								Name: "gpu-0",
								Counters: map[string]resourceapi.Counter{
									"mem": {Value: resource.MustParse("1")},
								},
							}},
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
						}},
					},
				},
			},
			expectedStats: Stats{
				NumCreates: 1,
			},
			expectedResourceSlices: []resourceapi.ResourceSlice{
				*MakeResourceSlice().Name(generatedName1).GenerateName(generateName).
					NodeOwnerReferences(ownerName, string(nodeUID)).NodeName(ownerName).
					PerDeviceNodeSelection(true).
					SharedCounters([]resourceapi.CounterSet{{
						Name: "gpu-0",
						Counters: map[string]resourceapi.Counter{
							"mem": {Value: resource.MustParse("1")},
						},
					}}).
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
					Pool(resourceapi.ResourcePool{Name: poolName, Generation: 1, ResourceSliceCount: 1}).
					Obj(),
			},
		},
		"drop-partitionable-device": {
			features: features{disablePartitionableDevices: true},
			nodeUID:  nodeUID,
			inputDriverResources: &DriverResources{
				Pools: map[string]Pool{
					poolName: {
						Generation: 1,
						Slices: []Slice{{
							PerDeviceNodeSelection: ptr.To(true),
							SharedCounters: []resourceapi.CounterSet{{
								Name: "gpu-0",
								Counters: map[string]resourceapi.Counter{
									"mem": {Value: resource.MustParse("1")},
								},
							}},
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
						}},
					},
				},
			},
			expectedStats: Stats{
				NumCreates: 1,
			},
			expectedResourceSlices: []resourceapi.ResourceSlice{
				*MakeResourceSlice().Name(generatedName1).GenerateName(generateName).
					NodeOwnerReferences(ownerName, string(nodeUID)).NodeName(ownerName).
					Driver(driverName).
					Devices([]resourceapi.Device{newDevice(deviceName)}).
					Pool(resourceapi.ResourcePool{Name: poolName, Generation: 1, ResourceSliceCount: 1}).
					Obj(),
			},
			expectedError: `create ResourceSlice: pool "pool", slice #0: some fields were dropped by the apiserver, probably because these features are disabled: DRAPartitionableDevices`,
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
			})
			defer ctrl.Stop()
			require.NoError(t, err, "unexpected controller creation error")

			// Process work items in the queue until the queue is empty.
			// Processing races with informers adding new work items,
			// but the desired state should already be reached in the
			// first iteration, so all following iterations should be nops.
			ctrl.run(ctx)

			// Check ResourceSlices
			resourceSlices, err := kubeClient.ResourceV1beta2().ResourceSlices().List(ctx, metav1.ListOptions{})
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

			ctrl.Stop()
			switch {
			case test.expectedError != "" && len(controllerErrors) == 0:
				t.Errorf("expected error, got none: %s", test.expectedError)
			case test.expectedError == "" && len(controllerErrors) > 0:
				t.Errorf("expected no error, got:\n  %s", strings.Join(formatErrors(controllerErrors), "\n  "))
			case test.expectedError != "" && len(controllerErrors) != 1:
				t.Errorf("expected one error %q, got:\n  %s", test.expectedError, strings.Join(formatErrors(controllerErrors), "\n  "))
			case test.expectedError != "":
				assert.Equal(t, test.expectedError, controllerErrors[0].Error())
			}
		})
	}
}

func formatErrors(errs []error) []string {
	var errMsgs []string
	for _, err := range errs {
		var droppedFields *DroppedFieldsError
		var errMsg string
		if errors.As(err, &droppedFields) {
			errMsg = fmt.Sprintf("%v\n%s", err, cmp.Diff(droppedFields.DesiredSlice.Spec, droppedFields.ActualSlice.Spec))
		} else {
			errMsg = err.Error()
		}
		errMsgs = append(errMsgs, errMsg)
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
	disableDeviceTaints         bool
	disablePartitionableDevices bool
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
				Pool:    resourceapi.ResourcePool{},
				Devices: []resourceapi.Device{},
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

type nodeNameField string

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
		default:
			panic(fmt.Sprintf("unsupported resourceapi.Device field type %T", field))
		}
	}
	return device
}
