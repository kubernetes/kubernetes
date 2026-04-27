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
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
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
					NodeOwnerReferences(ownerName, string(nodeUID)).NodeName(ownerName).
					Driver(driverName).Devices([]resourceapi.Device{}).
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
				MakeResourceSlice().Name(generateName1+"random1").UID(generateName1+"random1").
					NodeOwnerReferences(ownerName, string(nodeUID)).NodeName(ownerName).
					Driver(driverName).Devices([]resourceapi.Device{}).
					Pool(resourceapi.ResourcePool{Name: poolName, Generation: 1, ResourceSliceCount: 1}).Obj(),
				// matching device
				MakeResourceSlice().Name(generateName2+"random2").UID(generateName2+"random2").
					NodeOwnerReferences(ownerName, string(nodeUID)).NodeName(ownerName).
					Driver(driverName).Devices([]resourceapi.Device{newDevice(deviceName)}).
					Pool(resourceapi.ResourcePool{Name: poolName, Generation: 2, ResourceSliceCount: 1}).Obj(),
				// no devices
				MakeResourceSlice().Name(generateName3+"random3").UID(generateName3+"random3").
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
					NodeOwnerReferences(ownerName, string(nodeUID)).
					PerDeviceNodeSelection(true).
					Driver(driverName).Devices([]resourceapi.Device{newDevice(deviceName, nodeSelector)}).
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
					AppOwnerReferences(ownerName).
					AllNodes(false).
					NodeName("").
					NodeSelector(nil).
					PerDeviceNodeSelection(false). // Should be dropped.
					Driver(driverName).
					Pool(resourceapi.ResourcePool{Name: poolName, Generation: 1, ResourceSliceCount: 2}).
					Obj(),
				*MakeResourceSlice().Name(resourceSlice2).GenerateName(generateName2).
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
					NodeOwnerReferences(ownerName, string(nodeUID)).NodeName(ownerName).
					Driver(driverName).Devices([]resourceapi.Device{newDevice(deviceName1)}).
					Pool(resourceapi.ResourcePool{Name: poolName, Generation: 2, ResourceSliceCount: 3}).Obj(),
				*MakeResourceSlice().Name(resourceSlice2).GenerateName(generateName2).
					NodeOwnerReferences(ownerName, string(nodeUID)).NodeName(ownerName).
					Driver(driverName).Devices([]resourceapi.Device{newDevice(deviceName2)}).
					Pool(resourceapi.ResourcePool{Name: poolName, Generation: 2, ResourceSliceCount: 3}).Obj(),
				*MakeResourceSlice().Name(resourceSlice3).GenerateName(generateName3).
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
					NodeOwnerReferences(ownerName, string(nodeUID)).NodeName(ownerName).
					Driver(driverName).Devices([]resourceapi.Device{newDevice(deviceName1)}).
					Pool(resourceapi.ResourcePool{Name: poolName, Generation: 2, ResourceSliceCount: 3}).Obj(),
				*MakeResourceSlice().Name(resourceSlice2).GenerateName(generateName2).UID(resourceSlice2).ResourceVersion("1").
					NodeOwnerReferences(ownerName, string(nodeUID)).NodeName(ownerName).
					Driver(driverName).Devices([]resourceapi.Device{newDevice(deviceName2)}).
					Pool(resourceapi.ResourcePool{Name: poolName, Generation: 2, ResourceSliceCount: 3}).Obj(),
				*MakeResourceSlice().Name(generateName3+"1").GenerateName(generateName3).
					NodeOwnerReferences(ownerName, string(nodeUID)).NodeName(ownerName).
					Driver(driverName).Devices([]resourceapi.Device{newDevice(deviceName3)}).
					Pool(resourceapi.ResourcePool{Name: poolName, Generation: 2, ResourceSliceCount: 3}).Obj(),
			},
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

// TestControllerUpdateReconcilePoolWithNameValidation verifies that Update rejects
// invalid pool sets when ReconcilePoolWithName is set
func TestControllerUpdateReconcilePoolWithNameValidation(t *testing.T) {
	const poolName = "pool"

	testcases := map[string]struct {
		resources      *DriverResources
		expectedErrors []string
	}{
		"multiple pools returns error": {
			resources: &DriverResources{
				Pools: map[string]Pool{
					poolName:     {Slices: []Slice{{Devices: []resourceapi.Device{}}}},
					"other-pool": {Slices: []Slice{{Devices: []resourceapi.Device{}}}},
				},
			},
			expectedErrors: []string{"found 2 pools; expected exactly one pool with this name"},
		},

		"wrong pool only returns error": {
			resources: &DriverResources{
				Pools: map[string]Pool{
					"other-pool": {Slices: []Slice{{Devices: []resourceapi.Device{}}}},
				},
			},
			expectedErrors: []string{"found 1 pools; expected exactly one pool with this name"},
		},

		"empty pools succeeds": {
			resources: &DriverResources{Pools: map[string]Pool{}},
		},

		"single matching pool succeeds": {
			resources: &DriverResources{
				Pools: map[string]Pool{
					poolName: {Slices: []Slice{{Devices: []resourceapi.Device{}}}},
				},
			},
		},
	}

	for name, tc := range testcases {
		t.Run(name, func(t *testing.T) {
			ctrl := &Controller{
				reconcilePoolWithName: poolName,
				queue:                 ptr.To(workqueue.Mock[string]{}),
				errorHandler: func(_ context.Context, err error, _ string) {
					if len(tc.expectedErrors) > 0 {
						require.Error(t, err)
						for _, expectedError := range tc.expectedErrors {
							assert.Contains(t, err.Error(), expectedError)
						}
						return
					}

					require.NoError(t, err)
				},
			}
			ctrl.Update(tc.resources)
		})
	}
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
	disableBindingConditions    bool
	disableDeviceTaints         bool
	disablePartitionableDevices bool
	disableConsumableCapacity   bool
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
