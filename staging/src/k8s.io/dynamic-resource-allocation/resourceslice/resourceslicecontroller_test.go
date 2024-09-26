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
	"fmt"
	"sort"
	"sync"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	v1 "k8s.io/api/core/v1"
	resourceapi "k8s.io/api/resource/v1alpha3"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/kubernetes/fake"
	k8stesting "k8s.io/client-go/testing"
	"k8s.io/klog/v2/ktesting"
	"k8s.io/utils/ptr"
)

func TestControllerSyncPool(t *testing.T) {

	var (
		node           = "node"
		nodeUID        = types.UID("node-uid")
		driveName      = "driver"
		poolName       = "pool"
		poolName1      = "pool-1"
		deviceName     = "device"
		resourceSlice  = "resource-slice"
		resourceSlice1 = "resource-slice-1"
		resourceSlice2 = "resource-slice-2"
		resourceSlice3 = "resource-slice-3"
	)

	testCases := map[string]struct {
		nodeUID types.UID
		// initialObjects is a list of initial resource slices to be used in the test.
		initialObjects         []runtime.Object
		initialOtherObjects    []runtime.Object
		poolName               string
		inputDriverResources   *DriverResources
		expectedResourceSlices []resourceapi.ResourceSlice
		// expectedResourceSliceTotalCount is the expected total count of resource slice.
		expectedResourceSliceTotalCount int
		// expectedResourceSliceChangeCount is the expected count of
		// resource slice changes (created or deleted).
		expectedResourceSliceChangeCount int
	}{
		"no-resourceslices-existing-with-no-devices": {
			nodeUID:        nodeUID,
			initialObjects: []runtime.Object{},
			poolName:       poolName,
			inputDriverResources: &DriverResources{
				Pools: map[string]Pool{
					poolName: {
						Devices: []resourceapi.Device{},
					},
				},
			},
			expectedResourceSlices: []resourceapi.ResourceSlice{
				*MakeResourceSlice().Name(node+"-"+driveName+"-0").GenerateName(node+"-"+driveName+"-").
					OwnerReferences(node, string(nodeUID)).NodeName(node).
					Driver(driveName).Devices([]resourceapi.Device{}).
					Pool(resourceapi.ResourcePool{Name: poolName, Generation: 1, ResourceSliceCount: 1}).Obj(),
			},
			expectedResourceSliceTotalCount:  1,
			expectedResourceSliceChangeCount: 1,
		},
		"no-resourceslices-existing-with-one-device": {
			nodeUID:        nodeUID,
			initialObjects: []runtime.Object{},
			poolName:       poolName,
			inputDriverResources: &DriverResources{
				Pools: map[string]Pool{
					poolName: {
						Generation: 1,
						Devices: []resourceapi.Device{
							{
								Name: deviceName,
							},
						},
					},
				},
			},
			expectedResourceSlices: []resourceapi.ResourceSlice{
				*MakeResourceSlice().Name(node+"-"+driveName+"-0").GenerateName(node+"-"+driveName+"-").
					OwnerReferences(node, string(nodeUID)).NodeName(node).
					Driver(driveName).Devices([]resourceapi.Device{{Name: deviceName}}).
					Pool(resourceapi.ResourcePool{Name: poolName, Generation: 2, ResourceSliceCount: 1}).Obj(),
			},
			expectedResourceSliceTotalCount:  1,
			expectedResourceSliceChangeCount: 1,
		},
		"no-resourceslice-existing-with-mismatching-resource-pool-name": {
			nodeUID:        nodeUID,
			initialObjects: []runtime.Object{},
			poolName:       poolName,
			inputDriverResources: &DriverResources{
				Pools: map[string]Pool{
					poolName1: {
						Generation: 1,
						Devices: []resourceapi.Device{
							{
								Name: deviceName,
							},
						},
					},
				},
			},
			expectedResourceSlices:           nil,
			expectedResourceSliceTotalCount:  0,
			expectedResourceSliceChangeCount: 0,
		},
		"single-resourceslice-existing-with-one-device": {
			nodeUID: nodeUID,
			initialObjects: []runtime.Object{
				// no devices
				MakeResourceSlice().Name(resourceSlice1).UID(resourceSlice1).
					OwnerReferences(node, string(nodeUID)).NodeName(node).
					Driver(driveName).Devices([]resourceapi.Device{}).
					Pool(resourceapi.ResourcePool{Name: poolName, Generation: 1, ResourceSliceCount: 1}).Obj(),
			},
			poolName: poolName,
			inputDriverResources: &DriverResources{
				Pools: map[string]Pool{
					poolName: {
						Devices: []resourceapi.Device{
							{
								Name: deviceName,
							},
						},
					},
				},
			},
			expectedResourceSlices: []resourceapi.ResourceSlice{
				*MakeResourceSlice().Name(resourceSlice1).UID(resourceSlice1).
					OwnerReferences(node, string(nodeUID)).NodeName(node).
					Driver(driveName).Devices([]resourceapi.Device{{Name: deviceName}}).
					Pool(resourceapi.ResourcePool{Name: poolName, Generation: 1, ResourceSliceCount: 1}).Obj(),
			},
			expectedResourceSliceTotalCount:  1,
			expectedResourceSliceChangeCount: 0,
		},
		"single-resourceslice-existing-with-nodeUID-empty": {
			nodeUID: "",
			initialObjects: []runtime.Object{
				// no devices
				MakeResourceSlice().Name(resourceSlice1).UID(resourceSlice1).
					NodeName(node).
					Driver(driveName).Devices([]resourceapi.Device{}).
					Pool(resourceapi.ResourcePool{Name: poolName, Generation: 1, ResourceSliceCount: 1}).Obj(),
			},
			initialOtherObjects: []runtime.Object{
				newNode(node, nodeUID),
			},
			poolName: poolName,
			inputDriverResources: &DriverResources{
				Pools: map[string]Pool{
					poolName: {
						Devices: []resourceapi.Device{
							{
								Name: deviceName,
							},
						},
					},
				},
			},
			expectedResourceSlices: []resourceapi.ResourceSlice{
				*MakeResourceSlice().Name(resourceSlice1).UID(resourceSlice1).
					OwnerReferences(node, string(nodeUID)).NodeName(node).
					Driver(driveName).Devices([]resourceapi.Device{{Name: deviceName}}).
					Pool(resourceapi.ResourcePool{Name: poolName, Generation: 1, ResourceSliceCount: 1}).Obj(),
			},
			expectedResourceSliceTotalCount:  1,
			expectedResourceSliceChangeCount: 0,
		},
		"single-resourceslice-existing-with-mismatching-resource-pool-name": {
			nodeUID: nodeUID,
			initialObjects: []runtime.Object{
				// no devices
				MakeResourceSlice().Name(resourceSlice1).UID(resourceSlice1).
					OwnerReferences(node, string(nodeUID)).NodeName(node).
					Driver(driveName).Devices([]resourceapi.Device{}).
					Pool(resourceapi.ResourcePool{Name: poolName, Generation: 1, ResourceSliceCount: 1}).Obj(),
			},
			poolName: poolName,
			inputDriverResources: &DriverResources{
				Pools: map[string]Pool{
					poolName1: {
						Generation: 1,
						Devices: []resourceapi.Device{
							{
								Name: deviceName,
							},
						},
					},
				},
			},
			expectedResourceSlices:           nil,
			expectedResourceSliceTotalCount:  0,
			expectedResourceSliceChangeCount: -1,
		},
		"single-resourceslice-with-more-than-128-devices-with-no-device": {
			nodeUID: nodeUID,
			initialObjects: []runtime.Object{
				// no devices
				MakeResourceSlice().Name(resourceSlice1).UID(resourceSlice1).
					OwnerReferences(node, string(nodeUID)).NodeName(node).
					Driver(driveName).Devices([]resourceapi.Device{}).
					Pool(resourceapi.ResourcePool{Name: poolName, Generation: 1, ResourceSliceCount: 1}).Obj(),
			},
			poolName: poolName,
			inputDriverResources: &DriverResources{
				Pools: map[string]Pool{
					poolName: {
						// We need to synchronize a total of 128 + 10 devices
						// The naming convention for the devices follows the pattern from "device-000000" to "device-000137"
						Devices: generateDevices(0, resourceapi.ResourceSliceMaxDevices+10),
					},
				},
			},
			expectedResourceSlices:           generateExpectedResourceSlices(resourceSlice, node, string(nodeUID), driveName, poolName, resourceapi.ResourceSliceMaxDevices+10, 2, 1),
			expectedResourceSliceTotalCount:  2,
			expectedResourceSliceChangeCount: 1,
		},
		"single-resourceslices-with-more-than-128-devices-with-noneeded-device": {
			nodeUID: nodeUID,
			initialObjects: []runtime.Object{
				// 10 noneeded devices, we have to remove those devices
				// The naming convention for the devices follows the pattern from "device-000300" to "device-000309"
				MakeResourceSlice().Name(resourceSlice1).UID(resourceSlice1).
					OwnerReferences(node, string(nodeUID)).NodeName(node).
					Driver(driveName).Devices(generateDevices(300, 10)).
					Pool(resourceapi.ResourcePool{Name: poolName, Generation: 1, ResourceSliceCount: 1}).Obj(),
			},
			poolName: poolName,
			inputDriverResources: &DriverResources{
				Pools: map[string]Pool{
					poolName: {
						// We need to synchronize a total of 128*2 + 10 devices
						// The naming convention for the devices follows the pattern from "device-000000" to "device-000265"
						Devices: generateDevices(0, resourceapi.ResourceSliceMaxDevices*2+10),
					},
				},
			},
			expectedResourceSlices:           generateExpectedResourceSlices(resourceSlice, node, string(nodeUID), driveName, poolName, resourceapi.ResourceSliceMaxDevices*2+10, 2, 1),
			expectedResourceSliceTotalCount:  3,
			expectedResourceSliceChangeCount: 2,
		},
		"multiple-resourceslices-existing": {
			nodeUID: nodeUID,
			initialObjects: []runtime.Object{
				MakeResourceSlice().Name(resourceSlice1).UID(resourceSlice1).
					OwnerReferences(node, string(nodeUID)).NodeName(node).
					Driver(driveName).Devices([]resourceapi.Device{}).
					Pool(resourceapi.ResourcePool{Name: poolName, Generation: 1, ResourceSliceCount: 1}).Obj(),
				MakeResourceSlice().Name(resourceSlice2).UID(resourceSlice2).
					OwnerReferences(node, string(nodeUID)).NodeName(node).
					Driver(driveName).Devices([]resourceapi.Device{}).
					Pool(resourceapi.ResourcePool{Name: poolName, Generation: 1, ResourceSliceCount: 1}).Obj(),
				MakeResourceSlice().Name(resourceSlice3).UID(resourceSlice3).
					OwnerReferences(node, string(nodeUID)).NodeName(node).
					Driver(driveName).Devices([]resourceapi.Device{}).
					Pool(resourceapi.ResourcePool{Name: poolName, Generation: 1, ResourceSliceCount: 1}).Obj(),
			},
			poolName: poolName,
			inputDriverResources: &DriverResources{
				Pools: map[string]Pool{
					poolName: {
						Devices: []resourceapi.Device{
							{
								Name: deviceName,
							},
						},
					},
				},
			},
			expectedResourceSlices: []resourceapi.ResourceSlice{
				*MakeResourceSlice().Name(resourceSlice1).UID(resourceSlice1).
					OwnerReferences(node, string(nodeUID)).NodeName(node).
					Driver(driveName).Devices([]resourceapi.Device{{Name: deviceName}}).
					Pool(resourceapi.ResourcePool{Name: poolName, Generation: 2, ResourceSliceCount: 1}).Obj(),
			},
			expectedResourceSliceTotalCount:  1,
			expectedResourceSliceChangeCount: -2,
		},
		"multiple-resourceslice-existing-with-different-resource-pool-generation": {
			nodeUID: nodeUID,
			initialObjects: []runtime.Object{
				// no devices
				MakeResourceSlice().Name(resourceSlice1).UID(resourceSlice1).
					OwnerReferences(node, string(nodeUID)).NodeName(node).
					Driver(driveName).Devices([]resourceapi.Device{}).
					Pool(resourceapi.ResourcePool{Name: poolName, Generation: 1, ResourceSliceCount: 1}).Obj(),
				// no devices
				MakeResourceSlice().Name(resourceSlice2).UID(resourceSlice2).
					OwnerReferences(node, string(nodeUID)).NodeName(node).
					Driver(driveName).Devices([]resourceapi.Device{}).
					Pool(resourceapi.ResourcePool{Name: poolName, Generation: 2, ResourceSliceCount: 1}).Obj(),
				// no devices
				MakeResourceSlice().Name(resourceSlice3).UID(resourceSlice3).
					OwnerReferences(node, string(nodeUID)).NodeName(node).
					Driver(driveName).Devices([]resourceapi.Device{}).
					Pool(resourceapi.ResourcePool{Name: poolName, Generation: 1, ResourceSliceCount: 1}).Obj(),
			},
			poolName: poolName,
			inputDriverResources: &DriverResources{
				Pools: map[string]Pool{
					poolName: {
						Generation: 3,
						Devices: []resourceapi.Device{
							{
								Name: deviceName,
							},
						},
					},
				},
			},
			expectedResourceSlices: []resourceapi.ResourceSlice{
				*MakeResourceSlice().Name(resourceSlice2).UID(resourceSlice2).
					OwnerReferences(node, string(nodeUID)).NodeName(node).
					Driver(driveName).Devices([]resourceapi.Device{{Name: deviceName}}).
					Pool(resourceapi.ResourcePool{Name: poolName, Generation: 4, ResourceSliceCount: 1}).Obj(),
			},
			expectedResourceSliceTotalCount:  1,
			expectedResourceSliceChangeCount: -2,
		},
		"multiple-resourceslices-with-more-than-128-devices-with-no-device": {
			nodeUID: nodeUID,
			initialObjects: []runtime.Object{
				// no devices
				MakeResourceSlice().Name(resourceSlice1).UID(resourceSlice1).
					OwnerReferences(node, string(nodeUID)).NodeName(node).
					Driver(driveName).Devices([]resourceapi.Device{}).
					Pool(resourceapi.ResourcePool{Name: poolName, Generation: 1, ResourceSliceCount: 1}).Obj(),
				// no devices
				MakeResourceSlice().Name(resourceSlice2).UID(resourceSlice2).
					OwnerReferences(node, string(nodeUID)).NodeName(node).
					Driver(driveName).Devices([]resourceapi.Device{}).
					Pool(resourceapi.ResourcePool{Name: poolName, Generation: 1, ResourceSliceCount: 1}).Obj(),
			},
			poolName: poolName,
			inputDriverResources: &DriverResources{
				Pools: map[string]Pool{
					poolName: {
						// We need to synchronize a total of 128*2 + 10 devices
						// The naming convention for the devices follows the pattern from "device-000000" to "device-000265"
						Devices: generateDevices(0, resourceapi.ResourceSliceMaxDevices*2+10),
					},
				},
			},
			expectedResourceSlices:           generateExpectedResourceSlices(resourceSlice, node, string(nodeUID), driveName, poolName, resourceapi.ResourceSliceMaxDevices*2+10, 2, 2),
			expectedResourceSliceTotalCount:  3,
			expectedResourceSliceChangeCount: 1,
		},
		"multiple-resourceslices-with-more-than-128-devices-with-needed-device-and-noneeded-device": {
			nodeUID: nodeUID,
			initialObjects: []runtime.Object{
				// 20 needed devices, we have to keep those devices
				// The naming convention for the devices follows the pattern from "device-000010" to "device-000029"
				MakeResourceSlice().Name(resourceSlice1).UID(resourceSlice1).
					OwnerReferences(node, string(nodeUID)).NodeName(node).
					Driver(driveName).Devices(generateDevices(10, 20)).
					Pool(resourceapi.ResourcePool{Name: poolName, Generation: 1, ResourceSliceCount: 1}).Obj(),
				// 10 noneeded devices, we have to remove those devices
				// The naming convention for the devices follows the pattern from "device-000300" to "device-000309"
				MakeResourceSlice().Name(resourceSlice2).UID(resourceSlice2).
					OwnerReferences(node, string(nodeUID)).NodeName(node).
					Driver(driveName).Devices(generateDevices(300, 10)).
					Pool(resourceapi.ResourcePool{Name: poolName, Generation: 1, ResourceSliceCount: 1}).Obj(),
			},
			poolName: poolName,
			inputDriverResources: &DriverResources{
				Pools: map[string]Pool{
					poolName: {
						// We need to synchronize a total of 128*2 + 10 devices
						// The naming convention for the devices follows the pattern from "device-000000" to "device-000265"
						Devices: generateDevices(0, resourceapi.ResourceSliceMaxDevices*2+10),
					},
				},
			},
			expectedResourceSlices:           generateExpectedResourceSlices(resourceSlice, node, string(nodeUID), driveName, poolName, resourceapi.ResourceSliceMaxDevices*2+10, 2, 2),
			expectedResourceSliceTotalCount:  3,
			expectedResourceSliceChangeCount: 1,
		},
		"multiple-resourceslices-with-more-than-128-devices-with-needed-device-and-noneeded-device-and-no-device": {
			nodeUID: nodeUID,
			initialObjects: []runtime.Object{
				// 50 needed devices, we have to keep those devices
				// The naming convention for the devices follows the pattern from "device-000010" to "device-000059"
				MakeResourceSlice().Name(resourceSlice1).UID(resourceSlice1).
					OwnerReferences(node, string(nodeUID)).NodeName(node).
					Driver(driveName).Devices(generateDevices(10, 50)).
					Pool(resourceapi.ResourcePool{Name: poolName, Generation: 1, ResourceSliceCount: 1}).Obj(),
				// 80 noneeded devices, we have to remove those devices
				// The naming convention for the devices follows the pattern from "device-000400" to "device-000479"
				MakeResourceSlice().Name(resourceSlice2).UID(resourceSlice2).
					OwnerReferences(node, string(nodeUID)).NodeName(node).
					Driver(driveName).Devices(generateDevices(400, 80)).
					Pool(resourceapi.ResourcePool{Name: poolName, Generation: 1, ResourceSliceCount: 1}).Obj(),
				// no devices
				MakeResourceSlice().Name(resourceSlice3).UID(resourceSlice3).
					OwnerReferences(node, string(nodeUID)).NodeName(node).
					Driver(driveName).Devices([]resourceapi.Device{}).
					Pool(resourceapi.ResourcePool{Name: poolName, Generation: 1, ResourceSliceCount: 1}).Obj(),
			},
			poolName: poolName,
			inputDriverResources: &DriverResources{
				Pools: map[string]Pool{
					poolName: {
						// We need to synchronize a total of 128*3 + 10 devices
						// The naming convention for the devices follows the pattern from "device-000000" to "device-000393"
						Devices: generateDevices(0, resourceapi.ResourceSliceMaxDevices*3+10),
					},
				},
			},
			expectedResourceSlices:           generateExpectedResourceSlices(resourceSlice, node, string(nodeUID), driveName, poolName, resourceapi.ResourceSliceMaxDevices*3+10, 2, 3),
			expectedResourceSliceTotalCount:  4,
			expectedResourceSliceChangeCount: 1,
		},
		"multiple-resourceslices-with-no-devices": {
			nodeUID: nodeUID,
			initialObjects: []runtime.Object{
				// 50 needed devices, we have to keep those devices
				// The naming convention for the devices follows the pattern from "device-000010" to "device-000059"
				MakeResourceSlice().Name(resourceSlice1).UID(resourceSlice1).
					OwnerReferences(node, string(nodeUID)).NodeName(node).
					Driver(driveName).Devices(generateDevices(10, 50)).
					Pool(resourceapi.ResourcePool{Name: poolName, Generation: 1, ResourceSliceCount: 1}).Obj(),
				// 80 noneeded devices, we have to remove those devices
				// The naming convention for the devices follows the pattern from "device-000300" to "device-000379"
				MakeResourceSlice().Name(resourceSlice2).UID(resourceSlice2).
					OwnerReferences(node, string(nodeUID)).NodeName(node).
					Driver(driveName).Devices(generateDevices(300, 80)).
					Pool(resourceapi.ResourcePool{Name: poolName, Generation: 1, ResourceSliceCount: 1}).Obj(),
				// no devices
				MakeResourceSlice().Name(resourceSlice3).UID(resourceSlice3).
					OwnerReferences(node, string(nodeUID)).NodeName(node).
					Driver(driveName).Devices([]resourceapi.Device{}).
					Pool(resourceapi.ResourcePool{Name: poolName, Generation: 1, ResourceSliceCount: 1}).Obj(),
			},
			poolName: poolName,
			inputDriverResources: &DriverResources{
				Pools: map[string]Pool{
					poolName: {
						Devices: []resourceapi.Device{},
					},
				},
			},
			expectedResourceSlices: []resourceapi.ResourceSlice{
				*MakeResourceSlice().Name(resourceSlice1).UID(resourceSlice1).
					OwnerReferences(node, string(nodeUID)).NodeName(node).
					Driver(driveName).Devices([]resourceapi.Device{}).
					Pool(resourceapi.ResourcePool{Name: poolName, Generation: 2, ResourceSliceCount: 1}).Obj(),
			},
			expectedResourceSliceTotalCount:  1,
			expectedResourceSliceChangeCount: -2,
		},
		"no-resourceslices-with-more-than-128-devices-one": {
			nodeUID:        nodeUID,
			initialObjects: []runtime.Object{},
			poolName:       poolName,
			inputDriverResources: &DriverResources{
				Pools: map[string]Pool{
					poolName: {
						Generation: 1,
						// We need to synchronize a total of 128*2 + 10 devices
						// The naming convention for the devices follows the pattern from "device-000000" to "device-000265"
						Devices: generateDevices(0, resourceapi.ResourceSliceMaxDevices*2+10),
					},
				},
			},
			expectedResourceSlices:           generateExpectedResourceSlices(resourceSlice, node, string(nodeUID), driveName, poolName, resourceapi.ResourceSliceMaxDevices*2+10, 2, 0),
			expectedResourceSliceTotalCount:  3,
			expectedResourceSliceChangeCount: 3,
		},
		"no-resourceslices-with-more-than-128-devices-two": {
			nodeUID:        nodeUID,
			initialObjects: []runtime.Object{},
			poolName:       poolName,
			inputDriverResources: &DriverResources{
				Pools: map[string]Pool{
					poolName: {
						Generation: 1,
						// We need to synchronize a total of 128*3 + 10 devices
						// The naming convention for the devices follows the pattern from "device-000000" to "device-000393"
						Devices: generateDevices(0, resourceapi.ResourceSliceMaxDevices*3+10),
					},
				},
			},
			expectedResourceSlices:           generateExpectedResourceSlices(resourceSlice, node, string(nodeUID), driveName, poolName, resourceapi.ResourceSliceMaxDevices*3+10, 2, 0),
			expectedResourceSliceTotalCount:  4,
			expectedResourceSliceChangeCount: 4,
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
			kubeClient := createTestClient(inputObjects...)
			owner := Owner{
				APIVersion: "v1",
				Kind:       "Node",
				Name:       node,
				UID:        test.nodeUID,
			}
			ctrl, err := newController(ctx, kubeClient, driveName, owner, test.inputDriverResources)
			defer ctrl.Stop()
			require.NoError(t, err, "unexpected controller creation error")
			err = ctrl.syncPool(ctx, test.poolName)
			require.NoError(t, err, "unexpected syncPool error")

			// Check ResourceSlices
			resourceSlices, err := kubeClient.ResourceV1alpha3().ResourceSlices().List(ctx, metav1.ListOptions{})
			require.NoError(t, err, "list resource slices")

			sortResourceSlices(test.expectedResourceSlices)
			sortResourceSlices(resourceSlices.Items)
			assert.Equal(t, test.expectedResourceSlices, resourceSlices.Items)
			resourceSlicesLen := len(resourceSlices.Items)
			initResourceSliceLen := len(test.initialObjects)
			assert.Equal(t, test.expectedResourceSliceTotalCount, resourceSlicesLen)
			assert.Equal(t, test.expectedResourceSliceChangeCount, resourceSlicesLen-initResourceSliceLen)
		})
	}
}

func sortResourceSlices(slices []resourceapi.ResourceSlice) {
	sort.Slice(slices, func(i, j int) bool {
		if len(slices[i].Name) == 0 && len(slices[j].Name) == 0 {
			return slices[i].ObjectMeta.GenerateName < slices[j].ObjectMeta.GenerateName
		}
		return slices[i].Name < slices[j].Name
	})
}

func newNode(name string, uid types.UID) *v1.Node {
	return &v1.Node{
		ObjectMeta: metav1.ObjectMeta{
			Name: name,
			UID:  uid,
		},
	}
}

func createTestClient(objects ...runtime.Object) *fake.Clientset {
	fakeClient := fake.NewSimpleClientset(objects...)
	fakeClient.PrependReactor("create", "resourceslices", createResourceSliceReactor())
	return fakeClient
}

// createResourceSliceReactor implements the logic required for the GenerateName field to work when using
// the fake client. Add it with client.PrependReactor to your fake client.
func createResourceSliceReactor() func(action k8stesting.Action) (handled bool, ret runtime.Object, err error) {
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
		return false, nil, nil
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

// OwnerReferences sets the value of ResourceSlice.ObjectMeta.OwnerReferences
func (r *ResourceSliceWrapper) OwnerReferences(nodeName, nodeUID string) *ResourceSliceWrapper {
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
	r.Spec.NodeName = nodeName
	return r
}

// NodeSelector sets the value of ResourceSlice.Spec.NodeSelector
func (r *ResourceSliceWrapper) NodeSelector(nodeSelector *v1.NodeSelector) *ResourceSliceWrapper {
	r.Spec.NodeSelector = nodeSelector
	return r
}

// AllNodes sets the value of ResourceSlice.Spec.AllNodes
func (r *ResourceSliceWrapper) AllNodes(allNodes bool) *ResourceSliceWrapper {
	r.Spec.AllNodes = allNodes
	return r
}

// Devices sets the value of ResourceSlice.Spec.Devices
func (r *ResourceSliceWrapper) Devices(devices []resourceapi.Device) *ResourceSliceWrapper {
	r.Spec.Devices = devices
	return r
}

// generateExpectedResourceSlices generates the expected ResourceSlice objects for a given test case.
func generateExpectedResourceSlices(baseName, nodeName, nodeUID, driveName, poolName string, totalDevices, generation int, initialSlices int) []resourceapi.ResourceSlice {
	var slices []resourceapi.ResourceSlice
	maxDevices := 128
	resourceSliceCount := (totalDevices + maxDevices - 1) / maxDevices
	// Initialize the slices array to ensure that its length is large enough
	for i := 1; i <= initialSlices; i++ {
		slice := MakeResourceSlice().Name(fmt.Sprintf("%s-%d", baseName, i)).UID(fmt.Sprintf("%s-%d", baseName, i)).
			OwnerReferences(nodeName, string(nodeUID)).NodeName(nodeName).
			Driver(driveName).Devices([]resourceapi.Device{}).
			Pool(resourceapi.ResourcePool{Name: poolName, Generation: 1}).Obj()
		slices = append(slices, *slice)
	}

	// Add additional slices if needed
	if resourceSliceCount > initialSlices {
		for i := 0; i < resourceSliceCount-initialSlices; i++ {
			slice := MakeResourceSlice().Name(fmt.Sprintf(nodeName+"-"+driveName+"-%d", i)).GenerateName(nodeName+"-"+driveName+"-").
				OwnerReferences(nodeName, string(nodeUID)).NodeName(nodeName).
				Driver(driveName).Devices([]resourceapi.Device{}).
				Pool(resourceapi.ResourcePool{Name: poolName, Generation: 1}).Obj()
			slices = append(slices, *slice)
		}
	}
	// Generate a list of devices and ensure that the device names are consecutive
	currentDeviceIndex := 0
	for i := 0; i < resourceSliceCount; i++ {
		start := i * maxDevices
		end := start + maxDevices
		if end > totalDevices {
			end = totalDevices
		}
		devices := generateDevices(currentDeviceIndex, end-start)
		slices[i].Spec.Devices = devices
		slices[i].Spec.Pool.ResourceSliceCount = int64(resourceSliceCount)
		slices[i].Spec.Pool.Generation = int64(generation)
		currentDeviceIndex += end - start
	}
	return slices
}

// generateDevices generates a list of devices with consecutive names.
// The number of devices generated is determined by the total number of devices
// divided by the number of slices.
func generateDevices(startIndex int, count int) []resourceapi.Device {
	devices := make([]resourceapi.Device, count)
	for i := 0; i < count; i++ {
		// Generate a unique device name using the index
		deviceName := fmt.Sprintf("device-%06d", startIndex+i)
		devices[i] = resourceapi.Device{Name: deviceName}
	}
	return devices
}
