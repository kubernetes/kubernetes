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
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	v1 "k8s.io/api/core/v1"
	resourceapi "k8s.io/api/resource/v1alpha3"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/kubernetes/fake"
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
		resourceSlice1 = "resource-slice-1"
		resourceSlice2 = "resource-slice-2"
		resourceSlice3 = "resource-slice-3"
	)

	testCases := map[string]struct {
		nodeUID                types.UID
		initialObjects         []runtime.Object
		poolName               string
		inputDriverResources   *DriverResources
		expectedResourceSlices []resourceapi.ResourceSlice
	}{
		"single-resourceslice-existing": {
			nodeUID: nodeUID,
			initialObjects: []runtime.Object{
				newResourceSlice(resourceSlice1, node, driveName, poolName, 1),
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
				*newExpectResourceSlice(resourceSlice1, node, string(nodeUID), driveName, poolName, deviceName, false, 1),
			},
		},
		"single-resourceslice-existing-with-nodeUID-empty": {
			nodeUID: "",
			initialObjects: []runtime.Object{
				newNode(node, nodeUID),
				newResourceSlice(resourceSlice1, node, driveName, poolName, 1),
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
				*newExpectResourceSlice(resourceSlice1, node, string(nodeUID), driveName, poolName, deviceName, false, 1),
			},
		},
		"multiple-resourceslices-existing": {
			nodeUID: nodeUID,
			initialObjects: []runtime.Object{
				newResourceSlice(resourceSlice1, node, driveName, poolName, 1),
				newResourceSlice(resourceSlice2, node, driveName, poolName, 1),
				newResourceSlice(resourceSlice3, node, driveName, poolName, 1),
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
				*newExpectResourceSlice(resourceSlice1, node, string(nodeUID), driveName, poolName, deviceName, false, 1),
			},
		},
		"no-resourceslices-existing": {
			nodeUID:        nodeUID,
			initialObjects: []runtime.Object{},
			poolName:       poolName,
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
				*newExpectResourceSlice(node+"-"+driveName+"-", node, string(nodeUID), driveName, poolName, deviceName, true, 0),
			},
		},
		"single-resourceslice-existing-with-different-resource-pool-generation": {
			nodeUID: nodeUID,
			initialObjects: []runtime.Object{
				newResourceSlice(resourceSlice1, node, driveName, poolName, 2),
				newResourceSlice(resourceSlice2, node, driveName, poolName, 1),
				newResourceSlice(resourceSlice3, node, driveName, poolName, 1),
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
				*newExpectResourceSlice(resourceSlice1, node, string(nodeUID), driveName, poolName, deviceName, false, 3),
			},
		},
		"single-resourceslice-existing-with-mismatching-resource-pool-name": {
			nodeUID: nodeUID,
			initialObjects: []runtime.Object{
				newResourceSlice(resourceSlice1, node, driveName, poolName, 1),
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
			expectedResourceSlices: nil,
		},
	}

	for name, test := range testCases {
		t.Run(name, func(t *testing.T) {
			_, ctx := ktesting.NewTestContext(t)
			kubeClient := fake.NewSimpleClientset(test.initialObjects...)
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
			assert.Equal(t, test.expectedResourceSlices, resourceSlices.Items, "unexpected ResourceSlices")
		})
	}
}

func newNode(name string, uid types.UID) *v1.Node {
	return &v1.Node{
		ObjectMeta: metav1.ObjectMeta{
			Name: name,
			UID:  uid,
		},
	}
}

func newResourceSlice(name, nodeName, driveName, poolName string, poolGeneration int64) *resourceapi.ResourceSlice {
	return &resourceapi.ResourceSlice{
		ObjectMeta: metav1.ObjectMeta{
			Name: name,
			UID:  types.UID(name),
		},
		Spec: resourceapi.ResourceSliceSpec{
			NodeName: nodeName,
			Driver:   driveName,
			Pool: resourceapi.ResourcePool{
				Name:               poolName,
				ResourceSliceCount: 1,
				Generation:         poolGeneration,
			},
		},
	}
}

func newExpectResourceSlice(name, nodeName, nodeUID, driveName, poolName, deviceName string, generateName bool, poolGeneration int64) *resourceapi.ResourceSlice {
	resourceSlice := &resourceapi.ResourceSlice{
		ObjectMeta: metav1.ObjectMeta{
			OwnerReferences: []metav1.OwnerReference{
				{
					APIVersion: "v1",
					Kind:       "Node",
					Name:       nodeName,
					UID:        types.UID(nodeUID),
					Controller: ptr.To(true),
				},
			},
		},
		Spec: resourceapi.ResourceSliceSpec{
			NodeName: nodeName,
			Driver:   driveName,
			Pool: resourceapi.ResourcePool{
				Name:               poolName,
				ResourceSliceCount: 1,
				Generation:         poolGeneration,
			},
			Devices: []resourceapi.Device{{Name: deviceName}},
		},
	}

	if generateName {
		resourceSlice.ObjectMeta.GenerateName = name
	} else {
		resourceSlice.ObjectMeta.Name = name
		resourceSlice.ObjectMeta.UID = types.UID(name)
	}
	return resourceSlice
}
