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

package dra

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	resourceapi "k8s.io/api/resource/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes/fake"
	"k8s.io/utils/ptr"

	"k8s.io/kubernetes/test/utils/ktesting"
)

const testNodeName = "worker-a"

func TestGetTopologyHintsUsesAllocatedDeviceNUMA(t *testing.T) {
	tCtx := ktesting.Init(t)

	pod := genTestPod()
	pod.Spec.NodeName = testNodeName
	pod.Spec.Containers[0].Resources.Claims[0].Request = requestName

	claim := genTestClaim(claimName, driverName, deviceName, string(pod.UID))
	slice := makeTestResourceSlice(testNodeName, deviceName, map[resourceapi.QualifiedName]resourceapi.DeviceAttribute{
		resourceapi.QualifiedName("dra.net/numaNode"): {IntValue: ptr.To(int64(1))},
	})

	manager, err := NewManager(tCtx.Logger(), fake.NewSimpleClientset(claim, slice), t.TempDir())
	require.NoError(t, err)

	hints := manager.GetTopologyHints(pod, &pod.Spec.Containers[0])
	resourceName := topologyResourceName(claimName, requestName)

	require.Contains(t, hints, resourceName)
	require.Len(t, hints[resourceName], 1)
	assert.True(t, hints[resourceName][0].Preferred)
	assert.ElementsMatch(t, []int{1}, hints[resourceName][0].NUMANodeAffinity.GetBits())
}

func TestGetTopologyHintsSkipsTopologyWhenNUMAAttributeUnavailable(t *testing.T) {
	tCtx := ktesting.Init(t)

	pod := genTestPod()
	pod.Spec.NodeName = testNodeName

	claim := genTestClaim(claimName, driverName, deviceName, string(pod.UID))
	slice := makeTestResourceSlice(testNodeName, deviceName, map[resourceapi.QualifiedName]resourceapi.DeviceAttribute{
		resourceapi.QualifiedName("test-driver/model"): {StringValue: ptr.To("x")},
	})

	manager, err := NewManager(tCtx.Logger(), fake.NewSimpleClientset(claim, slice), t.TempDir())
	require.NoError(t, err)

	hints := manager.GetTopologyHints(pod, &pod.Spec.Containers[0])
	resourceName := topologyResourceName(claimName, "")

	require.Contains(t, hints, resourceName)
	assert.Nil(t, hints[resourceName])
}

func TestGetTopologyHintsFailsClosedWhenAllocatedDeviceCannotBeResolved(t *testing.T) {
	tCtx := ktesting.Init(t)

	pod := genTestPod()
	pod.Spec.NodeName = testNodeName

	claim := genTestClaim(claimName, driverName, deviceName, string(pod.UID))
	slice := makeTestResourceSlice(testNodeName, "different-device", map[resourceapi.QualifiedName]resourceapi.DeviceAttribute{
		resourceapi.QualifiedName("dra.net/numaNode"): {IntValue: ptr.To(int64(0))},
	})

	manager, err := NewManager(tCtx.Logger(), fake.NewSimpleClientset(claim, slice), t.TempDir())
	require.NoError(t, err)

	hints := manager.GetTopologyHints(pod, &pod.Spec.Containers[0])
	resourceName := topologyResourceName(claimName, "")

	require.Contains(t, hints, resourceName)
	assert.NotNil(t, hints[resourceName])
	assert.Empty(t, hints[resourceName])
}

func makeTestResourceSlice(nodeName, device string, attributes map[resourceapi.QualifiedName]resourceapi.DeviceAttribute) *resourceapi.ResourceSlice {
	return &resourceapi.ResourceSlice{
		ObjectMeta: metav1.ObjectMeta{
			Name: "slice-" + device,
		},
		Spec: resourceapi.ResourceSliceSpec{
			Driver:   driverName,
			NodeName: ptr.To(nodeName),
			Pool: resourceapi.ResourcePool{
				Name:               poolName,
				Generation:         1,
				ResourceSliceCount: 1,
			},
			Devices: []resourceapi.Device{{
				Name:       device,
				Attributes: attributes,
			}},
		},
	}
}
