/*
Copyright 2020 The Kubernetes Authors.

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

package devicemanager

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"k8s.io/apimachinery/pkg/util/sets"
	pluginapi "k8s.io/kubelet/pkg/apis/deviceplugin/v1beta1"
	"k8s.io/kubernetes/pkg/kubelet/cm/devicemanager/checkpoint"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	"k8s.io/kubernetes/test/utils/ktesting"
)

func TestGetContainerDevices(t *testing.T) {
	podDevices := newPodDevices()
	resourceName1 := "domain1.com/resource1"
	podID := "pod1"
	contID := "con1"
	devices := checkpoint.DevicesPerNUMA{0: []string{"dev1"}, 1: []string{"dev1"}}

	podDevices.insert(podID, contID, resourceName1,
		devices,
		newContainerAllocateResponse(
			withDevices(map[string]string{"/dev/r1dev1": "/dev/r1dev1", "/dev/r1dev2": "/dev/r1dev2"}),
			withMounts(map[string]string{"/home/r1lib1": "/usr/r1lib1"}),
		),
	)

	resContDevices := podDevices.getContainerDevices(podID, contID)
	contDevices, ok := resContDevices[resourceName1]
	require.True(t, ok, "resource %q not present", resourceName1)

	for devID, plugInfo := range contDevices {
		nodes := plugInfo.GetTopology().GetNodes()
		require.Equal(t, len(nodes), len(devices), "Incorrect container devices: %v - %v (nodes %v)", devices, contDevices, nodes)

		for _, node := range plugInfo.GetTopology().GetNodes() {
			dev, ok := devices[node.ID]
			require.True(t, ok, "NUMA id %v doesn't exist in result", node.ID)
			require.Equal(t, devID, dev[0], "Can't find device %s in result", dev[0])
		}
	}
}

func TestPodDevicesReservationLifecycle(t *testing.T) {
	logger, _ := ktesting.NewTestContext(t)
	const (
		podUID                = "pod"
		containerName         = "container"
		committedContainer    = "committed-container"
		resourceName          = "example.com/resource"
		committedResourceName = "example.com/committed-resource"
		deviceID              = "dev0"
		committedDeviceID     = "dev1"
	)

	podDevices := newPodDevices()
	require.True(t, podDevices.reserve(podUID, containerName, resourceName))
	require.False(t, podDevices.reserve(podUID, containerName, resourceName), "a second allocation must not replace an in-flight reservation")
	require.True(t, podDevices.addDevicesToReservation(podUID, containerName, resourceName, sets.New[string](deviceID)))

	assert.False(t, podDevices.hasPod(podUID), "reservations must not be exposed as committed pod allocations")
	assert.Empty(t, podDevices.pods())
	assert.Nil(t, podDevices.containerDevices(podUID, containerName, resourceName))
	assert.Nil(t, podDevices.getContainerDevices(podUID, containerName))
	assert.Nil(t, podDevices.deviceRunContainerOptions(logger, podUID, containerName))
	assert.Empty(t, podDevices.toCheckpointData(logger))
	assert.Equal(t, sets.New[string](deviceID), podDevices.devices()[resourceName], "reservations must still make devices unavailable")

	// Garbage collection must remove the pod's committed allocations without
	// removing an in-flight reservation owned by that same pod.
	podDevices.insert(podUID, committedContainer, committedResourceName, constructDevices([]string{committedDeviceID}), newContainerAllocateResponse())
	assert.True(t, podDevices.hasPod(podUID))
	podsWithReservations := podDevices.delete([]string{podUID})
	assert.Equal(t, []string{podUID}, podsWithReservations)
	assert.False(t, podDevices.hasPod(podUID))
	assert.Nil(t, podDevices.containerDevices(podUID, committedContainer, committedResourceName))
	assert.Equal(t, sets.New[string](deviceID), podDevices.devices()[resourceName])

	response := newContainerAllocateResponse()
	require.True(t, podDevices.commitReservation(podUID, containerName, resourceName, constructDevices([]string{deviceID}), response))
	assert.True(t, podDevices.hasPod(podUID))
	assert.Equal(t, sets.New[string](deviceID), podDevices.containerDevices(podUID, containerName, resourceName))
	assert.False(t, podDevices.rollbackReservation(podUID, containerName, resourceName), "a committed allocation must not be rolled back as a reservation")
	assert.Empty(t, podDevices.delete([]string{podUID}), "committed-only deletion must not report reservations")
}

func TestPodDevicesRollbackReservation(t *testing.T) {
	const (
		podUID        = "pod"
		containerName = "container"
		resourceName  = "example.com/resource"
	)

	podDevices := newPodDevices()
	require.True(t, podDevices.reserve(podUID, containerName, resourceName))
	require.True(t, podDevices.addDevicesToReservation(podUID, containerName, resourceName, sets.New[string]("dev0")))
	require.True(t, podDevices.rollbackReservation(podUID, containerName, resourceName))
	assert.Empty(t, podDevices.devs)
	assert.Empty(t, podDevices.devices())
}

func TestPodDevices(t *testing.T) {
	const (
		podUID           = "pod"
		reservedPodUID   = "reserved-pod"
		container1       = "container1"
		container2       = "container2"
		resource1        = "example.com/resource1"
		resource2        = "example.com/resource2"
		committedDevice1 = "dev1"
		committedDevice2 = "dev2"
		committedDevice3 = "dev3"
		reservedDevice   = "reserved-dev"
	)

	podDevices := newPodDevices()
	podDevices.insert(podUID, container1, resource1,
		constructDevices([]string{committedDevice1}),
		newContainerAllocateResponse())
	podDevices.insert(podUID, container2, resource1,
		constructDevices([]string{committedDevice2}),
		newContainerAllocateResponse())
	podDevices.insert(podUID, container1, resource2,
		constructDevices([]string{committedDevice3}),
		newContainerAllocateResponse())

	require.True(t, podDevices.reserve(
		reservedPodUID, container1, resource1))
	require.True(t, podDevices.addDevicesToReservation(
		reservedPodUID, container1, resource1,
		sets.New[string](reservedDevice)))

	assert.Equal(t, sets.New[string](committedDevice1, committedDevice2),
		podDevices.podDevices(podUID, resource1))
	assert.Equal(t, sets.New[string](committedDevice3),
		podDevices.podDevices(podUID, resource2))
	assert.Empty(t, podDevices.podDevices(podUID, "example.com/unknown"))
	assert.Empty(t, podDevices.podDevices("unknown-pod", resource1))
	assert.Empty(t, podDevices.podDevices(reservedPodUID, resource1),
		"in-flight reservations must not be exposed as pod devices")
}

func TestResourceDeviceInstanceFilter(t *testing.T) {
	var expected string
	var cond map[string]sets.Set[string]
	var resp ResourceDeviceInstances
	devs := ResourceDeviceInstances{
		"foo": DeviceInstances{
			"dev-foo1": &pluginapi.Device{
				ID: "foo1",
			},
			"dev-foo2": &pluginapi.Device{
				ID: "foo2",
			},
			"dev-foo3": &pluginapi.Device{
				ID: "foo3",
			},
		},
		"bar": DeviceInstances{
			"dev-bar1": &pluginapi.Device{
				ID: "bar1",
			},
			"dev-bar2": &pluginapi.Device{
				ID: "bar2",
			},
			"dev-bar3": &pluginapi.Device{
				ID: "bar3",
			},
		},
		"baz": DeviceInstances{
			"dev-baz1": &pluginapi.Device{
				ID: "baz1",
			},
			"dev-baz2": &pluginapi.Device{
				ID: "baz2",
			},
			"dev-baz3": &pluginapi.Device{
				ID: "baz3",
			},
		},
	}

	resp = devs.Filter(map[string]sets.Set[string]{})
	expected = `{}`
	expectResourceDeviceInstances(t, resp, expected)

	cond = map[string]sets.Set[string]{
		"foo": sets.New[string]("dev-foo1", "dev-foo2"),
		"bar": sets.New[string]("dev-bar1"),
	}
	resp = devs.Filter(cond)
	expected = `{"bar":{"dev-bar1":{"ID":"bar1"}},"foo":{"dev-foo1":{"ID":"foo1"},"dev-foo2":{"ID":"foo2"}}}`
	expectResourceDeviceInstances(t, resp, expected)

	cond = map[string]sets.Set[string]{
		"foo": sets.New[string]("dev-foo1", "dev-foo2", "dev-foo3"),
		"bar": sets.New[string]("dev-bar1", "dev-bar2", "dev-bar3"),
		"baz": sets.New[string]("dev-baz1", "dev-baz2", "dev-baz3"),
	}
	resp = devs.Filter(cond)
	expected = `{"bar":{"dev-bar1":{"ID":"bar1"},"dev-bar2":{"ID":"bar2"},"dev-bar3":{"ID":"bar3"}},"baz":{"dev-baz1":{"ID":"baz1"},"dev-baz2":{"ID":"baz2"},"dev-baz3":{"ID":"baz3"}},"foo":{"dev-foo1":{"ID":"foo1"},"dev-foo2":{"ID":"foo2"},"dev-foo3":{"ID":"foo3"}}}`
	expectResourceDeviceInstances(t, resp, expected)

	cond = map[string]sets.Set[string]{
		"foo": sets.New[string]("dev-foo1", "dev-foo2", "dev-foo3", "dev-foo4"),
		"bar": sets.New[string]("dev-bar1", "dev-bar2", "dev-bar3", "dev-bar4"),
		"baz": sets.New[string]("dev-baz1", "dev-baz2", "dev-baz3", "dev-bar4"),
	}
	resp = devs.Filter(cond)
	expected = `{"bar":{"dev-bar1":{"ID":"bar1"},"dev-bar2":{"ID":"bar2"},"dev-bar3":{"ID":"bar3"}},"baz":{"dev-baz1":{"ID":"baz1"},"dev-baz2":{"ID":"baz2"},"dev-baz3":{"ID":"baz3"}},"foo":{"dev-foo1":{"ID":"foo1"},"dev-foo2":{"ID":"foo2"},"dev-foo3":{"ID":"foo3"}}}`
	expectResourceDeviceInstances(t, resp, expected)

	cond = map[string]sets.Set[string]{
		"foo": sets.New[string]("dev-foo1", "dev-foo4", "dev-foo7"),
		"bar": sets.New[string]("dev-bar1", "dev-bar4", "dev-bar7"),
		"baz": sets.New[string]("dev-baz1", "dev-baz4", "dev-baz7"),
	}
	resp = devs.Filter(cond)
	expected = `{"bar":{"dev-bar1":{"ID":"bar1"}},"baz":{"dev-baz1":{"ID":"baz1"}},"foo":{"dev-foo1":{"ID":"foo1"}}}`
	expectResourceDeviceInstances(t, resp, expected)

}

func expectResourceDeviceInstances(t *testing.T, resp ResourceDeviceInstances, expected string) {
	// per docs in https://pkg.go.dev/encoding/json#Marshal
	// "Map values encode as JSON objects. The map's key type must either be a string, an integer type, or
	// implement encoding.TextMarshaler. The map keys are sorted [...]"
	// so this check is expected to be stable and not flaky
	data, err := json.Marshal(resp)
	if err != nil {
		t.Fatalf("unexpected JSON marshalling error: %v", err)
	}
	got := string(data)
	if got != expected {
		t.Errorf("expected %q got %q", expected, got)
	}
}

func TestDeviceRunContainerOptions(t *testing.T) {
	logger, _ := ktesting.NewTestContext(t)
	const (
		podUID        = "pod"
		containerName = "container"
		resource1     = "example1.com/resource1"
		resource2     = "example2.com/resource2"
	)
	testCases := []struct {
		description          string
		responsesPerResource map[string]*pluginapi.ContainerAllocateResponse
		expected             *DeviceRunContainerOptions
	}{
		{
			description: "empty response",
			responsesPerResource: map[string]*pluginapi.ContainerAllocateResponse{
				resource1: newContainerAllocateResponse(),
			},
			expected: &DeviceRunContainerOptions{},
		},
		{
			description: "cdi devices are handled",
			responsesPerResource: map[string]*pluginapi.ContainerAllocateResponse{
				resource1: newContainerAllocateResponse(
					withCDIDevices("vendor1.com/class1=device1", "vendor2.com/class2=device2"),
				),
			},
			expected: &DeviceRunContainerOptions{
				CDIDevices: []kubecontainer.CDIDevice{
					{Name: "vendor1.com/class1=device1"},
					{Name: "vendor2.com/class2=device2"},
				},
			},
		},
		{
			description: "cdi devices from multiple resources are handled",
			responsesPerResource: map[string]*pluginapi.ContainerAllocateResponse{
				resource1: newContainerAllocateResponse(
					withCDIDevices("vendor1.com/class1=device1", "vendor2.com/class2=device2"),
				),
				resource2: newContainerAllocateResponse(
					withCDIDevices("vendor3.com/class3=device3", "vendor4.com/class4=device4"),
				),
			},
			expected: &DeviceRunContainerOptions{
				CDIDevices: []kubecontainer.CDIDevice{
					{Name: "vendor1.com/class1=device1"},
					{Name: "vendor2.com/class2=device2"},
					{Name: "vendor3.com/class3=device3"},
					{Name: "vendor4.com/class4=device4"},
				},
			},
		},
		{
			description: "duplicate cdi devices are skipped",
			responsesPerResource: map[string]*pluginapi.ContainerAllocateResponse{
				resource1: newContainerAllocateResponse(
					withCDIDevices("vendor1.com/class1=device1", "vendor2.com/class2=device2"),
				),
				resource2: newContainerAllocateResponse(
					withCDIDevices("vendor2.com/class2=device2", "vendor3.com/class3=device3"),
				),
			},
			expected: &DeviceRunContainerOptions{
				CDIDevices: []kubecontainer.CDIDevice{
					{Name: "vendor1.com/class1=device1"},
					{Name: "vendor2.com/class2=device2"},
					{Name: "vendor3.com/class3=device3"},
				},
			},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.description, func(t *testing.T) {
			as := assert.New(t)

			podDevices := newPodDevices()
			for resourceName, response := range tc.responsesPerResource {
				podDevices.insert("pod", "container", resourceName,
					nil,
					response,
				)
			}
			opts := podDevices.deviceRunContainerOptions(logger, podUID, containerName)

			// The exact ordering of the options depends on the order of the resources in the map.
			// We therefore use `ElementsMatch` instead of `Equal` on the member slices.
			as.ElementsMatch(tc.expected.Annotations, opts.Annotations)
			as.ElementsMatch(tc.expected.CDIDevices, opts.CDIDevices)
			as.ElementsMatch(tc.expected.Devices, opts.Devices)
			as.ElementsMatch(tc.expected.Envs, opts.Envs)
			as.ElementsMatch(tc.expected.Mounts, opts.Mounts)
		})
	}
}

func TestGetPodAndContainerForDevice(t *testing.T) {
	podDevices := newPodDevices()
	const (
		resource1     = "domain1.com/resource1"
		resource2     = "domain2.com/resource2"
		pod1          = "pod1"
		pod2          = "pod2"
		reservedPod   = "reserved-pod"
		container1    = "container1"
		container2    = "container2"
		deviceID      = "dev1"
		reservedDevID = "reserved-dev"
	)

	podDevices.insert(pod1, container1, resource1,
		constructDevices([]string{deviceID}),
		newContainerAllocateResponse())
	// Device IDs are unique only within a resource, so the same ID may be
	// committed to a different pod and container under another resource.
	podDevices.insert(pod2, container2, resource2,
		constructDevices([]string{deviceID}),
		newContainerAllocateResponse())
	require.True(t, podDevices.reserve(
		reservedPod, container1, resource1))
	require.True(t, podDevices.addDevicesToReservation(
		reservedPod, container1, resource1,
		sets.New[string](reservedDevID)))

	testCases := []struct {
		name              string
		resource          string
		device            string
		expectedPod       string
		expectedContainer string
	}{
		{
			name:              "committed device",
			resource:          resource1,
			device:            deviceID,
			expectedPod:       pod1,
			expectedContainer: container1,
		},
		{
			name:              "same device ID under another resource",
			resource:          resource2,
			device:            deviceID,
			expectedPod:       pod2,
			expectedContainer: container2,
		},
		{
			name:     "reserved device",
			resource: resource1,
			device:   reservedDevID,
		},
		{
			name:     "unknown device",
			resource: resource1,
			device:   "unknown-device",
		},
		{
			name:     "unknown resource",
			resource: "domain3.com/unknown",
			device:   deviceID,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			podUID, containerName := podDevices.getPodAndContainerForDevice(
				tc.resource, tc.device)
			assert.Equal(t, tc.expectedPod, podUID)
			assert.Equal(t, tc.expectedContainer, containerName)
		})
	}
}
