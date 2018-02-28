/*
Copyright 2017 The Kubernetes Authors.

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
	"flag"
	"fmt"
	"io/ioutil"
	"os"
	"reflect"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/uuid"
	pluginapi "k8s.io/kubernetes/pkg/kubelet/apis/deviceplugin/v1beta1"
	"k8s.io/kubernetes/pkg/kubelet/lifecycle"
	utilstore "k8s.io/kubernetes/pkg/kubelet/util/store"
	"k8s.io/kubernetes/pkg/scheduler/schedulercache"
	utilfs "k8s.io/kubernetes/pkg/util/filesystem"
)

const (
	testResourceName = "fake-domain/resource"
)

func tmpSocketDir() (socketDir, socketName, pluginSocketName string, err error) {
	socketDir, err = ioutil.TempDir("", "device_plugin")
	if err != nil {
		return
	}
	socketName = socketDir + "/server.sock"
	pluginSocketName = socketDir + "/device-plugin.sock"
	return
}

func TestNewManagerImpl(t *testing.T) {
	socketDir, socketName, _, err := tmpSocketDir()
	require.NoError(t, err)
	defer os.RemoveAll(socketDir)
	_, err = newManagerImpl(socketName)
	require.NoError(t, err)
	os.RemoveAll(socketDir)
}

func TestNewManagerImplStart(t *testing.T) {
	socketDir, socketName, pluginSocketName, err := tmpSocketDir()
	require.NoError(t, err)
	defer os.RemoveAll(socketDir)
	m, p := setup(t, []*pluginapi.Device{}, func(n string, a, u, r []pluginapi.Device) {}, socketName, pluginSocketName)
	cleanup(t, m, p)
}

// Tests that the device plugin manager correctly handles registration and re-registration by
// making sure that after registration, devices are correctly updated and if a re-registration
// happens, we will NOT delete devices; and no orphaned devices left.
func TestDevicePluginReRegistration(t *testing.T) {
	socketDir, socketName, pluginSocketName, err := tmpSocketDir()
	require.NoError(t, err)
	defer os.RemoveAll(socketDir)
	devs := []*pluginapi.Device{
		{ID: "Dev1", Health: pluginapi.Healthy},
		{ID: "Dev2", Health: pluginapi.Healthy},
	}
	devsForRegistration := []*pluginapi.Device{
		{ID: "Dev3", Health: pluginapi.Healthy},
	}
	for _, preStartContainerFlag := range []bool{false, true} {

		expCallbackCount := int32(0)
		callbackCount := int32(0)
		callbackChan := make(chan int32)
		callback := func(n string, a, u, r []pluginapi.Device) {
			callbackCount++
			if callbackCount > atomic.LoadInt32(&expCallbackCount) {
				t.FailNow()
			}
			callbackChan <- callbackCount
		}
		m, p1 := setup(t, devs, callback, socketName, pluginSocketName)
		atomic.StoreInt32(&expCallbackCount, 1)
		p1.Register(socketName, testResourceName, preStartContainerFlag)
		// Wait for the first callback to be issued.

		select {
		case <-callbackChan:
			break
		case <-time.After(time.Second):
			t.FailNow()
		}
		devices := m.Devices()
		require.Equal(t, 2, len(devices[testResourceName]), "Devices are not updated.")

		p2 := NewDevicePluginStub(devs, pluginSocketName+".new")
		err = p2.Start()
		require.NoError(t, err)
		atomic.StoreInt32(&expCallbackCount, 2)
		p2.Register(socketName, testResourceName, preStartContainerFlag)
		// Wait for the second callback to be issued.
		select {
		case <-callbackChan:
			break
		case <-time.After(time.Second):
			t.FailNow()
		}

		devices2 := m.Devices()
		require.Equal(t, 2, len(devices2[testResourceName]), "Devices shouldn't change.")

		// Test the scenario that a plugin re-registers with different devices.
		p3 := NewDevicePluginStub(devsForRegistration, pluginSocketName+".third")
		err = p3.Start()
		require.NoError(t, err)
		atomic.StoreInt32(&expCallbackCount, 3)
		p3.Register(socketName, testResourceName, preStartContainerFlag)
		// Wait for the second callback to be issued.
		select {
		case <-callbackChan:
			break
		case <-time.After(time.Second):
			t.FailNow()
		}
		devices3 := m.Devices()
		require.Equal(t, 1, len(devices3[testResourceName]), "Devices of plugin previously registered should be removed.")
		p2.Stop()
		p3.Stop()
		cleanup(t, m, p1)
		close(callbackChan)
	}
}

func setup(t *testing.T, devs []*pluginapi.Device, callback monitorCallback, socketName string, pluginSocketName string) (Manager, *Stub) {
	m, err := newManagerImpl(socketName)
	require.NoError(t, err)

	m.callback = callback

	activePods := func() []*v1.Pod {
		return []*v1.Pod{}
	}
	err = m.Start(activePods, &sourcesReadyStub{})
	require.NoError(t, err)

	p := NewDevicePluginStub(devs, pluginSocketName)
	err = p.Start()
	require.NoError(t, err)

	return m, p
}

func cleanup(t *testing.T, m Manager, p *Stub) {
	p.Stop()
	m.Stop()
}

func TestUpdateCapacityAllocatable(t *testing.T) {
	socketDir, socketName, _, err := tmpSocketDir()
	require.NoError(t, err)
	defer os.RemoveAll(socketDir)
	testManager, err := newManagerImpl(socketName)
	as := assert.New(t)
	as.NotNil(testManager)
	as.Nil(err)

	devs := []pluginapi.Device{
		{ID: "Device1", Health: pluginapi.Healthy},
		{ID: "Device2", Health: pluginapi.Healthy},
		{ID: "Device3", Health: pluginapi.Unhealthy},
	}
	callback := testManager.genericDeviceUpdateCallback

	// Adds three devices for resource1, two healthy and one unhealthy.
	// Expects capacity for resource1 to be 2.
	resourceName1 := "domain1.com/resource1"
	testManager.endpoints[resourceName1] = &endpointImpl{devices: make(map[string]pluginapi.Device)}
	callback(resourceName1, devs, []pluginapi.Device{}, []pluginapi.Device{})
	capacity, allocatable, removedResources := testManager.GetCapacity()
	resource1Capacity, ok := capacity[v1.ResourceName(resourceName1)]
	as.True(ok)
	resource1Allocatable, ok := allocatable[v1.ResourceName(resourceName1)]
	as.True(ok)
	as.Equal(int64(3), resource1Capacity.Value())
	as.Equal(int64(2), resource1Allocatable.Value())
	as.Equal(0, len(removedResources))

	// Deletes an unhealthy device should NOT change allocatable but change capacity.
	callback(resourceName1, []pluginapi.Device{}, []pluginapi.Device{}, []pluginapi.Device{devs[2]})
	capacity, allocatable, removedResources = testManager.GetCapacity()
	resource1Capacity, ok = capacity[v1.ResourceName(resourceName1)]
	as.True(ok)
	resource1Allocatable, ok = allocatable[v1.ResourceName(resourceName1)]
	as.True(ok)
	as.Equal(int64(2), resource1Capacity.Value())
	as.Equal(int64(2), resource1Allocatable.Value())
	as.Equal(0, len(removedResources))

	// Updates a healthy device to unhealthy should reduce allocatable by 1.
	dev2 := devs[1]
	dev2.Health = pluginapi.Unhealthy
	callback(resourceName1, []pluginapi.Device{}, []pluginapi.Device{dev2}, []pluginapi.Device{})
	capacity, allocatable, removedResources = testManager.GetCapacity()
	resource1Capacity, ok = capacity[v1.ResourceName(resourceName1)]
	as.True(ok)
	resource1Allocatable, ok = allocatable[v1.ResourceName(resourceName1)]
	as.True(ok)
	as.Equal(int64(2), resource1Capacity.Value())
	as.Equal(int64(1), resource1Allocatable.Value())
	as.Equal(0, len(removedResources))

	// Deletes a healthy device should reduce capacity and allocatable by 1.
	callback(resourceName1, []pluginapi.Device{}, []pluginapi.Device{}, []pluginapi.Device{devs[0]})
	capacity, allocatable, removedResources = testManager.GetCapacity()
	resource1Capacity, ok = capacity[v1.ResourceName(resourceName1)]
	as.True(ok)
	resource1Allocatable, ok = allocatable[v1.ResourceName(resourceName1)]
	as.True(ok)
	as.Equal(int64(0), resource1Allocatable.Value())
	as.Equal(int64(1), resource1Capacity.Value())
	as.Equal(0, len(removedResources))

	// Tests adding another resource.
	resourceName2 := "resource2"
	testManager.endpoints[resourceName2] = &endpointImpl{devices: make(map[string]pluginapi.Device)}
	callback(resourceName2, devs, []pluginapi.Device{}, []pluginapi.Device{})
	capacity, allocatable, removedResources = testManager.GetCapacity()
	as.Equal(2, len(capacity))
	resource2Capacity, ok := capacity[v1.ResourceName(resourceName2)]
	as.True(ok)
	resource2Allocatable, ok := allocatable[v1.ResourceName(resourceName2)]
	as.True(ok)
	as.Equal(int64(3), resource2Capacity.Value())
	as.Equal(int64(2), resource2Allocatable.Value())
	as.Equal(0, len(removedResources))

	// Removes resourceName1 endpoint. Verifies testManager.GetCapacity() reports that resourceName1
	// is removed from capacity and it no longer exists in healthyDevices after the call.
	delete(testManager.endpoints, resourceName1)
	capacity, allocatable, removed := testManager.GetCapacity()
	as.Equal([]string{resourceName1}, removed)
	_, ok = capacity[v1.ResourceName(resourceName1)]
	as.False(ok)
	val, ok := capacity[v1.ResourceName(resourceName2)]
	as.True(ok)
	as.Equal(int64(3), val.Value())
	_, ok = testManager.healthyDevices[resourceName1]
	as.False(ok)
	_, ok = testManager.unhealthyDevices[resourceName1]
	as.False(ok)
	fmt.Println("removed: ", removed)
	as.Equal(1, len(removed))

}

func constructDevices(devices []string) sets.String {
	ret := sets.NewString()
	for _, dev := range devices {
		ret.Insert(dev)
	}
	return ret
}

func constructAllocResp(devices, mounts, envs map[string]string) *pluginapi.ContainerAllocateResponse {
	resp := &pluginapi.ContainerAllocateResponse{}
	for k, v := range devices {
		resp.Devices = append(resp.Devices, &pluginapi.DeviceSpec{
			HostPath:      k,
			ContainerPath: v,
			Permissions:   "mrw",
		})
	}
	for k, v := range mounts {
		resp.Mounts = append(resp.Mounts, &pluginapi.Mount{
			ContainerPath: k,
			HostPath:      v,
			ReadOnly:      true,
		})
	}
	resp.Envs = make(map[string]string)
	for k, v := range envs {
		resp.Envs[k] = v
	}
	return resp
}

func TestCheckpoint(t *testing.T) {
	resourceName1 := "domain1.com/resource1"
	resourceName2 := "domain2.com/resource2"

	as := assert.New(t)
	tmpDir, err := ioutil.TempDir("", "checkpoint")
	as.Nil(err)
	defer os.RemoveAll(tmpDir)
	testManager := &ManagerImpl{
		socketdir:        tmpDir,
		healthyDevices:   make(map[string]sets.String),
		allocatedDevices: make(map[string]sets.String),
		podDevices:       make(podDevices),
	}
	testManager.store, _ = utilstore.NewFileStore("/tmp/", utilfs.DefaultFs{})

	testManager.podDevices.insert("pod1", "con1", resourceName1,
		constructDevices([]string{"dev1", "dev2"}),
		constructAllocResp(map[string]string{"/dev/r1dev1": "/dev/r1dev1", "/dev/r1dev2": "/dev/r1dev2"},
			map[string]string{"/home/r1lib1": "/usr/r1lib1"}, map[string]string{}))
	testManager.podDevices.insert("pod1", "con1", resourceName2,
		constructDevices([]string{"dev1", "dev2"}),
		constructAllocResp(map[string]string{"/dev/r2dev1": "/dev/r2dev1", "/dev/r2dev2": "/dev/r2dev2"},
			map[string]string{"/home/r2lib1": "/usr/r2lib1"},
			map[string]string{"r2devices": "dev1 dev2"}))
	testManager.podDevices.insert("pod1", "con2", resourceName1,
		constructDevices([]string{"dev3"}),
		constructAllocResp(map[string]string{"/dev/r1dev3": "/dev/r1dev3"},
			map[string]string{"/home/r1lib1": "/usr/r1lib1"}, map[string]string{}))
	testManager.podDevices.insert("pod2", "con1", resourceName1,
		constructDevices([]string{"dev4"}),
		constructAllocResp(map[string]string{"/dev/r1dev4": "/dev/r1dev4"},
			map[string]string{"/home/r1lib1": "/usr/r1lib1"}, map[string]string{}))

	testManager.healthyDevices[resourceName1] = sets.NewString()
	testManager.healthyDevices[resourceName1].Insert("dev1")
	testManager.healthyDevices[resourceName1].Insert("dev2")
	testManager.healthyDevices[resourceName1].Insert("dev3")
	testManager.healthyDevices[resourceName1].Insert("dev4")
	testManager.healthyDevices[resourceName1].Insert("dev5")
	testManager.healthyDevices[resourceName2] = sets.NewString()
	testManager.healthyDevices[resourceName2].Insert("dev1")
	testManager.healthyDevices[resourceName2].Insert("dev2")

	expectedPodDevices := testManager.podDevices
	expectedAllocatedDevices := testManager.podDevices.devices()
	expectedAllDevices := testManager.healthyDevices

	err = testManager.writeCheckpoint()

	as.Nil(err)
	testManager.podDevices = make(podDevices)
	err = testManager.readCheckpoint()
	as.Nil(err)

	as.Equal(len(expectedPodDevices), len(testManager.podDevices))
	for podUID, containerDevices := range expectedPodDevices {
		for conName, resources := range containerDevices {
			for resource := range resources {
				expDevices := expectedPodDevices.containerDevices(podUID, conName, resource)
				testDevices := testManager.podDevices.containerDevices(podUID, conName, resource)
				as.True(reflect.DeepEqual(expDevices, testDevices))
				opts1 := expectedPodDevices.deviceRunContainerOptions(podUID, conName)
				opts2 := testManager.podDevices.deviceRunContainerOptions(podUID, conName)
				as.Equal(len(opts1.Envs), len(opts2.Envs))
				as.Equal(len(opts1.Mounts), len(opts2.Mounts))
				as.Equal(len(opts1.Devices), len(opts2.Devices))
			}
		}
	}
	as.True(reflect.DeepEqual(expectedAllocatedDevices, testManager.allocatedDevices))
	as.True(reflect.DeepEqual(expectedAllDevices, testManager.healthyDevices))
}

type activePodsStub struct {
	activePods []*v1.Pod
}

func (a *activePodsStub) getActivePods() []*v1.Pod {
	return a.activePods
}

func (a *activePodsStub) updateActivePods(newPods []*v1.Pod) {
	a.activePods = newPods
}

type MockEndpoint struct {
	allocateFunc func(devs []string) (*pluginapi.AllocateResponse, error)
	initChan     chan []string
}

func (m *MockEndpoint) stop() {}
func (m *MockEndpoint) run()  {}

func (m *MockEndpoint) getDevices() []pluginapi.Device {
	return []pluginapi.Device{}
}

func (m *MockEndpoint) callback(resourceName string, added, updated, deleted []pluginapi.Device) {}

func (m *MockEndpoint) preStartContainer(devs []string) (*pluginapi.PreStartContainerResponse, error) {
	m.initChan <- devs
	return &pluginapi.PreStartContainerResponse{}, nil
}

func (m *MockEndpoint) allocate(devs []string) (*pluginapi.AllocateResponse, error) {
	if m.allocateFunc != nil {
		return m.allocateFunc(devs)
	}
	return nil, nil
}

func makePod(limits v1.ResourceList) *v1.Pod {
	return &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			UID: uuid.NewUUID(),
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Resources: v1.ResourceRequirements{
						Limits: limits,
					},
				},
			},
		},
	}
}

func getTestManager(tmpDir string, activePods ActivePodsFunc, testRes []TestResource, opts map[string]*pluginapi.DevicePluginOptions) *ManagerImpl {
	monitorCallback := func(resourceName string, added, updated, deleted []pluginapi.Device) {}
	testManager := &ManagerImpl{
		socketdir:        tmpDir,
		callback:         monitorCallback,
		healthyDevices:   make(map[string]sets.String),
		allocatedDevices: make(map[string]sets.String),
		endpoints:        make(map[string]endpoint),
		pluginOpts:       opts,
		podDevices:       make(podDevices),
		activePods:       activePods,
		sourcesReady:     &sourcesReadyStub{},
	}
	testManager.store, _ = utilstore.NewFileStore("/tmp/", utilfs.DefaultFs{})
	for _, res := range testRes {
		testManager.healthyDevices[res.resourceName] = sets.NewString()
		for _, dev := range res.devs {
			testManager.healthyDevices[res.resourceName].Insert(dev)
		}
		if res.resourceName == "domain1.com/resource1" {
			testManager.endpoints[res.resourceName] = &MockEndpoint{
				allocateFunc: allocateStubFunc(),
			}
		}
		if res.resourceName == "domain2.com/resource2" {
			testManager.endpoints[res.resourceName] = &MockEndpoint{
				allocateFunc: func(devs []string) (*pluginapi.AllocateResponse, error) {
					resp := new(pluginapi.ContainerAllocateResponse)
					resp.Envs = make(map[string]string)
					for _, dev := range devs {
						switch dev {
						case "dev3":
							resp.Envs["key2"] = "val2"

						case "dev4":
							resp.Envs["key2"] = "val3"
						}
					}
					resps := new(pluginapi.AllocateResponse)
					resps.ContainerResponses = append(resps.ContainerResponses, resp)
					return resps, nil
				},
			}
		}
	}
	return testManager
}

func getTestNodeInfo(allocatable v1.ResourceList) *schedulercache.NodeInfo {
	cachedNode := &v1.Node{
		Status: v1.NodeStatus{
			Allocatable: allocatable,
		},
	}
	nodeInfo := &schedulercache.NodeInfo{}
	nodeInfo.SetNode(cachedNode)
	return nodeInfo
}

type TestResource struct {
	resourceName     string
	resourceQuantity resource.Quantity
	devs             []string
}

func TestPodContainerDeviceAllocation(t *testing.T) {
	flag.Set("alsologtostderr", fmt.Sprintf("%t", true))
	res1 := TestResource{
		resourceName:     "domain1.com/resource1",
		resourceQuantity: *resource.NewQuantity(int64(2), resource.DecimalSI),
		devs:             []string{"dev1", "dev2"},
	}
	res2 := TestResource{
		resourceName:     "domain2.com/resource2",
		resourceQuantity: *resource.NewQuantity(int64(1), resource.DecimalSI),
		devs:             []string{"dev3", "dev4"},
	}
	testResources := make([]TestResource, 2)
	testResources = append(testResources, res1)
	testResources = append(testResources, res2)
	as := require.New(t)
	podsStub := activePodsStub{
		activePods: []*v1.Pod{},
	}
	tmpDir, err := ioutil.TempDir("", "checkpoint")
	as.Nil(err)
	defer os.RemoveAll(tmpDir)
	nodeInfo := getTestNodeInfo(v1.ResourceList{})
	pluginOpts := make(map[string]*pluginapi.DevicePluginOptions)
	testManager := getTestManager(tmpDir, podsStub.getActivePods, testResources, pluginOpts)

	testPods := []*v1.Pod{
		makePod(v1.ResourceList{
			v1.ResourceName(res1.resourceName): res1.resourceQuantity,
			v1.ResourceName("cpu"):             res1.resourceQuantity,
			v1.ResourceName(res2.resourceName): res2.resourceQuantity}),
		makePod(v1.ResourceList{
			v1.ResourceName(res1.resourceName): res2.resourceQuantity}),
		makePod(v1.ResourceList{
			v1.ResourceName(res2.resourceName): res2.resourceQuantity}),
	}
	testCases := []struct {
		description               string
		testPod                   *v1.Pod
		expectedContainerOptsLen  []int
		expectedAllocatedResName1 int
		expectedAllocatedResName2 int
		expErr                    error
	}{
		{
			description:               "Successful allocation of two Res1 resources and one Res2 resource",
			testPod:                   testPods[0],
			expectedContainerOptsLen:  []int{3, 2, 2},
			expectedAllocatedResName1: 2,
			expectedAllocatedResName2: 1,
			expErr: nil,
		},
		{
			description:               "Requesting to create a pod without enough resources should fail",
			testPod:                   testPods[1],
			expectedContainerOptsLen:  nil,
			expectedAllocatedResName1: 2,
			expectedAllocatedResName2: 1,
			expErr: fmt.Errorf("requested number of devices unavailable for domain1.com/resource1. Requested: 1, Available: 0"),
		},
		{
			description:               "Successful allocation of all available Res1 resources and Res2 resources",
			testPod:                   testPods[2],
			expectedContainerOptsLen:  []int{0, 0, 1},
			expectedAllocatedResName1: 2,
			expectedAllocatedResName2: 2,
			expErr: nil,
		},
	}
	activePods := []*v1.Pod{}
	for _, testCase := range testCases {
		pod := testCase.testPod
		activePods = append(activePods, pod)
		podsStub.updateActivePods(activePods)
		err := testManager.Allocate(nodeInfo, &lifecycle.PodAdmitAttributes{Pod: pod})
		if !reflect.DeepEqual(err, testCase.expErr) {
			t.Errorf("DevicePluginManager error (%v). expected error: %v but got: %v",
				testCase.description, testCase.expErr, err)
		}
		runContainerOpts, err := testManager.GetDeviceRunContainerOptions(pod, &pod.Spec.Containers[0])
		as.Nil(err)
		if testCase.expectedContainerOptsLen == nil {
			as.Nil(runContainerOpts)
		} else {
			as.Equal(len(runContainerOpts.Devices), testCase.expectedContainerOptsLen[0])
			as.Equal(len(runContainerOpts.Mounts), testCase.expectedContainerOptsLen[1])
			as.Equal(len(runContainerOpts.Envs), testCase.expectedContainerOptsLen[2])
		}
		as.Equal(testCase.expectedAllocatedResName1, testManager.allocatedDevices[res1.resourceName].Len())
		as.Equal(testCase.expectedAllocatedResName2, testManager.allocatedDevices[res2.resourceName].Len())
	}

}

func TestInitContainerDeviceAllocation(t *testing.T) {
	// Requesting to create a pod that requests resourceName1 in init containers and normal containers
	// should succeed with devices allocated to init containers reallocated to normal containers.
	res1 := TestResource{
		resourceName:     "domain1.com/resource1",
		resourceQuantity: *resource.NewQuantity(int64(2), resource.DecimalSI),
		devs:             []string{"dev1", "dev2"},
	}
	res2 := TestResource{
		resourceName:     "domain2.com/resource2",
		resourceQuantity: *resource.NewQuantity(int64(1), resource.DecimalSI),
		devs:             []string{"dev3", "dev4"},
	}
	testResources := make([]TestResource, 2)
	testResources = append(testResources, res1)
	testResources = append(testResources, res2)
	as := require.New(t)
	podsStub := activePodsStub{
		activePods: []*v1.Pod{},
	}
	nodeInfo := getTestNodeInfo(v1.ResourceList{})
	tmpDir, err := ioutil.TempDir("", "checkpoint")
	as.Nil(err)
	defer os.RemoveAll(tmpDir)
	pluginOpts := make(map[string]*pluginapi.DevicePluginOptions)
	testManager := getTestManager(tmpDir, podsStub.getActivePods, testResources, pluginOpts)

	podWithPluginResourcesInInitContainers := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			UID: uuid.NewUUID(),
		},
		Spec: v1.PodSpec{
			InitContainers: []v1.Container{
				{
					Name: string(uuid.NewUUID()),
					Resources: v1.ResourceRequirements{
						Limits: v1.ResourceList{
							v1.ResourceName(res1.resourceName): res2.resourceQuantity,
						},
					},
				},
				{
					Name: string(uuid.NewUUID()),
					Resources: v1.ResourceRequirements{
						Limits: v1.ResourceList{
							v1.ResourceName(res1.resourceName): res1.resourceQuantity,
						},
					},
				},
			},
			Containers: []v1.Container{
				{
					Name: string(uuid.NewUUID()),
					Resources: v1.ResourceRequirements{
						Limits: v1.ResourceList{
							v1.ResourceName(res1.resourceName): res2.resourceQuantity,
							v1.ResourceName(res2.resourceName): res2.resourceQuantity,
						},
					},
				},
				{
					Name: string(uuid.NewUUID()),
					Resources: v1.ResourceRequirements{
						Limits: v1.ResourceList{
							v1.ResourceName(res1.resourceName): res2.resourceQuantity,
							v1.ResourceName(res2.resourceName): res2.resourceQuantity,
						},
					},
				},
			},
		},
	}
	podsStub.updateActivePods([]*v1.Pod{podWithPluginResourcesInInitContainers})
	err = testManager.Allocate(nodeInfo, &lifecycle.PodAdmitAttributes{Pod: podWithPluginResourcesInInitContainers})
	as.Nil(err)
	podUID := string(podWithPluginResourcesInInitContainers.UID)
	initCont1 := podWithPluginResourcesInInitContainers.Spec.InitContainers[0].Name
	initCont2 := podWithPluginResourcesInInitContainers.Spec.InitContainers[1].Name
	normalCont1 := podWithPluginResourcesInInitContainers.Spec.Containers[0].Name
	normalCont2 := podWithPluginResourcesInInitContainers.Spec.Containers[1].Name
	initCont1Devices := testManager.podDevices.containerDevices(podUID, initCont1, res1.resourceName)
	initCont2Devices := testManager.podDevices.containerDevices(podUID, initCont2, res1.resourceName)
	normalCont1Devices := testManager.podDevices.containerDevices(podUID, normalCont1, res1.resourceName)
	normalCont2Devices := testManager.podDevices.containerDevices(podUID, normalCont2, res1.resourceName)
	as.Equal(1, initCont1Devices.Len())
	as.Equal(2, initCont2Devices.Len())
	as.Equal(1, normalCont1Devices.Len())
	as.Equal(1, normalCont2Devices.Len())
	as.True(initCont2Devices.IsSuperset(initCont1Devices))
	as.True(initCont2Devices.IsSuperset(normalCont1Devices))
	as.True(initCont2Devices.IsSuperset(normalCont2Devices))
	as.Equal(0, normalCont1Devices.Intersection(normalCont2Devices).Len())
}

func TestSanitizeNodeAllocatable(t *testing.T) {
	resourceName1 := "domain1.com/resource1"
	devID1 := "dev1"

	resourceName2 := "domain2.com/resource2"
	devID2 := "dev2"

	as := assert.New(t)
	monitorCallback := func(resourceName string, added, updated, deleted []pluginapi.Device) {}

	testManager := &ManagerImpl{
		callback:         monitorCallback,
		healthyDevices:   make(map[string]sets.String),
		allocatedDevices: make(map[string]sets.String),
		podDevices:       make(podDevices),
	}
	testManager.store, _ = utilstore.NewFileStore("/tmp/", utilfs.DefaultFs{})
	// require one of resource1 and one of resource2
	testManager.allocatedDevices[resourceName1] = sets.NewString()
	testManager.allocatedDevices[resourceName1].Insert(devID1)
	testManager.allocatedDevices[resourceName2] = sets.NewString()
	testManager.allocatedDevices[resourceName2].Insert(devID2)

	cachedNode := &v1.Node{
		Status: v1.NodeStatus{
			Allocatable: v1.ResourceList{
				// has no resource1 and two of resource2
				v1.ResourceName(resourceName2): *resource.NewQuantity(int64(2), resource.DecimalSI),
			},
		},
	}
	nodeInfo := &schedulercache.NodeInfo{}
	nodeInfo.SetNode(cachedNode)

	testManager.sanitizeNodeAllocatable(nodeInfo)

	allocatableScalarResources := nodeInfo.AllocatableResource().ScalarResources
	// allocatable in nodeInfo is less than needed, should update
	as.Equal(1, int(allocatableScalarResources[v1.ResourceName(resourceName1)]))
	// allocatable in nodeInfo is more than needed, should skip updating
	as.Equal(2, int(allocatableScalarResources[v1.ResourceName(resourceName2)]))
}

func TestDevicePreStartContainer(t *testing.T) {
	// Ensures that if device manager is indicated to invoke `PreStartContainer` RPC
	// by device plugin, then device manager invokes PreStartContainer at endpoint interface.
	// Also verifies that final allocation of mounts, envs etc is same as expected.
	res1 := TestResource{
		resourceName:     "domain1.com/resource1",
		resourceQuantity: *resource.NewQuantity(int64(2), resource.DecimalSI),
		devs:             []string{"dev1", "dev2"},
	}
	as := require.New(t)
	podsStub := activePodsStub{
		activePods: []*v1.Pod{},
	}
	tmpDir, err := ioutil.TempDir("", "checkpoint")
	as.Nil(err)
	defer os.RemoveAll(tmpDir)
	nodeInfo := getTestNodeInfo(v1.ResourceList{})
	pluginOpts := make(map[string]*pluginapi.DevicePluginOptions)
	pluginOpts[res1.resourceName] = &pluginapi.DevicePluginOptions{PreStartRequired: true}

	testManager := getTestManager(tmpDir, podsStub.getActivePods, []TestResource{res1}, pluginOpts)

	ch := make(chan []string, 1)
	testManager.endpoints[res1.resourceName] = &MockEndpoint{
		initChan:     ch,
		allocateFunc: allocateStubFunc(),
	}

	pod := makePod(v1.ResourceList{
		v1.ResourceName(res1.resourceName): res1.resourceQuantity})
	activePods := []*v1.Pod{}
	activePods = append(activePods, pod)
	podsStub.updateActivePods(activePods)
	err = testManager.Allocate(nodeInfo, &lifecycle.PodAdmitAttributes{Pod: pod})
	as.Nil(err)
	runContainerOpts, err := testManager.GetDeviceRunContainerOptions(pod, &pod.Spec.Containers[0])
	as.Nil(err)
	var initializedDevs []string
	select {
	case <-time.After(time.Second):
		t.Fatalf("Timed out while waiting on channel for response from PreStartContainer RPC stub")
	case initializedDevs = <-ch:
		break
	}

	as.Contains(initializedDevs, "dev1")
	as.Contains(initializedDevs, "dev2")
	as.Equal(len(initializedDevs), len(res1.devs))

	expectedResps, err := allocateStubFunc()([]string{"dev1", "dev2"})
	as.Nil(err)
	as.Equal(1, len(expectedResps.ContainerResponses))
	expectedResp := expectedResps.ContainerResponses[0]
	as.Equal(len(runContainerOpts.Devices), len(expectedResp.Devices))
	as.Equal(len(runContainerOpts.Mounts), len(expectedResp.Mounts))
	as.Equal(len(runContainerOpts.Envs), len(expectedResp.Envs))
}

func allocateStubFunc() func(devs []string) (*pluginapi.AllocateResponse, error) {
	return func(devs []string) (*pluginapi.AllocateResponse, error) {
		resp := new(pluginapi.ContainerAllocateResponse)
		resp.Envs = make(map[string]string)
		for _, dev := range devs {
			switch dev {
			case "dev1":
				resp.Devices = append(resp.Devices, &pluginapi.DeviceSpec{
					ContainerPath: "/dev/aaa",
					HostPath:      "/dev/aaa",
					Permissions:   "mrw",
				})

				resp.Devices = append(resp.Devices, &pluginapi.DeviceSpec{
					ContainerPath: "/dev/bbb",
					HostPath:      "/dev/bbb",
					Permissions:   "mrw",
				})

				resp.Mounts = append(resp.Mounts, &pluginapi.Mount{
					ContainerPath: "/container_dir1/file1",
					HostPath:      "host_dir1/file1",
					ReadOnly:      true,
				})

			case "dev2":
				resp.Devices = append(resp.Devices, &pluginapi.DeviceSpec{
					ContainerPath: "/dev/ccc",
					HostPath:      "/dev/ccc",
					Permissions:   "mrw",
				})

				resp.Mounts = append(resp.Mounts, &pluginapi.Mount{
					ContainerPath: "/container_dir1/file2",
					HostPath:      "host_dir1/file2",
					ReadOnly:      true,
				})

				resp.Envs["key1"] = "val1"
			}
		}
		resps := new(pluginapi.AllocateResponse)
		resps.ContainerResponses = append(resps.ContainerResponses, resp)
		return resps, nil
	}
}
