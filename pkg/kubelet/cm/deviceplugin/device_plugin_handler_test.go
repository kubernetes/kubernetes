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

package deviceplugin

import (
	"flag"
	"fmt"
	"reflect"
	"testing"

	"github.com/stretchr/testify/assert"

	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/uuid"
	pluginapi "k8s.io/kubernetes/pkg/kubelet/apis/deviceplugin/v1alpha"
)

func TestUpdateCapacity(t *testing.T) {
	var expected = v1.ResourceList{}
	as := assert.New(t)
	verifyCapacityFunc := func(updates v1.ResourceList) {
		as.Equal(expected, updates)
	}
	testHandler, err := NewHandlerImpl(verifyCapacityFunc)
	as.NotNil(testHandler)
	as.Nil(err)

	devs := []pluginapi.Device{
		{ID: "Device1", Health: pluginapi.Healthy},
		{ID: "Device2", Health: pluginapi.Healthy},
		{ID: "Device3", Health: pluginapi.Unhealthy},
	}

	resourceName := "resource1"
	// Adds three devices for resource1, two healthy and one unhealthy.
	// Expects capacity for resource1 to be 2.
	expected[v1.ResourceName(resourceName)] = *resource.NewQuantity(int64(2), resource.DecimalSI)
	testHandler.devicePluginManagerMonitorCallback(resourceName, devs, []pluginapi.Device{}, []pluginapi.Device{})
	// Deletes an unhealthy device should NOT change capacity.
	testHandler.devicePluginManagerMonitorCallback(resourceName, []pluginapi.Device{}, []pluginapi.Device{}, []pluginapi.Device{devs[2]})
	// Updates a healthy device to unhealthy should reduce capacity by 1.
	expected[v1.ResourceName(resourceName)] = *resource.NewQuantity(int64(1), resource.DecimalSI)
	// Deletes a healthy device should reduce capacity by 1.
	expected[v1.ResourceName(resourceName)] = *resource.NewQuantity(int64(0), resource.DecimalSI)
	// Tests adding another resource.
	delete(expected, v1.ResourceName(resourceName))
	resourceName2 := "resource2"
	expected[v1.ResourceName(resourceName2)] = *resource.NewQuantity(int64(2), resource.DecimalSI)
	testHandler.devicePluginManagerMonitorCallback(resourceName2, devs, []pluginapi.Device{}, []pluginapi.Device{})
}

type stringPairType struct {
	value1 string
	value2 string
}

// DevicePluginManager stub to test device Allocation behavior.
type DevicePluginManagerTestStub struct {
	// All data structs are keyed by resourceName+DevId
	devRuntimeDevices map[string][]stringPairType
	devRuntimeMounts  map[string][]stringPairType
	devRuntimeEnvs    map[string][]stringPairType
}

func NewDevicePluginManagerTestStub() (*DevicePluginManagerTestStub, error) {
	return &DevicePluginManagerTestStub{
		devRuntimeDevices: make(map[string][]stringPairType),
		devRuntimeMounts:  make(map[string][]stringPairType),
		devRuntimeEnvs:    make(map[string][]stringPairType),
	}, nil
}

func (m *DevicePluginManagerTestStub) Start() error {
	return nil
}

func (m *DevicePluginManagerTestStub) Devices() map[string][]pluginapi.Device {
	return make(map[string][]pluginapi.Device)
}

func (m *DevicePluginManagerTestStub) Allocate(resourceName string, devIds []string) (*pluginapi.AllocateResponse, error) {
	resp := new(pluginapi.AllocateResponse)
	resp.Envs = make(map[string]string)
	for _, id := range devIds {
		key := resourceName + id
		fmt.Printf("Alloc device %v for resource %v\n", id, resourceName)
		for _, dev := range m.devRuntimeDevices[key] {
			fmt.Printf("Add dev %v %v\n", dev.value1, dev.value2)
			resp.Devices = append(resp.Devices, &pluginapi.DeviceSpec{
				ContainerPath: dev.value1,
				HostPath:      dev.value2,
				Permissions:   "mrw",
			})
		}
		for _, mount := range m.devRuntimeMounts[key] {
			fmt.Printf("Add mount %v %v\n", mount.value1, mount.value2)
			resp.Mounts = append(resp.Mounts, &pluginapi.Mount{
				ContainerPath: mount.value1,
				HostPath:      mount.value2,
				ReadOnly:      true,
			})
		}
		for _, env := range m.devRuntimeEnvs[key] {
			fmt.Printf("Add env %v %v\n", env.value1, env.value2)
			resp.Envs[env.value1] = env.value2
		}
	}
	return resp, nil
}

func (m *DevicePluginManagerTestStub) Stop() error {
	return nil
}

func (m *DevicePluginManagerTestStub) CheckpointFile() string {
	return "/tmp/device-plugin-checkpoint"
}

func constructDevices(devices []string) sets.String {
	ret := sets.NewString()
	for _, dev := range devices {
		ret.Insert(dev)
	}
	return ret
}

func constructAllocResp(devices, mounts, envs map[string]string) *pluginapi.AllocateResponse {
	resp := &pluginapi.AllocateResponse{}
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

	m, err := NewDevicePluginManagerTestStub()
	as := assert.New(t)
	as.Nil(err)

	testHandler := &HandlerImpl{
		devicePluginManager: m,
		allDevices:          make(map[string]sets.String),
		allocatedDevices:    make(map[string]sets.String),
		podDevices:          make(podDevices),
	}

	testHandler.podDevices.insert("pod1", "con1", resourceName1,
		constructDevices([]string{"dev1", "dev2"}),
		constructAllocResp(map[string]string{"/dev/r1dev1": "/dev/r1dev1", "/dev/r1dev2": "/dev/r1dev2"},
			map[string]string{"/home/r1lib1": "/usr/r1lib1"}, map[string]string{}))
	testHandler.podDevices.insert("pod1", "con1", resourceName2,
		constructDevices([]string{"dev1", "dev2"}),
		constructAllocResp(map[string]string{"/dev/r2dev1": "/dev/r2dev1", "/dev/r2dev2": "/dev/r2dev2"},
			map[string]string{"/home/r2lib1": "/usr/r2lib1"},
			map[string]string{"r2devices": "dev1 dev2"}))
	testHandler.podDevices.insert("pod1", "con2", resourceName1,
		constructDevices([]string{"dev3"}),
		constructAllocResp(map[string]string{"/dev/r1dev3": "/dev/r1dev3"},
			map[string]string{"/home/r1lib1": "/usr/r1lib1"}, map[string]string{}))
	testHandler.podDevices.insert("pod2", "con1", resourceName1,
		constructDevices([]string{"dev4"}),
		constructAllocResp(map[string]string{"/dev/r1dev4": "/dev/r1dev4"},
			map[string]string{"/home/r1lib1": "/usr/r1lib1"}, map[string]string{}))

	expectedPodDevices := testHandler.podDevices
	expectedAllocatedDevices := testHandler.podDevices.devices()

	err = testHandler.writeCheckpoint()
	as.Nil(err)
	testHandler.podDevices = make(podDevices)
	err = testHandler.readCheckpoint()
	as.Nil(err)

	as.Equal(len(expectedPodDevices), len(testHandler.podDevices))
	for podUID, containerDevices := range expectedPodDevices {
		for conName, resources := range containerDevices {
			for resource := range resources {
				as.True(reflect.DeepEqual(
					expectedPodDevices.containerDevices(podUID, conName, resource),
					testHandler.podDevices.containerDevices(podUID, conName, resource)))
				opts1 := expectedPodDevices.deviceRunContainerOptions(podUID, conName)
				opts2 := testHandler.podDevices.deviceRunContainerOptions(podUID, conName)
				as.Equal(len(opts1.Envs), len(opts2.Envs))
				as.Equal(len(opts1.Mounts), len(opts2.Mounts))
				as.Equal(len(opts1.Devices), len(opts2.Devices))
			}
		}
	}
	as.True(reflect.DeepEqual(expectedAllocatedDevices, testHandler.allocatedDevices))
}

func TestPodContainerDeviceAllocation(t *testing.T) {
	flag.Set("alsologtostderr", fmt.Sprintf("%t", true))
	var logLevel string
	flag.StringVar(&logLevel, "logLevel", "4", "test")
	flag.Lookup("v").Value.Set(logLevel)

	var activePods []*v1.Pod
	resourceName1 := "domain1.com/resource1"
	resourceQuantity1 := *resource.NewQuantity(int64(2), resource.DecimalSI)
	devID1 := "dev1"
	devID2 := "dev2"
	resourceName2 := "domain2.com/resource2"
	resourceQuantity2 := *resource.NewQuantity(int64(1), resource.DecimalSI)
	devID3 := "dev3"
	devID4 := "dev4"

	m, err := NewDevicePluginManagerTestStub()
	as := assert.New(t)
	as.Nil(err)
	monitorCallback := func(resourceName string, added, updated, deleted []pluginapi.Device) {}

	testHandler := &HandlerImpl{
		devicePluginManager:                m,
		devicePluginManagerMonitorCallback: monitorCallback,
		allDevices:                         make(map[string]sets.String),
		allocatedDevices:                   make(map[string]sets.String),
		podDevices:                         make(podDevices),
	}
	testHandler.allDevices[resourceName1] = sets.NewString()
	testHandler.allDevices[resourceName1].Insert(devID1)
	testHandler.allDevices[resourceName1].Insert(devID2)
	testHandler.allDevices[resourceName2] = sets.NewString()
	testHandler.allDevices[resourceName2].Insert(devID3)
	testHandler.allDevices[resourceName2].Insert(devID4)

	m.devRuntimeDevices[resourceName1+devID1] = append(m.devRuntimeDevices[resourceName1+devID1], stringPairType{"/dev/aaa", "/dev/aaa"})
	m.devRuntimeDevices[resourceName1+devID1] = append(m.devRuntimeDevices[resourceName1+devID1], stringPairType{"/dev/bbb", "/dev/bbb"})
	m.devRuntimeDevices[resourceName1+devID2] = append(m.devRuntimeDevices[resourceName1+devID2], stringPairType{"/dev/ccc", "/dev/ccc"})
	m.devRuntimeMounts[resourceName1+devID1] = append(m.devRuntimeMounts[resourceName1+devID1], stringPairType{"/container_dir1/file1", "host_dir1/file1"})
	m.devRuntimeMounts[resourceName1+devID2] = append(m.devRuntimeMounts[resourceName1+devID2], stringPairType{"/container_dir1/file2", "host_dir1/file2"})
	m.devRuntimeEnvs[resourceName1+devID2] = append(m.devRuntimeEnvs[resourceName1+devID2], stringPairType{"key1", "val1"})
	m.devRuntimeEnvs[resourceName2+devID3] = append(m.devRuntimeEnvs[resourceName2+devID3], stringPairType{"key2", "val2"})
	m.devRuntimeEnvs[resourceName2+devID4] = append(m.devRuntimeEnvs[resourceName2+devID4], stringPairType{"key2", "val3"})

	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			UID: uuid.NewUUID(),
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Name: string(uuid.NewUUID()),
					Resources: v1.ResourceRequirements{
						Limits: v1.ResourceList{
							v1.ResourceName(resourceName1): resourceQuantity1,
							v1.ResourceName("cpu"):         resourceQuantity1,
							v1.ResourceName(resourceName2): resourceQuantity2,
						},
					},
				},
			},
		},
	}

	activePods = append(activePods, pod)
	err = testHandler.Allocate(pod, &pod.Spec.Containers[0], activePods)
	as.Nil(err)
	runContainerOpts := testHandler.GetDeviceRunContainerOptions(pod, &pod.Spec.Containers[0])
	as.Equal(len(runContainerOpts.Devices), 3)
	as.Equal(len(runContainerOpts.Mounts), 2)
	as.Equal(len(runContainerOpts.Envs), 2)

	// Requesting to create a pod without enough resources should fail.
	as.Equal(2, testHandler.allocatedDevices[resourceName1].Len())
	failPod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			UID: uuid.NewUUID(),
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Name: string(uuid.NewUUID()),
					Resources: v1.ResourceRequirements{
						Limits: v1.ResourceList{
							v1.ResourceName(resourceName1): resourceQuantity2,
						},
					},
				},
			},
		},
	}
	err = testHandler.Allocate(failPod, &failPod.Spec.Containers[0], activePods)
	as.NotNil(err)
	runContainerOpts2 := testHandler.GetDeviceRunContainerOptions(failPod, &failPod.Spec.Containers[0])
	as.Nil(runContainerOpts2)

	// Requesting to create a new pod with a single resourceName2 should succeed.
	newPod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			UID: uuid.NewUUID(),
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Name: string(uuid.NewUUID()),
					Resources: v1.ResourceRequirements{
						Limits: v1.ResourceList{
							v1.ResourceName(resourceName2): resourceQuantity2,
						},
					},
				},
			},
		},
	}
	err = testHandler.Allocate(newPod, &newPod.Spec.Containers[0], activePods)
	as.Nil(err)
	runContainerOpts3 := testHandler.GetDeviceRunContainerOptions(newPod, &newPod.Spec.Containers[0])
	as.Equal(1, len(runContainerOpts3.Envs))
}
