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

package cm

import (
	"flag"
	"fmt"
	"strconv"
	"testing"

	"github.com/stretchr/testify/assert"

	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/uuid"
	pluginapi "k8s.io/kubernetes/pkg/kubelet/apis/deviceplugin/v1alpha1"
	"k8s.io/kubernetes/pkg/kubelet/deviceplugin"
)

func TestStart(t *testing.T) {
	as := assert.New(t)
	updateCapacityFunc := func(updates v1.ResourceList) {}

	testDevicePluginHandler, err := NewDevicePluginHandlerImpl(updateCapacityFunc)
	as.NotNil(testDevicePluginHandler)
	as.Nil(err)

	err = testDevicePluginHandler.Start()
	as.Nil(err)
}

func TestUpdateCapacity(t *testing.T) {
	var expected = v1.ResourceList{}
	as := assert.New(t)
	verifyCapacityFunc := func(updates v1.ResourceList) {
		as.Equal(expected, updates)
	}
	testDevicePluginHandler, err := NewDevicePluginHandlerImpl(verifyCapacityFunc)
	as.NotNil(testDevicePluginHandler)
	as.Nil(err)

	devs := []*pluginapi.Device{
		{ID: "Device1", Health: pluginapi.Healthy},
		{ID: "Device2", Health: pluginapi.Healthy},
		{ID: "Device3", Health: pluginapi.Unhealthy},
	}

	resourceName := "resource1"
	// Adds three devices for resource1, two healthy and one unhealthy.
	// Expects capacity for resource1 to be 2.
	expected[v1.ResourceName(resourceName)] = *resource.NewQuantity(int64(2), resource.DecimalSI)
	testDevicePluginHandler.deviceMonitorCallback(resourceName, devs, []*pluginapi.Device{}, []*pluginapi.Device{})
	// Deletes an unhealthy device should NOT change capacity.
	testDevicePluginHandler.deviceMonitorCallback(resourceName, []*pluginapi.Device{}, []*pluginapi.Device{}, []*pluginapi.Device{devs[2]})
	// Updates a healthy device to unhealthy should reduce capacity by 1.
	expected[v1.ResourceName(resourceName)] = *resource.NewQuantity(int64(1), resource.DecimalSI)
	// Deletes a healthy device should reduce capacity by 1.
	expected[v1.ResourceName(resourceName)] = *resource.NewQuantity(int64(0), resource.DecimalSI)
	// Tests adding another resource.
	delete(expected, v1.ResourceName(resourceName))
	resourceName2 := "resource2"
	expected[v1.ResourceName(resourceName2)] = *resource.NewQuantity(int64(2), resource.DecimalSI)
	testDevicePluginHandler.deviceMonitorCallback(resourceName2, devs, []*pluginapi.Device{}, []*pluginapi.Device{})
}

type stringPairType struct {
	value1 string
	value2 string
}

func TestCheckpoint(t *testing.T) {
	resourceName1 := "domain1.com/resource1"
	resourceName2 := "domain2.com/resource2"

	as := assert.New(t)

	testDevicePluginHandler := &DevicePluginHandlerImpl{
		allDevices:       make(map[string]sets.String),
		allocatedDevices: make(map[string]podDevices),
		socketdir:        "/tmp/",
	}
	testDevicePluginHandler.allocatedDevices[resourceName1] = make(podDevices)
	testDevicePluginHandler.allocatedDevices[resourceName1].insert("pod1", "con1", "dev1")
	testDevicePluginHandler.allocatedDevices[resourceName1].insert("pod1", "con1", "dev2")
	testDevicePluginHandler.allocatedDevices[resourceName1].insert("pod1", "con2", "dev1")
	testDevicePluginHandler.allocatedDevices[resourceName1].insert("pod2", "con1", "dev1")
	testDevicePluginHandler.allocatedDevices[resourceName2] = make(podDevices)
	testDevicePluginHandler.allocatedDevices[resourceName2].insert("pod1", "con1", "dev3")
	testDevicePluginHandler.allocatedDevices[resourceName2].insert("pod1", "con1", "dev4")

	err := testDevicePluginHandler.writeCheckpoint()
	as.Nil(err)
	expected := testDevicePluginHandler.allocatedDevices
	testDevicePluginHandler.allocatedDevices = make(map[string]podDevices)
	err = testDevicePluginHandler.readCheckpoint()
	as.Nil(err)
	as.Equal(expected, testDevicePluginHandler.allocatedDevices)
}

func TestPodContainerDeviceAllocation(t *testing.T) {
	flag.Set("alsologtostderr", fmt.Sprintf("%t", true))
	var logLevel string
	flag.StringVar(&logLevel, "logLevel", "4", "test")
	flag.Lookup("v").Value.Set(logLevel)

	var activePods []*v1.Pod
	resourceName1 := "domain1.com/resource1"
	resourceQuantity1 := *resource.NewQuantity(int64(2), resource.DecimalSI)
	devId1 := "dev1"
	devId2 := "dev2"
	resourceName2 := "domain2.com/resource2"
	resourceQuantity2 := *resource.NewQuantity(int64(1), resource.DecimalSI)
	devId3 := "dev3"
	devId4 := "dev4"

	as := assert.New(t)
	monitorCallback := func(resourceName string, added, updated, deleted []*pluginapi.Device) {}

	testDevicePluginHandler := &DevicePluginHandlerImpl{
		deviceMonitorCallback: monitorCallback,
		allDevices:            make(map[string]sets.String),
		allocatedDevices:      make(map[string]podDevices),
		socketdir:             "/tmp/",
	}
	testDevicePluginHandler.allDevices[resourceName1] = sets.NewString()
	testDevicePluginHandler.allDevices[resourceName1].Insert(devId1)
	testDevicePluginHandler.allDevices[resourceName1].Insert(devId2)
	testDevicePluginHandler.allDevices[resourceName2] = sets.NewString()
	testDevicePluginHandler.allDevices[resourceName2].Insert(devId3)
	testDevicePluginHandler.allDevices[resourceName2].Insert(devId4)

	Resources1devs := []*pluginapi.Device{
		{ID: devId1, Health: pluginapi.Healthy},
		{ID: devId2, Health: pluginapi.Healthy},
	}

	Resources2devs := []*pluginapi.Device{
		{ID: devId3, Health: pluginapi.Healthy},
		{ID: devId4, Health: pluginapi.Healthy},
	}

	devs := map[string][]*pluginapi.Device{
		resourceName1: Resources1devs,
		resourceName2: Resources2devs,
	}

	testDevicePluginHandler.endpoints = make(map[string]*deviceplugin.Endpoint)
	for count, resource := range []string{resourceName1, resourceName2} {
		socket := "/tmp/resource" + strconv.Itoa(count) + ".sock"
		p := deviceplugin.NewDevicePluginStub(devs[resource], socket, resource)
		err := p.Start()
		as.Nil(err)

		defer p.Stop()

		endpoint, err := deviceplugin.NewEndpoint(socket, resource, testDevicePluginHandler.deviceMonitorCallback)
		as.Nil(err)
		testDevicePluginHandler.endpoints[resource] = endpoint
	}

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

	cm := &containerManagerImpl{
		devicePluginHandler: testDevicePluginHandler,
	}
	activePods = append(activePods, pod)
	runContainerOpts, err := cm.GetResources(pod, &pod.Spec.Containers[0], activePods)
	as.Equal(len(runContainerOpts.Devices), 3)
	// Two devices require to mount the same path. Expects a single mount entry to be created.
	as.Equal(len(runContainerOpts.Mounts), 2)
	as.Equal(len(runContainerOpts.Envs), 2)

	// Requesting to create a pod without enough resources should fail.
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
							v1.ResourceName(resourceName1): resourceQuantity1,
						},
					},
				},
			},
		},
	}
	runContainerOpts2, err := cm.GetResources(failPod, &failPod.Spec.Containers[0], activePods)
	as.NotNil(err)
	as.Equal(len(runContainerOpts2.Devices), 0)
	as.Equal(len(runContainerOpts2.Mounts), 0)
	as.Equal(len(runContainerOpts2.Envs), 0)

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
	runContainerOpts3, err := cm.GetResources(newPod, &newPod.Spec.Containers[0], activePods)
	as.Nil(err)
	as.Equal(len(runContainerOpts3.Envs), 1)
}
