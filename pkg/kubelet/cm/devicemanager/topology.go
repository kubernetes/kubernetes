/*
Copyright 2019 The Kubernetes Authors.

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
	"reflect"

	"k8s.io/klog"

	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/kubernetes/pkg/kubelet/cm/topologymanager"
	"k8s.io/kubernetes/pkg/kubelet/cm/topologymanager/socketmask"

	pluginapi "k8s.io/kubernetes/pkg/kubelet/apis/deviceplugin/v1beta1"
)

type topology struct {
	largestSocket int64
	deviceMask    []socketmask.SocketMask
	affinity      bool
	allDevices    map[string][]pluginapi.Device
}

// GetTopologyHints calculates a BitMask based on device socket availability for an incoming container
func (m *ManagerImpl) GetTopologyHints(pod v1.Pod, container v1.Container) topologymanager.TopologyHints {
	topo := &topology{
		largestSocket: int64(-1),
		affinity:      true,
		allDevices:    m.allDevices,
	}
	klog.Infof("Devices in GetTopologyHints: %v", topo.allDevices)
	var finalCrossSocketMask socketmask.SocketMask
	count := false
	topo.largestSocket = topo.getLargestSocket()
	deviceTriggered := false
	klog.Infof("Largest Socket in Devices: %v", topo.largestSocket)
	for resourceObj, amountObj := range container.Resources.Requests {
		resource := string(resourceObj)
		amount := int64(amountObj.Value())
		if m.isDevicePluginResource(resource) {
			deviceTriggered = true
			klog.Infof("%v is a resource managed by device manager.", resource)
			klog.Infof("Health Devices: %v", m.healthyDevices[resource])
			if _, ok := m.healthyDevices[resource]; !ok {
				klog.Infof("No healthy devices for resource %v", resource)
				continue
			}
			available := m.getAvailableDevices(resource)

			if int64(available.Len()) < amount {
				klog.Infof("requested number of devices unavailable for %s. Requested: %d, Available: %d", resource, amount, available.Len())
				continue
			}
			klog.Infof("[devicemanager] Available devices for resource %v: %v", resource, available)
			deviceSocketAvail := topo.getDevicesPerSocket(resource, available)

			var mask socketmask.SocketMask
			var crossSocket socketmask.SocketMask
			crossSocket = make([]int64, (topo.largestSocket + 1))
			var overwriteDeviceMask []socketmask.SocketMask
			for socket, amountAvail := range deviceSocketAvail {
				klog.Infof("Socket: %v, Avail: %v", socket, amountAvail)
				mask = nil
				if amountAvail >= amount {
					mask = topo.calculateDeviceMask(socket)
					klog.Infof("Mask: %v", mask)
					if !count {
						klog.Infof("Not Count. Device Mask: %v", topo.deviceMask)
						topo.deviceMask = append(topo.deviceMask, mask)
					} else {
						klog.Infof("Count. Device Mask: %v", topo.deviceMask)
						overwriteDeviceMask = append(overwriteDeviceMask, checkIfMaskEqualsStoreMask(topo.deviceMask, mask)...)
						klog.Infof("OverwriteDeviceMask: %v", overwriteDeviceMask)
					}
				}
				//crossSocket can be duplicate of mask need to remove if so
				crossSocket[socket] = 1
			}
			if !count {
				finalCrossSocketMask = crossSocket
			} else {
				topo.deviceMask = overwriteDeviceMask
				klog.Infof("DeviceMask: %v", topo.deviceMask)
				if !reflect.DeepEqual(finalCrossSocketMask, crossSocket) {
					finalCrossSocketMask = topo.calculateAllDeviceMask(finalCrossSocketMask, crossSocket)
				}
			}
			klog.Infof("deviceMask: %v", topo.deviceMask)
			klog.Infof("finalCrossSocketMask: %v", finalCrossSocketMask)

			count = true
		}
	}
	if deviceTriggered {
		topo.deviceMask = append(topo.deviceMask, finalCrossSocketMask)
		topo.affinity = calculateIfDeviceHasSocketAffinity(topo.deviceMask)
	}
	klog.Infof("DeviceMask %v: Device Affinity: %v", topo.deviceMask, topo.affinity)
	return topologymanager.TopologyHints{
		SocketAffinity: topo.deviceMask,
		Affinity:       topo.affinity,
	}
}

func (m *ManagerImpl) getAvailableDevices(resource string) sets.String {
	// Gets Devices in use.
	m.updateAllocatedDevices(m.activePods())
	devicesInUse := m.allocatedDevices[resource]
	klog.Infof("Devices in use:%v", devicesInUse)

	// Gets a list of available devices.
	available := m.healthyDevices[resource].Difference(devicesInUse)
	return available
}

func (t *topology) getLargestSocket() int64 {
	largestSocket := int64(-1)
	for _, list := range t.allDevices {
		for _, device := range list {
			if device.Topology.Socket > largestSocket {
				largestSocket = device.Topology.Socket
			}
		}
	}
	return largestSocket
}

func (t *topology) getDevicesPerSocket(resource string, available sets.String) map[int64]int64 {
	deviceSocketAvail := make(map[int64]int64)
	for availID := range available {
		for _, device := range t.allDevices[resource] {
			klog.Infof("[device-manager] AvailID: %v DeviceID: %v", availID, device)
			if availID == device.ID {
				socket := device.Topology.Socket
				deviceSocketAvail[socket]++
			}
		}
	}
	return deviceSocketAvail
}

func (t *topology) calculateDeviceMask(socket int64) socketmask.SocketMask {
	var mask socketmask.SocketMask
	for i := int64(0); i < t.largestSocket+1; i++ {
		if i == socket {
			mask = append(mask, 1)
		} else {
			mask = append(mask, 0)
		}
	}
	klog.Infof("Mask: %v", mask)
	return mask
}

func checkIfMaskEqualsStoreMask(existingDeviceMask []socketmask.SocketMask, newMask socketmask.SocketMask) []socketmask.SocketMask {
	var newDeviceMask []socketmask.SocketMask
	for _, storedMask := range existingDeviceMask {
		klog.Infof("For. StoredMask: %v", storedMask)
		if reflect.DeepEqual(storedMask, newMask) {
			klog.Infof("DeepEqual.")
			newDeviceMask = append(newDeviceMask, storedMask)
		}
	}
	return newDeviceMask
}

func (t *topology) calculateAllDeviceMask(finalSocketMask, crossSocket socketmask.SocketMask) socketmask.SocketMask {
	var tempSocketMask socketmask.SocketMask
	tempSocketMask = make([]int64, t.largestSocket+1)
	for i, bit := range finalSocketMask {
		klog.Infof("i %v for cross Socket, bit %v crossSocket[i] %v or result %v", bit, crossSocket[i], bit|crossSocket[i])
		tempSocketMask[i] = bit | crossSocket[i]
	}
	klog.Infof("TempSocketMask: %v", tempSocketMask)
	finalSocketMask = tempSocketMask
	return finalSocketMask
}

func calculateIfDeviceHasSocketAffinity(deviceMask []socketmask.SocketMask) bool {
	affinity := false
	for _, outerMask := range deviceMask {
		for _, innerMask := range outerMask {
			if innerMask == 0 {
				affinity = true
				break
			}
		}
	}
	return affinity
}
