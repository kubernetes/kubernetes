package devicemanager

import (
	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/klog"
	pluginapi "k8s.io/kubernetes/pkg/kubelet/apis/deviceplugin/v1beta1"
	"k8s.io/kubernetes/pkg/kubelet/cm/topologymanager"
	"k8s.io/kubernetes/pkg/kubelet/cm/topologymanager/socketmask"
)

//GetTopologyHints implements the TopologyManager HintProvider Interface which ensures the Device Manager is consulted when Topology Aware Hints for each container are created
func (m *ManagerImpl) GetTopologyHints(pod v1.Pod, container v1.Container) ([]topologymanager.TopologyHint, bool) {
	klog.Infof("Devices in GetTopologyHints: %v", m.allDevices)

	var deviceHints []topologymanager.TopologyHint
	admit := false

	var tempMaskSet []topologymanager.TopologyHint
	allDeviceSockets := make(map[int]bool)

	firstIteration := true
	containerRequiresDevice := false

	for resourceObj, amountObj := range container.Resources.Requests {
		resource := string(resourceObj)
		amount := int64(amountObj.Value())
		if m.isDevicePluginResource(resource) {
			klog.Infof("%v is a resource managed by device manager.", resource)
			klog.Infof("Health Devices: %v", m.healthyDevices[resource])
			containerRequiresDevice = true
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

			deviceSocketAvail := getDevicesPerSocket(resource, available, m.allDevices)

			for socket, amountAvail := range deviceSocketAvail {
				klog.Infof("Socket: %v, Avail: %v AmountReq: %v", socket, amountAvail, amount)
				mask, _ := socketmask.NewSocketMask(int(socket))
				klog.Infof("Socket Mask: %v", mask.String())
				if amountAvail >= amount {
					if firstIteration {
						tempMaskSet = append(tempMaskSet, topologymanager.TopologyHint{SocketMask: mask})
					} else {
						isEqual := checkIfMaskEqualsStoreMask(deviceHints, mask)
						if isEqual {
							tempMaskSet = append(tempMaskSet, topologymanager.TopologyHint{SocketMask: mask})
						}
					}
				}
				allDeviceSockets[int(socket)] = true
			}
			firstIteration = false
			deviceHints = tempMaskSet
			tempMaskSet = nil
		}
	}

	if containerRequiresDevice {
		if len(allDeviceSockets) > 1 {
			var allDeviceSocketsInt []int
			for socket := range allDeviceSockets {
				allDeviceSocketsInt = append(allDeviceSocketsInt, socket)
			}
			crossSocketMask, _ := socketmask.NewSocketMask(allDeviceSocketsInt...)
			klog.Infof("CrossSocketMask: %v", crossSocketMask.String())
			deviceHints = append(deviceHints, topologymanager.TopologyHint{SocketMask: crossSocketMask})
		}
		admit = calculateIfDeviceHasSocketAffinity(deviceHints)
		klog.Infof("DeviceHints: %v, Admit: %v", deviceHints, admit)
	}

	return deviceHints, admit
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

func getDevicesPerSocket(resource string, available sets.String, allDevices map[string][]pluginapi.Device) map[int64]int64 {
	deviceSocketAvail := make(map[int64]int64)
	for availID := range available {
		for _, device := range allDevices[resource] {
			if availID == device.ID {
				socket := device.Topology.Socket
				deviceSocketAvail[socket]++
			}
		}
	}
	return deviceSocketAvail
}

func checkIfMaskEqualsStoreMask(existingDeviceHints []topologymanager.TopologyHint, newMask socketmask.SocketMask) bool {
	maskEqual := false
	for _, storedHint := range existingDeviceHints {
		if storedHint.SocketMask.IsEqual(newMask) {
			maskEqual = true
			break
		}
	}
	return maskEqual
}

func calculateIfDeviceHasSocketAffinity(deviceHints []topologymanager.TopologyHint) bool {
	admit := false
	for _, hint := range deviceHints {
		if hint.SocketMask.Count() == 1 {
			admit = true
			break
		}
	}
	return admit
}
