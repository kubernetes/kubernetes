package device

import pluginapi "k8s.io/kubelet/pkg/apis/deviceplugin/v1beta1"

// DeviceInstances is a mapping device name -> plugin device data
type DeviceInstances map[string]pluginapi.Device

// ResourceDeviceInstances is a mapping resource name -> DeviceInstances
type ResourceDeviceInstances map[string]DeviceInstances

func NewResourceDeviceInstances() ResourceDeviceInstances {
	return make(ResourceDeviceInstances)
}

func (rdev ResourceDeviceInstances) Clone() ResourceDeviceInstances {
	clone := NewResourceDeviceInstances()
	for resourceName, resourceDevs := range rdev {
		clone[resourceName] = make(map[string]pluginapi.Device)
		for devID, dev := range resourceDevs {
			clone[resourceName][devID] = dev
		}
	}
	return clone
}
