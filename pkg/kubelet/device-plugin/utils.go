package deviceplugin

import (
	"fmt"
	"strings"

	"k8s.io/api/core/v1"

	pluginapi "k8s.io/kubernetes/pkg/kubelet/apis/device-plugin/v1alpha1"
)

func NewDevice(name, kind, vendor string) *pluginapi.Device {
	return &pluginapi.Device{
		Name:   name,
		Kind:   kind,
		Vendor: vendor,
	}
}

func NewDeviceHealth(name, kind, vendor, health string) *pluginapi.DeviceHealth {
	return &pluginapi.DeviceHealth{
		Name:   name,
		Kind:   kind,
		Vendor: vendor,
		Health: health,
	}
}

func NewError(err string) *pluginapi.Error {
	return &pluginapi.Error{
		Error:  true,
		Reason: err,
	}
}

func HasDevice(d *pluginapi.Device, devs []*pluginapi.Device) (int, bool) {
	name := DeviceKey(d)

	for i, d := range devs {
		if DeviceKey(d) != name {
			continue
		}

		return i, true
	}

	return -1, false
}

func deleteDev(d *pluginapi.Device, devs []*pluginapi.Device) []*pluginapi.Device {
	i, ok := HasDevice(d, devs)
	if !ok {
		return devs
	}

	devs[i], devs[len(devs)-1] = devs[len(devs)-1], devs[i]

	return devs[:len(devs)-1]
}

func deleteDevAt(i int, devs []*pluginapi.Device) []*pluginapi.Device {
	devs[i], devs[len(devs)-1] = devs[len(devs)-1], devs[i]
	return devs[:len(devs)-1]
}

const invalidChars = "/-"

func IsDevsValid(devs []*pluginapi.Device, vendor string) error {
	for _, d := range devs {
		if err := IsDevValid(d, vendor); err != nil {
			return err
		}
	}

	return nil
}

func IsDevValid(d *pluginapi.Device, vendor string) error {
	d.Vendor = strings.TrimSpace(d.Vendor)
	d.Kind = strings.TrimSpace(d.Kind)
	d.Name = strings.TrimSpace(d.Name)

	if d.Vendor != vendor {
		return fmt.Errorf("Vendor mismatch for device %+v "+
			"with vendor %s", d, vendor)
	}

	if d.Kind == "" || d.Name == "" {
		return fmt.Errorf("Invalid Empty Kind or name for device %+vs", d)
	}

	if d.Kind != "nvidia-gpu" && strings.ContainsAny(d.Kind, invalidChars) {
		return fmt.Errorf("Invalid device Kind '%s' should not contain any of '%s'",
			d.Kind, invalidChars)
	}
	return nil
}

// Expecting vendor to be trimed
func IsVendorValid(vendor string) error {
	if vendor == "" {
		return fmt.Errorf("Invalid Empty Vendor")
	}

	if vendor != "nvidia-gpu" && strings.ContainsAny(vendor, invalidChars) {
		return fmt.Errorf("Invalid vendor '%s' should not contain any of '%s'",
			vendor, invalidChars)
	}

	return nil
}

func DeviceName(k v1.ResourceName) (bool, string) {
	key := string(k)
	if k != v1.ResourceNvidiaGPU && !strings.HasPrefix(key, v1.ResourceOpaqueIntPrefix) {
		return false, ""
	}
	var name string
	if k == v1.ResourceNvidiaGPU {
		name = "nvidia-gpu"
	} else {
		name = strings.TrimPrefix(key, v1.ResourceOpaqueIntPrefix)
	}

	return true, name
}

func VendorDeviceKey(d *pluginapi.Device) string {
	return d.Vendor + "-" + DeviceKey(d)
}

func DeviceKey(d *pluginapi.Device) string {
	return d.Kind + "-" + d.Name
}
