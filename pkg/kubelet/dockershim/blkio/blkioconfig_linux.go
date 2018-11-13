// +build linux

package blkio

import (
	"encoding/json"
	"fmt"

	"github.com/opencontainers/runc/libcontainer/configs"
	"k8s.io/apimachinery/pkg/api/resource"
)

const (
	// copy from dockershim/docker_service.go:61

	containerTypeLabelKey       = "io.kubernetes.docker.type"
	containerTypeLabelSandbox   = "podsandbox"
	containerTypeLabelContainer = "container"
	containerLogPathLabelKey    = "io.kubernetes.container.logpath"
	sandboxIDLabelKey           = "io.kubernetes.sandbox.id"

	BlkioKey = "annotation.io.kubernetes.container.blkio"

	GraphDriverName          = "devicemapper"
	GraphDriverDeviceIdKey   = "DeviceId"
	GraphDriverDeviceNameKey = "DeviceName"
	GraphDriverDeviceSizeKey = "DeviceSize"

	RootfsDeviceKey = "rootfs"

	BlkioSubsystemName = "blkio"

	DeviceWeightMin = 10
	DeviceWeightMax = 1000
	DeviceBPSMin    = 1000
	DeviceIOPSMin   = 5
)

// Value resource.Quantity  `json:"value"`
type deviceValue struct {
	Device string `json:"device"`
	Value  string `json:"value"`
}

type Blkio struct {
	Weight          uint16        `json:"weight,omitempty"`
	WeightDevice    []deviceValue `json:"weight_device,omitempty"`
	DeviceReadBps   []deviceValue `json:"device_read_bps,omitempty"`
	DeviceWriteBps  []deviceValue `json:"device_write_bps,omitempty"`
	DeviceReadIOps  []deviceValue `json:"device_read_iops,omitempty"`
	DeviceWriteIOps []deviceValue `json:"device_write_iops,omitempty"`
}

func getBlkioDeviceValue(d *deviceValue, rootfsDevice string) (q, Major, Minor int64, err error) {
	quantity, err := resource.ParseQuantity(d.Value)
	if err != nil {
		return 0, 0, 0, fmt.Errorf("the %s.DeviceValue:%+v format error. %s", BlkioKey, d, err.Error())
	}
	q = quantity.Value()

	if d.Device == RootfsDeviceKey {
		d.Device = rootfsDevice

	}
	cfg, err := DeviceFromPath(d.Device, "")
	if err != nil {
		return 0, 0, 0, err
	}
	return q, cfg.Major, cfg.Minor, nil
}

func getBlkioWeightDevice(d *deviceValue, rootfsDevice string) (w configs.WeightDevice, err error) {
	q, Major, Minor, err := getBlkioDeviceValue(d, rootfsDevice)
	w.Weight = uint16(q & 0x00ffff)
	w.Major = Major
	w.Minor = Minor
	return w, err
}

func getBlkioThrottleDevice(d *deviceValue, rootfsDevice string) (t configs.ThrottleDevice, err error) {
	q, Major, Minor, err := getBlkioDeviceValue(d, rootfsDevice)
	t.Rate = uint64(q)
	t.Major = Major
	t.Minor = Minor
	return t, err
}

func getBlkioResource(b *Blkio, rootfsDevice string) (blkioResource configs.Resources, err error) {
	if b.Weight != 0 && (b.Weight < DeviceWeightMin || b.Weight > DeviceWeightMax) {
		return blkioResource, fmt.Errorf("blkio.weight:%d is out of rage. Currently allowed range of weights is from %d to %d.", b.Weight, DeviceWeightMin, DeviceWeightMax)
	}
	blkioResource.BlkioWeight = b.Weight

	for _, w := range b.WeightDevice {
		d, err := getBlkioWeightDevice(&w, rootfsDevice)
		if err != nil {
			return blkioResource, err
		}
		if d.Weight != 0 && (d.Weight < DeviceWeightMin || d.Weight > DeviceWeightMax) {
			return blkioResource, fmt.Errorf("blkio.weight_device:%d is out of rage. Currently allowed range of weights is from %d to %d.", d.Weight, DeviceWeightMin, DeviceWeightMax)
		}
		blkioResource.BlkioWeightDevice = append(blkioResource.BlkioWeightDevice, &d)
	}
	for _, w := range b.DeviceReadBps {
		t, err := getBlkioThrottleDevice(&w, rootfsDevice)
		if err != nil {
			return blkioResource, err
		}
		if t.Rate != 0 && t.Rate < DeviceBPSMin {
			return blkioResource, fmt.Errorf("blkio.device_read_bps:%d is out of rage. The Rate must be greater than %d.", t.Rate, DeviceBPSMin)
		}
		blkioResource.BlkioThrottleReadBpsDevice = append(blkioResource.BlkioThrottleReadBpsDevice, &t)
	}
	for _, w := range b.DeviceWriteBps {
		t, err := getBlkioThrottleDevice(&w, rootfsDevice)
		if err != nil {
			return blkioResource, err
		}
		if t.Rate != 0 && t.Rate < DeviceBPSMin {
			return blkioResource, fmt.Errorf("blkio.device_write_bps:%d is out of rage. The Rate must be greater than %d.", t.Rate, DeviceBPSMin)
		}
		blkioResource.BlkioThrottleWriteBpsDevice = append(blkioResource.BlkioThrottleWriteBpsDevice, &t)
	}
	for _, w := range b.DeviceReadIOps {
		t, err := getBlkioThrottleDevice(&w, rootfsDevice)
		if err != nil {
			return blkioResource, err
		}
		if t.Rate != 0 && t.Rate < DeviceIOPSMin {
			return blkioResource, fmt.Errorf("blkio.device_read_iops:%d is out of rage. The Rate must be greater than %d.", t.Rate, DeviceIOPSMin)
		}
		blkioResource.BlkioThrottleReadIOPSDevice = append(blkioResource.BlkioThrottleReadIOPSDevice, &t)
	}
	for _, w := range b.DeviceWriteIOps {
		t, err := getBlkioThrottleDevice(&w, rootfsDevice)
		if err != nil {
			return blkioResource, err
		}
		if t.Rate != 0 && t.Rate < DeviceIOPSMin {
			return blkioResource, fmt.Errorf("blkio.device_write_iops:%d is out of rage. The Rate must be greater than %d.", t.Rate, DeviceIOPSMin)
		}
		blkioResource.BlkioThrottleWriteIOPSDevice = append(blkioResource.BlkioThrottleWriteIOPSDevice, &t)
	}
	return blkioResource, nil
}

func cgroupToString(cgroup *configs.Cgroup) string {
	if cgroup == nil {
		return ""
	}
	data, _ := json.Marshal(cgroup)
	return string(data)
}
