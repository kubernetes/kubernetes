// Copyright (c) 2015-2016, NVIDIA CORPORATION. All rights reserved.

package nvidia

import (
	"sort"

	"github.com/haniceboy/nvidia-docker/tools/src/cuda"
	"github.com/haniceboy/nvidia-docker/tools/src/nvml"
)

type NVMLDev nvml.Device
type CUDADev cuda.Device

type Device struct {
	*NVMLDev
	*CUDADev
}

type NVMLDevStatus nvml.DeviceStatus

type DeviceStatus struct {
	*NVMLDevStatus
}

func (d *Device) Status() (*DeviceStatus, error) {
	s, err := (*nvml.Device)(d.NVMLDev).Status()
	if err != nil {
		return nil, err
	}

	return &DeviceStatus{(*NVMLDevStatus)(s)}, nil
}

type deviceSorter struct {
	devs []Device
}

func (s *deviceSorter) Len() int {
	return len(s.devs)
}

func (s *deviceSorter) Swap(i, j int) {
	s.devs[i], s.devs[j] = s.devs[j], s.devs[i]
}

func (s *deviceSorter) Less(i, j int) bool {
	return s.devs[i].PCI.BusID < s.devs[j].PCI.BusID
}

func LookupDevices() (devs []Device, err error) {
	var i uint

	n, err := nvml.GetDeviceCount()
	if err != nil {
		return nil, err
	}
	devs = make([]Device, 0, n)

	for i = 0; i < n; i++ {
		nd, err := nvml.NewDevice(i)
		if err != nil {
			return nil, err
		}
		cd, err := cuda.NewDevice(nd.PCI.BusID)
		if err != nil {
			return nil, err
		}
		devs = append(devs, Device{(*NVMLDev)(nd), (*CUDADev)(cd)})
	}

	for i = 0; i < n-1; i++ {
		for j := i + 1; j < n; j++ {
			ok, err := cuda.CanAccessPeer(
				(*cuda.Device)(devs[i].CUDADev),
				(*cuda.Device)(devs[j].CUDADev),
			)
			if err != nil {
				return nil, err
			}
			if ok {
				l, err := nvml.GetP2PLink(
					(*nvml.Device)(devs[i].NVMLDev),
					(*nvml.Device)(devs[j].NVMLDev),
				)
				if err != nil {
					return nil, err
				}
				devs[i].Topology = append(devs[i].Topology, nvml.P2PLink{devs[j].PCI.BusID, l})
				devs[j].Topology = append(devs[j].Topology, nvml.P2PLink{devs[i].PCI.BusID, l})
			}
		}
	}
	sort.Sort(&deviceSorter{devs})
	return
}

func LookupDevicePaths() ([]string, error) {
	var i uint

	n, err := nvml.GetDeviceCount()
	if err != nil {
		return nil, err
	}
	paths := make([]string, 0, n)

	for i = 0; i < n; i++ {
		p, err := nvml.GetDevicePath(i)
		if err != nil {
			return nil, err
		}
		paths = append(paths, p)
	}
	return paths, nil
}
