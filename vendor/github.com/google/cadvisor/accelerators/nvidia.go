// Copyright 2017 Google Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
package accelerators

import (
	"bufio"
	"fmt"
	"io/ioutil"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"time"

	info "github.com/google/cadvisor/info/v1"

	"github.com/mindprince/gonvml"
	"k8s.io/klog"
)

type NvidiaManager struct {
	sync.Mutex

	// true if there are NVIDIA devices present on the node
	devicesPresent bool

	// true if the NVML library (libnvidia-ml.so.1) was loaded successfully
	nvmlInitialized bool

	// nvidiaDevices is a map from device minor number to a handle that can be used to get metrics about the device
	nvidiaDevices map[int]gonvml.Device
}

var sysFsPCIDevicesPath = "/sys/bus/pci/devices/"

const nvidiaVendorId = "0x10de"

// Setup initializes NVML if nvidia devices are present on the node.
func (nm *NvidiaManager) Setup() {
	if !detectDevices(nvidiaVendorId) {
		klog.V(4).Info("No NVIDIA devices found.")
		return
	}

	nm.devicesPresent = true

	initializeNVML(nm)
}

// detectDevices returns true if a device with given pci id is present on the node.
func detectDevices(vendorId string) bool {
	devices, err := ioutil.ReadDir(sysFsPCIDevicesPath)
	if err != nil {
		klog.Warningf("Error reading %q: %v", sysFsPCIDevicesPath, err)
		return false
	}

	for _, device := range devices {
		vendorPath := filepath.Join(sysFsPCIDevicesPath, device.Name(), "vendor")
		content, err := ioutil.ReadFile(vendorPath)
		if err != nil {
			klog.V(4).Infof("Error while reading %q: %v", vendorPath, err)
			continue
		}
		if strings.EqualFold(strings.TrimSpace(string(content)), vendorId) {
			klog.V(3).Infof("Found device with vendorId %q", vendorId)
			return true
		}
	}
	return false
}

// initializeNVML initializes the NVML library and sets up the nvmlDevices map.
// This is defined as a variable to help in testing.
var initializeNVML = func(nm *NvidiaManager) {
	if err := gonvml.Initialize(); err != nil {
		// This is under a logging level because otherwise we may cause
		// log spam if the drivers/nvml is not installed on the system.
		klog.V(4).Infof("Could not initialize NVML: %v", err)
		return
	}
	nm.nvmlInitialized = true
	numDevices, err := gonvml.DeviceCount()
	if err != nil {
		klog.Warningf("GPU metrics would not be available. Failed to get the number of nvidia devices: %v", err)
		return
	}
	klog.V(1).Infof("NVML initialized. Number of nvidia devices: %v", numDevices)
	nm.nvidiaDevices = make(map[int]gonvml.Device, numDevices)
	for i := 0; i < int(numDevices); i++ {
		device, err := gonvml.DeviceHandleByIndex(uint(i))
		if err != nil {
			klog.Warningf("Failed to get nvidia device handle %d: %v", i, err)
			continue
		}
		minorNumber, err := device.MinorNumber()
		if err != nil {
			klog.Warningf("Failed to get nvidia device minor number: %v", err)
			continue
		}
		nm.nvidiaDevices[int(minorNumber)] = device
	}
}

// Destroy shuts down NVML.
func (nm *NvidiaManager) Destroy() {
	if nm.nvmlInitialized {
		gonvml.Shutdown()
	}
}

// GetCollector returns a collector that can fetch nvidia gpu metrics for nvidia devices
// present in the devices.list file in the given devicesCgroupPath.
func (nm *NvidiaManager) GetCollector(devicesCgroupPath string) (AcceleratorCollector, error) {
	nc := &NvidiaCollector{}

	if !nm.devicesPresent {
		return nc, nil
	}
	// Makes sure that we don't call initializeNVML() concurrently and
	// that we only call initializeNVML() when it's not initialized.
	nm.Lock()
	if !nm.nvmlInitialized {
		initializeNVML(nm)
	}
	if !nm.nvmlInitialized || len(nm.nvidiaDevices) == 0 {
		nm.Unlock()
		return nc, nil
	}
	nm.Unlock()
	nvidiaMinorNumbers, err := parseDevicesCgroup(devicesCgroupPath)
	if err != nil {
		return nc, err
	}
	for _, minor := range nvidiaMinorNumbers {
		device, ok := nm.nvidiaDevices[minor]
		if !ok {
			return nc, fmt.Errorf("nvidia device minor number %d not found in cached devices", minor)
		}
		nc.Devices = append(nc.Devices, device)
	}
	return nc, nil
}

// parseDevicesCgroup parses the devices cgroup devices.list file for the container
// and returns a list of minor numbers corresponding to NVIDIA GPU devices that the
// container is allowed to access. In cases where the container has access to all
// devices or all NVIDIA devices but the devices are not enumerated separately in
// the devices.list file, we return an empty list.
// This is defined as a variable to help in testing.
var parseDevicesCgroup = func(devicesCgroupPath string) ([]int, error) {
	// Always return a non-nil slice
	nvidiaMinorNumbers := []int{}

	devicesList := filepath.Join(devicesCgroupPath, "devices.list")
	f, err := os.Open(devicesList)
	if err != nil {
		return nvidiaMinorNumbers, fmt.Errorf("error while opening devices cgroup file %q: %v", devicesList, err)
	}
	defer f.Close()

	s := bufio.NewScanner(f)

	// See https://www.kernel.org/doc/Documentation/cgroup-v1/devices.txt for the file format
	for s.Scan() {
		text := s.Text()

		fields := strings.Fields(text)
		if len(fields) != 3 {
			return nvidiaMinorNumbers, fmt.Errorf("invalid devices cgroup entry %q: must contain three whitespace-separated fields", text)
		}

		// Split the second field to find out major:minor numbers
		majorMinor := strings.Split(fields[1], ":")
		if len(majorMinor) != 2 {
			return nvidiaMinorNumbers, fmt.Errorf("invalid devices cgroup entry %q: second field should have one colon", text)
		}

		// NVIDIA graphics devices are character devices with major number 195.
		// https://github.com/torvalds/linux/blob/v4.13/Documentation/admin-guide/devices.txt#L2583
		if fields[0] == "c" && majorMinor[0] == "195" {
			minorNumber, err := strconv.Atoi(majorMinor[1])
			if err != nil {
				return nvidiaMinorNumbers, fmt.Errorf("invalid devices cgroup entry %q: minor number is not integer", text)
			}
			// We don't want devices like nvidiactl (195:255) and nvidia-modeset (195:254)
			if minorNumber < 128 {
				nvidiaMinorNumbers = append(nvidiaMinorNumbers, minorNumber)
			}
			// We are ignoring the "195:*" case
			// where the container has access to all NVIDIA devices on the machine.
		}
		// We are ignoring the "*:*" case
		// where the container has access to all devices on the machine.
	}
	return nvidiaMinorNumbers, nil
}

type NvidiaCollector struct {
	// Exposed for testing
	Devices []gonvml.Device
}

// UpdateStats updates the stats for NVIDIA GPUs (if any) attached to the container.
func (nc *NvidiaCollector) UpdateStats(stats *info.ContainerStats) error {
	for _, device := range nc.Devices {
		model, err := device.Name()
		if err != nil {
			return fmt.Errorf("error while getting gpu name: %v", err)
		}
		uuid, err := device.UUID()
		if err != nil {
			return fmt.Errorf("error while getting gpu uuid: %v", err)
		}
		memoryTotal, memoryUsed, err := device.MemoryInfo()
		if err != nil {
			return fmt.Errorf("error while getting gpu memory info: %v", err)
		}
		//TODO: Use housekeepingInterval
		utilizationGPU, err := device.AverageGPUUtilization(10 * time.Second)
		if err != nil {
			return fmt.Errorf("error while getting gpu utilization: %v", err)
		}

		stats.Accelerators = append(stats.Accelerators, info.AcceleratorStats{
			Make:        "nvidia",
			Model:       model,
			ID:          uuid,
			MemoryTotal: memoryTotal,
			MemoryUsed:  memoryUsed,
			DutyCycle:   uint64(utilizationGPU),
		})
	}
	return nil
}
