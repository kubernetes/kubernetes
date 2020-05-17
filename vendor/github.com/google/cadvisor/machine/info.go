// Copyright 2014 Google Inc. All Rights Reserved.
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

package machine

import (
	"bytes"
	"flag"
	"io/ioutil"
	"path/filepath"
	"strings"
	"time"

	"github.com/google/cadvisor/fs"
	info "github.com/google/cadvisor/info/v1"
	"github.com/google/cadvisor/nvm"
	"github.com/google/cadvisor/utils/cloudinfo"
	"github.com/google/cadvisor/utils/sysfs"
	"github.com/google/cadvisor/utils/sysinfo"

	"k8s.io/klog/v2"

	"golang.org/x/sys/unix"
)

const hugepagesDirectory = "/sys/kernel/mm/hugepages/"
const memoryControllerPath = "/sys/devices/system/edac/mc/"

var machineIDFilePath = flag.String("machine_id_file", "/etc/machine-id,/var/lib/dbus/machine-id", "Comma-separated list of files to check for machine-id. Use the first one that exists.")
var bootIDFilePath = flag.String("boot_id_file", "/proc/sys/kernel/random/boot_id", "Comma-separated list of files to check for boot-id. Use the first one that exists.")

func getInfoFromFiles(filePaths string) string {
	if len(filePaths) == 0 {
		return ""
	}
	for _, file := range strings.Split(filePaths, ",") {
		id, err := ioutil.ReadFile(file)
		if err == nil {
			return strings.TrimSpace(string(id))
		}
	}
	klog.Warningf("Couldn't collect info from any of the files in %q", filePaths)
	return ""
}

func Info(sysFs sysfs.SysFs, fsInfo fs.FsInfo, inHostNamespace bool) (*info.MachineInfo, error) {
	rootFs := "/"
	if !inHostNamespace {
		rootFs = "/rootfs"
	}

	cpuinfo, err := ioutil.ReadFile(filepath.Join(rootFs, "/proc/cpuinfo"))
	if err != nil {
		return nil, err
	}
	clockSpeed, err := GetClockSpeed(cpuinfo)
	if err != nil {
		return nil, err
	}

	memoryCapacity, err := GetMachineMemoryCapacity()
	if err != nil {
		return nil, err
	}

	memoryByType, err := GetMachineMemoryByType(memoryControllerPath)
	if err != nil {
		return nil, err
	}

	nvmInfo, err := nvm.GetInfo()
	if err != nil {
		return nil, err
	}

	hugePagesInfo, err := sysinfo.GetHugePagesInfo(sysFs, hugepagesDirectory)
	if err != nil {
		return nil, err
	}

	filesystems, err := fsInfo.GetGlobalFsInfo()
	if err != nil {
		klog.Errorf("Failed to get global filesystem information: %v", err)
	}

	diskMap, err := sysinfo.GetBlockDeviceInfo(sysFs)
	if err != nil {
		klog.Errorf("Failed to get disk map: %v", err)
	}

	netDevices, err := sysinfo.GetNetworkDevices(sysFs)
	if err != nil {
		klog.Errorf("Failed to get network devices: %v", err)
	}

	topology, numCores, err := GetTopology(sysFs)
	if err != nil {
		klog.Errorf("Failed to get topology information: %v", err)
	}

	systemUUID, err := sysinfo.GetSystemUUID(sysFs)
	if err != nil {
		klog.Errorf("Failed to get system UUID: %v", err)
	}

	realCloudInfo := cloudinfo.NewRealCloudInfo()
	cloudProvider := realCloudInfo.GetCloudProvider()
	instanceType := realCloudInfo.GetInstanceType()
	instanceID := realCloudInfo.GetInstanceID()

	machineInfo := &info.MachineInfo{
		Timestamp:        time.Now(),
		NumCores:         numCores,
		NumPhysicalCores: GetPhysicalCores(cpuinfo),
		NumSockets:       GetSockets(cpuinfo),
		CpuFrequency:     clockSpeed,
		MemoryCapacity:   memoryCapacity,
		MemoryByType:     memoryByType,
		NVMInfo:          nvmInfo,
		HugePages:        hugePagesInfo,
		DiskMap:          diskMap,
		NetworkDevices:   netDevices,
		Topology:         topology,
		MachineID:        getInfoFromFiles(filepath.Join(rootFs, *machineIDFilePath)),
		SystemUUID:       systemUUID,
		BootID:           getInfoFromFiles(filepath.Join(rootFs, *bootIDFilePath)),
		CloudProvider:    cloudProvider,
		InstanceType:     instanceType,
		InstanceID:       instanceID,
	}

	for i := range filesystems {
		fs := filesystems[i]
		inodes := uint64(0)
		if fs.Inodes != nil {
			inodes = *fs.Inodes
		}
		machineInfo.Filesystems = append(machineInfo.Filesystems, info.FsInfo{Device: fs.Device, DeviceMajor: uint64(fs.Major), DeviceMinor: uint64(fs.Minor), Type: fs.Type.String(), Capacity: fs.Capacity, Inodes: inodes, HasInodes: fs.Inodes != nil})
	}

	return machineInfo, nil
}

func ContainerOsVersion() string {
	os, err := getOperatingSystem()
	if err != nil {
		os = "Unknown"
	}
	return os
}

func KernelVersion() string {
	uname := &unix.Utsname{}

	if err := unix.Uname(uname); err != nil {
		return "Unknown"
	}

	return string(uname.Release[:bytes.IndexByte(uname.Release[:], 0)])
}
