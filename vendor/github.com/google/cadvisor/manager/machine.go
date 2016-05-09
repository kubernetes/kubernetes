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

package manager

import (
	"bytes"
	"flag"
	"io/ioutil"
	"path/filepath"
	"strings"
	"syscall"

	"github.com/google/cadvisor/container/docker"
	"github.com/google/cadvisor/fs"
	info "github.com/google/cadvisor/info/v1"
	"github.com/google/cadvisor/utils/cloudinfo"
	"github.com/google/cadvisor/utils/machine"
	"github.com/google/cadvisor/utils/sysfs"
	"github.com/google/cadvisor/utils/sysinfo"
	version "github.com/google/cadvisor/version"

	"github.com/golang/glog"
)

var machineIdFilePath = flag.String("machine_id_file", "/etc/machine-id,/var/lib/dbus/machine-id", "Comma-separated list of files to check for machine-id. Use the first one that exists.")
var bootIdFilePath = flag.String("boot_id_file", "/proc/sys/kernel/random/boot_id", "Comma-separated list of files to check for boot-id. Use the first one that exists.")

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
	glog.Infof("Couldn't collect info from any of the files in %q", filePaths)
	return ""
}

func getMachineInfo(sysFs sysfs.SysFs, fsInfo fs.FsInfo, inHostNamespace bool) (*info.MachineInfo, error) {
	rootFs := "/"
	if !inHostNamespace {
		rootFs = "/rootfs"
	}

	cpuinfo, err := ioutil.ReadFile(filepath.Join(rootFs, "/proc/cpuinfo"))
	clockSpeed, err := machine.GetClockSpeed(cpuinfo)
	if err != nil {
		return nil, err
	}

	memoryCapacity, err := machine.GetMachineMemoryCapacity()
	if err != nil {
		return nil, err
	}

	filesystems, err := fsInfo.GetGlobalFsInfo()
	if err != nil {
		glog.Errorf("Failed to get global filesystem information: %v", err)
	}

	diskMap, err := sysinfo.GetBlockDeviceInfo(sysFs)
	if err != nil {
		glog.Errorf("Failed to get disk map: %v", err)
	}

	netDevices, err := sysinfo.GetNetworkDevices(sysFs)
	if err != nil {
		glog.Errorf("Failed to get network devices: %v", err)
	}

	topology, numCores, err := machine.GetTopology(sysFs, string(cpuinfo))
	if err != nil {
		glog.Errorf("Failed to get topology information: %v", err)
	}

	systemUUID, err := sysinfo.GetSystemUUID(sysFs)
	if err != nil {
		glog.Errorf("Failed to get system UUID: %v", err)
	}

	realCloudInfo := cloudinfo.NewRealCloudInfo()
	cloudProvider := realCloudInfo.GetCloudProvider()
	instanceType := realCloudInfo.GetInstanceType()
	instanceID := realCloudInfo.GetInstanceID()

	machineInfo := &info.MachineInfo{
		NumCores:       numCores,
		CpuFrequency:   clockSpeed,
		MemoryCapacity: memoryCapacity,
		DiskMap:        diskMap,
		NetworkDevices: netDevices,
		Topology:       topology,
		MachineID:      getInfoFromFiles(filepath.Join(rootFs, *machineIdFilePath)),
		SystemUUID:     systemUUID,
		BootID:         getInfoFromFiles(filepath.Join(rootFs, *bootIdFilePath)),
		CloudProvider:  cloudProvider,
		InstanceType:   instanceType,
		InstanceID:     instanceID,
	}

	for _, fs := range filesystems {
		machineInfo.Filesystems = append(machineInfo.Filesystems, info.FsInfo{Device: fs.Device, Type: fs.Type.String(), Capacity: fs.Capacity, Inodes: fs.Inodes})
	}

	return machineInfo, nil
}

func getVersionInfo() (*info.VersionInfo, error) {

	kernel_version := getKernelVersion()
	container_os := getContainerOsVersion()
	docker_version := getDockerVersion()

	return &info.VersionInfo{
		KernelVersion:      kernel_version,
		ContainerOsVersion: container_os,
		DockerVersion:      docker_version,
		CadvisorVersion:    version.Info["version"],
		CadvisorRevision:   version.Info["revision"],
	}, nil
}

func getContainerOsVersion() string {
	container_os := "Unknown"
	os_release, err := ioutil.ReadFile("/etc/os-release")
	if err == nil {
		// We might be running in a busybox or some hand-crafted image.
		// It's useful to know why cadvisor didn't come up.
		for _, line := range strings.Split(string(os_release), "\n") {
			parsed := strings.Split(line, "\"")
			if len(parsed) == 3 && parsed[0] == "PRETTY_NAME=" {
				container_os = parsed[1]
				break
			}
		}
	}
	return container_os
}

func getDockerVersion() string {
	docker_version := "Unknown"
	client, err := docker.Client()
	if err == nil {
		version, err := client.Version()
		if err == nil {
			docker_version = version.Get("Version")
		}
	}
	return docker_version
}

func getKernelVersion() string {
	uname := &syscall.Utsname{}

	if err := syscall.Uname(uname); err != nil {
		return "Unknown"
	}

	release := make([]byte, len(uname.Release))
	i := 0
	for _, c := range uname.Release {
		release[i] = byte(c)
		i++
	}
	release = release[:bytes.IndexByte(release, 0)]

	return string(release)
}
