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

// Handler for "raw" containers.
package raw

import (
	"fmt"

	"github.com/google/cadvisor/container"
	"github.com/google/cadvisor/container/common"
	"github.com/google/cadvisor/container/libcontainer"
	"github.com/google/cadvisor/fs"
	info "github.com/google/cadvisor/info/v1"
	"github.com/google/cadvisor/machine"

	"github.com/golang/glog"
	"github.com/opencontainers/runc/libcontainer/cgroups"
	cgroupfs "github.com/opencontainers/runc/libcontainer/cgroups/fs"
	"github.com/opencontainers/runc/libcontainer/configs"
)

type rawContainerHandler struct {
	// Name of the container for this handler.
	name               string
	cgroupSubsystems   *libcontainer.CgroupSubsystems
	machineInfoFactory info.MachineInfoFactory

	// Absolute path to the cgroup hierarchies of this container.
	// (e.g.: "cpu" -> "/sys/fs/cgroup/cpu/test")
	cgroupPaths map[string]string

	// Manager of this container's cgroups.
	cgroupManager cgroups.Manager

	fsInfo         fs.FsInfo
	externalMounts []common.Mount

	rootFs string

	// Metrics to be ignored.
	ignoreMetrics container.MetricSet

	pid int
}

func isRootCgroup(name string) bool {
	return name == "/"
}

func newRawContainerHandler(name string, cgroupSubsystems *libcontainer.CgroupSubsystems, machineInfoFactory info.MachineInfoFactory, fsInfo fs.FsInfo, watcher *common.InotifyWatcher, rootFs string, ignoreMetrics container.MetricSet) (container.ContainerHandler, error) {
	cgroupPaths := common.MakeCgroupPaths(cgroupSubsystems.MountPoints, name)

	cHints, err := common.GetContainerHintsFromFile(*common.ArgContainerHints)
	if err != nil {
		return nil, err
	}

	// Generate the equivalent cgroup manager for this container.
	cgroupManager := &cgroupfs.Manager{
		Cgroups: &configs.Cgroup{
			Name: name,
		},
		Paths: cgroupPaths,
	}

	var externalMounts []common.Mount
	for _, container := range cHints.AllHosts {
		if name == container.FullName {
			externalMounts = container.Mounts
			break
		}
	}

	pid := 0
	if isRootCgroup(name) {
		pid = 1
	}

	return &rawContainerHandler{
		name:               name,
		cgroupSubsystems:   cgroupSubsystems,
		machineInfoFactory: machineInfoFactory,
		cgroupPaths:        cgroupPaths,
		cgroupManager:      cgroupManager,
		fsInfo:             fsInfo,
		externalMounts:     externalMounts,
		rootFs:             rootFs,
		ignoreMetrics:      ignoreMetrics,
		pid:                pid,
	}, nil
}

func (self *rawContainerHandler) ContainerReference() (info.ContainerReference, error) {
	// We only know the container by its one name.
	return info.ContainerReference{
		Name: self.name,
	}, nil
}

func (self *rawContainerHandler) GetRootNetworkDevices() ([]info.NetInfo, error) {
	nd := []info.NetInfo{}
	if isRootCgroup(self.name) {
		mi, err := self.machineInfoFactory.GetMachineInfo()
		if err != nil {
			return nd, err
		}
		return mi.NetworkDevices, nil
	}
	return nd, nil
}

// Nothing to start up.
func (self *rawContainerHandler) Start() {}

// Nothing to clean up.
func (self *rawContainerHandler) Cleanup() {}

func (self *rawContainerHandler) GetSpec() (info.ContainerSpec, error) {
	const hasNetwork = false
	hasFilesystem := isRootCgroup(self.name) || len(self.externalMounts) > 0
	spec, err := common.GetSpec(self.cgroupPaths, self.machineInfoFactory, hasNetwork, hasFilesystem)
	if err != nil {
		return spec, err
	}

	if isRootCgroup(self.name) {
		// Check physical network devices for root container.
		nd, err := self.GetRootNetworkDevices()
		if err != nil {
			return spec, err
		}
		spec.HasNetwork = spec.HasNetwork || len(nd) != 0

		// Get memory and swap limits of the running machine
		memLimit, err := machine.GetMachineMemoryCapacity()
		if err != nil {
			glog.Warningf("failed to obtain memory limit for machine container")
			spec.HasMemory = false
		} else {
			spec.Memory.Limit = uint64(memLimit)
			// Spec is marked to have memory only if the memory limit is set
			spec.HasMemory = true
		}

		swapLimit, err := machine.GetMachineSwapCapacity()
		if err != nil {
			glog.Warningf("failed to obtain swap limit for machine container")
		} else {
			spec.Memory.SwapLimit = uint64(swapLimit)
		}
	}

	return spec, nil
}

func fsToFsStats(fs *fs.Fs) info.FsStats {
	inodes := uint64(0)
	inodesFree := uint64(0)
	hasInodes := fs.InodesFree != nil
	if hasInodes {
		inodes = *fs.Inodes
		inodesFree = *fs.InodesFree
	}
	return info.FsStats{
		Device:          fs.Device,
		Type:            fs.Type.String(),
		Limit:           fs.Capacity,
		Usage:           fs.Capacity - fs.Free,
		HasInodes:       hasInodes,
		Inodes:          inodes,
		InodesFree:      inodesFree,
		Available:       fs.Available,
		ReadsCompleted:  fs.DiskStats.ReadsCompleted,
		ReadsMerged:     fs.DiskStats.ReadsMerged,
		SectorsRead:     fs.DiskStats.SectorsRead,
		ReadTime:        fs.DiskStats.ReadTime,
		WritesCompleted: fs.DiskStats.WritesCompleted,
		WritesMerged:    fs.DiskStats.WritesMerged,
		SectorsWritten:  fs.DiskStats.SectorsWritten,
		WriteTime:       fs.DiskStats.WriteTime,
		IoInProgress:    fs.DiskStats.IoInProgress,
		IoTime:          fs.DiskStats.IoTime,
		WeightedIoTime:  fs.DiskStats.WeightedIoTime,
	}
}

func (self *rawContainerHandler) getFsStats(stats *info.ContainerStats) error {
	var allFs []fs.Fs
	// Get Filesystem information only for the root cgroup.
	if isRootCgroup(self.name) {
		filesystems, err := self.fsInfo.GetGlobalFsInfo()
		if err != nil {
			return err
		}
		for i := range filesystems {
			fs := filesystems[i]
			stats.Filesystem = append(stats.Filesystem, fsToFsStats(&fs))
		}
		allFs = filesystems
	} else if len(self.externalMounts) > 0 {
		var mountSet map[string]struct{}
		mountSet = make(map[string]struct{})
		for _, mount := range self.externalMounts {
			mountSet[mount.HostDir] = struct{}{}
		}
		filesystems, err := self.fsInfo.GetFsInfoForPath(mountSet)
		if err != nil {
			return err
		}
		for i := range filesystems {
			fs := filesystems[i]
			stats.Filesystem = append(stats.Filesystem, fsToFsStats(&fs))
		}
		allFs = filesystems
	}

	common.AssignDeviceNamesToDiskStats(&fsNamer{fs: allFs, factory: self.machineInfoFactory}, &stats.DiskIo)
	return nil
}

func (self *rawContainerHandler) GetStats() (*info.ContainerStats, error) {
	stats, err := libcontainer.GetStats(self.cgroupManager, self.rootFs, self.pid, self.ignoreMetrics)
	if err != nil {
		return stats, err
	}

	// Get filesystem stats.
	err = self.getFsStats(stats)
	if err != nil {
		return stats, err
	}

	return stats, nil
}

func (self *rawContainerHandler) GetCgroupPath(resource string) (string, error) {
	path, ok := self.cgroupPaths[resource]
	if !ok {
		return "", fmt.Errorf("could not find path for resource %q for container %q\n", resource, self.name)
	}
	return path, nil
}

func (self *rawContainerHandler) GetContainerLabels() map[string]string {
	return map[string]string{}
}

func (self *rawContainerHandler) GetContainerIPAddress() string {
	// the IP address for the raw container corresponds to the system ip address.
	return "127.0.0.1"
}

func (self *rawContainerHandler) ListContainers(listType container.ListType) ([]info.ContainerReference, error) {
	return common.ListContainers(self.name, self.cgroupPaths, listType)
}

func (self *rawContainerHandler) ListProcesses(listType container.ListType) ([]int, error) {
	return libcontainer.GetProcesses(self.cgroupManager)
}

func (self *rawContainerHandler) Exists() bool {
	return common.CgroupExists(self.cgroupPaths)
}

func (self *rawContainerHandler) Type() container.ContainerType {
	return container.ContainerTypeRaw
}

type fsNamer struct {
	fs      []fs.Fs
	factory info.MachineInfoFactory
	info    common.DeviceNamer
}

func (n *fsNamer) DeviceName(major, minor uint64) (string, bool) {
	for _, info := range n.fs {
		if uint64(info.Major) == major && uint64(info.Minor) == minor {
			return info.Device, true
		}
	}
	if n.info == nil {
		mi, err := n.factory.GetMachineInfo()
		if err != nil {
			return "", false
		}
		n.info = (*common.MachineInfoNamer)(mi)
	}
	return n.info.DeviceName(major, minor)
}
