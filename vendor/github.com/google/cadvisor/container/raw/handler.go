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

//go:build linux

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

	"github.com/opencontainers/cgroups"
	"k8s.io/klog/v2"
)

type rawContainerHandler struct {
	// Name of the container for this handler.
	name               string
	machineInfoFactory info.MachineInfoFactory

	// Absolute path to the cgroup hierarchies of this container.
	// (e.g.: "cpu" -> "/sys/fs/cgroup/cpu/test")
	cgroupPaths map[string]string

	fsInfo          fs.FsInfo
	externalMounts  []common.Mount
	includedMetrics container.MetricSet

	libcontainerHandler *libcontainer.Handler
}

func isRootCgroup(name string) bool {
	return name == "/"
}

func newRawContainerHandler(name string, cgroupSubsystems map[string]string, machineInfoFactory info.MachineInfoFactory, fsInfo fs.FsInfo, watcher *common.InotifyWatcher, rootFs string, includedMetrics container.MetricSet) (container.ContainerHandler, error) {
	cHints, err := common.GetContainerHintsFromFile(*common.ArgContainerHints)
	if err != nil {
		return nil, err
	}

	cgroupPaths := common.MakeCgroupPaths(cgroupSubsystems, name)

	cgroupManager, err := libcontainer.NewCgroupManager(name, cgroupPaths)
	if err != nil {
		return nil, err
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

		// delete pids from cgroup paths because /sys/fs/cgroup/pids/pids.current not exist
		delete(cgroupPaths, "pids")
	}

	handler := libcontainer.NewHandler(cgroupManager, rootFs, pid, includedMetrics)

	return &rawContainerHandler{
		name:                name,
		machineInfoFactory:  machineInfoFactory,
		cgroupPaths:         cgroupPaths,
		fsInfo:              fsInfo,
		externalMounts:      externalMounts,
		includedMetrics:     includedMetrics,
		libcontainerHandler: handler,
	}, nil
}

func (h *rawContainerHandler) ContainerReference() (info.ContainerReference, error) {
	// We only know the container by its one name.
	return info.ContainerReference{
		Name: h.name,
	}, nil
}

func (h *rawContainerHandler) GetRootNetworkDevices() ([]info.NetInfo, error) {
	nd := []info.NetInfo{}
	if isRootCgroup(h.name) {
		mi, err := h.machineInfoFactory.GetMachineInfo()
		if err != nil {
			return nd, err
		}
		return mi.NetworkDevices, nil
	}
	return nd, nil
}

// Nothing to start up.
func (h *rawContainerHandler) Start() {}

// Nothing to clean up.
func (h *rawContainerHandler) Cleanup() {}

func (h *rawContainerHandler) GetSpec() (info.ContainerSpec, error) {
	const hasNetwork = false
	hasFilesystem := isRootCgroup(h.name) || len(h.externalMounts) > 0
	spec, err := common.GetSpec(h.cgroupPaths, h.machineInfoFactory, hasNetwork, hasFilesystem)
	if err != nil {
		return spec, err
	}

	if isRootCgroup(h.name) {
		// Check physical network devices for root container.
		nd, err := h.GetRootNetworkDevices()
		if err != nil {
			return spec, err
		}
		spec.HasNetwork = spec.HasNetwork || len(nd) != 0

		// Get memory and swap limits of the running machine
		memLimit, err := machine.GetMachineMemoryCapacity()
		if err != nil {
			klog.Warningf("failed to obtain memory limit for machine container")
			spec.HasMemory = false
		} else {
			spec.Memory.Limit = uint64(memLimit)
			// Spec is marked to have memory only if the memory limit is set
			spec.HasMemory = true
		}

		swapLimit, err := machine.GetMachineSwapCapacity()
		if err != nil {
			klog.Warningf("failed to obtain swap limit for machine container")
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

func (h *rawContainerHandler) getFsStats(stats *info.ContainerStats) error {
	var filesystems []fs.Fs
	var err error
	// Early exist if no disk metrics are to be collected.
	if !h.includedMetrics.Has(container.DiskUsageMetrics) && !h.includedMetrics.Has(container.DiskIOMetrics) {
		return nil
	}

	// Get Filesystem information only for the root cgroup.
	if isRootCgroup(h.name) {
		filesystems, err = h.fsInfo.GetGlobalFsInfo()
		if err != nil {
			return err
		}
	} else {
		if len(h.externalMounts) > 0 {
			mountSet := make(map[string]struct{})
			for _, mount := range h.externalMounts {
				mountSet[mount.HostDir] = struct{}{}
			}
			filesystems, err = h.fsInfo.GetFsInfoForPath(mountSet)
			if err != nil {
				return err
			}
		}
	}

	if h.includedMetrics.Has(container.DiskUsageMetrics) {
		for i := range filesystems {
			fs := filesystems[i]
			stats.Filesystem = append(stats.Filesystem, fsToFsStats(&fs))
		}
	}

	if h.includedMetrics.Has(container.DiskIOMetrics) {
		common.AssignDeviceNamesToDiskStats(&fsNamer{fs: filesystems, factory: h.machineInfoFactory}, &stats.DiskIo)

	}
	return nil
}

func (h *rawContainerHandler) GetStats() (*info.ContainerStats, error) {
	if *disableRootCgroupStats && isRootCgroup(h.name) {
		return nil, nil
	}
	stats, err := h.libcontainerHandler.GetStats()
	if err != nil {
		return stats, err
	}

	// Get filesystem stats.
	err = h.getFsStats(stats)
	if err != nil {
		return stats, err
	}

	return stats, nil
}

func (h *rawContainerHandler) GetCgroupPath(resource string) (string, error) {
	var res string
	if !cgroups.IsCgroup2UnifiedMode() {
		res = resource
	}
	path, ok := h.cgroupPaths[res]
	if !ok {
		return "", fmt.Errorf("could not find path for resource %q for container %q", resource, h.name)
	}
	return path, nil
}

func (h *rawContainerHandler) GetContainerLabels() map[string]string {
	return map[string]string{}
}

func (h *rawContainerHandler) GetContainerIPAddress() string {
	// the IP address for the raw container corresponds to the system ip address.
	return "127.0.0.1"
}

func (h *rawContainerHandler) ListContainers(listType container.ListType) ([]info.ContainerReference, error) {
	return common.ListContainers(h.name, h.cgroupPaths, listType)
}

func (h *rawContainerHandler) ListProcesses(listType container.ListType) ([]int, error) {
	return h.libcontainerHandler.GetProcesses()
}

func (h *rawContainerHandler) Exists() bool {
	return common.CgroupExists(h.cgroupPaths)
}

func (h *rawContainerHandler) Type() container.ContainerType {
	return container.ContainerTypeRaw
}

func (h *rawContainerHandler) GetExitCode() (int, error) {
	return -1, fmt.Errorf("exit codes not applicable for raw cgroup containers")
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
