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
	"io/ioutil"
	"path"
	"strings"

	"github.com/google/cadvisor/container"
	"github.com/google/cadvisor/container/common"
	"github.com/google/cadvisor/container/libcontainer"
	"github.com/google/cadvisor/fs"
	info "github.com/google/cadvisor/info/v1"
	"github.com/google/cadvisor/utils/machine"

	"github.com/golang/glog"
	"github.com/opencontainers/runc/libcontainer/cgroups"
	cgroupfs "github.com/opencontainers/runc/libcontainer/cgroups/fs"
	"github.com/opencontainers/runc/libcontainer/configs"
	"golang.org/x/exp/inotify"
)

type rawContainerHandler struct {
	// Name of the container for this handler.
	name               string
	cgroupSubsystems   *libcontainer.CgroupSubsystems
	machineInfoFactory info.MachineInfoFactory

	// Inotify event watcher.
	watcher *common.InotifyWatcher

	// Signal for watcher thread to stop.
	stopWatcher chan error

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
		stopWatcher:        make(chan error),
		cgroupPaths:        cgroupPaths,
		cgroupManager:      cgroupManager,
		fsInfo:             fsInfo,
		externalMounts:     externalMounts,
		watcher:            watcher,
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

func (self *rawContainerHandler) getFsStats(stats *info.ContainerStats) error {
	// Get Filesystem information only for the root cgroup.
	if isRootCgroup(self.name) {
		filesystems, err := self.fsInfo.GetGlobalFsInfo()
		if err != nil {
			return err
		}
		for _, fs := range filesystems {
			stats.Filesystem = append(stats.Filesystem,
				info.FsStats{
					Device:          fs.Device,
					Type:            fs.Type.String(),
					Limit:           fs.Capacity,
					Usage:           fs.Capacity - fs.Free,
					Available:       fs.Available,
					InodesFree:      fs.InodesFree,
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
				})
		}
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
		for _, fs := range filesystems {
			stats.Filesystem = append(stats.Filesystem,
				info.FsStats{
					Device:          fs.Device,
					Type:            fs.Type.String(),
					Limit:           fs.Capacity,
					Usage:           fs.Capacity - fs.Free,
					InodesFree:      fs.InodesFree,
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
				})
		}
	}
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

func (self *rawContainerHandler) ListContainers(listType container.ListType) ([]info.ContainerReference, error) {
	containers := make(map[string]struct{})
	for _, cgroupPath := range self.cgroupPaths {
		err := common.ListDirectories(cgroupPath, self.name, listType == container.ListRecursive, containers)
		if err != nil {
			return nil, err
		}
	}

	// Make into container references.
	ret := make([]info.ContainerReference, 0, len(containers))
	for cont := range containers {
		ret = append(ret, info.ContainerReference{
			Name: cont,
		})
	}

	return ret, nil
}

func (self *rawContainerHandler) ListThreads(listType container.ListType) ([]int, error) {
	// TODO(vmarmol): Implement
	return nil, nil
}

func (self *rawContainerHandler) ListProcesses(listType container.ListType) ([]int, error) {
	return libcontainer.GetProcesses(self.cgroupManager)
}

// Watches the specified directory and all subdirectories. Returns whether the path was
// already being watched and an error (if any).
func (self *rawContainerHandler) watchDirectory(dir string, containerName string) (bool, error) {
	alreadyWatching, err := self.watcher.AddWatch(containerName, dir)
	if err != nil {
		return alreadyWatching, err
	}

	// Remove the watch if further operations failed.
	cleanup := true
	defer func() {
		if cleanup {
			_, err := self.watcher.RemoveWatch(containerName, dir)
			if err != nil {
				glog.Warningf("Failed to remove inotify watch for %q: %v", dir, err)
			}
		}
	}()

	// TODO(vmarmol): We should re-do this once we're done to ensure directories were not added in the meantime.
	// Watch subdirectories as well.
	entries, err := ioutil.ReadDir(dir)
	if err != nil {
		return alreadyWatching, err
	}
	for _, entry := range entries {
		if entry.IsDir() {
			// TODO(vmarmol): We don't have to fail here, maybe we can recover and try to get as many registrations as we can.
			_, err = self.watchDirectory(path.Join(dir, entry.Name()), path.Join(containerName, entry.Name()))
			if err != nil {
				return alreadyWatching, err
			}
		}
	}

	cleanup = false
	return alreadyWatching, nil
}

func (self *rawContainerHandler) processEvent(event *inotify.Event, events chan container.SubcontainerEvent) error {
	// Convert the inotify event type to a container create or delete.
	var eventType container.SubcontainerEventType
	switch {
	case (event.Mask & inotify.IN_CREATE) > 0:
		eventType = container.SubcontainerAdd
	case (event.Mask & inotify.IN_DELETE) > 0:
		eventType = container.SubcontainerDelete
	case (event.Mask & inotify.IN_MOVED_FROM) > 0:
		eventType = container.SubcontainerDelete
	case (event.Mask & inotify.IN_MOVED_TO) > 0:
		eventType = container.SubcontainerAdd
	default:
		// Ignore other events.
		return nil
	}

	// Derive the container name from the path name.
	var containerName string
	for _, mount := range self.cgroupSubsystems.Mounts {
		mountLocation := path.Clean(mount.Mountpoint) + "/"
		if strings.HasPrefix(event.Name, mountLocation) {
			containerName = event.Name[len(mountLocation)-1:]
			break
		}
	}
	if containerName == "" {
		return fmt.Errorf("unable to detect container from watch event on directory %q", event.Name)
	}

	// Maintain the watch for the new or deleted container.
	switch {
	case eventType == container.SubcontainerAdd:
		// New container was created, watch it.
		alreadyWatched, err := self.watchDirectory(event.Name, containerName)
		if err != nil {
			return err
		}

		// Only report container creation once.
		if alreadyWatched {
			return nil
		}
	case eventType == container.SubcontainerDelete:
		// Container was deleted, stop watching for it.
		lastWatched, err := self.watcher.RemoveWatch(containerName, event.Name)
		if err != nil {
			return err
		}

		// Only report container deletion once.
		if !lastWatched {
			return nil
		}
	default:
		return fmt.Errorf("unknown event type %v", eventType)
	}

	// Deliver the event.
	events <- container.SubcontainerEvent{
		EventType: eventType,
		Name:      containerName,
	}

	return nil
}

func (self *rawContainerHandler) WatchSubcontainers(events chan container.SubcontainerEvent) error {
	// Watch this container (all its cgroups) and all subdirectories.
	for _, cgroupPath := range self.cgroupPaths {
		_, err := self.watchDirectory(cgroupPath, self.name)
		if err != nil {
			return err
		}
	}

	// Process the events received from the kernel.
	go func() {
		for {
			select {
			case event := <-self.watcher.Event():
				err := self.processEvent(event, events)
				if err != nil {
					glog.Warningf("Error while processing event (%+v): %v", event, err)
				}
			case err := <-self.watcher.Error():
				glog.Warningf("Error while watching %q:", self.name, err)
			case <-self.stopWatcher:
				err := self.watcher.Close()
				if err == nil {
					self.stopWatcher <- err
					return
				}
			}
		}
	}()

	return nil
}

func (self *rawContainerHandler) StopWatchingSubcontainers() error {
	// Rendezvous with the watcher thread.
	self.stopWatcher <- nil
	return <-self.stopWatcher
}

func (self *rawContainerHandler) Exists() bool {
	return common.CgroupExists(self.cgroupPaths)
}
