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

// Handler for CRI-O containers.
package crio

import (
	"fmt"
	"path"
	"strconv"
	"strings"
	"time"

	"github.com/google/cadvisor/container"
	"github.com/google/cadvisor/container/common"
	containerlibcontainer "github.com/google/cadvisor/container/libcontainer"
	"github.com/google/cadvisor/fs"
	info "github.com/google/cadvisor/info/v1"

	"github.com/opencontainers/runc/libcontainer/cgroups"
	cgroupfs "github.com/opencontainers/runc/libcontainer/cgroups/fs"
	libcontainerconfigs "github.com/opencontainers/runc/libcontainer/configs"
)

type crioContainerHandler struct {
	name               string
	id                 string
	aliases            []string
	machineInfoFactory info.MachineInfoFactory

	// Absolute path to the cgroup hierarchies of this container.
	// (e.g.: "cpu" -> "/sys/fs/cgroup/cpu/test")
	cgroupPaths map[string]string

	// Manager of this container's cgroups.
	cgroupManager cgroups.Manager

	// the CRI-O storage driver
	storageDriver    storageDriver
	fsInfo           fs.FsInfo
	rootfsStorageDir string

	// Time at which this container was created.
	creationTime time.Time

	// Metadata associated with the container.
	labels map[string]string
	envs   map[string]string

	// TODO
	// crio version handling...

	// The container PID used to switch namespaces as required
	pid int

	// Image name used for this container.
	image string

	// The host root FS to read
	rootFs string

	// The network mode of the container
	// TODO

	// Filesystem handler.
	fsHandler common.FsHandler

	// The IP address of the container
	ipAddress string

	ignoreMetrics container.MetricSet

	// container restart count
	restartCount int
}

var _ container.ContainerHandler = &crioContainerHandler{}

// newCrioContainerHandler returns a new container.ContainerHandler
func newCrioContainerHandler(
	client crioClient,
	name string,
	machineInfoFactory info.MachineInfoFactory,
	fsInfo fs.FsInfo,
	storageDriver storageDriver,
	storageDir string,
	cgroupSubsystems *containerlibcontainer.CgroupSubsystems,
	inHostNamespace bool,
	metadataEnvs []string,
	ignoreMetrics container.MetricSet,
) (container.ContainerHandler, error) {
	// Create the cgroup paths.
	cgroupPaths := make(map[string]string, len(cgroupSubsystems.MountPoints))
	for key, val := range cgroupSubsystems.MountPoints {
		cgroupPaths[key] = path.Join(val, name)
	}

	// Generate the equivalent cgroup manager for this container.
	cgroupManager := &cgroupfs.Manager{
		Cgroups: &libcontainerconfigs.Cgroup{
			Name: name,
		},
		Paths: cgroupPaths,
	}

	rootFs := "/"
	if !inHostNamespace {
		rootFs = "/rootfs"
		storageDir = path.Join(rootFs, storageDir)
	}

	id := ContainerNameToCrioId(name)

	cInfo, err := client.ContainerInfo(id)
	if err != nil {
		return nil, err
	}

	// passed to fs handler below ...
	// XXX: this is using the full container logpath, as constructed by the CRI
	// /var/log/pods/<pod_uuid>/container_instance.log
	// It's not actually a log dir, as the CRI doesn't have per-container dirs
	// under /var/log/pods/<pod_uuid>/
	// We can't use /var/log/pods/<pod_uuid>/ to count per-container log usage.
	// We use the container log file directly.
	storageLogDir := cInfo.LogPath

	// Determine the rootfs storage dir
	rootfsStorageDir := cInfo.Root
	// TODO(runcom): CRI-O doesn't strip /merged but we need to in order to
	// get device ID from root, otherwise, it's going to error out as overlay
	// mounts doesn't have fixed dev ids.
	rootfsStorageDir = strings.TrimSuffix(rootfsStorageDir, "/merged")

	// TODO: extract object mother method
	handler := &crioContainerHandler{
		id:                 id,
		name:               name,
		machineInfoFactory: machineInfoFactory,
		cgroupPaths:        cgroupPaths,
		cgroupManager:      cgroupManager,
		storageDriver:      storageDriver,
		fsInfo:             fsInfo,
		rootFs:             rootFs,
		rootfsStorageDir:   rootfsStorageDir,
		envs:               make(map[string]string),
		ignoreMetrics:      ignoreMetrics,
	}

	handler.creationTime = time.Unix(0, cInfo.CreatedTime)
	handler.pid = cInfo.Pid
	handler.aliases = append(handler.aliases, cInfo.Name, id)
	handler.labels = cInfo.Labels
	handler.image = cInfo.Image
	// TODO: we wantd to know graph driver DeviceId (dont think this is needed now)

	// ignore err and get zero as default, this happens with sandboxes, not sure why...
	// kube isn't sending restart count in labels for sandboxes.
	restartCount, _ := strconv.Atoi(cInfo.Annotations["io.kubernetes.container.restartCount"])
	handler.restartCount = restartCount

	handler.ipAddress = cInfo.IP

	// we optionally collect disk usage metrics
	if !ignoreMetrics.Has(container.DiskUsageMetrics) {
		handler.fsHandler = common.NewFsHandler(common.DefaultPeriod, rootfsStorageDir, storageLogDir, fsInfo)
	}
	// TODO for env vars we wanted to show from container.Config.Env from whitelist
	//for _, exposedEnv := range metadataEnvs {
	//glog.Infof("TODO env whitelist: %v", exposedEnv)
	//}

	return handler, nil
}

func (self *crioContainerHandler) Start() {
	if self.fsHandler != nil {
		self.fsHandler.Start()
	}
}

func (self *crioContainerHandler) Cleanup() {
	if self.fsHandler != nil {
		self.fsHandler.Stop()
	}
}

func (self *crioContainerHandler) ContainerReference() (info.ContainerReference, error) {
	return info.ContainerReference{
		Id:        self.id,
		Name:      self.name,
		Aliases:   self.aliases,
		Namespace: CrioNamespace,
		Labels:    self.labels,
	}, nil
}

func (self *crioContainerHandler) needNet() bool {
	if !self.ignoreMetrics.Has(container.NetworkUsageMetrics) {
		return self.labels["io.kubernetes.container.name"] == "POD"
	}
	return false
}

func (self *crioContainerHandler) GetSpec() (info.ContainerSpec, error) {
	hasFilesystem := !self.ignoreMetrics.Has(container.DiskUsageMetrics)
	spec, err := common.GetSpec(self.cgroupPaths, self.machineInfoFactory, self.needNet(), hasFilesystem)

	spec.Labels = self.labels
	// Only adds restartcount label if it's greater than 0
	if self.restartCount > 0 {
		spec.Labels["restartcount"] = strconv.Itoa(self.restartCount)
	}
	spec.Envs = self.envs
	spec.Image = self.image

	return spec, err
}

func (self *crioContainerHandler) getFsStats(stats *info.ContainerStats) error {
	mi, err := self.machineInfoFactory.GetMachineInfo()
	if err != nil {
		return err
	}

	if !self.ignoreMetrics.Has(container.DiskIOMetrics) {
		common.AssignDeviceNamesToDiskStats((*common.MachineInfoNamer)(mi), &stats.DiskIo)
	}

	if self.ignoreMetrics.Has(container.DiskUsageMetrics) {
		return nil
	}
	var device string
	switch self.storageDriver {
	case overlay2StorageDriver, overlayStorageDriver:
		deviceInfo, err := self.fsInfo.GetDirFsDevice(self.rootfsStorageDir)
		if err != nil {
			return fmt.Errorf("unable to determine device info for dir: %v: %v", self.rootfsStorageDir, err)
		}
		device = deviceInfo.Device
	default:
		return nil
	}

	var (
		limit  uint64
		fsType string
	)

	// crio does not impose any filesystem limits for containers. So use capacity as limit.
	for _, fs := range mi.Filesystems {
		if fs.Device == device {
			limit = fs.Capacity
			fsType = fs.Type
			break
		}
	}

	fsStat := info.FsStats{Device: device, Type: fsType, Limit: limit}
	usage := self.fsHandler.Usage()
	fsStat.BaseUsage = usage.BaseUsageBytes
	fsStat.Usage = usage.TotalUsageBytes
	fsStat.Inodes = usage.InodeUsage

	stats.Filesystem = append(stats.Filesystem, fsStat)

	return nil
}

func (self *crioContainerHandler) GetStats() (*info.ContainerStats, error) {
	stats, err := containerlibcontainer.GetStats(self.cgroupManager, self.rootFs, self.pid, self.ignoreMetrics)
	if err != nil {
		return stats, err
	}
	// Clean up stats for containers that don't have their own network - this
	// includes containers running in Kubernetes pods that use the network of the
	// infrastructure container. This stops metrics being reported multiple times
	// for each container in a pod.
	if !self.needNet() {
		stats.Network = info.NetworkStats{}
	}

	// Get filesystem stats.
	err = self.getFsStats(stats)
	if err != nil {
		return stats, err
	}

	return stats, nil
}

func (self *crioContainerHandler) ListContainers(listType container.ListType) ([]info.ContainerReference, error) {
	// No-op for Docker driver.
	return []info.ContainerReference{}, nil
}

func (self *crioContainerHandler) GetCgroupPath(resource string) (string, error) {
	path, ok := self.cgroupPaths[resource]
	if !ok {
		return "", fmt.Errorf("could not find path for resource %q for container %q\n", resource, self.name)
	}
	return path, nil
}

func (self *crioContainerHandler) GetContainerLabels() map[string]string {
	return self.labels
}

func (self *crioContainerHandler) GetContainerIPAddress() string {
	return self.ipAddress
}

func (self *crioContainerHandler) ListProcesses(listType container.ListType) ([]int, error) {
	return containerlibcontainer.GetProcesses(self.cgroupManager)
}

func (self *crioContainerHandler) Exists() bool {
	return common.CgroupExists(self.cgroupPaths)
}

func (self *crioContainerHandler) Type() container.ContainerType {
	return container.ContainerTypeCrio
}
