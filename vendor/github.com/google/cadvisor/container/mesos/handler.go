// Copyright 2018 Google Inc. All Rights Reserved.
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

// Handler for "mesos" containers.
package mesos

import (
	"fmt"
	"path"

	"github.com/google/cadvisor/container"
	"github.com/google/cadvisor/container/common"
	containerlibcontainer "github.com/google/cadvisor/container/libcontainer"
	"github.com/google/cadvisor/fs"
	info "github.com/google/cadvisor/info/v1"

	cgroupfs "github.com/opencontainers/runc/libcontainer/cgroups/fs"
	libcontainerconfigs "github.com/opencontainers/runc/libcontainer/configs"
)

type mesosContainerHandler struct {
	// Name of the container for this handler.
	name string

	// machineInfoFactory provides info.MachineInfo
	machineInfoFactory info.MachineInfoFactory

	// Absolute path to the cgroup hierarchies of this container.
	// (e.g.: "cpu" -> "/sys/fs/cgroup/cpu/test")
	cgroupPaths map[string]string

	// File System Info
	fsInfo fs.FsInfo

	// Metrics to be included.
	includedMetrics container.MetricSet

	labels map[string]string

	// Reference to the container
	reference info.ContainerReference

	libcontainerHandler *containerlibcontainer.Handler
}

func isRootCgroup(name string) bool {
	return name == "/"
}

func newMesosContainerHandler(
	name string,
	cgroupSubsystems *containerlibcontainer.CgroupSubsystems,
	machineInfoFactory info.MachineInfoFactory,
	fsInfo fs.FsInfo,
	includedMetrics container.MetricSet,
	inHostNamespace bool,
	client mesosAgentClient,
) (container.ContainerHandler, error) {
	cgroupPaths := common.MakeCgroupPaths(cgroupSubsystems.MountPoints, name)
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
	}

	id := ContainerNameToMesosId(name)

	cinfo, err := client.ContainerInfo(id)

	if err != nil {
		return nil, err
	}

	labels := cinfo.labels
	pid, err := client.ContainerPid(id)
	if err != nil {
		return nil, err
	}

	libcontainerHandler := containerlibcontainer.NewHandler(cgroupManager, rootFs, pid, includedMetrics)

	reference := info.ContainerReference{
		Id:        id,
		Name:      name,
		Namespace: MesosNamespace,
		Aliases:   []string{id, name},
	}

	handler := &mesosContainerHandler{
		name:                name,
		machineInfoFactory:  machineInfoFactory,
		cgroupPaths:         cgroupPaths,
		fsInfo:              fsInfo,
		includedMetrics:     includedMetrics,
		labels:              labels,
		reference:           reference,
		libcontainerHandler: libcontainerHandler,
	}

	return handler, nil
}

func (self *mesosContainerHandler) ContainerReference() (info.ContainerReference, error) {
	// We only know the container by its one name.
	return self.reference, nil
}

// Nothing to start up.
func (self *mesosContainerHandler) Start() {}

// Nothing to clean up.
func (self *mesosContainerHandler) Cleanup() {}

func (self *mesosContainerHandler) GetSpec() (info.ContainerSpec, error) {
	// TODO: Since we dont collect disk usage and network stats for mesos containers, we set
	// hasFilesystem and hasNetwork to false. Revisit when we support disk usage, network
	// stats for mesos containers.
	hasNetwork := false
	hasFilesystem := false

	spec, err := common.GetSpec(self.cgroupPaths, self.machineInfoFactory, hasNetwork, hasFilesystem)
	if err != nil {
		return spec, err
	}

	spec.Labels = self.labels

	return spec, nil
}

func (self *mesosContainerHandler) getFsStats(stats *info.ContainerStats) error {

	mi, err := self.machineInfoFactory.GetMachineInfo()
	if err != nil {
		return err
	}

	if self.includedMetrics.Has(container.DiskIOMetrics) {
		common.AssignDeviceNamesToDiskStats((*common.MachineInfoNamer)(mi), &stats.DiskIo)
	}

	return nil
}

func (self *mesosContainerHandler) GetStats() (*info.ContainerStats, error) {
	stats, err := self.libcontainerHandler.GetStats()
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

func (self *mesosContainerHandler) GetCgroupPath(resource string) (string, error) {
	path, ok := self.cgroupPaths[resource]
	if !ok {
		return "", fmt.Errorf("could not find path for resource %q for container %q\n", resource, self.name)
	}
	return path, nil
}

func (self *mesosContainerHandler) GetContainerLabels() map[string]string {
	return self.labels
}

func (self *mesosContainerHandler) GetContainerIPAddress() string {
	// the IP address for the mesos container corresponds to the system ip address.
	return "127.0.0.1"
}

func (self *mesosContainerHandler) ListContainers(listType container.ListType) ([]info.ContainerReference, error) {
	return common.ListContainers(self.name, self.cgroupPaths, listType)
}

func (self *mesosContainerHandler) ListProcesses(listType container.ListType) ([]int, error) {
	return self.libcontainerHandler.GetProcesses()
}

func (self *mesosContainerHandler) Exists() bool {
	return common.CgroupExists(self.cgroupPaths)
}

func (self *mesosContainerHandler) Type() container.ContainerType {
	return container.ContainerTypeMesos
}
