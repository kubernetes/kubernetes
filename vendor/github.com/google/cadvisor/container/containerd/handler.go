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

// Handler for containerd containers.
package containerd

import (
	"encoding/json"
	"fmt"
	"path"
	"strings"
	"time"

	"github.com/opencontainers/runc/libcontainer/cgroups"
	cgroupfs "github.com/opencontainers/runc/libcontainer/cgroups/fs"
	libcontainerconfigs "github.com/opencontainers/runc/libcontainer/configs"
	"golang.org/x/net/context"

	"github.com/google/cadvisor/container"
	"github.com/google/cadvisor/container/common"
	containerlibcontainer "github.com/google/cadvisor/container/libcontainer"
	"github.com/google/cadvisor/fs"
	info "github.com/google/cadvisor/info/v1"
	specs "github.com/opencontainers/runtime-spec/specs-go"
)

type containerdContainerHandler struct {
	client             containerdClient
	name               string
	id                 string
	aliases            []string
	machineInfoFactory info.MachineInfoFactory
	// Absolute path to the cgroup hierarchies of this container.
	// (e.g.: "cpu" -> "/sys/fs/cgroup/cpu/test")
	cgroupPaths map[string]string
	// Manager of this container's cgroups.
	cgroupManager cgroups.Manager
	fsInfo        fs.FsInfo
	poolName      string
	// Time at which this container was created.
	creationTime time.Time
	// Metadata associated with the container.
	labels map[string]string
	envs   map[string]string
	// The container PID used to switch namespaces as required
	pid int
	// Image name used for this container.
	image string
	// The host root FS to read
	rootFs string
	// Filesystem handler.
	ignoreMetrics container.MetricSet
}

var _ container.ContainerHandler = &containerdContainerHandler{}

// newContainerdContainerHandler returns a new container.ContainerHandler
func newContainerdContainerHandler(
	client containerdClient,
	name string,
	machineInfoFactory info.MachineInfoFactory,
	fsInfo fs.FsInfo,
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

	id := ContainerNameToContainerdID(name)
	// We assume that if load fails then the container is not known to containerd.
	ctx := context.Background()
	cntr, err := client.LoadContainer(ctx, id)
	if err != nil {
		return nil, err
	}

	var spec specs.Spec
	if err := json.Unmarshal(cntr.Spec.Value, &spec); err != nil {
		return nil, err
	}

	taskPid, err := client.TaskPid(ctx, id)
	if err != nil {
		return nil, err
	}
	rootfs := "/"
	if !inHostNamespace {
		rootfs = "/rootfs"
	}

	handler := &containerdContainerHandler{
		id:                 id,
		client:             client,
		name:               name,
		machineInfoFactory: machineInfoFactory,
		cgroupPaths:        cgroupPaths,
		cgroupManager:      cgroupManager,
		rootFs:             rootfs,
		fsInfo:             fsInfo,
		envs:               make(map[string]string),
		labels:             make(map[string]string),
		ignoreMetrics:      ignoreMetrics,
		pid:                int(taskPid),
		creationTime:       cntr.CreatedAt,
	}
	// Add the name and bare ID as aliases of the container.
	handler.labels = cntr.Labels
	handler.image = cntr.Image
	handler.aliases = []string{id, name}
	for _, envVar := range spec.Process.Env {
		if envVar != "" {
			splits := strings.SplitN(envVar, "=", 2)
			if len(splits) == 2 {
				handler.envs[splits[0]] = splits[1]
			}
		}
	}

	return handler, nil
}

func (self *containerdContainerHandler) ContainerReference() (info.ContainerReference, error) {
	return info.ContainerReference{
		Id:        self.id,
		Name:      self.name,
		Namespace: k8sContainerdNamespace,
		Labels:    self.labels,
		Aliases:   self.aliases,
	}, nil
}

func (self *containerdContainerHandler) needNet() bool {
	// Since containerd does not handle networking ideally we need to return based
	// on ignoreMetrics list. Here the assumption is the presence of cri-containerd
	// label
	if !self.ignoreMetrics.Has(container.NetworkUsageMetrics) {
		//TODO change it to exported cri-containerd constants
		return self.labels["io.cri-containerd.kind"] == "sandbox"
	}
	return false
}

func (self *containerdContainerHandler) GetSpec() (info.ContainerSpec, error) {
	// TODO: Since we dont collect disk usage stats for containerd, we set hasFilesystem
	// to false. Revisit when we support disk usage stats for containerd
	hasFilesystem := false
	spec, err := common.GetSpec(self.cgroupPaths, self.machineInfoFactory, self.needNet(), hasFilesystem)
	spec.Labels = self.labels
	spec.Envs = self.envs
	spec.Image = self.image

	return spec, err
}

func (self *containerdContainerHandler) getFsStats(stats *info.ContainerStats) error {
	mi, err := self.machineInfoFactory.GetMachineInfo()
	if err != nil {
		return err
	}

	if !self.ignoreMetrics.Has(container.DiskIOMetrics) {
		common.AssignDeviceNamesToDiskStats((*common.MachineInfoNamer)(mi), &stats.DiskIo)
	}
	return nil
}

func (self *containerdContainerHandler) GetStats() (*info.ContainerStats, error) {
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
	return stats, err
}

func (self *containerdContainerHandler) ListContainers(listType container.ListType) ([]info.ContainerReference, error) {
	return []info.ContainerReference{}, nil
}

func (self *containerdContainerHandler) GetCgroupPath(resource string) (string, error) {
	path, ok := self.cgroupPaths[resource]
	if !ok {
		return "", fmt.Errorf("could not find path for resource %q for container %q\n", resource, self.name)
	}
	return path, nil
}

func (self *containerdContainerHandler) GetContainerLabels() map[string]string {
	return self.labels
}

func (self *containerdContainerHandler) ListProcesses(listType container.ListType) ([]int, error) {
	return containerlibcontainer.GetProcesses(self.cgroupManager)
}

func (self *containerdContainerHandler) Exists() bool {
	return common.CgroupExists(self.cgroupPaths)
}

func (self *containerdContainerHandler) Type() container.ContainerType {
	return container.ContainerTypeContainerd
}

func (self *containerdContainerHandler) Start() {
}

func (self *containerdContainerHandler) Cleanup() {
}

func (self *containerdContainerHandler) GetContainerIPAddress() string {
	// containerd doesnt take care of networking.So it doesnt maintain networking states
	return ""
}
