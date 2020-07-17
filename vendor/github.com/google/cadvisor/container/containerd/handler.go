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
	"strings"
	"time"

	"github.com/containerd/containerd/errdefs"
	"golang.org/x/net/context"

	"github.com/google/cadvisor/container"
	"github.com/google/cadvisor/container/common"
	containerlibcontainer "github.com/google/cadvisor/container/libcontainer"
	"github.com/google/cadvisor/fs"
	info "github.com/google/cadvisor/info/v1"
	specs "github.com/opencontainers/runtime-spec/specs-go"
)

type containerdContainerHandler struct {
	machineInfoFactory info.MachineInfoFactory
	// Absolute path to the cgroup hierarchies of this container.
	// (e.g.: "cpu" -> "/sys/fs/cgroup/cpu/test")
	cgroupPaths map[string]string
	fsInfo      fs.FsInfo
	// Metadata associated with the container.
	reference info.ContainerReference
	envs      map[string]string
	labels    map[string]string
	// Image name used for this container.
	image string
	// Filesystem handler.
	includedMetrics container.MetricSet

	libcontainerHandler *containerlibcontainer.Handler
}

var _ container.ContainerHandler = &containerdContainerHandler{}

// newContainerdContainerHandler returns a new container.ContainerHandler
func newContainerdContainerHandler(
	client ContainerdClient,
	name string,
	machineInfoFactory info.MachineInfoFactory,
	fsInfo fs.FsInfo,
	cgroupSubsystems *containerlibcontainer.CgroupSubsystems,
	inHostNamespace bool,
	metadataEnvs []string,
	includedMetrics container.MetricSet,
) (container.ContainerHandler, error) {
	// Create the cgroup paths.
	cgroupPaths := common.MakeCgroupPaths(cgroupSubsystems.MountPoints, name)

	// Generate the equivalent cgroup manager for this container.
	cgroupManager, err := containerlibcontainer.NewCgroupManager(name, cgroupPaths)
	if err != nil {
		return nil, err
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

	// Cgroup is created during task creation. When cadvisor sees the cgroup,
	// task may not be fully created yet. Use a retry+backoff to tolerant the
	// race condition.
	// TODO(random-liu): Use cri-containerd client to talk with cri-containerd
	// instead. cri-containerd has some internal synchronization to make sure
	// `ContainerStatus` only returns result after `StartContainer` finishes.
	var taskPid uint32
	backoff := 100 * time.Millisecond
	retry := 5
	for {
		taskPid, err = client.TaskPid(ctx, id)
		if err == nil {
			break
		}
		retry--
		if !errdefs.IsNotFound(err) || retry == 0 {
			return nil, err
		}
		time.Sleep(backoff)
		backoff *= 2
	}

	rootfs := "/"
	if !inHostNamespace {
		rootfs = "/rootfs"
	}

	containerReference := info.ContainerReference{
		Id:        id,
		Name:      name,
		Namespace: k8sContainerdNamespace,
		Aliases:   []string{id, name},
	}

	libcontainerHandler := containerlibcontainer.NewHandler(cgroupManager, rootfs, int(taskPid), includedMetrics)

	handler := &containerdContainerHandler{
		machineInfoFactory:  machineInfoFactory,
		cgroupPaths:         cgroupPaths,
		fsInfo:              fsInfo,
		envs:                make(map[string]string),
		labels:              cntr.Labels,
		includedMetrics:     includedMetrics,
		reference:           containerReference,
		libcontainerHandler: libcontainerHandler,
	}
	// Add the name and bare ID as aliases of the container.
	handler.image = cntr.Image
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

func (h *containerdContainerHandler) ContainerReference() (info.ContainerReference, error) {
	return h.reference, nil
}

func (h *containerdContainerHandler) needNet() bool {
	// Since containerd does not handle networking ideally we need to return based
	// on includedMetrics list. Here the assumption is the presence of cri-containerd
	// label
	if h.includedMetrics.Has(container.NetworkUsageMetrics) {
		//TODO change it to exported cri-containerd constants
		return h.labels["io.cri-containerd.kind"] == "sandbox"
	}
	return false
}

func (h *containerdContainerHandler) GetSpec() (info.ContainerSpec, error) {
	// TODO: Since we dont collect disk usage stats for containerd, we set hasFilesystem
	// to false. Revisit when we support disk usage stats for containerd
	hasFilesystem := false
	spec, err := common.GetSpec(h.cgroupPaths, h.machineInfoFactory, h.needNet(), hasFilesystem)
	spec.Labels = h.labels
	spec.Envs = h.envs
	spec.Image = h.image

	return spec, err
}

func (h *containerdContainerHandler) getFsStats(stats *info.ContainerStats) error {
	mi, err := h.machineInfoFactory.GetMachineInfo()
	if err != nil {
		return err
	}

	if h.includedMetrics.Has(container.DiskIOMetrics) {
		common.AssignDeviceNamesToDiskStats((*common.MachineInfoNamer)(mi), &stats.DiskIo)
	}
	return nil
}

func (h *containerdContainerHandler) GetStats() (*info.ContainerStats, error) {
	stats, err := h.libcontainerHandler.GetStats()
	if err != nil {
		return stats, err
	}
	// Clean up stats for containers that don't have their own network - this
	// includes containers running in Kubernetes pods that use the network of the
	// infrastructure container. This stops metrics being reported multiple times
	// for each container in a pod.
	if !h.needNet() {
		stats.Network = info.NetworkStats{}
	}

	// Get filesystem stats.
	err = h.getFsStats(stats)
	return stats, err
}

func (h *containerdContainerHandler) ListContainers(listType container.ListType) ([]info.ContainerReference, error) {
	return []info.ContainerReference{}, nil
}

func (h *containerdContainerHandler) GetCgroupPath(resource string) (string, error) {
	path, ok := h.cgroupPaths[resource]
	if !ok {
		return "", fmt.Errorf("could not find path for resource %q for container %q", resource, h.reference.Name)
	}
	return path, nil
}

func (h *containerdContainerHandler) GetContainerLabels() map[string]string {
	return h.labels
}

func (h *containerdContainerHandler) ListProcesses(listType container.ListType) ([]int, error) {
	return h.libcontainerHandler.GetProcesses()
}

func (h *containerdContainerHandler) Exists() bool {
	return common.CgroupExists(h.cgroupPaths)
}

func (h *containerdContainerHandler) Type() container.ContainerType {
	return container.ContainerTypeContainerd
}

func (h *containerdContainerHandler) Start() {
}

func (h *containerdContainerHandler) Cleanup() {
}

func (h *containerdContainerHandler) GetContainerIPAddress() string {
	// containerd doesnt take care of networking.So it doesnt maintain networking states
	return ""
}
