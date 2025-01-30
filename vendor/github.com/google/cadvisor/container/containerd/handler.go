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
	"errors"
	"fmt"
	"strings"
	"time"

	"github.com/containerd/errdefs"
	"github.com/opencontainers/runc/libcontainer/cgroups"
	specs "github.com/opencontainers/runtime-spec/specs-go"
	"golang.org/x/net/context"

	"github.com/google/cadvisor/container"
	"github.com/google/cadvisor/container/common"
	containerlibcontainer "github.com/google/cadvisor/container/libcontainer"
	"github.com/google/cadvisor/fs"
	info "github.com/google/cadvisor/info/v1"
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
	cgroupSubsystems map[string]string,
	inHostNamespace bool,
	metadataEnvAllowList []string,
	includedMetrics container.MetricSet,
) (container.ContainerHandler, error) {
	// Create the cgroup paths.
	cgroupPaths := common.MakeCgroupPaths(cgroupSubsystems, name)

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

		// Retry when task is not created yet or task is in unknown state (likely in process of initializing)
		isRetriableError := errdefs.IsNotFound(err) || errors.Is(err, ErrTaskIsInUnknownState)
		if !isRetriableError || retry == 0 {
			return nil, err
		}

		retry--
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

	// Containers that don't have their own network -- this includes
	// containers running in Kubernetes pods that use the network of the
	// infrastructure container -- does not need their stats to be
	// reported. This stops metrics being reported multiple times for each
	// container in a pod.
	metrics := common.RemoveNetMetrics(includedMetrics, cntr.Labels["io.cri-containerd.kind"] != "sandbox")

	libcontainerHandler := containerlibcontainer.NewHandler(cgroupManager, rootfs, int(taskPid), metrics)

	handler := &containerdContainerHandler{
		machineInfoFactory:  machineInfoFactory,
		cgroupPaths:         cgroupPaths,
		fsInfo:              fsInfo,
		envs:                make(map[string]string),
		labels:              cntr.Labels,
		includedMetrics:     metrics,
		reference:           containerReference,
		libcontainerHandler: libcontainerHandler,
	}
	// Add the name and bare ID as aliases of the container.
	handler.image = cntr.Image

	for _, exposedEnv := range metadataEnvAllowList {
		if exposedEnv == "" {
			// if no containerdEnvWhitelist provided, len(metadataEnvAllowList) == 1, metadataEnvAllowList[0] == ""
			continue
		}

		for _, envVar := range spec.Process.Env {
			if envVar != "" {
				splits := strings.SplitN(envVar, "=", 2)
				if len(splits) == 2 && strings.HasPrefix(splits[0], exposedEnv) {
					handler.envs[splits[0]] = splits[1]
				}
			}
		}
	}

	return handler, nil
}

func (h *containerdContainerHandler) ContainerReference() (info.ContainerReference, error) {
	return h.reference, nil
}

func (h *containerdContainerHandler) GetSpec() (info.ContainerSpec, error) {
	// TODO: Since we dont collect disk usage stats for containerd, we set hasFilesystem
	// to false. Revisit when we support disk usage stats for containerd
	hasFilesystem := false
	hasNet := h.includedMetrics.Has(container.NetworkUsageMetrics)
	spec, err := common.GetSpec(h.cgroupPaths, h.machineInfoFactory, hasNet, hasFilesystem)
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

	// Get filesystem stats.
	err = h.getFsStats(stats)
	return stats, err
}

func (h *containerdContainerHandler) ListContainers(listType container.ListType) ([]info.ContainerReference, error) {
	return []info.ContainerReference{}, nil
}

func (h *containerdContainerHandler) GetCgroupPath(resource string) (string, error) {
	var res string
	if !cgroups.IsCgroup2UnifiedMode() {
		res = resource
	}
	path, ok := h.cgroupPaths[res]
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
