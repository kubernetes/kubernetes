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

// Handler for Docker containers.
package docker

import (
	"fmt"
	"os"
	"path"
	"strconv"
	"strings"
	"time"

	"github.com/google/cadvisor/container"
	"github.com/google/cadvisor/container/common"
	dockerutil "github.com/google/cadvisor/container/docker/utils"
	containerlibcontainer "github.com/google/cadvisor/container/libcontainer"
	"github.com/google/cadvisor/devicemapper"
	"github.com/google/cadvisor/fs"
	info "github.com/google/cadvisor/info/v1"
	"github.com/google/cadvisor/zfs"
	"github.com/opencontainers/runc/libcontainer/cgroups"

	docker "github.com/docker/docker/client"
	"golang.org/x/net/context"
)

const (
	// The read write layers exist here.
	aufsRWLayer     = "diff"
	overlayRWLayer  = "upper"
	overlay2RWLayer = "diff"

	// Path to the directory where docker stores log files if the json logging driver is enabled.
	pathToContainersDir = "containers"
)

type dockerContainerHandler struct {
	// machineInfoFactory provides info.MachineInfo
	machineInfoFactory info.MachineInfoFactory

	// Absolute path to the cgroup hierarchies of this container.
	// (e.g.: "cpu" -> "/sys/fs/cgroup/cpu/test")
	cgroupPaths map[string]string

	// the docker storage driver
	storageDriver    StorageDriver
	fsInfo           fs.FsInfo
	rootfsStorageDir string

	// Time at which this container was created.
	creationTime time.Time

	// Metadata associated with the container.
	envs   map[string]string
	labels map[string]string

	// Image name used for this container.
	image string

	// Filesystem handler.
	fsHandler common.FsHandler

	// The IP address of the container
	ipAddress string

	includedMetrics container.MetricSet

	// the devicemapper poolname
	poolName string

	// zfsParent is the parent for docker zfs
	zfsParent string

	// Reference to the container
	reference info.ContainerReference

	libcontainerHandler *containerlibcontainer.Handler
}

var _ container.ContainerHandler = &dockerContainerHandler{}

func getRwLayerID(containerID, storageDir string, sd StorageDriver, dockerVersion []int) (string, error) {
	const (
		// Docker version >=1.10.0 have a randomized ID for the root fs of a container.
		randomizedRWLayerMinorVersion = 10
		rwLayerIDFile                 = "mount-id"
	)
	if (dockerVersion[0] <= 1) && (dockerVersion[1] < randomizedRWLayerMinorVersion) {
		return containerID, nil
	}

	bytes, err := os.ReadFile(path.Join(storageDir, "image", string(sd), "layerdb", "mounts", containerID, rwLayerIDFile))
	if err != nil {
		return "", fmt.Errorf("failed to identify the read-write layer ID for container %q. - %v", containerID, err)
	}
	return string(bytes), err
}

// newDockerContainerHandler returns a new container.ContainerHandler
func newDockerContainerHandler(
	client *docker.Client,
	name string,
	machineInfoFactory info.MachineInfoFactory,
	fsInfo fs.FsInfo,
	storageDriver StorageDriver,
	storageDir string,
	cgroupSubsystems map[string]string,
	inHostNamespace bool,
	metadataEnvAllowList []string,
	dockerVersion []int,
	includedMetrics container.MetricSet,
	thinPoolName string,
	thinPoolWatcher *devicemapper.ThinPoolWatcher,
	zfsWatcher *zfs.ZfsWatcher,
) (container.ContainerHandler, error) {
	// Create the cgroup paths.
	cgroupPaths := common.MakeCgroupPaths(cgroupSubsystems, name)

	// Generate the equivalent cgroup manager for this container.
	cgroupManager, err := containerlibcontainer.NewCgroupManager(name, cgroupPaths)
	if err != nil {
		return nil, err
	}

	rootFs := "/"
	if !inHostNamespace {
		rootFs = "/rootfs"
		storageDir = path.Join(rootFs, storageDir)
	}

	id := dockerutil.ContainerNameToId(name)

	// Add the Containers dir where the log files are stored.
	// FIXME: Give `otherStorageDir` a more descriptive name.
	otherStorageDir := path.Join(storageDir, pathToContainersDir, id)

	rwLayerID, err := getRwLayerID(id, storageDir, storageDriver, dockerVersion)
	if err != nil {
		return nil, err
	}

	// Determine the rootfs storage dir OR the pool name to determine the device.
	// For devicemapper, we only need the thin pool name, and that is passed in to this call
	rootfsStorageDir, zfsFilesystem, zfsParent, err := DetermineDeviceStorage(storageDriver, storageDir, rwLayerID)
	if err != nil {
		return nil, fmt.Errorf("unable to determine device storage: %v", err)
	}

	// We assume that if Inspect fails then the container is not known to docker.
	ctnr, err := client.ContainerInspect(context.Background(), id)
	if err != nil {
		return nil, fmt.Errorf("failed to inspect container %q: %v", id, err)
	}

	// Do not report network metrics for containers that share netns with another container.
	metrics := common.RemoveNetMetrics(includedMetrics, ctnr.HostConfig.NetworkMode.IsContainer())

	// TODO: extract object mother method
	handler := &dockerContainerHandler{
		machineInfoFactory: machineInfoFactory,
		cgroupPaths:        cgroupPaths,
		fsInfo:             fsInfo,
		storageDriver:      storageDriver,
		poolName:           thinPoolName,
		rootfsStorageDir:   rootfsStorageDir,
		envs:               make(map[string]string),
		labels:             ctnr.Config.Labels,
		includedMetrics:    metrics,
		zfsParent:          zfsParent,
	}
	// Timestamp returned by Docker is in time.RFC3339Nano format.
	handler.creationTime, err = time.Parse(time.RFC3339Nano, ctnr.Created)
	if err != nil {
		// This should not happen, report the error just in case
		return nil, fmt.Errorf("failed to parse the create timestamp %q for container %q: %v", ctnr.Created, id, err)
	}
	handler.libcontainerHandler = containerlibcontainer.NewHandler(cgroupManager, rootFs, ctnr.State.Pid, metrics)

	// Add the name and bare ID as aliases of the container.
	handler.reference = info.ContainerReference{
		Id:        id,
		Name:      name,
		Aliases:   []string{strings.TrimPrefix(ctnr.Name, "/"), id},
		Namespace: DockerNamespace,
	}
	handler.image = ctnr.Config.Image
	// Only adds restartcount label if it's greater than 0
	if ctnr.RestartCount > 0 {
		handler.labels["restartcount"] = strconv.Itoa(ctnr.RestartCount)
	}

	// Obtain the IP address for the container.
	// If the NetworkMode starts with 'container:' then we need to use the IP address of the container specified.
	// This happens in cases such as kubernetes where the containers doesn't have an IP address itself and we need to use the pod's address
	ipAddress := ctnr.NetworkSettings.IPAddress
	networkMode := string(ctnr.HostConfig.NetworkMode)
	if ipAddress == "" && strings.HasPrefix(networkMode, "container:") {
		containerID := strings.TrimPrefix(networkMode, "container:")
		c, err := client.ContainerInspect(context.Background(), containerID)
		if err != nil {
			return nil, fmt.Errorf("failed to inspect container %q: %v", id, err)
		}
		ipAddress = c.NetworkSettings.IPAddress
	}

	handler.ipAddress = ipAddress

	if includedMetrics.Has(container.DiskUsageMetrics) {
		handler.fsHandler = &FsHandler{
			FsHandler:       common.NewFsHandler(common.DefaultPeriod, rootfsStorageDir, otherStorageDir, fsInfo),
			ThinPoolWatcher: thinPoolWatcher,
			ZfsWatcher:      zfsWatcher,
			DeviceID:        ctnr.GraphDriver.Data["DeviceId"],
			ZfsFilesystem:   zfsFilesystem,
		}
	}

	// split env vars to get metadata map.
	for _, exposedEnv := range metadataEnvAllowList {
		if exposedEnv == "" {
			// if no dockerEnvWhitelist provided, len(metadataEnvAllowList) == 1, metadataEnvAllowList[0] == ""
			continue
		}

		for _, envVar := range ctnr.Config.Env {
			if envVar != "" {
				splits := strings.SplitN(envVar, "=", 2)
				if len(splits) == 2 && strings.HasPrefix(splits[0], exposedEnv) {
					handler.envs[strings.ToLower(splits[0])] = splits[1]
				}
			}
		}
	}

	return handler, nil
}

func DetermineDeviceStorage(storageDriver StorageDriver, storageDir string, rwLayerID string) (
	rootfsStorageDir string, zfsFilesystem string, zfsParent string, err error) {
	switch storageDriver {
	case AufsStorageDriver:
		rootfsStorageDir = path.Join(storageDir, string(AufsStorageDriver), aufsRWLayer, rwLayerID)
	case OverlayStorageDriver:
		rootfsStorageDir = path.Join(storageDir, string(storageDriver), rwLayerID, overlayRWLayer)
	case Overlay2StorageDriver:
		rootfsStorageDir = path.Join(storageDir, string(storageDriver), rwLayerID, overlay2RWLayer)
	case VfsStorageDriver:
		rootfsStorageDir = path.Join(storageDir)
	case ZfsStorageDriver:
		var status info.DockerStatus
		status, err = Status()
		if err != nil {
			return
		}
		zfsParent = status.DriverStatus[dockerutil.DriverStatusParentDataset]
		zfsFilesystem = path.Join(zfsParent, rwLayerID)
	}
	return
}

func (h *dockerContainerHandler) Start() {
	if h.fsHandler != nil {
		h.fsHandler.Start()
	}
}

func (h *dockerContainerHandler) Cleanup() {
	if h.fsHandler != nil {
		h.fsHandler.Stop()
	}
}

func (h *dockerContainerHandler) ContainerReference() (info.ContainerReference, error) {
	return h.reference, nil
}

func (h *dockerContainerHandler) GetSpec() (info.ContainerSpec, error) {
	hasFilesystem := h.includedMetrics.Has(container.DiskUsageMetrics)
	hasNetwork := h.includedMetrics.Has(container.NetworkUsageMetrics)
	spec, err := common.GetSpec(h.cgroupPaths, h.machineInfoFactory, hasNetwork, hasFilesystem)

	spec.Labels = h.labels
	spec.Envs = h.envs
	spec.Image = h.image
	spec.CreationTime = h.creationTime

	return spec, err
}

// TODO(vmarmol): Get from libcontainer API instead of cgroup manager when we don't have to support older Dockers.
func (h *dockerContainerHandler) GetStats() (*info.ContainerStats, error) {
	stats, err := h.libcontainerHandler.GetStats()
	if err != nil {
		return stats, err
	}

	// Get filesystem stats.
	err = FsStats(stats, h.machineInfoFactory, h.includedMetrics, h.storageDriver,
		h.fsHandler, h.fsInfo, h.poolName, h.rootfsStorageDir, h.zfsParent)
	if err != nil {
		return stats, err
	}

	return stats, nil
}

func (h *dockerContainerHandler) ListContainers(listType container.ListType) ([]info.ContainerReference, error) {
	// No-op for Docker driver.
	return []info.ContainerReference{}, nil
}

func (h *dockerContainerHandler) GetCgroupPath(resource string) (string, error) {
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

func (h *dockerContainerHandler) GetContainerLabels() map[string]string {
	return h.labels
}

func (h *dockerContainerHandler) GetContainerIPAddress() string {
	return h.ipAddress
}

func (h *dockerContainerHandler) ListProcesses(listType container.ListType) ([]int, error) {
	return h.libcontainerHandler.GetProcesses()
}

func (h *dockerContainerHandler) Exists() bool {
	return common.CgroupExists(h.cgroupPaths)
}

func (h *dockerContainerHandler) Type() container.ContainerType {
	return container.ContainerTypeDocker
}
