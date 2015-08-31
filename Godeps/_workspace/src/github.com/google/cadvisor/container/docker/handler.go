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
	"math"
	"path"
	"strings"
	"time"

	"github.com/docker/libcontainer/cgroups"
	cgroup_fs "github.com/docker/libcontainer/cgroups/fs"
	libcontainerConfigs "github.com/docker/libcontainer/configs"
	"github.com/fsouza/go-dockerclient"
	"github.com/google/cadvisor/container"
	containerLibcontainer "github.com/google/cadvisor/container/libcontainer"
	"github.com/google/cadvisor/fs"
	info "github.com/google/cadvisor/info/v1"
	"github.com/google/cadvisor/utils"
)

// Path to aufs dir where all the files exist.
// aufs/layers is ignored here since it does not hold a lot of data.
// aufs/mnt contains the mount points used to compose the rootfs. Hence it is also ignored.
var pathToAufsDir = "aufs/diff"

type dockerContainerHandler struct {
	client             *docker.Client
	name               string
	id                 string
	aliases            []string
	machineInfoFactory info.MachineInfoFactory

	// Path to the libcontainer config file.
	libcontainerConfigPath string

	// Path to the libcontainer state file.
	libcontainerStatePath string

	// TODO(vmarmol): Remove when we depend on a newer Docker.
	// Path to the libcontainer pid file.
	libcontainerPidPath string

	// Absolute path to the cgroup hierarchies of this container.
	// (e.g.: "cpu" -> "/sys/fs/cgroup/cpu/test")
	cgroupPaths map[string]string

	// Manager of this container's cgroups.
	cgroupManager cgroups.Manager

	usesAufsDriver bool
	fsInfo         fs.FsInfo
	storageDirs    []string

	// Time at which this container was created.
	creationTime time.Time

	// Metadata labels associated with the container.
	labels map[string]string

	// The container PID used to switch namespaces as required
	pid int
}

func newDockerContainerHandler(
	client *docker.Client,
	name string,
	machineInfoFactory info.MachineInfoFactory,
	fsInfo fs.FsInfo,
	usesAufsDriver bool,
	cgroupSubsystems *containerLibcontainer.CgroupSubsystems,
) (container.ContainerHandler, error) {
	// Create the cgroup paths.
	cgroupPaths := make(map[string]string, len(cgroupSubsystems.MountPoints))
	for key, val := range cgroupSubsystems.MountPoints {
		cgroupPaths[key] = path.Join(val, name)
	}

	// Generate the equivalent cgroup manager for this container.
	cgroupManager := &cgroup_fs.Manager{
		Cgroups: &libcontainerConfigs.Cgroup{
			Name: name,
		},
		Paths: cgroupPaths,
	}

	id := ContainerNameToDockerId(name)
	handler := &dockerContainerHandler{
		id:                 id,
		client:             client,
		name:               name,
		machineInfoFactory: machineInfoFactory,
		cgroupPaths:        cgroupPaths,
		cgroupManager:      cgroupManager,
		usesAufsDriver:     usesAufsDriver,
		fsInfo:             fsInfo,
	}
	handler.storageDirs = append(handler.storageDirs, path.Join(*dockerRootDir, pathToAufsDir, id))

	// We assume that if Inspect fails then the container is not known to docker.
	ctnr, err := client.InspectContainer(id)
	if err != nil {
		return nil, fmt.Errorf("failed to inspect container %q: %v", id, err)
	}
	handler.creationTime = ctnr.Created
	handler.pid = ctnr.State.Pid

	// Add the name and bare ID as aliases of the container.
	handler.aliases = append(handler.aliases, strings.TrimPrefix(ctnr.Name, "/"))
	handler.aliases = append(handler.aliases, id)
	handler.labels = ctnr.Config.Labels

	return handler, nil
}

func (self *dockerContainerHandler) ContainerReference() (info.ContainerReference, error) {
	return info.ContainerReference{
		Name:      self.name,
		Aliases:   self.aliases,
		Namespace: DockerNamespace,
	}, nil
}

func (self *dockerContainerHandler) readLibcontainerConfig() (*libcontainerConfigs.Config, error) {
	config, err := containerLibcontainer.ReadConfig(*dockerRootDir, *dockerRunDir, self.id)
	if err != nil {
		return nil, fmt.Errorf("failed to read libcontainer config: %v", err)
	}

	// Replace cgroup parent and name with our own since we may be running in a different context.
	if config.Cgroups == nil {
		config.Cgroups = new(libcontainerConfigs.Cgroup)
	}
	config.Cgroups.Name = self.name
	config.Cgroups.Parent = "/"

	return config, nil
}

func libcontainerConfigToContainerSpec(config *libcontainerConfigs.Config, mi *info.MachineInfo) info.ContainerSpec {
	var spec info.ContainerSpec
	spec.HasMemory = true
	spec.Memory.Limit = math.MaxUint64
	spec.Memory.SwapLimit = math.MaxUint64
	if config.Cgroups.Memory > 0 {
		spec.Memory.Limit = uint64(config.Cgroups.Memory)
	}
	if config.Cgroups.MemorySwap > 0 {
		spec.Memory.SwapLimit = uint64(config.Cgroups.MemorySwap)
	}

	// Get CPU info
	spec.HasCpu = true
	spec.Cpu.Limit = 1024
	if config.Cgroups.CpuShares != 0 {
		spec.Cpu.Limit = uint64(config.Cgroups.CpuShares)
	}
	spec.Cpu.Mask = utils.FixCpuMask(config.Cgroups.CpusetCpus, mi.NumCores)

	// Docker reports a loop device for containers with --net=host. Ignore
	// those too.
	networkCount := 0
	for _, n := range config.Networks {
		if n.Type != "loopback" {
			networkCount += 1
		}
	}

	spec.HasNetwork = networkCount > 0 || hasNetNs(config.Namespaces)
	spec.HasDiskIo = true

	return spec
}

func hasNetNs(namespaces libcontainerConfigs.Namespaces) bool {
	for _, ns := range namespaces {
		if ns.Type == libcontainerConfigs.NEWNET {
			return true
		}
	}
	return false
}

func (self *dockerContainerHandler) GetSpec() (info.ContainerSpec, error) {
	mi, err := self.machineInfoFactory.GetMachineInfo()
	if err != nil {
		return info.ContainerSpec{}, err
	}
	libcontainerConfig, err := self.readLibcontainerConfig()
	if err != nil {
		return info.ContainerSpec{}, err
	}

	spec := libcontainerConfigToContainerSpec(libcontainerConfig, mi)
	spec.CreationTime = self.creationTime
	if self.usesAufsDriver {
		spec.HasFilesystem = true
	}
	spec.Labels = self.labels

	return spec, err
}

func (self *dockerContainerHandler) getFsStats(stats *info.ContainerStats) error {
	// No support for non-aufs storage drivers.
	if !self.usesAufsDriver {
		return nil
	}

	// As of now we assume that all the storage dirs are on the same device.
	// The first storage dir will be that of the image layers.
	deviceInfo, err := self.fsInfo.GetDirFsDevice(self.storageDirs[0])
	if err != nil {
		return err
	}

	mi, err := self.machineInfoFactory.GetMachineInfo()
	if err != nil {
		return err
	}
	var limit uint64 = 0
	// Docker does not impose any filesystem limits for containers. So use capacity as limit.
	for _, fs := range mi.Filesystems {
		if fs.Device == deviceInfo.Device {
			limit = fs.Capacity
			break
		}
	}

	fsStat := info.FsStats{Device: deviceInfo.Device, Limit: limit}

	var usage uint64 = 0
	for _, dir := range self.storageDirs {
		// TODO(Vishh): Add support for external mounts.
		dirUsage, err := self.fsInfo.GetDirUsage(dir)
		if err != nil {
			return err
		}
		usage += dirUsage
	}
	fsStat.Usage = usage
	stats.Filesystem = append(stats.Filesystem, fsStat)

	return nil
}

// TODO(vmarmol): Get from libcontainer API instead of cgroup manager when we don't have to support older Dockers.
func (self *dockerContainerHandler) GetStats() (*info.ContainerStats, error) {
	config, err := self.readLibcontainerConfig()
	if err != nil {
		return nil, err
	}

	var networkInterfaces []string
	if len(config.Networks) > 0 {
		// ContainerStats only reports stat for one network device.
		// TODO(vmarmol): Handle multiple physical network devices.
		for _, n := range config.Networks {
			// Take the first non-loopback.
			if n.Type != "loopback" {
				networkInterfaces = []string{n.HostInterfaceName}
				break
			}
		}
	}
	stats, err := containerLibcontainer.GetStats(self.cgroupManager, networkInterfaces, self.pid)
	if err != nil {
		return stats, err
	}

	// TODO(rjnagal): Remove the conversion when network stats are read from libcontainer.
	convertInterfaceStats(&stats.Network.InterfaceStats)
	for i := range stats.Network.Interfaces {
		convertInterfaceStats(&stats.Network.Interfaces[i])
	}

	// Get filesystem stats.
	err = self.getFsStats(stats)
	if err != nil {
		return stats, err
	}

	return stats, nil
}

func convertInterfaceStats(stats *info.InterfaceStats) {
	net := *stats

	// Ingress for host veth is from the container.
	// Hence tx_bytes stat on the host veth is actually number of bytes received by the container.
	stats.RxBytes = net.TxBytes
	stats.RxPackets = net.TxPackets
	stats.RxErrors = net.TxErrors
	stats.RxDropped = net.TxDropped
	stats.TxBytes = net.RxBytes
	stats.TxPackets = net.RxPackets
	stats.TxErrors = net.RxErrors
	stats.TxDropped = net.RxDropped
}

func (self *dockerContainerHandler) ListContainers(listType container.ListType) ([]info.ContainerReference, error) {
	if self.name != "/docker" {
		return []info.ContainerReference{}, nil
	}
	opt := docker.ListContainersOptions{
		All: true,
	}
	containers, err := self.client.ListContainers(opt)
	if err != nil {
		return nil, err
	}

	ret := make([]info.ContainerReference, 0, len(containers)+1)
	for _, c := range containers {
		if !strings.HasPrefix(c.Status, "Up ") {
			continue
		}

		ref := info.ContainerReference{
			Name:      FullContainerName(c.ID),
			Aliases:   append(c.Names, c.ID),
			Namespace: DockerNamespace,
		}
		ret = append(ret, ref)
	}

	return ret, nil
}

func (self *dockerContainerHandler) GetCgroupPath(resource string) (string, error) {
	path, ok := self.cgroupPaths[resource]
	if !ok {
		return "", fmt.Errorf("could not find path for resource %q for container %q\n", resource, self.name)
	}
	return path, nil
}

func (self *dockerContainerHandler) ListThreads(listType container.ListType) ([]int, error) {
	// TODO(vmarmol): Implement.
	return nil, nil
}

func (self *dockerContainerHandler) GetContainerLabels() map[string]string {
	return self.labels
}

func (self *dockerContainerHandler) ListProcesses(listType container.ListType) ([]int, error) {
	return containerLibcontainer.GetProcesses(self.cgroupManager)
}

func (self *dockerContainerHandler) WatchSubcontainers(events chan container.SubcontainerEvent) error {
	return fmt.Errorf("watch is unimplemented in the Docker container driver")
}

func (self *dockerContainerHandler) StopWatchingSubcontainers() error {
	// No-op for Docker driver.
	return nil
}

func (self *dockerContainerHandler) Exists() bool {
	return containerLibcontainer.Exists(*dockerRootDir, *dockerRunDir, self.id)
}

func DockerInfo() (map[string]string, error) {
	client, err := docker.NewClient(*ArgDockerEndpoint)
	if err != nil {
		return nil, fmt.Errorf("unable to communicate with docker daemon: %v", err)
	}
	info, err := client.Info()
	if err != nil {
		return nil, err
	}
	return info.Map(), nil
}

func DockerImages() ([]docker.APIImages, error) {
	client, err := docker.NewClient(*ArgDockerEndpoint)
	if err != nil {
		return nil, fmt.Errorf("unable to communicate with docker daemon: %v", err)
	}
	images, err := client.ListImages(docker.ListImagesOptions{All: false})
	if err != nil {
		return nil, err
	}
	return images, nil
}
