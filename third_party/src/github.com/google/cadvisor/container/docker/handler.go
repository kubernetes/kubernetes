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

package docker

import (
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"math"
	"os"
	"path"
	"path/filepath"
	"strings"

	"github.com/docker/libcontainer"
	"github.com/docker/libcontainer/cgroups"
	"github.com/docker/libcontainer/cgroups/fs"
	"github.com/fsouza/go-dockerclient"
	"github.com/google/cadvisor/container"
	containerLibcontainer "github.com/google/cadvisor/container/libcontainer"
	"github.com/google/cadvisor/info"
	"github.com/google/cadvisor/utils"
)

// Basepath to all container specific information that libcontainer stores.
const dockerRootDir = "/var/lib/docker/execdriver/native"

var fileNotFound = errors.New("file not found")

type dockerContainerHandler struct {
	client             *docker.Client
	name               string
	parent             string
	id                 string
	aliases            []string
	machineInfoFactory info.MachineInfoFactory
	useSystemd         bool
}

func newDockerContainerHandler(
	client *docker.Client,
	name string,
	machineInfoFactory info.MachineInfoFactory,
	useSystemd bool,
) (container.ContainerHandler, error) {
	handler := &dockerContainerHandler{
		client:             client,
		name:               name,
		machineInfoFactory: machineInfoFactory,
		useSystemd:         useSystemd,
	}
	if handler.isDockerRoot() {
		return handler, nil
	}
	parent, id, err := containerLibcontainer.SplitName(name)
	if err != nil {
		return nil, fmt.Errorf("invalid docker container %v: %v", name, err)
	}
	handler.parent = parent
	handler.id = id
	ctnr, err := client.InspectContainer(id)
	// We assume that if Inspect fails then the container is not known to docker.
	if err != nil {
		return nil, fmt.Errorf("failed to inspect container %s - %s\n", id, err)
	}
	handler.aliases = append(handler.aliases, path.Join("/docker", ctnr.Name))
	return handler, nil
}

func (self *dockerContainerHandler) ContainerReference() (info.ContainerReference, error) {
	return info.ContainerReference{
		Name:    self.name,
		Aliases: self.aliases,
	}, nil
}

func (self *dockerContainerHandler) isDockerRoot() bool {
	return self.name == "/docker"
}

// TODO(vmarmol): Switch to getting this from libcontainer once we have a solid API.
func (self *dockerContainerHandler) readLibcontainerConfig() (config *libcontainer.Config, err error) {
	configPath := path.Join(dockerRootDir, self.id, "container.json")
	if !utils.FileExists(configPath) {
		// TODO(vishh): Return file name as well once we have a better error interface.
		err = fileNotFound
		return
	}
	f, err := os.Open(configPath)
	if err != nil {
		return nil, fmt.Errorf("failed to open %s - %s\n", configPath, err)
	}
	defer f.Close()
	d := json.NewDecoder(f)
	retConfig := new(libcontainer.Config)
	err = d.Decode(retConfig)
	if err != nil {
		return
	}
	config = retConfig

	// Replace cgroup parent and name with our own since we may be running in a different context.
	config.Cgroups.Parent = self.parent
	config.Cgroups.Name = self.id

	return
}

func (self *dockerContainerHandler) readLibcontainerState() (state *libcontainer.State, err error) {
	statePath := path.Join(dockerRootDir, self.id, "state.json")
	if !utils.FileExists(statePath) {
		// TODO(vmarmol): Remove this once we can depend on a newer Docker.
		// Libcontainer changed how its state was stored, try the old way of a "pid" file
		if utils.FileExists(path.Join(dockerRootDir, self.id, "pid")) {
			// We don't need the old state, return an empty state and we'll gracefully degrade.
			state = new(libcontainer.State)
			return
		}

		// TODO(vishh): Return file name as well once we have a better error interface.
		err = fileNotFound
		return
	}
	f, err := os.Open(statePath)
	if err != nil {
		return nil, fmt.Errorf("failed to open %s - %s\n", statePath, err)
	}
	defer f.Close()
	d := json.NewDecoder(f)
	retState := new(libcontainer.State)
	err = d.Decode(retState)
	if err != nil {
		return
	}
	state = retState

	return
}

func libcontainerConfigToContainerSpec(config *libcontainer.Config, mi *info.MachineInfo) *info.ContainerSpec {
	spec := new(info.ContainerSpec)
	spec.Memory = new(info.MemorySpec)
	spec.Memory.Limit = math.MaxUint64
	spec.Memory.SwapLimit = math.MaxUint64
	if config.Cgroups.Memory > 0 {
		spec.Memory.Limit = uint64(config.Cgroups.Memory)
	}
	if config.Cgroups.MemorySwap > 0 {
		spec.Memory.SwapLimit = uint64(config.Cgroups.MemorySwap)
	}

	// Get CPU info
	spec.Cpu = new(info.CpuSpec)
	spec.Cpu.Limit = 1024
	if config.Cgroups.CpuShares != 0 {
		spec.Cpu.Limit = uint64(config.Cgroups.CpuShares)
	}
	if config.Cgroups.CpusetCpus == "" {
		// All cores are active.
		spec.Cpu.Mask = fmt.Sprintf("0-%d", mi.NumCores-1)
	} else {
		spec.Cpu.Mask = config.Cgroups.CpusetCpus
	}
	return spec
}

func (self *dockerContainerHandler) GetSpec() (spec *info.ContainerSpec, err error) {
	if self.isDockerRoot() {
		return &info.ContainerSpec{}, nil
	}
	mi, err := self.machineInfoFactory.GetMachineInfo()
	if err != nil {
		return
	}
	libcontainerConfig, err := self.readLibcontainerConfig()
	if err != nil {
		return
	}

	spec = libcontainerConfigToContainerSpec(libcontainerConfig, mi)
	return
}

func (self *dockerContainerHandler) GetStats() (stats *info.ContainerStats, err error) {
	if self.isDockerRoot() {
		return &info.ContainerStats{}, nil
	}
	config, err := self.readLibcontainerConfig()
	if err != nil {
		if err == fileNotFound {
			log.Printf("Libcontainer config not found for container %q", self.name)
			return &info.ContainerStats{}, nil
		}
		return
	}
	state, err := self.readLibcontainerState()
	if err != nil {
		if err == fileNotFound {
			log.Printf("Libcontainer state not found for container %q", self.name)
			return &info.ContainerStats{}, nil
		}
		return
	}

	return containerLibcontainer.GetStats(config, state)
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

	// On non-systemd systems Docker containers are under /docker.
	containerPrefix := "/docker"
	if self.useSystemd {
		containerPrefix = "/system.slice"
	}

	ret := make([]info.ContainerReference, 0, len(containers)+1)
	for _, c := range containers {
		if !strings.HasPrefix(c.Status, "Up ") {
			continue
		}

		ref := info.ContainerReference{
			Name:    filepath.Join(containerPrefix, c.ID),
			Aliases: c.Names,
		}
		ret = append(ret, ref)
	}

	return ret, nil
}

func (self *dockerContainerHandler) ListThreads(listType container.ListType) ([]int, error) {
	return nil, nil
}

func (self *dockerContainerHandler) ListProcesses(listType container.ListType) ([]int, error) {
	c := &cgroups.Cgroup{
		Parent: self.parent,
		Name:   self.id,
	}
	return fs.GetPids(c)
}
