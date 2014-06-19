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
	"bufio"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path"
	"strings"
	"time"

	"github.com/docker/libcontainer"
	"github.com/docker/libcontainer/cgroups"
	"github.com/docker/libcontainer/cgroups/fs"
	"github.com/fsouza/go-dockerclient"
	"github.com/google/cadvisor/container"
	"github.com/google/cadvisor/info"
)

type dockerContainerHandler struct {
	client             *docker.Client
	name               string
	aliases            []string
	machineInfoFactory info.MachineInfoFactory
}

func newDockerContainerHandler(
	client *docker.Client,
	name string,
	machineInfoFactory info.MachineInfoFactory,
) (container.ContainerHandler, error) {
	handler := &dockerContainerHandler{
		client:             client,
		name:               name,
		machineInfoFactory: machineInfoFactory,
	}
	if !handler.isDockerContainer() {
		return handler, nil
	}
	_, id, err := handler.splitName()
	if err != nil {
		return nil, fmt.Errorf("invalid docker container %v: %v", name, err)
	}
	ctnr, err := client.InspectContainer(id)
	if err != nil {
		return nil, fmt.Errorf("unable to inspect container %v: %v", name, err)
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

func (self *dockerContainerHandler) splitName() (string, string, error) {
	parent, id := path.Split(self.name)
	cgroupSelf, err := os.Open("/proc/self/cgroup")
	if err != nil {
		return "", "", err
	}
	scanner := bufio.NewScanner(cgroupSelf)

	subsys := []string{"memory", "cpu"}
	nestedLevels := 0
	for scanner.Scan() {
		line := scanner.Text()
		elems := strings.Split(line, ":")
		if len(elems) < 3 {
			continue
		}
		for _, s := range subsys {
			if elems[1] == s {
				// count how many nested docker containers are there.
				nestedLevels = strings.Count(elems[2], "/docker")
				break
			}
		}
	}
	if nestedLevels > 0 {
		// we are running inside a docker container
		upperLevel := strings.Repeat("../../", nestedLevels)
		//parent = strings.Join([]string{parent, upperLevel}, "/")
		parent = fmt.Sprintf("%v%v", upperLevel, parent)
	}
	return parent, id, nil
}

func (self *dockerContainerHandler) isDockerRoot() bool {
	// TODO(dengnan): Should we consider other cases?
	return self.name == "/docker"
}

func (self *dockerContainerHandler) isRootContainer() bool {
	return self.name == "/"
}

func (self *dockerContainerHandler) isDockerContainer() bool {
	return (!self.isDockerRoot()) && (!self.isRootContainer())
}

// TODO(vmarmol): Switch to getting this from libcontainer once we have a solid API.
func readLibcontainerSpec(id string) (spec *libcontainer.Container, err error) {
	dir := "/var/lib/docker/execdriver/native"
	configPath := path.Join(dir, id, "container.json")
	f, err := os.Open(configPath)
	if err != nil {
		return
	}
	defer f.Close()
	d := json.NewDecoder(f)
	ret := new(libcontainer.Container)
	err = d.Decode(ret)
	if err != nil {
		return
	}
	spec = ret
	return
}

func libcontainerConfigToContainerSpec(config *libcontainer.Container, mi *info.MachineInfo) *info.ContainerSpec {
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
	n := (mi.NumCores + 63) / 64
	spec.Cpu.Mask.Data = make([]uint64, n)
	for i := 0; i < n; i++ {
		spec.Cpu.Mask.Data[i] = math.MaxUint64
	}
	// TODO(vmarmol): Get CPUs from config.Cgroups.CpusetCpus
	return spec
}

func (self *dockerContainerHandler) GetSpec() (spec *info.ContainerSpec, err error) {
	if !self.isDockerContainer() {
		spec = new(info.ContainerSpec)
		return
	}
	mi, err := self.machineInfoFactory.GetMachineInfo()
	if err != nil {
		return
	}
	_, id, err := self.splitName()
	if err != nil {
		return
	}
	libcontainerSpec, err := readLibcontainerSpec(id)
	if err != nil {
		return
	}

	spec = libcontainerConfigToContainerSpec(libcontainerSpec, mi)
	return
}

func libcontainerToContainerStats(s *cgroups.Stats, mi *info.MachineInfo) *info.ContainerStats {
	ret := new(info.ContainerStats)
	ret.Timestamp = time.Now()
	ret.Cpu = new(info.CpuStats)
	ret.Cpu.Usage.User = s.CpuStats.CpuUsage.UsageInUsermode
	ret.Cpu.Usage.System = s.CpuStats.CpuUsage.UsageInKernelmode
	n := len(s.CpuStats.CpuUsage.PercpuUsage)
	ret.Cpu.Usage.PerCpu = make([]uint64, n)

	ret.Cpu.Usage.Total = 0
	for i := 0; i < n; i++ {
		ret.Cpu.Usage.PerCpu[i] = s.CpuStats.CpuUsage.PercpuUsage[i]
		ret.Cpu.Usage.Total += s.CpuStats.CpuUsage.PercpuUsage[i]
	}
	ret.Memory = new(info.MemoryStats)
	ret.Memory.Usage = s.MemoryStats.Usage
	if v, ok := s.MemoryStats.Stats["pgfault"]; ok {
		ret.Memory.ContainerData.Pgfault = v
		ret.Memory.HierarchicalData.Pgfault = v
	}
	if v, ok := s.MemoryStats.Stats["pgmajfault"]; ok {
		ret.Memory.ContainerData.Pgmajfault = v
		ret.Memory.HierarchicalData.Pgmajfault = v
	}
	return ret
}

func (self *dockerContainerHandler) GetStats() (stats *info.ContainerStats, err error) {
	if !self.isDockerContainer() {
		// Return empty stats for root containers.
		stats = new(info.ContainerStats)
		stats.Timestamp = time.Now()
		return
	}
	mi, err := self.machineInfoFactory.GetMachineInfo()
	if err != nil {
		return
	}
	parent, id, err := self.splitName()
	if err != nil {
		return
	}
	cg := &cgroups.Cgroup{
		Parent: parent,
		Name:   id,
	}
	s, err := fs.GetStats(cg)
	if err != nil {
		return
	}
	stats = libcontainerToContainerStats(s, mi)
	return
}

func (self *dockerContainerHandler) ListContainers(listType container.ListType) ([]info.ContainerReference, error) {
	if self.isDockerContainer() {
		return nil, nil
	}
	if self.isRootContainer() && listType == container.LIST_SELF {
		return []info.ContainerReference{info.ContainerReference{Name: "/docker"}}, nil
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
		path := fmt.Sprintf("/docker/%v", c.ID)
		aliases := c.Names
		ref := info.ContainerReference{
			Name:    path,
			Aliases: aliases,
		}
		ret = append(ret, ref)
	}
	if self.isRootContainer() {
		ret = append(ret, info.ContainerReference{Name: "/docker"})
	}
	return ret, nil
}

func (self *dockerContainerHandler) ListThreads(listType container.ListType) ([]int, error) {
	return nil, nil
}

func (self *dockerContainerHandler) ListProcesses(listType container.ListType) ([]int, error) {
	return nil, nil
}
