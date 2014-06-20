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

// A container object

package lmctfy

import (
	"fmt"
	"os/exec"
	"strings"
	"syscall"
	"time"

	"code.google.com/p/goprotobuf/proto"
	"github.com/google/cadvisor/container"
	"github.com/google/cadvisor/info"
)

type lmctfyContainerHandler struct {
	// Container name
	Name string
}

const (
	lmctfyBinary     = "lmctfy"
	notFoundExitCode = 5
)

// Create a new
func New(name string) (container.ContainerHandler, error) {
	el := &lmctfyContainerHandler{
		Name: name,
	}
	return el, nil
}

func (self *lmctfyContainerHandler) ContainerReference() (info.ContainerReference, error) {
	return info.ContainerReference{Name: self.Name}, nil
}

func getExitCode(err error) int {
	msg, ok := err.(*exec.ExitError)
	if ok {
		return msg.Sys().(syscall.WaitStatus).ExitStatus()
	}
	return -1
}

func protobufToContainerSpec(pspec *ContainerSpec) *info.ContainerSpec {
	ret := new(info.ContainerSpec)
	if pspec.GetCpu() != nil {
		cpuspec := new(info.CpuSpec)
		cpuspec.Limit = pspec.GetCpu().GetLimit()
		cpuspec.MaxLimit = pspec.GetCpu().GetMaxLimit()
		if pspec.GetCpu().GetMask() != nil {
			cpuspec.Mask.Data = pspec.GetCpu().GetMask().GetData()
		}
		ret.Cpu = cpuspec
	}
	if pspec.GetMemory() != nil {
		pmem := pspec.GetMemory()
		memspec := new(info.MemorySpec)
		memspec.Limit = uint64(pmem.GetLimit())
		memspec.Reservation = uint64(pmem.GetReservation())
		memspec.SwapLimit = uint64(pmem.GetSwapLimit())
		ret.Memory = memspec
	}
	return ret
}

// Gets spec.
func (c *lmctfyContainerHandler) GetSpec() (*info.ContainerSpec, error) {
	// Run lmctfy spec "container_name" and get spec.
	// Ignore if the container was not found.
	cmd := exec.Command(lmctfyBinary, "spec", string(c.Name))
	data, err := cmd.Output()
	if err != nil && getExitCode(err) != notFoundExitCode {
		return nil, fmt.Errorf("unable to run command %v spec %v: %v", lmctfyBinary, c.Name, err)
	}

	// Parse output into a protobuf.
	pspec := &ContainerSpec{}
	err = proto.UnmarshalText(string(data), pspec)
	if err != nil {
		return nil, err
	}
	spec := protobufToContainerSpec(pspec)
	return spec, nil
}

func protobufToMemoryData(pmd *MemoryStats_MemoryData, data *info.MemoryStatsMemoryData) {
	if pmd == nil {
		return
	}
	data.Pgfault = uint64(pmd.GetPgfault())
	data.Pgmajfault = uint64(pmd.GetPgmajfault())
	return
}

func protobufToContainerStats(pstats *ContainerStats) *info.ContainerStats {
	ret := new(info.ContainerStats)
	if pstats.GetCpu() != nil {
		pcpu := pstats.GetCpu()
		cpustats := new(info.CpuStats)
		cpustats.Usage.Total = pcpu.GetUsage().GetTotal()
		percpu := pcpu.GetUsage().GetPerCpu()
		if len(percpu) > 0 {
			cpustats.Usage.PerCpu = make([]uint64, len(percpu))
			for i, p := range percpu {
				cpustats.Usage.PerCpu[i] = uint64(p)
			}
		}
		cpustats.Usage.User = uint64(pcpu.GetUsage().GetUser())
		cpustats.Usage.System = uint64(pcpu.GetUsage().GetSystem())
		cpustats.Load = pcpu.GetLoad()
		ret.Cpu = cpustats
	}
	if pstats.GetMemory() != nil {
		pmem := pstats.GetMemory()
		memstats := new(info.MemoryStats)
		memstats.Limit = uint64(pmem.GetLimit())
		memstats.Usage = uint64(pmem.GetUsage())
		protobufToMemoryData(pmem.GetContainerData(), &memstats.ContainerData)
		protobufToMemoryData(pmem.GetHierarchicalData(), &memstats.HierarchicalData)
		ret.Memory = memstats
	}
	return ret
}

// Gets full stats.
func (c *lmctfyContainerHandler) GetStats() (*info.ContainerStats, error) {
	// Ignore if the container was not found.
	cmd := exec.Command(lmctfyBinary, "stats", "full", string(c.Name))
	data, err := cmd.Output()
	if err != nil && getExitCode(err) != notFoundExitCode {
		return nil, fmt.Errorf("unable to run command %v stats full %v: %v", lmctfyBinary, c.Name, err)
	}

	// Parse output into a protobuf.
	pstats := &ContainerStats{}
	err = proto.UnmarshalText(string(data), pstats)
	if err != nil {
		return nil, err
	}
	stats := protobufToContainerStats(pstats)
	stats.Timestamp = time.Now()
	return stats, nil
}

// Gets all subcontainers.
func (c *lmctfyContainerHandler) ListContainers(listType container.ListType) ([]info.ContainerReference, error) {
	// Prepare the arguments.
	args := []string{"list", "containers", "-v"}
	if listType == container.LIST_RECURSIVE {
		args = append(args, "-r")
	}
	args = append(args, c.Name)

	// Run the command.
	cmd := exec.Command(lmctfyBinary, args...)
	data, err := cmd.Output()
	if err != nil && getExitCode(err) != notFoundExitCode {
		return nil, err
	}

	// Parse lines as container names.
	if len(data) == 0 {
		return nil, nil
	}
	names := strings.Split(string(data), "\n")
	containerNames := make([]info.ContainerReference, 0, len(names))
	for _, name := range names {
		if len(name) != 0 {
			ref := info.ContainerReference{Name: name}
			containerNames = append(containerNames, ref)
		}
	}
	return containerNames, nil
}

// TODO(vmarmol): Implement
func (c *lmctfyContainerHandler) ListThreads(listType container.ListType) ([]int, error) {
	return []int{}, nil
}
func (c *lmctfyContainerHandler) ListProcesses(listType container.ListType) ([]int, error) {
	return []int{}, nil
}
