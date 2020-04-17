// +build windows

/*
Copyright 2017 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

// Package winstats provides a client to get node and pod level stats on windows
package winstats

import (
	"syscall"
	"time"
	"unsafe"

	cadvisorapi "github.com/google/cadvisor/info/v1"
	cadvisorapiv2 "github.com/google/cadvisor/info/v2"
)

var (
	procGetDiskFreeSpaceEx = modkernel32.NewProc("GetDiskFreeSpaceExW")
)

// Client is an interface that is used to get stats information.
type Client interface {
	WinContainerInfos() (map[string]cadvisorapiv2.ContainerInfo, error)
	WinMachineInfo() (*cadvisorapi.MachineInfo, error)
	WinVersionInfo() (*cadvisorapi.VersionInfo, error)
	GetDirFsInfo(path string) (cadvisorapiv2.FsInfo, error)
}

// StatsClient is a client that implements the Client interface
type StatsClient struct {
	client winNodeStatsClient
}

type winNodeStatsClient interface {
	startMonitoring() error
	getNodeMetrics() (nodeMetrics, error)
	getNodeInfo() nodeInfo
	getMachineInfo() (*cadvisorapi.MachineInfo, error)
	getVersionInfo() (*cadvisorapi.VersionInfo, error)
}

type nodeMetrics struct {
	cpuUsageCoreNanoSeconds   uint64
	cpuUsageNanoCores         uint64
	memoryPrivWorkingSetBytes uint64
	memoryCommittedBytes      uint64
	timeStamp                 time.Time
	interfaceStats            []cadvisorapi.InterfaceStats
}

type nodeInfo struct {
	memoryPhysicalCapacityBytes uint64
	kernelVersion               string
	osImageVersion              string
	// startTime is the time when the node was started
	startTime time.Time
}

type cpuUsageCoreNanoSecondsCache struct {
	latestValue   uint64
	previousValue uint64
}

// newClient constructs a Client.
func newClient(statsNodeClient winNodeStatsClient) (Client, error) {
	statsClient := new(StatsClient)
	statsClient.client = statsNodeClient

	err := statsClient.client.startMonitoring()
	if err != nil {
		return nil, err
	}

	return statsClient, nil
}

// WinContainerInfos returns a map of container infos. The map contains node and
// pod level stats. Analogous to cadvisor GetContainerInfoV2 method.
func (c *StatsClient) WinContainerInfos() (map[string]cadvisorapiv2.ContainerInfo, error) {
	infos := make(map[string]cadvisorapiv2.ContainerInfo)
	rootContainerInfo, err := c.createRootContainerInfo()
	if err != nil {
		return nil, err
	}

	infos["/"] = *rootContainerInfo

	return infos, nil
}

// WinMachineInfo returns a cadvisorapi.MachineInfo with details about the
// node machine. Analogous to cadvisor MachineInfo method.
func (c *StatsClient) WinMachineInfo() (*cadvisorapi.MachineInfo, error) {
	return c.client.getMachineInfo()
}

// WinVersionInfo returns a  cadvisorapi.VersionInfo with version info of
// the kernel and docker runtime. Analogous to cadvisor VersionInfo method.
func (c *StatsClient) WinVersionInfo() (*cadvisorapi.VersionInfo, error) {
	return c.client.getVersionInfo()
}

func (c *StatsClient) createRootContainerInfo() (*cadvisorapiv2.ContainerInfo, error) {
	nodeMetrics, err := c.client.getNodeMetrics()
	if err != nil {
		return nil, err
	}

	var stats []*cadvisorapiv2.ContainerStats
	stats = append(stats, &cadvisorapiv2.ContainerStats{
		Timestamp: nodeMetrics.timeStamp,
		Cpu: &cadvisorapi.CpuStats{
			Usage: cadvisorapi.CpuUsage{
				Total: nodeMetrics.cpuUsageCoreNanoSeconds,
			},
		},
		CpuInst: &cadvisorapiv2.CpuInstStats{
			Usage: cadvisorapiv2.CpuInstUsage{
				Total: nodeMetrics.cpuUsageNanoCores,
			},
		},
		Memory: &cadvisorapi.MemoryStats{
			WorkingSet: nodeMetrics.memoryPrivWorkingSetBytes,
			Usage:      nodeMetrics.memoryCommittedBytes,
		},
		Network: &cadvisorapiv2.NetworkStats{
			Interfaces: nodeMetrics.interfaceStats,
		},
	})

	nodeInfo := c.client.getNodeInfo()
	rootInfo := cadvisorapiv2.ContainerInfo{
		Spec: cadvisorapiv2.ContainerSpec{
			CreationTime: nodeInfo.startTime,
			HasCpu:       true,
			HasMemory:    true,
			HasNetwork:   true,
			Memory: cadvisorapiv2.MemorySpec{
				Limit: nodeInfo.memoryPhysicalCapacityBytes,
			},
		},
		Stats: stats,
	}

	return &rootInfo, nil
}

// GetDirFsInfo returns filesystem capacity and usage information.
func (c *StatsClient) GetDirFsInfo(path string) (cadvisorapiv2.FsInfo, error) {
	var freeBytesAvailable, totalNumberOfBytes, totalNumberOfFreeBytes int64
	var err error

	ret, _, err := syscall.Syscall6(
		procGetDiskFreeSpaceEx.Addr(),
		4,
		uintptr(unsafe.Pointer(syscall.StringToUTF16Ptr(path))),
		uintptr(unsafe.Pointer(&freeBytesAvailable)),
		uintptr(unsafe.Pointer(&totalNumberOfBytes)),
		uintptr(unsafe.Pointer(&totalNumberOfFreeBytes)),
		0,
		0,
	)
	if ret == 0 {
		return cadvisorapiv2.FsInfo{}, err
	}

	return cadvisorapiv2.FsInfo{
		Timestamp: time.Now(),
		Capacity:  uint64(totalNumberOfBytes),
		Available: uint64(freeBytesAvailable),
		Usage:     uint64(totalNumberOfBytes - freeBytesAvailable),
	}, nil
}
