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
	"regexp"
	"time"

	cadvisorapi "github.com/google/cadvisor/info/v1"
	cadvisorapiv2 "github.com/google/cadvisor/info/v2"
)

// Client is an interface that is used to get stats information.
type Client interface {
	WinContainerInfos() (map[string]cadvisorapiv2.ContainerInfo, error)
	WinMachineInfo() (*cadvisorapi.MachineInfo, error)
	WinVersionInfo() (*cadvisorapi.VersionInfo, error)
}

// StatsClient is a client that implements the Client interface
type statsClient struct {
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
	memoryPrivWorkingSetBytes uint64
	memoryCommittedBytes      uint64
	timeStamp                 time.Time
}

type nodeInfo struct {
	memoryPhysicalCapacityBytes uint64
	kernelVersion               string
	osImageVersion              string
	// startTime is the time when the node was started
	startTime time.Time
}

// newClient constructs a Client.
func newClient(statsNodeClient winNodeStatsClient) (Client, error) {
	statsClient := new(statsClient)
	statsClient.client = statsNodeClient

	err := statsClient.client.startMonitoring()
	if err != nil {
		return nil, err
	}

	return statsClient, nil
}

// WinContainerInfos returns a map of container infos. The map contains node and
// pod level stats. Analogous to cadvisor GetContainerInfoV2 method.
func (c *statsClient) WinContainerInfos() (map[string]cadvisorapiv2.ContainerInfo, error) {
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
func (c *statsClient) WinMachineInfo() (*cadvisorapi.MachineInfo, error) {
	return c.client.getMachineInfo()
}

// WinVersionInfo returns a  cadvisorapi.VersionInfo with version info of
// the kernel and docker runtime. Analogous to cadvisor VersionInfo method.
func (c *statsClient) WinVersionInfo() (*cadvisorapi.VersionInfo, error) {
	return c.client.getVersionInfo()
}

func (c *statsClient) createRootContainerInfo() (*cadvisorapiv2.ContainerInfo, error) {
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
		Memory: &cadvisorapi.MemoryStats{
			WorkingSet: nodeMetrics.memoryPrivWorkingSetBytes,
			Usage:      nodeMetrics.memoryCommittedBytes,
		},
	})

	nodeInfo := c.client.getNodeInfo()
	rootInfo := cadvisorapiv2.ContainerInfo{
		Spec: cadvisorapiv2.ContainerSpec{
			CreationTime: nodeInfo.startTime,
			HasCpu:       true,
			HasMemory:    true,
			Memory: cadvisorapiv2.MemorySpec{
				Limit: nodeInfo.memoryPhysicalCapacityBytes,
			},
		},
		Stats: stats,
	}

	return &rootInfo, nil
}

// extractVersionNumber gets the version number from the full version string on Windows
// e.g. extracts "10.0.14393" from "Microsoft Windows [Version 10.0.14393]"
func extractVersionNumber(fullVersion string) string {
	re := regexp.MustCompile("[^0-9.]")
	version := re.ReplaceAllString(fullVersion, "")
	return version
}
