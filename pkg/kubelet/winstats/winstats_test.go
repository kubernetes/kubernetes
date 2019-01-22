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

package winstats

import (
	"testing"
	"time"

	cadvisorapi "github.com/google/cadvisor/info/v1"
	cadvisorapiv2 "github.com/google/cadvisor/info/v2"
	"github.com/stretchr/testify/assert"
)

var timeStamp = time.Now()

type fakeWinNodeStatsClient struct{}

func (f fakeWinNodeStatsClient) startMonitoring() error {
	return nil
}

func (f fakeWinNodeStatsClient) getNodeMetrics() (nodeMetrics, error) {
	return nodeMetrics{
		cpuUsageCoreNanoSeconds:   123,
		memoryPrivWorkingSetBytes: 1234,
		memoryCommittedBytes:      12345,
		timeStamp:                 timeStamp,
	}, nil
}

func (f fakeWinNodeStatsClient) getNodeInfo() nodeInfo {
	return nodeInfo{
		kernelVersion:               "v42",
		memoryPhysicalCapacityBytes: 1.6e+10,
	}
}
func (f fakeWinNodeStatsClient) getMachineInfo() (*cadvisorapi.MachineInfo, error) {
	return &cadvisorapi.MachineInfo{
		NumCores:       4,
		MemoryCapacity: 1.6e+10,
		MachineID:      "somehostname",
	}, nil
}

func (f fakeWinNodeStatsClient) getVersionInfo() (*cadvisorapi.VersionInfo, error) {
	return &cadvisorapi.VersionInfo{
		KernelVersion: "v42",
	}, nil
}

func TestWinContainerInfos(t *testing.T) {
	c := getClient(t)

	actualRootInfos, err := c.WinContainerInfos()
	assert.NoError(t, err)

	var stats []*cadvisorapiv2.ContainerStats
	stats = append(stats, &cadvisorapiv2.ContainerStats{
		Timestamp: timeStamp,
		Cpu: &cadvisorapi.CpuStats{
			Usage: cadvisorapi.CpuUsage{
				Total: 123,
			},
		},
		Memory: &cadvisorapi.MemoryStats{
			WorkingSet: 1234,
			Usage:      12345,
		},
	})
	infos := make(map[string]cadvisorapiv2.ContainerInfo)
	infos["/"] = cadvisorapiv2.ContainerInfo{
		Spec: cadvisorapiv2.ContainerSpec{
			HasCpu:    true,
			HasMemory: true,
			Memory: cadvisorapiv2.MemorySpec{
				Limit: 1.6e+10,
			},
		},
		Stats: stats,
	}

	assert.Equal(t, actualRootInfos, infos)
}

func TestWinMachineInfo(t *testing.T) {
	c := getClient(t)

	machineInfo, err := c.WinMachineInfo()
	assert.NoError(t, err)
	assert.Equal(t, machineInfo, &cadvisorapi.MachineInfo{
		NumCores:       4,
		MemoryCapacity: 1.6e+10,
		MachineID:      "somehostname"})
}

func TestWinVersionInfo(t *testing.T) {
	c := getClient(t)

	versionInfo, err := c.WinVersionInfo()
	assert.NoError(t, err)
	assert.Equal(t, versionInfo, &cadvisorapi.VersionInfo{
		KernelVersion: "v42"})
}

func getClient(t *testing.T) Client {
	f := fakeWinNodeStatsClient{}
	c, err := newClient(f)
	assert.NoError(t, err)
	assert.NotNil(t, c)
	return c
}
