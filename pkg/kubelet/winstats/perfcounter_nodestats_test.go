//go:build windows

/*
Copyright 2023 The Kubernetes Authors.

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
	"os"
	"os/exec"
	"strconv"
	"strings"
	"testing"
	"time"

	cadvisorapi "github.com/google/cadvisor/info/v1"
	"github.com/google/uuid"
	"github.com/stretchr/testify/assert"

	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/klog/v2/ktesting"
)

func TestMonitoring(t *testing.T) {
	logger, _ := ktesting.NewTestContext(t)
	counterClient, err := NewPerfCounterClient(logger)
	assert.NoError(t, err)

	// assert that startMonitoring has been called. nodeInfo should be set.
	assert.NotNil(t, counterClient.(*StatsClient).client.getNodeInfo())

	// Wait until we get a non-zero node metrics.
	if pollErr := wait.Poll(100*time.Millisecond, 5*perfCounterUpdatePeriod, func() (bool, error) {
		metrics, _ := counterClient.(*StatsClient).client.getNodeMetrics()
		if metrics.memoryPrivWorkingSetBytes != 0 {
			return true, nil
		}

		return false, nil
	}); pollErr != nil {
		t.Fatalf("Encountered error: `%v'", pollErr)
	}
}

func TestGetMachineInfo(t *testing.T) {
	logger, _ := ktesting.NewTestContext(t)
	p := perfCounterNodeStatsClient{
		nodeInfo: nodeInfo{
			memoryPhysicalCapacityBytes: 100,
		},
	}

	machineInfo, err := p.getMachineInfo(logger)
	assert.NoError(t, err)
	assert.Equal(t, uint64(100), machineInfo.MemoryCapacity)
	hostname, _ := os.Hostname()
	assert.Equal(t, hostname, machineInfo.MachineID)

	// Check if it's an UUID.
	_, err = uuid.Parse(machineInfo.SystemUUID)
	assert.NoError(t, err)

	id, err := strconv.Atoi(machineInfo.BootID)
	assert.NoError(t, err)
	assert.NotZero(t, id)
}

func TestGetVersionInfo(t *testing.T) {
	client := perfCounterNodeStatsClient{
		nodeInfo: nodeInfo{
			kernelVersion:  "foo",
			osImageVersion: "lish",
		},
	}

	info, _ := client.getVersionInfo()
	expected := &cadvisorapi.VersionInfo{
		KernelVersion:      "foo",
		ContainerOsVersion: "lish",
	}
	assert.Equal(t, expected, info)
}

func TestCollectMetricsData(t *testing.T) {
	logger, _ := ktesting.NewTestContext(t)
	p := perfCounterNodeStatsClient{}

	cpuCounter := &fakePerfCounterImpl{
		value:      1,
		raiseError: true,
	}
	memWorkingSetCounter := &fakePerfCounterImpl{
		value:      2,
		raiseError: true,
	}
	memCommittedBytesCounter := &fakePerfCounterImpl{
		value:      3,
		raiseError: true,
	}
	networkAdapterCounter := newFakedNetworkCounters(true)

	// Checking the error cases first.
	p.collectMetricsData(logger, cpuCounter, memWorkingSetCounter, memCommittedBytesCounter, networkAdapterCounter)
	metrics, _ := p.getNodeMetrics()
	expectedMetrics := nodeMetrics{}
	assert.Equal(t, expectedMetrics, metrics)

	cpuCounter.raiseError = false
	p.collectMetricsData(logger, cpuCounter, memWorkingSetCounter, memCommittedBytesCounter, networkAdapterCounter)
	metrics, _ = p.getNodeMetrics()
	assert.Equal(t, expectedMetrics, metrics)

	memWorkingSetCounter.raiseError = false
	p.collectMetricsData(logger, cpuCounter, memWorkingSetCounter, memCommittedBytesCounter, networkAdapterCounter)
	metrics, _ = p.getNodeMetrics()
	assert.Equal(t, expectedMetrics, metrics)

	memCommittedBytesCounter.raiseError = false
	p.collectMetricsData(logger, cpuCounter, memWorkingSetCounter, memCommittedBytesCounter, networkAdapterCounter)
	metrics, _ = p.getNodeMetrics()
	assert.Equal(t, expectedMetrics, metrics)

	networkAdapterCounter = newFakedNetworkCounters(false)
	p.collectMetricsData(logger, cpuCounter, memWorkingSetCounter, memCommittedBytesCounter, networkAdapterCounter)
	metrics, _ = p.getNodeMetrics()
	expectedMetrics = nodeMetrics{
		cpuUsageCoreNanoSeconds:   uint64(ProcessorCount()) * 1e7,
		cpuUsageNanoCores:         0,
		memoryPrivWorkingSetBytes: 2,
		memoryCommittedBytes:      3,
		interfaceStats:            networkAdapterCounter.listInterfaceStats(),
	}
	assert.WithinDuration(t, time.Now(), metrics.timeStamp, 20*time.Millisecond)
	expectedMetrics.timeStamp = metrics.timeStamp
	assert.Equal(t, expectedMetrics, metrics)
}

func TestConvertCPUValue(t *testing.T) {
	testCases := []struct {
		cpuValue uint64
		expected uint64
	}{
		{cpuValue: uint64(50), expected: uint64(2000000000)},
		{cpuValue: uint64(0), expected: uint64(0)},
		{cpuValue: uint64(100), expected: uint64(4000000000)},
	}
	var cpuCores = 4

	for _, tc := range testCases {
		p := perfCounterNodeStatsClient{}
		newValue := p.convertCPUValue(cpuCores, tc.cpuValue)
		assert.Equal(t, tc.expected, newValue)
	}
}

func TestGetCPUUsageNanoCores(t *testing.T) {
	// Scaled expected unit test values by the frequency the CPU perf counters are polled.
	perfCounterUpdatePeriodSeconds := uint64(perfCounterUpdatePeriod / time.Second)
	testCases := []struct {
		latestValue   uint64
		previousValue uint64
		expected      uint64
	}{
		{latestValue: uint64(0), previousValue: uint64(0), expected: uint64(0)},
		{latestValue: uint64(2000000000), previousValue: uint64(0), expected: uint64(200000000 * perfCounterUpdatePeriodSeconds)},
		{latestValue: uint64(5000000000), previousValue: uint64(2000000000), expected: uint64(300000000 * perfCounterUpdatePeriodSeconds)},
	}

	for _, tc := range testCases {
		p := perfCounterNodeStatsClient{}
		p.cpuUsageCoreNanoSecondsCache = cpuUsageCoreNanoSecondsCache{
			latestValue:   tc.latestValue,
			previousValue: tc.previousValue,
		}
		cpuUsageNanoCores := p.getCPUUsageNanoCores()
		assert.Equal(t, tc.expected, cpuUsageNanoCores)
	}
}

func testGetPhysicallyInstalledSystemMemoryBytes(t *testing.T) {
	totalMemory, err := getPhysicallyInstalledSystemMemoryBytes()
	assert.NoError(t, err)
	assert.NotZero(t, totalMemory)
}

func TestGetSystemUUID(t *testing.T) {
	uuidFromRegistry, err := getSystemUUID()
	assert.NoError(t, err)

	uuidFromWmi, err := exec.Command("powershell.exe", "Get-WmiObject", "Win32_ComputerSystemProduct", "|", "Select-Object", "-ExpandProperty UUID").Output()
	assert.NoError(t, err)
	uuidFromWmiString := strings.Trim(string(uuidFromWmi), "\r\n")
	assert.Equal(t, uuidFromWmiString, uuidFromRegistry)
}
