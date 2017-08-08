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
	"errors"
	"os"
	"os/exec"
	"runtime"
	"strings"
	"sync"
	"time"

	"github.com/golang/glog"
	cadvisorapi "github.com/google/cadvisor/info/v1"
	"github.com/lxn/win"
	"k8s.io/apimachinery/pkg/util/wait"
)

// NewPerfCounterClient creates a client using perf counters
func NewPerfCounterClient() (Client, error) {
	return newClient(&perfCounterNodeStatsClient{})
}

// perfCounterNodeStatsClient is a client that provides Windows Stats via PerfCounters
type perfCounterNodeStatsClient struct {
	nodeMetrics
	mu sync.RWMutex // mu protects nodeMetrics
	nodeInfo
}

func (p *perfCounterNodeStatsClient) startMonitoring() error {
	memory, err := getPhysicallyInstalledSystemMemoryBytes()
	if err != nil {
		return err
	}

	version, err := exec.Command("cmd", "/C", "ver").Output()
	if err != nil {
		return err
	}

	osImageVersion := strings.TrimSpace(string(version))
	kernelVersion := extractVersionNumber(osImageVersion)
	p.nodeInfo = nodeInfo{
		kernelVersion:               kernelVersion,
		osImageVersion:              osImageVersion,
		memoryPhysicalCapacityBytes: memory,
		startTime:                   time.Now(),
	}

	cpuCounter, err := newPerfCounter(cpuQuery)
	if err != nil {
		return err
	}

	memWorkingSetCounter, err := newPerfCounter(memoryPrivWorkingSetQuery)
	if err != nil {
		return err
	}

	memCommittedBytesCounter, err := newPerfCounter(memoryCommittedBytesQuery)
	if err != nil {
		return err
	}

	go wait.Forever(func() {
		p.collectMetricsData(cpuCounter, memWorkingSetCounter, memCommittedBytesCounter)
	}, perfCounterUpdatePeriod)

	return nil
}

func (p *perfCounterNodeStatsClient) getMachineInfo() (*cadvisorapi.MachineInfo, error) {
	hostname, err := os.Hostname()
	if err != nil {
		return nil, err
	}

	return &cadvisorapi.MachineInfo{
		NumCores:       runtime.NumCPU(),
		MemoryCapacity: p.nodeInfo.memoryPhysicalCapacityBytes,
		MachineID:      hostname,
	}, nil
}

func (p *perfCounterNodeStatsClient) getVersionInfo() (*cadvisorapi.VersionInfo, error) {
	return &cadvisorapi.VersionInfo{
		KernelVersion:      p.nodeInfo.kernelVersion,
		ContainerOsVersion: p.nodeInfo.osImageVersion,
	}, nil
}

func (p *perfCounterNodeStatsClient) getNodeMetrics() (nodeMetrics, error) {
	p.mu.RLock()
	defer p.mu.RUnlock()
	return p.nodeMetrics, nil
}

func (p *perfCounterNodeStatsClient) getNodeInfo() nodeInfo {
	return p.nodeInfo
}

func (p *perfCounterNodeStatsClient) collectMetricsData(cpuCounter, memWorkingSetCounter, memCommittedBytesCounter *perfCounter) {
	cpuValue, err := cpuCounter.getData()
	if err != nil {
		glog.Errorf("Unable to get cpu perf counter data; err: %v", err)
		return
	}

	memWorkingSetValue, err := memWorkingSetCounter.getData()
	if err != nil {
		glog.Errorf("Unable to get memWorkingSet perf counter data; err: %v", err)
		return
	}

	memCommittedBytesValue, err := memCommittedBytesCounter.getData()
	if err != nil {
		glog.Errorf("Unable to get memCommittedBytes perf counter data; err: %v", err)
		return
	}

	p.mu.Lock()
	defer p.mu.Unlock()
	p.nodeMetrics = nodeMetrics{
		cpuUsageCoreNanoSeconds:   p.convertCPUValue(cpuValue),
		memoryPrivWorkingSetBytes: memWorkingSetValue,
		memoryCommittedBytes:      memCommittedBytesValue,
		timeStamp:                 time.Now(),
	}
}

func (p *perfCounterNodeStatsClient) convertCPUValue(cpuValue uint64) uint64 {
	cpuCores := runtime.NumCPU()
	// This converts perf counter data which is cpu percentage for all cores into nanoseconds.
	// The formula is (cpuPercentage / 100.0) * #cores * 1e+9 (nano seconds). More info here:
	// https://github.com/kubernetes/heapster/issues/650
	newValue := p.nodeMetrics.cpuUsageCoreNanoSeconds + uint64((float64(cpuValue)/100.0)*float64(cpuCores)*1e9)
	return newValue
}

func getPhysicallyInstalledSystemMemoryBytes() (uint64, error) {
	var physicalMemoryKiloBytes uint64

	if ok := win.GetPhysicallyInstalledSystemMemory(&physicalMemoryKiloBytes); !ok {
		return 0, errors.New("unable to read physical memory")
	}

	return physicalMemoryKiloBytes * 1024, nil // convert kilobytes to bytes
}
