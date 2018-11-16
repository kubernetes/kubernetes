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
	"runtime"
	"sync"
	"time"
	"unsafe"

	cadvisorapi "github.com/google/cadvisor/info/v1"
	"golang.org/x/sys/windows"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/klog"
)

// MemoryStatusEx is the same as Windows structure MEMORYSTATUSEX
// https://msdn.microsoft.com/en-us/library/windows/desktop/aa366770(v=vs.85).aspx
type MemoryStatusEx struct {
	Length               uint32
	MemoryLoad           uint32
	TotalPhys            uint64
	AvailPhys            uint64
	TotalPageFile        uint64
	AvailPageFile        uint64
	TotalVirtual         uint64
	AvailVirtual         uint64
	AvailExtendedVirtual uint64
}

var (
	modkernel32              = windows.NewLazySystemDLL("kernel32.dll")
	procGlobalMemoryStatusEx = modkernel32.NewProc("GlobalMemoryStatusEx")
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

	kernelVersion, err := getKernelVersion()
	if err != nil {
		return err
	}

	osImageVersion, err := getOSImageVersion()
	if err != nil {
		return err
	}

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
		klog.Errorf("Unable to get cpu perf counter data; err: %v", err)
		return
	}

	memWorkingSetValue, err := memWorkingSetCounter.getData()
	if err != nil {
		klog.Errorf("Unable to get memWorkingSet perf counter data; err: %v", err)
		return
	}

	memCommittedBytesValue, err := memCommittedBytesCounter.getData()
	if err != nil {
		klog.Errorf("Unable to get memCommittedBytes perf counter data; err: %v", err)
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
	// We use GlobalMemoryStatusEx instead of GetPhysicallyInstalledSystemMemory
	// on Windows node for the following reasons:
	// 1. GetPhysicallyInstalledSystemMemory retrieves the amount of physically
	// installed RAM from the computer's SMBIOS firmware tables.
	// https://msdn.microsoft.com/en-us/library/windows/desktop/cc300158(v=vs.85).aspx
	// On some VM, it is unable to read data from SMBIOS and fails with ERROR_INVALID_DATA.
	// 2. On Linux node, total physical memory is read from MemTotal in /proc/meminfo.
	// GlobalMemoryStatusEx returns the amount of physical memory that is available
	// for the operating system to use. The amount returned by GlobalMemoryStatusEx
	// is closer in parity with Linux
	// https://www.kernel.org/doc/Documentation/filesystems/proc.txt
	var statex MemoryStatusEx
	statex.Length = uint32(unsafe.Sizeof(statex))
	ret, _, _ := procGlobalMemoryStatusEx.Call(uintptr(unsafe.Pointer(&statex)))

	if ret == 0 {
		return 0, errors.New("unable to read physical memory")
	}

	return statex.TotalPhys, nil
}
