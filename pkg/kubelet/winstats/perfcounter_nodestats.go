//go:build windows
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
	"fmt"
	"os"
	"runtime"
	"strings"
	"sync"
	"syscall"
	"time"
	"unsafe"

	"github.com/go-ole/go-ole"
	"github.com/go-ole/go-ole/oleutil"
	cadvisorapi "github.com/google/cadvisor/info/v1"
	"golang.org/x/sys/windows"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/klog/v2"
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
	modkernel32                 = windows.NewLazySystemDLL("kernel32.dll")
	procGlobalMemoryStatusEx    = modkernel32.NewProc("GlobalMemoryStatusEx")
	procGetActiveProcessorCount = modkernel32.NewProc("GetActiveProcessorCount")
)

const allProcessorGroups = 0xFFFF

// NewPerfCounterClient creates a client using perf counters
func NewPerfCounterClient() (Client, error) {
	// Initialize the cache
	initCache := cpuUsageCoreNanoSecondsCache{0, 0}
	return newClient(&perfCounterNodeStatsClient{
		cpuUsageCoreNanoSecondsCache: initCache,
	})
}

// perfCounterNodeStatsClient is a client that provides Windows Stats via PerfCounters
type perfCounterNodeStatsClient struct {
	nodeMetrics
	mu sync.RWMutex // mu protects nodeMetrics
	nodeInfo
	// cpuUsageCoreNanoSecondsCache caches the cpu usage for nodes.
	cpuUsageCoreNanoSecondsCache
}

func (p *perfCounterNodeStatsClient) startMonitoring() error {
	memory, err := getPhysicallyInstalledSystemMemoryBytes()
	if err != nil {
		return err
	}

	osInfo, err := GetOSInfo()
	if err != nil {
		return err
	}

	p.nodeInfo = nodeInfo{
		kernelVersion:               osInfo.GetPatchVersion(),
		osImageVersion:              osInfo.ProductName,
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

	networkAdapterCounter, err := newNetworkCounters()
	if err != nil {
		return err
	}

	go wait.Forever(func() {
		p.collectMetricsData(cpuCounter, memWorkingSetCounter, memCommittedBytesCounter, networkAdapterCounter)
	}, perfCounterUpdatePeriod)

	// Cache the CPU usage every defaultCachePeriod
	go wait.Forever(func() {
		newValue := p.nodeMetrics.cpuUsageCoreNanoSeconds
		p.mu.Lock()
		defer p.mu.Unlock()
		p.cpuUsageCoreNanoSecondsCache = cpuUsageCoreNanoSecondsCache{
			previousValue: p.cpuUsageCoreNanoSecondsCache.latestValue,
			latestValue:   newValue,
		}
	}, defaultCachePeriod)

	return nil
}

func (p *perfCounterNodeStatsClient) getMachineInfo() (*cadvisorapi.MachineInfo, error) {
	hostname, err := os.Hostname()
	if err != nil {
		return nil, err
	}

	systemUUID, err := getSystemUUID()
	if err != nil {
		return nil, err
	}

	return &cadvisorapi.MachineInfo{
		NumCores:       processorCount(),
		MemoryCapacity: p.nodeInfo.memoryPhysicalCapacityBytes,
		MachineID:      hostname,
		SystemUUID:     systemUUID,
	}, nil
}

// runtime.NumCPU() will only return the information for a single Processor Group.
// Since a single group can only hold 64 logical processors, this
// means when there are more they will be divided into multiple groups.
// For the above reason, procGetActiveProcessorCount is used to get the
// cpu count for all processor groups of the windows node.
// more notes for this issue:
// same issue in moby: https://github.com/moby/moby/issues/38935#issuecomment-744638345
// solution in hcsshim: https://github.com/microsoft/hcsshim/blob/master/internal/processorinfo/processor_count.go
func processorCount() int {
	if amount := getActiveProcessorCount(allProcessorGroups); amount != 0 {
		return int(amount)
	}
	return runtime.NumCPU()
}

func getActiveProcessorCount(groupNumber uint16) int {
	r0, _, _ := syscall.Syscall(procGetActiveProcessorCount.Addr(), 1, uintptr(groupNumber), 0, 0)
	return int(r0)
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

func (p *perfCounterNodeStatsClient) collectMetricsData(cpuCounter, memWorkingSetCounter, memCommittedBytesCounter *perfCounter, networkAdapterCounter *networkCounter) {
	cpuValue, err := cpuCounter.getData()
	cpuCores := runtime.NumCPU()
	if err != nil {
		klog.ErrorS(err, "Unable to get cpu perf counter data")
		return
	}

	memWorkingSetValue, err := memWorkingSetCounter.getData()
	if err != nil {
		klog.ErrorS(err, "Unable to get memWorkingSet perf counter data")
		return
	}

	memCommittedBytesValue, err := memCommittedBytesCounter.getData()
	if err != nil {
		klog.ErrorS(err, "Unable to get memCommittedBytes perf counter data")
		return
	}

	networkAdapterStats, err := networkAdapterCounter.getData()
	if err != nil {
		klog.ErrorS(err, "Unable to get network adapter perf counter data")
		return
	}

	p.mu.Lock()
	defer p.mu.Unlock()
	p.nodeMetrics = nodeMetrics{
		cpuUsageCoreNanoSeconds:   p.convertCPUValue(cpuCores, cpuValue),
		cpuUsageNanoCores:         p.getCPUUsageNanoCores(),
		memoryPrivWorkingSetBytes: memWorkingSetValue,
		memoryCommittedBytes:      memCommittedBytesValue,
		interfaceStats:            networkAdapterStats,
		timeStamp:                 time.Now(),
	}
}

func (p *perfCounterNodeStatsClient) convertCPUValue(cpuCores int, cpuValue uint64) uint64 {
	// This converts perf counter data which is cpu percentage for all cores into nanoseconds.
	// The formula is (cpuPercentage / 100.0) * #cores * 1e+9 (nano seconds). More info here:
	// https://github.com/kubernetes/heapster/issues/650
	newValue := p.nodeMetrics.cpuUsageCoreNanoSeconds + uint64((float64(cpuValue)/100.0)*float64(cpuCores)*1e9)
	return newValue
}

func (p *perfCounterNodeStatsClient) getCPUUsageNanoCores() uint64 {
	cachePeriodSeconds := uint64(defaultCachePeriod / time.Second)
	cpuUsageNanoCores := (p.cpuUsageCoreNanoSecondsCache.latestValue - p.cpuUsageCoreNanoSecondsCache.previousValue) / cachePeriodSeconds
	return cpuUsageNanoCores
}

func getSystemUUID() (string, error) {
	obj, err := queryWMI("Win32_ComputerSystemProduct", "UUID")
	if err != nil {
		return "", err
	}

	if len(obj) == 0 {
		return "", fmt.Errorf("Could not get the System UUID.")
	}

	return obj[0]["UUID"], nil
}

// queryWMI returns an array of items, each item containing the selected properties requested.
func queryWMI(className string, selectedProperties ...string) ([]map[string]string, error) {
	if err := ole.CoInitialize(0); err != nil {
		oleerr := err.(*ole.OleError)

		// S_OK: The COM library was initialized successfully on this thread.
		// S_False (0x00000001): The COM library is already initialized on this thread.
		// S_False is not defined in the ole library.
		// See: https://docs.microsoft.com/en-us/windows/win32/api/combaseapi/nf-combaseapi-coinitializeex#return-value
		if oleerr.Code() != ole.S_OK && oleerr.Code() != 0x00000001 {
			return nil, err
		}
	} else {
		defer ole.CoUninitialize()
	}

	swbemLocator, _ := oleutil.CreateObject("WbemScripting.SWbemLocator")
	defer swbemLocator.Release()

	wmi, _ := swbemLocator.QueryInterface(ole.IID_IDispatch)
	defer wmi.Release()

	serviceRaw, _ := oleutil.CallMethod(wmi, "ConnectServer")
	service := serviceRaw.ToIDispatch()
	defer service.Release()

	query := fmt.Sprintf("Select %s from %s", strings.Join(selectedProperties, ","), className)
	resultRaw, err := oleutil.CallMethod(service, "ExecQuery", query)
	if err != nil {
		return nil, err
	}
	result := resultRaw.ToIDispatch()
	defer result.Release()

	countVar, err := oleutil.GetProperty(result, "Count")
	count := int(countVar.Val)

	values := make([]map[string]string, count)
	for i := 0; i < count; i++ {
		values[i] = make(map[string]string)
		itemRaw, err := oleutil.CallMethod(result, "ItemIndex", i)
		if err != nil {
			return nil, err
		}
		item := itemRaw.ToIDispatch()
		defer item.Release()

		for _, prop := range selectedProperties {
			val, err := oleutil.GetProperty(item, prop)
			if err != nil {
				return nil, err
			}
			values[i][prop] = val.ToString()
		}
	}

	return values, nil
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
