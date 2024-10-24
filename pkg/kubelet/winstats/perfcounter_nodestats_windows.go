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
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	kubefeatures "k8s.io/kubernetes/pkg/features"
	"os"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"syscall"
	"time"
	"unsafe"

	cadvisorapi "github.com/google/cadvisor/info/v1"
	"github.com/pkg/errors"
	"golang.org/x/sys/windows"
	"golang.org/x/sys/windows/registry"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/klog/v2"
)

const (
	bootIdRegistry = `SYSTEM\CurrentControlSet\Control\Session Manager\Memory Management\PrefetchParameters`
	bootIdKey      = `BootId`
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

// PerformanceInfo is the same as Windows structure PERFORMANCE_INFORMATION
// https://learn.microsoft.com/en-us/windows/win32/api/psapi/ns-psapi-performance_information
type PerformanceInformation struct {
	cb                     uint32
	CommitTotalPages       uint64
	CommitLimitPages       uint64
	CommitPeakPages        uint64
	PhysicalTotalPages     uint64
	PhysicalAvailablePages uint64
	SystemCachePages       uint64
	KernelTotalPages       uint64
	KernelPagesPages       uint64
	KernelNonpagedPages    uint64
	PageSize               uint64
	HandleCount            uint32
	ProcessCount           uint32
	ThreadCount            uint32
}

var (
	// kernel32.dll system calls
	modkernel32                 = windows.NewLazySystemDLL("kernel32.dll")
	procGlobalMemoryStatusEx    = modkernel32.NewProc("GlobalMemoryStatusEx")
	procGetActiveProcessorCount = modkernel32.NewProc("GetActiveProcessorCount")
	// psapi.dll system calls
	modpsapi               = windows.NewLazySystemDLL("psapi.dll")
	procGetPerformanceInfo = modpsapi.NewProc("GetPerformanceInfo")
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

	bootId, err := getBootID()
	if err != nil {
		return nil, err
	}

	mi := &cadvisorapi.MachineInfo{
		NumCores:       ProcessorCount(),
		MemoryCapacity: p.nodeInfo.memoryPhysicalCapacityBytes,
		MachineID:      hostname,
		SystemUUID:     systemUUID,
		BootID:         bootId,
	}

	if utilfeature.DefaultFeatureGate.Enabled(kubefeatures.WindowsCPUAndMemoryAffinity) {
		numOfPysicalCores, numOfSockets, topology, err := processorInfo(RelationAll)
		if err != nil {
			return nil, err
		}

		mi.NumPhysicalCores = numOfPysicalCores
		mi.NumSockets = numOfSockets
		mi.Topology = topology
	}

	return mi, nil
}

// runtime.NumCPU() will only return the information for a single Processor Group.
// Since a single group can only hold 64 logical processors, this
// means when there are more they will be divided into multiple groups.
// For the above reason, procGetActiveProcessorCount is used to get the
// cpu count for all processor groups of the windows node.
// more notes for this issue:
// same issue in moby: https://github.com/moby/moby/issues/38935#issuecomment-744638345
// solution in hcsshim: https://github.com/microsoft/hcsshim/blob/master/internal/processorinfo/processor_count.go
func ProcessorCount() int {
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

func (p *perfCounterNodeStatsClient) collectMetricsData(cpuCounter, memWorkingSetCounter, memCommittedBytesCounter perfCounter, networkAdapterCounter *networkCounter) {
	cpuValue, err := cpuCounter.getData()
	cpuCores := ProcessorCount()
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
	perfCounterUpdatePeriodSeconds := uint64(perfCounterUpdatePeriod / time.Second)
	cpuUsageNanoCores := ((p.cpuUsageCoreNanoSecondsCache.latestValue - p.cpuUsageCoreNanoSecondsCache.previousValue) * perfCounterUpdatePeriodSeconds) / cachePeriodSeconds
	return cpuUsageNanoCores
}

func getSystemUUID() (string, error) {
	k, err := registry.OpenKey(registry.LOCAL_MACHINE, `SYSTEM\HardwareConfig`, registry.QUERY_VALUE)
	if err != nil {
		return "", errors.Wrap(err, "failed to open registry key HKLM\\SYSTEM\\HardwareConfig")
	}
	defer k.Close()

	uuid, _, err := k.GetStringValue("LastConfig")
	if err != nil {
		return "", errors.Wrap(err, "failed to read registry value LastConfig from key HKLM\\SYSTEM\\HardwareConfig")
	}

	uuid = strings.Trim(uuid, "{")
	uuid = strings.Trim(uuid, "}")
	uuid = strings.ToUpper(uuid)
	return uuid, nil
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

func GetPerformanceInfo() (*PerformanceInformation, error) {
	var pi PerformanceInformation
	pi.cb = uint32(unsafe.Sizeof(pi))
	ret, _, _ := procGetPerformanceInfo.Call(uintptr(unsafe.Pointer(&pi)), uintptr(pi.cb))
	if ret == 0 {
		return nil, errors.New("unable to read Windows performance information")
	}
	return &pi, nil
}

func getBootID() (string, error) {
	regKey, err := registry.OpenKey(registry.LOCAL_MACHINE, bootIdRegistry, registry.READ)
	if err != nil {
		return "", err
	}
	defer regKey.Close()
	regValue, _, err := regKey.GetIntegerValue(bootIdKey)
	if err != nil {
		return "", err
	}
	return strconv.FormatUint(regValue, 10), nil
}
