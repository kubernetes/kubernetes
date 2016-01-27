// Copyright (c) 2015-2016, NVIDIA CORPORATION. All rights reserved.

package nvml

// #cgo LDFLAGS: -lnvidia-ml
// #include "nvml_dl.h"
import "C"

import (
	"errors"
	"fmt"
)

const (
	szDriver   = C.NVML_SYSTEM_DRIVER_VERSION_BUFFER_SIZE
	szModel    = C.NVML_DEVICE_NAME_BUFFER_SIZE
	szUUID     = C.NVML_DEVICE_UUID_BUFFER_SIZE
	szProcs    = 32
	szProcName = 64
)

var (
	ErrCPUAffinity        = errors.New("failed to retrieve CPU affinity")
	ErrUnsupportedP2PLink = errors.New("unsupported P2P link type")
)

type P2PLinkType uint

const (
	P2PLinkUnknown P2PLinkType = iota
	P2PLinkCrossCPU
	P2PLinkSameCPU
	P2PLinkHostBridge
	P2PLinkMultiSwitch
	P2PLinkSingleSwitch
	P2PLinkSameBoard
)

type P2PLink struct {
	BusID string
	Link  P2PLinkType
}

func (t P2PLinkType) String() string {
	switch t {
	case P2PLinkCrossCPU:
		return "Cross CPU socket"
	case P2PLinkSameCPU:
		return "Same CPU socket"
	case P2PLinkHostBridge:
		return "Host PCI bridge"
	case P2PLinkMultiSwitch:
		return "Multiple PCI switches"
	case P2PLinkSingleSwitch:
		return "Single PCI switch"
	case P2PLinkSameBoard:
		return "Same board"
	case P2PLinkUnknown:
	}
	return "???"
}

type ClockInfo struct {
	Core   uint
	Memory uint
}

type PCIInfo struct {
	BusID     string
	BAR1      uint64
	Bandwidth uint
}

type Device struct {
	handle C.nvmlDevice_t

	Model       string
	UUID        string
	Path        string
	Power       uint
	CPUAffinity uint
	PCI         PCIInfo
	Clocks      ClockInfo
	Topology    []P2PLink
}

type UtilizationInfo struct {
	GPU     uint
	Encoder uint
	Decoder uint
}

type PCIThroughputInfo struct {
	RX uint
	TX uint
}

type PCIStatusInfo struct {
	BAR1Used   uint64
	Throughput PCIThroughputInfo
}

type ECCErrorsInfo struct {
	L1Cache uint64
	L2Cache uint64
	Global  uint64
}

type MemoryInfo struct {
	GlobalUsed uint64
	ECCErrors  ECCErrorsInfo
}

type ProcessInfo struct {
	PID  uint
	Name string
}

type DeviceStatus struct {
	Power       uint
	Temperature uint
	Utilization UtilizationInfo
	Memory      MemoryInfo
	Clocks      ClockInfo
	PCI         PCIStatusInfo
	Processes   []ProcessInfo
}

func nvmlErr(ret C.nvmlReturn_t) error {
	if ret == C.NVML_SUCCESS {
		return nil
	}
	err := C.GoString(C.nvmlErrorString(ret))
	return errors.New(err)
}

func assert(ret C.nvmlReturn_t) {
	if err := nvmlErr(ret); err != nil {
		panic(err)
	}
}

func Init() error {
	if err := C.nvmlInit_dl(); err != nil {
		return errors.New(C.GoString(err))
	}
	return nvmlErr(C.nvmlInit())
}

func Shutdown() error {
	C.nvmlShutdown_dl()
	return nvmlErr(C.nvmlShutdown())
}

func GetDeviceCount() (uint, error) {
	var n C.uint

	err := nvmlErr(C.nvmlDeviceGetCount(&n))
	return uint(n), err
}

func GetDriverVersion() (string, error) {
	var driver [szDriver]C.char

	err := nvmlErr(C.nvmlSystemGetDriverVersion(&driver[0], szDriver))
	return C.GoString(&driver[0]), err
}

var pcieGenToBandwidth = map[int]uint{
	1: 250, // MB/s
	2: 500,
	3: 985,
	4: 1969,
}

func NewDevice(idx uint) (device *Device, err error) {
	var (
		dev   C.nvmlDevice_t
		model [szModel]C.char
		uuid  [szUUID]C.char
		pci   C.nvmlPciInfo_t
		minor C.uint
		bar1  C.nvmlBAR1Memory_t
		power C.uint
		clock [2]C.uint
		pciel [2]C.uint
		cpus  cpuSet
	)

	defer func() {
		if r := recover(); r != nil {
			err = r.(error)
		}
	}()

	assert(C.nvmlDeviceGetHandleByIndex(C.uint(idx), &dev))
	assert(C.nvmlDeviceGetName(dev, &model[0], szModel))
	assert(C.nvmlDeviceGetUUID(dev, &uuid[0], szUUID))
	assert(C.nvmlDeviceGetPciInfo(dev, &pci))
	assert(C.nvmlDeviceGetMinorNumber(dev, &minor))
	assert(C.nvmlDeviceGetBAR1MemoryInfo(dev, &bar1))
	assert(C.nvmlDeviceGetPowerManagementLimit(dev, &power))
	assert(C.nvmlDeviceGetMaxClockInfo(dev, C.NVML_CLOCK_SM, &clock[0]))
	assert(C.nvmlDeviceGetMaxClockInfo(dev, C.NVML_CLOCK_MEM, &clock[1]))
	assert(C.nvmlDeviceGetMaxPcieLinkGeneration(dev, &pciel[0]))
	assert(C.nvmlDeviceGetMaxPcieLinkWidth(dev, &pciel[1]))
	assert(C.nvmlDeviceGetCpuAffinity(dev, C.uint(len(cpus)), (*C.ulong)(&cpus[0])))
	node, err := getCPUNode(cpus)
	if err != nil {
		return nil, err
	}

	device = &Device{
		handle:      dev,
		Model:       C.GoString(&model[0]),
		UUID:        C.GoString(&uuid[0]),
		Path:        fmt.Sprintf("/dev/nvidia%d", uint(minor)),
		Power:       uint(power / 1000),
		CPUAffinity: node,
		PCI: PCIInfo{
			BusID:     C.GoString(&pci.busId[0]),
			BAR1:      uint64(bar1.bar1Total / (1024 * 1024)),
			Bandwidth: pcieGenToBandwidth[int(pciel[0])] * uint(pciel[1]) / 1000,
		},
		Clocks: ClockInfo{
			Core:   uint(clock[0]),
			Memory: uint(clock[1]),
		},
	}
	return
}

func (d *Device) Status() (status *DeviceStatus, err error) {
	var (
		power      C.uint
		temp       C.uint
		usage      C.nvmlUtilization_t
		encoder    [2]C.uint
		decoder    [2]C.uint
		mem        C.nvmlMemory_t
		ecc        [3]C.ulonglong
		clock      [2]C.uint
		bar1       C.nvmlBAR1Memory_t
		throughput [2]C.uint
		procname   [szProcName]C.char
		procs      [szProcs]C.nvmlProcessInfo_t
		nprocs     = C.uint(szProcs)
	)

	defer func() {
		if r := recover(); r != nil {
			err = r.(error)
		}
	}()

	assert(C.nvmlDeviceGetPowerUsage(d.handle, &power))
	assert(C.nvmlDeviceGetTemperature(d.handle, C.NVML_TEMPERATURE_GPU, &temp))
	assert(C.nvmlDeviceGetUtilizationRates(d.handle, &usage))
	assert(C.nvmlDeviceGetEncoderUtilization(d.handle, &encoder[0], &encoder[1]))
	assert(C.nvmlDeviceGetDecoderUtilization(d.handle, &decoder[0], &decoder[1]))
	assert(C.nvmlDeviceGetMemoryInfo(d.handle, &mem))
	assert(C.nvmlDeviceGetClockInfo(d.handle, C.NVML_CLOCK_SM, &clock[0]))
	assert(C.nvmlDeviceGetClockInfo(d.handle, C.NVML_CLOCK_MEM, &clock[1]))
	assert(C.nvmlDeviceGetBAR1MemoryInfo(d.handle, &bar1))
	assert(C.nvmlDeviceGetComputeRunningProcesses(d.handle, &nprocs, &procs[0]))

	status = &DeviceStatus{
		Power:       uint(power / 1000),
		Temperature: uint(temp),
		Utilization: UtilizationInfo{
			GPU:     uint(usage.gpu),
			Encoder: uint(encoder[0]),
			Decoder: uint(decoder[0]),
		},
		Memory: MemoryInfo{
			GlobalUsed: uint64(mem.used / (1024 * 1024)),
		},
		Clocks: ClockInfo{
			Core:   uint(clock[0]),
			Memory: uint(clock[1]),
		},
		PCI: PCIStatusInfo{
			BAR1Used: uint64(bar1.bar1Used / (1024 * 1024)),
		},
	}

	r := C.nvmlDeviceGetMemoryErrorCounter(d.handle, C.NVML_MEMORY_ERROR_TYPE_UNCORRECTED, C.NVML_VOLATILE_ECC,
		C.NVML_MEMORY_LOCATION_L1_CACHE, &ecc[0])
	if r != C.NVML_ERROR_NOT_SUPPORTED { // only supported on Tesla cards
		assert(r)
		assert(C.nvmlDeviceGetMemoryErrorCounter(d.handle, C.NVML_MEMORY_ERROR_TYPE_UNCORRECTED, C.NVML_VOLATILE_ECC,
			C.NVML_MEMORY_LOCATION_L2_CACHE, &ecc[1]))
		assert(C.nvmlDeviceGetMemoryErrorCounter(d.handle, C.NVML_MEMORY_ERROR_TYPE_UNCORRECTED, C.NVML_VOLATILE_ECC,
			C.NVML_MEMORY_LOCATION_DEVICE_MEMORY, &ecc[2]))
		status.Memory.ECCErrors = ECCErrorsInfo{uint64(ecc[0]), uint64(ecc[1]), uint64(ecc[2])}
	}

	r = C.nvmlDeviceGetPcieThroughput(d.handle, C.NVML_PCIE_UTIL_RX_BYTES, &throughput[0])
	if r != C.NVML_ERROR_NOT_SUPPORTED { // only supported on Maxwell or newer
		assert(r)
		assert(C.nvmlDeviceGetPcieThroughput(d.handle, C.NVML_PCIE_UTIL_TX_BYTES, &throughput[1]))
		status.PCI.Throughput = PCIThroughputInfo{uint(throughput[0]), uint(throughput[1])}
	}

	status.Processes = make([]ProcessInfo, nprocs)
	for i := range status.Processes {
		status.Processes[i].PID = uint(procs[i].pid)
		assert(C.nvmlSystemGetProcessName(procs[i].pid, &procname[0], szProcName))
		status.Processes[i].Name = C.GoString(&procname[0])
	}
	return
}

func GetP2PLink(dev1, dev2 *Device) (link P2PLinkType, err error) {
	var level C.nvmlGpuTopologyLevel_t

	r := C.nvmlDeviceGetTopologyCommonAncestor_dl(dev1.handle, dev2.handle, &level)
	if r == C.NVML_ERROR_FUNCTION_NOT_FOUND {
		return P2PLinkUnknown, nil
	}
	if err = nvmlErr(r); err != nil {
		return
	}
	switch level {
	case C.NVML_TOPOLOGY_INTERNAL:
		link = P2PLinkSameBoard
	case C.NVML_TOPOLOGY_SINGLE:
		link = P2PLinkSingleSwitch
	case C.NVML_TOPOLOGY_MULTIPLE:
		link = P2PLinkMultiSwitch
	case C.NVML_TOPOLOGY_HOSTBRIDGE:
		link = P2PLinkHostBridge
	case C.NVML_TOPOLOGY_CPU:
		link = P2PLinkSameCPU
	case C.NVML_TOPOLOGY_SYSTEM:
		link = P2PLinkCrossCPU
	default:
		err = ErrUnsupportedP2PLink
	}
	return
}

func GetDevicePath(idx uint) (path string, err error) {
	var dev C.nvmlDevice_t
	var minor C.uint

	err = nvmlErr(C.nvmlDeviceGetHandleByIndex(C.uint(idx), &dev))
	if err != nil {
		return
	}
	err = nvmlErr(C.nvmlDeviceGetMinorNumber(dev, &minor))
	path = fmt.Sprintf("/dev/nvidia%d", uint(minor))
	return
}
