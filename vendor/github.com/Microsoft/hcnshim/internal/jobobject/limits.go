//go:build windows

package jobobject

import (
	"errors"
	"fmt"
	"unsafe"

	"github.com/Microsoft/hcnshim/internal/winapi"
	"golang.org/x/sys/windows"
)

const (
	memoryLimitMax uint64 = 0xffffffffffffffff
)

func isFlagSet(flag, controlFlags uint32) bool {
	return (flag & controlFlags) == flag
}

// SetResourceLimits sets resource limits on the job object (cpu, memory, storage).
func (job *JobObject) SetResourceLimits(limits *JobLimits) error {
	// Go through and check what limits were specified and apply them to the job.
	if limits.MemoryLimitInBytes != 0 {
		if err := job.SetMemoryLimit(limits.MemoryLimitInBytes); err != nil {
			return fmt.Errorf("failed to set job object memory limit: %w", err)
		}
	}

	if limits.CPULimit != 0 {
		if err := job.SetCPULimit(RateBased, limits.CPULimit); err != nil {
			return fmt.Errorf("failed to set job object cpu limit: %w", err)
		}
	} else if limits.CPUWeight != 0 {
		if err := job.SetCPULimit(WeightBased, limits.CPUWeight); err != nil {
			return fmt.Errorf("failed to set job object cpu limit: %w", err)
		}
	}

	if limits.MaxBandwidth != 0 || limits.MaxIOPS != 0 {
		if err := job.SetIOLimit(limits.MaxBandwidth, limits.MaxIOPS); err != nil {
			return fmt.Errorf("failed to set io limit on job object: %w", err)
		}
	}
	return nil
}

// SetTerminateOnLastHandleClose sets the job object flag that specifies that the job should terminate
// all processes in the job on the last open handle being closed.
func (job *JobObject) SetTerminateOnLastHandleClose() error {
	info, err := job.getExtendedInformation()
	if err != nil {
		return err
	}
	info.BasicLimitInformation.LimitFlags |= windows.JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
	return job.setExtendedInformation(info)
}

// SetMemoryLimit sets the memory limit of the job object based on the given `memoryLimitInBytes`.
func (job *JobObject) SetMemoryLimit(memoryLimitInBytes uint64) error {
	if memoryLimitInBytes >= memoryLimitMax {
		return errors.New("memory limit specified exceeds the max size")
	}

	info, err := job.getExtendedInformation()
	if err != nil {
		return err
	}

	info.JobMemoryLimit = uintptr(memoryLimitInBytes)
	info.BasicLimitInformation.LimitFlags |= windows.JOB_OBJECT_LIMIT_JOB_MEMORY
	return job.setExtendedInformation(info)
}

// GetMemoryLimit gets the memory limit in bytes of the job object.
func (job *JobObject) GetMemoryLimit() (uint64, error) {
	info, err := job.getExtendedInformation()
	if err != nil {
		return 0, err
	}
	return uint64(info.JobMemoryLimit), nil
}

// SetCPULimit sets the CPU limit depending on the specified `CPURateControlType` to
// `rateControlValue` for the job object.
func (job *JobObject) SetCPULimit(rateControlType CPURateControlType, rateControlValue uint32) error {
	cpuInfo, err := job.getCPURateControlInformation()
	if err != nil {
		return err
	}
	switch rateControlType {
	case WeightBased:
		if rateControlValue < cpuWeightMin || rateControlValue > cpuWeightMax {
			return fmt.Errorf("processor weight value of `%d` is invalid", rateControlValue)
		}
		cpuInfo.ControlFlags |= winapi.JOB_OBJECT_CPU_RATE_CONTROL_ENABLE | winapi.JOB_OBJECT_CPU_RATE_CONTROL_WEIGHT_BASED
		cpuInfo.Value = rateControlValue
	case RateBased:
		if rateControlValue < cpuLimitMin || rateControlValue > cpuLimitMax {
			return fmt.Errorf("processor rate of `%d` is invalid", rateControlValue)
		}
		cpuInfo.ControlFlags |= winapi.JOB_OBJECT_CPU_RATE_CONTROL_ENABLE | winapi.JOB_OBJECT_CPU_RATE_CONTROL_HARD_CAP
		cpuInfo.Value = rateControlValue
	default:
		return errors.New("invalid job object cpu rate control type")
	}
	return job.setCPURateControlInfo(cpuInfo)
}

// GetCPULimit gets the cpu limits for the job object.
// `rateControlType` is used to indicate what type of cpu limit to query for.
func (job *JobObject) GetCPULimit(rateControlType CPURateControlType) (uint32, error) {
	info, err := job.getCPURateControlInformation()
	if err != nil {
		return 0, err
	}

	if !isFlagSet(winapi.JOB_OBJECT_CPU_RATE_CONTROL_ENABLE, info.ControlFlags) {
		return 0, errors.New("the job does not have cpu rate control enabled")
	}

	switch rateControlType {
	case WeightBased:
		if !isFlagSet(winapi.JOB_OBJECT_CPU_RATE_CONTROL_WEIGHT_BASED, info.ControlFlags) {
			return 0, errors.New("cannot get cpu weight for job object without cpu weight option set")
		}
	case RateBased:
		if !isFlagSet(winapi.JOB_OBJECT_CPU_RATE_CONTROL_HARD_CAP, info.ControlFlags) {
			return 0, errors.New("cannot get cpu rate hard cap for job object without cpu rate hard cap option set")
		}
	default:
		return 0, errors.New("invalid job object cpu rate control type")
	}
	return info.Value, nil
}

// SetCPUAffinity sets the processor affinity for the job object.
// The affinity is passed in as a bitmask.
func (job *JobObject) SetCPUAffinity(affinityBitMask uint64) error {
	info, err := job.getExtendedInformation()
	if err != nil {
		return err
	}
	info.BasicLimitInformation.LimitFlags |= uint32(windows.JOB_OBJECT_LIMIT_AFFINITY)

	// We really, really shouldn't be running on 32 bit, but just in case (and to satisfy CodeQL) ...
	const maxUintptr = ^uintptr(0)
	if affinityBitMask > uint64(maxUintptr) {
		return fmt.Errorf("affinity bitmask (%d) exceeds max allowable value (%d)", affinityBitMask, maxUintptr)
	}

	info.BasicLimitInformation.Affinity = uintptr(affinityBitMask)
	return job.setExtendedInformation(info)
}

// GetCPUAffinity gets the processor affinity for the job object.
// The returned affinity is a bitmask.
func (job *JobObject) GetCPUAffinity() (uint64, error) {
	info, err := job.getExtendedInformation()
	if err != nil {
		return 0, err
	}
	return uint64(info.BasicLimitInformation.Affinity), nil
}

// SetIOLimit sets the IO limits specified on the job object.
func (job *JobObject) SetIOLimit(maxBandwidth, maxIOPS int64) error {
	ioInfo, err := job.getIOLimit()
	if err != nil {
		return err
	}
	ioInfo.ControlFlags |= winapi.JOB_OBJECT_IO_RATE_CONTROL_ENABLE
	if maxBandwidth != 0 {
		ioInfo.MaxBandwidth = maxBandwidth
	}
	if maxIOPS != 0 {
		ioInfo.MaxIops = maxIOPS
	}
	return job.setIORateControlInfo(ioInfo)
}

// GetIOMaxBandwidthLimit gets the max bandwidth for the job object.
func (job *JobObject) GetIOMaxBandwidthLimit() (int64, error) {
	info, err := job.getIOLimit()
	if err != nil {
		return 0, err
	}
	return info.MaxBandwidth, nil
}

// GetIOMaxIopsLimit gets the max iops for the job object.
func (job *JobObject) GetIOMaxIopsLimit() (int64, error) {
	info, err := job.getIOLimit()
	if err != nil {
		return 0, err
	}
	return info.MaxIops, nil
}

// Helper function for getting a job object's extended information.
func (job *JobObject) getExtendedInformation() (*windows.JOBOBJECT_EXTENDED_LIMIT_INFORMATION, error) {
	job.handleLock.RLock()
	defer job.handleLock.RUnlock()

	if job.handle == 0 {
		return nil, ErrAlreadyClosed
	}

	info := windows.JOBOBJECT_EXTENDED_LIMIT_INFORMATION{}
	if err := winapi.QueryInformationJobObject(
		job.handle,
		windows.JobObjectExtendedLimitInformation,
		unsafe.Pointer(&info),
		uint32(unsafe.Sizeof(info)),
		nil,
	); err != nil {
		return nil, fmt.Errorf("query %v returned error: %w", info, err)
	}
	return &info, nil
}

// Helper function for getting a job object's CPU rate control information.
func (job *JobObject) getCPURateControlInformation() (*winapi.JOBOBJECT_CPU_RATE_CONTROL_INFORMATION, error) {
	job.handleLock.RLock()
	defer job.handleLock.RUnlock()

	if job.handle == 0 {
		return nil, ErrAlreadyClosed
	}

	info := winapi.JOBOBJECT_CPU_RATE_CONTROL_INFORMATION{}
	if err := winapi.QueryInformationJobObject(
		job.handle,
		windows.JobObjectCpuRateControlInformation,
		unsafe.Pointer(&info),
		uint32(unsafe.Sizeof(info)),
		nil,
	); err != nil {
		return nil, fmt.Errorf("query %v returned error: %w", info, err)
	}
	return &info, nil
}

// Helper function for setting a job object's extended information.
func (job *JobObject) setExtendedInformation(info *windows.JOBOBJECT_EXTENDED_LIMIT_INFORMATION) error {
	job.handleLock.RLock()
	defer job.handleLock.RUnlock()

	if job.handle == 0 {
		return ErrAlreadyClosed
	}

	if _, err := windows.SetInformationJobObject(
		job.handle,
		windows.JobObjectExtendedLimitInformation,
		uintptr(unsafe.Pointer(info)),
		uint32(unsafe.Sizeof(*info)),
	); err != nil {
		return fmt.Errorf("failed to set Extended info %v on job object: %w", info, err)
	}
	return nil
}

// Helper function for querying job handle for IO limit information.
func (job *JobObject) getIOLimit() (*winapi.JOBOBJECT_IO_RATE_CONTROL_INFORMATION, error) {
	job.handleLock.RLock()
	defer job.handleLock.RUnlock()

	if job.handle == 0 {
		return nil, ErrAlreadyClosed
	}

	ioInfo := &winapi.JOBOBJECT_IO_RATE_CONTROL_INFORMATION{}
	var blockCount uint32 = 1

	if _, err := winapi.QueryIoRateControlInformationJobObject(
		job.handle,
		nil,
		&ioInfo,
		&blockCount,
	); err != nil {
		return nil, fmt.Errorf("query %v returned error: %w", ioInfo, err)
	}

	if !isFlagSet(winapi.JOB_OBJECT_IO_RATE_CONTROL_ENABLE, ioInfo.ControlFlags) {
		return nil, fmt.Errorf("query %v cannot get IO limits for job object without IO rate control option set", ioInfo)
	}
	return ioInfo, nil
}

// Helper function for setting a job object's IO rate control information.
func (job *JobObject) setIORateControlInfo(ioInfo *winapi.JOBOBJECT_IO_RATE_CONTROL_INFORMATION) error {
	job.handleLock.RLock()
	defer job.handleLock.RUnlock()

	if job.handle == 0 {
		return ErrAlreadyClosed
	}

	if _, err := winapi.SetIoRateControlInformationJobObject(job.handle, ioInfo); err != nil {
		return fmt.Errorf("failed to set IO limit info %v on job object: %w", ioInfo, err)
	}
	return nil
}

// Helper function for setting a job object's CPU rate control information.
func (job *JobObject) setCPURateControlInfo(cpuInfo *winapi.JOBOBJECT_CPU_RATE_CONTROL_INFORMATION) error {
	job.handleLock.RLock()
	defer job.handleLock.RUnlock()

	if job.handle == 0 {
		return ErrAlreadyClosed
	}
	if _, err := windows.SetInformationJobObject(
		job.handle,
		windows.JobObjectCpuRateControlInformation,
		uintptr(unsafe.Pointer(cpuInfo)),
		uint32(unsafe.Sizeof(cpuInfo)),
	); err != nil {
		return fmt.Errorf("failed to set cpu limit info %v on job object: %w", cpuInfo, err)
	}
	return nil
}
