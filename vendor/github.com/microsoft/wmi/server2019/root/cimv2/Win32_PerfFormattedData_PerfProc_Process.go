// Copyright 2019 (c) Microsoft Corporation.
// Licensed under the MIT license.

//
// Author:
//      Auto Generated on 9/18/2020 using wmigen
//      Source root.CIMV2
//////////////////////////////////////////////
package cimv2

import (
	"github.com/microsoft/wmi/pkg/base/query"
	"github.com/microsoft/wmi/pkg/errors"
	cim "github.com/microsoft/wmi/pkg/wmiinstance"
	"reflect"
)

// Win32_PerfFormattedData_PerfProc_Process struct
type Win32_PerfFormattedData_PerfProc_Process struct {
	*Win32_PerfFormattedData

	//
	CreatingProcessID uint32

	//
	ElapsedTime uint64

	//
	HandleCount uint32

	//
	IDProcess uint32

	//
	IODataBytesPersec uint64

	//
	IODataOperationsPersec uint64

	//
	IOOtherBytesPersec uint64

	//
	IOOtherOperationsPersec uint64

	//
	IOReadBytesPersec uint64

	//
	IOReadOperationsPersec uint64

	//
	IOWriteBytesPersec uint64

	//
	IOWriteOperationsPersec uint64

	//
	PageFaultsPersec uint32

	//
	PageFileBytes uint64

	//
	PageFileBytesPeak uint64

	//
	PercentPrivilegedTime uint64

	//
	PercentProcessorTime uint64

	//
	PercentUserTime uint64

	//
	PoolNonpagedBytes uint32

	//
	PoolPagedBytes uint32

	//
	PriorityBase uint32

	//
	PrivateBytes uint64

	//
	ThreadCount uint32

	//
	VirtualBytes uint64

	//
	VirtualBytesPeak uint64

	//
	WorkingSet uint64

	//
	WorkingSetPeak uint64

	//
	WorkingSetPrivate uint64
}

func NewWin32_PerfFormattedData_PerfProc_ProcessEx1(instance *cim.WmiInstance) (newInstance *Win32_PerfFormattedData_PerfProc_Process, err error) {
	tmp, err := NewWin32_PerfFormattedDataEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfFormattedData_PerfProc_Process{
		Win32_PerfFormattedData: tmp,
	}
	return
}

func NewWin32_PerfFormattedData_PerfProc_ProcessEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_PerfFormattedData_PerfProc_Process, err error) {
	tmp, err := NewWin32_PerfFormattedDataEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfFormattedData_PerfProc_Process{
		Win32_PerfFormattedData: tmp,
	}
	return
}

// SetCreatingProcessID sets the value of CreatingProcessID for the instance
func (instance *Win32_PerfFormattedData_PerfProc_Process) SetPropertyCreatingProcessID(value uint32) (err error) {
	return instance.SetProperty("CreatingProcessID", (value))
}

// GetCreatingProcessID gets the value of CreatingProcessID for the instance
func (instance *Win32_PerfFormattedData_PerfProc_Process) GetPropertyCreatingProcessID() (value uint32, err error) {
	retValue, err := instance.GetProperty("CreatingProcessID")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint32)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint32 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint32(valuetmp)

	return
}

// SetElapsedTime sets the value of ElapsedTime for the instance
func (instance *Win32_PerfFormattedData_PerfProc_Process) SetPropertyElapsedTime(value uint64) (err error) {
	return instance.SetProperty("ElapsedTime", (value))
}

// GetElapsedTime gets the value of ElapsedTime for the instance
func (instance *Win32_PerfFormattedData_PerfProc_Process) GetPropertyElapsedTime() (value uint64, err error) {
	retValue, err := instance.GetProperty("ElapsedTime")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint64)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint64 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint64(valuetmp)

	return
}

// SetHandleCount sets the value of HandleCount for the instance
func (instance *Win32_PerfFormattedData_PerfProc_Process) SetPropertyHandleCount(value uint32) (err error) {
	return instance.SetProperty("HandleCount", (value))
}

// GetHandleCount gets the value of HandleCount for the instance
func (instance *Win32_PerfFormattedData_PerfProc_Process) GetPropertyHandleCount() (value uint32, err error) {
	retValue, err := instance.GetProperty("HandleCount")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint32)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint32 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint32(valuetmp)

	return
}

// SetIDProcess sets the value of IDProcess for the instance
func (instance *Win32_PerfFormattedData_PerfProc_Process) SetPropertyIDProcess(value uint32) (err error) {
	return instance.SetProperty("IDProcess", (value))
}

// GetIDProcess gets the value of IDProcess for the instance
func (instance *Win32_PerfFormattedData_PerfProc_Process) GetPropertyIDProcess() (value uint32, err error) {
	retValue, err := instance.GetProperty("IDProcess")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint32)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint32 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint32(valuetmp)

	return
}

// SetIODataBytesPersec sets the value of IODataBytesPersec for the instance
func (instance *Win32_PerfFormattedData_PerfProc_Process) SetPropertyIODataBytesPersec(value uint64) (err error) {
	return instance.SetProperty("IODataBytesPersec", (value))
}

// GetIODataBytesPersec gets the value of IODataBytesPersec for the instance
func (instance *Win32_PerfFormattedData_PerfProc_Process) GetPropertyIODataBytesPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("IODataBytesPersec")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint64)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint64 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint64(valuetmp)

	return
}

// SetIODataOperationsPersec sets the value of IODataOperationsPersec for the instance
func (instance *Win32_PerfFormattedData_PerfProc_Process) SetPropertyIODataOperationsPersec(value uint64) (err error) {
	return instance.SetProperty("IODataOperationsPersec", (value))
}

// GetIODataOperationsPersec gets the value of IODataOperationsPersec for the instance
func (instance *Win32_PerfFormattedData_PerfProc_Process) GetPropertyIODataOperationsPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("IODataOperationsPersec")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint64)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint64 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint64(valuetmp)

	return
}

// SetIOOtherBytesPersec sets the value of IOOtherBytesPersec for the instance
func (instance *Win32_PerfFormattedData_PerfProc_Process) SetPropertyIOOtherBytesPersec(value uint64) (err error) {
	return instance.SetProperty("IOOtherBytesPersec", (value))
}

// GetIOOtherBytesPersec gets the value of IOOtherBytesPersec for the instance
func (instance *Win32_PerfFormattedData_PerfProc_Process) GetPropertyIOOtherBytesPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("IOOtherBytesPersec")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint64)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint64 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint64(valuetmp)

	return
}

// SetIOOtherOperationsPersec sets the value of IOOtherOperationsPersec for the instance
func (instance *Win32_PerfFormattedData_PerfProc_Process) SetPropertyIOOtherOperationsPersec(value uint64) (err error) {
	return instance.SetProperty("IOOtherOperationsPersec", (value))
}

// GetIOOtherOperationsPersec gets the value of IOOtherOperationsPersec for the instance
func (instance *Win32_PerfFormattedData_PerfProc_Process) GetPropertyIOOtherOperationsPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("IOOtherOperationsPersec")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint64)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint64 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint64(valuetmp)

	return
}

// SetIOReadBytesPersec sets the value of IOReadBytesPersec for the instance
func (instance *Win32_PerfFormattedData_PerfProc_Process) SetPropertyIOReadBytesPersec(value uint64) (err error) {
	return instance.SetProperty("IOReadBytesPersec", (value))
}

// GetIOReadBytesPersec gets the value of IOReadBytesPersec for the instance
func (instance *Win32_PerfFormattedData_PerfProc_Process) GetPropertyIOReadBytesPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("IOReadBytesPersec")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint64)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint64 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint64(valuetmp)

	return
}

// SetIOReadOperationsPersec sets the value of IOReadOperationsPersec for the instance
func (instance *Win32_PerfFormattedData_PerfProc_Process) SetPropertyIOReadOperationsPersec(value uint64) (err error) {
	return instance.SetProperty("IOReadOperationsPersec", (value))
}

// GetIOReadOperationsPersec gets the value of IOReadOperationsPersec for the instance
func (instance *Win32_PerfFormattedData_PerfProc_Process) GetPropertyIOReadOperationsPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("IOReadOperationsPersec")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint64)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint64 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint64(valuetmp)

	return
}

// SetIOWriteBytesPersec sets the value of IOWriteBytesPersec for the instance
func (instance *Win32_PerfFormattedData_PerfProc_Process) SetPropertyIOWriteBytesPersec(value uint64) (err error) {
	return instance.SetProperty("IOWriteBytesPersec", (value))
}

// GetIOWriteBytesPersec gets the value of IOWriteBytesPersec for the instance
func (instance *Win32_PerfFormattedData_PerfProc_Process) GetPropertyIOWriteBytesPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("IOWriteBytesPersec")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint64)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint64 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint64(valuetmp)

	return
}

// SetIOWriteOperationsPersec sets the value of IOWriteOperationsPersec for the instance
func (instance *Win32_PerfFormattedData_PerfProc_Process) SetPropertyIOWriteOperationsPersec(value uint64) (err error) {
	return instance.SetProperty("IOWriteOperationsPersec", (value))
}

// GetIOWriteOperationsPersec gets the value of IOWriteOperationsPersec for the instance
func (instance *Win32_PerfFormattedData_PerfProc_Process) GetPropertyIOWriteOperationsPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("IOWriteOperationsPersec")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint64)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint64 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint64(valuetmp)

	return
}

// SetPageFaultsPersec sets the value of PageFaultsPersec for the instance
func (instance *Win32_PerfFormattedData_PerfProc_Process) SetPropertyPageFaultsPersec(value uint32) (err error) {
	return instance.SetProperty("PageFaultsPersec", (value))
}

// GetPageFaultsPersec gets the value of PageFaultsPersec for the instance
func (instance *Win32_PerfFormattedData_PerfProc_Process) GetPropertyPageFaultsPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("PageFaultsPersec")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint32)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint32 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint32(valuetmp)

	return
}

// SetPageFileBytes sets the value of PageFileBytes for the instance
func (instance *Win32_PerfFormattedData_PerfProc_Process) SetPropertyPageFileBytes(value uint64) (err error) {
	return instance.SetProperty("PageFileBytes", (value))
}

// GetPageFileBytes gets the value of PageFileBytes for the instance
func (instance *Win32_PerfFormattedData_PerfProc_Process) GetPropertyPageFileBytes() (value uint64, err error) {
	retValue, err := instance.GetProperty("PageFileBytes")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint64)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint64 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint64(valuetmp)

	return
}

// SetPageFileBytesPeak sets the value of PageFileBytesPeak for the instance
func (instance *Win32_PerfFormattedData_PerfProc_Process) SetPropertyPageFileBytesPeak(value uint64) (err error) {
	return instance.SetProperty("PageFileBytesPeak", (value))
}

// GetPageFileBytesPeak gets the value of PageFileBytesPeak for the instance
func (instance *Win32_PerfFormattedData_PerfProc_Process) GetPropertyPageFileBytesPeak() (value uint64, err error) {
	retValue, err := instance.GetProperty("PageFileBytesPeak")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint64)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint64 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint64(valuetmp)

	return
}

// SetPercentPrivilegedTime sets the value of PercentPrivilegedTime for the instance
func (instance *Win32_PerfFormattedData_PerfProc_Process) SetPropertyPercentPrivilegedTime(value uint64) (err error) {
	return instance.SetProperty("PercentPrivilegedTime", (value))
}

// GetPercentPrivilegedTime gets the value of PercentPrivilegedTime for the instance
func (instance *Win32_PerfFormattedData_PerfProc_Process) GetPropertyPercentPrivilegedTime() (value uint64, err error) {
	retValue, err := instance.GetProperty("PercentPrivilegedTime")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint64)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint64 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint64(valuetmp)

	return
}

// SetPercentProcessorTime sets the value of PercentProcessorTime for the instance
func (instance *Win32_PerfFormattedData_PerfProc_Process) SetPropertyPercentProcessorTime(value uint64) (err error) {
	return instance.SetProperty("PercentProcessorTime", (value))
}

// GetPercentProcessorTime gets the value of PercentProcessorTime for the instance
func (instance *Win32_PerfFormattedData_PerfProc_Process) GetPropertyPercentProcessorTime() (value uint64, err error) {
	retValue, err := instance.GetProperty("PercentProcessorTime")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint64)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint64 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint64(valuetmp)

	return
}

// SetPercentUserTime sets the value of PercentUserTime for the instance
func (instance *Win32_PerfFormattedData_PerfProc_Process) SetPropertyPercentUserTime(value uint64) (err error) {
	return instance.SetProperty("PercentUserTime", (value))
}

// GetPercentUserTime gets the value of PercentUserTime for the instance
func (instance *Win32_PerfFormattedData_PerfProc_Process) GetPropertyPercentUserTime() (value uint64, err error) {
	retValue, err := instance.GetProperty("PercentUserTime")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint64)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint64 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint64(valuetmp)

	return
}

// SetPoolNonpagedBytes sets the value of PoolNonpagedBytes for the instance
func (instance *Win32_PerfFormattedData_PerfProc_Process) SetPropertyPoolNonpagedBytes(value uint32) (err error) {
	return instance.SetProperty("PoolNonpagedBytes", (value))
}

// GetPoolNonpagedBytes gets the value of PoolNonpagedBytes for the instance
func (instance *Win32_PerfFormattedData_PerfProc_Process) GetPropertyPoolNonpagedBytes() (value uint32, err error) {
	retValue, err := instance.GetProperty("PoolNonpagedBytes")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint32)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint32 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint32(valuetmp)

	return
}

// SetPoolPagedBytes sets the value of PoolPagedBytes for the instance
func (instance *Win32_PerfFormattedData_PerfProc_Process) SetPropertyPoolPagedBytes(value uint32) (err error) {
	return instance.SetProperty("PoolPagedBytes", (value))
}

// GetPoolPagedBytes gets the value of PoolPagedBytes for the instance
func (instance *Win32_PerfFormattedData_PerfProc_Process) GetPropertyPoolPagedBytes() (value uint32, err error) {
	retValue, err := instance.GetProperty("PoolPagedBytes")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint32)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint32 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint32(valuetmp)

	return
}

// SetPriorityBase sets the value of PriorityBase for the instance
func (instance *Win32_PerfFormattedData_PerfProc_Process) SetPropertyPriorityBase(value uint32) (err error) {
	return instance.SetProperty("PriorityBase", (value))
}

// GetPriorityBase gets the value of PriorityBase for the instance
func (instance *Win32_PerfFormattedData_PerfProc_Process) GetPropertyPriorityBase() (value uint32, err error) {
	retValue, err := instance.GetProperty("PriorityBase")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint32)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint32 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint32(valuetmp)

	return
}

// SetPrivateBytes sets the value of PrivateBytes for the instance
func (instance *Win32_PerfFormattedData_PerfProc_Process) SetPropertyPrivateBytes(value uint64) (err error) {
	return instance.SetProperty("PrivateBytes", (value))
}

// GetPrivateBytes gets the value of PrivateBytes for the instance
func (instance *Win32_PerfFormattedData_PerfProc_Process) GetPropertyPrivateBytes() (value uint64, err error) {
	retValue, err := instance.GetProperty("PrivateBytes")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint64)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint64 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint64(valuetmp)

	return
}

// SetThreadCount sets the value of ThreadCount for the instance
func (instance *Win32_PerfFormattedData_PerfProc_Process) SetPropertyThreadCount(value uint32) (err error) {
	return instance.SetProperty("ThreadCount", (value))
}

// GetThreadCount gets the value of ThreadCount for the instance
func (instance *Win32_PerfFormattedData_PerfProc_Process) GetPropertyThreadCount() (value uint32, err error) {
	retValue, err := instance.GetProperty("ThreadCount")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint32)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint32 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint32(valuetmp)

	return
}

// SetVirtualBytes sets the value of VirtualBytes for the instance
func (instance *Win32_PerfFormattedData_PerfProc_Process) SetPropertyVirtualBytes(value uint64) (err error) {
	return instance.SetProperty("VirtualBytes", (value))
}

// GetVirtualBytes gets the value of VirtualBytes for the instance
func (instance *Win32_PerfFormattedData_PerfProc_Process) GetPropertyVirtualBytes() (value uint64, err error) {
	retValue, err := instance.GetProperty("VirtualBytes")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint64)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint64 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint64(valuetmp)

	return
}

// SetVirtualBytesPeak sets the value of VirtualBytesPeak for the instance
func (instance *Win32_PerfFormattedData_PerfProc_Process) SetPropertyVirtualBytesPeak(value uint64) (err error) {
	return instance.SetProperty("VirtualBytesPeak", (value))
}

// GetVirtualBytesPeak gets the value of VirtualBytesPeak for the instance
func (instance *Win32_PerfFormattedData_PerfProc_Process) GetPropertyVirtualBytesPeak() (value uint64, err error) {
	retValue, err := instance.GetProperty("VirtualBytesPeak")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint64)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint64 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint64(valuetmp)

	return
}

// SetWorkingSet sets the value of WorkingSet for the instance
func (instance *Win32_PerfFormattedData_PerfProc_Process) SetPropertyWorkingSet(value uint64) (err error) {
	return instance.SetProperty("WorkingSet", (value))
}

// GetWorkingSet gets the value of WorkingSet for the instance
func (instance *Win32_PerfFormattedData_PerfProc_Process) GetPropertyWorkingSet() (value uint64, err error) {
	retValue, err := instance.GetProperty("WorkingSet")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint64)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint64 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint64(valuetmp)

	return
}

// SetWorkingSetPeak sets the value of WorkingSetPeak for the instance
func (instance *Win32_PerfFormattedData_PerfProc_Process) SetPropertyWorkingSetPeak(value uint64) (err error) {
	return instance.SetProperty("WorkingSetPeak", (value))
}

// GetWorkingSetPeak gets the value of WorkingSetPeak for the instance
func (instance *Win32_PerfFormattedData_PerfProc_Process) GetPropertyWorkingSetPeak() (value uint64, err error) {
	retValue, err := instance.GetProperty("WorkingSetPeak")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint64)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint64 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint64(valuetmp)

	return
}

// SetWorkingSetPrivate sets the value of WorkingSetPrivate for the instance
func (instance *Win32_PerfFormattedData_PerfProc_Process) SetPropertyWorkingSetPrivate(value uint64) (err error) {
	return instance.SetProperty("WorkingSetPrivate", (value))
}

// GetWorkingSetPrivate gets the value of WorkingSetPrivate for the instance
func (instance *Win32_PerfFormattedData_PerfProc_Process) GetPropertyWorkingSetPrivate() (value uint64, err error) {
	retValue, err := instance.GetProperty("WorkingSetPrivate")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint64)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint64 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint64(valuetmp)

	return
}
