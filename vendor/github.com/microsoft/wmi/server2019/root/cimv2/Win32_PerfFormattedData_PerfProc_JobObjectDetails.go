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

// Win32_PerfFormattedData_PerfProc_JobObjectDetails struct
type Win32_PerfFormattedData_PerfProc_JobObjectDetails struct {
	*Win32_PerfFormattedData

	//
	CreatingProcessID uint64

	//
	ElapsedTime uint64

	//
	HandleCount uint32

	//
	IDProcess uint64

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
}

func NewWin32_PerfFormattedData_PerfProc_JobObjectDetailsEx1(instance *cim.WmiInstance) (newInstance *Win32_PerfFormattedData_PerfProc_JobObjectDetails, err error) {
	tmp, err := NewWin32_PerfFormattedDataEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfFormattedData_PerfProc_JobObjectDetails{
		Win32_PerfFormattedData: tmp,
	}
	return
}

func NewWin32_PerfFormattedData_PerfProc_JobObjectDetailsEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_PerfFormattedData_PerfProc_JobObjectDetails, err error) {
	tmp, err := NewWin32_PerfFormattedDataEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfFormattedData_PerfProc_JobObjectDetails{
		Win32_PerfFormattedData: tmp,
	}
	return
}

// SetCreatingProcessID sets the value of CreatingProcessID for the instance
func (instance *Win32_PerfFormattedData_PerfProc_JobObjectDetails) SetPropertyCreatingProcessID(value uint64) (err error) {
	return instance.SetProperty("CreatingProcessID", (value))
}

// GetCreatingProcessID gets the value of CreatingProcessID for the instance
func (instance *Win32_PerfFormattedData_PerfProc_JobObjectDetails) GetPropertyCreatingProcessID() (value uint64, err error) {
	retValue, err := instance.GetProperty("CreatingProcessID")
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

// SetElapsedTime sets the value of ElapsedTime for the instance
func (instance *Win32_PerfFormattedData_PerfProc_JobObjectDetails) SetPropertyElapsedTime(value uint64) (err error) {
	return instance.SetProperty("ElapsedTime", (value))
}

// GetElapsedTime gets the value of ElapsedTime for the instance
func (instance *Win32_PerfFormattedData_PerfProc_JobObjectDetails) GetPropertyElapsedTime() (value uint64, err error) {
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
func (instance *Win32_PerfFormattedData_PerfProc_JobObjectDetails) SetPropertyHandleCount(value uint32) (err error) {
	return instance.SetProperty("HandleCount", (value))
}

// GetHandleCount gets the value of HandleCount for the instance
func (instance *Win32_PerfFormattedData_PerfProc_JobObjectDetails) GetPropertyHandleCount() (value uint32, err error) {
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
func (instance *Win32_PerfFormattedData_PerfProc_JobObjectDetails) SetPropertyIDProcess(value uint64) (err error) {
	return instance.SetProperty("IDProcess", (value))
}

// GetIDProcess gets the value of IDProcess for the instance
func (instance *Win32_PerfFormattedData_PerfProc_JobObjectDetails) GetPropertyIDProcess() (value uint64, err error) {
	retValue, err := instance.GetProperty("IDProcess")
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

// SetIODataBytesPersec sets the value of IODataBytesPersec for the instance
func (instance *Win32_PerfFormattedData_PerfProc_JobObjectDetails) SetPropertyIODataBytesPersec(value uint64) (err error) {
	return instance.SetProperty("IODataBytesPersec", (value))
}

// GetIODataBytesPersec gets the value of IODataBytesPersec for the instance
func (instance *Win32_PerfFormattedData_PerfProc_JobObjectDetails) GetPropertyIODataBytesPersec() (value uint64, err error) {
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
func (instance *Win32_PerfFormattedData_PerfProc_JobObjectDetails) SetPropertyIODataOperationsPersec(value uint64) (err error) {
	return instance.SetProperty("IODataOperationsPersec", (value))
}

// GetIODataOperationsPersec gets the value of IODataOperationsPersec for the instance
func (instance *Win32_PerfFormattedData_PerfProc_JobObjectDetails) GetPropertyIODataOperationsPersec() (value uint64, err error) {
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
func (instance *Win32_PerfFormattedData_PerfProc_JobObjectDetails) SetPropertyIOOtherBytesPersec(value uint64) (err error) {
	return instance.SetProperty("IOOtherBytesPersec", (value))
}

// GetIOOtherBytesPersec gets the value of IOOtherBytesPersec for the instance
func (instance *Win32_PerfFormattedData_PerfProc_JobObjectDetails) GetPropertyIOOtherBytesPersec() (value uint64, err error) {
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
func (instance *Win32_PerfFormattedData_PerfProc_JobObjectDetails) SetPropertyIOOtherOperationsPersec(value uint64) (err error) {
	return instance.SetProperty("IOOtherOperationsPersec", (value))
}

// GetIOOtherOperationsPersec gets the value of IOOtherOperationsPersec for the instance
func (instance *Win32_PerfFormattedData_PerfProc_JobObjectDetails) GetPropertyIOOtherOperationsPersec() (value uint64, err error) {
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
func (instance *Win32_PerfFormattedData_PerfProc_JobObjectDetails) SetPropertyIOReadBytesPersec(value uint64) (err error) {
	return instance.SetProperty("IOReadBytesPersec", (value))
}

// GetIOReadBytesPersec gets the value of IOReadBytesPersec for the instance
func (instance *Win32_PerfFormattedData_PerfProc_JobObjectDetails) GetPropertyIOReadBytesPersec() (value uint64, err error) {
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
func (instance *Win32_PerfFormattedData_PerfProc_JobObjectDetails) SetPropertyIOReadOperationsPersec(value uint64) (err error) {
	return instance.SetProperty("IOReadOperationsPersec", (value))
}

// GetIOReadOperationsPersec gets the value of IOReadOperationsPersec for the instance
func (instance *Win32_PerfFormattedData_PerfProc_JobObjectDetails) GetPropertyIOReadOperationsPersec() (value uint64, err error) {
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
func (instance *Win32_PerfFormattedData_PerfProc_JobObjectDetails) SetPropertyIOWriteBytesPersec(value uint64) (err error) {
	return instance.SetProperty("IOWriteBytesPersec", (value))
}

// GetIOWriteBytesPersec gets the value of IOWriteBytesPersec for the instance
func (instance *Win32_PerfFormattedData_PerfProc_JobObjectDetails) GetPropertyIOWriteBytesPersec() (value uint64, err error) {
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
func (instance *Win32_PerfFormattedData_PerfProc_JobObjectDetails) SetPropertyIOWriteOperationsPersec(value uint64) (err error) {
	return instance.SetProperty("IOWriteOperationsPersec", (value))
}

// GetIOWriteOperationsPersec gets the value of IOWriteOperationsPersec for the instance
func (instance *Win32_PerfFormattedData_PerfProc_JobObjectDetails) GetPropertyIOWriteOperationsPersec() (value uint64, err error) {
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
func (instance *Win32_PerfFormattedData_PerfProc_JobObjectDetails) SetPropertyPageFaultsPersec(value uint32) (err error) {
	return instance.SetProperty("PageFaultsPersec", (value))
}

// GetPageFaultsPersec gets the value of PageFaultsPersec for the instance
func (instance *Win32_PerfFormattedData_PerfProc_JobObjectDetails) GetPropertyPageFaultsPersec() (value uint32, err error) {
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
func (instance *Win32_PerfFormattedData_PerfProc_JobObjectDetails) SetPropertyPageFileBytes(value uint64) (err error) {
	return instance.SetProperty("PageFileBytes", (value))
}

// GetPageFileBytes gets the value of PageFileBytes for the instance
func (instance *Win32_PerfFormattedData_PerfProc_JobObjectDetails) GetPropertyPageFileBytes() (value uint64, err error) {
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
func (instance *Win32_PerfFormattedData_PerfProc_JobObjectDetails) SetPropertyPageFileBytesPeak(value uint64) (err error) {
	return instance.SetProperty("PageFileBytesPeak", (value))
}

// GetPageFileBytesPeak gets the value of PageFileBytesPeak for the instance
func (instance *Win32_PerfFormattedData_PerfProc_JobObjectDetails) GetPropertyPageFileBytesPeak() (value uint64, err error) {
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
func (instance *Win32_PerfFormattedData_PerfProc_JobObjectDetails) SetPropertyPercentPrivilegedTime(value uint64) (err error) {
	return instance.SetProperty("PercentPrivilegedTime", (value))
}

// GetPercentPrivilegedTime gets the value of PercentPrivilegedTime for the instance
func (instance *Win32_PerfFormattedData_PerfProc_JobObjectDetails) GetPropertyPercentPrivilegedTime() (value uint64, err error) {
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
func (instance *Win32_PerfFormattedData_PerfProc_JobObjectDetails) SetPropertyPercentProcessorTime(value uint64) (err error) {
	return instance.SetProperty("PercentProcessorTime", (value))
}

// GetPercentProcessorTime gets the value of PercentProcessorTime for the instance
func (instance *Win32_PerfFormattedData_PerfProc_JobObjectDetails) GetPropertyPercentProcessorTime() (value uint64, err error) {
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
func (instance *Win32_PerfFormattedData_PerfProc_JobObjectDetails) SetPropertyPercentUserTime(value uint64) (err error) {
	return instance.SetProperty("PercentUserTime", (value))
}

// GetPercentUserTime gets the value of PercentUserTime for the instance
func (instance *Win32_PerfFormattedData_PerfProc_JobObjectDetails) GetPropertyPercentUserTime() (value uint64, err error) {
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
func (instance *Win32_PerfFormattedData_PerfProc_JobObjectDetails) SetPropertyPoolNonpagedBytes(value uint32) (err error) {
	return instance.SetProperty("PoolNonpagedBytes", (value))
}

// GetPoolNonpagedBytes gets the value of PoolNonpagedBytes for the instance
func (instance *Win32_PerfFormattedData_PerfProc_JobObjectDetails) GetPropertyPoolNonpagedBytes() (value uint32, err error) {
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
func (instance *Win32_PerfFormattedData_PerfProc_JobObjectDetails) SetPropertyPoolPagedBytes(value uint32) (err error) {
	return instance.SetProperty("PoolPagedBytes", (value))
}

// GetPoolPagedBytes gets the value of PoolPagedBytes for the instance
func (instance *Win32_PerfFormattedData_PerfProc_JobObjectDetails) GetPropertyPoolPagedBytes() (value uint32, err error) {
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
func (instance *Win32_PerfFormattedData_PerfProc_JobObjectDetails) SetPropertyPriorityBase(value uint32) (err error) {
	return instance.SetProperty("PriorityBase", (value))
}

// GetPriorityBase gets the value of PriorityBase for the instance
func (instance *Win32_PerfFormattedData_PerfProc_JobObjectDetails) GetPropertyPriorityBase() (value uint32, err error) {
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
func (instance *Win32_PerfFormattedData_PerfProc_JobObjectDetails) SetPropertyPrivateBytes(value uint64) (err error) {
	return instance.SetProperty("PrivateBytes", (value))
}

// GetPrivateBytes gets the value of PrivateBytes for the instance
func (instance *Win32_PerfFormattedData_PerfProc_JobObjectDetails) GetPropertyPrivateBytes() (value uint64, err error) {
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
func (instance *Win32_PerfFormattedData_PerfProc_JobObjectDetails) SetPropertyThreadCount(value uint32) (err error) {
	return instance.SetProperty("ThreadCount", (value))
}

// GetThreadCount gets the value of ThreadCount for the instance
func (instance *Win32_PerfFormattedData_PerfProc_JobObjectDetails) GetPropertyThreadCount() (value uint32, err error) {
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
func (instance *Win32_PerfFormattedData_PerfProc_JobObjectDetails) SetPropertyVirtualBytes(value uint64) (err error) {
	return instance.SetProperty("VirtualBytes", (value))
}

// GetVirtualBytes gets the value of VirtualBytes for the instance
func (instance *Win32_PerfFormattedData_PerfProc_JobObjectDetails) GetPropertyVirtualBytes() (value uint64, err error) {
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
func (instance *Win32_PerfFormattedData_PerfProc_JobObjectDetails) SetPropertyVirtualBytesPeak(value uint64) (err error) {
	return instance.SetProperty("VirtualBytesPeak", (value))
}

// GetVirtualBytesPeak gets the value of VirtualBytesPeak for the instance
func (instance *Win32_PerfFormattedData_PerfProc_JobObjectDetails) GetPropertyVirtualBytesPeak() (value uint64, err error) {
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
func (instance *Win32_PerfFormattedData_PerfProc_JobObjectDetails) SetPropertyWorkingSet(value uint64) (err error) {
	return instance.SetProperty("WorkingSet", (value))
}

// GetWorkingSet gets the value of WorkingSet for the instance
func (instance *Win32_PerfFormattedData_PerfProc_JobObjectDetails) GetPropertyWorkingSet() (value uint64, err error) {
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
func (instance *Win32_PerfFormattedData_PerfProc_JobObjectDetails) SetPropertyWorkingSetPeak(value uint64) (err error) {
	return instance.SetProperty("WorkingSetPeak", (value))
}

// GetWorkingSetPeak gets the value of WorkingSetPeak for the instance
func (instance *Win32_PerfFormattedData_PerfProc_JobObjectDetails) GetPropertyWorkingSetPeak() (value uint64, err error) {
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
