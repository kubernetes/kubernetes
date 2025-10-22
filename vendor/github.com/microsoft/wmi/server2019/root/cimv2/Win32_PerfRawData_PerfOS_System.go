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

// Win32_PerfRawData_PerfOS_System struct
type Win32_PerfRawData_PerfOS_System struct {
	*Win32_PerfRawData

	//
	AlignmentFixupsPersec uint32

	//
	ContextSwitchesPersec uint32

	//
	ExceptionDispatchesPersec uint32

	//
	FileControlBytesPersec uint64

	//
	FileControlOperationsPersec uint32

	//
	FileDataOperationsPersec uint32

	//
	FileReadBytesPersec uint64

	//
	FileReadOperationsPersec uint32

	//
	FileWriteBytesPersec uint64

	//
	FileWriteOperationsPersec uint32

	//
	FloatingEmulationsPersec uint32

	//
	PercentRegistryQuotaInUse uint32

	//
	PercentRegistryQuotaInUse_Base uint32

	//
	Processes uint32

	//
	ProcessorQueueLength uint32

	//
	SystemCallsPersec uint32

	//
	SystemUpTime uint64

	//
	Threads uint32
}

func NewWin32_PerfRawData_PerfOS_SystemEx1(instance *cim.WmiInstance) (newInstance *Win32_PerfRawData_PerfOS_System, err error) {
	tmp, err := NewWin32_PerfRawDataEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfRawData_PerfOS_System{
		Win32_PerfRawData: tmp,
	}
	return
}

func NewWin32_PerfRawData_PerfOS_SystemEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_PerfRawData_PerfOS_System, err error) {
	tmp, err := NewWin32_PerfRawDataEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfRawData_PerfOS_System{
		Win32_PerfRawData: tmp,
	}
	return
}

// SetAlignmentFixupsPersec sets the value of AlignmentFixupsPersec for the instance
func (instance *Win32_PerfRawData_PerfOS_System) SetPropertyAlignmentFixupsPersec(value uint32) (err error) {
	return instance.SetProperty("AlignmentFixupsPersec", (value))
}

// GetAlignmentFixupsPersec gets the value of AlignmentFixupsPersec for the instance
func (instance *Win32_PerfRawData_PerfOS_System) GetPropertyAlignmentFixupsPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("AlignmentFixupsPersec")
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

// SetContextSwitchesPersec sets the value of ContextSwitchesPersec for the instance
func (instance *Win32_PerfRawData_PerfOS_System) SetPropertyContextSwitchesPersec(value uint32) (err error) {
	return instance.SetProperty("ContextSwitchesPersec", (value))
}

// GetContextSwitchesPersec gets the value of ContextSwitchesPersec for the instance
func (instance *Win32_PerfRawData_PerfOS_System) GetPropertyContextSwitchesPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("ContextSwitchesPersec")
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

// SetExceptionDispatchesPersec sets the value of ExceptionDispatchesPersec for the instance
func (instance *Win32_PerfRawData_PerfOS_System) SetPropertyExceptionDispatchesPersec(value uint32) (err error) {
	return instance.SetProperty("ExceptionDispatchesPersec", (value))
}

// GetExceptionDispatchesPersec gets the value of ExceptionDispatchesPersec for the instance
func (instance *Win32_PerfRawData_PerfOS_System) GetPropertyExceptionDispatchesPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("ExceptionDispatchesPersec")
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

// SetFileControlBytesPersec sets the value of FileControlBytesPersec for the instance
func (instance *Win32_PerfRawData_PerfOS_System) SetPropertyFileControlBytesPersec(value uint64) (err error) {
	return instance.SetProperty("FileControlBytesPersec", (value))
}

// GetFileControlBytesPersec gets the value of FileControlBytesPersec for the instance
func (instance *Win32_PerfRawData_PerfOS_System) GetPropertyFileControlBytesPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("FileControlBytesPersec")
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

// SetFileControlOperationsPersec sets the value of FileControlOperationsPersec for the instance
func (instance *Win32_PerfRawData_PerfOS_System) SetPropertyFileControlOperationsPersec(value uint32) (err error) {
	return instance.SetProperty("FileControlOperationsPersec", (value))
}

// GetFileControlOperationsPersec gets the value of FileControlOperationsPersec for the instance
func (instance *Win32_PerfRawData_PerfOS_System) GetPropertyFileControlOperationsPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("FileControlOperationsPersec")
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

// SetFileDataOperationsPersec sets the value of FileDataOperationsPersec for the instance
func (instance *Win32_PerfRawData_PerfOS_System) SetPropertyFileDataOperationsPersec(value uint32) (err error) {
	return instance.SetProperty("FileDataOperationsPersec", (value))
}

// GetFileDataOperationsPersec gets the value of FileDataOperationsPersec for the instance
func (instance *Win32_PerfRawData_PerfOS_System) GetPropertyFileDataOperationsPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("FileDataOperationsPersec")
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

// SetFileReadBytesPersec sets the value of FileReadBytesPersec for the instance
func (instance *Win32_PerfRawData_PerfOS_System) SetPropertyFileReadBytesPersec(value uint64) (err error) {
	return instance.SetProperty("FileReadBytesPersec", (value))
}

// GetFileReadBytesPersec gets the value of FileReadBytesPersec for the instance
func (instance *Win32_PerfRawData_PerfOS_System) GetPropertyFileReadBytesPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("FileReadBytesPersec")
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

// SetFileReadOperationsPersec sets the value of FileReadOperationsPersec for the instance
func (instance *Win32_PerfRawData_PerfOS_System) SetPropertyFileReadOperationsPersec(value uint32) (err error) {
	return instance.SetProperty("FileReadOperationsPersec", (value))
}

// GetFileReadOperationsPersec gets the value of FileReadOperationsPersec for the instance
func (instance *Win32_PerfRawData_PerfOS_System) GetPropertyFileReadOperationsPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("FileReadOperationsPersec")
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

// SetFileWriteBytesPersec sets the value of FileWriteBytesPersec for the instance
func (instance *Win32_PerfRawData_PerfOS_System) SetPropertyFileWriteBytesPersec(value uint64) (err error) {
	return instance.SetProperty("FileWriteBytesPersec", (value))
}

// GetFileWriteBytesPersec gets the value of FileWriteBytesPersec for the instance
func (instance *Win32_PerfRawData_PerfOS_System) GetPropertyFileWriteBytesPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("FileWriteBytesPersec")
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

// SetFileWriteOperationsPersec sets the value of FileWriteOperationsPersec for the instance
func (instance *Win32_PerfRawData_PerfOS_System) SetPropertyFileWriteOperationsPersec(value uint32) (err error) {
	return instance.SetProperty("FileWriteOperationsPersec", (value))
}

// GetFileWriteOperationsPersec gets the value of FileWriteOperationsPersec for the instance
func (instance *Win32_PerfRawData_PerfOS_System) GetPropertyFileWriteOperationsPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("FileWriteOperationsPersec")
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

// SetFloatingEmulationsPersec sets the value of FloatingEmulationsPersec for the instance
func (instance *Win32_PerfRawData_PerfOS_System) SetPropertyFloatingEmulationsPersec(value uint32) (err error) {
	return instance.SetProperty("FloatingEmulationsPersec", (value))
}

// GetFloatingEmulationsPersec gets the value of FloatingEmulationsPersec for the instance
func (instance *Win32_PerfRawData_PerfOS_System) GetPropertyFloatingEmulationsPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("FloatingEmulationsPersec")
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

// SetPercentRegistryQuotaInUse sets the value of PercentRegistryQuotaInUse for the instance
func (instance *Win32_PerfRawData_PerfOS_System) SetPropertyPercentRegistryQuotaInUse(value uint32) (err error) {
	return instance.SetProperty("PercentRegistryQuotaInUse", (value))
}

// GetPercentRegistryQuotaInUse gets the value of PercentRegistryQuotaInUse for the instance
func (instance *Win32_PerfRawData_PerfOS_System) GetPropertyPercentRegistryQuotaInUse() (value uint32, err error) {
	retValue, err := instance.GetProperty("PercentRegistryQuotaInUse")
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

// SetPercentRegistryQuotaInUse_Base sets the value of PercentRegistryQuotaInUse_Base for the instance
func (instance *Win32_PerfRawData_PerfOS_System) SetPropertyPercentRegistryQuotaInUse_Base(value uint32) (err error) {
	return instance.SetProperty("PercentRegistryQuotaInUse_Base", (value))
}

// GetPercentRegistryQuotaInUse_Base gets the value of PercentRegistryQuotaInUse_Base for the instance
func (instance *Win32_PerfRawData_PerfOS_System) GetPropertyPercentRegistryQuotaInUse_Base() (value uint32, err error) {
	retValue, err := instance.GetProperty("PercentRegistryQuotaInUse_Base")
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

// SetProcesses sets the value of Processes for the instance
func (instance *Win32_PerfRawData_PerfOS_System) SetPropertyProcesses(value uint32) (err error) {
	return instance.SetProperty("Processes", (value))
}

// GetProcesses gets the value of Processes for the instance
func (instance *Win32_PerfRawData_PerfOS_System) GetPropertyProcesses() (value uint32, err error) {
	retValue, err := instance.GetProperty("Processes")
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

// SetProcessorQueueLength sets the value of ProcessorQueueLength for the instance
func (instance *Win32_PerfRawData_PerfOS_System) SetPropertyProcessorQueueLength(value uint32) (err error) {
	return instance.SetProperty("ProcessorQueueLength", (value))
}

// GetProcessorQueueLength gets the value of ProcessorQueueLength for the instance
func (instance *Win32_PerfRawData_PerfOS_System) GetPropertyProcessorQueueLength() (value uint32, err error) {
	retValue, err := instance.GetProperty("ProcessorQueueLength")
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

// SetSystemCallsPersec sets the value of SystemCallsPersec for the instance
func (instance *Win32_PerfRawData_PerfOS_System) SetPropertySystemCallsPersec(value uint32) (err error) {
	return instance.SetProperty("SystemCallsPersec", (value))
}

// GetSystemCallsPersec gets the value of SystemCallsPersec for the instance
func (instance *Win32_PerfRawData_PerfOS_System) GetPropertySystemCallsPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("SystemCallsPersec")
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

// SetSystemUpTime sets the value of SystemUpTime for the instance
func (instance *Win32_PerfRawData_PerfOS_System) SetPropertySystemUpTime(value uint64) (err error) {
	return instance.SetProperty("SystemUpTime", (value))
}

// GetSystemUpTime gets the value of SystemUpTime for the instance
func (instance *Win32_PerfRawData_PerfOS_System) GetPropertySystemUpTime() (value uint64, err error) {
	retValue, err := instance.GetProperty("SystemUpTime")
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

// SetThreads sets the value of Threads for the instance
func (instance *Win32_PerfRawData_PerfOS_System) SetPropertyThreads(value uint32) (err error) {
	return instance.SetProperty("Threads", (value))
}

// GetThreads gets the value of Threads for the instance
func (instance *Win32_PerfRawData_PerfOS_System) GetPropertyThreads() (value uint32, err error) {
	retValue, err := instance.GetProperty("Threads")
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
