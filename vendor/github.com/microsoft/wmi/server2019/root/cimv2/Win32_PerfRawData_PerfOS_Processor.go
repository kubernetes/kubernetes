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

// Win32_PerfRawData_PerfOS_Processor struct
type Win32_PerfRawData_PerfOS_Processor struct {
	*Win32_PerfRawData

	//
	C1TransitionsPersec uint64

	//
	C2TransitionsPersec uint64

	//
	C3TransitionsPersec uint64

	//
	DPCRate uint32

	//
	DPCsQueuedPersec uint32

	//
	InterruptsPersec uint32

	//
	PercentC1Time uint64

	//
	PercentC2Time uint64

	//
	PercentC3Time uint64

	//
	PercentDPCTime uint64

	//
	PercentIdleTime uint64

	//
	PercentInterruptTime uint64

	//
	PercentPrivilegedTime uint64

	//
	PercentProcessorTime uint64

	//
	PercentUserTime uint64
}

func NewWin32_PerfRawData_PerfOS_ProcessorEx1(instance *cim.WmiInstance) (newInstance *Win32_PerfRawData_PerfOS_Processor, err error) {
	tmp, err := NewWin32_PerfRawDataEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfRawData_PerfOS_Processor{
		Win32_PerfRawData: tmp,
	}
	return
}

func NewWin32_PerfRawData_PerfOS_ProcessorEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_PerfRawData_PerfOS_Processor, err error) {
	tmp, err := NewWin32_PerfRawDataEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfRawData_PerfOS_Processor{
		Win32_PerfRawData: tmp,
	}
	return
}

// SetC1TransitionsPersec sets the value of C1TransitionsPersec for the instance
func (instance *Win32_PerfRawData_PerfOS_Processor) SetPropertyC1TransitionsPersec(value uint64) (err error) {
	return instance.SetProperty("C1TransitionsPersec", (value))
}

// GetC1TransitionsPersec gets the value of C1TransitionsPersec for the instance
func (instance *Win32_PerfRawData_PerfOS_Processor) GetPropertyC1TransitionsPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("C1TransitionsPersec")
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

// SetC2TransitionsPersec sets the value of C2TransitionsPersec for the instance
func (instance *Win32_PerfRawData_PerfOS_Processor) SetPropertyC2TransitionsPersec(value uint64) (err error) {
	return instance.SetProperty("C2TransitionsPersec", (value))
}

// GetC2TransitionsPersec gets the value of C2TransitionsPersec for the instance
func (instance *Win32_PerfRawData_PerfOS_Processor) GetPropertyC2TransitionsPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("C2TransitionsPersec")
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

// SetC3TransitionsPersec sets the value of C3TransitionsPersec for the instance
func (instance *Win32_PerfRawData_PerfOS_Processor) SetPropertyC3TransitionsPersec(value uint64) (err error) {
	return instance.SetProperty("C3TransitionsPersec", (value))
}

// GetC3TransitionsPersec gets the value of C3TransitionsPersec for the instance
func (instance *Win32_PerfRawData_PerfOS_Processor) GetPropertyC3TransitionsPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("C3TransitionsPersec")
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

// SetDPCRate sets the value of DPCRate for the instance
func (instance *Win32_PerfRawData_PerfOS_Processor) SetPropertyDPCRate(value uint32) (err error) {
	return instance.SetProperty("DPCRate", (value))
}

// GetDPCRate gets the value of DPCRate for the instance
func (instance *Win32_PerfRawData_PerfOS_Processor) GetPropertyDPCRate() (value uint32, err error) {
	retValue, err := instance.GetProperty("DPCRate")
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

// SetDPCsQueuedPersec sets the value of DPCsQueuedPersec for the instance
func (instance *Win32_PerfRawData_PerfOS_Processor) SetPropertyDPCsQueuedPersec(value uint32) (err error) {
	return instance.SetProperty("DPCsQueuedPersec", (value))
}

// GetDPCsQueuedPersec gets the value of DPCsQueuedPersec for the instance
func (instance *Win32_PerfRawData_PerfOS_Processor) GetPropertyDPCsQueuedPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("DPCsQueuedPersec")
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

// SetInterruptsPersec sets the value of InterruptsPersec for the instance
func (instance *Win32_PerfRawData_PerfOS_Processor) SetPropertyInterruptsPersec(value uint32) (err error) {
	return instance.SetProperty("InterruptsPersec", (value))
}

// GetInterruptsPersec gets the value of InterruptsPersec for the instance
func (instance *Win32_PerfRawData_PerfOS_Processor) GetPropertyInterruptsPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("InterruptsPersec")
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

// SetPercentC1Time sets the value of PercentC1Time for the instance
func (instance *Win32_PerfRawData_PerfOS_Processor) SetPropertyPercentC1Time(value uint64) (err error) {
	return instance.SetProperty("PercentC1Time", (value))
}

// GetPercentC1Time gets the value of PercentC1Time for the instance
func (instance *Win32_PerfRawData_PerfOS_Processor) GetPropertyPercentC1Time() (value uint64, err error) {
	retValue, err := instance.GetProperty("PercentC1Time")
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

// SetPercentC2Time sets the value of PercentC2Time for the instance
func (instance *Win32_PerfRawData_PerfOS_Processor) SetPropertyPercentC2Time(value uint64) (err error) {
	return instance.SetProperty("PercentC2Time", (value))
}

// GetPercentC2Time gets the value of PercentC2Time for the instance
func (instance *Win32_PerfRawData_PerfOS_Processor) GetPropertyPercentC2Time() (value uint64, err error) {
	retValue, err := instance.GetProperty("PercentC2Time")
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

// SetPercentC3Time sets the value of PercentC3Time for the instance
func (instance *Win32_PerfRawData_PerfOS_Processor) SetPropertyPercentC3Time(value uint64) (err error) {
	return instance.SetProperty("PercentC3Time", (value))
}

// GetPercentC3Time gets the value of PercentC3Time for the instance
func (instance *Win32_PerfRawData_PerfOS_Processor) GetPropertyPercentC3Time() (value uint64, err error) {
	retValue, err := instance.GetProperty("PercentC3Time")
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

// SetPercentDPCTime sets the value of PercentDPCTime for the instance
func (instance *Win32_PerfRawData_PerfOS_Processor) SetPropertyPercentDPCTime(value uint64) (err error) {
	return instance.SetProperty("PercentDPCTime", (value))
}

// GetPercentDPCTime gets the value of PercentDPCTime for the instance
func (instance *Win32_PerfRawData_PerfOS_Processor) GetPropertyPercentDPCTime() (value uint64, err error) {
	retValue, err := instance.GetProperty("PercentDPCTime")
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

// SetPercentIdleTime sets the value of PercentIdleTime for the instance
func (instance *Win32_PerfRawData_PerfOS_Processor) SetPropertyPercentIdleTime(value uint64) (err error) {
	return instance.SetProperty("PercentIdleTime", (value))
}

// GetPercentIdleTime gets the value of PercentIdleTime for the instance
func (instance *Win32_PerfRawData_PerfOS_Processor) GetPropertyPercentIdleTime() (value uint64, err error) {
	retValue, err := instance.GetProperty("PercentIdleTime")
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

// SetPercentInterruptTime sets the value of PercentInterruptTime for the instance
func (instance *Win32_PerfRawData_PerfOS_Processor) SetPropertyPercentInterruptTime(value uint64) (err error) {
	return instance.SetProperty("PercentInterruptTime", (value))
}

// GetPercentInterruptTime gets the value of PercentInterruptTime for the instance
func (instance *Win32_PerfRawData_PerfOS_Processor) GetPropertyPercentInterruptTime() (value uint64, err error) {
	retValue, err := instance.GetProperty("PercentInterruptTime")
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
func (instance *Win32_PerfRawData_PerfOS_Processor) SetPropertyPercentPrivilegedTime(value uint64) (err error) {
	return instance.SetProperty("PercentPrivilegedTime", (value))
}

// GetPercentPrivilegedTime gets the value of PercentPrivilegedTime for the instance
func (instance *Win32_PerfRawData_PerfOS_Processor) GetPropertyPercentPrivilegedTime() (value uint64, err error) {
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
func (instance *Win32_PerfRawData_PerfOS_Processor) SetPropertyPercentProcessorTime(value uint64) (err error) {
	return instance.SetProperty("PercentProcessorTime", (value))
}

// GetPercentProcessorTime gets the value of PercentProcessorTime for the instance
func (instance *Win32_PerfRawData_PerfOS_Processor) GetPropertyPercentProcessorTime() (value uint64, err error) {
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
func (instance *Win32_PerfRawData_PerfOS_Processor) SetPropertyPercentUserTime(value uint64) (err error) {
	return instance.SetProperty("PercentUserTime", (value))
}

// GetPercentUserTime gets the value of PercentUserTime for the instance
func (instance *Win32_PerfRawData_PerfOS_Processor) GetPropertyPercentUserTime() (value uint64, err error) {
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
