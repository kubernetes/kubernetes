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

// Win32_PerfRawData_Counters_ProcessorInformation struct
type Win32_PerfRawData_Counters_ProcessorInformation struct {
	*Win32_PerfRawData

	//
	AverageIdleTime uint64

	//
	AverageIdleTime_Base uint64

	//
	C1TransitionsPersec uint64

	//
	C2TransitionsPersec uint64

	//
	C3TransitionsPersec uint64

	//
	ClockInterruptsPersec uint32

	//
	DPCRate uint32

	//
	DPCsQueuedPersec uint32

	//
	IdleBreakEventsPersec uint64

	//
	InterruptsPersec uint32

	//
	ParkingStatus uint32

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
	PercentofMaximumFrequency uint32

	//
	PercentPerformanceLimit uint32

	//
	PercentPriorityTime uint64

	//
	PercentPrivilegedTime uint64

	//
	PercentPrivilegedUtility uint64

	//
	PercentPrivilegedUtility_Base uint32

	//
	PercentProcessorPerformance uint64

	//
	PercentProcessorPerformance_Base uint32

	//
	PercentProcessorTime uint64

	//
	PercentProcessorUtility uint64

	//
	PercentProcessorUtility_Base uint32

	//
	PercentUserTime uint64

	//
	PerformanceLimitFlags uint32

	//
	ProcessorFrequency uint32

	//
	ProcessorStateFlags uint32
}

func NewWin32_PerfRawData_Counters_ProcessorInformationEx1(instance *cim.WmiInstance) (newInstance *Win32_PerfRawData_Counters_ProcessorInformation, err error) {
	tmp, err := NewWin32_PerfRawDataEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfRawData_Counters_ProcessorInformation{
		Win32_PerfRawData: tmp,
	}
	return
}

func NewWin32_PerfRawData_Counters_ProcessorInformationEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_PerfRawData_Counters_ProcessorInformation, err error) {
	tmp, err := NewWin32_PerfRawDataEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfRawData_Counters_ProcessorInformation{
		Win32_PerfRawData: tmp,
	}
	return
}

// SetAverageIdleTime sets the value of AverageIdleTime for the instance
func (instance *Win32_PerfRawData_Counters_ProcessorInformation) SetPropertyAverageIdleTime(value uint64) (err error) {
	return instance.SetProperty("AverageIdleTime", (value))
}

// GetAverageIdleTime gets the value of AverageIdleTime for the instance
func (instance *Win32_PerfRawData_Counters_ProcessorInformation) GetPropertyAverageIdleTime() (value uint64, err error) {
	retValue, err := instance.GetProperty("AverageIdleTime")
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

// SetAverageIdleTime_Base sets the value of AverageIdleTime_Base for the instance
func (instance *Win32_PerfRawData_Counters_ProcessorInformation) SetPropertyAverageIdleTime_Base(value uint64) (err error) {
	return instance.SetProperty("AverageIdleTime_Base", (value))
}

// GetAverageIdleTime_Base gets the value of AverageIdleTime_Base for the instance
func (instance *Win32_PerfRawData_Counters_ProcessorInformation) GetPropertyAverageIdleTime_Base() (value uint64, err error) {
	retValue, err := instance.GetProperty("AverageIdleTime_Base")
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

// SetC1TransitionsPersec sets the value of C1TransitionsPersec for the instance
func (instance *Win32_PerfRawData_Counters_ProcessorInformation) SetPropertyC1TransitionsPersec(value uint64) (err error) {
	return instance.SetProperty("C1TransitionsPersec", (value))
}

// GetC1TransitionsPersec gets the value of C1TransitionsPersec for the instance
func (instance *Win32_PerfRawData_Counters_ProcessorInformation) GetPropertyC1TransitionsPersec() (value uint64, err error) {
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
func (instance *Win32_PerfRawData_Counters_ProcessorInformation) SetPropertyC2TransitionsPersec(value uint64) (err error) {
	return instance.SetProperty("C2TransitionsPersec", (value))
}

// GetC2TransitionsPersec gets the value of C2TransitionsPersec for the instance
func (instance *Win32_PerfRawData_Counters_ProcessorInformation) GetPropertyC2TransitionsPersec() (value uint64, err error) {
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
func (instance *Win32_PerfRawData_Counters_ProcessorInformation) SetPropertyC3TransitionsPersec(value uint64) (err error) {
	return instance.SetProperty("C3TransitionsPersec", (value))
}

// GetC3TransitionsPersec gets the value of C3TransitionsPersec for the instance
func (instance *Win32_PerfRawData_Counters_ProcessorInformation) GetPropertyC3TransitionsPersec() (value uint64, err error) {
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

// SetClockInterruptsPersec sets the value of ClockInterruptsPersec for the instance
func (instance *Win32_PerfRawData_Counters_ProcessorInformation) SetPropertyClockInterruptsPersec(value uint32) (err error) {
	return instance.SetProperty("ClockInterruptsPersec", (value))
}

// GetClockInterruptsPersec gets the value of ClockInterruptsPersec for the instance
func (instance *Win32_PerfRawData_Counters_ProcessorInformation) GetPropertyClockInterruptsPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("ClockInterruptsPersec")
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

// SetDPCRate sets the value of DPCRate for the instance
func (instance *Win32_PerfRawData_Counters_ProcessorInformation) SetPropertyDPCRate(value uint32) (err error) {
	return instance.SetProperty("DPCRate", (value))
}

// GetDPCRate gets the value of DPCRate for the instance
func (instance *Win32_PerfRawData_Counters_ProcessorInformation) GetPropertyDPCRate() (value uint32, err error) {
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
func (instance *Win32_PerfRawData_Counters_ProcessorInformation) SetPropertyDPCsQueuedPersec(value uint32) (err error) {
	return instance.SetProperty("DPCsQueuedPersec", (value))
}

// GetDPCsQueuedPersec gets the value of DPCsQueuedPersec for the instance
func (instance *Win32_PerfRawData_Counters_ProcessorInformation) GetPropertyDPCsQueuedPersec() (value uint32, err error) {
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

// SetIdleBreakEventsPersec sets the value of IdleBreakEventsPersec for the instance
func (instance *Win32_PerfRawData_Counters_ProcessorInformation) SetPropertyIdleBreakEventsPersec(value uint64) (err error) {
	return instance.SetProperty("IdleBreakEventsPersec", (value))
}

// GetIdleBreakEventsPersec gets the value of IdleBreakEventsPersec for the instance
func (instance *Win32_PerfRawData_Counters_ProcessorInformation) GetPropertyIdleBreakEventsPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("IdleBreakEventsPersec")
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

// SetInterruptsPersec sets the value of InterruptsPersec for the instance
func (instance *Win32_PerfRawData_Counters_ProcessorInformation) SetPropertyInterruptsPersec(value uint32) (err error) {
	return instance.SetProperty("InterruptsPersec", (value))
}

// GetInterruptsPersec gets the value of InterruptsPersec for the instance
func (instance *Win32_PerfRawData_Counters_ProcessorInformation) GetPropertyInterruptsPersec() (value uint32, err error) {
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

// SetParkingStatus sets the value of ParkingStatus for the instance
func (instance *Win32_PerfRawData_Counters_ProcessorInformation) SetPropertyParkingStatus(value uint32) (err error) {
	return instance.SetProperty("ParkingStatus", (value))
}

// GetParkingStatus gets the value of ParkingStatus for the instance
func (instance *Win32_PerfRawData_Counters_ProcessorInformation) GetPropertyParkingStatus() (value uint32, err error) {
	retValue, err := instance.GetProperty("ParkingStatus")
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
func (instance *Win32_PerfRawData_Counters_ProcessorInformation) SetPropertyPercentC1Time(value uint64) (err error) {
	return instance.SetProperty("PercentC1Time", (value))
}

// GetPercentC1Time gets the value of PercentC1Time for the instance
func (instance *Win32_PerfRawData_Counters_ProcessorInformation) GetPropertyPercentC1Time() (value uint64, err error) {
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
func (instance *Win32_PerfRawData_Counters_ProcessorInformation) SetPropertyPercentC2Time(value uint64) (err error) {
	return instance.SetProperty("PercentC2Time", (value))
}

// GetPercentC2Time gets the value of PercentC2Time for the instance
func (instance *Win32_PerfRawData_Counters_ProcessorInformation) GetPropertyPercentC2Time() (value uint64, err error) {
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
func (instance *Win32_PerfRawData_Counters_ProcessorInformation) SetPropertyPercentC3Time(value uint64) (err error) {
	return instance.SetProperty("PercentC3Time", (value))
}

// GetPercentC3Time gets the value of PercentC3Time for the instance
func (instance *Win32_PerfRawData_Counters_ProcessorInformation) GetPropertyPercentC3Time() (value uint64, err error) {
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
func (instance *Win32_PerfRawData_Counters_ProcessorInformation) SetPropertyPercentDPCTime(value uint64) (err error) {
	return instance.SetProperty("PercentDPCTime", (value))
}

// GetPercentDPCTime gets the value of PercentDPCTime for the instance
func (instance *Win32_PerfRawData_Counters_ProcessorInformation) GetPropertyPercentDPCTime() (value uint64, err error) {
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
func (instance *Win32_PerfRawData_Counters_ProcessorInformation) SetPropertyPercentIdleTime(value uint64) (err error) {
	return instance.SetProperty("PercentIdleTime", (value))
}

// GetPercentIdleTime gets the value of PercentIdleTime for the instance
func (instance *Win32_PerfRawData_Counters_ProcessorInformation) GetPropertyPercentIdleTime() (value uint64, err error) {
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
func (instance *Win32_PerfRawData_Counters_ProcessorInformation) SetPropertyPercentInterruptTime(value uint64) (err error) {
	return instance.SetProperty("PercentInterruptTime", (value))
}

// GetPercentInterruptTime gets the value of PercentInterruptTime for the instance
func (instance *Win32_PerfRawData_Counters_ProcessorInformation) GetPropertyPercentInterruptTime() (value uint64, err error) {
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

// SetPercentofMaximumFrequency sets the value of PercentofMaximumFrequency for the instance
func (instance *Win32_PerfRawData_Counters_ProcessorInformation) SetPropertyPercentofMaximumFrequency(value uint32) (err error) {
	return instance.SetProperty("PercentofMaximumFrequency", (value))
}

// GetPercentofMaximumFrequency gets the value of PercentofMaximumFrequency for the instance
func (instance *Win32_PerfRawData_Counters_ProcessorInformation) GetPropertyPercentofMaximumFrequency() (value uint32, err error) {
	retValue, err := instance.GetProperty("PercentofMaximumFrequency")
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

// SetPercentPerformanceLimit sets the value of PercentPerformanceLimit for the instance
func (instance *Win32_PerfRawData_Counters_ProcessorInformation) SetPropertyPercentPerformanceLimit(value uint32) (err error) {
	return instance.SetProperty("PercentPerformanceLimit", (value))
}

// GetPercentPerformanceLimit gets the value of PercentPerformanceLimit for the instance
func (instance *Win32_PerfRawData_Counters_ProcessorInformation) GetPropertyPercentPerformanceLimit() (value uint32, err error) {
	retValue, err := instance.GetProperty("PercentPerformanceLimit")
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

// SetPercentPriorityTime sets the value of PercentPriorityTime for the instance
func (instance *Win32_PerfRawData_Counters_ProcessorInformation) SetPropertyPercentPriorityTime(value uint64) (err error) {
	return instance.SetProperty("PercentPriorityTime", (value))
}

// GetPercentPriorityTime gets the value of PercentPriorityTime for the instance
func (instance *Win32_PerfRawData_Counters_ProcessorInformation) GetPropertyPercentPriorityTime() (value uint64, err error) {
	retValue, err := instance.GetProperty("PercentPriorityTime")
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
func (instance *Win32_PerfRawData_Counters_ProcessorInformation) SetPropertyPercentPrivilegedTime(value uint64) (err error) {
	return instance.SetProperty("PercentPrivilegedTime", (value))
}

// GetPercentPrivilegedTime gets the value of PercentPrivilegedTime for the instance
func (instance *Win32_PerfRawData_Counters_ProcessorInformation) GetPropertyPercentPrivilegedTime() (value uint64, err error) {
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

// SetPercentPrivilegedUtility sets the value of PercentPrivilegedUtility for the instance
func (instance *Win32_PerfRawData_Counters_ProcessorInformation) SetPropertyPercentPrivilegedUtility(value uint64) (err error) {
	return instance.SetProperty("PercentPrivilegedUtility", (value))
}

// GetPercentPrivilegedUtility gets the value of PercentPrivilegedUtility for the instance
func (instance *Win32_PerfRawData_Counters_ProcessorInformation) GetPropertyPercentPrivilegedUtility() (value uint64, err error) {
	retValue, err := instance.GetProperty("PercentPrivilegedUtility")
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

// SetPercentPrivilegedUtility_Base sets the value of PercentPrivilegedUtility_Base for the instance
func (instance *Win32_PerfRawData_Counters_ProcessorInformation) SetPropertyPercentPrivilegedUtility_Base(value uint32) (err error) {
	return instance.SetProperty("PercentPrivilegedUtility_Base", (value))
}

// GetPercentPrivilegedUtility_Base gets the value of PercentPrivilegedUtility_Base for the instance
func (instance *Win32_PerfRawData_Counters_ProcessorInformation) GetPropertyPercentPrivilegedUtility_Base() (value uint32, err error) {
	retValue, err := instance.GetProperty("PercentPrivilegedUtility_Base")
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

// SetPercentProcessorPerformance sets the value of PercentProcessorPerformance for the instance
func (instance *Win32_PerfRawData_Counters_ProcessorInformation) SetPropertyPercentProcessorPerformance(value uint64) (err error) {
	return instance.SetProperty("PercentProcessorPerformance", (value))
}

// GetPercentProcessorPerformance gets the value of PercentProcessorPerformance for the instance
func (instance *Win32_PerfRawData_Counters_ProcessorInformation) GetPropertyPercentProcessorPerformance() (value uint64, err error) {
	retValue, err := instance.GetProperty("PercentProcessorPerformance")
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

// SetPercentProcessorPerformance_Base sets the value of PercentProcessorPerformance_Base for the instance
func (instance *Win32_PerfRawData_Counters_ProcessorInformation) SetPropertyPercentProcessorPerformance_Base(value uint32) (err error) {
	return instance.SetProperty("PercentProcessorPerformance_Base", (value))
}

// GetPercentProcessorPerformance_Base gets the value of PercentProcessorPerformance_Base for the instance
func (instance *Win32_PerfRawData_Counters_ProcessorInformation) GetPropertyPercentProcessorPerformance_Base() (value uint32, err error) {
	retValue, err := instance.GetProperty("PercentProcessorPerformance_Base")
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

// SetPercentProcessorTime sets the value of PercentProcessorTime for the instance
func (instance *Win32_PerfRawData_Counters_ProcessorInformation) SetPropertyPercentProcessorTime(value uint64) (err error) {
	return instance.SetProperty("PercentProcessorTime", (value))
}

// GetPercentProcessorTime gets the value of PercentProcessorTime for the instance
func (instance *Win32_PerfRawData_Counters_ProcessorInformation) GetPropertyPercentProcessorTime() (value uint64, err error) {
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

// SetPercentProcessorUtility sets the value of PercentProcessorUtility for the instance
func (instance *Win32_PerfRawData_Counters_ProcessorInformation) SetPropertyPercentProcessorUtility(value uint64) (err error) {
	return instance.SetProperty("PercentProcessorUtility", (value))
}

// GetPercentProcessorUtility gets the value of PercentProcessorUtility for the instance
func (instance *Win32_PerfRawData_Counters_ProcessorInformation) GetPropertyPercentProcessorUtility() (value uint64, err error) {
	retValue, err := instance.GetProperty("PercentProcessorUtility")
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

// SetPercentProcessorUtility_Base sets the value of PercentProcessorUtility_Base for the instance
func (instance *Win32_PerfRawData_Counters_ProcessorInformation) SetPropertyPercentProcessorUtility_Base(value uint32) (err error) {
	return instance.SetProperty("PercentProcessorUtility_Base", (value))
}

// GetPercentProcessorUtility_Base gets the value of PercentProcessorUtility_Base for the instance
func (instance *Win32_PerfRawData_Counters_ProcessorInformation) GetPropertyPercentProcessorUtility_Base() (value uint32, err error) {
	retValue, err := instance.GetProperty("PercentProcessorUtility_Base")
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

// SetPercentUserTime sets the value of PercentUserTime for the instance
func (instance *Win32_PerfRawData_Counters_ProcessorInformation) SetPropertyPercentUserTime(value uint64) (err error) {
	return instance.SetProperty("PercentUserTime", (value))
}

// GetPercentUserTime gets the value of PercentUserTime for the instance
func (instance *Win32_PerfRawData_Counters_ProcessorInformation) GetPropertyPercentUserTime() (value uint64, err error) {
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

// SetPerformanceLimitFlags sets the value of PerformanceLimitFlags for the instance
func (instance *Win32_PerfRawData_Counters_ProcessorInformation) SetPropertyPerformanceLimitFlags(value uint32) (err error) {
	return instance.SetProperty("PerformanceLimitFlags", (value))
}

// GetPerformanceLimitFlags gets the value of PerformanceLimitFlags for the instance
func (instance *Win32_PerfRawData_Counters_ProcessorInformation) GetPropertyPerformanceLimitFlags() (value uint32, err error) {
	retValue, err := instance.GetProperty("PerformanceLimitFlags")
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

// SetProcessorFrequency sets the value of ProcessorFrequency for the instance
func (instance *Win32_PerfRawData_Counters_ProcessorInformation) SetPropertyProcessorFrequency(value uint32) (err error) {
	return instance.SetProperty("ProcessorFrequency", (value))
}

// GetProcessorFrequency gets the value of ProcessorFrequency for the instance
func (instance *Win32_PerfRawData_Counters_ProcessorInformation) GetPropertyProcessorFrequency() (value uint32, err error) {
	retValue, err := instance.GetProperty("ProcessorFrequency")
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

// SetProcessorStateFlags sets the value of ProcessorStateFlags for the instance
func (instance *Win32_PerfRawData_Counters_ProcessorInformation) SetPropertyProcessorStateFlags(value uint32) (err error) {
	return instance.SetProperty("ProcessorStateFlags", (value))
}

// GetProcessorStateFlags gets the value of ProcessorStateFlags for the instance
func (instance *Win32_PerfRawData_Counters_ProcessorInformation) GetPropertyProcessorStateFlags() (value uint32, err error) {
	retValue, err := instance.GetProperty("ProcessorStateFlags")
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
