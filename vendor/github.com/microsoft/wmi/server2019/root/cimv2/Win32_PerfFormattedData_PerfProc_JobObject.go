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

// Win32_PerfFormattedData_PerfProc_JobObject struct
type Win32_PerfFormattedData_PerfProc_JobObject struct {
	*Win32_PerfFormattedData

	//
	CurrentPercentKernelModeTime uint64

	//
	CurrentPercentProcessorTime uint64

	//
	CurrentPercentUserModeTime uint64

	//
	PagesPerSec uint32

	//
	ProcessCountActive uint32

	//
	ProcessCountTerminated uint32

	//
	ProcessCountTotal uint32

	//
	ThisPeriodmSecKernelMode uint64

	//
	ThisPeriodmSecProcessor uint64

	//
	ThisPeriodmSecUserMode uint64

	//
	TotalmSecKernelMode uint64

	//
	TotalmSecProcessor uint64

	//
	TotalmSecUserMode uint64
}

func NewWin32_PerfFormattedData_PerfProc_JobObjectEx1(instance *cim.WmiInstance) (newInstance *Win32_PerfFormattedData_PerfProc_JobObject, err error) {
	tmp, err := NewWin32_PerfFormattedDataEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfFormattedData_PerfProc_JobObject{
		Win32_PerfFormattedData: tmp,
	}
	return
}

func NewWin32_PerfFormattedData_PerfProc_JobObjectEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_PerfFormattedData_PerfProc_JobObject, err error) {
	tmp, err := NewWin32_PerfFormattedDataEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfFormattedData_PerfProc_JobObject{
		Win32_PerfFormattedData: tmp,
	}
	return
}

// SetCurrentPercentKernelModeTime sets the value of CurrentPercentKernelModeTime for the instance
func (instance *Win32_PerfFormattedData_PerfProc_JobObject) SetPropertyCurrentPercentKernelModeTime(value uint64) (err error) {
	return instance.SetProperty("CurrentPercentKernelModeTime", (value))
}

// GetCurrentPercentKernelModeTime gets the value of CurrentPercentKernelModeTime for the instance
func (instance *Win32_PerfFormattedData_PerfProc_JobObject) GetPropertyCurrentPercentKernelModeTime() (value uint64, err error) {
	retValue, err := instance.GetProperty("CurrentPercentKernelModeTime")
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

// SetCurrentPercentProcessorTime sets the value of CurrentPercentProcessorTime for the instance
func (instance *Win32_PerfFormattedData_PerfProc_JobObject) SetPropertyCurrentPercentProcessorTime(value uint64) (err error) {
	return instance.SetProperty("CurrentPercentProcessorTime", (value))
}

// GetCurrentPercentProcessorTime gets the value of CurrentPercentProcessorTime for the instance
func (instance *Win32_PerfFormattedData_PerfProc_JobObject) GetPropertyCurrentPercentProcessorTime() (value uint64, err error) {
	retValue, err := instance.GetProperty("CurrentPercentProcessorTime")
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

// SetCurrentPercentUserModeTime sets the value of CurrentPercentUserModeTime for the instance
func (instance *Win32_PerfFormattedData_PerfProc_JobObject) SetPropertyCurrentPercentUserModeTime(value uint64) (err error) {
	return instance.SetProperty("CurrentPercentUserModeTime", (value))
}

// GetCurrentPercentUserModeTime gets the value of CurrentPercentUserModeTime for the instance
func (instance *Win32_PerfFormattedData_PerfProc_JobObject) GetPropertyCurrentPercentUserModeTime() (value uint64, err error) {
	retValue, err := instance.GetProperty("CurrentPercentUserModeTime")
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

// SetPagesPerSec sets the value of PagesPerSec for the instance
func (instance *Win32_PerfFormattedData_PerfProc_JobObject) SetPropertyPagesPerSec(value uint32) (err error) {
	return instance.SetProperty("PagesPerSec", (value))
}

// GetPagesPerSec gets the value of PagesPerSec for the instance
func (instance *Win32_PerfFormattedData_PerfProc_JobObject) GetPropertyPagesPerSec() (value uint32, err error) {
	retValue, err := instance.GetProperty("PagesPerSec")
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

// SetProcessCountActive sets the value of ProcessCountActive for the instance
func (instance *Win32_PerfFormattedData_PerfProc_JobObject) SetPropertyProcessCountActive(value uint32) (err error) {
	return instance.SetProperty("ProcessCountActive", (value))
}

// GetProcessCountActive gets the value of ProcessCountActive for the instance
func (instance *Win32_PerfFormattedData_PerfProc_JobObject) GetPropertyProcessCountActive() (value uint32, err error) {
	retValue, err := instance.GetProperty("ProcessCountActive")
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

// SetProcessCountTerminated sets the value of ProcessCountTerminated for the instance
func (instance *Win32_PerfFormattedData_PerfProc_JobObject) SetPropertyProcessCountTerminated(value uint32) (err error) {
	return instance.SetProperty("ProcessCountTerminated", (value))
}

// GetProcessCountTerminated gets the value of ProcessCountTerminated for the instance
func (instance *Win32_PerfFormattedData_PerfProc_JobObject) GetPropertyProcessCountTerminated() (value uint32, err error) {
	retValue, err := instance.GetProperty("ProcessCountTerminated")
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

// SetProcessCountTotal sets the value of ProcessCountTotal for the instance
func (instance *Win32_PerfFormattedData_PerfProc_JobObject) SetPropertyProcessCountTotal(value uint32) (err error) {
	return instance.SetProperty("ProcessCountTotal", (value))
}

// GetProcessCountTotal gets the value of ProcessCountTotal for the instance
func (instance *Win32_PerfFormattedData_PerfProc_JobObject) GetPropertyProcessCountTotal() (value uint32, err error) {
	retValue, err := instance.GetProperty("ProcessCountTotal")
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

// SetThisPeriodmSecKernelMode sets the value of ThisPeriodmSecKernelMode for the instance
func (instance *Win32_PerfFormattedData_PerfProc_JobObject) SetPropertyThisPeriodmSecKernelMode(value uint64) (err error) {
	return instance.SetProperty("ThisPeriodmSecKernelMode", (value))
}

// GetThisPeriodmSecKernelMode gets the value of ThisPeriodmSecKernelMode for the instance
func (instance *Win32_PerfFormattedData_PerfProc_JobObject) GetPropertyThisPeriodmSecKernelMode() (value uint64, err error) {
	retValue, err := instance.GetProperty("ThisPeriodmSecKernelMode")
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

// SetThisPeriodmSecProcessor sets the value of ThisPeriodmSecProcessor for the instance
func (instance *Win32_PerfFormattedData_PerfProc_JobObject) SetPropertyThisPeriodmSecProcessor(value uint64) (err error) {
	return instance.SetProperty("ThisPeriodmSecProcessor", (value))
}

// GetThisPeriodmSecProcessor gets the value of ThisPeriodmSecProcessor for the instance
func (instance *Win32_PerfFormattedData_PerfProc_JobObject) GetPropertyThisPeriodmSecProcessor() (value uint64, err error) {
	retValue, err := instance.GetProperty("ThisPeriodmSecProcessor")
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

// SetThisPeriodmSecUserMode sets the value of ThisPeriodmSecUserMode for the instance
func (instance *Win32_PerfFormattedData_PerfProc_JobObject) SetPropertyThisPeriodmSecUserMode(value uint64) (err error) {
	return instance.SetProperty("ThisPeriodmSecUserMode", (value))
}

// GetThisPeriodmSecUserMode gets the value of ThisPeriodmSecUserMode for the instance
func (instance *Win32_PerfFormattedData_PerfProc_JobObject) GetPropertyThisPeriodmSecUserMode() (value uint64, err error) {
	retValue, err := instance.GetProperty("ThisPeriodmSecUserMode")
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

// SetTotalmSecKernelMode sets the value of TotalmSecKernelMode for the instance
func (instance *Win32_PerfFormattedData_PerfProc_JobObject) SetPropertyTotalmSecKernelMode(value uint64) (err error) {
	return instance.SetProperty("TotalmSecKernelMode", (value))
}

// GetTotalmSecKernelMode gets the value of TotalmSecKernelMode for the instance
func (instance *Win32_PerfFormattedData_PerfProc_JobObject) GetPropertyTotalmSecKernelMode() (value uint64, err error) {
	retValue, err := instance.GetProperty("TotalmSecKernelMode")
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

// SetTotalmSecProcessor sets the value of TotalmSecProcessor for the instance
func (instance *Win32_PerfFormattedData_PerfProc_JobObject) SetPropertyTotalmSecProcessor(value uint64) (err error) {
	return instance.SetProperty("TotalmSecProcessor", (value))
}

// GetTotalmSecProcessor gets the value of TotalmSecProcessor for the instance
func (instance *Win32_PerfFormattedData_PerfProc_JobObject) GetPropertyTotalmSecProcessor() (value uint64, err error) {
	retValue, err := instance.GetProperty("TotalmSecProcessor")
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

// SetTotalmSecUserMode sets the value of TotalmSecUserMode for the instance
func (instance *Win32_PerfFormattedData_PerfProc_JobObject) SetPropertyTotalmSecUserMode(value uint64) (err error) {
	return instance.SetProperty("TotalmSecUserMode", (value))
}

// GetTotalmSecUserMode gets the value of TotalmSecUserMode for the instance
func (instance *Win32_PerfFormattedData_PerfProc_JobObject) GetPropertyTotalmSecUserMode() (value uint64, err error) {
	retValue, err := instance.GetProperty("TotalmSecUserMode")
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
