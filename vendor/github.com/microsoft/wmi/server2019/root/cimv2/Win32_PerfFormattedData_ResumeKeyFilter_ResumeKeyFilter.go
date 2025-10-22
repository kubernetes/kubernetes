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

// Win32_PerfFormattedData_ResumeKeyFilter_ResumeKeyFilter struct
type Win32_PerfFormattedData_ResumeKeyFilter_ResumeKeyFilter struct {
	*Win32_PerfFormattedData

	//
	CancelledHandleCount uint64

	//
	CurrentActiveHandleCount uint64

	//
	CurrentInactiveHandleCount uint64

	//
	FSFailedResumeHandleCount uint64

	//
	ReplayedHandleCount uint64

	//
	ResumedHandleCount uint64

	//
	RKFailedResumeHandleCount uint64

	//
	SuspendedHandleCount uint64
}

func NewWin32_PerfFormattedData_ResumeKeyFilter_ResumeKeyFilterEx1(instance *cim.WmiInstance) (newInstance *Win32_PerfFormattedData_ResumeKeyFilter_ResumeKeyFilter, err error) {
	tmp, err := NewWin32_PerfFormattedDataEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfFormattedData_ResumeKeyFilter_ResumeKeyFilter{
		Win32_PerfFormattedData: tmp,
	}
	return
}

func NewWin32_PerfFormattedData_ResumeKeyFilter_ResumeKeyFilterEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_PerfFormattedData_ResumeKeyFilter_ResumeKeyFilter, err error) {
	tmp, err := NewWin32_PerfFormattedDataEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfFormattedData_ResumeKeyFilter_ResumeKeyFilter{
		Win32_PerfFormattedData: tmp,
	}
	return
}

// SetCancelledHandleCount sets the value of CancelledHandleCount for the instance
func (instance *Win32_PerfFormattedData_ResumeKeyFilter_ResumeKeyFilter) SetPropertyCancelledHandleCount(value uint64) (err error) {
	return instance.SetProperty("CancelledHandleCount", (value))
}

// GetCancelledHandleCount gets the value of CancelledHandleCount for the instance
func (instance *Win32_PerfFormattedData_ResumeKeyFilter_ResumeKeyFilter) GetPropertyCancelledHandleCount() (value uint64, err error) {
	retValue, err := instance.GetProperty("CancelledHandleCount")
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

// SetCurrentActiveHandleCount sets the value of CurrentActiveHandleCount for the instance
func (instance *Win32_PerfFormattedData_ResumeKeyFilter_ResumeKeyFilter) SetPropertyCurrentActiveHandleCount(value uint64) (err error) {
	return instance.SetProperty("CurrentActiveHandleCount", (value))
}

// GetCurrentActiveHandleCount gets the value of CurrentActiveHandleCount for the instance
func (instance *Win32_PerfFormattedData_ResumeKeyFilter_ResumeKeyFilter) GetPropertyCurrentActiveHandleCount() (value uint64, err error) {
	retValue, err := instance.GetProperty("CurrentActiveHandleCount")
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

// SetCurrentInactiveHandleCount sets the value of CurrentInactiveHandleCount for the instance
func (instance *Win32_PerfFormattedData_ResumeKeyFilter_ResumeKeyFilter) SetPropertyCurrentInactiveHandleCount(value uint64) (err error) {
	return instance.SetProperty("CurrentInactiveHandleCount", (value))
}

// GetCurrentInactiveHandleCount gets the value of CurrentInactiveHandleCount for the instance
func (instance *Win32_PerfFormattedData_ResumeKeyFilter_ResumeKeyFilter) GetPropertyCurrentInactiveHandleCount() (value uint64, err error) {
	retValue, err := instance.GetProperty("CurrentInactiveHandleCount")
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

// SetFSFailedResumeHandleCount sets the value of FSFailedResumeHandleCount for the instance
func (instance *Win32_PerfFormattedData_ResumeKeyFilter_ResumeKeyFilter) SetPropertyFSFailedResumeHandleCount(value uint64) (err error) {
	return instance.SetProperty("FSFailedResumeHandleCount", (value))
}

// GetFSFailedResumeHandleCount gets the value of FSFailedResumeHandleCount for the instance
func (instance *Win32_PerfFormattedData_ResumeKeyFilter_ResumeKeyFilter) GetPropertyFSFailedResumeHandleCount() (value uint64, err error) {
	retValue, err := instance.GetProperty("FSFailedResumeHandleCount")
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

// SetReplayedHandleCount sets the value of ReplayedHandleCount for the instance
func (instance *Win32_PerfFormattedData_ResumeKeyFilter_ResumeKeyFilter) SetPropertyReplayedHandleCount(value uint64) (err error) {
	return instance.SetProperty("ReplayedHandleCount", (value))
}

// GetReplayedHandleCount gets the value of ReplayedHandleCount for the instance
func (instance *Win32_PerfFormattedData_ResumeKeyFilter_ResumeKeyFilter) GetPropertyReplayedHandleCount() (value uint64, err error) {
	retValue, err := instance.GetProperty("ReplayedHandleCount")
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

// SetResumedHandleCount sets the value of ResumedHandleCount for the instance
func (instance *Win32_PerfFormattedData_ResumeKeyFilter_ResumeKeyFilter) SetPropertyResumedHandleCount(value uint64) (err error) {
	return instance.SetProperty("ResumedHandleCount", (value))
}

// GetResumedHandleCount gets the value of ResumedHandleCount for the instance
func (instance *Win32_PerfFormattedData_ResumeKeyFilter_ResumeKeyFilter) GetPropertyResumedHandleCount() (value uint64, err error) {
	retValue, err := instance.GetProperty("ResumedHandleCount")
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

// SetRKFailedResumeHandleCount sets the value of RKFailedResumeHandleCount for the instance
func (instance *Win32_PerfFormattedData_ResumeKeyFilter_ResumeKeyFilter) SetPropertyRKFailedResumeHandleCount(value uint64) (err error) {
	return instance.SetProperty("RKFailedResumeHandleCount", (value))
}

// GetRKFailedResumeHandleCount gets the value of RKFailedResumeHandleCount for the instance
func (instance *Win32_PerfFormattedData_ResumeKeyFilter_ResumeKeyFilter) GetPropertyRKFailedResumeHandleCount() (value uint64, err error) {
	retValue, err := instance.GetProperty("RKFailedResumeHandleCount")
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

// SetSuspendedHandleCount sets the value of SuspendedHandleCount for the instance
func (instance *Win32_PerfFormattedData_ResumeKeyFilter_ResumeKeyFilter) SetPropertySuspendedHandleCount(value uint64) (err error) {
	return instance.SetProperty("SuspendedHandleCount", (value))
}

// GetSuspendedHandleCount gets the value of SuspendedHandleCount for the instance
func (instance *Win32_PerfFormattedData_ResumeKeyFilter_ResumeKeyFilter) GetPropertySuspendedHandleCount() (value uint64, err error) {
	retValue, err := instance.GetProperty("SuspendedHandleCount")
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
