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

// Win32_PerfFormattedData_Counters_Netlogon struct
type Win32_PerfFormattedData_Counters_Netlogon struct {
	*Win32_PerfFormattedData

	//
	AverageSemaphoreHoldTime uint32

	//
	LastAuthenticationTime uint32

	//
	SemaphoreAcquires uint64

	//
	SemaphoreHolders uint32

	//
	SemaphoreTimeouts uint64

	//
	SemaphoreWaiters uint32
}

func NewWin32_PerfFormattedData_Counters_NetlogonEx1(instance *cim.WmiInstance) (newInstance *Win32_PerfFormattedData_Counters_Netlogon, err error) {
	tmp, err := NewWin32_PerfFormattedDataEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfFormattedData_Counters_Netlogon{
		Win32_PerfFormattedData: tmp,
	}
	return
}

func NewWin32_PerfFormattedData_Counters_NetlogonEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_PerfFormattedData_Counters_Netlogon, err error) {
	tmp, err := NewWin32_PerfFormattedDataEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfFormattedData_Counters_Netlogon{
		Win32_PerfFormattedData: tmp,
	}
	return
}

// SetAverageSemaphoreHoldTime sets the value of AverageSemaphoreHoldTime for the instance
func (instance *Win32_PerfFormattedData_Counters_Netlogon) SetPropertyAverageSemaphoreHoldTime(value uint32) (err error) {
	return instance.SetProperty("AverageSemaphoreHoldTime", (value))
}

// GetAverageSemaphoreHoldTime gets the value of AverageSemaphoreHoldTime for the instance
func (instance *Win32_PerfFormattedData_Counters_Netlogon) GetPropertyAverageSemaphoreHoldTime() (value uint32, err error) {
	retValue, err := instance.GetProperty("AverageSemaphoreHoldTime")
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

// SetLastAuthenticationTime sets the value of LastAuthenticationTime for the instance
func (instance *Win32_PerfFormattedData_Counters_Netlogon) SetPropertyLastAuthenticationTime(value uint32) (err error) {
	return instance.SetProperty("LastAuthenticationTime", (value))
}

// GetLastAuthenticationTime gets the value of LastAuthenticationTime for the instance
func (instance *Win32_PerfFormattedData_Counters_Netlogon) GetPropertyLastAuthenticationTime() (value uint32, err error) {
	retValue, err := instance.GetProperty("LastAuthenticationTime")
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

// SetSemaphoreAcquires sets the value of SemaphoreAcquires for the instance
func (instance *Win32_PerfFormattedData_Counters_Netlogon) SetPropertySemaphoreAcquires(value uint64) (err error) {
	return instance.SetProperty("SemaphoreAcquires", (value))
}

// GetSemaphoreAcquires gets the value of SemaphoreAcquires for the instance
func (instance *Win32_PerfFormattedData_Counters_Netlogon) GetPropertySemaphoreAcquires() (value uint64, err error) {
	retValue, err := instance.GetProperty("SemaphoreAcquires")
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

// SetSemaphoreHolders sets the value of SemaphoreHolders for the instance
func (instance *Win32_PerfFormattedData_Counters_Netlogon) SetPropertySemaphoreHolders(value uint32) (err error) {
	return instance.SetProperty("SemaphoreHolders", (value))
}

// GetSemaphoreHolders gets the value of SemaphoreHolders for the instance
func (instance *Win32_PerfFormattedData_Counters_Netlogon) GetPropertySemaphoreHolders() (value uint32, err error) {
	retValue, err := instance.GetProperty("SemaphoreHolders")
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

// SetSemaphoreTimeouts sets the value of SemaphoreTimeouts for the instance
func (instance *Win32_PerfFormattedData_Counters_Netlogon) SetPropertySemaphoreTimeouts(value uint64) (err error) {
	return instance.SetProperty("SemaphoreTimeouts", (value))
}

// GetSemaphoreTimeouts gets the value of SemaphoreTimeouts for the instance
func (instance *Win32_PerfFormattedData_Counters_Netlogon) GetPropertySemaphoreTimeouts() (value uint64, err error) {
	retValue, err := instance.GetProperty("SemaphoreTimeouts")
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

// SetSemaphoreWaiters sets the value of SemaphoreWaiters for the instance
func (instance *Win32_PerfFormattedData_Counters_Netlogon) SetPropertySemaphoreWaiters(value uint32) (err error) {
	return instance.SetProperty("SemaphoreWaiters", (value))
}

// GetSemaphoreWaiters gets the value of SemaphoreWaiters for the instance
func (instance *Win32_PerfFormattedData_Counters_Netlogon) GetPropertySemaphoreWaiters() (value uint32, err error) {
	retValue, err := instance.GetProperty("SemaphoreWaiters")
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
