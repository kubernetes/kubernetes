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

// Win32_PerfRawData_Counters_EventTracingforWindows struct
type Win32_PerfRawData_Counters_EventTracingforWindows struct {
	*Win32_PerfRawData

	//
	TotalMemoryUsageNonPagedPool uint32

	//
	TotalMemoryUsagePagedPool uint32

	//
	TotalNumberofActiveSessions uint32

	//
	TotalNumberofDistinctDisabledProviders uint32

	//
	TotalNumberofDistinctEnabledProviders uint32

	//
	TotalNumberofDistinctPreEnabledProviders uint32
}

func NewWin32_PerfRawData_Counters_EventTracingforWindowsEx1(instance *cim.WmiInstance) (newInstance *Win32_PerfRawData_Counters_EventTracingforWindows, err error) {
	tmp, err := NewWin32_PerfRawDataEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfRawData_Counters_EventTracingforWindows{
		Win32_PerfRawData: tmp,
	}
	return
}

func NewWin32_PerfRawData_Counters_EventTracingforWindowsEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_PerfRawData_Counters_EventTracingforWindows, err error) {
	tmp, err := NewWin32_PerfRawDataEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfRawData_Counters_EventTracingforWindows{
		Win32_PerfRawData: tmp,
	}
	return
}

// SetTotalMemoryUsageNonPagedPool sets the value of TotalMemoryUsageNonPagedPool for the instance
func (instance *Win32_PerfRawData_Counters_EventTracingforWindows) SetPropertyTotalMemoryUsageNonPagedPool(value uint32) (err error) {
	return instance.SetProperty("TotalMemoryUsageNonPagedPool", (value))
}

// GetTotalMemoryUsageNonPagedPool gets the value of TotalMemoryUsageNonPagedPool for the instance
func (instance *Win32_PerfRawData_Counters_EventTracingforWindows) GetPropertyTotalMemoryUsageNonPagedPool() (value uint32, err error) {
	retValue, err := instance.GetProperty("TotalMemoryUsageNonPagedPool")
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

// SetTotalMemoryUsagePagedPool sets the value of TotalMemoryUsagePagedPool for the instance
func (instance *Win32_PerfRawData_Counters_EventTracingforWindows) SetPropertyTotalMemoryUsagePagedPool(value uint32) (err error) {
	return instance.SetProperty("TotalMemoryUsagePagedPool", (value))
}

// GetTotalMemoryUsagePagedPool gets the value of TotalMemoryUsagePagedPool for the instance
func (instance *Win32_PerfRawData_Counters_EventTracingforWindows) GetPropertyTotalMemoryUsagePagedPool() (value uint32, err error) {
	retValue, err := instance.GetProperty("TotalMemoryUsagePagedPool")
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

// SetTotalNumberofActiveSessions sets the value of TotalNumberofActiveSessions for the instance
func (instance *Win32_PerfRawData_Counters_EventTracingforWindows) SetPropertyTotalNumberofActiveSessions(value uint32) (err error) {
	return instance.SetProperty("TotalNumberofActiveSessions", (value))
}

// GetTotalNumberofActiveSessions gets the value of TotalNumberofActiveSessions for the instance
func (instance *Win32_PerfRawData_Counters_EventTracingforWindows) GetPropertyTotalNumberofActiveSessions() (value uint32, err error) {
	retValue, err := instance.GetProperty("TotalNumberofActiveSessions")
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

// SetTotalNumberofDistinctDisabledProviders sets the value of TotalNumberofDistinctDisabledProviders for the instance
func (instance *Win32_PerfRawData_Counters_EventTracingforWindows) SetPropertyTotalNumberofDistinctDisabledProviders(value uint32) (err error) {
	return instance.SetProperty("TotalNumberofDistinctDisabledProviders", (value))
}

// GetTotalNumberofDistinctDisabledProviders gets the value of TotalNumberofDistinctDisabledProviders for the instance
func (instance *Win32_PerfRawData_Counters_EventTracingforWindows) GetPropertyTotalNumberofDistinctDisabledProviders() (value uint32, err error) {
	retValue, err := instance.GetProperty("TotalNumberofDistinctDisabledProviders")
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

// SetTotalNumberofDistinctEnabledProviders sets the value of TotalNumberofDistinctEnabledProviders for the instance
func (instance *Win32_PerfRawData_Counters_EventTracingforWindows) SetPropertyTotalNumberofDistinctEnabledProviders(value uint32) (err error) {
	return instance.SetProperty("TotalNumberofDistinctEnabledProviders", (value))
}

// GetTotalNumberofDistinctEnabledProviders gets the value of TotalNumberofDistinctEnabledProviders for the instance
func (instance *Win32_PerfRawData_Counters_EventTracingforWindows) GetPropertyTotalNumberofDistinctEnabledProviders() (value uint32, err error) {
	retValue, err := instance.GetProperty("TotalNumberofDistinctEnabledProviders")
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

// SetTotalNumberofDistinctPreEnabledProviders sets the value of TotalNumberofDistinctPreEnabledProviders for the instance
func (instance *Win32_PerfRawData_Counters_EventTracingforWindows) SetPropertyTotalNumberofDistinctPreEnabledProviders(value uint32) (err error) {
	return instance.SetProperty("TotalNumberofDistinctPreEnabledProviders", (value))
}

// GetTotalNumberofDistinctPreEnabledProviders gets the value of TotalNumberofDistinctPreEnabledProviders for the instance
func (instance *Win32_PerfRawData_Counters_EventTracingforWindows) GetPropertyTotalNumberofDistinctPreEnabledProviders() (value uint32, err error) {
	retValue, err := instance.GetProperty("TotalNumberofDistinctPreEnabledProviders")
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
