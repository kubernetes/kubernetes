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

// Win32_PerfRawData_Counters_XHCICommonBuffer struct
type Win32_PerfRawData_Counters_XHCICommonBuffer struct {
	*Win32_PerfRawData

	//
	AllocationCount uint32

	//
	FreeCount uint32

	//
	PagesInUse uint32

	//
	PagesTotal uint32
}

func NewWin32_PerfRawData_Counters_XHCICommonBufferEx1(instance *cim.WmiInstance) (newInstance *Win32_PerfRawData_Counters_XHCICommonBuffer, err error) {
	tmp, err := NewWin32_PerfRawDataEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfRawData_Counters_XHCICommonBuffer{
		Win32_PerfRawData: tmp,
	}
	return
}

func NewWin32_PerfRawData_Counters_XHCICommonBufferEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_PerfRawData_Counters_XHCICommonBuffer, err error) {
	tmp, err := NewWin32_PerfRawDataEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfRawData_Counters_XHCICommonBuffer{
		Win32_PerfRawData: tmp,
	}
	return
}

// SetAllocationCount sets the value of AllocationCount for the instance
func (instance *Win32_PerfRawData_Counters_XHCICommonBuffer) SetPropertyAllocationCount(value uint32) (err error) {
	return instance.SetProperty("AllocationCount", (value))
}

// GetAllocationCount gets the value of AllocationCount for the instance
func (instance *Win32_PerfRawData_Counters_XHCICommonBuffer) GetPropertyAllocationCount() (value uint32, err error) {
	retValue, err := instance.GetProperty("AllocationCount")
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

// SetFreeCount sets the value of FreeCount for the instance
func (instance *Win32_PerfRawData_Counters_XHCICommonBuffer) SetPropertyFreeCount(value uint32) (err error) {
	return instance.SetProperty("FreeCount", (value))
}

// GetFreeCount gets the value of FreeCount for the instance
func (instance *Win32_PerfRawData_Counters_XHCICommonBuffer) GetPropertyFreeCount() (value uint32, err error) {
	retValue, err := instance.GetProperty("FreeCount")
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

// SetPagesInUse sets the value of PagesInUse for the instance
func (instance *Win32_PerfRawData_Counters_XHCICommonBuffer) SetPropertyPagesInUse(value uint32) (err error) {
	return instance.SetProperty("PagesInUse", (value))
}

// GetPagesInUse gets the value of PagesInUse for the instance
func (instance *Win32_PerfRawData_Counters_XHCICommonBuffer) GetPropertyPagesInUse() (value uint32, err error) {
	retValue, err := instance.GetProperty("PagesInUse")
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

// SetPagesTotal sets the value of PagesTotal for the instance
func (instance *Win32_PerfRawData_Counters_XHCICommonBuffer) SetPropertyPagesTotal(value uint32) (err error) {
	return instance.SetProperty("PagesTotal", (value))
}

// GetPagesTotal gets the value of PagesTotal for the instance
func (instance *Win32_PerfRawData_Counters_XHCICommonBuffer) GetPropertyPagesTotal() (value uint32, err error) {
	retValue, err := instance.GetProperty("PagesTotal")
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
