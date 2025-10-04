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

// Win32_PageFileUsage struct
type Win32_PageFileUsage struct {
	*CIM_LogicalElement

	//
	AllocatedBaseSize uint32

	//
	CurrentUsage uint32

	//
	PeakUsage uint32

	//
	TempPageFile bool
}

func NewWin32_PageFileUsageEx1(instance *cim.WmiInstance) (newInstance *Win32_PageFileUsage, err error) {
	tmp, err := NewCIM_LogicalElementEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_PageFileUsage{
		CIM_LogicalElement: tmp,
	}
	return
}

func NewWin32_PageFileUsageEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_PageFileUsage, err error) {
	tmp, err := NewCIM_LogicalElementEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_PageFileUsage{
		CIM_LogicalElement: tmp,
	}
	return
}

// SetAllocatedBaseSize sets the value of AllocatedBaseSize for the instance
func (instance *Win32_PageFileUsage) SetPropertyAllocatedBaseSize(value uint32) (err error) {
	return instance.SetProperty("AllocatedBaseSize", (value))
}

// GetAllocatedBaseSize gets the value of AllocatedBaseSize for the instance
func (instance *Win32_PageFileUsage) GetPropertyAllocatedBaseSize() (value uint32, err error) {
	retValue, err := instance.GetProperty("AllocatedBaseSize")
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

// SetCurrentUsage sets the value of CurrentUsage for the instance
func (instance *Win32_PageFileUsage) SetPropertyCurrentUsage(value uint32) (err error) {
	return instance.SetProperty("CurrentUsage", (value))
}

// GetCurrentUsage gets the value of CurrentUsage for the instance
func (instance *Win32_PageFileUsage) GetPropertyCurrentUsage() (value uint32, err error) {
	retValue, err := instance.GetProperty("CurrentUsage")
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

// SetPeakUsage sets the value of PeakUsage for the instance
func (instance *Win32_PageFileUsage) SetPropertyPeakUsage(value uint32) (err error) {
	return instance.SetProperty("PeakUsage", (value))
}

// GetPeakUsage gets the value of PeakUsage for the instance
func (instance *Win32_PageFileUsage) GetPropertyPeakUsage() (value uint32, err error) {
	retValue, err := instance.GetProperty("PeakUsage")
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

// SetTempPageFile sets the value of TempPageFile for the instance
func (instance *Win32_PageFileUsage) SetPropertyTempPageFile(value bool) (err error) {
	return instance.SetProperty("TempPageFile", (value))
}

// GetTempPageFile gets the value of TempPageFile for the instance
func (instance *Win32_PageFileUsage) GetPropertyTempPageFile() (value bool, err error) {
	retValue, err := instance.GetProperty("TempPageFile")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(bool)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " bool is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = bool(valuetmp)

	return
}
