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

// Win32_PerfFormattedData_NETFramework_NETCLRExceptions struct
type Win32_PerfFormattedData_NETFramework_NETCLRExceptions struct {
	*Win32_PerfFormattedData

	//
	NumberofExcepsThrown uint32

	//
	NumberofExcepsThrownPersec uint32

	//
	NumberofFiltersPersec uint32

	//
	NumberofFinallysPersec uint32

	//
	ThrowToCatchDepthPersec uint32
}

func NewWin32_PerfFormattedData_NETFramework_NETCLRExceptionsEx1(instance *cim.WmiInstance) (newInstance *Win32_PerfFormattedData_NETFramework_NETCLRExceptions, err error) {
	tmp, err := NewWin32_PerfFormattedDataEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfFormattedData_NETFramework_NETCLRExceptions{
		Win32_PerfFormattedData: tmp,
	}
	return
}

func NewWin32_PerfFormattedData_NETFramework_NETCLRExceptionsEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_PerfFormattedData_NETFramework_NETCLRExceptions, err error) {
	tmp, err := NewWin32_PerfFormattedDataEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfFormattedData_NETFramework_NETCLRExceptions{
		Win32_PerfFormattedData: tmp,
	}
	return
}

// SetNumberofExcepsThrown sets the value of NumberofExcepsThrown for the instance
func (instance *Win32_PerfFormattedData_NETFramework_NETCLRExceptions) SetPropertyNumberofExcepsThrown(value uint32) (err error) {
	return instance.SetProperty("NumberofExcepsThrown", (value))
}

// GetNumberofExcepsThrown gets the value of NumberofExcepsThrown for the instance
func (instance *Win32_PerfFormattedData_NETFramework_NETCLRExceptions) GetPropertyNumberofExcepsThrown() (value uint32, err error) {
	retValue, err := instance.GetProperty("NumberofExcepsThrown")
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

// SetNumberofExcepsThrownPersec sets the value of NumberofExcepsThrownPersec for the instance
func (instance *Win32_PerfFormattedData_NETFramework_NETCLRExceptions) SetPropertyNumberofExcepsThrownPersec(value uint32) (err error) {
	return instance.SetProperty("NumberofExcepsThrownPersec", (value))
}

// GetNumberofExcepsThrownPersec gets the value of NumberofExcepsThrownPersec for the instance
func (instance *Win32_PerfFormattedData_NETFramework_NETCLRExceptions) GetPropertyNumberofExcepsThrownPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("NumberofExcepsThrownPersec")
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

// SetNumberofFiltersPersec sets the value of NumberofFiltersPersec for the instance
func (instance *Win32_PerfFormattedData_NETFramework_NETCLRExceptions) SetPropertyNumberofFiltersPersec(value uint32) (err error) {
	return instance.SetProperty("NumberofFiltersPersec", (value))
}

// GetNumberofFiltersPersec gets the value of NumberofFiltersPersec for the instance
func (instance *Win32_PerfFormattedData_NETFramework_NETCLRExceptions) GetPropertyNumberofFiltersPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("NumberofFiltersPersec")
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

// SetNumberofFinallysPersec sets the value of NumberofFinallysPersec for the instance
func (instance *Win32_PerfFormattedData_NETFramework_NETCLRExceptions) SetPropertyNumberofFinallysPersec(value uint32) (err error) {
	return instance.SetProperty("NumberofFinallysPersec", (value))
}

// GetNumberofFinallysPersec gets the value of NumberofFinallysPersec for the instance
func (instance *Win32_PerfFormattedData_NETFramework_NETCLRExceptions) GetPropertyNumberofFinallysPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("NumberofFinallysPersec")
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

// SetThrowToCatchDepthPersec sets the value of ThrowToCatchDepthPersec for the instance
func (instance *Win32_PerfFormattedData_NETFramework_NETCLRExceptions) SetPropertyThrowToCatchDepthPersec(value uint32) (err error) {
	return instance.SetProperty("ThrowToCatchDepthPersec", (value))
}

// GetThrowToCatchDepthPersec gets the value of ThrowToCatchDepthPersec for the instance
func (instance *Win32_PerfFormattedData_NETFramework_NETCLRExceptions) GetPropertyThrowToCatchDepthPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("ThrowToCatchDepthPersec")
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
