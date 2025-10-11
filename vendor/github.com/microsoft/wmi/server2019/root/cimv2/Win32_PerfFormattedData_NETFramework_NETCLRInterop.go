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

// Win32_PerfFormattedData_NETFramework_NETCLRInterop struct
type Win32_PerfFormattedData_NETFramework_NETCLRInterop struct {
	*Win32_PerfFormattedData

	//
	NumberofCCWs uint32

	//
	Numberofmarshalling uint32

	//
	NumberofStubs uint32

	//
	NumberofTLBexportsPersec uint32

	//
	NumberofTLBimportsPersec uint32
}

func NewWin32_PerfFormattedData_NETFramework_NETCLRInteropEx1(instance *cim.WmiInstance) (newInstance *Win32_PerfFormattedData_NETFramework_NETCLRInterop, err error) {
	tmp, err := NewWin32_PerfFormattedDataEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfFormattedData_NETFramework_NETCLRInterop{
		Win32_PerfFormattedData: tmp,
	}
	return
}

func NewWin32_PerfFormattedData_NETFramework_NETCLRInteropEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_PerfFormattedData_NETFramework_NETCLRInterop, err error) {
	tmp, err := NewWin32_PerfFormattedDataEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfFormattedData_NETFramework_NETCLRInterop{
		Win32_PerfFormattedData: tmp,
	}
	return
}

// SetNumberofCCWs sets the value of NumberofCCWs for the instance
func (instance *Win32_PerfFormattedData_NETFramework_NETCLRInterop) SetPropertyNumberofCCWs(value uint32) (err error) {
	return instance.SetProperty("NumberofCCWs", (value))
}

// GetNumberofCCWs gets the value of NumberofCCWs for the instance
func (instance *Win32_PerfFormattedData_NETFramework_NETCLRInterop) GetPropertyNumberofCCWs() (value uint32, err error) {
	retValue, err := instance.GetProperty("NumberofCCWs")
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

// SetNumberofmarshalling sets the value of Numberofmarshalling for the instance
func (instance *Win32_PerfFormattedData_NETFramework_NETCLRInterop) SetPropertyNumberofmarshalling(value uint32) (err error) {
	return instance.SetProperty("Numberofmarshalling", (value))
}

// GetNumberofmarshalling gets the value of Numberofmarshalling for the instance
func (instance *Win32_PerfFormattedData_NETFramework_NETCLRInterop) GetPropertyNumberofmarshalling() (value uint32, err error) {
	retValue, err := instance.GetProperty("Numberofmarshalling")
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

// SetNumberofStubs sets the value of NumberofStubs for the instance
func (instance *Win32_PerfFormattedData_NETFramework_NETCLRInterop) SetPropertyNumberofStubs(value uint32) (err error) {
	return instance.SetProperty("NumberofStubs", (value))
}

// GetNumberofStubs gets the value of NumberofStubs for the instance
func (instance *Win32_PerfFormattedData_NETFramework_NETCLRInterop) GetPropertyNumberofStubs() (value uint32, err error) {
	retValue, err := instance.GetProperty("NumberofStubs")
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

// SetNumberofTLBexportsPersec sets the value of NumberofTLBexportsPersec for the instance
func (instance *Win32_PerfFormattedData_NETFramework_NETCLRInterop) SetPropertyNumberofTLBexportsPersec(value uint32) (err error) {
	return instance.SetProperty("NumberofTLBexportsPersec", (value))
}

// GetNumberofTLBexportsPersec gets the value of NumberofTLBexportsPersec for the instance
func (instance *Win32_PerfFormattedData_NETFramework_NETCLRInterop) GetPropertyNumberofTLBexportsPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("NumberofTLBexportsPersec")
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

// SetNumberofTLBimportsPersec sets the value of NumberofTLBimportsPersec for the instance
func (instance *Win32_PerfFormattedData_NETFramework_NETCLRInterop) SetPropertyNumberofTLBimportsPersec(value uint32) (err error) {
	return instance.SetProperty("NumberofTLBimportsPersec", (value))
}

// GetNumberofTLBimportsPersec gets the value of NumberofTLBimportsPersec for the instance
func (instance *Win32_PerfFormattedData_NETFramework_NETCLRInterop) GetPropertyNumberofTLBimportsPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("NumberofTLBimportsPersec")
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
