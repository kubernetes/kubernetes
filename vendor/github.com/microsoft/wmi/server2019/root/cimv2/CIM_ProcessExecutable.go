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

// CIM_ProcessExecutable struct
type CIM_ProcessExecutable struct {
	*CIM_Dependency

	//
	BaseAddress uint64

	//
	GlobalProcessCount uint32

	//
	ModuleInstance uint32

	//
	ProcessCount uint32
}

func NewCIM_ProcessExecutableEx1(instance *cim.WmiInstance) (newInstance *CIM_ProcessExecutable, err error) {
	tmp, err := NewCIM_DependencyEx1(instance)

	if err != nil {
		return
	}
	newInstance = &CIM_ProcessExecutable{
		CIM_Dependency: tmp,
	}
	return
}

func NewCIM_ProcessExecutableEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *CIM_ProcessExecutable, err error) {
	tmp, err := NewCIM_DependencyEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &CIM_ProcessExecutable{
		CIM_Dependency: tmp,
	}
	return
}

// SetBaseAddress sets the value of BaseAddress for the instance
func (instance *CIM_ProcessExecutable) SetPropertyBaseAddress(value uint64) (err error) {
	return instance.SetProperty("BaseAddress", (value))
}

// GetBaseAddress gets the value of BaseAddress for the instance
func (instance *CIM_ProcessExecutable) GetPropertyBaseAddress() (value uint64, err error) {
	retValue, err := instance.GetProperty("BaseAddress")
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

// SetGlobalProcessCount sets the value of GlobalProcessCount for the instance
func (instance *CIM_ProcessExecutable) SetPropertyGlobalProcessCount(value uint32) (err error) {
	return instance.SetProperty("GlobalProcessCount", (value))
}

// GetGlobalProcessCount gets the value of GlobalProcessCount for the instance
func (instance *CIM_ProcessExecutable) GetPropertyGlobalProcessCount() (value uint32, err error) {
	retValue, err := instance.GetProperty("GlobalProcessCount")
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

// SetModuleInstance sets the value of ModuleInstance for the instance
func (instance *CIM_ProcessExecutable) SetPropertyModuleInstance(value uint32) (err error) {
	return instance.SetProperty("ModuleInstance", (value))
}

// GetModuleInstance gets the value of ModuleInstance for the instance
func (instance *CIM_ProcessExecutable) GetPropertyModuleInstance() (value uint32, err error) {
	retValue, err := instance.GetProperty("ModuleInstance")
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

// SetProcessCount sets the value of ProcessCount for the instance
func (instance *CIM_ProcessExecutable) SetPropertyProcessCount(value uint32) (err error) {
	return instance.SetProperty("ProcessCount", (value))
}

// GetProcessCount gets the value of ProcessCount for the instance
func (instance *CIM_ProcessExecutable) GetPropertyProcessCount() (value uint32, err error) {
	retValue, err := instance.GetProperty("ProcessCount")
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
