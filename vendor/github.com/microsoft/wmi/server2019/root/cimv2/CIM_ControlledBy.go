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

// CIM_ControlledBy struct
type CIM_ControlledBy struct {
	*CIM_DeviceConnection

	//
	AccessState uint16

	//
	NumberOfHardResets uint32

	//
	NumberOfSoftResets uint32
}

func NewCIM_ControlledByEx1(instance *cim.WmiInstance) (newInstance *CIM_ControlledBy, err error) {
	tmp, err := NewCIM_DeviceConnectionEx1(instance)

	if err != nil {
		return
	}
	newInstance = &CIM_ControlledBy{
		CIM_DeviceConnection: tmp,
	}
	return
}

func NewCIM_ControlledByEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *CIM_ControlledBy, err error) {
	tmp, err := NewCIM_DeviceConnectionEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &CIM_ControlledBy{
		CIM_DeviceConnection: tmp,
	}
	return
}

// SetAccessState sets the value of AccessState for the instance
func (instance *CIM_ControlledBy) SetPropertyAccessState(value uint16) (err error) {
	return instance.SetProperty("AccessState", (value))
}

// GetAccessState gets the value of AccessState for the instance
func (instance *CIM_ControlledBy) GetPropertyAccessState() (value uint16, err error) {
	retValue, err := instance.GetProperty("AccessState")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint16)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint16 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint16(valuetmp)

	return
}

// SetNumberOfHardResets sets the value of NumberOfHardResets for the instance
func (instance *CIM_ControlledBy) SetPropertyNumberOfHardResets(value uint32) (err error) {
	return instance.SetProperty("NumberOfHardResets", (value))
}

// GetNumberOfHardResets gets the value of NumberOfHardResets for the instance
func (instance *CIM_ControlledBy) GetPropertyNumberOfHardResets() (value uint32, err error) {
	retValue, err := instance.GetProperty("NumberOfHardResets")
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

// SetNumberOfSoftResets sets the value of NumberOfSoftResets for the instance
func (instance *CIM_ControlledBy) SetPropertyNumberOfSoftResets(value uint32) (err error) {
	return instance.SetProperty("NumberOfSoftResets", (value))
}

// GetNumberOfSoftResets gets the value of NumberOfSoftResets for the instance
func (instance *CIM_ControlledBy) GetPropertyNumberOfSoftResets() (value uint32, err error) {
	retValue, err := instance.GetProperty("NumberOfSoftResets")
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
