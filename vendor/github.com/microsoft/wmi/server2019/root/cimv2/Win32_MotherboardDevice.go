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

// Win32_MotherboardDevice struct
type Win32_MotherboardDevice struct {
	*CIM_LogicalDevice

	//
	PrimaryBusType string

	//
	RevisionNumber string

	//
	SecondaryBusType string
}

func NewWin32_MotherboardDeviceEx1(instance *cim.WmiInstance) (newInstance *Win32_MotherboardDevice, err error) {
	tmp, err := NewCIM_LogicalDeviceEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_MotherboardDevice{
		CIM_LogicalDevice: tmp,
	}
	return
}

func NewWin32_MotherboardDeviceEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_MotherboardDevice, err error) {
	tmp, err := NewCIM_LogicalDeviceEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_MotherboardDevice{
		CIM_LogicalDevice: tmp,
	}
	return
}

// SetPrimaryBusType sets the value of PrimaryBusType for the instance
func (instance *Win32_MotherboardDevice) SetPropertyPrimaryBusType(value string) (err error) {
	return instance.SetProperty("PrimaryBusType", (value))
}

// GetPrimaryBusType gets the value of PrimaryBusType for the instance
func (instance *Win32_MotherboardDevice) GetPropertyPrimaryBusType() (value string, err error) {
	retValue, err := instance.GetProperty("PrimaryBusType")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(string)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " string is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = string(valuetmp)

	return
}

// SetRevisionNumber sets the value of RevisionNumber for the instance
func (instance *Win32_MotherboardDevice) SetPropertyRevisionNumber(value string) (err error) {
	return instance.SetProperty("RevisionNumber", (value))
}

// GetRevisionNumber gets the value of RevisionNumber for the instance
func (instance *Win32_MotherboardDevice) GetPropertyRevisionNumber() (value string, err error) {
	retValue, err := instance.GetProperty("RevisionNumber")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(string)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " string is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = string(valuetmp)

	return
}

// SetSecondaryBusType sets the value of SecondaryBusType for the instance
func (instance *Win32_MotherboardDevice) SetPropertySecondaryBusType(value string) (err error) {
	return instance.SetProperty("SecondaryBusType", (value))
}

// GetSecondaryBusType gets the value of SecondaryBusType for the instance
func (instance *Win32_MotherboardDevice) GetPropertySecondaryBusType() (value string, err error) {
	retValue, err := instance.GetProperty("SecondaryBusType")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(string)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " string is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = string(valuetmp)

	return
}
