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

// Win32_ServiceControl struct
type Win32_ServiceControl struct {
	*Win32_MSIResource

	//
	Arguments string

	//
	Event string

	//
	ID string

	//
	Name string

	//
	ProductCode string

	//
	Wait uint16
}

func NewWin32_ServiceControlEx1(instance *cim.WmiInstance) (newInstance *Win32_ServiceControl, err error) {
	tmp, err := NewWin32_MSIResourceEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_ServiceControl{
		Win32_MSIResource: tmp,
	}
	return
}

func NewWin32_ServiceControlEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_ServiceControl, err error) {
	tmp, err := NewWin32_MSIResourceEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_ServiceControl{
		Win32_MSIResource: tmp,
	}
	return
}

// SetArguments sets the value of Arguments for the instance
func (instance *Win32_ServiceControl) SetPropertyArguments(value string) (err error) {
	return instance.SetProperty("Arguments", (value))
}

// GetArguments gets the value of Arguments for the instance
func (instance *Win32_ServiceControl) GetPropertyArguments() (value string, err error) {
	retValue, err := instance.GetProperty("Arguments")
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

// SetEvent sets the value of Event for the instance
func (instance *Win32_ServiceControl) SetPropertyEvent(value string) (err error) {
	return instance.SetProperty("Event", (value))
}

// GetEvent gets the value of Event for the instance
func (instance *Win32_ServiceControl) GetPropertyEvent() (value string, err error) {
	retValue, err := instance.GetProperty("Event")
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

// SetID sets the value of ID for the instance
func (instance *Win32_ServiceControl) SetPropertyID(value string) (err error) {
	return instance.SetProperty("ID", (value))
}

// GetID gets the value of ID for the instance
func (instance *Win32_ServiceControl) GetPropertyID() (value string, err error) {
	retValue, err := instance.GetProperty("ID")
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

// SetName sets the value of Name for the instance
func (instance *Win32_ServiceControl) SetPropertyName(value string) (err error) {
	return instance.SetProperty("Name", (value))
}

// GetName gets the value of Name for the instance
func (instance *Win32_ServiceControl) GetPropertyName() (value string, err error) {
	retValue, err := instance.GetProperty("Name")
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

// SetProductCode sets the value of ProductCode for the instance
func (instance *Win32_ServiceControl) SetPropertyProductCode(value string) (err error) {
	return instance.SetProperty("ProductCode", (value))
}

// GetProductCode gets the value of ProductCode for the instance
func (instance *Win32_ServiceControl) GetPropertyProductCode() (value string, err error) {
	retValue, err := instance.GetProperty("ProductCode")
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

// SetWait sets the value of Wait for the instance
func (instance *Win32_ServiceControl) SetPropertyWait(value uint16) (err error) {
	return instance.SetProperty("Wait", (value))
}

// GetWait gets the value of Wait for the instance
func (instance *Win32_ServiceControl) GetPropertyWait() (value uint16, err error) {
	retValue, err := instance.GetProperty("Wait")
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
