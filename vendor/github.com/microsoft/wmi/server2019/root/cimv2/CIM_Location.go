// Copyright 2019 (c) Microsoft Corporation.
// Licensed under the MIT license.

//
// Author:
//      Auto Generated on 9/18/2020 using wmigen
//      Source root.CIMV2
//////////////////////////////////////////////
package cimv2

import (
	"github.com/microsoft/wmi/pkg/base/instance"
	"github.com/microsoft/wmi/pkg/base/query"
	"github.com/microsoft/wmi/pkg/errors"
	cim "github.com/microsoft/wmi/pkg/wmiinstance"
	"reflect"
)

// CIM_Location struct
type CIM_Location struct {
	*cim.WmiInstance

	//
	Address string

	//
	Name string

	//
	PhysicalPosition string
}

func NewCIM_LocationEx1(instance *cim.WmiInstance) (newInstance *CIM_Location, err error) {
	tmp, err := instance, nil

	if err != nil {
		return
	}
	newInstance = &CIM_Location{
		WmiInstance: tmp,
	}
	return
}

func NewCIM_LocationEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *CIM_Location, err error) {
	tmp, err := instance.GetWmiInstance(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &CIM_Location{
		WmiInstance: tmp,
	}
	return
}

// SetAddress sets the value of Address for the instance
func (instance *CIM_Location) SetPropertyAddress(value string) (err error) {
	return instance.SetProperty("Address", (value))
}

// GetAddress gets the value of Address for the instance
func (instance *CIM_Location) GetPropertyAddress() (value string, err error) {
	retValue, err := instance.GetProperty("Address")
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
func (instance *CIM_Location) SetPropertyName(value string) (err error) {
	return instance.SetProperty("Name", (value))
}

// GetName gets the value of Name for the instance
func (instance *CIM_Location) GetPropertyName() (value string, err error) {
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

// SetPhysicalPosition sets the value of PhysicalPosition for the instance
func (instance *CIM_Location) SetPropertyPhysicalPosition(value string) (err error) {
	return instance.SetProperty("PhysicalPosition", (value))
}

// GetPhysicalPosition gets the value of PhysicalPosition for the instance
func (instance *CIM_Location) GetPropertyPhysicalPosition() (value string, err error) {
	retValue, err := instance.GetProperty("PhysicalPosition")
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
