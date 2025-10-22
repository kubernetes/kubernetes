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

// CIM_ChassisInRack struct
type CIM_ChassisInRack struct {
	*CIM_Container

	//
	BottomU uint16
}

func NewCIM_ChassisInRackEx1(instance *cim.WmiInstance) (newInstance *CIM_ChassisInRack, err error) {
	tmp, err := NewCIM_ContainerEx1(instance)

	if err != nil {
		return
	}
	newInstance = &CIM_ChassisInRack{
		CIM_Container: tmp,
	}
	return
}

func NewCIM_ChassisInRackEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *CIM_ChassisInRack, err error) {
	tmp, err := NewCIM_ContainerEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &CIM_ChassisInRack{
		CIM_Container: tmp,
	}
	return
}

// SetBottomU sets the value of BottomU for the instance
func (instance *CIM_ChassisInRack) SetPropertyBottomU(value uint16) (err error) {
	return instance.SetProperty("BottomU", (value))
}

// GetBottomU gets the value of BottomU for the instance
func (instance *CIM_ChassisInRack) GetPropertyBottomU() (value uint16, err error) {
	retValue, err := instance.GetProperty("BottomU")
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
