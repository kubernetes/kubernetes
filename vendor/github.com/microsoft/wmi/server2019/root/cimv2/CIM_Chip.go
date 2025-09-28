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

// CIM_Chip struct
type CIM_Chip struct {
	*CIM_PhysicalComponent

	//
	FormFactor uint16
}

func NewCIM_ChipEx1(instance *cim.WmiInstance) (newInstance *CIM_Chip, err error) {
	tmp, err := NewCIM_PhysicalComponentEx1(instance)

	if err != nil {
		return
	}
	newInstance = &CIM_Chip{
		CIM_PhysicalComponent: tmp,
	}
	return
}

func NewCIM_ChipEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *CIM_Chip, err error) {
	tmp, err := NewCIM_PhysicalComponentEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &CIM_Chip{
		CIM_PhysicalComponent: tmp,
	}
	return
}

// SetFormFactor sets the value of FormFactor for the instance
func (instance *CIM_Chip) SetPropertyFormFactor(value uint16) (err error) {
	return instance.SetProperty("FormFactor", (value))
}

// GetFormFactor gets the value of FormFactor for the instance
func (instance *CIM_Chip) GetPropertyFormFactor() (value uint16, err error) {
	retValue, err := instance.GetProperty("FormFactor")
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
