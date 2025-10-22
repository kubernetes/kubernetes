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

// CIM_VideoBIOSElement struct
type CIM_VideoBIOSElement struct {
	*CIM_SoftwareElement

	//
	IsShadowed bool
}

func NewCIM_VideoBIOSElementEx1(instance *cim.WmiInstance) (newInstance *CIM_VideoBIOSElement, err error) {
	tmp, err := NewCIM_SoftwareElementEx1(instance)

	if err != nil {
		return
	}
	newInstance = &CIM_VideoBIOSElement{
		CIM_SoftwareElement: tmp,
	}
	return
}

func NewCIM_VideoBIOSElementEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *CIM_VideoBIOSElement, err error) {
	tmp, err := NewCIM_SoftwareElementEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &CIM_VideoBIOSElement{
		CIM_SoftwareElement: tmp,
	}
	return
}

// SetIsShadowed sets the value of IsShadowed for the instance
func (instance *CIM_VideoBIOSElement) SetPropertyIsShadowed(value bool) (err error) {
	return instance.SetProperty("IsShadowed", (value))
}

// GetIsShadowed gets the value of IsShadowed for the instance
func (instance *CIM_VideoBIOSElement) GetPropertyIsShadowed() (value bool, err error) {
	retValue, err := instance.GetProperty("IsShadowed")
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
