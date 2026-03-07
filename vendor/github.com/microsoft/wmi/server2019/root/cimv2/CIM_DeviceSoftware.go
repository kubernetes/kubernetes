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

// CIM_DeviceSoftware struct
type CIM_DeviceSoftware struct {
	*CIM_Dependency

	//
	Purpose uint16

	//
	PurposeDescription string
}

func NewCIM_DeviceSoftwareEx1(instance *cim.WmiInstance) (newInstance *CIM_DeviceSoftware, err error) {
	tmp, err := NewCIM_DependencyEx1(instance)

	if err != nil {
		return
	}
	newInstance = &CIM_DeviceSoftware{
		CIM_Dependency: tmp,
	}
	return
}

func NewCIM_DeviceSoftwareEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *CIM_DeviceSoftware, err error) {
	tmp, err := NewCIM_DependencyEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &CIM_DeviceSoftware{
		CIM_Dependency: tmp,
	}
	return
}

// SetPurpose sets the value of Purpose for the instance
func (instance *CIM_DeviceSoftware) SetPropertyPurpose(value uint16) (err error) {
	return instance.SetProperty("Purpose", (value))
}

// GetPurpose gets the value of Purpose for the instance
func (instance *CIM_DeviceSoftware) GetPropertyPurpose() (value uint16, err error) {
	retValue, err := instance.GetProperty("Purpose")
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

// SetPurposeDescription sets the value of PurposeDescription for the instance
func (instance *CIM_DeviceSoftware) SetPropertyPurposeDescription(value string) (err error) {
	return instance.SetProperty("PurposeDescription", (value))
}

// GetPurposeDescription gets the value of PurposeDescription for the instance
func (instance *CIM_DeviceSoftware) GetPropertyPurposeDescription() (value string, err error) {
	retValue, err := instance.GetProperty("PurposeDescription")
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
