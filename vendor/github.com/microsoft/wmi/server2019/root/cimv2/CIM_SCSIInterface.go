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

// CIM_SCSIInterface struct
type CIM_SCSIInterface struct {
	*CIM_ControlledBy

	//
	SCSIRetries uint32

	//
	SCSITimeouts uint32
}

func NewCIM_SCSIInterfaceEx1(instance *cim.WmiInstance) (newInstance *CIM_SCSIInterface, err error) {
	tmp, err := NewCIM_ControlledByEx1(instance)

	if err != nil {
		return
	}
	newInstance = &CIM_SCSIInterface{
		CIM_ControlledBy: tmp,
	}
	return
}

func NewCIM_SCSIInterfaceEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *CIM_SCSIInterface, err error) {
	tmp, err := NewCIM_ControlledByEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &CIM_SCSIInterface{
		CIM_ControlledBy: tmp,
	}
	return
}

// SetSCSIRetries sets the value of SCSIRetries for the instance
func (instance *CIM_SCSIInterface) SetPropertySCSIRetries(value uint32) (err error) {
	return instance.SetProperty("SCSIRetries", (value))
}

// GetSCSIRetries gets the value of SCSIRetries for the instance
func (instance *CIM_SCSIInterface) GetPropertySCSIRetries() (value uint32, err error) {
	retValue, err := instance.GetProperty("SCSIRetries")
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

// SetSCSITimeouts sets the value of SCSITimeouts for the instance
func (instance *CIM_SCSIInterface) SetPropertySCSITimeouts(value uint32) (err error) {
	return instance.SetProperty("SCSITimeouts", (value))
}

// GetSCSITimeouts gets the value of SCSITimeouts for the instance
func (instance *CIM_SCSIInterface) GetPropertySCSITimeouts() (value uint32, err error) {
	retValue, err := instance.GetProperty("SCSITimeouts")
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
