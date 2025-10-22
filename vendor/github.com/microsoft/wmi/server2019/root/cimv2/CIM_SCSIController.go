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

// CIM_SCSIController struct
type CIM_SCSIController struct {
	*CIM_Controller

	//
	ControllerTimeouts uint32

	//
	MaxDataWidth uint32

	//
	MaxTransferRate uint64

	//
	ProtectionManagement uint16
}

func NewCIM_SCSIControllerEx1(instance *cim.WmiInstance) (newInstance *CIM_SCSIController, err error) {
	tmp, err := NewCIM_ControllerEx1(instance)

	if err != nil {
		return
	}
	newInstance = &CIM_SCSIController{
		CIM_Controller: tmp,
	}
	return
}

func NewCIM_SCSIControllerEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *CIM_SCSIController, err error) {
	tmp, err := NewCIM_ControllerEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &CIM_SCSIController{
		CIM_Controller: tmp,
	}
	return
}

// SetControllerTimeouts sets the value of ControllerTimeouts for the instance
func (instance *CIM_SCSIController) SetPropertyControllerTimeouts(value uint32) (err error) {
	return instance.SetProperty("ControllerTimeouts", (value))
}

// GetControllerTimeouts gets the value of ControllerTimeouts for the instance
func (instance *CIM_SCSIController) GetPropertyControllerTimeouts() (value uint32, err error) {
	retValue, err := instance.GetProperty("ControllerTimeouts")
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

// SetMaxDataWidth sets the value of MaxDataWidth for the instance
func (instance *CIM_SCSIController) SetPropertyMaxDataWidth(value uint32) (err error) {
	return instance.SetProperty("MaxDataWidth", (value))
}

// GetMaxDataWidth gets the value of MaxDataWidth for the instance
func (instance *CIM_SCSIController) GetPropertyMaxDataWidth() (value uint32, err error) {
	retValue, err := instance.GetProperty("MaxDataWidth")
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

// SetMaxTransferRate sets the value of MaxTransferRate for the instance
func (instance *CIM_SCSIController) SetPropertyMaxTransferRate(value uint64) (err error) {
	return instance.SetProperty("MaxTransferRate", (value))
}

// GetMaxTransferRate gets the value of MaxTransferRate for the instance
func (instance *CIM_SCSIController) GetPropertyMaxTransferRate() (value uint64, err error) {
	retValue, err := instance.GetProperty("MaxTransferRate")
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

// SetProtectionManagement sets the value of ProtectionManagement for the instance
func (instance *CIM_SCSIController) SetPropertyProtectionManagement(value uint16) (err error) {
	return instance.SetProperty("ProtectionManagement", (value))
}

// GetProtectionManagement gets the value of ProtectionManagement for the instance
func (instance *CIM_SCSIController) GetPropertyProtectionManagement() (value uint16, err error) {
	retValue, err := instance.GetProperty("ProtectionManagement")
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
