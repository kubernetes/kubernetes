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

// CIM_SerialInterface struct
type CIM_SerialInterface struct {
	*CIM_ControlledBy

	//
	FlowControlInfo uint16

	//
	NumberOfStopBits uint16

	//
	ParityInfo uint16
}

func NewCIM_SerialInterfaceEx1(instance *cim.WmiInstance) (newInstance *CIM_SerialInterface, err error) {
	tmp, err := NewCIM_ControlledByEx1(instance)

	if err != nil {
		return
	}
	newInstance = &CIM_SerialInterface{
		CIM_ControlledBy: tmp,
	}
	return
}

func NewCIM_SerialInterfaceEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *CIM_SerialInterface, err error) {
	tmp, err := NewCIM_ControlledByEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &CIM_SerialInterface{
		CIM_ControlledBy: tmp,
	}
	return
}

// SetFlowControlInfo sets the value of FlowControlInfo for the instance
func (instance *CIM_SerialInterface) SetPropertyFlowControlInfo(value uint16) (err error) {
	return instance.SetProperty("FlowControlInfo", (value))
}

// GetFlowControlInfo gets the value of FlowControlInfo for the instance
func (instance *CIM_SerialInterface) GetPropertyFlowControlInfo() (value uint16, err error) {
	retValue, err := instance.GetProperty("FlowControlInfo")
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

// SetNumberOfStopBits sets the value of NumberOfStopBits for the instance
func (instance *CIM_SerialInterface) SetPropertyNumberOfStopBits(value uint16) (err error) {
	return instance.SetProperty("NumberOfStopBits", (value))
}

// GetNumberOfStopBits gets the value of NumberOfStopBits for the instance
func (instance *CIM_SerialInterface) GetPropertyNumberOfStopBits() (value uint16, err error) {
	retValue, err := instance.GetProperty("NumberOfStopBits")
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

// SetParityInfo sets the value of ParityInfo for the instance
func (instance *CIM_SerialInterface) SetPropertyParityInfo(value uint16) (err error) {
	return instance.SetProperty("ParityInfo", (value))
}

// GetParityInfo gets the value of ParityInfo for the instance
func (instance *CIM_SerialInterface) GetPropertyParityInfo() (value uint16, err error) {
	retValue, err := instance.GetProperty("ParityInfo")
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
