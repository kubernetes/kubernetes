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

// CIM_BinarySensor struct
type CIM_BinarySensor struct {
	*CIM_Sensor

	//
	CurrentReading bool

	//
	ExpectedReading bool

	//
	InterpretationOfFalse string

	//
	InterpretationOfTrue string
}

func NewCIM_BinarySensorEx1(instance *cim.WmiInstance) (newInstance *CIM_BinarySensor, err error) {
	tmp, err := NewCIM_SensorEx1(instance)

	if err != nil {
		return
	}
	newInstance = &CIM_BinarySensor{
		CIM_Sensor: tmp,
	}
	return
}

func NewCIM_BinarySensorEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *CIM_BinarySensor, err error) {
	tmp, err := NewCIM_SensorEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &CIM_BinarySensor{
		CIM_Sensor: tmp,
	}
	return
}

// SetCurrentReading sets the value of CurrentReading for the instance
func (instance *CIM_BinarySensor) SetPropertyCurrentReading(value bool) (err error) {
	return instance.SetProperty("CurrentReading", (value))
}

// GetCurrentReading gets the value of CurrentReading for the instance
func (instance *CIM_BinarySensor) GetPropertyCurrentReading() (value bool, err error) {
	retValue, err := instance.GetProperty("CurrentReading")
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

// SetExpectedReading sets the value of ExpectedReading for the instance
func (instance *CIM_BinarySensor) SetPropertyExpectedReading(value bool) (err error) {
	return instance.SetProperty("ExpectedReading", (value))
}

// GetExpectedReading gets the value of ExpectedReading for the instance
func (instance *CIM_BinarySensor) GetPropertyExpectedReading() (value bool, err error) {
	retValue, err := instance.GetProperty("ExpectedReading")
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

// SetInterpretationOfFalse sets the value of InterpretationOfFalse for the instance
func (instance *CIM_BinarySensor) SetPropertyInterpretationOfFalse(value string) (err error) {
	return instance.SetProperty("InterpretationOfFalse", (value))
}

// GetInterpretationOfFalse gets the value of InterpretationOfFalse for the instance
func (instance *CIM_BinarySensor) GetPropertyInterpretationOfFalse() (value string, err error) {
	retValue, err := instance.GetProperty("InterpretationOfFalse")
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

// SetInterpretationOfTrue sets the value of InterpretationOfTrue for the instance
func (instance *CIM_BinarySensor) SetPropertyInterpretationOfTrue(value string) (err error) {
	return instance.SetProperty("InterpretationOfTrue", (value))
}

// GetInterpretationOfTrue gets the value of InterpretationOfTrue for the instance
func (instance *CIM_BinarySensor) GetPropertyInterpretationOfTrue() (value string, err error) {
	retValue, err := instance.GetProperty("InterpretationOfTrue")
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
