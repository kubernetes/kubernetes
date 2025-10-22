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

// CIM_DiscreteSensor struct
type CIM_DiscreteSensor struct {
	*CIM_Sensor

	//
	AcceptableValues []string

	//
	CurrentReading string

	//
	PossibleValues []string
}

func NewCIM_DiscreteSensorEx1(instance *cim.WmiInstance) (newInstance *CIM_DiscreteSensor, err error) {
	tmp, err := NewCIM_SensorEx1(instance)

	if err != nil {
		return
	}
	newInstance = &CIM_DiscreteSensor{
		CIM_Sensor: tmp,
	}
	return
}

func NewCIM_DiscreteSensorEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *CIM_DiscreteSensor, err error) {
	tmp, err := NewCIM_SensorEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &CIM_DiscreteSensor{
		CIM_Sensor: tmp,
	}
	return
}

// SetAcceptableValues sets the value of AcceptableValues for the instance
func (instance *CIM_DiscreteSensor) SetPropertyAcceptableValues(value []string) (err error) {
	return instance.SetProperty("AcceptableValues", (value))
}

// GetAcceptableValues gets the value of AcceptableValues for the instance
func (instance *CIM_DiscreteSensor) GetPropertyAcceptableValues() (value []string, err error) {
	retValue, err := instance.GetProperty("AcceptableValues")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	for _, interfaceValue := range retValue.([]interface{}) {
		valuetmp, ok := interfaceValue.(string)
		if !ok {
			err = errors.Wrapf(errors.InvalidType, " string is Invalid. Expected %s", reflect.TypeOf(interfaceValue))
			return
		}
		value = append(value, string(valuetmp))
	}

	return
}

// SetCurrentReading sets the value of CurrentReading for the instance
func (instance *CIM_DiscreteSensor) SetPropertyCurrentReading(value string) (err error) {
	return instance.SetProperty("CurrentReading", (value))
}

// GetCurrentReading gets the value of CurrentReading for the instance
func (instance *CIM_DiscreteSensor) GetPropertyCurrentReading() (value string, err error) {
	retValue, err := instance.GetProperty("CurrentReading")
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

// SetPossibleValues sets the value of PossibleValues for the instance
func (instance *CIM_DiscreteSensor) SetPropertyPossibleValues(value []string) (err error) {
	return instance.SetProperty("PossibleValues", (value))
}

// GetPossibleValues gets the value of PossibleValues for the instance
func (instance *CIM_DiscreteSensor) GetPropertyPossibleValues() (value []string, err error) {
	retValue, err := instance.GetProperty("PossibleValues")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	for _, interfaceValue := range retValue.([]interface{}) {
		valuetmp, ok := interfaceValue.(string)
		if !ok {
			err = errors.Wrapf(errors.InvalidType, " string is Invalid. Expected %s", reflect.TypeOf(interfaceValue))
			return
		}
		value = append(value, string(valuetmp))
	}

	return
}
