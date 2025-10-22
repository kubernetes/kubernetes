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

// CIM_Chassis struct
type CIM_Chassis struct {
	*CIM_PhysicalFrame

	//
	ChassisTypes []uint16

	//
	CurrentRequiredOrProduced int16

	//
	HeatGeneration uint16

	//
	NumberOfPowerCords uint16

	//
	TypeDescriptions []string
}

func NewCIM_ChassisEx1(instance *cim.WmiInstance) (newInstance *CIM_Chassis, err error) {
	tmp, err := NewCIM_PhysicalFrameEx1(instance)

	if err != nil {
		return
	}
	newInstance = &CIM_Chassis{
		CIM_PhysicalFrame: tmp,
	}
	return
}

func NewCIM_ChassisEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *CIM_Chassis, err error) {
	tmp, err := NewCIM_PhysicalFrameEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &CIM_Chassis{
		CIM_PhysicalFrame: tmp,
	}
	return
}

// SetChassisTypes sets the value of ChassisTypes for the instance
func (instance *CIM_Chassis) SetPropertyChassisTypes(value []uint16) (err error) {
	return instance.SetProperty("ChassisTypes", (value))
}

// GetChassisTypes gets the value of ChassisTypes for the instance
func (instance *CIM_Chassis) GetPropertyChassisTypes() (value []uint16, err error) {
	retValue, err := instance.GetProperty("ChassisTypes")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	for _, interfaceValue := range retValue.([]interface{}) {
		valuetmp, ok := interfaceValue.(uint16)
		if !ok {
			err = errors.Wrapf(errors.InvalidType, " uint16 is Invalid. Expected %s", reflect.TypeOf(interfaceValue))
			return
		}
		value = append(value, uint16(valuetmp))
	}

	return
}

// SetCurrentRequiredOrProduced sets the value of CurrentRequiredOrProduced for the instance
func (instance *CIM_Chassis) SetPropertyCurrentRequiredOrProduced(value int16) (err error) {
	return instance.SetProperty("CurrentRequiredOrProduced", (value))
}

// GetCurrentRequiredOrProduced gets the value of CurrentRequiredOrProduced for the instance
func (instance *CIM_Chassis) GetPropertyCurrentRequiredOrProduced() (value int16, err error) {
	retValue, err := instance.GetProperty("CurrentRequiredOrProduced")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(int16)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " int16 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = int16(valuetmp)

	return
}

// SetHeatGeneration sets the value of HeatGeneration for the instance
func (instance *CIM_Chassis) SetPropertyHeatGeneration(value uint16) (err error) {
	return instance.SetProperty("HeatGeneration", (value))
}

// GetHeatGeneration gets the value of HeatGeneration for the instance
func (instance *CIM_Chassis) GetPropertyHeatGeneration() (value uint16, err error) {
	retValue, err := instance.GetProperty("HeatGeneration")
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

// SetNumberOfPowerCords sets the value of NumberOfPowerCords for the instance
func (instance *CIM_Chassis) SetPropertyNumberOfPowerCords(value uint16) (err error) {
	return instance.SetProperty("NumberOfPowerCords", (value))
}

// GetNumberOfPowerCords gets the value of NumberOfPowerCords for the instance
func (instance *CIM_Chassis) GetPropertyNumberOfPowerCords() (value uint16, err error) {
	retValue, err := instance.GetProperty("NumberOfPowerCords")
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

// SetTypeDescriptions sets the value of TypeDescriptions for the instance
func (instance *CIM_Chassis) SetPropertyTypeDescriptions(value []string) (err error) {
	return instance.SetProperty("TypeDescriptions", (value))
}

// GetTypeDescriptions gets the value of TypeDescriptions for the instance
func (instance *CIM_Chassis) GetPropertyTypeDescriptions() (value []string, err error) {
	retValue, err := instance.GetProperty("TypeDescriptions")
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
