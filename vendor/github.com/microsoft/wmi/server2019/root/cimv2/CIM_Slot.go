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

// CIM_Slot struct
type CIM_Slot struct {
	*CIM_PhysicalConnector

	//
	HeightAllowed float32

	//
	LengthAllowed float32

	//
	MaxDataWidth uint16

	//
	Number uint16

	//
	PurposeDescription string

	//
	SpecialPurpose bool

	//
	SupportsHotPlug bool

	//
	ThermalRating uint32

	//
	VccMixedVoltageSupport []uint16

	//
	VppMixedVoltageSupport []uint16
}

func NewCIM_SlotEx1(instance *cim.WmiInstance) (newInstance *CIM_Slot, err error) {
	tmp, err := NewCIM_PhysicalConnectorEx1(instance)

	if err != nil {
		return
	}
	newInstance = &CIM_Slot{
		CIM_PhysicalConnector: tmp,
	}
	return
}

func NewCIM_SlotEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *CIM_Slot, err error) {
	tmp, err := NewCIM_PhysicalConnectorEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &CIM_Slot{
		CIM_PhysicalConnector: tmp,
	}
	return
}

// SetHeightAllowed sets the value of HeightAllowed for the instance
func (instance *CIM_Slot) SetPropertyHeightAllowed(value float32) (err error) {
	return instance.SetProperty("HeightAllowed", (value))
}

// GetHeightAllowed gets the value of HeightAllowed for the instance
func (instance *CIM_Slot) GetPropertyHeightAllowed() (value float32, err error) {
	retValue, err := instance.GetProperty("HeightAllowed")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(float32)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " float32 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = float32(valuetmp)

	return
}

// SetLengthAllowed sets the value of LengthAllowed for the instance
func (instance *CIM_Slot) SetPropertyLengthAllowed(value float32) (err error) {
	return instance.SetProperty("LengthAllowed", (value))
}

// GetLengthAllowed gets the value of LengthAllowed for the instance
func (instance *CIM_Slot) GetPropertyLengthAllowed() (value float32, err error) {
	retValue, err := instance.GetProperty("LengthAllowed")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(float32)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " float32 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = float32(valuetmp)

	return
}

// SetMaxDataWidth sets the value of MaxDataWidth for the instance
func (instance *CIM_Slot) SetPropertyMaxDataWidth(value uint16) (err error) {
	return instance.SetProperty("MaxDataWidth", (value))
}

// GetMaxDataWidth gets the value of MaxDataWidth for the instance
func (instance *CIM_Slot) GetPropertyMaxDataWidth() (value uint16, err error) {
	retValue, err := instance.GetProperty("MaxDataWidth")
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

// SetNumber sets the value of Number for the instance
func (instance *CIM_Slot) SetPropertyNumber(value uint16) (err error) {
	return instance.SetProperty("Number", (value))
}

// GetNumber gets the value of Number for the instance
func (instance *CIM_Slot) GetPropertyNumber() (value uint16, err error) {
	retValue, err := instance.GetProperty("Number")
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
func (instance *CIM_Slot) SetPropertyPurposeDescription(value string) (err error) {
	return instance.SetProperty("PurposeDescription", (value))
}

// GetPurposeDescription gets the value of PurposeDescription for the instance
func (instance *CIM_Slot) GetPropertyPurposeDescription() (value string, err error) {
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

// SetSpecialPurpose sets the value of SpecialPurpose for the instance
func (instance *CIM_Slot) SetPropertySpecialPurpose(value bool) (err error) {
	return instance.SetProperty("SpecialPurpose", (value))
}

// GetSpecialPurpose gets the value of SpecialPurpose for the instance
func (instance *CIM_Slot) GetPropertySpecialPurpose() (value bool, err error) {
	retValue, err := instance.GetProperty("SpecialPurpose")
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

// SetSupportsHotPlug sets the value of SupportsHotPlug for the instance
func (instance *CIM_Slot) SetPropertySupportsHotPlug(value bool) (err error) {
	return instance.SetProperty("SupportsHotPlug", (value))
}

// GetSupportsHotPlug gets the value of SupportsHotPlug for the instance
func (instance *CIM_Slot) GetPropertySupportsHotPlug() (value bool, err error) {
	retValue, err := instance.GetProperty("SupportsHotPlug")
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

// SetThermalRating sets the value of ThermalRating for the instance
func (instance *CIM_Slot) SetPropertyThermalRating(value uint32) (err error) {
	return instance.SetProperty("ThermalRating", (value))
}

// GetThermalRating gets the value of ThermalRating for the instance
func (instance *CIM_Slot) GetPropertyThermalRating() (value uint32, err error) {
	retValue, err := instance.GetProperty("ThermalRating")
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

// SetVccMixedVoltageSupport sets the value of VccMixedVoltageSupport for the instance
func (instance *CIM_Slot) SetPropertyVccMixedVoltageSupport(value []uint16) (err error) {
	return instance.SetProperty("VccMixedVoltageSupport", (value))
}

// GetVccMixedVoltageSupport gets the value of VccMixedVoltageSupport for the instance
func (instance *CIM_Slot) GetPropertyVccMixedVoltageSupport() (value []uint16, err error) {
	retValue, err := instance.GetProperty("VccMixedVoltageSupport")
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

// SetVppMixedVoltageSupport sets the value of VppMixedVoltageSupport for the instance
func (instance *CIM_Slot) SetPropertyVppMixedVoltageSupport(value []uint16) (err error) {
	return instance.SetProperty("VppMixedVoltageSupport", (value))
}

// GetVppMixedVoltageSupport gets the value of VppMixedVoltageSupport for the instance
func (instance *CIM_Slot) GetPropertyVppMixedVoltageSupport() (value []uint16, err error) {
	retValue, err := instance.GetProperty("VppMixedVoltageSupport")
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
