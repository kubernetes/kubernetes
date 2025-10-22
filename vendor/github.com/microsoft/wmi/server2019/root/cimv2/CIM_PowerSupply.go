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

// CIM_PowerSupply struct
type CIM_PowerSupply struct {
	*CIM_LogicalDevice

	//
	ActiveInputVoltage uint16

	//
	IsSwitchingSupply bool

	//
	Range1InputFrequencyHigh uint32

	//
	Range1InputFrequencyLow uint32

	//
	Range1InputVoltageHigh uint32

	//
	Range1InputVoltageLow uint32

	//
	Range2InputFrequencyHigh uint32

	//
	Range2InputFrequencyLow uint32

	//
	Range2InputVoltageHigh uint32

	//
	Range2InputVoltageLow uint32

	//
	TotalOutputPower uint32

	//
	TypeOfRangeSwitching uint16
}

func NewCIM_PowerSupplyEx1(instance *cim.WmiInstance) (newInstance *CIM_PowerSupply, err error) {
	tmp, err := NewCIM_LogicalDeviceEx1(instance)

	if err != nil {
		return
	}
	newInstance = &CIM_PowerSupply{
		CIM_LogicalDevice: tmp,
	}
	return
}

func NewCIM_PowerSupplyEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *CIM_PowerSupply, err error) {
	tmp, err := NewCIM_LogicalDeviceEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &CIM_PowerSupply{
		CIM_LogicalDevice: tmp,
	}
	return
}

// SetActiveInputVoltage sets the value of ActiveInputVoltage for the instance
func (instance *CIM_PowerSupply) SetPropertyActiveInputVoltage(value uint16) (err error) {
	return instance.SetProperty("ActiveInputVoltage", (value))
}

// GetActiveInputVoltage gets the value of ActiveInputVoltage for the instance
func (instance *CIM_PowerSupply) GetPropertyActiveInputVoltage() (value uint16, err error) {
	retValue, err := instance.GetProperty("ActiveInputVoltage")
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

// SetIsSwitchingSupply sets the value of IsSwitchingSupply for the instance
func (instance *CIM_PowerSupply) SetPropertyIsSwitchingSupply(value bool) (err error) {
	return instance.SetProperty("IsSwitchingSupply", (value))
}

// GetIsSwitchingSupply gets the value of IsSwitchingSupply for the instance
func (instance *CIM_PowerSupply) GetPropertyIsSwitchingSupply() (value bool, err error) {
	retValue, err := instance.GetProperty("IsSwitchingSupply")
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

// SetRange1InputFrequencyHigh sets the value of Range1InputFrequencyHigh for the instance
func (instance *CIM_PowerSupply) SetPropertyRange1InputFrequencyHigh(value uint32) (err error) {
	return instance.SetProperty("Range1InputFrequencyHigh", (value))
}

// GetRange1InputFrequencyHigh gets the value of Range1InputFrequencyHigh for the instance
func (instance *CIM_PowerSupply) GetPropertyRange1InputFrequencyHigh() (value uint32, err error) {
	retValue, err := instance.GetProperty("Range1InputFrequencyHigh")
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

// SetRange1InputFrequencyLow sets the value of Range1InputFrequencyLow for the instance
func (instance *CIM_PowerSupply) SetPropertyRange1InputFrequencyLow(value uint32) (err error) {
	return instance.SetProperty("Range1InputFrequencyLow", (value))
}

// GetRange1InputFrequencyLow gets the value of Range1InputFrequencyLow for the instance
func (instance *CIM_PowerSupply) GetPropertyRange1InputFrequencyLow() (value uint32, err error) {
	retValue, err := instance.GetProperty("Range1InputFrequencyLow")
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

// SetRange1InputVoltageHigh sets the value of Range1InputVoltageHigh for the instance
func (instance *CIM_PowerSupply) SetPropertyRange1InputVoltageHigh(value uint32) (err error) {
	return instance.SetProperty("Range1InputVoltageHigh", (value))
}

// GetRange1InputVoltageHigh gets the value of Range1InputVoltageHigh for the instance
func (instance *CIM_PowerSupply) GetPropertyRange1InputVoltageHigh() (value uint32, err error) {
	retValue, err := instance.GetProperty("Range1InputVoltageHigh")
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

// SetRange1InputVoltageLow sets the value of Range1InputVoltageLow for the instance
func (instance *CIM_PowerSupply) SetPropertyRange1InputVoltageLow(value uint32) (err error) {
	return instance.SetProperty("Range1InputVoltageLow", (value))
}

// GetRange1InputVoltageLow gets the value of Range1InputVoltageLow for the instance
func (instance *CIM_PowerSupply) GetPropertyRange1InputVoltageLow() (value uint32, err error) {
	retValue, err := instance.GetProperty("Range1InputVoltageLow")
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

// SetRange2InputFrequencyHigh sets the value of Range2InputFrequencyHigh for the instance
func (instance *CIM_PowerSupply) SetPropertyRange2InputFrequencyHigh(value uint32) (err error) {
	return instance.SetProperty("Range2InputFrequencyHigh", (value))
}

// GetRange2InputFrequencyHigh gets the value of Range2InputFrequencyHigh for the instance
func (instance *CIM_PowerSupply) GetPropertyRange2InputFrequencyHigh() (value uint32, err error) {
	retValue, err := instance.GetProperty("Range2InputFrequencyHigh")
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

// SetRange2InputFrequencyLow sets the value of Range2InputFrequencyLow for the instance
func (instance *CIM_PowerSupply) SetPropertyRange2InputFrequencyLow(value uint32) (err error) {
	return instance.SetProperty("Range2InputFrequencyLow", (value))
}

// GetRange2InputFrequencyLow gets the value of Range2InputFrequencyLow for the instance
func (instance *CIM_PowerSupply) GetPropertyRange2InputFrequencyLow() (value uint32, err error) {
	retValue, err := instance.GetProperty("Range2InputFrequencyLow")
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

// SetRange2InputVoltageHigh sets the value of Range2InputVoltageHigh for the instance
func (instance *CIM_PowerSupply) SetPropertyRange2InputVoltageHigh(value uint32) (err error) {
	return instance.SetProperty("Range2InputVoltageHigh", (value))
}

// GetRange2InputVoltageHigh gets the value of Range2InputVoltageHigh for the instance
func (instance *CIM_PowerSupply) GetPropertyRange2InputVoltageHigh() (value uint32, err error) {
	retValue, err := instance.GetProperty("Range2InputVoltageHigh")
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

// SetRange2InputVoltageLow sets the value of Range2InputVoltageLow for the instance
func (instance *CIM_PowerSupply) SetPropertyRange2InputVoltageLow(value uint32) (err error) {
	return instance.SetProperty("Range2InputVoltageLow", (value))
}

// GetRange2InputVoltageLow gets the value of Range2InputVoltageLow for the instance
func (instance *CIM_PowerSupply) GetPropertyRange2InputVoltageLow() (value uint32, err error) {
	retValue, err := instance.GetProperty("Range2InputVoltageLow")
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

// SetTotalOutputPower sets the value of TotalOutputPower for the instance
func (instance *CIM_PowerSupply) SetPropertyTotalOutputPower(value uint32) (err error) {
	return instance.SetProperty("TotalOutputPower", (value))
}

// GetTotalOutputPower gets the value of TotalOutputPower for the instance
func (instance *CIM_PowerSupply) GetPropertyTotalOutputPower() (value uint32, err error) {
	retValue, err := instance.GetProperty("TotalOutputPower")
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

// SetTypeOfRangeSwitching sets the value of TypeOfRangeSwitching for the instance
func (instance *CIM_PowerSupply) SetPropertyTypeOfRangeSwitching(value uint16) (err error) {
	return instance.SetProperty("TypeOfRangeSwitching", (value))
}

// GetTypeOfRangeSwitching gets the value of TypeOfRangeSwitching for the instance
func (instance *CIM_PowerSupply) GetPropertyTypeOfRangeSwitching() (value uint16, err error) {
	retValue, err := instance.GetProperty("TypeOfRangeSwitching")
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
