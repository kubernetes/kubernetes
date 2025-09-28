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

// CIM_Battery struct
type CIM_Battery struct {
	*CIM_LogicalDevice

	//
	BatteryStatus uint16

	//
	Chemistry uint16

	//
	DesignCapacity uint32

	//
	DesignVoltage uint64

	//
	EstimatedChargeRemaining uint16

	//
	EstimatedRunTime uint32

	//
	ExpectedLife uint32

	//
	FullChargeCapacity uint32

	//
	MaxRechargeTime uint32

	//
	SmartBatteryVersion string

	//
	TimeOnBattery uint32

	//
	TimeToFullCharge uint32
}

func NewCIM_BatteryEx1(instance *cim.WmiInstance) (newInstance *CIM_Battery, err error) {
	tmp, err := NewCIM_LogicalDeviceEx1(instance)

	if err != nil {
		return
	}
	newInstance = &CIM_Battery{
		CIM_LogicalDevice: tmp,
	}
	return
}

func NewCIM_BatteryEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *CIM_Battery, err error) {
	tmp, err := NewCIM_LogicalDeviceEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &CIM_Battery{
		CIM_LogicalDevice: tmp,
	}
	return
}

// SetBatteryStatus sets the value of BatteryStatus for the instance
func (instance *CIM_Battery) SetPropertyBatteryStatus(value uint16) (err error) {
	return instance.SetProperty("BatteryStatus", (value))
}

// GetBatteryStatus gets the value of BatteryStatus for the instance
func (instance *CIM_Battery) GetPropertyBatteryStatus() (value uint16, err error) {
	retValue, err := instance.GetProperty("BatteryStatus")
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

// SetChemistry sets the value of Chemistry for the instance
func (instance *CIM_Battery) SetPropertyChemistry(value uint16) (err error) {
	return instance.SetProperty("Chemistry", (value))
}

// GetChemistry gets the value of Chemistry for the instance
func (instance *CIM_Battery) GetPropertyChemistry() (value uint16, err error) {
	retValue, err := instance.GetProperty("Chemistry")
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

// SetDesignCapacity sets the value of DesignCapacity for the instance
func (instance *CIM_Battery) SetPropertyDesignCapacity(value uint32) (err error) {
	return instance.SetProperty("DesignCapacity", (value))
}

// GetDesignCapacity gets the value of DesignCapacity for the instance
func (instance *CIM_Battery) GetPropertyDesignCapacity() (value uint32, err error) {
	retValue, err := instance.GetProperty("DesignCapacity")
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

// SetDesignVoltage sets the value of DesignVoltage for the instance
func (instance *CIM_Battery) SetPropertyDesignVoltage(value uint64) (err error) {
	return instance.SetProperty("DesignVoltage", (value))
}

// GetDesignVoltage gets the value of DesignVoltage for the instance
func (instance *CIM_Battery) GetPropertyDesignVoltage() (value uint64, err error) {
	retValue, err := instance.GetProperty("DesignVoltage")
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

// SetEstimatedChargeRemaining sets the value of EstimatedChargeRemaining for the instance
func (instance *CIM_Battery) SetPropertyEstimatedChargeRemaining(value uint16) (err error) {
	return instance.SetProperty("EstimatedChargeRemaining", (value))
}

// GetEstimatedChargeRemaining gets the value of EstimatedChargeRemaining for the instance
func (instance *CIM_Battery) GetPropertyEstimatedChargeRemaining() (value uint16, err error) {
	retValue, err := instance.GetProperty("EstimatedChargeRemaining")
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

// SetEstimatedRunTime sets the value of EstimatedRunTime for the instance
func (instance *CIM_Battery) SetPropertyEstimatedRunTime(value uint32) (err error) {
	return instance.SetProperty("EstimatedRunTime", (value))
}

// GetEstimatedRunTime gets the value of EstimatedRunTime for the instance
func (instance *CIM_Battery) GetPropertyEstimatedRunTime() (value uint32, err error) {
	retValue, err := instance.GetProperty("EstimatedRunTime")
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

// SetExpectedLife sets the value of ExpectedLife for the instance
func (instance *CIM_Battery) SetPropertyExpectedLife(value uint32) (err error) {
	return instance.SetProperty("ExpectedLife", (value))
}

// GetExpectedLife gets the value of ExpectedLife for the instance
func (instance *CIM_Battery) GetPropertyExpectedLife() (value uint32, err error) {
	retValue, err := instance.GetProperty("ExpectedLife")
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

// SetFullChargeCapacity sets the value of FullChargeCapacity for the instance
func (instance *CIM_Battery) SetPropertyFullChargeCapacity(value uint32) (err error) {
	return instance.SetProperty("FullChargeCapacity", (value))
}

// GetFullChargeCapacity gets the value of FullChargeCapacity for the instance
func (instance *CIM_Battery) GetPropertyFullChargeCapacity() (value uint32, err error) {
	retValue, err := instance.GetProperty("FullChargeCapacity")
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

// SetMaxRechargeTime sets the value of MaxRechargeTime for the instance
func (instance *CIM_Battery) SetPropertyMaxRechargeTime(value uint32) (err error) {
	return instance.SetProperty("MaxRechargeTime", (value))
}

// GetMaxRechargeTime gets the value of MaxRechargeTime for the instance
func (instance *CIM_Battery) GetPropertyMaxRechargeTime() (value uint32, err error) {
	retValue, err := instance.GetProperty("MaxRechargeTime")
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

// SetSmartBatteryVersion sets the value of SmartBatteryVersion for the instance
func (instance *CIM_Battery) SetPropertySmartBatteryVersion(value string) (err error) {
	return instance.SetProperty("SmartBatteryVersion", (value))
}

// GetSmartBatteryVersion gets the value of SmartBatteryVersion for the instance
func (instance *CIM_Battery) GetPropertySmartBatteryVersion() (value string, err error) {
	retValue, err := instance.GetProperty("SmartBatteryVersion")
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

// SetTimeOnBattery sets the value of TimeOnBattery for the instance
func (instance *CIM_Battery) SetPropertyTimeOnBattery(value uint32) (err error) {
	return instance.SetProperty("TimeOnBattery", (value))
}

// GetTimeOnBattery gets the value of TimeOnBattery for the instance
func (instance *CIM_Battery) GetPropertyTimeOnBattery() (value uint32, err error) {
	retValue, err := instance.GetProperty("TimeOnBattery")
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

// SetTimeToFullCharge sets the value of TimeToFullCharge for the instance
func (instance *CIM_Battery) SetPropertyTimeToFullCharge(value uint32) (err error) {
	return instance.SetProperty("TimeToFullCharge", (value))
}

// GetTimeToFullCharge gets the value of TimeToFullCharge for the instance
func (instance *CIM_Battery) GetPropertyTimeToFullCharge() (value uint32, err error) {
	retValue, err := instance.GetProperty("TimeToFullCharge")
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
