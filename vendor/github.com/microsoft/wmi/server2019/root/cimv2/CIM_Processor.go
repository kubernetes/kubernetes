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

// CIM_Processor struct
type CIM_Processor struct {
	*CIM_LogicalDevice

	//
	AddressWidth uint16

	//
	CurrentClockSpeed uint32

	//
	DataWidth uint16

	//
	Family uint16

	//
	LoadPercentage uint16

	//
	MaxClockSpeed uint32

	//
	OtherFamilyDescription string

	//
	Role string

	//
	Stepping string

	//
	UniqueId string

	//
	UpgradeMethod uint16
}

func NewCIM_ProcessorEx1(instance *cim.WmiInstance) (newInstance *CIM_Processor, err error) {
	tmp, err := NewCIM_LogicalDeviceEx1(instance)

	if err != nil {
		return
	}
	newInstance = &CIM_Processor{
		CIM_LogicalDevice: tmp,
	}
	return
}

func NewCIM_ProcessorEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *CIM_Processor, err error) {
	tmp, err := NewCIM_LogicalDeviceEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &CIM_Processor{
		CIM_LogicalDevice: tmp,
	}
	return
}

// SetAddressWidth sets the value of AddressWidth for the instance
func (instance *CIM_Processor) SetPropertyAddressWidth(value uint16) (err error) {
	return instance.SetProperty("AddressWidth", (value))
}

// GetAddressWidth gets the value of AddressWidth for the instance
func (instance *CIM_Processor) GetPropertyAddressWidth() (value uint16, err error) {
	retValue, err := instance.GetProperty("AddressWidth")
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

// SetCurrentClockSpeed sets the value of CurrentClockSpeed for the instance
func (instance *CIM_Processor) SetPropertyCurrentClockSpeed(value uint32) (err error) {
	return instance.SetProperty("CurrentClockSpeed", (value))
}

// GetCurrentClockSpeed gets the value of CurrentClockSpeed for the instance
func (instance *CIM_Processor) GetPropertyCurrentClockSpeed() (value uint32, err error) {
	retValue, err := instance.GetProperty("CurrentClockSpeed")
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

// SetDataWidth sets the value of DataWidth for the instance
func (instance *CIM_Processor) SetPropertyDataWidth(value uint16) (err error) {
	return instance.SetProperty("DataWidth", (value))
}

// GetDataWidth gets the value of DataWidth for the instance
func (instance *CIM_Processor) GetPropertyDataWidth() (value uint16, err error) {
	retValue, err := instance.GetProperty("DataWidth")
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

// SetFamily sets the value of Family for the instance
func (instance *CIM_Processor) SetPropertyFamily(value uint16) (err error) {
	return instance.SetProperty("Family", (value))
}

// GetFamily gets the value of Family for the instance
func (instance *CIM_Processor) GetPropertyFamily() (value uint16, err error) {
	retValue, err := instance.GetProperty("Family")
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

// SetLoadPercentage sets the value of LoadPercentage for the instance
func (instance *CIM_Processor) SetPropertyLoadPercentage(value uint16) (err error) {
	return instance.SetProperty("LoadPercentage", (value))
}

// GetLoadPercentage gets the value of LoadPercentage for the instance
func (instance *CIM_Processor) GetPropertyLoadPercentage() (value uint16, err error) {
	retValue, err := instance.GetProperty("LoadPercentage")
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

// SetMaxClockSpeed sets the value of MaxClockSpeed for the instance
func (instance *CIM_Processor) SetPropertyMaxClockSpeed(value uint32) (err error) {
	return instance.SetProperty("MaxClockSpeed", (value))
}

// GetMaxClockSpeed gets the value of MaxClockSpeed for the instance
func (instance *CIM_Processor) GetPropertyMaxClockSpeed() (value uint32, err error) {
	retValue, err := instance.GetProperty("MaxClockSpeed")
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

// SetOtherFamilyDescription sets the value of OtherFamilyDescription for the instance
func (instance *CIM_Processor) SetPropertyOtherFamilyDescription(value string) (err error) {
	return instance.SetProperty("OtherFamilyDescription", (value))
}

// GetOtherFamilyDescription gets the value of OtherFamilyDescription for the instance
func (instance *CIM_Processor) GetPropertyOtherFamilyDescription() (value string, err error) {
	retValue, err := instance.GetProperty("OtherFamilyDescription")
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

// SetRole sets the value of Role for the instance
func (instance *CIM_Processor) SetPropertyRole(value string) (err error) {
	return instance.SetProperty("Role", (value))
}

// GetRole gets the value of Role for the instance
func (instance *CIM_Processor) GetPropertyRole() (value string, err error) {
	retValue, err := instance.GetProperty("Role")
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

// SetStepping sets the value of Stepping for the instance
func (instance *CIM_Processor) SetPropertyStepping(value string) (err error) {
	return instance.SetProperty("Stepping", (value))
}

// GetStepping gets the value of Stepping for the instance
func (instance *CIM_Processor) GetPropertyStepping() (value string, err error) {
	retValue, err := instance.GetProperty("Stepping")
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

// SetUniqueId sets the value of UniqueId for the instance
func (instance *CIM_Processor) SetPropertyUniqueId(value string) (err error) {
	return instance.SetProperty("UniqueId", (value))
}

// GetUniqueId gets the value of UniqueId for the instance
func (instance *CIM_Processor) GetPropertyUniqueId() (value string, err error) {
	retValue, err := instance.GetProperty("UniqueId")
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

// SetUpgradeMethod sets the value of UpgradeMethod for the instance
func (instance *CIM_Processor) SetPropertyUpgradeMethod(value uint16) (err error) {
	return instance.SetProperty("UpgradeMethod", (value))
}

// GetUpgradeMethod gets the value of UpgradeMethod for the instance
func (instance *CIM_Processor) GetPropertyUpgradeMethod() (value uint16, err error) {
	retValue, err := instance.GetProperty("UpgradeMethod")
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
