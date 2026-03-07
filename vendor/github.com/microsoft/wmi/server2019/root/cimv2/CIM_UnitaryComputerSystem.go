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

// CIM_UnitaryComputerSystem struct
type CIM_UnitaryComputerSystem struct {
	*CIM_ComputerSystem

	//
	InitialLoadInfo []string

	//
	LastLoadInfo string

	//
	PowerManagementCapabilities []uint16

	//
	PowerManagementSupported bool

	//
	PowerState uint16

	//
	ResetCapability uint16
}

func NewCIM_UnitaryComputerSystemEx1(instance *cim.WmiInstance) (newInstance *CIM_UnitaryComputerSystem, err error) {
	tmp, err := NewCIM_ComputerSystemEx1(instance)

	if err != nil {
		return
	}
	newInstance = &CIM_UnitaryComputerSystem{
		CIM_ComputerSystem: tmp,
	}
	return
}

func NewCIM_UnitaryComputerSystemEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *CIM_UnitaryComputerSystem, err error) {
	tmp, err := NewCIM_ComputerSystemEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &CIM_UnitaryComputerSystem{
		CIM_ComputerSystem: tmp,
	}
	return
}

// SetInitialLoadInfo sets the value of InitialLoadInfo for the instance
func (instance *CIM_UnitaryComputerSystem) SetPropertyInitialLoadInfo(value []string) (err error) {
	return instance.SetProperty("InitialLoadInfo", (value))
}

// GetInitialLoadInfo gets the value of InitialLoadInfo for the instance
func (instance *CIM_UnitaryComputerSystem) GetPropertyInitialLoadInfo() (value []string, err error) {
	retValue, err := instance.GetProperty("InitialLoadInfo")
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

// SetLastLoadInfo sets the value of LastLoadInfo for the instance
func (instance *CIM_UnitaryComputerSystem) SetPropertyLastLoadInfo(value string) (err error) {
	return instance.SetProperty("LastLoadInfo", (value))
}

// GetLastLoadInfo gets the value of LastLoadInfo for the instance
func (instance *CIM_UnitaryComputerSystem) GetPropertyLastLoadInfo() (value string, err error) {
	retValue, err := instance.GetProperty("LastLoadInfo")
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

// SetPowerManagementCapabilities sets the value of PowerManagementCapabilities for the instance
func (instance *CIM_UnitaryComputerSystem) SetPropertyPowerManagementCapabilities(value []uint16) (err error) {
	return instance.SetProperty("PowerManagementCapabilities", (value))
}

// GetPowerManagementCapabilities gets the value of PowerManagementCapabilities for the instance
func (instance *CIM_UnitaryComputerSystem) GetPropertyPowerManagementCapabilities() (value []uint16, err error) {
	retValue, err := instance.GetProperty("PowerManagementCapabilities")
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

// SetPowerManagementSupported sets the value of PowerManagementSupported for the instance
func (instance *CIM_UnitaryComputerSystem) SetPropertyPowerManagementSupported(value bool) (err error) {
	return instance.SetProperty("PowerManagementSupported", (value))
}

// GetPowerManagementSupported gets the value of PowerManagementSupported for the instance
func (instance *CIM_UnitaryComputerSystem) GetPropertyPowerManagementSupported() (value bool, err error) {
	retValue, err := instance.GetProperty("PowerManagementSupported")
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

// SetPowerState sets the value of PowerState for the instance
func (instance *CIM_UnitaryComputerSystem) SetPropertyPowerState(value uint16) (err error) {
	return instance.SetProperty("PowerState", (value))
}

// GetPowerState gets the value of PowerState for the instance
func (instance *CIM_UnitaryComputerSystem) GetPropertyPowerState() (value uint16, err error) {
	retValue, err := instance.GetProperty("PowerState")
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

// SetResetCapability sets the value of ResetCapability for the instance
func (instance *CIM_UnitaryComputerSystem) SetPropertyResetCapability(value uint16) (err error) {
	return instance.SetProperty("ResetCapability", (value))
}

// GetResetCapability gets the value of ResetCapability for the instance
func (instance *CIM_UnitaryComputerSystem) GetPropertyResetCapability() (value uint16, err error) {
	retValue, err := instance.GetProperty("ResetCapability")
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

//

// <param name="PowerState" type="uint16 "></param>
// <param name="Time" type="string "></param>

// <param name="ReturnValue" type="uint32 "></param>
func (instance *CIM_UnitaryComputerSystem) SetPowerState( /* IN */ PowerState uint16,
	/* IN */ Time string) (result uint32, err error) {
	retVal, err := instance.InvokeMethodWithReturn("SetPowerState", PowerState, Time)
	if err != nil {
		return
	}
	result = uint32(retVal)
	return

}
