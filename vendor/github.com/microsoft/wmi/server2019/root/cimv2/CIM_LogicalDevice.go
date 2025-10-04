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

// CIM_LogicalDevice struct
type CIM_LogicalDevice struct {
	*CIM_LogicalElement

	//
	Availability uint16

	//
	ConfigManagerErrorCode uint32

	//
	ConfigManagerUserConfig bool

	//
	CreationClassName string

	//
	DeviceID string

	//
	ErrorCleared bool

	//
	ErrorDescription string

	//
	LastErrorCode uint32

	//
	PNPDeviceID string

	//
	PowerManagementCapabilities []uint16

	//
	PowerManagementSupported bool

	//
	StatusInfo uint16

	//
	SystemCreationClassName string

	//
	SystemName string
}

func NewCIM_LogicalDeviceEx1(instance *cim.WmiInstance) (newInstance *CIM_LogicalDevice, err error) {
	tmp, err := NewCIM_LogicalElementEx1(instance)

	if err != nil {
		return
	}
	newInstance = &CIM_LogicalDevice{
		CIM_LogicalElement: tmp,
	}
	return
}

func NewCIM_LogicalDeviceEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *CIM_LogicalDevice, err error) {
	tmp, err := NewCIM_LogicalElementEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &CIM_LogicalDevice{
		CIM_LogicalElement: tmp,
	}
	return
}

// SetAvailability sets the value of Availability for the instance
func (instance *CIM_LogicalDevice) SetPropertyAvailability(value uint16) (err error) {
	return instance.SetProperty("Availability", (value))
}

// GetAvailability gets the value of Availability for the instance
func (instance *CIM_LogicalDevice) GetPropertyAvailability() (value uint16, err error) {
	retValue, err := instance.GetProperty("Availability")
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

// SetConfigManagerErrorCode sets the value of ConfigManagerErrorCode for the instance
func (instance *CIM_LogicalDevice) SetPropertyConfigManagerErrorCode(value uint32) (err error) {
	return instance.SetProperty("ConfigManagerErrorCode", (value))
}

// GetConfigManagerErrorCode gets the value of ConfigManagerErrorCode for the instance
func (instance *CIM_LogicalDevice) GetPropertyConfigManagerErrorCode() (value uint32, err error) {
	retValue, err := instance.GetProperty("ConfigManagerErrorCode")
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

// SetConfigManagerUserConfig sets the value of ConfigManagerUserConfig for the instance
func (instance *CIM_LogicalDevice) SetPropertyConfigManagerUserConfig(value bool) (err error) {
	return instance.SetProperty("ConfigManagerUserConfig", (value))
}

// GetConfigManagerUserConfig gets the value of ConfigManagerUserConfig for the instance
func (instance *CIM_LogicalDevice) GetPropertyConfigManagerUserConfig() (value bool, err error) {
	retValue, err := instance.GetProperty("ConfigManagerUserConfig")
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

// SetCreationClassName sets the value of CreationClassName for the instance
func (instance *CIM_LogicalDevice) SetPropertyCreationClassName(value string) (err error) {
	return instance.SetProperty("CreationClassName", (value))
}

// GetCreationClassName gets the value of CreationClassName for the instance
func (instance *CIM_LogicalDevice) GetPropertyCreationClassName() (value string, err error) {
	retValue, err := instance.GetProperty("CreationClassName")
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

// SetDeviceID sets the value of DeviceID for the instance
func (instance *CIM_LogicalDevice) SetPropertyDeviceID(value string) (err error) {
	return instance.SetProperty("DeviceID", (value))
}

// GetDeviceID gets the value of DeviceID for the instance
func (instance *CIM_LogicalDevice) GetPropertyDeviceID() (value string, err error) {
	retValue, err := instance.GetProperty("DeviceID")
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

// SetErrorCleared sets the value of ErrorCleared for the instance
func (instance *CIM_LogicalDevice) SetPropertyErrorCleared(value bool) (err error) {
	return instance.SetProperty("ErrorCleared", (value))
}

// GetErrorCleared gets the value of ErrorCleared for the instance
func (instance *CIM_LogicalDevice) GetPropertyErrorCleared() (value bool, err error) {
	retValue, err := instance.GetProperty("ErrorCleared")
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

// SetErrorDescription sets the value of ErrorDescription for the instance
func (instance *CIM_LogicalDevice) SetPropertyErrorDescription(value string) (err error) {
	return instance.SetProperty("ErrorDescription", (value))
}

// GetErrorDescription gets the value of ErrorDescription for the instance
func (instance *CIM_LogicalDevice) GetPropertyErrorDescription() (value string, err error) {
	retValue, err := instance.GetProperty("ErrorDescription")
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

// SetLastErrorCode sets the value of LastErrorCode for the instance
func (instance *CIM_LogicalDevice) SetPropertyLastErrorCode(value uint32) (err error) {
	return instance.SetProperty("LastErrorCode", (value))
}

// GetLastErrorCode gets the value of LastErrorCode for the instance
func (instance *CIM_LogicalDevice) GetPropertyLastErrorCode() (value uint32, err error) {
	retValue, err := instance.GetProperty("LastErrorCode")
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

// SetPNPDeviceID sets the value of PNPDeviceID for the instance
func (instance *CIM_LogicalDevice) SetPropertyPNPDeviceID(value string) (err error) {
	return instance.SetProperty("PNPDeviceID", (value))
}

// GetPNPDeviceID gets the value of PNPDeviceID for the instance
func (instance *CIM_LogicalDevice) GetPropertyPNPDeviceID() (value string, err error) {
	retValue, err := instance.GetProperty("PNPDeviceID")
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
func (instance *CIM_LogicalDevice) SetPropertyPowerManagementCapabilities(value []uint16) (err error) {
	return instance.SetProperty("PowerManagementCapabilities", (value))
}

// GetPowerManagementCapabilities gets the value of PowerManagementCapabilities for the instance
func (instance *CIM_LogicalDevice) GetPropertyPowerManagementCapabilities() (value []uint16, err error) {
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
func (instance *CIM_LogicalDevice) SetPropertyPowerManagementSupported(value bool) (err error) {
	return instance.SetProperty("PowerManagementSupported", (value))
}

// GetPowerManagementSupported gets the value of PowerManagementSupported for the instance
func (instance *CIM_LogicalDevice) GetPropertyPowerManagementSupported() (value bool, err error) {
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

// SetStatusInfo sets the value of StatusInfo for the instance
func (instance *CIM_LogicalDevice) SetPropertyStatusInfo(value uint16) (err error) {
	return instance.SetProperty("StatusInfo", (value))
}

// GetStatusInfo gets the value of StatusInfo for the instance
func (instance *CIM_LogicalDevice) GetPropertyStatusInfo() (value uint16, err error) {
	retValue, err := instance.GetProperty("StatusInfo")
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

// SetSystemCreationClassName sets the value of SystemCreationClassName for the instance
func (instance *CIM_LogicalDevice) SetPropertySystemCreationClassName(value string) (err error) {
	return instance.SetProperty("SystemCreationClassName", (value))
}

// GetSystemCreationClassName gets the value of SystemCreationClassName for the instance
func (instance *CIM_LogicalDevice) GetPropertySystemCreationClassName() (value string, err error) {
	retValue, err := instance.GetProperty("SystemCreationClassName")
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

// SetSystemName sets the value of SystemName for the instance
func (instance *CIM_LogicalDevice) SetPropertySystemName(value string) (err error) {
	return instance.SetProperty("SystemName", (value))
}

// GetSystemName gets the value of SystemName for the instance
func (instance *CIM_LogicalDevice) GetPropertySystemName() (value string, err error) {
	retValue, err := instance.GetProperty("SystemName")
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

//

// <param name="PowerState" type="uint16 "></param>
// <param name="Time" type="string "></param>

// <param name="ReturnValue" type="uint32 "></param>
func (instance *CIM_LogicalDevice) SetPowerState( /* IN */ PowerState uint16,
	/* IN */ Time string) (result uint32, err error) {
	retVal, err := instance.InvokeMethodWithReturn("SetPowerState", PowerState, Time)
	if err != nil {
		return
	}
	result = uint32(retVal)
	return

}

//

// <param name="ReturnValue" type="uint32 "></param>
func (instance *CIM_LogicalDevice) Reset() (result uint32, err error) {
	retVal, err := instance.InvokeMethodWithReturn("Reset")
	if err != nil {
		return
	}
	result = uint32(retVal)
	return

}
