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

// CIM_USBDevice struct
type CIM_USBDevice struct {
	*CIM_LogicalDevice

	//
	ClassCode uint8

	//
	CurrentAlternateSettings []uint8

	//
	CurrentConfigValue uint8

	//
	NumberOfConfigs uint8

	//
	ProtocolCode uint8

	//
	SubclassCode uint8

	//
	USBVersion uint16
}

func NewCIM_USBDeviceEx1(instance *cim.WmiInstance) (newInstance *CIM_USBDevice, err error) {
	tmp, err := NewCIM_LogicalDeviceEx1(instance)

	if err != nil {
		return
	}
	newInstance = &CIM_USBDevice{
		CIM_LogicalDevice: tmp,
	}
	return
}

func NewCIM_USBDeviceEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *CIM_USBDevice, err error) {
	tmp, err := NewCIM_LogicalDeviceEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &CIM_USBDevice{
		CIM_LogicalDevice: tmp,
	}
	return
}

// SetClassCode sets the value of ClassCode for the instance
func (instance *CIM_USBDevice) SetPropertyClassCode(value uint8) (err error) {
	return instance.SetProperty("ClassCode", (value))
}

// GetClassCode gets the value of ClassCode for the instance
func (instance *CIM_USBDevice) GetPropertyClassCode() (value uint8, err error) {
	retValue, err := instance.GetProperty("ClassCode")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint8)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint8 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint8(valuetmp)

	return
}

// SetCurrentAlternateSettings sets the value of CurrentAlternateSettings for the instance
func (instance *CIM_USBDevice) SetPropertyCurrentAlternateSettings(value []uint8) (err error) {
	return instance.SetProperty("CurrentAlternateSettings", (value))
}

// GetCurrentAlternateSettings gets the value of CurrentAlternateSettings for the instance
func (instance *CIM_USBDevice) GetPropertyCurrentAlternateSettings() (value []uint8, err error) {
	retValue, err := instance.GetProperty("CurrentAlternateSettings")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	for _, interfaceValue := range retValue.([]interface{}) {
		valuetmp, ok := interfaceValue.(uint8)
		if !ok {
			err = errors.Wrapf(errors.InvalidType, " uint8 is Invalid. Expected %s", reflect.TypeOf(interfaceValue))
			return
		}
		value = append(value, uint8(valuetmp))
	}

	return
}

// SetCurrentConfigValue sets the value of CurrentConfigValue for the instance
func (instance *CIM_USBDevice) SetPropertyCurrentConfigValue(value uint8) (err error) {
	return instance.SetProperty("CurrentConfigValue", (value))
}

// GetCurrentConfigValue gets the value of CurrentConfigValue for the instance
func (instance *CIM_USBDevice) GetPropertyCurrentConfigValue() (value uint8, err error) {
	retValue, err := instance.GetProperty("CurrentConfigValue")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint8)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint8 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint8(valuetmp)

	return
}

// SetNumberOfConfigs sets the value of NumberOfConfigs for the instance
func (instance *CIM_USBDevice) SetPropertyNumberOfConfigs(value uint8) (err error) {
	return instance.SetProperty("NumberOfConfigs", (value))
}

// GetNumberOfConfigs gets the value of NumberOfConfigs for the instance
func (instance *CIM_USBDevice) GetPropertyNumberOfConfigs() (value uint8, err error) {
	retValue, err := instance.GetProperty("NumberOfConfigs")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint8)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint8 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint8(valuetmp)

	return
}

// SetProtocolCode sets the value of ProtocolCode for the instance
func (instance *CIM_USBDevice) SetPropertyProtocolCode(value uint8) (err error) {
	return instance.SetProperty("ProtocolCode", (value))
}

// GetProtocolCode gets the value of ProtocolCode for the instance
func (instance *CIM_USBDevice) GetPropertyProtocolCode() (value uint8, err error) {
	retValue, err := instance.GetProperty("ProtocolCode")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint8)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint8 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint8(valuetmp)

	return
}

// SetSubclassCode sets the value of SubclassCode for the instance
func (instance *CIM_USBDevice) SetPropertySubclassCode(value uint8) (err error) {
	return instance.SetProperty("SubclassCode", (value))
}

// GetSubclassCode gets the value of SubclassCode for the instance
func (instance *CIM_USBDevice) GetPropertySubclassCode() (value uint8, err error) {
	retValue, err := instance.GetProperty("SubclassCode")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint8)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint8 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint8(valuetmp)

	return
}

// SetUSBVersion sets the value of USBVersion for the instance
func (instance *CIM_USBDevice) SetPropertyUSBVersion(value uint16) (err error) {
	return instance.SetProperty("USBVersion", (value))
}

// GetUSBVersion gets the value of USBVersion for the instance
func (instance *CIM_USBDevice) GetPropertyUSBVersion() (value uint16, err error) {
	retValue, err := instance.GetProperty("USBVersion")
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

// <param name="RequestIndex" type="uint16 "></param>
// <param name="RequestLength" type="uint16 "></param>
// <param name="RequestType" type="uint8 "></param>
// <param name="RequestValue" type="uint16 "></param>

// <param name="Buffer" type="uint8 []"></param>
// <param name="RequestLength" type="uint16 "></param>
// <param name="ReturnValue" type="uint32 "></param>
func (instance *CIM_USBDevice) GetDescriptor( /* IN */ RequestType uint8,
	/* IN */ RequestValue uint16,
	/* IN */ RequestIndex uint16,
	/* IN/OUT */ RequestLength uint16,
	/* OUT */ Buffer []uint8) (result uint32, err error) {
	retVal, err := instance.InvokeMethod("GetDescriptor", RequestType, RequestValue, RequestIndex)
	if err != nil {
		return
	}
	retValue := retVal[0].(int32)
	result = uint32(retValue)
	return

}
