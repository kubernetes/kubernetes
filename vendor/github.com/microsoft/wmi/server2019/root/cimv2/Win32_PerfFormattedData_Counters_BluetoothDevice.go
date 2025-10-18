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

// Win32_PerfFormattedData_Counters_BluetoothDevice struct
type Win32_PerfFormattedData_Counters_BluetoothDevice struct {
	*Win32_PerfFormattedData

	//
	ClassicACLbytesreadPersec uint32

	//
	ClassicACLbyteswrittenPersec uint32

	//
	LEACLbytesreadPersec uint32

	//
	LEACLbyteswrittenPersec uint32

	//
	SCObytesreadPersec uint32

	//
	SCObyteswrittenPersec uint32
}

func NewWin32_PerfFormattedData_Counters_BluetoothDeviceEx1(instance *cim.WmiInstance) (newInstance *Win32_PerfFormattedData_Counters_BluetoothDevice, err error) {
	tmp, err := NewWin32_PerfFormattedDataEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfFormattedData_Counters_BluetoothDevice{
		Win32_PerfFormattedData: tmp,
	}
	return
}

func NewWin32_PerfFormattedData_Counters_BluetoothDeviceEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_PerfFormattedData_Counters_BluetoothDevice, err error) {
	tmp, err := NewWin32_PerfFormattedDataEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfFormattedData_Counters_BluetoothDevice{
		Win32_PerfFormattedData: tmp,
	}
	return
}

// SetClassicACLbytesreadPersec sets the value of ClassicACLbytesreadPersec for the instance
func (instance *Win32_PerfFormattedData_Counters_BluetoothDevice) SetPropertyClassicACLbytesreadPersec(value uint32) (err error) {
	return instance.SetProperty("ClassicACLbytesreadPersec", (value))
}

// GetClassicACLbytesreadPersec gets the value of ClassicACLbytesreadPersec for the instance
func (instance *Win32_PerfFormattedData_Counters_BluetoothDevice) GetPropertyClassicACLbytesreadPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("ClassicACLbytesreadPersec")
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

// SetClassicACLbyteswrittenPersec sets the value of ClassicACLbyteswrittenPersec for the instance
func (instance *Win32_PerfFormattedData_Counters_BluetoothDevice) SetPropertyClassicACLbyteswrittenPersec(value uint32) (err error) {
	return instance.SetProperty("ClassicACLbyteswrittenPersec", (value))
}

// GetClassicACLbyteswrittenPersec gets the value of ClassicACLbyteswrittenPersec for the instance
func (instance *Win32_PerfFormattedData_Counters_BluetoothDevice) GetPropertyClassicACLbyteswrittenPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("ClassicACLbyteswrittenPersec")
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

// SetLEACLbytesreadPersec sets the value of LEACLbytesreadPersec for the instance
func (instance *Win32_PerfFormattedData_Counters_BluetoothDevice) SetPropertyLEACLbytesreadPersec(value uint32) (err error) {
	return instance.SetProperty("LEACLbytesreadPersec", (value))
}

// GetLEACLbytesreadPersec gets the value of LEACLbytesreadPersec for the instance
func (instance *Win32_PerfFormattedData_Counters_BluetoothDevice) GetPropertyLEACLbytesreadPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("LEACLbytesreadPersec")
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

// SetLEACLbyteswrittenPersec sets the value of LEACLbyteswrittenPersec for the instance
func (instance *Win32_PerfFormattedData_Counters_BluetoothDevice) SetPropertyLEACLbyteswrittenPersec(value uint32) (err error) {
	return instance.SetProperty("LEACLbyteswrittenPersec", (value))
}

// GetLEACLbyteswrittenPersec gets the value of LEACLbyteswrittenPersec for the instance
func (instance *Win32_PerfFormattedData_Counters_BluetoothDevice) GetPropertyLEACLbyteswrittenPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("LEACLbyteswrittenPersec")
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

// SetSCObytesreadPersec sets the value of SCObytesreadPersec for the instance
func (instance *Win32_PerfFormattedData_Counters_BluetoothDevice) SetPropertySCObytesreadPersec(value uint32) (err error) {
	return instance.SetProperty("SCObytesreadPersec", (value))
}

// GetSCObytesreadPersec gets the value of SCObytesreadPersec for the instance
func (instance *Win32_PerfFormattedData_Counters_BluetoothDevice) GetPropertySCObytesreadPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("SCObytesreadPersec")
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

// SetSCObyteswrittenPersec sets the value of SCObyteswrittenPersec for the instance
func (instance *Win32_PerfFormattedData_Counters_BluetoothDevice) SetPropertySCObyteswrittenPersec(value uint32) (err error) {
	return instance.SetProperty("SCObyteswrittenPersec", (value))
}

// GetSCObyteswrittenPersec gets the value of SCObyteswrittenPersec for the instance
func (instance *Win32_PerfFormattedData_Counters_BluetoothDevice) GetPropertySCObyteswrittenPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("SCObyteswrittenPersec")
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
