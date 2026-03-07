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

// Win32_PerfFormattedData_WinNatCounters_WinNATInstance struct
type Win32_PerfFormattedData_WinNatCounters_WinNATInstance struct {
	*Win32_PerfFormattedData

	//
	TCPPortsAvailable uint32

	//
	TCPPortsInUse uint32

	//
	UDPPortsAvailable uint32

	//
	UDPPortsInUse uint32
}

func NewWin32_PerfFormattedData_WinNatCounters_WinNATInstanceEx1(instance *cim.WmiInstance) (newInstance *Win32_PerfFormattedData_WinNatCounters_WinNATInstance, err error) {
	tmp, err := NewWin32_PerfFormattedDataEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfFormattedData_WinNatCounters_WinNATInstance{
		Win32_PerfFormattedData: tmp,
	}
	return
}

func NewWin32_PerfFormattedData_WinNatCounters_WinNATInstanceEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_PerfFormattedData_WinNatCounters_WinNATInstance, err error) {
	tmp, err := NewWin32_PerfFormattedDataEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfFormattedData_WinNatCounters_WinNATInstance{
		Win32_PerfFormattedData: tmp,
	}
	return
}

// SetTCPPortsAvailable sets the value of TCPPortsAvailable for the instance
func (instance *Win32_PerfFormattedData_WinNatCounters_WinNATInstance) SetPropertyTCPPortsAvailable(value uint32) (err error) {
	return instance.SetProperty("TCPPortsAvailable", (value))
}

// GetTCPPortsAvailable gets the value of TCPPortsAvailable for the instance
func (instance *Win32_PerfFormattedData_WinNatCounters_WinNATInstance) GetPropertyTCPPortsAvailable() (value uint32, err error) {
	retValue, err := instance.GetProperty("TCPPortsAvailable")
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

// SetTCPPortsInUse sets the value of TCPPortsInUse for the instance
func (instance *Win32_PerfFormattedData_WinNatCounters_WinNATInstance) SetPropertyTCPPortsInUse(value uint32) (err error) {
	return instance.SetProperty("TCPPortsInUse", (value))
}

// GetTCPPortsInUse gets the value of TCPPortsInUse for the instance
func (instance *Win32_PerfFormattedData_WinNatCounters_WinNATInstance) GetPropertyTCPPortsInUse() (value uint32, err error) {
	retValue, err := instance.GetProperty("TCPPortsInUse")
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

// SetUDPPortsAvailable sets the value of UDPPortsAvailable for the instance
func (instance *Win32_PerfFormattedData_WinNatCounters_WinNATInstance) SetPropertyUDPPortsAvailable(value uint32) (err error) {
	return instance.SetProperty("UDPPortsAvailable", (value))
}

// GetUDPPortsAvailable gets the value of UDPPortsAvailable for the instance
func (instance *Win32_PerfFormattedData_WinNatCounters_WinNATInstance) GetPropertyUDPPortsAvailable() (value uint32, err error) {
	retValue, err := instance.GetProperty("UDPPortsAvailable")
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

// SetUDPPortsInUse sets the value of UDPPortsInUse for the instance
func (instance *Win32_PerfFormattedData_WinNatCounters_WinNATInstance) SetPropertyUDPPortsInUse(value uint32) (err error) {
	return instance.SetProperty("UDPPortsInUse", (value))
}

// GetUDPPortsInUse gets the value of UDPPortsInUse for the instance
func (instance *Win32_PerfFormattedData_WinNatCounters_WinNATInstance) GetPropertyUDPPortsInUse() (value uint32, err error) {
	retValue, err := instance.GetProperty("UDPPortsInUse")
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
