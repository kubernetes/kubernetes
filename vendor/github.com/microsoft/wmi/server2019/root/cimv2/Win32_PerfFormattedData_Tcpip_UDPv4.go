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

// Win32_PerfFormattedData_Tcpip_UDPv4 struct
type Win32_PerfFormattedData_Tcpip_UDPv4 struct {
	*Win32_PerfFormattedData

	//
	DatagramsNoPortPersec uint32

	//
	DatagramsPersec uint32

	//
	DatagramsReceivedErrors uint32

	//
	DatagramsReceivedPersec uint32

	//
	DatagramsSentPersec uint32
}

func NewWin32_PerfFormattedData_Tcpip_UDPv4Ex1(instance *cim.WmiInstance) (newInstance *Win32_PerfFormattedData_Tcpip_UDPv4, err error) {
	tmp, err := NewWin32_PerfFormattedDataEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfFormattedData_Tcpip_UDPv4{
		Win32_PerfFormattedData: tmp,
	}
	return
}

func NewWin32_PerfFormattedData_Tcpip_UDPv4Ex6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_PerfFormattedData_Tcpip_UDPv4, err error) {
	tmp, err := NewWin32_PerfFormattedDataEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfFormattedData_Tcpip_UDPv4{
		Win32_PerfFormattedData: tmp,
	}
	return
}

// SetDatagramsNoPortPersec sets the value of DatagramsNoPortPersec for the instance
func (instance *Win32_PerfFormattedData_Tcpip_UDPv4) SetPropertyDatagramsNoPortPersec(value uint32) (err error) {
	return instance.SetProperty("DatagramsNoPortPersec", (value))
}

// GetDatagramsNoPortPersec gets the value of DatagramsNoPortPersec for the instance
func (instance *Win32_PerfFormattedData_Tcpip_UDPv4) GetPropertyDatagramsNoPortPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("DatagramsNoPortPersec")
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

// SetDatagramsPersec sets the value of DatagramsPersec for the instance
func (instance *Win32_PerfFormattedData_Tcpip_UDPv4) SetPropertyDatagramsPersec(value uint32) (err error) {
	return instance.SetProperty("DatagramsPersec", (value))
}

// GetDatagramsPersec gets the value of DatagramsPersec for the instance
func (instance *Win32_PerfFormattedData_Tcpip_UDPv4) GetPropertyDatagramsPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("DatagramsPersec")
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

// SetDatagramsReceivedErrors sets the value of DatagramsReceivedErrors for the instance
func (instance *Win32_PerfFormattedData_Tcpip_UDPv4) SetPropertyDatagramsReceivedErrors(value uint32) (err error) {
	return instance.SetProperty("DatagramsReceivedErrors", (value))
}

// GetDatagramsReceivedErrors gets the value of DatagramsReceivedErrors for the instance
func (instance *Win32_PerfFormattedData_Tcpip_UDPv4) GetPropertyDatagramsReceivedErrors() (value uint32, err error) {
	retValue, err := instance.GetProperty("DatagramsReceivedErrors")
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

// SetDatagramsReceivedPersec sets the value of DatagramsReceivedPersec for the instance
func (instance *Win32_PerfFormattedData_Tcpip_UDPv4) SetPropertyDatagramsReceivedPersec(value uint32) (err error) {
	return instance.SetProperty("DatagramsReceivedPersec", (value))
}

// GetDatagramsReceivedPersec gets the value of DatagramsReceivedPersec for the instance
func (instance *Win32_PerfFormattedData_Tcpip_UDPv4) GetPropertyDatagramsReceivedPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("DatagramsReceivedPersec")
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

// SetDatagramsSentPersec sets the value of DatagramsSentPersec for the instance
func (instance *Win32_PerfFormattedData_Tcpip_UDPv4) SetPropertyDatagramsSentPersec(value uint32) (err error) {
	return instance.SetProperty("DatagramsSentPersec", (value))
}

// GetDatagramsSentPersec gets the value of DatagramsSentPersec for the instance
func (instance *Win32_PerfFormattedData_Tcpip_UDPv4) GetPropertyDatagramsSentPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("DatagramsSentPersec")
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
