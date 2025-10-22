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

// Win32_PerfFormattedData_NETCLRNetworking_NETCLRNetworking struct
type Win32_PerfFormattedData_NETCLRNetworking_NETCLRNetworking struct {
	*Win32_PerfFormattedData

	//
	BytesReceived uint64

	//
	BytesSent uint64

	//
	ConnectionsEstablished uint32

	//
	DatagramsReceived uint32

	//
	DatagramsSent uint32
}

func NewWin32_PerfFormattedData_NETCLRNetworking_NETCLRNetworkingEx1(instance *cim.WmiInstance) (newInstance *Win32_PerfFormattedData_NETCLRNetworking_NETCLRNetworking, err error) {
	tmp, err := NewWin32_PerfFormattedDataEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfFormattedData_NETCLRNetworking_NETCLRNetworking{
		Win32_PerfFormattedData: tmp,
	}
	return
}

func NewWin32_PerfFormattedData_NETCLRNetworking_NETCLRNetworkingEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_PerfFormattedData_NETCLRNetworking_NETCLRNetworking, err error) {
	tmp, err := NewWin32_PerfFormattedDataEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfFormattedData_NETCLRNetworking_NETCLRNetworking{
		Win32_PerfFormattedData: tmp,
	}
	return
}

// SetBytesReceived sets the value of BytesReceived for the instance
func (instance *Win32_PerfFormattedData_NETCLRNetworking_NETCLRNetworking) SetPropertyBytesReceived(value uint64) (err error) {
	return instance.SetProperty("BytesReceived", (value))
}

// GetBytesReceived gets the value of BytesReceived for the instance
func (instance *Win32_PerfFormattedData_NETCLRNetworking_NETCLRNetworking) GetPropertyBytesReceived() (value uint64, err error) {
	retValue, err := instance.GetProperty("BytesReceived")
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

// SetBytesSent sets the value of BytesSent for the instance
func (instance *Win32_PerfFormattedData_NETCLRNetworking_NETCLRNetworking) SetPropertyBytesSent(value uint64) (err error) {
	return instance.SetProperty("BytesSent", (value))
}

// GetBytesSent gets the value of BytesSent for the instance
func (instance *Win32_PerfFormattedData_NETCLRNetworking_NETCLRNetworking) GetPropertyBytesSent() (value uint64, err error) {
	retValue, err := instance.GetProperty("BytesSent")
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

// SetConnectionsEstablished sets the value of ConnectionsEstablished for the instance
func (instance *Win32_PerfFormattedData_NETCLRNetworking_NETCLRNetworking) SetPropertyConnectionsEstablished(value uint32) (err error) {
	return instance.SetProperty("ConnectionsEstablished", (value))
}

// GetConnectionsEstablished gets the value of ConnectionsEstablished for the instance
func (instance *Win32_PerfFormattedData_NETCLRNetworking_NETCLRNetworking) GetPropertyConnectionsEstablished() (value uint32, err error) {
	retValue, err := instance.GetProperty("ConnectionsEstablished")
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

// SetDatagramsReceived sets the value of DatagramsReceived for the instance
func (instance *Win32_PerfFormattedData_NETCLRNetworking_NETCLRNetworking) SetPropertyDatagramsReceived(value uint32) (err error) {
	return instance.SetProperty("DatagramsReceived", (value))
}

// GetDatagramsReceived gets the value of DatagramsReceived for the instance
func (instance *Win32_PerfFormattedData_NETCLRNetworking_NETCLRNetworking) GetPropertyDatagramsReceived() (value uint32, err error) {
	retValue, err := instance.GetProperty("DatagramsReceived")
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

// SetDatagramsSent sets the value of DatagramsSent for the instance
func (instance *Win32_PerfFormattedData_NETCLRNetworking_NETCLRNetworking) SetPropertyDatagramsSent(value uint32) (err error) {
	return instance.SetProperty("DatagramsSent", (value))
}

// GetDatagramsSent gets the value of DatagramsSent for the instance
func (instance *Win32_PerfFormattedData_NETCLRNetworking_NETCLRNetworking) GetPropertyDatagramsSent() (value uint32, err error) {
	retValue, err := instance.GetProperty("DatagramsSent")
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
