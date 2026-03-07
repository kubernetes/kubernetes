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

// Win32_PerfRawData_DdmCounterProvider_RAS struct
type Win32_PerfRawData_DdmCounterProvider_RAS struct {
	*Win32_PerfRawData

	//
	BytesReceivedByDisconnectedClients uint64

	//
	BytesTransmittedByDisconnectedClients uint64

	//
	FailedAuthentications uint32

	//
	MaxClients uint32

	//
	TotalClients uint32
}

func NewWin32_PerfRawData_DdmCounterProvider_RASEx1(instance *cim.WmiInstance) (newInstance *Win32_PerfRawData_DdmCounterProvider_RAS, err error) {
	tmp, err := NewWin32_PerfRawDataEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfRawData_DdmCounterProvider_RAS{
		Win32_PerfRawData: tmp,
	}
	return
}

func NewWin32_PerfRawData_DdmCounterProvider_RASEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_PerfRawData_DdmCounterProvider_RAS, err error) {
	tmp, err := NewWin32_PerfRawDataEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfRawData_DdmCounterProvider_RAS{
		Win32_PerfRawData: tmp,
	}
	return
}

// SetBytesReceivedByDisconnectedClients sets the value of BytesReceivedByDisconnectedClients for the instance
func (instance *Win32_PerfRawData_DdmCounterProvider_RAS) SetPropertyBytesReceivedByDisconnectedClients(value uint64) (err error) {
	return instance.SetProperty("BytesReceivedByDisconnectedClients", (value))
}

// GetBytesReceivedByDisconnectedClients gets the value of BytesReceivedByDisconnectedClients for the instance
func (instance *Win32_PerfRawData_DdmCounterProvider_RAS) GetPropertyBytesReceivedByDisconnectedClients() (value uint64, err error) {
	retValue, err := instance.GetProperty("BytesReceivedByDisconnectedClients")
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

// SetBytesTransmittedByDisconnectedClients sets the value of BytesTransmittedByDisconnectedClients for the instance
func (instance *Win32_PerfRawData_DdmCounterProvider_RAS) SetPropertyBytesTransmittedByDisconnectedClients(value uint64) (err error) {
	return instance.SetProperty("BytesTransmittedByDisconnectedClients", (value))
}

// GetBytesTransmittedByDisconnectedClients gets the value of BytesTransmittedByDisconnectedClients for the instance
func (instance *Win32_PerfRawData_DdmCounterProvider_RAS) GetPropertyBytesTransmittedByDisconnectedClients() (value uint64, err error) {
	retValue, err := instance.GetProperty("BytesTransmittedByDisconnectedClients")
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

// SetFailedAuthentications sets the value of FailedAuthentications for the instance
func (instance *Win32_PerfRawData_DdmCounterProvider_RAS) SetPropertyFailedAuthentications(value uint32) (err error) {
	return instance.SetProperty("FailedAuthentications", (value))
}

// GetFailedAuthentications gets the value of FailedAuthentications for the instance
func (instance *Win32_PerfRawData_DdmCounterProvider_RAS) GetPropertyFailedAuthentications() (value uint32, err error) {
	retValue, err := instance.GetProperty("FailedAuthentications")
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

// SetMaxClients sets the value of MaxClients for the instance
func (instance *Win32_PerfRawData_DdmCounterProvider_RAS) SetPropertyMaxClients(value uint32) (err error) {
	return instance.SetProperty("MaxClients", (value))
}

// GetMaxClients gets the value of MaxClients for the instance
func (instance *Win32_PerfRawData_DdmCounterProvider_RAS) GetPropertyMaxClients() (value uint32, err error) {
	retValue, err := instance.GetProperty("MaxClients")
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

// SetTotalClients sets the value of TotalClients for the instance
func (instance *Win32_PerfRawData_DdmCounterProvider_RAS) SetPropertyTotalClients(value uint32) (err error) {
	return instance.SetProperty("TotalClients", (value))
}

// GetTotalClients gets the value of TotalClients for the instance
func (instance *Win32_PerfRawData_DdmCounterProvider_RAS) GetPropertyTotalClients() (value uint32, err error) {
	retValue, err := instance.GetProperty("TotalClients")
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
