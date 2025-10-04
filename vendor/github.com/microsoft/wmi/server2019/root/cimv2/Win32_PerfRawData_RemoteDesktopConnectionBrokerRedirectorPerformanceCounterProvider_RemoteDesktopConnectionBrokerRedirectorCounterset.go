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

// Win32_PerfRawData_RemoteDesktopConnectionBrokerRedirectorPerformanceCounterProvider_RemoteDesktopConnectionBrokerRedirectorCounterset struct
type Win32_PerfRawData_RemoteDesktopConnectionBrokerRedirectorPerformanceCounterProvider_RemoteDesktopConnectionBrokerRedirectorCounterset struct {
	*Win32_PerfRawData

	//
	Connectiontime uint64

	//
	Contextacquisitionwaittime uint64

	//
	RPCContext uint64

	//
	ThreadswaitingforRPCContext uint64
}

func NewWin32_PerfRawData_RemoteDesktopConnectionBrokerRedirectorPerformanceCounterProvider_RemoteDesktopConnectionBrokerRedirectorCountersetEx1(instance *cim.WmiInstance) (newInstance *Win32_PerfRawData_RemoteDesktopConnectionBrokerRedirectorPerformanceCounterProvider_RemoteDesktopConnectionBrokerRedirectorCounterset, err error) {
	tmp, err := NewWin32_PerfRawDataEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfRawData_RemoteDesktopConnectionBrokerRedirectorPerformanceCounterProvider_RemoteDesktopConnectionBrokerRedirectorCounterset{
		Win32_PerfRawData: tmp,
	}
	return
}

func NewWin32_PerfRawData_RemoteDesktopConnectionBrokerRedirectorPerformanceCounterProvider_RemoteDesktopConnectionBrokerRedirectorCountersetEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_PerfRawData_RemoteDesktopConnectionBrokerRedirectorPerformanceCounterProvider_RemoteDesktopConnectionBrokerRedirectorCounterset, err error) {
	tmp, err := NewWin32_PerfRawDataEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfRawData_RemoteDesktopConnectionBrokerRedirectorPerformanceCounterProvider_RemoteDesktopConnectionBrokerRedirectorCounterset{
		Win32_PerfRawData: tmp,
	}
	return
}

// SetConnectiontime sets the value of Connectiontime for the instance
func (instance *Win32_PerfRawData_RemoteDesktopConnectionBrokerRedirectorPerformanceCounterProvider_RemoteDesktopConnectionBrokerRedirectorCounterset) SetPropertyConnectiontime(value uint64) (err error) {
	return instance.SetProperty("Connectiontime", (value))
}

// GetConnectiontime gets the value of Connectiontime for the instance
func (instance *Win32_PerfRawData_RemoteDesktopConnectionBrokerRedirectorPerformanceCounterProvider_RemoteDesktopConnectionBrokerRedirectorCounterset) GetPropertyConnectiontime() (value uint64, err error) {
	retValue, err := instance.GetProperty("Connectiontime")
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

// SetContextacquisitionwaittime sets the value of Contextacquisitionwaittime for the instance
func (instance *Win32_PerfRawData_RemoteDesktopConnectionBrokerRedirectorPerformanceCounterProvider_RemoteDesktopConnectionBrokerRedirectorCounterset) SetPropertyContextacquisitionwaittime(value uint64) (err error) {
	return instance.SetProperty("Contextacquisitionwaittime", (value))
}

// GetContextacquisitionwaittime gets the value of Contextacquisitionwaittime for the instance
func (instance *Win32_PerfRawData_RemoteDesktopConnectionBrokerRedirectorPerformanceCounterProvider_RemoteDesktopConnectionBrokerRedirectorCounterset) GetPropertyContextacquisitionwaittime() (value uint64, err error) {
	retValue, err := instance.GetProperty("Contextacquisitionwaittime")
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

// SetRPCContext sets the value of RPCContext for the instance
func (instance *Win32_PerfRawData_RemoteDesktopConnectionBrokerRedirectorPerformanceCounterProvider_RemoteDesktopConnectionBrokerRedirectorCounterset) SetPropertyRPCContext(value uint64) (err error) {
	return instance.SetProperty("RPCContext", (value))
}

// GetRPCContext gets the value of RPCContext for the instance
func (instance *Win32_PerfRawData_RemoteDesktopConnectionBrokerRedirectorPerformanceCounterProvider_RemoteDesktopConnectionBrokerRedirectorCounterset) GetPropertyRPCContext() (value uint64, err error) {
	retValue, err := instance.GetProperty("RPCContext")
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

// SetThreadswaitingforRPCContext sets the value of ThreadswaitingforRPCContext for the instance
func (instance *Win32_PerfRawData_RemoteDesktopConnectionBrokerRedirectorPerformanceCounterProvider_RemoteDesktopConnectionBrokerRedirectorCounterset) SetPropertyThreadswaitingforRPCContext(value uint64) (err error) {
	return instance.SetProperty("ThreadswaitingforRPCContext", (value))
}

// GetThreadswaitingforRPCContext gets the value of ThreadswaitingforRPCContext for the instance
func (instance *Win32_PerfRawData_RemoteDesktopConnectionBrokerRedirectorPerformanceCounterProvider_RemoteDesktopConnectionBrokerRedirectorCounterset) GetPropertyThreadswaitingforRPCContext() (value uint64, err error) {
	retValue, err := instance.GetProperty("ThreadswaitingforRPCContext")
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
