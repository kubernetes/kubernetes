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

// Win32_PerfFormattedData_Counters_HTTPServiceUrlGroups struct
type Win32_PerfFormattedData_Counters_HTTPServiceUrlGroups struct {
	*Win32_PerfFormattedData

	//
	AllRequests uint32

	//
	BytesReceivedRate uint64

	//
	BytesSentRate uint64

	//
	BytesTransferredRate uint64

	//
	ConnectionAttempts uint32

	//
	CurrentConnections uint32

	//
	GetRequests uint32

	//
	HeadRequests uint32

	//
	MaxConnections uint32
}

func NewWin32_PerfFormattedData_Counters_HTTPServiceUrlGroupsEx1(instance *cim.WmiInstance) (newInstance *Win32_PerfFormattedData_Counters_HTTPServiceUrlGroups, err error) {
	tmp, err := NewWin32_PerfFormattedDataEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfFormattedData_Counters_HTTPServiceUrlGroups{
		Win32_PerfFormattedData: tmp,
	}
	return
}

func NewWin32_PerfFormattedData_Counters_HTTPServiceUrlGroupsEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_PerfFormattedData_Counters_HTTPServiceUrlGroups, err error) {
	tmp, err := NewWin32_PerfFormattedDataEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfFormattedData_Counters_HTTPServiceUrlGroups{
		Win32_PerfFormattedData: tmp,
	}
	return
}

// SetAllRequests sets the value of AllRequests for the instance
func (instance *Win32_PerfFormattedData_Counters_HTTPServiceUrlGroups) SetPropertyAllRequests(value uint32) (err error) {
	return instance.SetProperty("AllRequests", (value))
}

// GetAllRequests gets the value of AllRequests for the instance
func (instance *Win32_PerfFormattedData_Counters_HTTPServiceUrlGroups) GetPropertyAllRequests() (value uint32, err error) {
	retValue, err := instance.GetProperty("AllRequests")
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

// SetBytesReceivedRate sets the value of BytesReceivedRate for the instance
func (instance *Win32_PerfFormattedData_Counters_HTTPServiceUrlGroups) SetPropertyBytesReceivedRate(value uint64) (err error) {
	return instance.SetProperty("BytesReceivedRate", (value))
}

// GetBytesReceivedRate gets the value of BytesReceivedRate for the instance
func (instance *Win32_PerfFormattedData_Counters_HTTPServiceUrlGroups) GetPropertyBytesReceivedRate() (value uint64, err error) {
	retValue, err := instance.GetProperty("BytesReceivedRate")
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

// SetBytesSentRate sets the value of BytesSentRate for the instance
func (instance *Win32_PerfFormattedData_Counters_HTTPServiceUrlGroups) SetPropertyBytesSentRate(value uint64) (err error) {
	return instance.SetProperty("BytesSentRate", (value))
}

// GetBytesSentRate gets the value of BytesSentRate for the instance
func (instance *Win32_PerfFormattedData_Counters_HTTPServiceUrlGroups) GetPropertyBytesSentRate() (value uint64, err error) {
	retValue, err := instance.GetProperty("BytesSentRate")
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

// SetBytesTransferredRate sets the value of BytesTransferredRate for the instance
func (instance *Win32_PerfFormattedData_Counters_HTTPServiceUrlGroups) SetPropertyBytesTransferredRate(value uint64) (err error) {
	return instance.SetProperty("BytesTransferredRate", (value))
}

// GetBytesTransferredRate gets the value of BytesTransferredRate for the instance
func (instance *Win32_PerfFormattedData_Counters_HTTPServiceUrlGroups) GetPropertyBytesTransferredRate() (value uint64, err error) {
	retValue, err := instance.GetProperty("BytesTransferredRate")
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

// SetConnectionAttempts sets the value of ConnectionAttempts for the instance
func (instance *Win32_PerfFormattedData_Counters_HTTPServiceUrlGroups) SetPropertyConnectionAttempts(value uint32) (err error) {
	return instance.SetProperty("ConnectionAttempts", (value))
}

// GetConnectionAttempts gets the value of ConnectionAttempts for the instance
func (instance *Win32_PerfFormattedData_Counters_HTTPServiceUrlGroups) GetPropertyConnectionAttempts() (value uint32, err error) {
	retValue, err := instance.GetProperty("ConnectionAttempts")
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

// SetCurrentConnections sets the value of CurrentConnections for the instance
func (instance *Win32_PerfFormattedData_Counters_HTTPServiceUrlGroups) SetPropertyCurrentConnections(value uint32) (err error) {
	return instance.SetProperty("CurrentConnections", (value))
}

// GetCurrentConnections gets the value of CurrentConnections for the instance
func (instance *Win32_PerfFormattedData_Counters_HTTPServiceUrlGroups) GetPropertyCurrentConnections() (value uint32, err error) {
	retValue, err := instance.GetProperty("CurrentConnections")
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

// SetGetRequests sets the value of GetRequests for the instance
func (instance *Win32_PerfFormattedData_Counters_HTTPServiceUrlGroups) SetPropertyGetRequests(value uint32) (err error) {
	return instance.SetProperty("GetRequests", (value))
}

// GetGetRequests gets the value of GetRequests for the instance
func (instance *Win32_PerfFormattedData_Counters_HTTPServiceUrlGroups) GetPropertyGetRequests() (value uint32, err error) {
	retValue, err := instance.GetProperty("GetRequests")
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

// SetHeadRequests sets the value of HeadRequests for the instance
func (instance *Win32_PerfFormattedData_Counters_HTTPServiceUrlGroups) SetPropertyHeadRequests(value uint32) (err error) {
	return instance.SetProperty("HeadRequests", (value))
}

// GetHeadRequests gets the value of HeadRequests for the instance
func (instance *Win32_PerfFormattedData_Counters_HTTPServiceUrlGroups) GetPropertyHeadRequests() (value uint32, err error) {
	retValue, err := instance.GetProperty("HeadRequests")
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

// SetMaxConnections sets the value of MaxConnections for the instance
func (instance *Win32_PerfFormattedData_Counters_HTTPServiceUrlGroups) SetPropertyMaxConnections(value uint32) (err error) {
	return instance.SetProperty("MaxConnections", (value))
}

// GetMaxConnections gets the value of MaxConnections for the instance
func (instance *Win32_PerfFormattedData_Counters_HTTPServiceUrlGroups) GetPropertyMaxConnections() (value uint32, err error) {
	retValue, err := instance.GetProperty("MaxConnections")
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
