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

// Win32_PerfRawData_Counters_HyperVVirtualMachineBusPipes struct
type Win32_PerfRawData_Counters_HyperVVirtualMachineBusPipes struct {
	*Win32_PerfRawData

	//
	BytesReadPersec uint64

	//
	BytesWrittenPersec uint64

	//
	ReadsPersec uint64

	//
	WritesPersec uint64
}

func NewWin32_PerfRawData_Counters_HyperVVirtualMachineBusPipesEx1(instance *cim.WmiInstance) (newInstance *Win32_PerfRawData_Counters_HyperVVirtualMachineBusPipes, err error) {
	tmp, err := NewWin32_PerfRawDataEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfRawData_Counters_HyperVVirtualMachineBusPipes{
		Win32_PerfRawData: tmp,
	}
	return
}

func NewWin32_PerfRawData_Counters_HyperVVirtualMachineBusPipesEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_PerfRawData_Counters_HyperVVirtualMachineBusPipes, err error) {
	tmp, err := NewWin32_PerfRawDataEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfRawData_Counters_HyperVVirtualMachineBusPipes{
		Win32_PerfRawData: tmp,
	}
	return
}

// SetBytesReadPersec sets the value of BytesReadPersec for the instance
func (instance *Win32_PerfRawData_Counters_HyperVVirtualMachineBusPipes) SetPropertyBytesReadPersec(value uint64) (err error) {
	return instance.SetProperty("BytesReadPersec", (value))
}

// GetBytesReadPersec gets the value of BytesReadPersec for the instance
func (instance *Win32_PerfRawData_Counters_HyperVVirtualMachineBusPipes) GetPropertyBytesReadPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("BytesReadPersec")
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

// SetBytesWrittenPersec sets the value of BytesWrittenPersec for the instance
func (instance *Win32_PerfRawData_Counters_HyperVVirtualMachineBusPipes) SetPropertyBytesWrittenPersec(value uint64) (err error) {
	return instance.SetProperty("BytesWrittenPersec", (value))
}

// GetBytesWrittenPersec gets the value of BytesWrittenPersec for the instance
func (instance *Win32_PerfRawData_Counters_HyperVVirtualMachineBusPipes) GetPropertyBytesWrittenPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("BytesWrittenPersec")
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

// SetReadsPersec sets the value of ReadsPersec for the instance
func (instance *Win32_PerfRawData_Counters_HyperVVirtualMachineBusPipes) SetPropertyReadsPersec(value uint64) (err error) {
	return instance.SetProperty("ReadsPersec", (value))
}

// GetReadsPersec gets the value of ReadsPersec for the instance
func (instance *Win32_PerfRawData_Counters_HyperVVirtualMachineBusPipes) GetPropertyReadsPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("ReadsPersec")
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

// SetWritesPersec sets the value of WritesPersec for the instance
func (instance *Win32_PerfRawData_Counters_HyperVVirtualMachineBusPipes) SetPropertyWritesPersec(value uint64) (err error) {
	return instance.SetProperty("WritesPersec", (value))
}

// GetWritesPersec gets the value of WritesPersec for the instance
func (instance *Win32_PerfRawData_Counters_HyperVVirtualMachineBusPipes) GetPropertyWritesPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("WritesPersec")
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
