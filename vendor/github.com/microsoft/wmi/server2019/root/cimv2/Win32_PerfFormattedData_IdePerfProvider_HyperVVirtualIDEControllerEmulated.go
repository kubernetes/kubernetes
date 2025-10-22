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

// Win32_PerfFormattedData_IdePerfProvider_HyperVVirtualIDEControllerEmulated struct
type Win32_PerfFormattedData_IdePerfProvider_HyperVVirtualIDEControllerEmulated struct {
	*Win32_PerfFormattedData

	//
	ReadBytesPersec uint64

	//
	ReadSectorsPersec uint64

	//
	WriteBytesPersec uint64

	//
	WrittenSectorsPersec uint64
}

func NewWin32_PerfFormattedData_IdePerfProvider_HyperVVirtualIDEControllerEmulatedEx1(instance *cim.WmiInstance) (newInstance *Win32_PerfFormattedData_IdePerfProvider_HyperVVirtualIDEControllerEmulated, err error) {
	tmp, err := NewWin32_PerfFormattedDataEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfFormattedData_IdePerfProvider_HyperVVirtualIDEControllerEmulated{
		Win32_PerfFormattedData: tmp,
	}
	return
}

func NewWin32_PerfFormattedData_IdePerfProvider_HyperVVirtualIDEControllerEmulatedEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_PerfFormattedData_IdePerfProvider_HyperVVirtualIDEControllerEmulated, err error) {
	tmp, err := NewWin32_PerfFormattedDataEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfFormattedData_IdePerfProvider_HyperVVirtualIDEControllerEmulated{
		Win32_PerfFormattedData: tmp,
	}
	return
}

// SetReadBytesPersec sets the value of ReadBytesPersec for the instance
func (instance *Win32_PerfFormattedData_IdePerfProvider_HyperVVirtualIDEControllerEmulated) SetPropertyReadBytesPersec(value uint64) (err error) {
	return instance.SetProperty("ReadBytesPersec", (value))
}

// GetReadBytesPersec gets the value of ReadBytesPersec for the instance
func (instance *Win32_PerfFormattedData_IdePerfProvider_HyperVVirtualIDEControllerEmulated) GetPropertyReadBytesPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("ReadBytesPersec")
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

// SetReadSectorsPersec sets the value of ReadSectorsPersec for the instance
func (instance *Win32_PerfFormattedData_IdePerfProvider_HyperVVirtualIDEControllerEmulated) SetPropertyReadSectorsPersec(value uint64) (err error) {
	return instance.SetProperty("ReadSectorsPersec", (value))
}

// GetReadSectorsPersec gets the value of ReadSectorsPersec for the instance
func (instance *Win32_PerfFormattedData_IdePerfProvider_HyperVVirtualIDEControllerEmulated) GetPropertyReadSectorsPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("ReadSectorsPersec")
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

// SetWriteBytesPersec sets the value of WriteBytesPersec for the instance
func (instance *Win32_PerfFormattedData_IdePerfProvider_HyperVVirtualIDEControllerEmulated) SetPropertyWriteBytesPersec(value uint64) (err error) {
	return instance.SetProperty("WriteBytesPersec", (value))
}

// GetWriteBytesPersec gets the value of WriteBytesPersec for the instance
func (instance *Win32_PerfFormattedData_IdePerfProvider_HyperVVirtualIDEControllerEmulated) GetPropertyWriteBytesPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("WriteBytesPersec")
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

// SetWrittenSectorsPersec sets the value of WrittenSectorsPersec for the instance
func (instance *Win32_PerfFormattedData_IdePerfProvider_HyperVVirtualIDEControllerEmulated) SetPropertyWrittenSectorsPersec(value uint64) (err error) {
	return instance.SetProperty("WrittenSectorsPersec", (value))
}

// GetWrittenSectorsPersec gets the value of WrittenSectorsPersec for the instance
func (instance *Win32_PerfFormattedData_IdePerfProvider_HyperVVirtualIDEControllerEmulated) GetPropertyWrittenSectorsPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("WrittenSectorsPersec")
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
