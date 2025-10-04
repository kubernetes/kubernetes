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

// Win32_PerfFormattedData_CsvNfltPerfProvider_ClusterCSVBlockRedirection struct
type Win32_PerfFormattedData_CsvNfltPerfProvider_ClusterCSVBlockRedirection struct {
	*Win32_PerfFormattedData

	//
	IOReadBytes uint64

	//
	IOReadBytesPersec uint64

	//
	IOReads uint64

	//
	IOReadsPersec uint64

	//
	IOWriteBytes uint64

	//
	IOWriteBytesPersec uint64

	//
	IOWrites uint64

	//
	IOWritesPersec uint64
}

func NewWin32_PerfFormattedData_CsvNfltPerfProvider_ClusterCSVBlockRedirectionEx1(instance *cim.WmiInstance) (newInstance *Win32_PerfFormattedData_CsvNfltPerfProvider_ClusterCSVBlockRedirection, err error) {
	tmp, err := NewWin32_PerfFormattedDataEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfFormattedData_CsvNfltPerfProvider_ClusterCSVBlockRedirection{
		Win32_PerfFormattedData: tmp,
	}
	return
}

func NewWin32_PerfFormattedData_CsvNfltPerfProvider_ClusterCSVBlockRedirectionEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_PerfFormattedData_CsvNfltPerfProvider_ClusterCSVBlockRedirection, err error) {
	tmp, err := NewWin32_PerfFormattedDataEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfFormattedData_CsvNfltPerfProvider_ClusterCSVBlockRedirection{
		Win32_PerfFormattedData: tmp,
	}
	return
}

// SetIOReadBytes sets the value of IOReadBytes for the instance
func (instance *Win32_PerfFormattedData_CsvNfltPerfProvider_ClusterCSVBlockRedirection) SetPropertyIOReadBytes(value uint64) (err error) {
	return instance.SetProperty("IOReadBytes", (value))
}

// GetIOReadBytes gets the value of IOReadBytes for the instance
func (instance *Win32_PerfFormattedData_CsvNfltPerfProvider_ClusterCSVBlockRedirection) GetPropertyIOReadBytes() (value uint64, err error) {
	retValue, err := instance.GetProperty("IOReadBytes")
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

// SetIOReadBytesPersec sets the value of IOReadBytesPersec for the instance
func (instance *Win32_PerfFormattedData_CsvNfltPerfProvider_ClusterCSVBlockRedirection) SetPropertyIOReadBytesPersec(value uint64) (err error) {
	return instance.SetProperty("IOReadBytesPersec", (value))
}

// GetIOReadBytesPersec gets the value of IOReadBytesPersec for the instance
func (instance *Win32_PerfFormattedData_CsvNfltPerfProvider_ClusterCSVBlockRedirection) GetPropertyIOReadBytesPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("IOReadBytesPersec")
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

// SetIOReads sets the value of IOReads for the instance
func (instance *Win32_PerfFormattedData_CsvNfltPerfProvider_ClusterCSVBlockRedirection) SetPropertyIOReads(value uint64) (err error) {
	return instance.SetProperty("IOReads", (value))
}

// GetIOReads gets the value of IOReads for the instance
func (instance *Win32_PerfFormattedData_CsvNfltPerfProvider_ClusterCSVBlockRedirection) GetPropertyIOReads() (value uint64, err error) {
	retValue, err := instance.GetProperty("IOReads")
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

// SetIOReadsPersec sets the value of IOReadsPersec for the instance
func (instance *Win32_PerfFormattedData_CsvNfltPerfProvider_ClusterCSVBlockRedirection) SetPropertyIOReadsPersec(value uint64) (err error) {
	return instance.SetProperty("IOReadsPersec", (value))
}

// GetIOReadsPersec gets the value of IOReadsPersec for the instance
func (instance *Win32_PerfFormattedData_CsvNfltPerfProvider_ClusterCSVBlockRedirection) GetPropertyIOReadsPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("IOReadsPersec")
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

// SetIOWriteBytes sets the value of IOWriteBytes for the instance
func (instance *Win32_PerfFormattedData_CsvNfltPerfProvider_ClusterCSVBlockRedirection) SetPropertyIOWriteBytes(value uint64) (err error) {
	return instance.SetProperty("IOWriteBytes", (value))
}

// GetIOWriteBytes gets the value of IOWriteBytes for the instance
func (instance *Win32_PerfFormattedData_CsvNfltPerfProvider_ClusterCSVBlockRedirection) GetPropertyIOWriteBytes() (value uint64, err error) {
	retValue, err := instance.GetProperty("IOWriteBytes")
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

// SetIOWriteBytesPersec sets the value of IOWriteBytesPersec for the instance
func (instance *Win32_PerfFormattedData_CsvNfltPerfProvider_ClusterCSVBlockRedirection) SetPropertyIOWriteBytesPersec(value uint64) (err error) {
	return instance.SetProperty("IOWriteBytesPersec", (value))
}

// GetIOWriteBytesPersec gets the value of IOWriteBytesPersec for the instance
func (instance *Win32_PerfFormattedData_CsvNfltPerfProvider_ClusterCSVBlockRedirection) GetPropertyIOWriteBytesPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("IOWriteBytesPersec")
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

// SetIOWrites sets the value of IOWrites for the instance
func (instance *Win32_PerfFormattedData_CsvNfltPerfProvider_ClusterCSVBlockRedirection) SetPropertyIOWrites(value uint64) (err error) {
	return instance.SetProperty("IOWrites", (value))
}

// GetIOWrites gets the value of IOWrites for the instance
func (instance *Win32_PerfFormattedData_CsvNfltPerfProvider_ClusterCSVBlockRedirection) GetPropertyIOWrites() (value uint64, err error) {
	retValue, err := instance.GetProperty("IOWrites")
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

// SetIOWritesPersec sets the value of IOWritesPersec for the instance
func (instance *Win32_PerfFormattedData_CsvNfltPerfProvider_ClusterCSVBlockRedirection) SetPropertyIOWritesPersec(value uint64) (err error) {
	return instance.SetProperty("IOWritesPersec", (value))
}

// GetIOWritesPersec gets the value of IOWritesPersec for the instance
func (instance *Win32_PerfFormattedData_CsvNfltPerfProvider_ClusterCSVBlockRedirection) GetPropertyIOWritesPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("IOWritesPersec")
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
