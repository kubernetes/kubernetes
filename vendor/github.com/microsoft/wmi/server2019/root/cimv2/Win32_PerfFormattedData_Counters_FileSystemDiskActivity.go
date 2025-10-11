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

// Win32_PerfFormattedData_Counters_FileSystemDiskActivity struct
type Win32_PerfFormattedData_Counters_FileSystemDiskActivity struct {
	*Win32_PerfFormattedData

	//
	FileSystemBytesRead uint64

	//
	FileSystemBytesWritten uint64
}

func NewWin32_PerfFormattedData_Counters_FileSystemDiskActivityEx1(instance *cim.WmiInstance) (newInstance *Win32_PerfFormattedData_Counters_FileSystemDiskActivity, err error) {
	tmp, err := NewWin32_PerfFormattedDataEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfFormattedData_Counters_FileSystemDiskActivity{
		Win32_PerfFormattedData: tmp,
	}
	return
}

func NewWin32_PerfFormattedData_Counters_FileSystemDiskActivityEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_PerfFormattedData_Counters_FileSystemDiskActivity, err error) {
	tmp, err := NewWin32_PerfFormattedDataEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfFormattedData_Counters_FileSystemDiskActivity{
		Win32_PerfFormattedData: tmp,
	}
	return
}

// SetFileSystemBytesRead sets the value of FileSystemBytesRead for the instance
func (instance *Win32_PerfFormattedData_Counters_FileSystemDiskActivity) SetPropertyFileSystemBytesRead(value uint64) (err error) {
	return instance.SetProperty("FileSystemBytesRead", (value))
}

// GetFileSystemBytesRead gets the value of FileSystemBytesRead for the instance
func (instance *Win32_PerfFormattedData_Counters_FileSystemDiskActivity) GetPropertyFileSystemBytesRead() (value uint64, err error) {
	retValue, err := instance.GetProperty("FileSystemBytesRead")
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

// SetFileSystemBytesWritten sets the value of FileSystemBytesWritten for the instance
func (instance *Win32_PerfFormattedData_Counters_FileSystemDiskActivity) SetPropertyFileSystemBytesWritten(value uint64) (err error) {
	return instance.SetProperty("FileSystemBytesWritten", (value))
}

// GetFileSystemBytesWritten gets the value of FileSystemBytesWritten for the instance
func (instance *Win32_PerfFormattedData_Counters_FileSystemDiskActivity) GetPropertyFileSystemBytesWritten() (value uint64, err error) {
	retValue, err := instance.GetProperty("FileSystemBytesWritten")
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
