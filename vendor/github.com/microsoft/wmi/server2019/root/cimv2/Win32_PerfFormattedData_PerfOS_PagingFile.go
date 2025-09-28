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

// Win32_PerfFormattedData_PerfOS_PagingFile struct
type Win32_PerfFormattedData_PerfOS_PagingFile struct {
	*Win32_PerfFormattedData

	//
	PercentUsage uint32

	//
	PercentUsagePeak uint32
}

func NewWin32_PerfFormattedData_PerfOS_PagingFileEx1(instance *cim.WmiInstance) (newInstance *Win32_PerfFormattedData_PerfOS_PagingFile, err error) {
	tmp, err := NewWin32_PerfFormattedDataEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfFormattedData_PerfOS_PagingFile{
		Win32_PerfFormattedData: tmp,
	}
	return
}

func NewWin32_PerfFormattedData_PerfOS_PagingFileEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_PerfFormattedData_PerfOS_PagingFile, err error) {
	tmp, err := NewWin32_PerfFormattedDataEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfFormattedData_PerfOS_PagingFile{
		Win32_PerfFormattedData: tmp,
	}
	return
}

// SetPercentUsage sets the value of PercentUsage for the instance
func (instance *Win32_PerfFormattedData_PerfOS_PagingFile) SetPropertyPercentUsage(value uint32) (err error) {
	return instance.SetProperty("PercentUsage", (value))
}

// GetPercentUsage gets the value of PercentUsage for the instance
func (instance *Win32_PerfFormattedData_PerfOS_PagingFile) GetPropertyPercentUsage() (value uint32, err error) {
	retValue, err := instance.GetProperty("PercentUsage")
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

// SetPercentUsagePeak sets the value of PercentUsagePeak for the instance
func (instance *Win32_PerfFormattedData_PerfOS_PagingFile) SetPropertyPercentUsagePeak(value uint32) (err error) {
	return instance.SetProperty("PercentUsagePeak", (value))
}

// GetPercentUsagePeak gets the value of PercentUsagePeak for the instance
func (instance *Win32_PerfFormattedData_PerfOS_PagingFile) GetPropertyPercentUsagePeak() (value uint32, err error) {
	retValue, err := instance.GetProperty("PercentUsagePeak")
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
