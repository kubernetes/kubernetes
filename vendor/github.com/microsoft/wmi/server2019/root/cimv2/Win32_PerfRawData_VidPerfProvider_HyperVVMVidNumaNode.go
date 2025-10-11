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

// Win32_PerfRawData_VidPerfProvider_HyperVVMVidNumaNode struct
type Win32_PerfRawData_VidPerfProvider_HyperVVMVidNumaNode struct {
	*Win32_PerfRawData

	//
	PageCount uint64

	//
	ProcessorCount uint64
}

func NewWin32_PerfRawData_VidPerfProvider_HyperVVMVidNumaNodeEx1(instance *cim.WmiInstance) (newInstance *Win32_PerfRawData_VidPerfProvider_HyperVVMVidNumaNode, err error) {
	tmp, err := NewWin32_PerfRawDataEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfRawData_VidPerfProvider_HyperVVMVidNumaNode{
		Win32_PerfRawData: tmp,
	}
	return
}

func NewWin32_PerfRawData_VidPerfProvider_HyperVVMVidNumaNodeEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_PerfRawData_VidPerfProvider_HyperVVMVidNumaNode, err error) {
	tmp, err := NewWin32_PerfRawDataEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfRawData_VidPerfProvider_HyperVVMVidNumaNode{
		Win32_PerfRawData: tmp,
	}
	return
}

// SetPageCount sets the value of PageCount for the instance
func (instance *Win32_PerfRawData_VidPerfProvider_HyperVVMVidNumaNode) SetPropertyPageCount(value uint64) (err error) {
	return instance.SetProperty("PageCount", (value))
}

// GetPageCount gets the value of PageCount for the instance
func (instance *Win32_PerfRawData_VidPerfProvider_HyperVVMVidNumaNode) GetPropertyPageCount() (value uint64, err error) {
	retValue, err := instance.GetProperty("PageCount")
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

// SetProcessorCount sets the value of ProcessorCount for the instance
func (instance *Win32_PerfRawData_VidPerfProvider_HyperVVMVidNumaNode) SetPropertyProcessorCount(value uint64) (err error) {
	return instance.SetProperty("ProcessorCount", (value))
}

// GetProcessorCount gets the value of ProcessorCount for the instance
func (instance *Win32_PerfRawData_VidPerfProvider_HyperVVMVidNumaNode) GetPropertyProcessorCount() (value uint64, err error) {
	retValue, err := instance.GetProperty("ProcessorCount")
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
