// Copyright 2019 (c) Microsoft Corporation.
// Licensed under the MIT license.

//
// Author:
//      Auto Generated on 3/19/2020 using wmigen
//      Source root.CIMV2
//////////////////////////////////////////////
package cimv2

import (
	"github.com/microsoft/wmi/pkg/base/query"
	cim "github.com/microsoft/wmi/pkg/wmiinstance"
)

// Win32_PerfRawData_MmPerfProvider_HyperVVMWorkerProcessMemoryManager struct
type Win32_PerfRawData_MmPerfProvider_HyperVVMWorkerProcessMemoryManager struct {
	*Win32_PerfRawData

	//
	MemoryBlockCount uint64
}

func NewWin32_PerfRawData_MmPerfProvider_HyperVVMWorkerProcessMemoryManagerEx1(instance *cim.WmiInstance) (newInstance *Win32_PerfRawData_MmPerfProvider_HyperVVMWorkerProcessMemoryManager, err error) {
	tmp, err := NewWin32_PerfRawDataEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfRawData_MmPerfProvider_HyperVVMWorkerProcessMemoryManager{
		Win32_PerfRawData: tmp,
	}
	return
}

func NewWin32_PerfRawData_MmPerfProvider_HyperVVMWorkerProcessMemoryManagerEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_PerfRawData_MmPerfProvider_HyperVVMWorkerProcessMemoryManager, err error) {
	tmp, err := NewWin32_PerfRawDataEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfRawData_MmPerfProvider_HyperVVMWorkerProcessMemoryManager{
		Win32_PerfRawData: tmp,
	}
	return
}

// SetMemoryBlockCount sets the value of MemoryBlockCount for the instance
func (instance *Win32_PerfRawData_MmPerfProvider_HyperVVMWorkerProcessMemoryManager) SetPropertyMemoryBlockCount(value uint64) (err error) {
	return instance.SetProperty("MemoryBlockCount", value)
}

// GetMemoryBlockCount gets the value of MemoryBlockCount for the instance
func (instance *Win32_PerfRawData_MmPerfProvider_HyperVVMWorkerProcessMemoryManager) GetPropertyMemoryBlockCount() (value uint64, err error) {
	retValue, err := instance.GetProperty("MemoryBlockCount")
	if err != nil {
		return
	}
	value, ok := retValue.(uint64)
	if !ok {
		// TODO: Set an error
	}
	return
}
