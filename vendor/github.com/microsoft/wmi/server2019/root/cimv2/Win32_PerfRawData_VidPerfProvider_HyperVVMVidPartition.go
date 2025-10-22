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

// Win32_PerfRawData_VidPerfProvider_HyperVVMVidPartition struct
type Win32_PerfRawData_VidPerfProvider_HyperVVMVidPartition struct {
	*Win32_PerfRawData

	//
	PhysicalPagesAllocated uint64

	//
	PreferredNUMANodeIndex uint64

	//
	RemotePhysicalPages uint64
}

func NewWin32_PerfRawData_VidPerfProvider_HyperVVMVidPartitionEx1(instance *cim.WmiInstance) (newInstance *Win32_PerfRawData_VidPerfProvider_HyperVVMVidPartition, err error) {
	tmp, err := NewWin32_PerfRawDataEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfRawData_VidPerfProvider_HyperVVMVidPartition{
		Win32_PerfRawData: tmp,
	}
	return
}

func NewWin32_PerfRawData_VidPerfProvider_HyperVVMVidPartitionEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_PerfRawData_VidPerfProvider_HyperVVMVidPartition, err error) {
	tmp, err := NewWin32_PerfRawDataEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfRawData_VidPerfProvider_HyperVVMVidPartition{
		Win32_PerfRawData: tmp,
	}
	return
}

// SetPhysicalPagesAllocated sets the value of PhysicalPagesAllocated for the instance
func (instance *Win32_PerfRawData_VidPerfProvider_HyperVVMVidPartition) SetPropertyPhysicalPagesAllocated(value uint64) (err error) {
	return instance.SetProperty("PhysicalPagesAllocated", (value))
}

// GetPhysicalPagesAllocated gets the value of PhysicalPagesAllocated for the instance
func (instance *Win32_PerfRawData_VidPerfProvider_HyperVVMVidPartition) GetPropertyPhysicalPagesAllocated() (value uint64, err error) {
	retValue, err := instance.GetProperty("PhysicalPagesAllocated")
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

// SetPreferredNUMANodeIndex sets the value of PreferredNUMANodeIndex for the instance
func (instance *Win32_PerfRawData_VidPerfProvider_HyperVVMVidPartition) SetPropertyPreferredNUMANodeIndex(value uint64) (err error) {
	return instance.SetProperty("PreferredNUMANodeIndex", (value))
}

// GetPreferredNUMANodeIndex gets the value of PreferredNUMANodeIndex for the instance
func (instance *Win32_PerfRawData_VidPerfProvider_HyperVVMVidPartition) GetPropertyPreferredNUMANodeIndex() (value uint64, err error) {
	retValue, err := instance.GetProperty("PreferredNUMANodeIndex")
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

// SetRemotePhysicalPages sets the value of RemotePhysicalPages for the instance
func (instance *Win32_PerfRawData_VidPerfProvider_HyperVVMVidPartition) SetPropertyRemotePhysicalPages(value uint64) (err error) {
	return instance.SetProperty("RemotePhysicalPages", (value))
}

// GetRemotePhysicalPages gets the value of RemotePhysicalPages for the instance
func (instance *Win32_PerfRawData_VidPerfProvider_HyperVVMVidPartition) GetPropertyRemotePhysicalPages() (value uint64, err error) {
	retValue, err := instance.GetProperty("RemotePhysicalPages")
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
