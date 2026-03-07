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

// Win32_PerfRawData_GmoPerfProvider_HyperVVMSaveSnapshotandRestore struct
type Win32_PerfRawData_GmoPerfProvider_HyperVVMSaveSnapshotandRestore struct {
	*Win32_PerfRawData

	//
	OperationTime uint32

	//
	RequestsActive uint32

	//
	RequestsDispatched uint32

	//
	RequestsHighPriority uint32

	//
	RequestsProcessed uint32

	//
	ThreadsSpawned uint32
}

func NewWin32_PerfRawData_GmoPerfProvider_HyperVVMSaveSnapshotandRestoreEx1(instance *cim.WmiInstance) (newInstance *Win32_PerfRawData_GmoPerfProvider_HyperVVMSaveSnapshotandRestore, err error) {
	tmp, err := NewWin32_PerfRawDataEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfRawData_GmoPerfProvider_HyperVVMSaveSnapshotandRestore{
		Win32_PerfRawData: tmp,
	}
	return
}

func NewWin32_PerfRawData_GmoPerfProvider_HyperVVMSaveSnapshotandRestoreEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_PerfRawData_GmoPerfProvider_HyperVVMSaveSnapshotandRestore, err error) {
	tmp, err := NewWin32_PerfRawDataEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfRawData_GmoPerfProvider_HyperVVMSaveSnapshotandRestore{
		Win32_PerfRawData: tmp,
	}
	return
}

// SetOperationTime sets the value of OperationTime for the instance
func (instance *Win32_PerfRawData_GmoPerfProvider_HyperVVMSaveSnapshotandRestore) SetPropertyOperationTime(value uint32) (err error) {
	return instance.SetProperty("OperationTime", (value))
}

// GetOperationTime gets the value of OperationTime for the instance
func (instance *Win32_PerfRawData_GmoPerfProvider_HyperVVMSaveSnapshotandRestore) GetPropertyOperationTime() (value uint32, err error) {
	retValue, err := instance.GetProperty("OperationTime")
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

// SetRequestsActive sets the value of RequestsActive for the instance
func (instance *Win32_PerfRawData_GmoPerfProvider_HyperVVMSaveSnapshotandRestore) SetPropertyRequestsActive(value uint32) (err error) {
	return instance.SetProperty("RequestsActive", (value))
}

// GetRequestsActive gets the value of RequestsActive for the instance
func (instance *Win32_PerfRawData_GmoPerfProvider_HyperVVMSaveSnapshotandRestore) GetPropertyRequestsActive() (value uint32, err error) {
	retValue, err := instance.GetProperty("RequestsActive")
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

// SetRequestsDispatched sets the value of RequestsDispatched for the instance
func (instance *Win32_PerfRawData_GmoPerfProvider_HyperVVMSaveSnapshotandRestore) SetPropertyRequestsDispatched(value uint32) (err error) {
	return instance.SetProperty("RequestsDispatched", (value))
}

// GetRequestsDispatched gets the value of RequestsDispatched for the instance
func (instance *Win32_PerfRawData_GmoPerfProvider_HyperVVMSaveSnapshotandRestore) GetPropertyRequestsDispatched() (value uint32, err error) {
	retValue, err := instance.GetProperty("RequestsDispatched")
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

// SetRequestsHighPriority sets the value of RequestsHighPriority for the instance
func (instance *Win32_PerfRawData_GmoPerfProvider_HyperVVMSaveSnapshotandRestore) SetPropertyRequestsHighPriority(value uint32) (err error) {
	return instance.SetProperty("RequestsHighPriority", (value))
}

// GetRequestsHighPriority gets the value of RequestsHighPriority for the instance
func (instance *Win32_PerfRawData_GmoPerfProvider_HyperVVMSaveSnapshotandRestore) GetPropertyRequestsHighPriority() (value uint32, err error) {
	retValue, err := instance.GetProperty("RequestsHighPriority")
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

// SetRequestsProcessed sets the value of RequestsProcessed for the instance
func (instance *Win32_PerfRawData_GmoPerfProvider_HyperVVMSaveSnapshotandRestore) SetPropertyRequestsProcessed(value uint32) (err error) {
	return instance.SetProperty("RequestsProcessed", (value))
}

// GetRequestsProcessed gets the value of RequestsProcessed for the instance
func (instance *Win32_PerfRawData_GmoPerfProvider_HyperVVMSaveSnapshotandRestore) GetPropertyRequestsProcessed() (value uint32, err error) {
	retValue, err := instance.GetProperty("RequestsProcessed")
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

// SetThreadsSpawned sets the value of ThreadsSpawned for the instance
func (instance *Win32_PerfRawData_GmoPerfProvider_HyperVVMSaveSnapshotandRestore) SetPropertyThreadsSpawned(value uint32) (err error) {
	return instance.SetProperty("ThreadsSpawned", (value))
}

// GetThreadsSpawned gets the value of ThreadsSpawned for the instance
func (instance *Win32_PerfRawData_GmoPerfProvider_HyperVVMSaveSnapshotandRestore) GetPropertyThreadsSpawned() (value uint32, err error) {
	retValue, err := instance.GetProperty("ThreadsSpawned")
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
