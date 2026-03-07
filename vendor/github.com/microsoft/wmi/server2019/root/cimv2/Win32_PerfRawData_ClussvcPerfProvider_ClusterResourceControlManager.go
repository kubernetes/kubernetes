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

// Win32_PerfRawData_ClussvcPerfProvider_ClusterResourceControlManager struct
type Win32_PerfRawData_ClussvcPerfProvider_ClusterResourceControlManager struct {
	*Win32_PerfRawData

	//
	GroupsOnline uint64

	//
	RHSProcesses uint64

	//
	RHSRestarts uint64
}

func NewWin32_PerfRawData_ClussvcPerfProvider_ClusterResourceControlManagerEx1(instance *cim.WmiInstance) (newInstance *Win32_PerfRawData_ClussvcPerfProvider_ClusterResourceControlManager, err error) {
	tmp, err := NewWin32_PerfRawDataEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfRawData_ClussvcPerfProvider_ClusterResourceControlManager{
		Win32_PerfRawData: tmp,
	}
	return
}

func NewWin32_PerfRawData_ClussvcPerfProvider_ClusterResourceControlManagerEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_PerfRawData_ClussvcPerfProvider_ClusterResourceControlManager, err error) {
	tmp, err := NewWin32_PerfRawDataEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfRawData_ClussvcPerfProvider_ClusterResourceControlManager{
		Win32_PerfRawData: tmp,
	}
	return
}

// SetGroupsOnline sets the value of GroupsOnline for the instance
func (instance *Win32_PerfRawData_ClussvcPerfProvider_ClusterResourceControlManager) SetPropertyGroupsOnline(value uint64) (err error) {
	return instance.SetProperty("GroupsOnline", (value))
}

// GetGroupsOnline gets the value of GroupsOnline for the instance
func (instance *Win32_PerfRawData_ClussvcPerfProvider_ClusterResourceControlManager) GetPropertyGroupsOnline() (value uint64, err error) {
	retValue, err := instance.GetProperty("GroupsOnline")
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

// SetRHSProcesses sets the value of RHSProcesses for the instance
func (instance *Win32_PerfRawData_ClussvcPerfProvider_ClusterResourceControlManager) SetPropertyRHSProcesses(value uint64) (err error) {
	return instance.SetProperty("RHSProcesses", (value))
}

// GetRHSProcesses gets the value of RHSProcesses for the instance
func (instance *Win32_PerfRawData_ClussvcPerfProvider_ClusterResourceControlManager) GetPropertyRHSProcesses() (value uint64, err error) {
	retValue, err := instance.GetProperty("RHSProcesses")
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

// SetRHSRestarts sets the value of RHSRestarts for the instance
func (instance *Win32_PerfRawData_ClussvcPerfProvider_ClusterResourceControlManager) SetPropertyRHSRestarts(value uint64) (err error) {
	return instance.SetProperty("RHSRestarts", (value))
}

// GetRHSRestarts gets the value of RHSRestarts for the instance
func (instance *Win32_PerfRawData_ClussvcPerfProvider_ClusterResourceControlManager) GetPropertyRHSRestarts() (value uint64, err error) {
	retValue, err := instance.GetProperty("RHSRestarts")
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
