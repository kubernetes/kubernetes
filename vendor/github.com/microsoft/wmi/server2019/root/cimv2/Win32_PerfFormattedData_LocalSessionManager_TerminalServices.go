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

// Win32_PerfFormattedData_LocalSessionManager_TerminalServices struct
type Win32_PerfFormattedData_LocalSessionManager_TerminalServices struct {
	*Win32_PerfFormattedData

	//
	ActiveSessions uint32

	//
	InactiveSessions uint32

	//
	TotalSessions uint32
}

func NewWin32_PerfFormattedData_LocalSessionManager_TerminalServicesEx1(instance *cim.WmiInstance) (newInstance *Win32_PerfFormattedData_LocalSessionManager_TerminalServices, err error) {
	tmp, err := NewWin32_PerfFormattedDataEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfFormattedData_LocalSessionManager_TerminalServices{
		Win32_PerfFormattedData: tmp,
	}
	return
}

func NewWin32_PerfFormattedData_LocalSessionManager_TerminalServicesEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_PerfFormattedData_LocalSessionManager_TerminalServices, err error) {
	tmp, err := NewWin32_PerfFormattedDataEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfFormattedData_LocalSessionManager_TerminalServices{
		Win32_PerfFormattedData: tmp,
	}
	return
}

// SetActiveSessions sets the value of ActiveSessions for the instance
func (instance *Win32_PerfFormattedData_LocalSessionManager_TerminalServices) SetPropertyActiveSessions(value uint32) (err error) {
	return instance.SetProperty("ActiveSessions", (value))
}

// GetActiveSessions gets the value of ActiveSessions for the instance
func (instance *Win32_PerfFormattedData_LocalSessionManager_TerminalServices) GetPropertyActiveSessions() (value uint32, err error) {
	retValue, err := instance.GetProperty("ActiveSessions")
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

// SetInactiveSessions sets the value of InactiveSessions for the instance
func (instance *Win32_PerfFormattedData_LocalSessionManager_TerminalServices) SetPropertyInactiveSessions(value uint32) (err error) {
	return instance.SetProperty("InactiveSessions", (value))
}

// GetInactiveSessions gets the value of InactiveSessions for the instance
func (instance *Win32_PerfFormattedData_LocalSessionManager_TerminalServices) GetPropertyInactiveSessions() (value uint32, err error) {
	retValue, err := instance.GetProperty("InactiveSessions")
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

// SetTotalSessions sets the value of TotalSessions for the instance
func (instance *Win32_PerfFormattedData_LocalSessionManager_TerminalServices) SetPropertyTotalSessions(value uint32) (err error) {
	return instance.SetProperty("TotalSessions", (value))
}

// GetTotalSessions gets the value of TotalSessions for the instance
func (instance *Win32_PerfFormattedData_LocalSessionManager_TerminalServices) GetPropertyTotalSessions() (value uint32, err error) {
	retValue, err := instance.GetProperty("TotalSessions")
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
