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

// Win32_TerminalService struct
type Win32_TerminalService struct {
	*Win32_Service

	//
	DisconnectedSessions uint32

	//
	TotalSessions uint32
}

func NewWin32_TerminalServiceEx1(instance *cim.WmiInstance) (newInstance *Win32_TerminalService, err error) {
	tmp, err := NewWin32_ServiceEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_TerminalService{
		Win32_Service: tmp,
	}
	return
}

func NewWin32_TerminalServiceEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_TerminalService, err error) {
	tmp, err := NewWin32_ServiceEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_TerminalService{
		Win32_Service: tmp,
	}
	return
}

// SetDisconnectedSessions sets the value of DisconnectedSessions for the instance
func (instance *Win32_TerminalService) SetPropertyDisconnectedSessions(value uint32) (err error) {
	return instance.SetProperty("DisconnectedSessions", (value))
}

// GetDisconnectedSessions gets the value of DisconnectedSessions for the instance
func (instance *Win32_TerminalService) GetPropertyDisconnectedSessions() (value uint32, err error) {
	retValue, err := instance.GetProperty("DisconnectedSessions")
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
func (instance *Win32_TerminalService) SetPropertyTotalSessions(value uint32) (err error) {
	return instance.SetProperty("TotalSessions", (value))
}

// GetTotalSessions gets the value of TotalSessions for the instance
func (instance *Win32_TerminalService) GetPropertyTotalSessions() (value uint32, err error) {
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
