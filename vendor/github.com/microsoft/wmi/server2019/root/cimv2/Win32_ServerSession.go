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

// Win32_ServerSession struct
type Win32_ServerSession struct {
	*CIM_LogicalElement

	//
	ActiveTime uint32

	//
	ClientType string

	//
	ComputerName string

	//
	IdleTime uint32

	//
	ResourcesOpened uint32

	//
	SessionType uint32

	//
	TransportName string

	//
	UserName string
}

func NewWin32_ServerSessionEx1(instance *cim.WmiInstance) (newInstance *Win32_ServerSession, err error) {
	tmp, err := NewCIM_LogicalElementEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_ServerSession{
		CIM_LogicalElement: tmp,
	}
	return
}

func NewWin32_ServerSessionEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_ServerSession, err error) {
	tmp, err := NewCIM_LogicalElementEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_ServerSession{
		CIM_LogicalElement: tmp,
	}
	return
}

// SetActiveTime sets the value of ActiveTime for the instance
func (instance *Win32_ServerSession) SetPropertyActiveTime(value uint32) (err error) {
	return instance.SetProperty("ActiveTime", (value))
}

// GetActiveTime gets the value of ActiveTime for the instance
func (instance *Win32_ServerSession) GetPropertyActiveTime() (value uint32, err error) {
	retValue, err := instance.GetProperty("ActiveTime")
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

// SetClientType sets the value of ClientType for the instance
func (instance *Win32_ServerSession) SetPropertyClientType(value string) (err error) {
	return instance.SetProperty("ClientType", (value))
}

// GetClientType gets the value of ClientType for the instance
func (instance *Win32_ServerSession) GetPropertyClientType() (value string, err error) {
	retValue, err := instance.GetProperty("ClientType")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(string)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " string is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = string(valuetmp)

	return
}

// SetComputerName sets the value of ComputerName for the instance
func (instance *Win32_ServerSession) SetPropertyComputerName(value string) (err error) {
	return instance.SetProperty("ComputerName", (value))
}

// GetComputerName gets the value of ComputerName for the instance
func (instance *Win32_ServerSession) GetPropertyComputerName() (value string, err error) {
	retValue, err := instance.GetProperty("ComputerName")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(string)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " string is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = string(valuetmp)

	return
}

// SetIdleTime sets the value of IdleTime for the instance
func (instance *Win32_ServerSession) SetPropertyIdleTime(value uint32) (err error) {
	return instance.SetProperty("IdleTime", (value))
}

// GetIdleTime gets the value of IdleTime for the instance
func (instance *Win32_ServerSession) GetPropertyIdleTime() (value uint32, err error) {
	retValue, err := instance.GetProperty("IdleTime")
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

// SetResourcesOpened sets the value of ResourcesOpened for the instance
func (instance *Win32_ServerSession) SetPropertyResourcesOpened(value uint32) (err error) {
	return instance.SetProperty("ResourcesOpened", (value))
}

// GetResourcesOpened gets the value of ResourcesOpened for the instance
func (instance *Win32_ServerSession) GetPropertyResourcesOpened() (value uint32, err error) {
	retValue, err := instance.GetProperty("ResourcesOpened")
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

// SetSessionType sets the value of SessionType for the instance
func (instance *Win32_ServerSession) SetPropertySessionType(value uint32) (err error) {
	return instance.SetProperty("SessionType", (value))
}

// GetSessionType gets the value of SessionType for the instance
func (instance *Win32_ServerSession) GetPropertySessionType() (value uint32, err error) {
	retValue, err := instance.GetProperty("SessionType")
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

// SetTransportName sets the value of TransportName for the instance
func (instance *Win32_ServerSession) SetPropertyTransportName(value string) (err error) {
	return instance.SetProperty("TransportName", (value))
}

// GetTransportName gets the value of TransportName for the instance
func (instance *Win32_ServerSession) GetPropertyTransportName() (value string, err error) {
	retValue, err := instance.GetProperty("TransportName")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(string)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " string is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = string(valuetmp)

	return
}

// SetUserName sets the value of UserName for the instance
func (instance *Win32_ServerSession) SetPropertyUserName(value string) (err error) {
	return instance.SetProperty("UserName", (value))
}

// GetUserName gets the value of UserName for the instance
func (instance *Win32_ServerSession) GetPropertyUserName() (value string, err error) {
	retValue, err := instance.GetProperty("UserName")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(string)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " string is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = string(valuetmp)

	return
}
