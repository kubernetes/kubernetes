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

// Win32_ServerConnection struct
type Win32_ServerConnection struct {
	*CIM_LogicalElement

	//
	ActiveTime uint32

	//
	ComputerName string

	//
	ConnectionID uint32

	//
	NumberOfFiles uint32

	//
	NumberOfUsers uint32

	//
	ShareName string

	//
	UserName string
}

func NewWin32_ServerConnectionEx1(instance *cim.WmiInstance) (newInstance *Win32_ServerConnection, err error) {
	tmp, err := NewCIM_LogicalElementEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_ServerConnection{
		CIM_LogicalElement: tmp,
	}
	return
}

func NewWin32_ServerConnectionEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_ServerConnection, err error) {
	tmp, err := NewCIM_LogicalElementEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_ServerConnection{
		CIM_LogicalElement: tmp,
	}
	return
}

// SetActiveTime sets the value of ActiveTime for the instance
func (instance *Win32_ServerConnection) SetPropertyActiveTime(value uint32) (err error) {
	return instance.SetProperty("ActiveTime", (value))
}

// GetActiveTime gets the value of ActiveTime for the instance
func (instance *Win32_ServerConnection) GetPropertyActiveTime() (value uint32, err error) {
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

// SetComputerName sets the value of ComputerName for the instance
func (instance *Win32_ServerConnection) SetPropertyComputerName(value string) (err error) {
	return instance.SetProperty("ComputerName", (value))
}

// GetComputerName gets the value of ComputerName for the instance
func (instance *Win32_ServerConnection) GetPropertyComputerName() (value string, err error) {
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

// SetConnectionID sets the value of ConnectionID for the instance
func (instance *Win32_ServerConnection) SetPropertyConnectionID(value uint32) (err error) {
	return instance.SetProperty("ConnectionID", (value))
}

// GetConnectionID gets the value of ConnectionID for the instance
func (instance *Win32_ServerConnection) GetPropertyConnectionID() (value uint32, err error) {
	retValue, err := instance.GetProperty("ConnectionID")
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

// SetNumberOfFiles sets the value of NumberOfFiles for the instance
func (instance *Win32_ServerConnection) SetPropertyNumberOfFiles(value uint32) (err error) {
	return instance.SetProperty("NumberOfFiles", (value))
}

// GetNumberOfFiles gets the value of NumberOfFiles for the instance
func (instance *Win32_ServerConnection) GetPropertyNumberOfFiles() (value uint32, err error) {
	retValue, err := instance.GetProperty("NumberOfFiles")
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

// SetNumberOfUsers sets the value of NumberOfUsers for the instance
func (instance *Win32_ServerConnection) SetPropertyNumberOfUsers(value uint32) (err error) {
	return instance.SetProperty("NumberOfUsers", (value))
}

// GetNumberOfUsers gets the value of NumberOfUsers for the instance
func (instance *Win32_ServerConnection) GetPropertyNumberOfUsers() (value uint32, err error) {
	retValue, err := instance.GetProperty("NumberOfUsers")
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

// SetShareName sets the value of ShareName for the instance
func (instance *Win32_ServerConnection) SetPropertyShareName(value string) (err error) {
	return instance.SetProperty("ShareName", (value))
}

// GetShareName gets the value of ShareName for the instance
func (instance *Win32_ServerConnection) GetPropertyShareName() (value string, err error) {
	retValue, err := instance.GetProperty("ShareName")
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
func (instance *Win32_ServerConnection) SetPropertyUserName(value string) (err error) {
	return instance.SetProperty("UserName", (value))
}

// GetUserName gets the value of UserName for the instance
func (instance *Win32_ServerConnection) GetPropertyUserName() (value string, err error) {
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
