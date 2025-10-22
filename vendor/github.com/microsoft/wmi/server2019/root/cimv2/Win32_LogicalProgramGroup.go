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

// Win32_LogicalProgramGroup struct
type Win32_LogicalProgramGroup struct {
	*Win32_ProgramGroupOrItem

	//
	GroupName string

	//
	UserName string
}

func NewWin32_LogicalProgramGroupEx1(instance *cim.WmiInstance) (newInstance *Win32_LogicalProgramGroup, err error) {
	tmp, err := NewWin32_ProgramGroupOrItemEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_LogicalProgramGroup{
		Win32_ProgramGroupOrItem: tmp,
	}
	return
}

func NewWin32_LogicalProgramGroupEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_LogicalProgramGroup, err error) {
	tmp, err := NewWin32_ProgramGroupOrItemEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_LogicalProgramGroup{
		Win32_ProgramGroupOrItem: tmp,
	}
	return
}

// SetGroupName sets the value of GroupName for the instance
func (instance *Win32_LogicalProgramGroup) SetPropertyGroupName(value string) (err error) {
	return instance.SetProperty("GroupName", (value))
}

// GetGroupName gets the value of GroupName for the instance
func (instance *Win32_LogicalProgramGroup) GetPropertyGroupName() (value string, err error) {
	retValue, err := instance.GetProperty("GroupName")
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
func (instance *Win32_LogicalProgramGroup) SetPropertyUserName(value string) (err error) {
	return instance.SetProperty("UserName", (value))
}

// GetUserName gets the value of UserName for the instance
func (instance *Win32_LogicalProgramGroup) GetPropertyUserName() (value string, err error) {
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
