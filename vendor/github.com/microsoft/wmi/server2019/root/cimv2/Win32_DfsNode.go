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

// Win32_DfsNode struct
type Win32_DfsNode struct {
	*CIM_LogicalElement

	//
	Root bool

	//
	State uint32

	//
	Timeout uint32
}

func NewWin32_DfsNodeEx1(instance *cim.WmiInstance) (newInstance *Win32_DfsNode, err error) {
	tmp, err := NewCIM_LogicalElementEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_DfsNode{
		CIM_LogicalElement: tmp,
	}
	return
}

func NewWin32_DfsNodeEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_DfsNode, err error) {
	tmp, err := NewCIM_LogicalElementEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_DfsNode{
		CIM_LogicalElement: tmp,
	}
	return
}

// SetRoot sets the value of Root for the instance
func (instance *Win32_DfsNode) SetPropertyRoot(value bool) (err error) {
	return instance.SetProperty("Root", (value))
}

// GetRoot gets the value of Root for the instance
func (instance *Win32_DfsNode) GetPropertyRoot() (value bool, err error) {
	retValue, err := instance.GetProperty("Root")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(bool)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " bool is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = bool(valuetmp)

	return
}

// SetState sets the value of State for the instance
func (instance *Win32_DfsNode) SetPropertyState(value uint32) (err error) {
	return instance.SetProperty("State", (value))
}

// GetState gets the value of State for the instance
func (instance *Win32_DfsNode) GetPropertyState() (value uint32, err error) {
	retValue, err := instance.GetProperty("State")
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

// SetTimeout sets the value of Timeout for the instance
func (instance *Win32_DfsNode) SetPropertyTimeout(value uint32) (err error) {
	return instance.SetProperty("Timeout", (value))
}

// GetTimeout gets the value of Timeout for the instance
func (instance *Win32_DfsNode) GetPropertyTimeout() (value uint32, err error) {
	retValue, err := instance.GetProperty("Timeout")
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

//

// <param name="Description" type="string "></param>
// <param name="DfsEntryPath" type="string "></param>
// <param name="ServerName" type="string "></param>
// <param name="ShareName" type="string "></param>

// <param name="ReturnValue" type="uint32 "></param>
func (instance *Win32_DfsNode) Create( /* IN */ DfsEntryPath string,
	/* IN */ ServerName string,
	/* IN */ ShareName string,
	/* OPTIONAL IN */ Description string) (result uint32, err error) {
	retVal, err := instance.InvokeMethodWithReturn("Create", DfsEntryPath, ServerName, ShareName, Description)
	if err != nil {
		return
	}
	result = uint32(retVal)
	return

}
