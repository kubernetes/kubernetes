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

// Win32_DuplicateFileAction struct
type Win32_DuplicateFileAction struct {
	*CIM_CopyFileAction

	//
	FileKey string
}

func NewWin32_DuplicateFileActionEx1(instance *cim.WmiInstance) (newInstance *Win32_DuplicateFileAction, err error) {
	tmp, err := NewCIM_CopyFileActionEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_DuplicateFileAction{
		CIM_CopyFileAction: tmp,
	}
	return
}

func NewWin32_DuplicateFileActionEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_DuplicateFileAction, err error) {
	tmp, err := NewCIM_CopyFileActionEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_DuplicateFileAction{
		CIM_CopyFileAction: tmp,
	}
	return
}

// SetFileKey sets the value of FileKey for the instance
func (instance *Win32_DuplicateFileAction) SetPropertyFileKey(value string) (err error) {
	return instance.SetProperty("FileKey", (value))
}

// GetFileKey gets the value of FileKey for the instance
func (instance *Win32_DuplicateFileAction) GetPropertyFileKey() (value string, err error) {
	retValue, err := instance.GetProperty("FileKey")
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
