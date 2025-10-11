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

// Win32_RemoveFileAction struct
type Win32_RemoveFileAction struct {
	*CIM_RemoveFileAction

	//
	DirProperty string

	//
	FileKey string

	//
	FileName string

	//
	InstallMode uint16
}

func NewWin32_RemoveFileActionEx1(instance *cim.WmiInstance) (newInstance *Win32_RemoveFileAction, err error) {
	tmp, err := NewCIM_RemoveFileActionEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_RemoveFileAction{
		CIM_RemoveFileAction: tmp,
	}
	return
}

func NewWin32_RemoveFileActionEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_RemoveFileAction, err error) {
	tmp, err := NewCIM_RemoveFileActionEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_RemoveFileAction{
		CIM_RemoveFileAction: tmp,
	}
	return
}

// SetDirProperty sets the value of DirProperty for the instance
func (instance *Win32_RemoveFileAction) SetPropertyDirProperty(value string) (err error) {
	return instance.SetProperty("DirProperty", (value))
}

// GetDirProperty gets the value of DirProperty for the instance
func (instance *Win32_RemoveFileAction) GetPropertyDirProperty() (value string, err error) {
	retValue, err := instance.GetProperty("DirProperty")
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

// SetFileKey sets the value of FileKey for the instance
func (instance *Win32_RemoveFileAction) SetPropertyFileKey(value string) (err error) {
	return instance.SetProperty("FileKey", (value))
}

// GetFileKey gets the value of FileKey for the instance
func (instance *Win32_RemoveFileAction) GetPropertyFileKey() (value string, err error) {
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

// SetFileName sets the value of FileName for the instance
func (instance *Win32_RemoveFileAction) SetPropertyFileName(value string) (err error) {
	return instance.SetProperty("FileName", (value))
}

// GetFileName gets the value of FileName for the instance
func (instance *Win32_RemoveFileAction) GetPropertyFileName() (value string, err error) {
	retValue, err := instance.GetProperty("FileName")
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

// SetInstallMode sets the value of InstallMode for the instance
func (instance *Win32_RemoveFileAction) SetPropertyInstallMode(value uint16) (err error) {
	return instance.SetProperty("InstallMode", (value))
}

// GetInstallMode gets the value of InstallMode for the instance
func (instance *Win32_RemoveFileAction) GetPropertyInstallMode() (value uint16, err error) {
	retValue, err := instance.GetProperty("InstallMode")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint16)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint16 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint16(valuetmp)

	return
}
