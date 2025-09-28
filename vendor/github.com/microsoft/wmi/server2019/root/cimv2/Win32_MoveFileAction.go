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

// Win32_MoveFileAction struct
type Win32_MoveFileAction struct {
	*CIM_FileAction

	//
	DestFolder string

	//
	DestName string

	//
	FileKey string

	//
	Options uint16

	//
	SourceFolder string

	//
	SourceName string
}

func NewWin32_MoveFileActionEx1(instance *cim.WmiInstance) (newInstance *Win32_MoveFileAction, err error) {
	tmp, err := NewCIM_FileActionEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_MoveFileAction{
		CIM_FileAction: tmp,
	}
	return
}

func NewWin32_MoveFileActionEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_MoveFileAction, err error) {
	tmp, err := NewCIM_FileActionEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_MoveFileAction{
		CIM_FileAction: tmp,
	}
	return
}

// SetDestFolder sets the value of DestFolder for the instance
func (instance *Win32_MoveFileAction) SetPropertyDestFolder(value string) (err error) {
	return instance.SetProperty("DestFolder", (value))
}

// GetDestFolder gets the value of DestFolder for the instance
func (instance *Win32_MoveFileAction) GetPropertyDestFolder() (value string, err error) {
	retValue, err := instance.GetProperty("DestFolder")
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

// SetDestName sets the value of DestName for the instance
func (instance *Win32_MoveFileAction) SetPropertyDestName(value string) (err error) {
	return instance.SetProperty("DestName", (value))
}

// GetDestName gets the value of DestName for the instance
func (instance *Win32_MoveFileAction) GetPropertyDestName() (value string, err error) {
	retValue, err := instance.GetProperty("DestName")
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
func (instance *Win32_MoveFileAction) SetPropertyFileKey(value string) (err error) {
	return instance.SetProperty("FileKey", (value))
}

// GetFileKey gets the value of FileKey for the instance
func (instance *Win32_MoveFileAction) GetPropertyFileKey() (value string, err error) {
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

// SetOptions sets the value of Options for the instance
func (instance *Win32_MoveFileAction) SetPropertyOptions(value uint16) (err error) {
	return instance.SetProperty("Options", (value))
}

// GetOptions gets the value of Options for the instance
func (instance *Win32_MoveFileAction) GetPropertyOptions() (value uint16, err error) {
	retValue, err := instance.GetProperty("Options")
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

// SetSourceFolder sets the value of SourceFolder for the instance
func (instance *Win32_MoveFileAction) SetPropertySourceFolder(value string) (err error) {
	return instance.SetProperty("SourceFolder", (value))
}

// GetSourceFolder gets the value of SourceFolder for the instance
func (instance *Win32_MoveFileAction) GetPropertySourceFolder() (value string, err error) {
	retValue, err := instance.GetProperty("SourceFolder")
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

// SetSourceName sets the value of SourceName for the instance
func (instance *Win32_MoveFileAction) SetPropertySourceName(value string) (err error) {
	return instance.SetProperty("SourceName", (value))
}

// GetSourceName gets the value of SourceName for the instance
func (instance *Win32_MoveFileAction) GetPropertySourceName() (value string, err error) {
	retValue, err := instance.GetProperty("SourceName")
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
