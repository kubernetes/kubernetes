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

// Win32_PageFile struct
type Win32_PageFile struct {
	*CIM_DataFile

	//
	FreeSpace uint32

	//
	InitialSize uint32

	//
	MaximumSize uint32
}

func NewWin32_PageFileEx1(instance *cim.WmiInstance) (newInstance *Win32_PageFile, err error) {
	tmp, err := NewCIM_DataFileEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_PageFile{
		CIM_DataFile: tmp,
	}
	return
}

func NewWin32_PageFileEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_PageFile, err error) {
	tmp, err := NewCIM_DataFileEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_PageFile{
		CIM_DataFile: tmp,
	}
	return
}

// SetFreeSpace sets the value of FreeSpace for the instance
func (instance *Win32_PageFile) SetPropertyFreeSpace(value uint32) (err error) {
	return instance.SetProperty("FreeSpace", (value))
}

// GetFreeSpace gets the value of FreeSpace for the instance
func (instance *Win32_PageFile) GetPropertyFreeSpace() (value uint32, err error) {
	retValue, err := instance.GetProperty("FreeSpace")
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

// SetInitialSize sets the value of InitialSize for the instance
func (instance *Win32_PageFile) SetPropertyInitialSize(value uint32) (err error) {
	return instance.SetProperty("InitialSize", (value))
}

// GetInitialSize gets the value of InitialSize for the instance
func (instance *Win32_PageFile) GetPropertyInitialSize() (value uint32, err error) {
	retValue, err := instance.GetProperty("InitialSize")
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

// SetMaximumSize sets the value of MaximumSize for the instance
func (instance *Win32_PageFile) SetPropertyMaximumSize(value uint32) (err error) {
	return instance.SetProperty("MaximumSize", (value))
}

// GetMaximumSize gets the value of MaximumSize for the instance
func (instance *Win32_PageFile) GetPropertyMaximumSize() (value uint32, err error) {
	retValue, err := instance.GetProperty("MaximumSize")
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
