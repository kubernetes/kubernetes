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

// Win32_CodecFile struct
type Win32_CodecFile struct {
	*CIM_DataFile

	//
	Group string
}

func NewWin32_CodecFileEx1(instance *cim.WmiInstance) (newInstance *Win32_CodecFile, err error) {
	tmp, err := NewCIM_DataFileEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_CodecFile{
		CIM_DataFile: tmp,
	}
	return
}

func NewWin32_CodecFileEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_CodecFile, err error) {
	tmp, err := NewCIM_DataFileEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_CodecFile{
		CIM_DataFile: tmp,
	}
	return
}

// SetGroup sets the value of Group for the instance
func (instance *Win32_CodecFile) SetPropertyGroup(value string) (err error) {
	return instance.SetProperty("Group", (value))
}

// GetGroup gets the value of Group for the instance
func (instance *Win32_CodecFile) GetPropertyGroup() (value string, err error) {
	retValue, err := instance.GetProperty("Group")
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
