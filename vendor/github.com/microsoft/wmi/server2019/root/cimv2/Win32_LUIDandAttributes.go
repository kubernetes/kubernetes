// Copyright 2019 (c) Microsoft Corporation.
// Licensed under the MIT license.

//
// Author:
//      Auto Generated on 9/18/2020 using wmigen
//      Source root.CIMV2
//////////////////////////////////////////////
package cimv2

import (
	"github.com/microsoft/wmi/pkg/base/instance"
	"github.com/microsoft/wmi/pkg/base/query"
	"github.com/microsoft/wmi/pkg/errors"
	cim "github.com/microsoft/wmi/pkg/wmiinstance"
	"reflect"
)

// Win32_LUIDandAttributes struct
type Win32_LUIDandAttributes struct {
	*cim.WmiInstance

	//
	Attributes uint32

	//
	LUID Win32_LUID
}

func NewWin32_LUIDandAttributesEx1(instance *cim.WmiInstance) (newInstance *Win32_LUIDandAttributes, err error) {
	tmp, err := instance, nil

	if err != nil {
		return
	}
	newInstance = &Win32_LUIDandAttributes{
		WmiInstance: tmp,
	}
	return
}

func NewWin32_LUIDandAttributesEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_LUIDandAttributes, err error) {
	tmp, err := instance.GetWmiInstance(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_LUIDandAttributes{
		WmiInstance: tmp,
	}
	return
}

// SetAttributes sets the value of Attributes for the instance
func (instance *Win32_LUIDandAttributes) SetPropertyAttributes(value uint32) (err error) {
	return instance.SetProperty("Attributes", (value))
}

// GetAttributes gets the value of Attributes for the instance
func (instance *Win32_LUIDandAttributes) GetPropertyAttributes() (value uint32, err error) {
	retValue, err := instance.GetProperty("Attributes")
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

// SetLUID sets the value of LUID for the instance
func (instance *Win32_LUIDandAttributes) SetPropertyLUID(value Win32_LUID) (err error) {
	return instance.SetProperty("LUID", (value))
}

// GetLUID gets the value of LUID for the instance
func (instance *Win32_LUIDandAttributes) GetPropertyLUID() (value Win32_LUID, err error) {
	retValue, err := instance.GetProperty("LUID")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(Win32_LUID)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " Win32_LUID is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = Win32_LUID(valuetmp)

	return
}
