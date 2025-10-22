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

// Win32_SIDandAttributes struct
type Win32_SIDandAttributes struct {
	*cim.WmiInstance

	//
	Attributes uint32

	//
	SID Win32_SID
}

func NewWin32_SIDandAttributesEx1(instance *cim.WmiInstance) (newInstance *Win32_SIDandAttributes, err error) {
	tmp, err := instance, nil

	if err != nil {
		return
	}
	newInstance = &Win32_SIDandAttributes{
		WmiInstance: tmp,
	}
	return
}

func NewWin32_SIDandAttributesEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_SIDandAttributes, err error) {
	tmp, err := instance.GetWmiInstance(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_SIDandAttributes{
		WmiInstance: tmp,
	}
	return
}

// SetAttributes sets the value of Attributes for the instance
func (instance *Win32_SIDandAttributes) SetPropertyAttributes(value uint32) (err error) {
	return instance.SetProperty("Attributes", (value))
}

// GetAttributes gets the value of Attributes for the instance
func (instance *Win32_SIDandAttributes) GetPropertyAttributes() (value uint32, err error) {
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

// SetSID sets the value of SID for the instance
func (instance *Win32_SIDandAttributes) SetPropertySID(value Win32_SID) (err error) {
	return instance.SetProperty("SID", (value))
}

// GetSID gets the value of SID for the instance
func (instance *Win32_SIDandAttributes) GetPropertySID() (value Win32_SID, err error) {
	retValue, err := instance.GetProperty("SID")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(Win32_SID)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " Win32_SID is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = Win32_SID(valuetmp)

	return
}
