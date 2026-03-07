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

// Win32_LUID struct
type Win32_LUID struct {
	*cim.WmiInstance

	//
	HighPart uint32

	//
	LowPart uint32
}

func NewWin32_LUIDEx1(instance *cim.WmiInstance) (newInstance *Win32_LUID, err error) {
	tmp, err := instance, nil

	if err != nil {
		return
	}
	newInstance = &Win32_LUID{
		WmiInstance: tmp,
	}
	return
}

func NewWin32_LUIDEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_LUID, err error) {
	tmp, err := instance.GetWmiInstance(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_LUID{
		WmiInstance: tmp,
	}
	return
}

// SetHighPart sets the value of HighPart for the instance
func (instance *Win32_LUID) SetPropertyHighPart(value uint32) (err error) {
	return instance.SetProperty("HighPart", (value))
}

// GetHighPart gets the value of HighPart for the instance
func (instance *Win32_LUID) GetPropertyHighPart() (value uint32, err error) {
	retValue, err := instance.GetProperty("HighPart")
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

// SetLowPart sets the value of LowPart for the instance
func (instance *Win32_LUID) SetPropertyLowPart(value uint32) (err error) {
	return instance.SetProperty("LowPart", (value))
}

// GetLowPart gets the value of LowPart for the instance
func (instance *Win32_LUID) GetPropertyLowPart() (value uint32, err error) {
	retValue, err := instance.GetProperty("LowPart")
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
