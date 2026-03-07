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

// Win32_IRQResource struct
type Win32_IRQResource struct {
	*CIM_IRQ

	//
	Hardware bool

	//
	Vector uint32
}

func NewWin32_IRQResourceEx1(instance *cim.WmiInstance) (newInstance *Win32_IRQResource, err error) {
	tmp, err := NewCIM_IRQEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_IRQResource{
		CIM_IRQ: tmp,
	}
	return
}

func NewWin32_IRQResourceEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_IRQResource, err error) {
	tmp, err := NewCIM_IRQEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_IRQResource{
		CIM_IRQ: tmp,
	}
	return
}

// SetHardware sets the value of Hardware for the instance
func (instance *Win32_IRQResource) SetPropertyHardware(value bool) (err error) {
	return instance.SetProperty("Hardware", (value))
}

// GetHardware gets the value of Hardware for the instance
func (instance *Win32_IRQResource) GetPropertyHardware() (value bool, err error) {
	retValue, err := instance.GetProperty("Hardware")
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

// SetVector sets the value of Vector for the instance
func (instance *Win32_IRQResource) SetPropertyVector(value uint32) (err error) {
	return instance.SetProperty("Vector", (value))
}

// GetVector gets the value of Vector for the instance
func (instance *Win32_IRQResource) GetPropertyVector() (value uint32, err error) {
	retValue, err := instance.GetProperty("Vector")
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
