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

// Win32_DMAChannel struct
type Win32_DMAChannel struct {
	*CIM_DMA

	//
	Port uint32
}

func NewWin32_DMAChannelEx1(instance *cim.WmiInstance) (newInstance *Win32_DMAChannel, err error) {
	tmp, err := NewCIM_DMAEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_DMAChannel{
		CIM_DMA: tmp,
	}
	return
}

func NewWin32_DMAChannelEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_DMAChannel, err error) {
	tmp, err := NewCIM_DMAEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_DMAChannel{
		CIM_DMA: tmp,
	}
	return
}

// SetPort sets the value of Port for the instance
func (instance *Win32_DMAChannel) SetPropertyPort(value uint32) (err error) {
	return instance.SetProperty("Port", (value))
}

// GetPort gets the value of Port for the instance
func (instance *Win32_DMAChannel) GetPropertyPort() (value uint32, err error) {
	retValue, err := instance.GetProperty("Port")
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
