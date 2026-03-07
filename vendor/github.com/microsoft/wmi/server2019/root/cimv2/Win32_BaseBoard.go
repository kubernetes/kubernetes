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

// Win32_BaseBoard struct
type Win32_BaseBoard struct {
	*CIM_Card

	//
	ConfigOptions []string

	//
	Product string
}

func NewWin32_BaseBoardEx1(instance *cim.WmiInstance) (newInstance *Win32_BaseBoard, err error) {
	tmp, err := NewCIM_CardEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_BaseBoard{
		CIM_Card: tmp,
	}
	return
}

func NewWin32_BaseBoardEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_BaseBoard, err error) {
	tmp, err := NewCIM_CardEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_BaseBoard{
		CIM_Card: tmp,
	}
	return
}

// SetConfigOptions sets the value of ConfigOptions for the instance
func (instance *Win32_BaseBoard) SetPropertyConfigOptions(value []string) (err error) {
	return instance.SetProperty("ConfigOptions", (value))
}

// GetConfigOptions gets the value of ConfigOptions for the instance
func (instance *Win32_BaseBoard) GetPropertyConfigOptions() (value []string, err error) {
	retValue, err := instance.GetProperty("ConfigOptions")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	for _, interfaceValue := range retValue.([]interface{}) {
		valuetmp, ok := interfaceValue.(string)
		if !ok {
			err = errors.Wrapf(errors.InvalidType, " string is Invalid. Expected %s", reflect.TypeOf(interfaceValue))
			return
		}
		value = append(value, string(valuetmp))
	}

	return
}

// SetProduct sets the value of Product for the instance
func (instance *Win32_BaseBoard) SetPropertyProduct(value string) (err error) {
	return instance.SetProperty("Product", (value))
}

// GetProduct gets the value of Product for the instance
func (instance *Win32_BaseBoard) GetPropertyProduct() (value string, err error) {
	retValue, err := instance.GetProperty("Product")
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
