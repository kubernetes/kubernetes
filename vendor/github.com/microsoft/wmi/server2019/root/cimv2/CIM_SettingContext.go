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

// CIM_SettingContext struct
type CIM_SettingContext struct {
	*cim.WmiInstance

	//
	Context CIM_Configuration

	//
	Setting CIM_Setting
}

func NewCIM_SettingContextEx1(instance *cim.WmiInstance) (newInstance *CIM_SettingContext, err error) {
	tmp, err := instance, nil

	if err != nil {
		return
	}
	newInstance = &CIM_SettingContext{
		WmiInstance: tmp,
	}
	return
}

func NewCIM_SettingContextEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *CIM_SettingContext, err error) {
	tmp, err := instance.GetWmiInstance(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &CIM_SettingContext{
		WmiInstance: tmp,
	}
	return
}

// SetContext sets the value of Context for the instance
func (instance *CIM_SettingContext) SetPropertyContext(value CIM_Configuration) (err error) {
	return instance.SetProperty("Context", (value))
}

// GetContext gets the value of Context for the instance
func (instance *CIM_SettingContext) GetPropertyContext() (value CIM_Configuration, err error) {
	retValue, err := instance.GetProperty("Context")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(CIM_Configuration)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " CIM_Configuration is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = CIM_Configuration(valuetmp)

	return
}

// SetSetting sets the value of Setting for the instance
func (instance *CIM_SettingContext) SetPropertySetting(value CIM_Setting) (err error) {
	return instance.SetProperty("Setting", (value))
}

// GetSetting gets the value of Setting for the instance
func (instance *CIM_SettingContext) GetPropertySetting() (value CIM_Setting, err error) {
	retValue, err := instance.GetProperty("Setting")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(CIM_Setting)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " CIM_Setting is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = CIM_Setting(valuetmp)

	return
}
