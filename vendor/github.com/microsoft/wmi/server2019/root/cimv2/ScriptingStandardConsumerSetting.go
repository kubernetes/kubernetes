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

// ScriptingStandardConsumerSetting struct
type ScriptingStandardConsumerSetting struct {
	*CIM_Setting

	//
	MaximumScripts uint32

	//
	Timeout uint32
}

func NewScriptingStandardConsumerSettingEx1(instance *cim.WmiInstance) (newInstance *ScriptingStandardConsumerSetting, err error) {
	tmp, err := NewCIM_SettingEx1(instance)

	if err != nil {
		return
	}
	newInstance = &ScriptingStandardConsumerSetting{
		CIM_Setting: tmp,
	}
	return
}

func NewScriptingStandardConsumerSettingEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *ScriptingStandardConsumerSetting, err error) {
	tmp, err := NewCIM_SettingEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &ScriptingStandardConsumerSetting{
		CIM_Setting: tmp,
	}
	return
}

// SetMaximumScripts sets the value of MaximumScripts for the instance
func (instance *ScriptingStandardConsumerSetting) SetPropertyMaximumScripts(value uint32) (err error) {
	return instance.SetProperty("MaximumScripts", (value))
}

// GetMaximumScripts gets the value of MaximumScripts for the instance
func (instance *ScriptingStandardConsumerSetting) GetPropertyMaximumScripts() (value uint32, err error) {
	retValue, err := instance.GetProperty("MaximumScripts")
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

// SetTimeout sets the value of Timeout for the instance
func (instance *ScriptingStandardConsumerSetting) SetPropertyTimeout(value uint32) (err error) {
	return instance.SetProperty("Timeout", (value))
}

// GetTimeout gets the value of Timeout for the instance
func (instance *ScriptingStandardConsumerSetting) GetPropertyTimeout() (value uint32, err error) {
	retValue, err := instance.GetProperty("Timeout")
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
