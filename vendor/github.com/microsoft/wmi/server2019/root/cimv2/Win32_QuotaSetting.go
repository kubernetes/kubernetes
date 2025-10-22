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

// Win32_QuotaSetting struct
type Win32_QuotaSetting struct {
	*CIM_Setting

	//
	DefaultLimit int64

	//
	DefaultWarningLimit int64

	//
	ExceededNotification bool

	//
	State uint32

	//
	VolumePath string

	//
	WarningExceededNotification bool
}

func NewWin32_QuotaSettingEx1(instance *cim.WmiInstance) (newInstance *Win32_QuotaSetting, err error) {
	tmp, err := NewCIM_SettingEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_QuotaSetting{
		CIM_Setting: tmp,
	}
	return
}

func NewWin32_QuotaSettingEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_QuotaSetting, err error) {
	tmp, err := NewCIM_SettingEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_QuotaSetting{
		CIM_Setting: tmp,
	}
	return
}

// SetDefaultLimit sets the value of DefaultLimit for the instance
func (instance *Win32_QuotaSetting) SetPropertyDefaultLimit(value int64) (err error) {
	return instance.SetProperty("DefaultLimit", (value))
}

// GetDefaultLimit gets the value of DefaultLimit for the instance
func (instance *Win32_QuotaSetting) GetPropertyDefaultLimit() (value int64, err error) {
	retValue, err := instance.GetProperty("DefaultLimit")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(int64)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " int64 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = int64(valuetmp)

	return
}

// SetDefaultWarningLimit sets the value of DefaultWarningLimit for the instance
func (instance *Win32_QuotaSetting) SetPropertyDefaultWarningLimit(value int64) (err error) {
	return instance.SetProperty("DefaultWarningLimit", (value))
}

// GetDefaultWarningLimit gets the value of DefaultWarningLimit for the instance
func (instance *Win32_QuotaSetting) GetPropertyDefaultWarningLimit() (value int64, err error) {
	retValue, err := instance.GetProperty("DefaultWarningLimit")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(int64)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " int64 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = int64(valuetmp)

	return
}

// SetExceededNotification sets the value of ExceededNotification for the instance
func (instance *Win32_QuotaSetting) SetPropertyExceededNotification(value bool) (err error) {
	return instance.SetProperty("ExceededNotification", (value))
}

// GetExceededNotification gets the value of ExceededNotification for the instance
func (instance *Win32_QuotaSetting) GetPropertyExceededNotification() (value bool, err error) {
	retValue, err := instance.GetProperty("ExceededNotification")
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

// SetState sets the value of State for the instance
func (instance *Win32_QuotaSetting) SetPropertyState(value uint32) (err error) {
	return instance.SetProperty("State", (value))
}

// GetState gets the value of State for the instance
func (instance *Win32_QuotaSetting) GetPropertyState() (value uint32, err error) {
	retValue, err := instance.GetProperty("State")
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

// SetVolumePath sets the value of VolumePath for the instance
func (instance *Win32_QuotaSetting) SetPropertyVolumePath(value string) (err error) {
	return instance.SetProperty("VolumePath", (value))
}

// GetVolumePath gets the value of VolumePath for the instance
func (instance *Win32_QuotaSetting) GetPropertyVolumePath() (value string, err error) {
	retValue, err := instance.GetProperty("VolumePath")
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

// SetWarningExceededNotification sets the value of WarningExceededNotification for the instance
func (instance *Win32_QuotaSetting) SetPropertyWarningExceededNotification(value bool) (err error) {
	return instance.SetProperty("WarningExceededNotification", (value))
}

// GetWarningExceededNotification gets the value of WarningExceededNotification for the instance
func (instance *Win32_QuotaSetting) GetPropertyWarningExceededNotification() (value bool, err error) {
	retValue, err := instance.GetProperty("WarningExceededNotification")
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
