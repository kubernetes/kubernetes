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

// Win32_PerfRawData_LSM_UserInputDelayperProcess struct
type Win32_PerfRawData_LSM_UserInputDelayperProcess struct {
	*Win32_PerfRawData

	//
	MaxInputDelay uint64

	//
	MaxInputDelay_Base uint32
}

func NewWin32_PerfRawData_LSM_UserInputDelayperProcessEx1(instance *cim.WmiInstance) (newInstance *Win32_PerfRawData_LSM_UserInputDelayperProcess, err error) {
	tmp, err := NewWin32_PerfRawDataEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfRawData_LSM_UserInputDelayperProcess{
		Win32_PerfRawData: tmp,
	}
	return
}

func NewWin32_PerfRawData_LSM_UserInputDelayperProcessEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_PerfRawData_LSM_UserInputDelayperProcess, err error) {
	tmp, err := NewWin32_PerfRawDataEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfRawData_LSM_UserInputDelayperProcess{
		Win32_PerfRawData: tmp,
	}
	return
}

// SetMaxInputDelay sets the value of MaxInputDelay for the instance
func (instance *Win32_PerfRawData_LSM_UserInputDelayperProcess) SetPropertyMaxInputDelay(value uint64) (err error) {
	return instance.SetProperty("MaxInputDelay", (value))
}

// GetMaxInputDelay gets the value of MaxInputDelay for the instance
func (instance *Win32_PerfRawData_LSM_UserInputDelayperProcess) GetPropertyMaxInputDelay() (value uint64, err error) {
	retValue, err := instance.GetProperty("MaxInputDelay")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint64)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint64 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint64(valuetmp)

	return
}

// SetMaxInputDelay_Base sets the value of MaxInputDelay_Base for the instance
func (instance *Win32_PerfRawData_LSM_UserInputDelayperProcess) SetPropertyMaxInputDelay_Base(value uint32) (err error) {
	return instance.SetProperty("MaxInputDelay_Base", (value))
}

// GetMaxInputDelay_Base gets the value of MaxInputDelay_Base for the instance
func (instance *Win32_PerfRawData_LSM_UserInputDelayperProcess) GetPropertyMaxInputDelay_Base() (value uint32, err error) {
	retValue, err := instance.GetProperty("MaxInputDelay_Base")
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
