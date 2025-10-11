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

// Win32_ThreadTrace struct
type Win32_ThreadTrace struct {
	*Win32_SystemTrace

	//
	ProcessID uint32

	//
	ThreadID uint32
}

func NewWin32_ThreadTraceEx1(instance *cim.WmiInstance) (newInstance *Win32_ThreadTrace, err error) {
	tmp, err := NewWin32_SystemTraceEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_ThreadTrace{
		Win32_SystemTrace: tmp,
	}
	return
}

func NewWin32_ThreadTraceEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_ThreadTrace, err error) {
	tmp, err := NewWin32_SystemTraceEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_ThreadTrace{
		Win32_SystemTrace: tmp,
	}
	return
}

// SetProcessID sets the value of ProcessID for the instance
func (instance *Win32_ThreadTrace) SetPropertyProcessID(value uint32) (err error) {
	return instance.SetProperty("ProcessID", (value))
}

// GetProcessID gets the value of ProcessID for the instance
func (instance *Win32_ThreadTrace) GetPropertyProcessID() (value uint32, err error) {
	retValue, err := instance.GetProperty("ProcessID")
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

// SetThreadID sets the value of ThreadID for the instance
func (instance *Win32_ThreadTrace) SetPropertyThreadID(value uint32) (err error) {
	return instance.SetProperty("ThreadID", (value))
}

// GetThreadID gets the value of ThreadID for the instance
func (instance *Win32_ThreadTrace) GetPropertyThreadID() (value uint32, err error) {
	retValue, err := instance.GetProperty("ThreadID")
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
