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

// Win32_NTLogEventLog struct
type Win32_NTLogEventLog struct {
	*cim.WmiInstance

	//
	Log Win32_NTEventlogFile

	//
	Record Win32_NTLogEvent
}

func NewWin32_NTLogEventLogEx1(instance *cim.WmiInstance) (newInstance *Win32_NTLogEventLog, err error) {
	tmp, err := instance, nil

	if err != nil {
		return
	}
	newInstance = &Win32_NTLogEventLog{
		WmiInstance: tmp,
	}
	return
}

func NewWin32_NTLogEventLogEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_NTLogEventLog, err error) {
	tmp, err := instance.GetWmiInstance(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_NTLogEventLog{
		WmiInstance: tmp,
	}
	return
}

// SetLog sets the value of Log for the instance
func (instance *Win32_NTLogEventLog) SetPropertyLog(value Win32_NTEventlogFile) (err error) {
	return instance.SetProperty("Log", (value))
}

// GetLog gets the value of Log for the instance
func (instance *Win32_NTLogEventLog) GetPropertyLog() (value Win32_NTEventlogFile, err error) {
	retValue, err := instance.GetProperty("Log")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(Win32_NTEventlogFile)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " Win32_NTEventlogFile is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = Win32_NTEventlogFile(valuetmp)

	return
}

// SetRecord sets the value of Record for the instance
func (instance *Win32_NTLogEventLog) SetPropertyRecord(value Win32_NTLogEvent) (err error) {
	return instance.SetProperty("Record", (value))
}

// GetRecord gets the value of Record for the instance
func (instance *Win32_NTLogEventLog) GetPropertyRecord() (value Win32_NTLogEvent, err error) {
	retValue, err := instance.GetProperty("Record")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(Win32_NTLogEvent)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " Win32_NTLogEvent is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = Win32_NTLogEvent(valuetmp)

	return
}
