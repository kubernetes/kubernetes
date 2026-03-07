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

// Win32_PerfRawData_AFDCounters_MicrosoftWinsockBSP struct
type Win32_PerfRawData_AFDCounters_MicrosoftWinsockBSP struct {
	*Win32_PerfRawData

	//
	DroppedDatagrams uint32

	//
	DroppedDatagramsPersec uint32

	//
	RejectedConnections uint32

	//
	RejectedConnectionsPersec uint32
}

func NewWin32_PerfRawData_AFDCounters_MicrosoftWinsockBSPEx1(instance *cim.WmiInstance) (newInstance *Win32_PerfRawData_AFDCounters_MicrosoftWinsockBSP, err error) {
	tmp, err := NewWin32_PerfRawDataEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfRawData_AFDCounters_MicrosoftWinsockBSP{
		Win32_PerfRawData: tmp,
	}
	return
}

func NewWin32_PerfRawData_AFDCounters_MicrosoftWinsockBSPEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_PerfRawData_AFDCounters_MicrosoftWinsockBSP, err error) {
	tmp, err := NewWin32_PerfRawDataEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfRawData_AFDCounters_MicrosoftWinsockBSP{
		Win32_PerfRawData: tmp,
	}
	return
}

// SetDroppedDatagrams sets the value of DroppedDatagrams for the instance
func (instance *Win32_PerfRawData_AFDCounters_MicrosoftWinsockBSP) SetPropertyDroppedDatagrams(value uint32) (err error) {
	return instance.SetProperty("DroppedDatagrams", (value))
}

// GetDroppedDatagrams gets the value of DroppedDatagrams for the instance
func (instance *Win32_PerfRawData_AFDCounters_MicrosoftWinsockBSP) GetPropertyDroppedDatagrams() (value uint32, err error) {
	retValue, err := instance.GetProperty("DroppedDatagrams")
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

// SetDroppedDatagramsPersec sets the value of DroppedDatagramsPersec for the instance
func (instance *Win32_PerfRawData_AFDCounters_MicrosoftWinsockBSP) SetPropertyDroppedDatagramsPersec(value uint32) (err error) {
	return instance.SetProperty("DroppedDatagramsPersec", (value))
}

// GetDroppedDatagramsPersec gets the value of DroppedDatagramsPersec for the instance
func (instance *Win32_PerfRawData_AFDCounters_MicrosoftWinsockBSP) GetPropertyDroppedDatagramsPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("DroppedDatagramsPersec")
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

// SetRejectedConnections sets the value of RejectedConnections for the instance
func (instance *Win32_PerfRawData_AFDCounters_MicrosoftWinsockBSP) SetPropertyRejectedConnections(value uint32) (err error) {
	return instance.SetProperty("RejectedConnections", (value))
}

// GetRejectedConnections gets the value of RejectedConnections for the instance
func (instance *Win32_PerfRawData_AFDCounters_MicrosoftWinsockBSP) GetPropertyRejectedConnections() (value uint32, err error) {
	retValue, err := instance.GetProperty("RejectedConnections")
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

// SetRejectedConnectionsPersec sets the value of RejectedConnectionsPersec for the instance
func (instance *Win32_PerfRawData_AFDCounters_MicrosoftWinsockBSP) SetPropertyRejectedConnectionsPersec(value uint32) (err error) {
	return instance.SetProperty("RejectedConnectionsPersec", (value))
}

// GetRejectedConnectionsPersec gets the value of RejectedConnectionsPersec for the instance
func (instance *Win32_PerfRawData_AFDCounters_MicrosoftWinsockBSP) GetPropertyRejectedConnectionsPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("RejectedConnectionsPersec")
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
