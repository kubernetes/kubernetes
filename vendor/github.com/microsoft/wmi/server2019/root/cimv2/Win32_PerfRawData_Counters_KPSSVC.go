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

// Win32_PerfRawData_Counters_KPSSVC struct
type Win32_PerfRawData_Counters_KPSSVC struct {
	*Win32_PerfRawData

	//
	FailedRequests uint32

	//
	IncomingArmoredRequests uint32

	//
	IncomingPasswordChangeRequests uint32

	//
	IncomingRequests uint32
}

func NewWin32_PerfRawData_Counters_KPSSVCEx1(instance *cim.WmiInstance) (newInstance *Win32_PerfRawData_Counters_KPSSVC, err error) {
	tmp, err := NewWin32_PerfRawDataEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfRawData_Counters_KPSSVC{
		Win32_PerfRawData: tmp,
	}
	return
}

func NewWin32_PerfRawData_Counters_KPSSVCEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_PerfRawData_Counters_KPSSVC, err error) {
	tmp, err := NewWin32_PerfRawDataEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfRawData_Counters_KPSSVC{
		Win32_PerfRawData: tmp,
	}
	return
}

// SetFailedRequests sets the value of FailedRequests for the instance
func (instance *Win32_PerfRawData_Counters_KPSSVC) SetPropertyFailedRequests(value uint32) (err error) {
	return instance.SetProperty("FailedRequests", (value))
}

// GetFailedRequests gets the value of FailedRequests for the instance
func (instance *Win32_PerfRawData_Counters_KPSSVC) GetPropertyFailedRequests() (value uint32, err error) {
	retValue, err := instance.GetProperty("FailedRequests")
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

// SetIncomingArmoredRequests sets the value of IncomingArmoredRequests for the instance
func (instance *Win32_PerfRawData_Counters_KPSSVC) SetPropertyIncomingArmoredRequests(value uint32) (err error) {
	return instance.SetProperty("IncomingArmoredRequests", (value))
}

// GetIncomingArmoredRequests gets the value of IncomingArmoredRequests for the instance
func (instance *Win32_PerfRawData_Counters_KPSSVC) GetPropertyIncomingArmoredRequests() (value uint32, err error) {
	retValue, err := instance.GetProperty("IncomingArmoredRequests")
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

// SetIncomingPasswordChangeRequests sets the value of IncomingPasswordChangeRequests for the instance
func (instance *Win32_PerfRawData_Counters_KPSSVC) SetPropertyIncomingPasswordChangeRequests(value uint32) (err error) {
	return instance.SetProperty("IncomingPasswordChangeRequests", (value))
}

// GetIncomingPasswordChangeRequests gets the value of IncomingPasswordChangeRequests for the instance
func (instance *Win32_PerfRawData_Counters_KPSSVC) GetPropertyIncomingPasswordChangeRequests() (value uint32, err error) {
	retValue, err := instance.GetProperty("IncomingPasswordChangeRequests")
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

// SetIncomingRequests sets the value of IncomingRequests for the instance
func (instance *Win32_PerfRawData_Counters_KPSSVC) SetPropertyIncomingRequests(value uint32) (err error) {
	return instance.SetProperty("IncomingRequests", (value))
}

// GetIncomingRequests gets the value of IncomingRequests for the instance
func (instance *Win32_PerfRawData_Counters_KPSSVC) GetPropertyIncomingRequests() (value uint32, err error) {
	retValue, err := instance.GetProperty("IncomingRequests")
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
