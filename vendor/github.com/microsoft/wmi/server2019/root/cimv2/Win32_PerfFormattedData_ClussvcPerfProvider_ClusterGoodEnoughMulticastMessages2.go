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

// Win32_PerfFormattedData_ClussvcPerfProvider_ClusterGoodEnoughMulticastMessages2 struct
type Win32_PerfFormattedData_ClussvcPerfProvider_ClusterGoodEnoughMulticastMessages2 struct {
	*Win32_PerfFormattedData

	//
	UnacknowledgedMessageCount uint64
}

func NewWin32_PerfFormattedData_ClussvcPerfProvider_ClusterGoodEnoughMulticastMessages2Ex1(instance *cim.WmiInstance) (newInstance *Win32_PerfFormattedData_ClussvcPerfProvider_ClusterGoodEnoughMulticastMessages2, err error) {
	tmp, err := NewWin32_PerfFormattedDataEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfFormattedData_ClussvcPerfProvider_ClusterGoodEnoughMulticastMessages2{
		Win32_PerfFormattedData: tmp,
	}
	return
}

func NewWin32_PerfFormattedData_ClussvcPerfProvider_ClusterGoodEnoughMulticastMessages2Ex6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_PerfFormattedData_ClussvcPerfProvider_ClusterGoodEnoughMulticastMessages2, err error) {
	tmp, err := NewWin32_PerfFormattedDataEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfFormattedData_ClussvcPerfProvider_ClusterGoodEnoughMulticastMessages2{
		Win32_PerfFormattedData: tmp,
	}
	return
}

// SetUnacknowledgedMessageCount sets the value of UnacknowledgedMessageCount for the instance
func (instance *Win32_PerfFormattedData_ClussvcPerfProvider_ClusterGoodEnoughMulticastMessages2) SetPropertyUnacknowledgedMessageCount(value uint64) (err error) {
	return instance.SetProperty("UnacknowledgedMessageCount", (value))
}

// GetUnacknowledgedMessageCount gets the value of UnacknowledgedMessageCount for the instance
func (instance *Win32_PerfFormattedData_ClussvcPerfProvider_ClusterGoodEnoughMulticastMessages2) GetPropertyUnacknowledgedMessageCount() (value uint64, err error) {
	retValue, err := instance.GetProperty("UnacknowledgedMessageCount")
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
