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

// Win32_PerfFormattedData_Counters_PacketDirectQueueDepth struct
type Win32_PerfFormattedData_Counters_PacketDirectQueueDepth struct {
	*Win32_PerfFormattedData

	//
	AverageQueueDepth uint32

	//
	PercentAverageQueueUtilization uint32
}

func NewWin32_PerfFormattedData_Counters_PacketDirectQueueDepthEx1(instance *cim.WmiInstance) (newInstance *Win32_PerfFormattedData_Counters_PacketDirectQueueDepth, err error) {
	tmp, err := NewWin32_PerfFormattedDataEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfFormattedData_Counters_PacketDirectQueueDepth{
		Win32_PerfFormattedData: tmp,
	}
	return
}

func NewWin32_PerfFormattedData_Counters_PacketDirectQueueDepthEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_PerfFormattedData_Counters_PacketDirectQueueDepth, err error) {
	tmp, err := NewWin32_PerfFormattedDataEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfFormattedData_Counters_PacketDirectQueueDepth{
		Win32_PerfFormattedData: tmp,
	}
	return
}

// SetAverageQueueDepth sets the value of AverageQueueDepth for the instance
func (instance *Win32_PerfFormattedData_Counters_PacketDirectQueueDepth) SetPropertyAverageQueueDepth(value uint32) (err error) {
	return instance.SetProperty("AverageQueueDepth", (value))
}

// GetAverageQueueDepth gets the value of AverageQueueDepth for the instance
func (instance *Win32_PerfFormattedData_Counters_PacketDirectQueueDepth) GetPropertyAverageQueueDepth() (value uint32, err error) {
	retValue, err := instance.GetProperty("AverageQueueDepth")
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

// SetPercentAverageQueueUtilization sets the value of PercentAverageQueueUtilization for the instance
func (instance *Win32_PerfFormattedData_Counters_PacketDirectQueueDepth) SetPropertyPercentAverageQueueUtilization(value uint32) (err error) {
	return instance.SetProperty("PercentAverageQueueUtilization", (value))
}

// GetPercentAverageQueueUtilization gets the value of PercentAverageQueueUtilization for the instance
func (instance *Win32_PerfFormattedData_Counters_PacketDirectQueueDepth) GetPropertyPercentAverageQueueUtilization() (value uint32, err error) {
	retValue, err := instance.GetProperty("PercentAverageQueueUtilization")
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
