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

// Win32_PerfRawData_Counters_VFPQoSQueueTotalOutboundNetworkTraffic struct
type Win32_PerfRawData_Counters_VFPQoSQueueTotalOutboundNetworkTraffic struct {
	*Win32_PerfRawData

	//
	TotalOutboundBytesDropped uint64

	//
	TotalOutboundPacketsDropped uint64
}

func NewWin32_PerfRawData_Counters_VFPQoSQueueTotalOutboundNetworkTrafficEx1(instance *cim.WmiInstance) (newInstance *Win32_PerfRawData_Counters_VFPQoSQueueTotalOutboundNetworkTraffic, err error) {
	tmp, err := NewWin32_PerfRawDataEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfRawData_Counters_VFPQoSQueueTotalOutboundNetworkTraffic{
		Win32_PerfRawData: tmp,
	}
	return
}

func NewWin32_PerfRawData_Counters_VFPQoSQueueTotalOutboundNetworkTrafficEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_PerfRawData_Counters_VFPQoSQueueTotalOutboundNetworkTraffic, err error) {
	tmp, err := NewWin32_PerfRawDataEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfRawData_Counters_VFPQoSQueueTotalOutboundNetworkTraffic{
		Win32_PerfRawData: tmp,
	}
	return
}

// SetTotalOutboundBytesDropped sets the value of TotalOutboundBytesDropped for the instance
func (instance *Win32_PerfRawData_Counters_VFPQoSQueueTotalOutboundNetworkTraffic) SetPropertyTotalOutboundBytesDropped(value uint64) (err error) {
	return instance.SetProperty("TotalOutboundBytesDropped", (value))
}

// GetTotalOutboundBytesDropped gets the value of TotalOutboundBytesDropped for the instance
func (instance *Win32_PerfRawData_Counters_VFPQoSQueueTotalOutboundNetworkTraffic) GetPropertyTotalOutboundBytesDropped() (value uint64, err error) {
	retValue, err := instance.GetProperty("TotalOutboundBytesDropped")
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

// SetTotalOutboundPacketsDropped sets the value of TotalOutboundPacketsDropped for the instance
func (instance *Win32_PerfRawData_Counters_VFPQoSQueueTotalOutboundNetworkTraffic) SetPropertyTotalOutboundPacketsDropped(value uint64) (err error) {
	return instance.SetProperty("TotalOutboundPacketsDropped", (value))
}

// GetTotalOutboundPacketsDropped gets the value of TotalOutboundPacketsDropped for the instance
func (instance *Win32_PerfRawData_Counters_VFPQoSQueueTotalOutboundNetworkTraffic) GetPropertyTotalOutboundPacketsDropped() (value uint64, err error) {
	retValue, err := instance.GetProperty("TotalOutboundPacketsDropped")
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
