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

// Win32_PerfFormattedData_Counters_HTTPServiceRequestQueues struct
type Win32_PerfFormattedData_Counters_HTTPServiceRequestQueues struct {
	*Win32_PerfFormattedData

	//
	ArrivalRate uint64

	//
	CacheHitRate uint64

	//
	CurrentQueueSize uint32

	//
	MaxQueueItemAge uint64

	//
	RejectedRequests uint64

	//
	RejectionRate uint64
}

func NewWin32_PerfFormattedData_Counters_HTTPServiceRequestQueuesEx1(instance *cim.WmiInstance) (newInstance *Win32_PerfFormattedData_Counters_HTTPServiceRequestQueues, err error) {
	tmp, err := NewWin32_PerfFormattedDataEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfFormattedData_Counters_HTTPServiceRequestQueues{
		Win32_PerfFormattedData: tmp,
	}
	return
}

func NewWin32_PerfFormattedData_Counters_HTTPServiceRequestQueuesEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_PerfFormattedData_Counters_HTTPServiceRequestQueues, err error) {
	tmp, err := NewWin32_PerfFormattedDataEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfFormattedData_Counters_HTTPServiceRequestQueues{
		Win32_PerfFormattedData: tmp,
	}
	return
}

// SetArrivalRate sets the value of ArrivalRate for the instance
func (instance *Win32_PerfFormattedData_Counters_HTTPServiceRequestQueues) SetPropertyArrivalRate(value uint64) (err error) {
	return instance.SetProperty("ArrivalRate", (value))
}

// GetArrivalRate gets the value of ArrivalRate for the instance
func (instance *Win32_PerfFormattedData_Counters_HTTPServiceRequestQueues) GetPropertyArrivalRate() (value uint64, err error) {
	retValue, err := instance.GetProperty("ArrivalRate")
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

// SetCacheHitRate sets the value of CacheHitRate for the instance
func (instance *Win32_PerfFormattedData_Counters_HTTPServiceRequestQueues) SetPropertyCacheHitRate(value uint64) (err error) {
	return instance.SetProperty("CacheHitRate", (value))
}

// GetCacheHitRate gets the value of CacheHitRate for the instance
func (instance *Win32_PerfFormattedData_Counters_HTTPServiceRequestQueues) GetPropertyCacheHitRate() (value uint64, err error) {
	retValue, err := instance.GetProperty("CacheHitRate")
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

// SetCurrentQueueSize sets the value of CurrentQueueSize for the instance
func (instance *Win32_PerfFormattedData_Counters_HTTPServiceRequestQueues) SetPropertyCurrentQueueSize(value uint32) (err error) {
	return instance.SetProperty("CurrentQueueSize", (value))
}

// GetCurrentQueueSize gets the value of CurrentQueueSize for the instance
func (instance *Win32_PerfFormattedData_Counters_HTTPServiceRequestQueues) GetPropertyCurrentQueueSize() (value uint32, err error) {
	retValue, err := instance.GetProperty("CurrentQueueSize")
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

// SetMaxQueueItemAge sets the value of MaxQueueItemAge for the instance
func (instance *Win32_PerfFormattedData_Counters_HTTPServiceRequestQueues) SetPropertyMaxQueueItemAge(value uint64) (err error) {
	return instance.SetProperty("MaxQueueItemAge", (value))
}

// GetMaxQueueItemAge gets the value of MaxQueueItemAge for the instance
func (instance *Win32_PerfFormattedData_Counters_HTTPServiceRequestQueues) GetPropertyMaxQueueItemAge() (value uint64, err error) {
	retValue, err := instance.GetProperty("MaxQueueItemAge")
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

// SetRejectedRequests sets the value of RejectedRequests for the instance
func (instance *Win32_PerfFormattedData_Counters_HTTPServiceRequestQueues) SetPropertyRejectedRequests(value uint64) (err error) {
	return instance.SetProperty("RejectedRequests", (value))
}

// GetRejectedRequests gets the value of RejectedRequests for the instance
func (instance *Win32_PerfFormattedData_Counters_HTTPServiceRequestQueues) GetPropertyRejectedRequests() (value uint64, err error) {
	retValue, err := instance.GetProperty("RejectedRequests")
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

// SetRejectionRate sets the value of RejectionRate for the instance
func (instance *Win32_PerfFormattedData_Counters_HTTPServiceRequestQueues) SetPropertyRejectionRate(value uint64) (err error) {
	return instance.SetProperty("RejectionRate", (value))
}

// GetRejectionRate gets the value of RejectionRate for the instance
func (instance *Win32_PerfFormattedData_Counters_HTTPServiceRequestQueues) GetPropertyRejectionRate() (value uint64, err error) {
	retValue, err := instance.GetProperty("RejectionRate")
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
