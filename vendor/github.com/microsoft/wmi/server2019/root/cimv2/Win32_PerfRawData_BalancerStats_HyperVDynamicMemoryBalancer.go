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

// Win32_PerfRawData_BalancerStats_HyperVDynamicMemoryBalancer struct
type Win32_PerfRawData_BalancerStats_HyperVDynamicMemoryBalancer struct {
	*Win32_PerfRawData

	//
	AvailableMemory uint32

	//
	AvailableMemoryForBalancing uint32

	//
	AveragePressure uint32

	//
	SystemCurrentPressure uint32
}

func NewWin32_PerfRawData_BalancerStats_HyperVDynamicMemoryBalancerEx1(instance *cim.WmiInstance) (newInstance *Win32_PerfRawData_BalancerStats_HyperVDynamicMemoryBalancer, err error) {
	tmp, err := NewWin32_PerfRawDataEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfRawData_BalancerStats_HyperVDynamicMemoryBalancer{
		Win32_PerfRawData: tmp,
	}
	return
}

func NewWin32_PerfRawData_BalancerStats_HyperVDynamicMemoryBalancerEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_PerfRawData_BalancerStats_HyperVDynamicMemoryBalancer, err error) {
	tmp, err := NewWin32_PerfRawDataEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfRawData_BalancerStats_HyperVDynamicMemoryBalancer{
		Win32_PerfRawData: tmp,
	}
	return
}

// SetAvailableMemory sets the value of AvailableMemory for the instance
func (instance *Win32_PerfRawData_BalancerStats_HyperVDynamicMemoryBalancer) SetPropertyAvailableMemory(value uint32) (err error) {
	return instance.SetProperty("AvailableMemory", (value))
}

// GetAvailableMemory gets the value of AvailableMemory for the instance
func (instance *Win32_PerfRawData_BalancerStats_HyperVDynamicMemoryBalancer) GetPropertyAvailableMemory() (value uint32, err error) {
	retValue, err := instance.GetProperty("AvailableMemory")
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

// SetAvailableMemoryForBalancing sets the value of AvailableMemoryForBalancing for the instance
func (instance *Win32_PerfRawData_BalancerStats_HyperVDynamicMemoryBalancer) SetPropertyAvailableMemoryForBalancing(value uint32) (err error) {
	return instance.SetProperty("AvailableMemoryForBalancing", (value))
}

// GetAvailableMemoryForBalancing gets the value of AvailableMemoryForBalancing for the instance
func (instance *Win32_PerfRawData_BalancerStats_HyperVDynamicMemoryBalancer) GetPropertyAvailableMemoryForBalancing() (value uint32, err error) {
	retValue, err := instance.GetProperty("AvailableMemoryForBalancing")
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

// SetAveragePressure sets the value of AveragePressure for the instance
func (instance *Win32_PerfRawData_BalancerStats_HyperVDynamicMemoryBalancer) SetPropertyAveragePressure(value uint32) (err error) {
	return instance.SetProperty("AveragePressure", (value))
}

// GetAveragePressure gets the value of AveragePressure for the instance
func (instance *Win32_PerfRawData_BalancerStats_HyperVDynamicMemoryBalancer) GetPropertyAveragePressure() (value uint32, err error) {
	retValue, err := instance.GetProperty("AveragePressure")
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

// SetSystemCurrentPressure sets the value of SystemCurrentPressure for the instance
func (instance *Win32_PerfRawData_BalancerStats_HyperVDynamicMemoryBalancer) SetPropertySystemCurrentPressure(value uint32) (err error) {
	return instance.SetProperty("SystemCurrentPressure", (value))
}

// GetSystemCurrentPressure gets the value of SystemCurrentPressure for the instance
func (instance *Win32_PerfRawData_BalancerStats_HyperVDynamicMemoryBalancer) GetPropertySystemCurrentPressure() (value uint32, err error) {
	retValue, err := instance.GetProperty("SystemCurrentPressure")
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
