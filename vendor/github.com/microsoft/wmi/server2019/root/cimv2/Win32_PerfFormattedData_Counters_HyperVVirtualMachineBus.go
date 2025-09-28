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

// Win32_PerfFormattedData_Counters_HyperVVirtualMachineBus struct
type Win32_PerfFormattedData_Counters_HyperVVirtualMachineBus struct {
	*Win32_PerfFormattedData

	//
	InterruptsReceivedPersec uint64

	//
	InterruptsSentPersec uint64

	//
	ThrottleEvents uint64
}

func NewWin32_PerfFormattedData_Counters_HyperVVirtualMachineBusEx1(instance *cim.WmiInstance) (newInstance *Win32_PerfFormattedData_Counters_HyperVVirtualMachineBus, err error) {
	tmp, err := NewWin32_PerfFormattedDataEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfFormattedData_Counters_HyperVVirtualMachineBus{
		Win32_PerfFormattedData: tmp,
	}
	return
}

func NewWin32_PerfFormattedData_Counters_HyperVVirtualMachineBusEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_PerfFormattedData_Counters_HyperVVirtualMachineBus, err error) {
	tmp, err := NewWin32_PerfFormattedDataEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfFormattedData_Counters_HyperVVirtualMachineBus{
		Win32_PerfFormattedData: tmp,
	}
	return
}

// SetInterruptsReceivedPersec sets the value of InterruptsReceivedPersec for the instance
func (instance *Win32_PerfFormattedData_Counters_HyperVVirtualMachineBus) SetPropertyInterruptsReceivedPersec(value uint64) (err error) {
	return instance.SetProperty("InterruptsReceivedPersec", (value))
}

// GetInterruptsReceivedPersec gets the value of InterruptsReceivedPersec for the instance
func (instance *Win32_PerfFormattedData_Counters_HyperVVirtualMachineBus) GetPropertyInterruptsReceivedPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("InterruptsReceivedPersec")
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

// SetInterruptsSentPersec sets the value of InterruptsSentPersec for the instance
func (instance *Win32_PerfFormattedData_Counters_HyperVVirtualMachineBus) SetPropertyInterruptsSentPersec(value uint64) (err error) {
	return instance.SetProperty("InterruptsSentPersec", (value))
}

// GetInterruptsSentPersec gets the value of InterruptsSentPersec for the instance
func (instance *Win32_PerfFormattedData_Counters_HyperVVirtualMachineBus) GetPropertyInterruptsSentPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("InterruptsSentPersec")
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

// SetThrottleEvents sets the value of ThrottleEvents for the instance
func (instance *Win32_PerfFormattedData_Counters_HyperVVirtualMachineBus) SetPropertyThrottleEvents(value uint64) (err error) {
	return instance.SetProperty("ThrottleEvents", (value))
}

// GetThrottleEvents gets the value of ThrottleEvents for the instance
func (instance *Win32_PerfFormattedData_Counters_HyperVVirtualMachineBus) GetPropertyThrottleEvents() (value uint64, err error) {
	retValue, err := instance.GetProperty("ThrottleEvents")
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
