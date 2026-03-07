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

// Win32_PerfFormattedData_Counters_XHCIInterrupter struct
type Win32_PerfFormattedData_Counters_XHCIInterrupter struct {
	*Win32_PerfFormattedData

	//
	DpcRequeueCount uint32

	//
	DPCsPersec uint32

	//
	EventRingFullCount uint32

	//
	EventsprocessedDPC uint64

	//
	InterruptsPersec uint32
}

func NewWin32_PerfFormattedData_Counters_XHCIInterrupterEx1(instance *cim.WmiInstance) (newInstance *Win32_PerfFormattedData_Counters_XHCIInterrupter, err error) {
	tmp, err := NewWin32_PerfFormattedDataEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfFormattedData_Counters_XHCIInterrupter{
		Win32_PerfFormattedData: tmp,
	}
	return
}

func NewWin32_PerfFormattedData_Counters_XHCIInterrupterEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_PerfFormattedData_Counters_XHCIInterrupter, err error) {
	tmp, err := NewWin32_PerfFormattedDataEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfFormattedData_Counters_XHCIInterrupter{
		Win32_PerfFormattedData: tmp,
	}
	return
}

// SetDpcRequeueCount sets the value of DpcRequeueCount for the instance
func (instance *Win32_PerfFormattedData_Counters_XHCIInterrupter) SetPropertyDpcRequeueCount(value uint32) (err error) {
	return instance.SetProperty("DpcRequeueCount", (value))
}

// GetDpcRequeueCount gets the value of DpcRequeueCount for the instance
func (instance *Win32_PerfFormattedData_Counters_XHCIInterrupter) GetPropertyDpcRequeueCount() (value uint32, err error) {
	retValue, err := instance.GetProperty("DpcRequeueCount")
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

// SetDPCsPersec sets the value of DPCsPersec for the instance
func (instance *Win32_PerfFormattedData_Counters_XHCIInterrupter) SetPropertyDPCsPersec(value uint32) (err error) {
	return instance.SetProperty("DPCsPersec", (value))
}

// GetDPCsPersec gets the value of DPCsPersec for the instance
func (instance *Win32_PerfFormattedData_Counters_XHCIInterrupter) GetPropertyDPCsPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("DPCsPersec")
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

// SetEventRingFullCount sets the value of EventRingFullCount for the instance
func (instance *Win32_PerfFormattedData_Counters_XHCIInterrupter) SetPropertyEventRingFullCount(value uint32) (err error) {
	return instance.SetProperty("EventRingFullCount", (value))
}

// GetEventRingFullCount gets the value of EventRingFullCount for the instance
func (instance *Win32_PerfFormattedData_Counters_XHCIInterrupter) GetPropertyEventRingFullCount() (value uint32, err error) {
	retValue, err := instance.GetProperty("EventRingFullCount")
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

// SetEventsprocessedDPC sets the value of EventsprocessedDPC for the instance
func (instance *Win32_PerfFormattedData_Counters_XHCIInterrupter) SetPropertyEventsprocessedDPC(value uint64) (err error) {
	return instance.SetProperty("EventsprocessedDPC", (value))
}

// GetEventsprocessedDPC gets the value of EventsprocessedDPC for the instance
func (instance *Win32_PerfFormattedData_Counters_XHCIInterrupter) GetPropertyEventsprocessedDPC() (value uint64, err error) {
	retValue, err := instance.GetProperty("EventsprocessedDPC")
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

// SetInterruptsPersec sets the value of InterruptsPersec for the instance
func (instance *Win32_PerfFormattedData_Counters_XHCIInterrupter) SetPropertyInterruptsPersec(value uint32) (err error) {
	return instance.SetProperty("InterruptsPersec", (value))
}

// GetInterruptsPersec gets the value of InterruptsPersec for the instance
func (instance *Win32_PerfFormattedData_Counters_XHCIInterrupter) GetPropertyInterruptsPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("InterruptsPersec")
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
