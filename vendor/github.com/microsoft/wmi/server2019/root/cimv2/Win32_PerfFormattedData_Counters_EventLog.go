// Copyright 2019 (c) Microsoft Corporation.
// Licensed under the MIT license.

//
// Author:
//      Auto Generated on 3/19/2020 using wmigen
//      Source root.CIMV2
//////////////////////////////////////////////
package cimv2

import (
	"github.com/microsoft/wmi/pkg/base/query"
	cim "github.com/microsoft/wmi/pkg/wmiinstance"
)

// Win32_PerfFormattedData_Counters_EventLog struct
type Win32_PerfFormattedData_Counters_EventLog struct {
	*Win32_PerfFormattedData

	//
	Activesubscriptions uint32

	//
	ELFRPCcallsPersec uint64

	//
	EnabledChannels uint32

	//
	EventfilteroperationsPersec uint64

	//
	EventsPersec uint64

	//
	WEVTRPCcallsPersec uint64
}

func NewWin32_PerfFormattedData_Counters_EventLogEx1(instance *cim.WmiInstance) (newInstance *Win32_PerfFormattedData_Counters_EventLog, err error) {
	tmp, err := NewWin32_PerfFormattedDataEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfFormattedData_Counters_EventLog{
		Win32_PerfFormattedData: tmp,
	}
	return
}

func NewWin32_PerfFormattedData_Counters_EventLogEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_PerfFormattedData_Counters_EventLog, err error) {
	tmp, err := NewWin32_PerfFormattedDataEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfFormattedData_Counters_EventLog{
		Win32_PerfFormattedData: tmp,
	}
	return
}

// SetActivesubscriptions sets the value of Activesubscriptions for the instance
func (instance *Win32_PerfFormattedData_Counters_EventLog) SetPropertyActivesubscriptions(value uint32) (err error) {
	return instance.SetProperty("Activesubscriptions", value)
}

// GetActivesubscriptions gets the value of Activesubscriptions for the instance
func (instance *Win32_PerfFormattedData_Counters_EventLog) GetPropertyActivesubscriptions() (value uint32, err error) {
	retValue, err := instance.GetProperty("Activesubscriptions")
	if err != nil {
		return
	}
	value, ok := retValue.(uint32)
	if !ok {
		// TODO: Set an error
	}
	return
}

// SetELFRPCcallsPersec sets the value of ELFRPCcallsPersec for the instance
func (instance *Win32_PerfFormattedData_Counters_EventLog) SetPropertyELFRPCcallsPersec(value uint64) (err error) {
	return instance.SetProperty("ELFRPCcallsPersec", value)
}

// GetELFRPCcallsPersec gets the value of ELFRPCcallsPersec for the instance
func (instance *Win32_PerfFormattedData_Counters_EventLog) GetPropertyELFRPCcallsPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("ELFRPCcallsPersec")
	if err != nil {
		return
	}
	value, ok := retValue.(uint64)
	if !ok {
		// TODO: Set an error
	}
	return
}

// SetEnabledChannels sets the value of EnabledChannels for the instance
func (instance *Win32_PerfFormattedData_Counters_EventLog) SetPropertyEnabledChannels(value uint32) (err error) {
	return instance.SetProperty("EnabledChannels", value)
}

// GetEnabledChannels gets the value of EnabledChannels for the instance
func (instance *Win32_PerfFormattedData_Counters_EventLog) GetPropertyEnabledChannels() (value uint32, err error) {
	retValue, err := instance.GetProperty("EnabledChannels")
	if err != nil {
		return
	}
	value, ok := retValue.(uint32)
	if !ok {
		// TODO: Set an error
	}
	return
}

// SetEventfilteroperationsPersec sets the value of EventfilteroperationsPersec for the instance
func (instance *Win32_PerfFormattedData_Counters_EventLog) SetPropertyEventfilteroperationsPersec(value uint64) (err error) {
	return instance.SetProperty("EventfilteroperationsPersec", value)
}

// GetEventfilteroperationsPersec gets the value of EventfilteroperationsPersec for the instance
func (instance *Win32_PerfFormattedData_Counters_EventLog) GetPropertyEventfilteroperationsPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("EventfilteroperationsPersec")
	if err != nil {
		return
	}
	value, ok := retValue.(uint64)
	if !ok {
		// TODO: Set an error
	}
	return
}

// SetEventsPersec sets the value of EventsPersec for the instance
func (instance *Win32_PerfFormattedData_Counters_EventLog) SetPropertyEventsPersec(value uint64) (err error) {
	return instance.SetProperty("EventsPersec", value)
}

// GetEventsPersec gets the value of EventsPersec for the instance
func (instance *Win32_PerfFormattedData_Counters_EventLog) GetPropertyEventsPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("EventsPersec")
	if err != nil {
		return
	}
	value, ok := retValue.(uint64)
	if !ok {
		// TODO: Set an error
	}
	return
}

// SetWEVTRPCcallsPersec sets the value of WEVTRPCcallsPersec for the instance
func (instance *Win32_PerfFormattedData_Counters_EventLog) SetPropertyWEVTRPCcallsPersec(value uint64) (err error) {
	return instance.SetProperty("WEVTRPCcallsPersec", value)
}

// GetWEVTRPCcallsPersec gets the value of WEVTRPCcallsPersec for the instance
func (instance *Win32_PerfFormattedData_Counters_EventLog) GetPropertyWEVTRPCcallsPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("WEVTRPCcallsPersec")
	if err != nil {
		return
	}
	value, ok := retValue.(uint64)
	if !ok {
		// TODO: Set an error
	}
	return
}
