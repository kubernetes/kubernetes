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

// Win32_PerfFormattedData_Counters_PerProcessorNetworkActivityCycles struct
type Win32_PerfFormattedData_Counters_PerProcessorNetworkActivityCycles struct {
	*Win32_PerfFormattedData

	//
	BuildScatterGatherCyclesPersec uint64

	//
	InterruptCyclesPersec uint64

	//
	InterruptDPCCyclesPersec uint64

	//
	InterruptDPCLatencyCyclesPersec uint64

	//
	MiniportReturnPacketCyclesPersec uint64

	//
	MiniportRSSIndirectionTableChangeCycles uint64

	//
	MiniportSendCyclesPersec uint64

	//
	NDISReceiveIndicationCyclesPersec uint64

	//
	NDISReturnPacketCyclesPersec uint64

	//
	NDISSendCompleteCyclesPersec uint64

	//
	NDISSendCyclesPersec uint64

	//
	StackReceiveIndicationCyclesPersec uint64

	//
	StackSendCompleteCyclesPersec uint64
}

func NewWin32_PerfFormattedData_Counters_PerProcessorNetworkActivityCyclesEx1(instance *cim.WmiInstance) (newInstance *Win32_PerfFormattedData_Counters_PerProcessorNetworkActivityCycles, err error) {
	tmp, err := NewWin32_PerfFormattedDataEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfFormattedData_Counters_PerProcessorNetworkActivityCycles{
		Win32_PerfFormattedData: tmp,
	}
	return
}

func NewWin32_PerfFormattedData_Counters_PerProcessorNetworkActivityCyclesEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_PerfFormattedData_Counters_PerProcessorNetworkActivityCycles, err error) {
	tmp, err := NewWin32_PerfFormattedDataEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfFormattedData_Counters_PerProcessorNetworkActivityCycles{
		Win32_PerfFormattedData: tmp,
	}
	return
}

// SetBuildScatterGatherCyclesPersec sets the value of BuildScatterGatherCyclesPersec for the instance
func (instance *Win32_PerfFormattedData_Counters_PerProcessorNetworkActivityCycles) SetPropertyBuildScatterGatherCyclesPersec(value uint64) (err error) {
	return instance.SetProperty("BuildScatterGatherCyclesPersec", (value))
}

// GetBuildScatterGatherCyclesPersec gets the value of BuildScatterGatherCyclesPersec for the instance
func (instance *Win32_PerfFormattedData_Counters_PerProcessorNetworkActivityCycles) GetPropertyBuildScatterGatherCyclesPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("BuildScatterGatherCyclesPersec")
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

// SetInterruptCyclesPersec sets the value of InterruptCyclesPersec for the instance
func (instance *Win32_PerfFormattedData_Counters_PerProcessorNetworkActivityCycles) SetPropertyInterruptCyclesPersec(value uint64) (err error) {
	return instance.SetProperty("InterruptCyclesPersec", (value))
}

// GetInterruptCyclesPersec gets the value of InterruptCyclesPersec for the instance
func (instance *Win32_PerfFormattedData_Counters_PerProcessorNetworkActivityCycles) GetPropertyInterruptCyclesPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("InterruptCyclesPersec")
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

// SetInterruptDPCCyclesPersec sets the value of InterruptDPCCyclesPersec for the instance
func (instance *Win32_PerfFormattedData_Counters_PerProcessorNetworkActivityCycles) SetPropertyInterruptDPCCyclesPersec(value uint64) (err error) {
	return instance.SetProperty("InterruptDPCCyclesPersec", (value))
}

// GetInterruptDPCCyclesPersec gets the value of InterruptDPCCyclesPersec for the instance
func (instance *Win32_PerfFormattedData_Counters_PerProcessorNetworkActivityCycles) GetPropertyInterruptDPCCyclesPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("InterruptDPCCyclesPersec")
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

// SetInterruptDPCLatencyCyclesPersec sets the value of InterruptDPCLatencyCyclesPersec for the instance
func (instance *Win32_PerfFormattedData_Counters_PerProcessorNetworkActivityCycles) SetPropertyInterruptDPCLatencyCyclesPersec(value uint64) (err error) {
	return instance.SetProperty("InterruptDPCLatencyCyclesPersec", (value))
}

// GetInterruptDPCLatencyCyclesPersec gets the value of InterruptDPCLatencyCyclesPersec for the instance
func (instance *Win32_PerfFormattedData_Counters_PerProcessorNetworkActivityCycles) GetPropertyInterruptDPCLatencyCyclesPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("InterruptDPCLatencyCyclesPersec")
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

// SetMiniportReturnPacketCyclesPersec sets the value of MiniportReturnPacketCyclesPersec for the instance
func (instance *Win32_PerfFormattedData_Counters_PerProcessorNetworkActivityCycles) SetPropertyMiniportReturnPacketCyclesPersec(value uint64) (err error) {
	return instance.SetProperty("MiniportReturnPacketCyclesPersec", (value))
}

// GetMiniportReturnPacketCyclesPersec gets the value of MiniportReturnPacketCyclesPersec for the instance
func (instance *Win32_PerfFormattedData_Counters_PerProcessorNetworkActivityCycles) GetPropertyMiniportReturnPacketCyclesPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("MiniportReturnPacketCyclesPersec")
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

// SetMiniportRSSIndirectionTableChangeCycles sets the value of MiniportRSSIndirectionTableChangeCycles for the instance
func (instance *Win32_PerfFormattedData_Counters_PerProcessorNetworkActivityCycles) SetPropertyMiniportRSSIndirectionTableChangeCycles(value uint64) (err error) {
	return instance.SetProperty("MiniportRSSIndirectionTableChangeCycles", (value))
}

// GetMiniportRSSIndirectionTableChangeCycles gets the value of MiniportRSSIndirectionTableChangeCycles for the instance
func (instance *Win32_PerfFormattedData_Counters_PerProcessorNetworkActivityCycles) GetPropertyMiniportRSSIndirectionTableChangeCycles() (value uint64, err error) {
	retValue, err := instance.GetProperty("MiniportRSSIndirectionTableChangeCycles")
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

// SetMiniportSendCyclesPersec sets the value of MiniportSendCyclesPersec for the instance
func (instance *Win32_PerfFormattedData_Counters_PerProcessorNetworkActivityCycles) SetPropertyMiniportSendCyclesPersec(value uint64) (err error) {
	return instance.SetProperty("MiniportSendCyclesPersec", (value))
}

// GetMiniportSendCyclesPersec gets the value of MiniportSendCyclesPersec for the instance
func (instance *Win32_PerfFormattedData_Counters_PerProcessorNetworkActivityCycles) GetPropertyMiniportSendCyclesPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("MiniportSendCyclesPersec")
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

// SetNDISReceiveIndicationCyclesPersec sets the value of NDISReceiveIndicationCyclesPersec for the instance
func (instance *Win32_PerfFormattedData_Counters_PerProcessorNetworkActivityCycles) SetPropertyNDISReceiveIndicationCyclesPersec(value uint64) (err error) {
	return instance.SetProperty("NDISReceiveIndicationCyclesPersec", (value))
}

// GetNDISReceiveIndicationCyclesPersec gets the value of NDISReceiveIndicationCyclesPersec for the instance
func (instance *Win32_PerfFormattedData_Counters_PerProcessorNetworkActivityCycles) GetPropertyNDISReceiveIndicationCyclesPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("NDISReceiveIndicationCyclesPersec")
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

// SetNDISReturnPacketCyclesPersec sets the value of NDISReturnPacketCyclesPersec for the instance
func (instance *Win32_PerfFormattedData_Counters_PerProcessorNetworkActivityCycles) SetPropertyNDISReturnPacketCyclesPersec(value uint64) (err error) {
	return instance.SetProperty("NDISReturnPacketCyclesPersec", (value))
}

// GetNDISReturnPacketCyclesPersec gets the value of NDISReturnPacketCyclesPersec for the instance
func (instance *Win32_PerfFormattedData_Counters_PerProcessorNetworkActivityCycles) GetPropertyNDISReturnPacketCyclesPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("NDISReturnPacketCyclesPersec")
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

// SetNDISSendCompleteCyclesPersec sets the value of NDISSendCompleteCyclesPersec for the instance
func (instance *Win32_PerfFormattedData_Counters_PerProcessorNetworkActivityCycles) SetPropertyNDISSendCompleteCyclesPersec(value uint64) (err error) {
	return instance.SetProperty("NDISSendCompleteCyclesPersec", (value))
}

// GetNDISSendCompleteCyclesPersec gets the value of NDISSendCompleteCyclesPersec for the instance
func (instance *Win32_PerfFormattedData_Counters_PerProcessorNetworkActivityCycles) GetPropertyNDISSendCompleteCyclesPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("NDISSendCompleteCyclesPersec")
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

// SetNDISSendCyclesPersec sets the value of NDISSendCyclesPersec for the instance
func (instance *Win32_PerfFormattedData_Counters_PerProcessorNetworkActivityCycles) SetPropertyNDISSendCyclesPersec(value uint64) (err error) {
	return instance.SetProperty("NDISSendCyclesPersec", (value))
}

// GetNDISSendCyclesPersec gets the value of NDISSendCyclesPersec for the instance
func (instance *Win32_PerfFormattedData_Counters_PerProcessorNetworkActivityCycles) GetPropertyNDISSendCyclesPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("NDISSendCyclesPersec")
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

// SetStackReceiveIndicationCyclesPersec sets the value of StackReceiveIndicationCyclesPersec for the instance
func (instance *Win32_PerfFormattedData_Counters_PerProcessorNetworkActivityCycles) SetPropertyStackReceiveIndicationCyclesPersec(value uint64) (err error) {
	return instance.SetProperty("StackReceiveIndicationCyclesPersec", (value))
}

// GetStackReceiveIndicationCyclesPersec gets the value of StackReceiveIndicationCyclesPersec for the instance
func (instance *Win32_PerfFormattedData_Counters_PerProcessorNetworkActivityCycles) GetPropertyStackReceiveIndicationCyclesPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("StackReceiveIndicationCyclesPersec")
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

// SetStackSendCompleteCyclesPersec sets the value of StackSendCompleteCyclesPersec for the instance
func (instance *Win32_PerfFormattedData_Counters_PerProcessorNetworkActivityCycles) SetPropertyStackSendCompleteCyclesPersec(value uint64) (err error) {
	return instance.SetProperty("StackSendCompleteCyclesPersec", (value))
}

// GetStackSendCompleteCyclesPersec gets the value of StackSendCompleteCyclesPersec for the instance
func (instance *Win32_PerfFormattedData_Counters_PerProcessorNetworkActivityCycles) GetPropertyStackSendCompleteCyclesPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("StackSendCompleteCyclesPersec")
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
