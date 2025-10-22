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

// Win32_PerfFormattedData_NvspSwitchProcStats_HyperVVirtualSwitchProcessor struct
type Win32_PerfFormattedData_NvspSwitchProcStats_HyperVVirtualSwitchProcessor struct {
	*Win32_PerfFormattedData

	//
	NumberofTransmitCompletesPersec uint64

	//
	NumberofVMQs uint64

	//
	PacketsfromExternalPersec uint64

	//
	PacketsfromInternalPersec uint64
}

func NewWin32_PerfFormattedData_NvspSwitchProcStats_HyperVVirtualSwitchProcessorEx1(instance *cim.WmiInstance) (newInstance *Win32_PerfFormattedData_NvspSwitchProcStats_HyperVVirtualSwitchProcessor, err error) {
	tmp, err := NewWin32_PerfFormattedDataEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfFormattedData_NvspSwitchProcStats_HyperVVirtualSwitchProcessor{
		Win32_PerfFormattedData: tmp,
	}
	return
}

func NewWin32_PerfFormattedData_NvspSwitchProcStats_HyperVVirtualSwitchProcessorEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_PerfFormattedData_NvspSwitchProcStats_HyperVVirtualSwitchProcessor, err error) {
	tmp, err := NewWin32_PerfFormattedDataEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfFormattedData_NvspSwitchProcStats_HyperVVirtualSwitchProcessor{
		Win32_PerfFormattedData: tmp,
	}
	return
}

// SetNumberofTransmitCompletesPersec sets the value of NumberofTransmitCompletesPersec for the instance
func (instance *Win32_PerfFormattedData_NvspSwitchProcStats_HyperVVirtualSwitchProcessor) SetPropertyNumberofTransmitCompletesPersec(value uint64) (err error) {
	return instance.SetProperty("NumberofTransmitCompletesPersec", (value))
}

// GetNumberofTransmitCompletesPersec gets the value of NumberofTransmitCompletesPersec for the instance
func (instance *Win32_PerfFormattedData_NvspSwitchProcStats_HyperVVirtualSwitchProcessor) GetPropertyNumberofTransmitCompletesPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("NumberofTransmitCompletesPersec")
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

// SetNumberofVMQs sets the value of NumberofVMQs for the instance
func (instance *Win32_PerfFormattedData_NvspSwitchProcStats_HyperVVirtualSwitchProcessor) SetPropertyNumberofVMQs(value uint64) (err error) {
	return instance.SetProperty("NumberofVMQs", (value))
}

// GetNumberofVMQs gets the value of NumberofVMQs for the instance
func (instance *Win32_PerfFormattedData_NvspSwitchProcStats_HyperVVirtualSwitchProcessor) GetPropertyNumberofVMQs() (value uint64, err error) {
	retValue, err := instance.GetProperty("NumberofVMQs")
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

// SetPacketsfromExternalPersec sets the value of PacketsfromExternalPersec for the instance
func (instance *Win32_PerfFormattedData_NvspSwitchProcStats_HyperVVirtualSwitchProcessor) SetPropertyPacketsfromExternalPersec(value uint64) (err error) {
	return instance.SetProperty("PacketsfromExternalPersec", (value))
}

// GetPacketsfromExternalPersec gets the value of PacketsfromExternalPersec for the instance
func (instance *Win32_PerfFormattedData_NvspSwitchProcStats_HyperVVirtualSwitchProcessor) GetPropertyPacketsfromExternalPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("PacketsfromExternalPersec")
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

// SetPacketsfromInternalPersec sets the value of PacketsfromInternalPersec for the instance
func (instance *Win32_PerfFormattedData_NvspSwitchProcStats_HyperVVirtualSwitchProcessor) SetPropertyPacketsfromInternalPersec(value uint64) (err error) {
	return instance.SetProperty("PacketsfromInternalPersec", (value))
}

// GetPacketsfromInternalPersec gets the value of PacketsfromInternalPersec for the instance
func (instance *Win32_PerfFormattedData_NvspSwitchProcStats_HyperVVirtualSwitchProcessor) GetPropertyPacketsfromInternalPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("PacketsfromInternalPersec")
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
