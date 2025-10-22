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

// Win32_PerfFormattedData_VmmsVirtualMachineStats_HyperVVirtualMachineHealthSummary struct
type Win32_PerfFormattedData_VmmsVirtualMachineStats_HyperVVirtualMachineHealthSummary struct {
	*Win32_PerfFormattedData

	//
	HealthCritical uint32

	//
	HealthOk uint32
}

func NewWin32_PerfFormattedData_VmmsVirtualMachineStats_HyperVVirtualMachineHealthSummaryEx1(instance *cim.WmiInstance) (newInstance *Win32_PerfFormattedData_VmmsVirtualMachineStats_HyperVVirtualMachineHealthSummary, err error) {
	tmp, err := NewWin32_PerfFormattedDataEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfFormattedData_VmmsVirtualMachineStats_HyperVVirtualMachineHealthSummary{
		Win32_PerfFormattedData: tmp,
	}
	return
}

func NewWin32_PerfFormattedData_VmmsVirtualMachineStats_HyperVVirtualMachineHealthSummaryEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_PerfFormattedData_VmmsVirtualMachineStats_HyperVVirtualMachineHealthSummary, err error) {
	tmp, err := NewWin32_PerfFormattedDataEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfFormattedData_VmmsVirtualMachineStats_HyperVVirtualMachineHealthSummary{
		Win32_PerfFormattedData: tmp,
	}
	return
}

// SetHealthCritical sets the value of HealthCritical for the instance
func (instance *Win32_PerfFormattedData_VmmsVirtualMachineStats_HyperVVirtualMachineHealthSummary) SetPropertyHealthCritical(value uint32) (err error) {
	return instance.SetProperty("HealthCritical", (value))
}

// GetHealthCritical gets the value of HealthCritical for the instance
func (instance *Win32_PerfFormattedData_VmmsVirtualMachineStats_HyperVVirtualMachineHealthSummary) GetPropertyHealthCritical() (value uint32, err error) {
	retValue, err := instance.GetProperty("HealthCritical")
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

// SetHealthOk sets the value of HealthOk for the instance
func (instance *Win32_PerfFormattedData_VmmsVirtualMachineStats_HyperVVirtualMachineHealthSummary) SetPropertyHealthOk(value uint32) (err error) {
	return instance.SetProperty("HealthOk", (value))
}

// GetHealthOk gets the value of HealthOk for the instance
func (instance *Win32_PerfFormattedData_VmmsVirtualMachineStats_HyperVVirtualMachineHealthSummary) GetPropertyHealthOk() (value uint32, err error) {
	retValue, err := instance.GetProperty("HealthOk")
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
