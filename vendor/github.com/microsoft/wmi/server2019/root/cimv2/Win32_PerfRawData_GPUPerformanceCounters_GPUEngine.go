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

// Win32_PerfRawData_GPUPerformanceCounters_GPUEngine struct
type Win32_PerfRawData_GPUPerformanceCounters_GPUEngine struct {
	*Win32_PerfRawData

	//
	RunningTime uint64

	//
	UtilizationPercentage uint64
}

func NewWin32_PerfRawData_GPUPerformanceCounters_GPUEngineEx1(instance *cim.WmiInstance) (newInstance *Win32_PerfRawData_GPUPerformanceCounters_GPUEngine, err error) {
	tmp, err := NewWin32_PerfRawDataEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfRawData_GPUPerformanceCounters_GPUEngine{
		Win32_PerfRawData: tmp,
	}
	return
}

func NewWin32_PerfRawData_GPUPerformanceCounters_GPUEngineEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_PerfRawData_GPUPerformanceCounters_GPUEngine, err error) {
	tmp, err := NewWin32_PerfRawDataEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfRawData_GPUPerformanceCounters_GPUEngine{
		Win32_PerfRawData: tmp,
	}
	return
}

// SetRunningTime sets the value of RunningTime for the instance
func (instance *Win32_PerfRawData_GPUPerformanceCounters_GPUEngine) SetPropertyRunningTime(value uint64) (err error) {
	return instance.SetProperty("RunningTime", (value))
}

// GetRunningTime gets the value of RunningTime for the instance
func (instance *Win32_PerfRawData_GPUPerformanceCounters_GPUEngine) GetPropertyRunningTime() (value uint64, err error) {
	retValue, err := instance.GetProperty("RunningTime")
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

// SetUtilizationPercentage sets the value of UtilizationPercentage for the instance
func (instance *Win32_PerfRawData_GPUPerformanceCounters_GPUEngine) SetPropertyUtilizationPercentage(value uint64) (err error) {
	return instance.SetProperty("UtilizationPercentage", (value))
}

// GetUtilizationPercentage gets the value of UtilizationPercentage for the instance
func (instance *Win32_PerfRawData_GPUPerformanceCounters_GPUEngine) GetPropertyUtilizationPercentage() (value uint64, err error) {
	retValue, err := instance.GetProperty("UtilizationPercentage")
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
