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

// Win32_Perf struct
type Win32_Perf struct {
	*CIM_StatisticalInformation

	//
	Frequency_Object uint64

	//
	Frequency_PerfTime uint64

	//
	Frequency_Sys100NS uint64

	//
	Timestamp_Object uint64

	//
	Timestamp_PerfTime uint64

	//
	Timestamp_Sys100NS uint64
}

func NewWin32_PerfEx1(instance *cim.WmiInstance) (newInstance *Win32_Perf, err error) {
	tmp, err := NewCIM_StatisticalInformationEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_Perf{
		CIM_StatisticalInformation: tmp,
	}
	return
}

func NewWin32_PerfEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_Perf, err error) {
	tmp, err := NewCIM_StatisticalInformationEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_Perf{
		CIM_StatisticalInformation: tmp,
	}
	return
}

// SetFrequency_Object sets the value of Frequency_Object for the instance
func (instance *Win32_Perf) SetPropertyFrequency_Object(value uint64) (err error) {
	return instance.SetProperty("Frequency_Object", (value))
}

// GetFrequency_Object gets the value of Frequency_Object for the instance
func (instance *Win32_Perf) GetPropertyFrequency_Object() (value uint64, err error) {
	retValue, err := instance.GetProperty("Frequency_Object")
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

// SetFrequency_PerfTime sets the value of Frequency_PerfTime for the instance
func (instance *Win32_Perf) SetPropertyFrequency_PerfTime(value uint64) (err error) {
	return instance.SetProperty("Frequency_PerfTime", (value))
}

// GetFrequency_PerfTime gets the value of Frequency_PerfTime for the instance
func (instance *Win32_Perf) GetPropertyFrequency_PerfTime() (value uint64, err error) {
	retValue, err := instance.GetProperty("Frequency_PerfTime")
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

// SetFrequency_Sys100NS sets the value of Frequency_Sys100NS for the instance
func (instance *Win32_Perf) SetPropertyFrequency_Sys100NS(value uint64) (err error) {
	return instance.SetProperty("Frequency_Sys100NS", (value))
}

// GetFrequency_Sys100NS gets the value of Frequency_Sys100NS for the instance
func (instance *Win32_Perf) GetPropertyFrequency_Sys100NS() (value uint64, err error) {
	retValue, err := instance.GetProperty("Frequency_Sys100NS")
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

// SetTimestamp_Object sets the value of Timestamp_Object for the instance
func (instance *Win32_Perf) SetPropertyTimestamp_Object(value uint64) (err error) {
	return instance.SetProperty("Timestamp_Object", (value))
}

// GetTimestamp_Object gets the value of Timestamp_Object for the instance
func (instance *Win32_Perf) GetPropertyTimestamp_Object() (value uint64, err error) {
	retValue, err := instance.GetProperty("Timestamp_Object")
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

// SetTimestamp_PerfTime sets the value of Timestamp_PerfTime for the instance
func (instance *Win32_Perf) SetPropertyTimestamp_PerfTime(value uint64) (err error) {
	return instance.SetProperty("Timestamp_PerfTime", (value))
}

// GetTimestamp_PerfTime gets the value of Timestamp_PerfTime for the instance
func (instance *Win32_Perf) GetPropertyTimestamp_PerfTime() (value uint64, err error) {
	retValue, err := instance.GetProperty("Timestamp_PerfTime")
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

// SetTimestamp_Sys100NS sets the value of Timestamp_Sys100NS for the instance
func (instance *Win32_Perf) SetPropertyTimestamp_Sys100NS(value uint64) (err error) {
	return instance.SetProperty("Timestamp_Sys100NS", (value))
}

// GetTimestamp_Sys100NS gets the value of Timestamp_Sys100NS for the instance
func (instance *Win32_Perf) GetPropertyTimestamp_Sys100NS() (value uint64, err error) {
	retValue, err := instance.GetProperty("Timestamp_Sys100NS")
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
