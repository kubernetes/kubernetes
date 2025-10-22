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

// Win32_PerfRawData_Counters_AppVClientStreamedDataPercentage struct
type Win32_PerfRawData_Counters_AppVClientStreamedDataPercentage struct {
	*Win32_PerfRawData

	//
	PrimaryFeaturePercentStreamed uint32
}

func NewWin32_PerfRawData_Counters_AppVClientStreamedDataPercentageEx1(instance *cim.WmiInstance) (newInstance *Win32_PerfRawData_Counters_AppVClientStreamedDataPercentage, err error) {
	tmp, err := NewWin32_PerfRawDataEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfRawData_Counters_AppVClientStreamedDataPercentage{
		Win32_PerfRawData: tmp,
	}
	return
}

func NewWin32_PerfRawData_Counters_AppVClientStreamedDataPercentageEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_PerfRawData_Counters_AppVClientStreamedDataPercentage, err error) {
	tmp, err := NewWin32_PerfRawDataEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfRawData_Counters_AppVClientStreamedDataPercentage{
		Win32_PerfRawData: tmp,
	}
	return
}

// SetPrimaryFeaturePercentStreamed sets the value of PrimaryFeaturePercentStreamed for the instance
func (instance *Win32_PerfRawData_Counters_AppVClientStreamedDataPercentage) SetPropertyPrimaryFeaturePercentStreamed(value uint32) (err error) {
	return instance.SetProperty("PrimaryFeaturePercentStreamed", (value))
}

// GetPrimaryFeaturePercentStreamed gets the value of PrimaryFeaturePercentStreamed for the instance
func (instance *Win32_PerfRawData_Counters_AppVClientStreamedDataPercentage) GetPropertyPrimaryFeaturePercentStreamed() (value uint32, err error) {
	retValue, err := instance.GetProperty("PrimaryFeaturePercentStreamed")
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
