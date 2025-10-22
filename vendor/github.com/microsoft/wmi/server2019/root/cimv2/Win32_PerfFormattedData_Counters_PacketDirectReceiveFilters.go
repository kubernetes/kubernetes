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

// Win32_PerfFormattedData_Counters_PacketDirectReceiveFilters struct
type Win32_PerfFormattedData_Counters_PacketDirectReceiveFilters struct {
	*Win32_PerfFormattedData

	//
	BytesMatched uint64

	//
	BytesMatchedPersec uint64

	//
	PacketsMatched uint64

	//
	PacketsMatchedPersec uint64
}

func NewWin32_PerfFormattedData_Counters_PacketDirectReceiveFiltersEx1(instance *cim.WmiInstance) (newInstance *Win32_PerfFormattedData_Counters_PacketDirectReceiveFilters, err error) {
	tmp, err := NewWin32_PerfFormattedDataEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfFormattedData_Counters_PacketDirectReceiveFilters{
		Win32_PerfFormattedData: tmp,
	}
	return
}

func NewWin32_PerfFormattedData_Counters_PacketDirectReceiveFiltersEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_PerfFormattedData_Counters_PacketDirectReceiveFilters, err error) {
	tmp, err := NewWin32_PerfFormattedDataEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfFormattedData_Counters_PacketDirectReceiveFilters{
		Win32_PerfFormattedData: tmp,
	}
	return
}

// SetBytesMatched sets the value of BytesMatched for the instance
func (instance *Win32_PerfFormattedData_Counters_PacketDirectReceiveFilters) SetPropertyBytesMatched(value uint64) (err error) {
	return instance.SetProperty("BytesMatched", (value))
}

// GetBytesMatched gets the value of BytesMatched for the instance
func (instance *Win32_PerfFormattedData_Counters_PacketDirectReceiveFilters) GetPropertyBytesMatched() (value uint64, err error) {
	retValue, err := instance.GetProperty("BytesMatched")
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

// SetBytesMatchedPersec sets the value of BytesMatchedPersec for the instance
func (instance *Win32_PerfFormattedData_Counters_PacketDirectReceiveFilters) SetPropertyBytesMatchedPersec(value uint64) (err error) {
	return instance.SetProperty("BytesMatchedPersec", (value))
}

// GetBytesMatchedPersec gets the value of BytesMatchedPersec for the instance
func (instance *Win32_PerfFormattedData_Counters_PacketDirectReceiveFilters) GetPropertyBytesMatchedPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("BytesMatchedPersec")
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

// SetPacketsMatched sets the value of PacketsMatched for the instance
func (instance *Win32_PerfFormattedData_Counters_PacketDirectReceiveFilters) SetPropertyPacketsMatched(value uint64) (err error) {
	return instance.SetProperty("PacketsMatched", (value))
}

// GetPacketsMatched gets the value of PacketsMatched for the instance
func (instance *Win32_PerfFormattedData_Counters_PacketDirectReceiveFilters) GetPropertyPacketsMatched() (value uint64, err error) {
	retValue, err := instance.GetProperty("PacketsMatched")
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

// SetPacketsMatchedPersec sets the value of PacketsMatchedPersec for the instance
func (instance *Win32_PerfFormattedData_Counters_PacketDirectReceiveFilters) SetPropertyPacketsMatchedPersec(value uint64) (err error) {
	return instance.SetProperty("PacketsMatchedPersec", (value))
}

// GetPacketsMatchedPersec gets the value of PacketsMatchedPersec for the instance
func (instance *Win32_PerfFormattedData_Counters_PacketDirectReceiveFilters) GetPropertyPacketsMatchedPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("PacketsMatchedPersec")
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
