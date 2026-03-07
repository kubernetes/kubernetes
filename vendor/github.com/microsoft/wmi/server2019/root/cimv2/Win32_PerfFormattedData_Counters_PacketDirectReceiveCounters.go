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

// Win32_PerfFormattedData_Counters_PacketDirectReceiveCounters struct
type Win32_PerfFormattedData_Counters_PacketDirectReceiveCounters struct {
	*Win32_PerfFormattedData

	//
	BytesReceived uint64

	//
	BytesReceivedPersec uint64

	//
	PacketsDropped uint64

	//
	PacketsDroppedPersec uint64

	//
	PacketsReceived uint64

	//
	PacketsReceivedPersec uint64
}

func NewWin32_PerfFormattedData_Counters_PacketDirectReceiveCountersEx1(instance *cim.WmiInstance) (newInstance *Win32_PerfFormattedData_Counters_PacketDirectReceiveCounters, err error) {
	tmp, err := NewWin32_PerfFormattedDataEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfFormattedData_Counters_PacketDirectReceiveCounters{
		Win32_PerfFormattedData: tmp,
	}
	return
}

func NewWin32_PerfFormattedData_Counters_PacketDirectReceiveCountersEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_PerfFormattedData_Counters_PacketDirectReceiveCounters, err error) {
	tmp, err := NewWin32_PerfFormattedDataEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfFormattedData_Counters_PacketDirectReceiveCounters{
		Win32_PerfFormattedData: tmp,
	}
	return
}

// SetBytesReceived sets the value of BytesReceived for the instance
func (instance *Win32_PerfFormattedData_Counters_PacketDirectReceiveCounters) SetPropertyBytesReceived(value uint64) (err error) {
	return instance.SetProperty("BytesReceived", (value))
}

// GetBytesReceived gets the value of BytesReceived for the instance
func (instance *Win32_PerfFormattedData_Counters_PacketDirectReceiveCounters) GetPropertyBytesReceived() (value uint64, err error) {
	retValue, err := instance.GetProperty("BytesReceived")
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

// SetBytesReceivedPersec sets the value of BytesReceivedPersec for the instance
func (instance *Win32_PerfFormattedData_Counters_PacketDirectReceiveCounters) SetPropertyBytesReceivedPersec(value uint64) (err error) {
	return instance.SetProperty("BytesReceivedPersec", (value))
}

// GetBytesReceivedPersec gets the value of BytesReceivedPersec for the instance
func (instance *Win32_PerfFormattedData_Counters_PacketDirectReceiveCounters) GetPropertyBytesReceivedPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("BytesReceivedPersec")
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

// SetPacketsDropped sets the value of PacketsDropped for the instance
func (instance *Win32_PerfFormattedData_Counters_PacketDirectReceiveCounters) SetPropertyPacketsDropped(value uint64) (err error) {
	return instance.SetProperty("PacketsDropped", (value))
}

// GetPacketsDropped gets the value of PacketsDropped for the instance
func (instance *Win32_PerfFormattedData_Counters_PacketDirectReceiveCounters) GetPropertyPacketsDropped() (value uint64, err error) {
	retValue, err := instance.GetProperty("PacketsDropped")
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

// SetPacketsDroppedPersec sets the value of PacketsDroppedPersec for the instance
func (instance *Win32_PerfFormattedData_Counters_PacketDirectReceiveCounters) SetPropertyPacketsDroppedPersec(value uint64) (err error) {
	return instance.SetProperty("PacketsDroppedPersec", (value))
}

// GetPacketsDroppedPersec gets the value of PacketsDroppedPersec for the instance
func (instance *Win32_PerfFormattedData_Counters_PacketDirectReceiveCounters) GetPropertyPacketsDroppedPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("PacketsDroppedPersec")
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

// SetPacketsReceived sets the value of PacketsReceived for the instance
func (instance *Win32_PerfFormattedData_Counters_PacketDirectReceiveCounters) SetPropertyPacketsReceived(value uint64) (err error) {
	return instance.SetProperty("PacketsReceived", (value))
}

// GetPacketsReceived gets the value of PacketsReceived for the instance
func (instance *Win32_PerfFormattedData_Counters_PacketDirectReceiveCounters) GetPropertyPacketsReceived() (value uint64, err error) {
	retValue, err := instance.GetProperty("PacketsReceived")
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

// SetPacketsReceivedPersec sets the value of PacketsReceivedPersec for the instance
func (instance *Win32_PerfFormattedData_Counters_PacketDirectReceiveCounters) SetPropertyPacketsReceivedPersec(value uint64) (err error) {
	return instance.SetProperty("PacketsReceivedPersec", (value))
}

// GetPacketsReceivedPersec gets the value of PacketsReceivedPersec for the instance
func (instance *Win32_PerfFormattedData_Counters_PacketDirectReceiveCounters) GetPropertyPacketsReceivedPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("PacketsReceivedPersec")
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
