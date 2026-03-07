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

// Win32_PerfRawData_Counters_PacketDirectTransmitCounters struct
type Win32_PerfRawData_Counters_PacketDirectTransmitCounters struct {
	*Win32_PerfRawData

	//
	BytesTransmitted uint64

	//
	BytesTransmittedPersec uint64

	//
	PacketsTransmitted uint64

	//
	PacketsTransmittedPersec uint64
}

func NewWin32_PerfRawData_Counters_PacketDirectTransmitCountersEx1(instance *cim.WmiInstance) (newInstance *Win32_PerfRawData_Counters_PacketDirectTransmitCounters, err error) {
	tmp, err := NewWin32_PerfRawDataEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfRawData_Counters_PacketDirectTransmitCounters{
		Win32_PerfRawData: tmp,
	}
	return
}

func NewWin32_PerfRawData_Counters_PacketDirectTransmitCountersEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_PerfRawData_Counters_PacketDirectTransmitCounters, err error) {
	tmp, err := NewWin32_PerfRawDataEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfRawData_Counters_PacketDirectTransmitCounters{
		Win32_PerfRawData: tmp,
	}
	return
}

// SetBytesTransmitted sets the value of BytesTransmitted for the instance
func (instance *Win32_PerfRawData_Counters_PacketDirectTransmitCounters) SetPropertyBytesTransmitted(value uint64) (err error) {
	return instance.SetProperty("BytesTransmitted", (value))
}

// GetBytesTransmitted gets the value of BytesTransmitted for the instance
func (instance *Win32_PerfRawData_Counters_PacketDirectTransmitCounters) GetPropertyBytesTransmitted() (value uint64, err error) {
	retValue, err := instance.GetProperty("BytesTransmitted")
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

// SetBytesTransmittedPersec sets the value of BytesTransmittedPersec for the instance
func (instance *Win32_PerfRawData_Counters_PacketDirectTransmitCounters) SetPropertyBytesTransmittedPersec(value uint64) (err error) {
	return instance.SetProperty("BytesTransmittedPersec", (value))
}

// GetBytesTransmittedPersec gets the value of BytesTransmittedPersec for the instance
func (instance *Win32_PerfRawData_Counters_PacketDirectTransmitCounters) GetPropertyBytesTransmittedPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("BytesTransmittedPersec")
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

// SetPacketsTransmitted sets the value of PacketsTransmitted for the instance
func (instance *Win32_PerfRawData_Counters_PacketDirectTransmitCounters) SetPropertyPacketsTransmitted(value uint64) (err error) {
	return instance.SetProperty("PacketsTransmitted", (value))
}

// GetPacketsTransmitted gets the value of PacketsTransmitted for the instance
func (instance *Win32_PerfRawData_Counters_PacketDirectTransmitCounters) GetPropertyPacketsTransmitted() (value uint64, err error) {
	retValue, err := instance.GetProperty("PacketsTransmitted")
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

// SetPacketsTransmittedPersec sets the value of PacketsTransmittedPersec for the instance
func (instance *Win32_PerfRawData_Counters_PacketDirectTransmitCounters) SetPropertyPacketsTransmittedPersec(value uint64) (err error) {
	return instance.SetProperty("PacketsTransmittedPersec", (value))
}

// GetPacketsTransmittedPersec gets the value of PacketsTransmittedPersec for the instance
func (instance *Win32_PerfRawData_Counters_PacketDirectTransmitCounters) GetPropertyPacketsTransmittedPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("PacketsTransmittedPersec")
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
