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

// Win32_PerfFormattedData_Counters_NetworkQoSPolicy struct
type Win32_PerfFormattedData_Counters_NetworkQoSPolicy struct {
	*Win32_PerfFormattedData

	//
	Bytestransmitted uint64

	//
	BytestransmittedPersec uint64

	//
	Packetsdropped uint32

	//
	PacketsdroppedPersec uint32

	//
	Packetstransmitted uint32

	//
	PacketstransmittedPersec uint32
}

func NewWin32_PerfFormattedData_Counters_NetworkQoSPolicyEx1(instance *cim.WmiInstance) (newInstance *Win32_PerfFormattedData_Counters_NetworkQoSPolicy, err error) {
	tmp, err := NewWin32_PerfFormattedDataEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfFormattedData_Counters_NetworkQoSPolicy{
		Win32_PerfFormattedData: tmp,
	}
	return
}

func NewWin32_PerfFormattedData_Counters_NetworkQoSPolicyEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_PerfFormattedData_Counters_NetworkQoSPolicy, err error) {
	tmp, err := NewWin32_PerfFormattedDataEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfFormattedData_Counters_NetworkQoSPolicy{
		Win32_PerfFormattedData: tmp,
	}
	return
}

// SetBytestransmitted sets the value of Bytestransmitted for the instance
func (instance *Win32_PerfFormattedData_Counters_NetworkQoSPolicy) SetPropertyBytestransmitted(value uint64) (err error) {
	return instance.SetProperty("Bytestransmitted", (value))
}

// GetBytestransmitted gets the value of Bytestransmitted for the instance
func (instance *Win32_PerfFormattedData_Counters_NetworkQoSPolicy) GetPropertyBytestransmitted() (value uint64, err error) {
	retValue, err := instance.GetProperty("Bytestransmitted")
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

// SetBytestransmittedPersec sets the value of BytestransmittedPersec for the instance
func (instance *Win32_PerfFormattedData_Counters_NetworkQoSPolicy) SetPropertyBytestransmittedPersec(value uint64) (err error) {
	return instance.SetProperty("BytestransmittedPersec", (value))
}

// GetBytestransmittedPersec gets the value of BytestransmittedPersec for the instance
func (instance *Win32_PerfFormattedData_Counters_NetworkQoSPolicy) GetPropertyBytestransmittedPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("BytestransmittedPersec")
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

// SetPacketsdropped sets the value of Packetsdropped for the instance
func (instance *Win32_PerfFormattedData_Counters_NetworkQoSPolicy) SetPropertyPacketsdropped(value uint32) (err error) {
	return instance.SetProperty("Packetsdropped", (value))
}

// GetPacketsdropped gets the value of Packetsdropped for the instance
func (instance *Win32_PerfFormattedData_Counters_NetworkQoSPolicy) GetPropertyPacketsdropped() (value uint32, err error) {
	retValue, err := instance.GetProperty("Packetsdropped")
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

// SetPacketsdroppedPersec sets the value of PacketsdroppedPersec for the instance
func (instance *Win32_PerfFormattedData_Counters_NetworkQoSPolicy) SetPropertyPacketsdroppedPersec(value uint32) (err error) {
	return instance.SetProperty("PacketsdroppedPersec", (value))
}

// GetPacketsdroppedPersec gets the value of PacketsdroppedPersec for the instance
func (instance *Win32_PerfFormattedData_Counters_NetworkQoSPolicy) GetPropertyPacketsdroppedPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("PacketsdroppedPersec")
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

// SetPacketstransmitted sets the value of Packetstransmitted for the instance
func (instance *Win32_PerfFormattedData_Counters_NetworkQoSPolicy) SetPropertyPacketstransmitted(value uint32) (err error) {
	return instance.SetProperty("Packetstransmitted", (value))
}

// GetPacketstransmitted gets the value of Packetstransmitted for the instance
func (instance *Win32_PerfFormattedData_Counters_NetworkQoSPolicy) GetPropertyPacketstransmitted() (value uint32, err error) {
	retValue, err := instance.GetProperty("Packetstransmitted")
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

// SetPacketstransmittedPersec sets the value of PacketstransmittedPersec for the instance
func (instance *Win32_PerfFormattedData_Counters_NetworkQoSPolicy) SetPropertyPacketstransmittedPersec(value uint32) (err error) {
	return instance.SetProperty("PacketstransmittedPersec", (value))
}

// GetPacketstransmittedPersec gets the value of PacketstransmittedPersec for the instance
func (instance *Win32_PerfFormattedData_Counters_NetworkQoSPolicy) GetPropertyPacketstransmittedPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("PacketstransmittedPersec")
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
