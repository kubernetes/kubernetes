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

// Win32_PerfRawData_EthernetPerfProvider_HyperVLegacyNetworkAdapter struct
type Win32_PerfRawData_EthernetPerfProvider_HyperVLegacyNetworkAdapter struct {
	*Win32_PerfRawData

	//
	BytesDropped uint64

	//
	BytesReceivedPersec uint64

	//
	BytesSentPersec uint64

	//
	FramesDropped uint64

	//
	FramesReceivedPersec uint64

	//
	FramesSentPersec uint64
}

func NewWin32_PerfRawData_EthernetPerfProvider_HyperVLegacyNetworkAdapterEx1(instance *cim.WmiInstance) (newInstance *Win32_PerfRawData_EthernetPerfProvider_HyperVLegacyNetworkAdapter, err error) {
	tmp, err := NewWin32_PerfRawDataEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfRawData_EthernetPerfProvider_HyperVLegacyNetworkAdapter{
		Win32_PerfRawData: tmp,
	}
	return
}

func NewWin32_PerfRawData_EthernetPerfProvider_HyperVLegacyNetworkAdapterEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_PerfRawData_EthernetPerfProvider_HyperVLegacyNetworkAdapter, err error) {
	tmp, err := NewWin32_PerfRawDataEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfRawData_EthernetPerfProvider_HyperVLegacyNetworkAdapter{
		Win32_PerfRawData: tmp,
	}
	return
}

// SetBytesDropped sets the value of BytesDropped for the instance
func (instance *Win32_PerfRawData_EthernetPerfProvider_HyperVLegacyNetworkAdapter) SetPropertyBytesDropped(value uint64) (err error) {
	return instance.SetProperty("BytesDropped", (value))
}

// GetBytesDropped gets the value of BytesDropped for the instance
func (instance *Win32_PerfRawData_EthernetPerfProvider_HyperVLegacyNetworkAdapter) GetPropertyBytesDropped() (value uint64, err error) {
	retValue, err := instance.GetProperty("BytesDropped")
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
func (instance *Win32_PerfRawData_EthernetPerfProvider_HyperVLegacyNetworkAdapter) SetPropertyBytesReceivedPersec(value uint64) (err error) {
	return instance.SetProperty("BytesReceivedPersec", (value))
}

// GetBytesReceivedPersec gets the value of BytesReceivedPersec for the instance
func (instance *Win32_PerfRawData_EthernetPerfProvider_HyperVLegacyNetworkAdapter) GetPropertyBytesReceivedPersec() (value uint64, err error) {
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

// SetBytesSentPersec sets the value of BytesSentPersec for the instance
func (instance *Win32_PerfRawData_EthernetPerfProvider_HyperVLegacyNetworkAdapter) SetPropertyBytesSentPersec(value uint64) (err error) {
	return instance.SetProperty("BytesSentPersec", (value))
}

// GetBytesSentPersec gets the value of BytesSentPersec for the instance
func (instance *Win32_PerfRawData_EthernetPerfProvider_HyperVLegacyNetworkAdapter) GetPropertyBytesSentPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("BytesSentPersec")
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

// SetFramesDropped sets the value of FramesDropped for the instance
func (instance *Win32_PerfRawData_EthernetPerfProvider_HyperVLegacyNetworkAdapter) SetPropertyFramesDropped(value uint64) (err error) {
	return instance.SetProperty("FramesDropped", (value))
}

// GetFramesDropped gets the value of FramesDropped for the instance
func (instance *Win32_PerfRawData_EthernetPerfProvider_HyperVLegacyNetworkAdapter) GetPropertyFramesDropped() (value uint64, err error) {
	retValue, err := instance.GetProperty("FramesDropped")
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

// SetFramesReceivedPersec sets the value of FramesReceivedPersec for the instance
func (instance *Win32_PerfRawData_EthernetPerfProvider_HyperVLegacyNetworkAdapter) SetPropertyFramesReceivedPersec(value uint64) (err error) {
	return instance.SetProperty("FramesReceivedPersec", (value))
}

// GetFramesReceivedPersec gets the value of FramesReceivedPersec for the instance
func (instance *Win32_PerfRawData_EthernetPerfProvider_HyperVLegacyNetworkAdapter) GetPropertyFramesReceivedPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("FramesReceivedPersec")
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

// SetFramesSentPersec sets the value of FramesSentPersec for the instance
func (instance *Win32_PerfRawData_EthernetPerfProvider_HyperVLegacyNetworkAdapter) SetPropertyFramesSentPersec(value uint64) (err error) {
	return instance.SetProperty("FramesSentPersec", (value))
}

// GetFramesSentPersec gets the value of FramesSentPersec for the instance
func (instance *Win32_PerfRawData_EthernetPerfProvider_HyperVLegacyNetworkAdapter) GetPropertyFramesSentPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("FramesSentPersec")
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
