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

// Win32_PerfRawData_NvspNicVRSSStats_HyperVVirtualNetworkAdapterVRSS struct
type Win32_PerfRawData_NvspNicVRSSStats_HyperVVirtualNetworkAdapterVRSS struct {
	*Win32_PerfRawData

	//
	PostmoveReceivePacketsPerSecond uint64

	//
	PostmoveReceivePacketsTotal uint64

	//
	PostmoveSendPacketCompletionsPerSecond uint64

	//
	PostmoveSendPacketCompletionsTotal uint64

	//
	PostmoveSendPacketsPerSecond uint64

	//
	PostmoveSendPacketsTotal uint64

	//
	ReceivePacketPerSecond uint64

	//
	ReceivePacketTotal uint64

	//
	ReceiveProcessor uint32

	//
	ReceiveProcessorGroup uint32

	//
	SendPacketCompletionsPerSecond uint64

	//
	SendPacketCompletionsTotal uint64

	//
	SendPacketPerSecond uint64

	//
	SendPacketTotal uint64

	//
	SendProcessor uint32

	//
	SendProcessorGroup uint32
}

func NewWin32_PerfRawData_NvspNicVRSSStats_HyperVVirtualNetworkAdapterVRSSEx1(instance *cim.WmiInstance) (newInstance *Win32_PerfRawData_NvspNicVRSSStats_HyperVVirtualNetworkAdapterVRSS, err error) {
	tmp, err := NewWin32_PerfRawDataEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfRawData_NvspNicVRSSStats_HyperVVirtualNetworkAdapterVRSS{
		Win32_PerfRawData: tmp,
	}
	return
}

func NewWin32_PerfRawData_NvspNicVRSSStats_HyperVVirtualNetworkAdapterVRSSEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_PerfRawData_NvspNicVRSSStats_HyperVVirtualNetworkAdapterVRSS, err error) {
	tmp, err := NewWin32_PerfRawDataEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfRawData_NvspNicVRSSStats_HyperVVirtualNetworkAdapterVRSS{
		Win32_PerfRawData: tmp,
	}
	return
}

// SetPostmoveReceivePacketsPerSecond sets the value of PostmoveReceivePacketsPerSecond for the instance
func (instance *Win32_PerfRawData_NvspNicVRSSStats_HyperVVirtualNetworkAdapterVRSS) SetPropertyPostmoveReceivePacketsPerSecond(value uint64) (err error) {
	return instance.SetProperty("PostmoveReceivePacketsPerSecond", (value))
}

// GetPostmoveReceivePacketsPerSecond gets the value of PostmoveReceivePacketsPerSecond for the instance
func (instance *Win32_PerfRawData_NvspNicVRSSStats_HyperVVirtualNetworkAdapterVRSS) GetPropertyPostmoveReceivePacketsPerSecond() (value uint64, err error) {
	retValue, err := instance.GetProperty("PostmoveReceivePacketsPerSecond")
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

// SetPostmoveReceivePacketsTotal sets the value of PostmoveReceivePacketsTotal for the instance
func (instance *Win32_PerfRawData_NvspNicVRSSStats_HyperVVirtualNetworkAdapterVRSS) SetPropertyPostmoveReceivePacketsTotal(value uint64) (err error) {
	return instance.SetProperty("PostmoveReceivePacketsTotal", (value))
}

// GetPostmoveReceivePacketsTotal gets the value of PostmoveReceivePacketsTotal for the instance
func (instance *Win32_PerfRawData_NvspNicVRSSStats_HyperVVirtualNetworkAdapterVRSS) GetPropertyPostmoveReceivePacketsTotal() (value uint64, err error) {
	retValue, err := instance.GetProperty("PostmoveReceivePacketsTotal")
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

// SetPostmoveSendPacketCompletionsPerSecond sets the value of PostmoveSendPacketCompletionsPerSecond for the instance
func (instance *Win32_PerfRawData_NvspNicVRSSStats_HyperVVirtualNetworkAdapterVRSS) SetPropertyPostmoveSendPacketCompletionsPerSecond(value uint64) (err error) {
	return instance.SetProperty("PostmoveSendPacketCompletionsPerSecond", (value))
}

// GetPostmoveSendPacketCompletionsPerSecond gets the value of PostmoveSendPacketCompletionsPerSecond for the instance
func (instance *Win32_PerfRawData_NvspNicVRSSStats_HyperVVirtualNetworkAdapterVRSS) GetPropertyPostmoveSendPacketCompletionsPerSecond() (value uint64, err error) {
	retValue, err := instance.GetProperty("PostmoveSendPacketCompletionsPerSecond")
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

// SetPostmoveSendPacketCompletionsTotal sets the value of PostmoveSendPacketCompletionsTotal for the instance
func (instance *Win32_PerfRawData_NvspNicVRSSStats_HyperVVirtualNetworkAdapterVRSS) SetPropertyPostmoveSendPacketCompletionsTotal(value uint64) (err error) {
	return instance.SetProperty("PostmoveSendPacketCompletionsTotal", (value))
}

// GetPostmoveSendPacketCompletionsTotal gets the value of PostmoveSendPacketCompletionsTotal for the instance
func (instance *Win32_PerfRawData_NvspNicVRSSStats_HyperVVirtualNetworkAdapterVRSS) GetPropertyPostmoveSendPacketCompletionsTotal() (value uint64, err error) {
	retValue, err := instance.GetProperty("PostmoveSendPacketCompletionsTotal")
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

// SetPostmoveSendPacketsPerSecond sets the value of PostmoveSendPacketsPerSecond for the instance
func (instance *Win32_PerfRawData_NvspNicVRSSStats_HyperVVirtualNetworkAdapterVRSS) SetPropertyPostmoveSendPacketsPerSecond(value uint64) (err error) {
	return instance.SetProperty("PostmoveSendPacketsPerSecond", (value))
}

// GetPostmoveSendPacketsPerSecond gets the value of PostmoveSendPacketsPerSecond for the instance
func (instance *Win32_PerfRawData_NvspNicVRSSStats_HyperVVirtualNetworkAdapterVRSS) GetPropertyPostmoveSendPacketsPerSecond() (value uint64, err error) {
	retValue, err := instance.GetProperty("PostmoveSendPacketsPerSecond")
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

// SetPostmoveSendPacketsTotal sets the value of PostmoveSendPacketsTotal for the instance
func (instance *Win32_PerfRawData_NvspNicVRSSStats_HyperVVirtualNetworkAdapterVRSS) SetPropertyPostmoveSendPacketsTotal(value uint64) (err error) {
	return instance.SetProperty("PostmoveSendPacketsTotal", (value))
}

// GetPostmoveSendPacketsTotal gets the value of PostmoveSendPacketsTotal for the instance
func (instance *Win32_PerfRawData_NvspNicVRSSStats_HyperVVirtualNetworkAdapterVRSS) GetPropertyPostmoveSendPacketsTotal() (value uint64, err error) {
	retValue, err := instance.GetProperty("PostmoveSendPacketsTotal")
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

// SetReceivePacketPerSecond sets the value of ReceivePacketPerSecond for the instance
func (instance *Win32_PerfRawData_NvspNicVRSSStats_HyperVVirtualNetworkAdapterVRSS) SetPropertyReceivePacketPerSecond(value uint64) (err error) {
	return instance.SetProperty("ReceivePacketPerSecond", (value))
}

// GetReceivePacketPerSecond gets the value of ReceivePacketPerSecond for the instance
func (instance *Win32_PerfRawData_NvspNicVRSSStats_HyperVVirtualNetworkAdapterVRSS) GetPropertyReceivePacketPerSecond() (value uint64, err error) {
	retValue, err := instance.GetProperty("ReceivePacketPerSecond")
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

// SetReceivePacketTotal sets the value of ReceivePacketTotal for the instance
func (instance *Win32_PerfRawData_NvspNicVRSSStats_HyperVVirtualNetworkAdapterVRSS) SetPropertyReceivePacketTotal(value uint64) (err error) {
	return instance.SetProperty("ReceivePacketTotal", (value))
}

// GetReceivePacketTotal gets the value of ReceivePacketTotal for the instance
func (instance *Win32_PerfRawData_NvspNicVRSSStats_HyperVVirtualNetworkAdapterVRSS) GetPropertyReceivePacketTotal() (value uint64, err error) {
	retValue, err := instance.GetProperty("ReceivePacketTotal")
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

// SetReceiveProcessor sets the value of ReceiveProcessor for the instance
func (instance *Win32_PerfRawData_NvspNicVRSSStats_HyperVVirtualNetworkAdapterVRSS) SetPropertyReceiveProcessor(value uint32) (err error) {
	return instance.SetProperty("ReceiveProcessor", (value))
}

// GetReceiveProcessor gets the value of ReceiveProcessor for the instance
func (instance *Win32_PerfRawData_NvspNicVRSSStats_HyperVVirtualNetworkAdapterVRSS) GetPropertyReceiveProcessor() (value uint32, err error) {
	retValue, err := instance.GetProperty("ReceiveProcessor")
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

// SetReceiveProcessorGroup sets the value of ReceiveProcessorGroup for the instance
func (instance *Win32_PerfRawData_NvspNicVRSSStats_HyperVVirtualNetworkAdapterVRSS) SetPropertyReceiveProcessorGroup(value uint32) (err error) {
	return instance.SetProperty("ReceiveProcessorGroup", (value))
}

// GetReceiveProcessorGroup gets the value of ReceiveProcessorGroup for the instance
func (instance *Win32_PerfRawData_NvspNicVRSSStats_HyperVVirtualNetworkAdapterVRSS) GetPropertyReceiveProcessorGroup() (value uint32, err error) {
	retValue, err := instance.GetProperty("ReceiveProcessorGroup")
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

// SetSendPacketCompletionsPerSecond sets the value of SendPacketCompletionsPerSecond for the instance
func (instance *Win32_PerfRawData_NvspNicVRSSStats_HyperVVirtualNetworkAdapterVRSS) SetPropertySendPacketCompletionsPerSecond(value uint64) (err error) {
	return instance.SetProperty("SendPacketCompletionsPerSecond", (value))
}

// GetSendPacketCompletionsPerSecond gets the value of SendPacketCompletionsPerSecond for the instance
func (instance *Win32_PerfRawData_NvspNicVRSSStats_HyperVVirtualNetworkAdapterVRSS) GetPropertySendPacketCompletionsPerSecond() (value uint64, err error) {
	retValue, err := instance.GetProperty("SendPacketCompletionsPerSecond")
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

// SetSendPacketCompletionsTotal sets the value of SendPacketCompletionsTotal for the instance
func (instance *Win32_PerfRawData_NvspNicVRSSStats_HyperVVirtualNetworkAdapterVRSS) SetPropertySendPacketCompletionsTotal(value uint64) (err error) {
	return instance.SetProperty("SendPacketCompletionsTotal", (value))
}

// GetSendPacketCompletionsTotal gets the value of SendPacketCompletionsTotal for the instance
func (instance *Win32_PerfRawData_NvspNicVRSSStats_HyperVVirtualNetworkAdapterVRSS) GetPropertySendPacketCompletionsTotal() (value uint64, err error) {
	retValue, err := instance.GetProperty("SendPacketCompletionsTotal")
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

// SetSendPacketPerSecond sets the value of SendPacketPerSecond for the instance
func (instance *Win32_PerfRawData_NvspNicVRSSStats_HyperVVirtualNetworkAdapterVRSS) SetPropertySendPacketPerSecond(value uint64) (err error) {
	return instance.SetProperty("SendPacketPerSecond", (value))
}

// GetSendPacketPerSecond gets the value of SendPacketPerSecond for the instance
func (instance *Win32_PerfRawData_NvspNicVRSSStats_HyperVVirtualNetworkAdapterVRSS) GetPropertySendPacketPerSecond() (value uint64, err error) {
	retValue, err := instance.GetProperty("SendPacketPerSecond")
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

// SetSendPacketTotal sets the value of SendPacketTotal for the instance
func (instance *Win32_PerfRawData_NvspNicVRSSStats_HyperVVirtualNetworkAdapterVRSS) SetPropertySendPacketTotal(value uint64) (err error) {
	return instance.SetProperty("SendPacketTotal", (value))
}

// GetSendPacketTotal gets the value of SendPacketTotal for the instance
func (instance *Win32_PerfRawData_NvspNicVRSSStats_HyperVVirtualNetworkAdapterVRSS) GetPropertySendPacketTotal() (value uint64, err error) {
	retValue, err := instance.GetProperty("SendPacketTotal")
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

// SetSendProcessor sets the value of SendProcessor for the instance
func (instance *Win32_PerfRawData_NvspNicVRSSStats_HyperVVirtualNetworkAdapterVRSS) SetPropertySendProcessor(value uint32) (err error) {
	return instance.SetProperty("SendProcessor", (value))
}

// GetSendProcessor gets the value of SendProcessor for the instance
func (instance *Win32_PerfRawData_NvspNicVRSSStats_HyperVVirtualNetworkAdapterVRSS) GetPropertySendProcessor() (value uint32, err error) {
	retValue, err := instance.GetProperty("SendProcessor")
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

// SetSendProcessorGroup sets the value of SendProcessorGroup for the instance
func (instance *Win32_PerfRawData_NvspNicVRSSStats_HyperVVirtualNetworkAdapterVRSS) SetPropertySendProcessorGroup(value uint32) (err error) {
	return instance.SetProperty("SendProcessorGroup", (value))
}

// GetSendProcessorGroup gets the value of SendProcessorGroup for the instance
func (instance *Win32_PerfRawData_NvspNicVRSSStats_HyperVVirtualNetworkAdapterVRSS) GetPropertySendProcessorGroup() (value uint32, err error) {
	retValue, err := instance.GetProperty("SendProcessorGroup")
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
