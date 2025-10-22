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

// Win32_PerfFormattedData_usbhub_USB struct
type Win32_PerfFormattedData_usbhub_USB struct {
	*Win32_PerfFormattedData

	//
	AvgBytesPerTransfer uint64

	//
	AvgmslatencyforISOtransfers uint64

	//
	BulkBytesPerSec uint32

	//
	ControlDataBytesPerSec uint32

	//
	ControllerPCIInterruptsPerSec uint32

	//
	ControllerWorkSignalsPerSec uint32

	//
	HostControllerAsyncCacheFlushCount uint32

	//
	HostControllerAsyncIdle uint32

	//
	HostControllerIdle uint32

	//
	HostControllerPeriodicCacheFlushCount uint32

	//
	HostControllerPeriodicIdle uint32

	//
	InterruptBytesPerSec uint32

	//
	IsochronousBytesPerSec uint32

	//
	IsoPacketErrorsPerSec uint32

	//
	PercentTotalBandwidthUsedforInterrupt uint32

	//
	PercentTotalBandwidthUsedforIso uint32

	//
	TransferErrorsPerSec uint32
}

func NewWin32_PerfFormattedData_usbhub_USBEx1(instance *cim.WmiInstance) (newInstance *Win32_PerfFormattedData_usbhub_USB, err error) {
	tmp, err := NewWin32_PerfFormattedDataEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfFormattedData_usbhub_USB{
		Win32_PerfFormattedData: tmp,
	}
	return
}

func NewWin32_PerfFormattedData_usbhub_USBEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_PerfFormattedData_usbhub_USB, err error) {
	tmp, err := NewWin32_PerfFormattedDataEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfFormattedData_usbhub_USB{
		Win32_PerfFormattedData: tmp,
	}
	return
}

// SetAvgBytesPerTransfer sets the value of AvgBytesPerTransfer for the instance
func (instance *Win32_PerfFormattedData_usbhub_USB) SetPropertyAvgBytesPerTransfer(value uint64) (err error) {
	return instance.SetProperty("AvgBytesPerTransfer", (value))
}

// GetAvgBytesPerTransfer gets the value of AvgBytesPerTransfer for the instance
func (instance *Win32_PerfFormattedData_usbhub_USB) GetPropertyAvgBytesPerTransfer() (value uint64, err error) {
	retValue, err := instance.GetProperty("AvgBytesPerTransfer")
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

// SetAvgmslatencyforISOtransfers sets the value of AvgmslatencyforISOtransfers for the instance
func (instance *Win32_PerfFormattedData_usbhub_USB) SetPropertyAvgmslatencyforISOtransfers(value uint64) (err error) {
	return instance.SetProperty("AvgmslatencyforISOtransfers", (value))
}

// GetAvgmslatencyforISOtransfers gets the value of AvgmslatencyforISOtransfers for the instance
func (instance *Win32_PerfFormattedData_usbhub_USB) GetPropertyAvgmslatencyforISOtransfers() (value uint64, err error) {
	retValue, err := instance.GetProperty("AvgmslatencyforISOtransfers")
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

// SetBulkBytesPerSec sets the value of BulkBytesPerSec for the instance
func (instance *Win32_PerfFormattedData_usbhub_USB) SetPropertyBulkBytesPerSec(value uint32) (err error) {
	return instance.SetProperty("BulkBytesPerSec", (value))
}

// GetBulkBytesPerSec gets the value of BulkBytesPerSec for the instance
func (instance *Win32_PerfFormattedData_usbhub_USB) GetPropertyBulkBytesPerSec() (value uint32, err error) {
	retValue, err := instance.GetProperty("BulkBytesPerSec")
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

// SetControlDataBytesPerSec sets the value of ControlDataBytesPerSec for the instance
func (instance *Win32_PerfFormattedData_usbhub_USB) SetPropertyControlDataBytesPerSec(value uint32) (err error) {
	return instance.SetProperty("ControlDataBytesPerSec", (value))
}

// GetControlDataBytesPerSec gets the value of ControlDataBytesPerSec for the instance
func (instance *Win32_PerfFormattedData_usbhub_USB) GetPropertyControlDataBytesPerSec() (value uint32, err error) {
	retValue, err := instance.GetProperty("ControlDataBytesPerSec")
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

// SetControllerPCIInterruptsPerSec sets the value of ControllerPCIInterruptsPerSec for the instance
func (instance *Win32_PerfFormattedData_usbhub_USB) SetPropertyControllerPCIInterruptsPerSec(value uint32) (err error) {
	return instance.SetProperty("ControllerPCIInterruptsPerSec", (value))
}

// GetControllerPCIInterruptsPerSec gets the value of ControllerPCIInterruptsPerSec for the instance
func (instance *Win32_PerfFormattedData_usbhub_USB) GetPropertyControllerPCIInterruptsPerSec() (value uint32, err error) {
	retValue, err := instance.GetProperty("ControllerPCIInterruptsPerSec")
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

// SetControllerWorkSignalsPerSec sets the value of ControllerWorkSignalsPerSec for the instance
func (instance *Win32_PerfFormattedData_usbhub_USB) SetPropertyControllerWorkSignalsPerSec(value uint32) (err error) {
	return instance.SetProperty("ControllerWorkSignalsPerSec", (value))
}

// GetControllerWorkSignalsPerSec gets the value of ControllerWorkSignalsPerSec for the instance
func (instance *Win32_PerfFormattedData_usbhub_USB) GetPropertyControllerWorkSignalsPerSec() (value uint32, err error) {
	retValue, err := instance.GetProperty("ControllerWorkSignalsPerSec")
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

// SetHostControllerAsyncCacheFlushCount sets the value of HostControllerAsyncCacheFlushCount for the instance
func (instance *Win32_PerfFormattedData_usbhub_USB) SetPropertyHostControllerAsyncCacheFlushCount(value uint32) (err error) {
	return instance.SetProperty("HostControllerAsyncCacheFlushCount", (value))
}

// GetHostControllerAsyncCacheFlushCount gets the value of HostControllerAsyncCacheFlushCount for the instance
func (instance *Win32_PerfFormattedData_usbhub_USB) GetPropertyHostControllerAsyncCacheFlushCount() (value uint32, err error) {
	retValue, err := instance.GetProperty("HostControllerAsyncCacheFlushCount")
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

// SetHostControllerAsyncIdle sets the value of HostControllerAsyncIdle for the instance
func (instance *Win32_PerfFormattedData_usbhub_USB) SetPropertyHostControllerAsyncIdle(value uint32) (err error) {
	return instance.SetProperty("HostControllerAsyncIdle", (value))
}

// GetHostControllerAsyncIdle gets the value of HostControllerAsyncIdle for the instance
func (instance *Win32_PerfFormattedData_usbhub_USB) GetPropertyHostControllerAsyncIdle() (value uint32, err error) {
	retValue, err := instance.GetProperty("HostControllerAsyncIdle")
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

// SetHostControllerIdle sets the value of HostControllerIdle for the instance
func (instance *Win32_PerfFormattedData_usbhub_USB) SetPropertyHostControllerIdle(value uint32) (err error) {
	return instance.SetProperty("HostControllerIdle", (value))
}

// GetHostControllerIdle gets the value of HostControllerIdle for the instance
func (instance *Win32_PerfFormattedData_usbhub_USB) GetPropertyHostControllerIdle() (value uint32, err error) {
	retValue, err := instance.GetProperty("HostControllerIdle")
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

// SetHostControllerPeriodicCacheFlushCount sets the value of HostControllerPeriodicCacheFlushCount for the instance
func (instance *Win32_PerfFormattedData_usbhub_USB) SetPropertyHostControllerPeriodicCacheFlushCount(value uint32) (err error) {
	return instance.SetProperty("HostControllerPeriodicCacheFlushCount", (value))
}

// GetHostControllerPeriodicCacheFlushCount gets the value of HostControllerPeriodicCacheFlushCount for the instance
func (instance *Win32_PerfFormattedData_usbhub_USB) GetPropertyHostControllerPeriodicCacheFlushCount() (value uint32, err error) {
	retValue, err := instance.GetProperty("HostControllerPeriodicCacheFlushCount")
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

// SetHostControllerPeriodicIdle sets the value of HostControllerPeriodicIdle for the instance
func (instance *Win32_PerfFormattedData_usbhub_USB) SetPropertyHostControllerPeriodicIdle(value uint32) (err error) {
	return instance.SetProperty("HostControllerPeriodicIdle", (value))
}

// GetHostControllerPeriodicIdle gets the value of HostControllerPeriodicIdle for the instance
func (instance *Win32_PerfFormattedData_usbhub_USB) GetPropertyHostControllerPeriodicIdle() (value uint32, err error) {
	retValue, err := instance.GetProperty("HostControllerPeriodicIdle")
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

// SetInterruptBytesPerSec sets the value of InterruptBytesPerSec for the instance
func (instance *Win32_PerfFormattedData_usbhub_USB) SetPropertyInterruptBytesPerSec(value uint32) (err error) {
	return instance.SetProperty("InterruptBytesPerSec", (value))
}

// GetInterruptBytesPerSec gets the value of InterruptBytesPerSec for the instance
func (instance *Win32_PerfFormattedData_usbhub_USB) GetPropertyInterruptBytesPerSec() (value uint32, err error) {
	retValue, err := instance.GetProperty("InterruptBytesPerSec")
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

// SetIsochronousBytesPerSec sets the value of IsochronousBytesPerSec for the instance
func (instance *Win32_PerfFormattedData_usbhub_USB) SetPropertyIsochronousBytesPerSec(value uint32) (err error) {
	return instance.SetProperty("IsochronousBytesPerSec", (value))
}

// GetIsochronousBytesPerSec gets the value of IsochronousBytesPerSec for the instance
func (instance *Win32_PerfFormattedData_usbhub_USB) GetPropertyIsochronousBytesPerSec() (value uint32, err error) {
	retValue, err := instance.GetProperty("IsochronousBytesPerSec")
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

// SetIsoPacketErrorsPerSec sets the value of IsoPacketErrorsPerSec for the instance
func (instance *Win32_PerfFormattedData_usbhub_USB) SetPropertyIsoPacketErrorsPerSec(value uint32) (err error) {
	return instance.SetProperty("IsoPacketErrorsPerSec", (value))
}

// GetIsoPacketErrorsPerSec gets the value of IsoPacketErrorsPerSec for the instance
func (instance *Win32_PerfFormattedData_usbhub_USB) GetPropertyIsoPacketErrorsPerSec() (value uint32, err error) {
	retValue, err := instance.GetProperty("IsoPacketErrorsPerSec")
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

// SetPercentTotalBandwidthUsedforInterrupt sets the value of PercentTotalBandwidthUsedforInterrupt for the instance
func (instance *Win32_PerfFormattedData_usbhub_USB) SetPropertyPercentTotalBandwidthUsedforInterrupt(value uint32) (err error) {
	return instance.SetProperty("PercentTotalBandwidthUsedforInterrupt", (value))
}

// GetPercentTotalBandwidthUsedforInterrupt gets the value of PercentTotalBandwidthUsedforInterrupt for the instance
func (instance *Win32_PerfFormattedData_usbhub_USB) GetPropertyPercentTotalBandwidthUsedforInterrupt() (value uint32, err error) {
	retValue, err := instance.GetProperty("PercentTotalBandwidthUsedforInterrupt")
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

// SetPercentTotalBandwidthUsedforIso sets the value of PercentTotalBandwidthUsedforIso for the instance
func (instance *Win32_PerfFormattedData_usbhub_USB) SetPropertyPercentTotalBandwidthUsedforIso(value uint32) (err error) {
	return instance.SetProperty("PercentTotalBandwidthUsedforIso", (value))
}

// GetPercentTotalBandwidthUsedforIso gets the value of PercentTotalBandwidthUsedforIso for the instance
func (instance *Win32_PerfFormattedData_usbhub_USB) GetPropertyPercentTotalBandwidthUsedforIso() (value uint32, err error) {
	retValue, err := instance.GetProperty("PercentTotalBandwidthUsedforIso")
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

// SetTransferErrorsPerSec sets the value of TransferErrorsPerSec for the instance
func (instance *Win32_PerfFormattedData_usbhub_USB) SetPropertyTransferErrorsPerSec(value uint32) (err error) {
	return instance.SetProperty("TransferErrorsPerSec", (value))
}

// GetTransferErrorsPerSec gets the value of TransferErrorsPerSec for the instance
func (instance *Win32_PerfFormattedData_usbhub_USB) GetPropertyTransferErrorsPerSec() (value uint32, err error) {
	retValue, err := instance.GetProperty("TransferErrorsPerSec")
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
