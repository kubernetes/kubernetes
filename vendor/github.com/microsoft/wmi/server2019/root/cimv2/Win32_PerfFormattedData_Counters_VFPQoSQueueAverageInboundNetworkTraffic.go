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

// Win32_PerfFormattedData_Counters_VFPQoSQueueAverageInboundNetworkTraffic struct
type Win32_PerfFormattedData_Counters_VFPQoSQueueAverageInboundNetworkTraffic struct {
	*Win32_PerfFormattedData

	//
	AverageInboundBytesAllowedThroughtheQueue uint64

	//
	AverageInboundBytesDropped uint64

	//
	AverageInboundBytesEnteringtheQueue uint64

	//
	AverageInboundBytesQueuedduetoBacklog uint64

	//
	AverageInboundBytesQueuedduetoInsufficientTokens uint64

	//
	AverageInboundBytesResumed uint64

	//
	AverageInboundPacketsAllowedThroughtheQueue uint64

	//
	AverageInboundPacketsDropped uint64

	//
	AverageInboundPacketsEnteringtheQueue uint64

	//
	AverageInboundPacketsQueuedduetoBacklog uint64

	//
	AverageInboundPacketsQueuedduetoInsufficientTokens uint64

	//
	AverageInboundPacketsResumed uint64
}

func NewWin32_PerfFormattedData_Counters_VFPQoSQueueAverageInboundNetworkTrafficEx1(instance *cim.WmiInstance) (newInstance *Win32_PerfFormattedData_Counters_VFPQoSQueueAverageInboundNetworkTraffic, err error) {
	tmp, err := NewWin32_PerfFormattedDataEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfFormattedData_Counters_VFPQoSQueueAverageInboundNetworkTraffic{
		Win32_PerfFormattedData: tmp,
	}
	return
}

func NewWin32_PerfFormattedData_Counters_VFPQoSQueueAverageInboundNetworkTrafficEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_PerfFormattedData_Counters_VFPQoSQueueAverageInboundNetworkTraffic, err error) {
	tmp, err := NewWin32_PerfFormattedDataEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfFormattedData_Counters_VFPQoSQueueAverageInboundNetworkTraffic{
		Win32_PerfFormattedData: tmp,
	}
	return
}

// SetAverageInboundBytesAllowedThroughtheQueue sets the value of AverageInboundBytesAllowedThroughtheQueue for the instance
func (instance *Win32_PerfFormattedData_Counters_VFPQoSQueueAverageInboundNetworkTraffic) SetPropertyAverageInboundBytesAllowedThroughtheQueue(value uint64) (err error) {
	return instance.SetProperty("AverageInboundBytesAllowedThroughtheQueue", (value))
}

// GetAverageInboundBytesAllowedThroughtheQueue gets the value of AverageInboundBytesAllowedThroughtheQueue for the instance
func (instance *Win32_PerfFormattedData_Counters_VFPQoSQueueAverageInboundNetworkTraffic) GetPropertyAverageInboundBytesAllowedThroughtheQueue() (value uint64, err error) {
	retValue, err := instance.GetProperty("AverageInboundBytesAllowedThroughtheQueue")
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

// SetAverageInboundBytesDropped sets the value of AverageInboundBytesDropped for the instance
func (instance *Win32_PerfFormattedData_Counters_VFPQoSQueueAverageInboundNetworkTraffic) SetPropertyAverageInboundBytesDropped(value uint64) (err error) {
	return instance.SetProperty("AverageInboundBytesDropped", (value))
}

// GetAverageInboundBytesDropped gets the value of AverageInboundBytesDropped for the instance
func (instance *Win32_PerfFormattedData_Counters_VFPQoSQueueAverageInboundNetworkTraffic) GetPropertyAverageInboundBytesDropped() (value uint64, err error) {
	retValue, err := instance.GetProperty("AverageInboundBytesDropped")
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

// SetAverageInboundBytesEnteringtheQueue sets the value of AverageInboundBytesEnteringtheQueue for the instance
func (instance *Win32_PerfFormattedData_Counters_VFPQoSQueueAverageInboundNetworkTraffic) SetPropertyAverageInboundBytesEnteringtheQueue(value uint64) (err error) {
	return instance.SetProperty("AverageInboundBytesEnteringtheQueue", (value))
}

// GetAverageInboundBytesEnteringtheQueue gets the value of AverageInboundBytesEnteringtheQueue for the instance
func (instance *Win32_PerfFormattedData_Counters_VFPQoSQueueAverageInboundNetworkTraffic) GetPropertyAverageInboundBytesEnteringtheQueue() (value uint64, err error) {
	retValue, err := instance.GetProperty("AverageInboundBytesEnteringtheQueue")
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

// SetAverageInboundBytesQueuedduetoBacklog sets the value of AverageInboundBytesQueuedduetoBacklog for the instance
func (instance *Win32_PerfFormattedData_Counters_VFPQoSQueueAverageInboundNetworkTraffic) SetPropertyAverageInboundBytesQueuedduetoBacklog(value uint64) (err error) {
	return instance.SetProperty("AverageInboundBytesQueuedduetoBacklog", (value))
}

// GetAverageInboundBytesQueuedduetoBacklog gets the value of AverageInboundBytesQueuedduetoBacklog for the instance
func (instance *Win32_PerfFormattedData_Counters_VFPQoSQueueAverageInboundNetworkTraffic) GetPropertyAverageInboundBytesQueuedduetoBacklog() (value uint64, err error) {
	retValue, err := instance.GetProperty("AverageInboundBytesQueuedduetoBacklog")
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

// SetAverageInboundBytesQueuedduetoInsufficientTokens sets the value of AverageInboundBytesQueuedduetoInsufficientTokens for the instance
func (instance *Win32_PerfFormattedData_Counters_VFPQoSQueueAverageInboundNetworkTraffic) SetPropertyAverageInboundBytesQueuedduetoInsufficientTokens(value uint64) (err error) {
	return instance.SetProperty("AverageInboundBytesQueuedduetoInsufficientTokens", (value))
}

// GetAverageInboundBytesQueuedduetoInsufficientTokens gets the value of AverageInboundBytesQueuedduetoInsufficientTokens for the instance
func (instance *Win32_PerfFormattedData_Counters_VFPQoSQueueAverageInboundNetworkTraffic) GetPropertyAverageInboundBytesQueuedduetoInsufficientTokens() (value uint64, err error) {
	retValue, err := instance.GetProperty("AverageInboundBytesQueuedduetoInsufficientTokens")
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

// SetAverageInboundBytesResumed sets the value of AverageInboundBytesResumed for the instance
func (instance *Win32_PerfFormattedData_Counters_VFPQoSQueueAverageInboundNetworkTraffic) SetPropertyAverageInboundBytesResumed(value uint64) (err error) {
	return instance.SetProperty("AverageInboundBytesResumed", (value))
}

// GetAverageInboundBytesResumed gets the value of AverageInboundBytesResumed for the instance
func (instance *Win32_PerfFormattedData_Counters_VFPQoSQueueAverageInboundNetworkTraffic) GetPropertyAverageInboundBytesResumed() (value uint64, err error) {
	retValue, err := instance.GetProperty("AverageInboundBytesResumed")
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

// SetAverageInboundPacketsAllowedThroughtheQueue sets the value of AverageInboundPacketsAllowedThroughtheQueue for the instance
func (instance *Win32_PerfFormattedData_Counters_VFPQoSQueueAverageInboundNetworkTraffic) SetPropertyAverageInboundPacketsAllowedThroughtheQueue(value uint64) (err error) {
	return instance.SetProperty("AverageInboundPacketsAllowedThroughtheQueue", (value))
}

// GetAverageInboundPacketsAllowedThroughtheQueue gets the value of AverageInboundPacketsAllowedThroughtheQueue for the instance
func (instance *Win32_PerfFormattedData_Counters_VFPQoSQueueAverageInboundNetworkTraffic) GetPropertyAverageInboundPacketsAllowedThroughtheQueue() (value uint64, err error) {
	retValue, err := instance.GetProperty("AverageInboundPacketsAllowedThroughtheQueue")
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

// SetAverageInboundPacketsDropped sets the value of AverageInboundPacketsDropped for the instance
func (instance *Win32_PerfFormattedData_Counters_VFPQoSQueueAverageInboundNetworkTraffic) SetPropertyAverageInboundPacketsDropped(value uint64) (err error) {
	return instance.SetProperty("AverageInboundPacketsDropped", (value))
}

// GetAverageInboundPacketsDropped gets the value of AverageInboundPacketsDropped for the instance
func (instance *Win32_PerfFormattedData_Counters_VFPQoSQueueAverageInboundNetworkTraffic) GetPropertyAverageInboundPacketsDropped() (value uint64, err error) {
	retValue, err := instance.GetProperty("AverageInboundPacketsDropped")
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

// SetAverageInboundPacketsEnteringtheQueue sets the value of AverageInboundPacketsEnteringtheQueue for the instance
func (instance *Win32_PerfFormattedData_Counters_VFPQoSQueueAverageInboundNetworkTraffic) SetPropertyAverageInboundPacketsEnteringtheQueue(value uint64) (err error) {
	return instance.SetProperty("AverageInboundPacketsEnteringtheQueue", (value))
}

// GetAverageInboundPacketsEnteringtheQueue gets the value of AverageInboundPacketsEnteringtheQueue for the instance
func (instance *Win32_PerfFormattedData_Counters_VFPQoSQueueAverageInboundNetworkTraffic) GetPropertyAverageInboundPacketsEnteringtheQueue() (value uint64, err error) {
	retValue, err := instance.GetProperty("AverageInboundPacketsEnteringtheQueue")
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

// SetAverageInboundPacketsQueuedduetoBacklog sets the value of AverageInboundPacketsQueuedduetoBacklog for the instance
func (instance *Win32_PerfFormattedData_Counters_VFPQoSQueueAverageInboundNetworkTraffic) SetPropertyAverageInboundPacketsQueuedduetoBacklog(value uint64) (err error) {
	return instance.SetProperty("AverageInboundPacketsQueuedduetoBacklog", (value))
}

// GetAverageInboundPacketsQueuedduetoBacklog gets the value of AverageInboundPacketsQueuedduetoBacklog for the instance
func (instance *Win32_PerfFormattedData_Counters_VFPQoSQueueAverageInboundNetworkTraffic) GetPropertyAverageInboundPacketsQueuedduetoBacklog() (value uint64, err error) {
	retValue, err := instance.GetProperty("AverageInboundPacketsQueuedduetoBacklog")
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

// SetAverageInboundPacketsQueuedduetoInsufficientTokens sets the value of AverageInboundPacketsQueuedduetoInsufficientTokens for the instance
func (instance *Win32_PerfFormattedData_Counters_VFPQoSQueueAverageInboundNetworkTraffic) SetPropertyAverageInboundPacketsQueuedduetoInsufficientTokens(value uint64) (err error) {
	return instance.SetProperty("AverageInboundPacketsQueuedduetoInsufficientTokens", (value))
}

// GetAverageInboundPacketsQueuedduetoInsufficientTokens gets the value of AverageInboundPacketsQueuedduetoInsufficientTokens for the instance
func (instance *Win32_PerfFormattedData_Counters_VFPQoSQueueAverageInboundNetworkTraffic) GetPropertyAverageInboundPacketsQueuedduetoInsufficientTokens() (value uint64, err error) {
	retValue, err := instance.GetProperty("AverageInboundPacketsQueuedduetoInsufficientTokens")
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

// SetAverageInboundPacketsResumed sets the value of AverageInboundPacketsResumed for the instance
func (instance *Win32_PerfFormattedData_Counters_VFPQoSQueueAverageInboundNetworkTraffic) SetPropertyAverageInboundPacketsResumed(value uint64) (err error) {
	return instance.SetProperty("AverageInboundPacketsResumed", (value))
}

// GetAverageInboundPacketsResumed gets the value of AverageInboundPacketsResumed for the instance
func (instance *Win32_PerfFormattedData_Counters_VFPQoSQueueAverageInboundNetworkTraffic) GetPropertyAverageInboundPacketsResumed() (value uint64, err error) {
	retValue, err := instance.GetProperty("AverageInboundPacketsResumed")
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
