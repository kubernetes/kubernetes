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

// Win32_PerfFormattedData_ClussvcPerfProvider_ClusterGoodEnoughMulticastMessages struct
type Win32_PerfFormattedData_ClussvcPerfProvider_ClusterGoodEnoughMulticastMessages struct {
	*Win32_PerfFormattedData

	//
	MessageQueueLength uint64

	//
	UnacknowledgedMessages uint64
}

func NewWin32_PerfFormattedData_ClussvcPerfProvider_ClusterGoodEnoughMulticastMessagesEx1(instance *cim.WmiInstance) (newInstance *Win32_PerfFormattedData_ClussvcPerfProvider_ClusterGoodEnoughMulticastMessages, err error) {
	tmp, err := NewWin32_PerfFormattedDataEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfFormattedData_ClussvcPerfProvider_ClusterGoodEnoughMulticastMessages{
		Win32_PerfFormattedData: tmp,
	}
	return
}

func NewWin32_PerfFormattedData_ClussvcPerfProvider_ClusterGoodEnoughMulticastMessagesEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_PerfFormattedData_ClussvcPerfProvider_ClusterGoodEnoughMulticastMessages, err error) {
	tmp, err := NewWin32_PerfFormattedDataEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfFormattedData_ClussvcPerfProvider_ClusterGoodEnoughMulticastMessages{
		Win32_PerfFormattedData: tmp,
	}
	return
}

// SetMessageQueueLength sets the value of MessageQueueLength for the instance
func (instance *Win32_PerfFormattedData_ClussvcPerfProvider_ClusterGoodEnoughMulticastMessages) SetPropertyMessageQueueLength(value uint64) (err error) {
	return instance.SetProperty("MessageQueueLength", (value))
}

// GetMessageQueueLength gets the value of MessageQueueLength for the instance
func (instance *Win32_PerfFormattedData_ClussvcPerfProvider_ClusterGoodEnoughMulticastMessages) GetPropertyMessageQueueLength() (value uint64, err error) {
	retValue, err := instance.GetProperty("MessageQueueLength")
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

// SetUnacknowledgedMessages sets the value of UnacknowledgedMessages for the instance
func (instance *Win32_PerfFormattedData_ClussvcPerfProvider_ClusterGoodEnoughMulticastMessages) SetPropertyUnacknowledgedMessages(value uint64) (err error) {
	return instance.SetProperty("UnacknowledgedMessages", (value))
}

// GetUnacknowledgedMessages gets the value of UnacknowledgedMessages for the instance
func (instance *Win32_PerfFormattedData_ClussvcPerfProvider_ClusterGoodEnoughMulticastMessages) GetPropertyUnacknowledgedMessages() (value uint64, err error) {
	retValue, err := instance.GetProperty("UnacknowledgedMessages")
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
