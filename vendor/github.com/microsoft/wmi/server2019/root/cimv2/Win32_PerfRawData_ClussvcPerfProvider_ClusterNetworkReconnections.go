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

// Win32_PerfRawData_ClussvcPerfProvider_ClusterNetworkReconnections struct
type Win32_PerfRawData_ClussvcPerfProvider_ClusterNetworkReconnections struct {
	*Win32_PerfRawData

	//
	NormalMessageQueueLength uint64

	//
	NormalMessageQueueLengthPersec uint64

	//
	ReconnectCount uint64

	//
	UnacknowledgedMessageQueueLength uint64

	//
	UnacknowledgedMessageQueueLengthPersec uint64

	//
	UrgentMessageQueueLength uint64

	//
	UrgentMessageQueueLengthPersec uint64
}

func NewWin32_PerfRawData_ClussvcPerfProvider_ClusterNetworkReconnectionsEx1(instance *cim.WmiInstance) (newInstance *Win32_PerfRawData_ClussvcPerfProvider_ClusterNetworkReconnections, err error) {
	tmp, err := NewWin32_PerfRawDataEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfRawData_ClussvcPerfProvider_ClusterNetworkReconnections{
		Win32_PerfRawData: tmp,
	}
	return
}

func NewWin32_PerfRawData_ClussvcPerfProvider_ClusterNetworkReconnectionsEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_PerfRawData_ClussvcPerfProvider_ClusterNetworkReconnections, err error) {
	tmp, err := NewWin32_PerfRawDataEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfRawData_ClussvcPerfProvider_ClusterNetworkReconnections{
		Win32_PerfRawData: tmp,
	}
	return
}

// SetNormalMessageQueueLength sets the value of NormalMessageQueueLength for the instance
func (instance *Win32_PerfRawData_ClussvcPerfProvider_ClusterNetworkReconnections) SetPropertyNormalMessageQueueLength(value uint64) (err error) {
	return instance.SetProperty("NormalMessageQueueLength", (value))
}

// GetNormalMessageQueueLength gets the value of NormalMessageQueueLength for the instance
func (instance *Win32_PerfRawData_ClussvcPerfProvider_ClusterNetworkReconnections) GetPropertyNormalMessageQueueLength() (value uint64, err error) {
	retValue, err := instance.GetProperty("NormalMessageQueueLength")
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

// SetNormalMessageQueueLengthPersec sets the value of NormalMessageQueueLengthPersec for the instance
func (instance *Win32_PerfRawData_ClussvcPerfProvider_ClusterNetworkReconnections) SetPropertyNormalMessageQueueLengthPersec(value uint64) (err error) {
	return instance.SetProperty("NormalMessageQueueLengthPersec", (value))
}

// GetNormalMessageQueueLengthPersec gets the value of NormalMessageQueueLengthPersec for the instance
func (instance *Win32_PerfRawData_ClussvcPerfProvider_ClusterNetworkReconnections) GetPropertyNormalMessageQueueLengthPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("NormalMessageQueueLengthPersec")
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

// SetReconnectCount sets the value of ReconnectCount for the instance
func (instance *Win32_PerfRawData_ClussvcPerfProvider_ClusterNetworkReconnections) SetPropertyReconnectCount(value uint64) (err error) {
	return instance.SetProperty("ReconnectCount", (value))
}

// GetReconnectCount gets the value of ReconnectCount for the instance
func (instance *Win32_PerfRawData_ClussvcPerfProvider_ClusterNetworkReconnections) GetPropertyReconnectCount() (value uint64, err error) {
	retValue, err := instance.GetProperty("ReconnectCount")
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

// SetUnacknowledgedMessageQueueLength sets the value of UnacknowledgedMessageQueueLength for the instance
func (instance *Win32_PerfRawData_ClussvcPerfProvider_ClusterNetworkReconnections) SetPropertyUnacknowledgedMessageQueueLength(value uint64) (err error) {
	return instance.SetProperty("UnacknowledgedMessageQueueLength", (value))
}

// GetUnacknowledgedMessageQueueLength gets the value of UnacknowledgedMessageQueueLength for the instance
func (instance *Win32_PerfRawData_ClussvcPerfProvider_ClusterNetworkReconnections) GetPropertyUnacknowledgedMessageQueueLength() (value uint64, err error) {
	retValue, err := instance.GetProperty("UnacknowledgedMessageQueueLength")
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

// SetUnacknowledgedMessageQueueLengthPersec sets the value of UnacknowledgedMessageQueueLengthPersec for the instance
func (instance *Win32_PerfRawData_ClussvcPerfProvider_ClusterNetworkReconnections) SetPropertyUnacknowledgedMessageQueueLengthPersec(value uint64) (err error) {
	return instance.SetProperty("UnacknowledgedMessageQueueLengthPersec", (value))
}

// GetUnacknowledgedMessageQueueLengthPersec gets the value of UnacknowledgedMessageQueueLengthPersec for the instance
func (instance *Win32_PerfRawData_ClussvcPerfProvider_ClusterNetworkReconnections) GetPropertyUnacknowledgedMessageQueueLengthPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("UnacknowledgedMessageQueueLengthPersec")
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

// SetUrgentMessageQueueLength sets the value of UrgentMessageQueueLength for the instance
func (instance *Win32_PerfRawData_ClussvcPerfProvider_ClusterNetworkReconnections) SetPropertyUrgentMessageQueueLength(value uint64) (err error) {
	return instance.SetProperty("UrgentMessageQueueLength", (value))
}

// GetUrgentMessageQueueLength gets the value of UrgentMessageQueueLength for the instance
func (instance *Win32_PerfRawData_ClussvcPerfProvider_ClusterNetworkReconnections) GetPropertyUrgentMessageQueueLength() (value uint64, err error) {
	retValue, err := instance.GetProperty("UrgentMessageQueueLength")
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

// SetUrgentMessageQueueLengthPersec sets the value of UrgentMessageQueueLengthPersec for the instance
func (instance *Win32_PerfRawData_ClussvcPerfProvider_ClusterNetworkReconnections) SetPropertyUrgentMessageQueueLengthPersec(value uint64) (err error) {
	return instance.SetProperty("UrgentMessageQueueLengthPersec", (value))
}

// GetUrgentMessageQueueLengthPersec gets the value of UrgentMessageQueueLengthPersec for the instance
func (instance *Win32_PerfRawData_ClussvcPerfProvider_ClusterNetworkReconnections) GetPropertyUrgentMessageQueueLengthPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("UrgentMessageQueueLengthPersec")
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
