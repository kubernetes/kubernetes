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

// Win32_PerfFormattedData_ClussvcPerfProvider_ClusterAPICalls struct
type Win32_PerfFormattedData_ClussvcPerfProvider_ClusterAPICalls struct {
	*Win32_PerfFormattedData

	//
	ClusterAPICallsPersec uint64

	//
	GroupAPICallsPersec uint64

	//
	KeyAPICallsPersec uint64

	//
	NetworkAPICallsPersec uint64

	//
	NetworkInterfaceAPICallsPersec uint64

	//
	NodeAPICallsPersec uint64

	//
	NotificationAPICallsPersec uint64

	//
	NotificationBatchAPICallsPersec uint64

	//
	ResourceAPICallsPersec uint64
}

func NewWin32_PerfFormattedData_ClussvcPerfProvider_ClusterAPICallsEx1(instance *cim.WmiInstance) (newInstance *Win32_PerfFormattedData_ClussvcPerfProvider_ClusterAPICalls, err error) {
	tmp, err := NewWin32_PerfFormattedDataEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfFormattedData_ClussvcPerfProvider_ClusterAPICalls{
		Win32_PerfFormattedData: tmp,
	}
	return
}

func NewWin32_PerfFormattedData_ClussvcPerfProvider_ClusterAPICallsEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_PerfFormattedData_ClussvcPerfProvider_ClusterAPICalls, err error) {
	tmp, err := NewWin32_PerfFormattedDataEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfFormattedData_ClussvcPerfProvider_ClusterAPICalls{
		Win32_PerfFormattedData: tmp,
	}
	return
}

// SetClusterAPICallsPersec sets the value of ClusterAPICallsPersec for the instance
func (instance *Win32_PerfFormattedData_ClussvcPerfProvider_ClusterAPICalls) SetPropertyClusterAPICallsPersec(value uint64) (err error) {
	return instance.SetProperty("ClusterAPICallsPersec", (value))
}

// GetClusterAPICallsPersec gets the value of ClusterAPICallsPersec for the instance
func (instance *Win32_PerfFormattedData_ClussvcPerfProvider_ClusterAPICalls) GetPropertyClusterAPICallsPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("ClusterAPICallsPersec")
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

// SetGroupAPICallsPersec sets the value of GroupAPICallsPersec for the instance
func (instance *Win32_PerfFormattedData_ClussvcPerfProvider_ClusterAPICalls) SetPropertyGroupAPICallsPersec(value uint64) (err error) {
	return instance.SetProperty("GroupAPICallsPersec", (value))
}

// GetGroupAPICallsPersec gets the value of GroupAPICallsPersec for the instance
func (instance *Win32_PerfFormattedData_ClussvcPerfProvider_ClusterAPICalls) GetPropertyGroupAPICallsPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("GroupAPICallsPersec")
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

// SetKeyAPICallsPersec sets the value of KeyAPICallsPersec for the instance
func (instance *Win32_PerfFormattedData_ClussvcPerfProvider_ClusterAPICalls) SetPropertyKeyAPICallsPersec(value uint64) (err error) {
	return instance.SetProperty("KeyAPICallsPersec", (value))
}

// GetKeyAPICallsPersec gets the value of KeyAPICallsPersec for the instance
func (instance *Win32_PerfFormattedData_ClussvcPerfProvider_ClusterAPICalls) GetPropertyKeyAPICallsPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("KeyAPICallsPersec")
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

// SetNetworkAPICallsPersec sets the value of NetworkAPICallsPersec for the instance
func (instance *Win32_PerfFormattedData_ClussvcPerfProvider_ClusterAPICalls) SetPropertyNetworkAPICallsPersec(value uint64) (err error) {
	return instance.SetProperty("NetworkAPICallsPersec", (value))
}

// GetNetworkAPICallsPersec gets the value of NetworkAPICallsPersec for the instance
func (instance *Win32_PerfFormattedData_ClussvcPerfProvider_ClusterAPICalls) GetPropertyNetworkAPICallsPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("NetworkAPICallsPersec")
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

// SetNetworkInterfaceAPICallsPersec sets the value of NetworkInterfaceAPICallsPersec for the instance
func (instance *Win32_PerfFormattedData_ClussvcPerfProvider_ClusterAPICalls) SetPropertyNetworkInterfaceAPICallsPersec(value uint64) (err error) {
	return instance.SetProperty("NetworkInterfaceAPICallsPersec", (value))
}

// GetNetworkInterfaceAPICallsPersec gets the value of NetworkInterfaceAPICallsPersec for the instance
func (instance *Win32_PerfFormattedData_ClussvcPerfProvider_ClusterAPICalls) GetPropertyNetworkInterfaceAPICallsPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("NetworkInterfaceAPICallsPersec")
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

// SetNodeAPICallsPersec sets the value of NodeAPICallsPersec for the instance
func (instance *Win32_PerfFormattedData_ClussvcPerfProvider_ClusterAPICalls) SetPropertyNodeAPICallsPersec(value uint64) (err error) {
	return instance.SetProperty("NodeAPICallsPersec", (value))
}

// GetNodeAPICallsPersec gets the value of NodeAPICallsPersec for the instance
func (instance *Win32_PerfFormattedData_ClussvcPerfProvider_ClusterAPICalls) GetPropertyNodeAPICallsPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("NodeAPICallsPersec")
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

// SetNotificationAPICallsPersec sets the value of NotificationAPICallsPersec for the instance
func (instance *Win32_PerfFormattedData_ClussvcPerfProvider_ClusterAPICalls) SetPropertyNotificationAPICallsPersec(value uint64) (err error) {
	return instance.SetProperty("NotificationAPICallsPersec", (value))
}

// GetNotificationAPICallsPersec gets the value of NotificationAPICallsPersec for the instance
func (instance *Win32_PerfFormattedData_ClussvcPerfProvider_ClusterAPICalls) GetPropertyNotificationAPICallsPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("NotificationAPICallsPersec")
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

// SetNotificationBatchAPICallsPersec sets the value of NotificationBatchAPICallsPersec for the instance
func (instance *Win32_PerfFormattedData_ClussvcPerfProvider_ClusterAPICalls) SetPropertyNotificationBatchAPICallsPersec(value uint64) (err error) {
	return instance.SetProperty("NotificationBatchAPICallsPersec", (value))
}

// GetNotificationBatchAPICallsPersec gets the value of NotificationBatchAPICallsPersec for the instance
func (instance *Win32_PerfFormattedData_ClussvcPerfProvider_ClusterAPICalls) GetPropertyNotificationBatchAPICallsPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("NotificationBatchAPICallsPersec")
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

// SetResourceAPICallsPersec sets the value of ResourceAPICallsPersec for the instance
func (instance *Win32_PerfFormattedData_ClussvcPerfProvider_ClusterAPICalls) SetPropertyResourceAPICallsPersec(value uint64) (err error) {
	return instance.SetProperty("ResourceAPICallsPersec", (value))
}

// GetResourceAPICallsPersec gets the value of ResourceAPICallsPersec for the instance
func (instance *Win32_PerfFormattedData_ClussvcPerfProvider_ClusterAPICalls) GetPropertyResourceAPICallsPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("ResourceAPICallsPersec")
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
