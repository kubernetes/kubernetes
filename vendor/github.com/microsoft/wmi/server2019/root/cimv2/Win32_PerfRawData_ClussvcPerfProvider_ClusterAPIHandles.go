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

// Win32_PerfRawData_ClussvcPerfProvider_ClusterAPIHandles struct
type Win32_PerfRawData_ClussvcPerfProvider_ClusterAPIHandles struct {
	*Win32_PerfRawData

	//
	BatchHandles uint64

	//
	BatchHandlesPersec uint64

	//
	ClusterHandles uint64

	//
	ClusterHandlesPersec uint64

	//
	GroupHandles uint64

	//
	GroupHandlesPersec uint64

	//
	KeyHandles uint64

	//
	KeyHandlesPersec uint64

	//
	NetworkHandles uint64

	//
	NetworkHandlesPersec uint64

	//
	NetworkInterfaceHandles uint64

	//
	NetworkInterfaceHandlesPersec uint64

	//
	NodeHandles uint64

	//
	NodeHandlesPersec uint64

	//
	NotificationHandles uint64

	//
	NotificationHandlesPersec uint64

	//
	ResourceHandles uint64

	//
	ResourceHandlesPersec uint64
}

func NewWin32_PerfRawData_ClussvcPerfProvider_ClusterAPIHandlesEx1(instance *cim.WmiInstance) (newInstance *Win32_PerfRawData_ClussvcPerfProvider_ClusterAPIHandles, err error) {
	tmp, err := NewWin32_PerfRawDataEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfRawData_ClussvcPerfProvider_ClusterAPIHandles{
		Win32_PerfRawData: tmp,
	}
	return
}

func NewWin32_PerfRawData_ClussvcPerfProvider_ClusterAPIHandlesEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_PerfRawData_ClussvcPerfProvider_ClusterAPIHandles, err error) {
	tmp, err := NewWin32_PerfRawDataEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfRawData_ClussvcPerfProvider_ClusterAPIHandles{
		Win32_PerfRawData: tmp,
	}
	return
}

// SetBatchHandles sets the value of BatchHandles for the instance
func (instance *Win32_PerfRawData_ClussvcPerfProvider_ClusterAPIHandles) SetPropertyBatchHandles(value uint64) (err error) {
	return instance.SetProperty("BatchHandles", (value))
}

// GetBatchHandles gets the value of BatchHandles for the instance
func (instance *Win32_PerfRawData_ClussvcPerfProvider_ClusterAPIHandles) GetPropertyBatchHandles() (value uint64, err error) {
	retValue, err := instance.GetProperty("BatchHandles")
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

// SetBatchHandlesPersec sets the value of BatchHandlesPersec for the instance
func (instance *Win32_PerfRawData_ClussvcPerfProvider_ClusterAPIHandles) SetPropertyBatchHandlesPersec(value uint64) (err error) {
	return instance.SetProperty("BatchHandlesPersec", (value))
}

// GetBatchHandlesPersec gets the value of BatchHandlesPersec for the instance
func (instance *Win32_PerfRawData_ClussvcPerfProvider_ClusterAPIHandles) GetPropertyBatchHandlesPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("BatchHandlesPersec")
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

// SetClusterHandles sets the value of ClusterHandles for the instance
func (instance *Win32_PerfRawData_ClussvcPerfProvider_ClusterAPIHandles) SetPropertyClusterHandles(value uint64) (err error) {
	return instance.SetProperty("ClusterHandles", (value))
}

// GetClusterHandles gets the value of ClusterHandles for the instance
func (instance *Win32_PerfRawData_ClussvcPerfProvider_ClusterAPIHandles) GetPropertyClusterHandles() (value uint64, err error) {
	retValue, err := instance.GetProperty("ClusterHandles")
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

// SetClusterHandlesPersec sets the value of ClusterHandlesPersec for the instance
func (instance *Win32_PerfRawData_ClussvcPerfProvider_ClusterAPIHandles) SetPropertyClusterHandlesPersec(value uint64) (err error) {
	return instance.SetProperty("ClusterHandlesPersec", (value))
}

// GetClusterHandlesPersec gets the value of ClusterHandlesPersec for the instance
func (instance *Win32_PerfRawData_ClussvcPerfProvider_ClusterAPIHandles) GetPropertyClusterHandlesPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("ClusterHandlesPersec")
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

// SetGroupHandles sets the value of GroupHandles for the instance
func (instance *Win32_PerfRawData_ClussvcPerfProvider_ClusterAPIHandles) SetPropertyGroupHandles(value uint64) (err error) {
	return instance.SetProperty("GroupHandles", (value))
}

// GetGroupHandles gets the value of GroupHandles for the instance
func (instance *Win32_PerfRawData_ClussvcPerfProvider_ClusterAPIHandles) GetPropertyGroupHandles() (value uint64, err error) {
	retValue, err := instance.GetProperty("GroupHandles")
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

// SetGroupHandlesPersec sets the value of GroupHandlesPersec for the instance
func (instance *Win32_PerfRawData_ClussvcPerfProvider_ClusterAPIHandles) SetPropertyGroupHandlesPersec(value uint64) (err error) {
	return instance.SetProperty("GroupHandlesPersec", (value))
}

// GetGroupHandlesPersec gets the value of GroupHandlesPersec for the instance
func (instance *Win32_PerfRawData_ClussvcPerfProvider_ClusterAPIHandles) GetPropertyGroupHandlesPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("GroupHandlesPersec")
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

// SetKeyHandles sets the value of KeyHandles for the instance
func (instance *Win32_PerfRawData_ClussvcPerfProvider_ClusterAPIHandles) SetPropertyKeyHandles(value uint64) (err error) {
	return instance.SetProperty("KeyHandles", (value))
}

// GetKeyHandles gets the value of KeyHandles for the instance
func (instance *Win32_PerfRawData_ClussvcPerfProvider_ClusterAPIHandles) GetPropertyKeyHandles() (value uint64, err error) {
	retValue, err := instance.GetProperty("KeyHandles")
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

// SetKeyHandlesPersec sets the value of KeyHandlesPersec for the instance
func (instance *Win32_PerfRawData_ClussvcPerfProvider_ClusterAPIHandles) SetPropertyKeyHandlesPersec(value uint64) (err error) {
	return instance.SetProperty("KeyHandlesPersec", (value))
}

// GetKeyHandlesPersec gets the value of KeyHandlesPersec for the instance
func (instance *Win32_PerfRawData_ClussvcPerfProvider_ClusterAPIHandles) GetPropertyKeyHandlesPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("KeyHandlesPersec")
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

// SetNetworkHandles sets the value of NetworkHandles for the instance
func (instance *Win32_PerfRawData_ClussvcPerfProvider_ClusterAPIHandles) SetPropertyNetworkHandles(value uint64) (err error) {
	return instance.SetProperty("NetworkHandles", (value))
}

// GetNetworkHandles gets the value of NetworkHandles for the instance
func (instance *Win32_PerfRawData_ClussvcPerfProvider_ClusterAPIHandles) GetPropertyNetworkHandles() (value uint64, err error) {
	retValue, err := instance.GetProperty("NetworkHandles")
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

// SetNetworkHandlesPersec sets the value of NetworkHandlesPersec for the instance
func (instance *Win32_PerfRawData_ClussvcPerfProvider_ClusterAPIHandles) SetPropertyNetworkHandlesPersec(value uint64) (err error) {
	return instance.SetProperty("NetworkHandlesPersec", (value))
}

// GetNetworkHandlesPersec gets the value of NetworkHandlesPersec for the instance
func (instance *Win32_PerfRawData_ClussvcPerfProvider_ClusterAPIHandles) GetPropertyNetworkHandlesPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("NetworkHandlesPersec")
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

// SetNetworkInterfaceHandles sets the value of NetworkInterfaceHandles for the instance
func (instance *Win32_PerfRawData_ClussvcPerfProvider_ClusterAPIHandles) SetPropertyNetworkInterfaceHandles(value uint64) (err error) {
	return instance.SetProperty("NetworkInterfaceHandles", (value))
}

// GetNetworkInterfaceHandles gets the value of NetworkInterfaceHandles for the instance
func (instance *Win32_PerfRawData_ClussvcPerfProvider_ClusterAPIHandles) GetPropertyNetworkInterfaceHandles() (value uint64, err error) {
	retValue, err := instance.GetProperty("NetworkInterfaceHandles")
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

// SetNetworkInterfaceHandlesPersec sets the value of NetworkInterfaceHandlesPersec for the instance
func (instance *Win32_PerfRawData_ClussvcPerfProvider_ClusterAPIHandles) SetPropertyNetworkInterfaceHandlesPersec(value uint64) (err error) {
	return instance.SetProperty("NetworkInterfaceHandlesPersec", (value))
}

// GetNetworkInterfaceHandlesPersec gets the value of NetworkInterfaceHandlesPersec for the instance
func (instance *Win32_PerfRawData_ClussvcPerfProvider_ClusterAPIHandles) GetPropertyNetworkInterfaceHandlesPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("NetworkInterfaceHandlesPersec")
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

// SetNodeHandles sets the value of NodeHandles for the instance
func (instance *Win32_PerfRawData_ClussvcPerfProvider_ClusterAPIHandles) SetPropertyNodeHandles(value uint64) (err error) {
	return instance.SetProperty("NodeHandles", (value))
}

// GetNodeHandles gets the value of NodeHandles for the instance
func (instance *Win32_PerfRawData_ClussvcPerfProvider_ClusterAPIHandles) GetPropertyNodeHandles() (value uint64, err error) {
	retValue, err := instance.GetProperty("NodeHandles")
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

// SetNodeHandlesPersec sets the value of NodeHandlesPersec for the instance
func (instance *Win32_PerfRawData_ClussvcPerfProvider_ClusterAPIHandles) SetPropertyNodeHandlesPersec(value uint64) (err error) {
	return instance.SetProperty("NodeHandlesPersec", (value))
}

// GetNodeHandlesPersec gets the value of NodeHandlesPersec for the instance
func (instance *Win32_PerfRawData_ClussvcPerfProvider_ClusterAPIHandles) GetPropertyNodeHandlesPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("NodeHandlesPersec")
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

// SetNotificationHandles sets the value of NotificationHandles for the instance
func (instance *Win32_PerfRawData_ClussvcPerfProvider_ClusterAPIHandles) SetPropertyNotificationHandles(value uint64) (err error) {
	return instance.SetProperty("NotificationHandles", (value))
}

// GetNotificationHandles gets the value of NotificationHandles for the instance
func (instance *Win32_PerfRawData_ClussvcPerfProvider_ClusterAPIHandles) GetPropertyNotificationHandles() (value uint64, err error) {
	retValue, err := instance.GetProperty("NotificationHandles")
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

// SetNotificationHandlesPersec sets the value of NotificationHandlesPersec for the instance
func (instance *Win32_PerfRawData_ClussvcPerfProvider_ClusterAPIHandles) SetPropertyNotificationHandlesPersec(value uint64) (err error) {
	return instance.SetProperty("NotificationHandlesPersec", (value))
}

// GetNotificationHandlesPersec gets the value of NotificationHandlesPersec for the instance
func (instance *Win32_PerfRawData_ClussvcPerfProvider_ClusterAPIHandles) GetPropertyNotificationHandlesPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("NotificationHandlesPersec")
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

// SetResourceHandles sets the value of ResourceHandles for the instance
func (instance *Win32_PerfRawData_ClussvcPerfProvider_ClusterAPIHandles) SetPropertyResourceHandles(value uint64) (err error) {
	return instance.SetProperty("ResourceHandles", (value))
}

// GetResourceHandles gets the value of ResourceHandles for the instance
func (instance *Win32_PerfRawData_ClussvcPerfProvider_ClusterAPIHandles) GetPropertyResourceHandles() (value uint64, err error) {
	retValue, err := instance.GetProperty("ResourceHandles")
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

// SetResourceHandlesPersec sets the value of ResourceHandlesPersec for the instance
func (instance *Win32_PerfRawData_ClussvcPerfProvider_ClusterAPIHandles) SetPropertyResourceHandlesPersec(value uint64) (err error) {
	return instance.SetProperty("ResourceHandlesPersec", (value))
}

// GetResourceHandlesPersec gets the value of ResourceHandlesPersec for the instance
func (instance *Win32_PerfRawData_ClussvcPerfProvider_ClusterAPIHandles) GetPropertyResourceHandlesPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("ResourceHandlesPersec")
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
