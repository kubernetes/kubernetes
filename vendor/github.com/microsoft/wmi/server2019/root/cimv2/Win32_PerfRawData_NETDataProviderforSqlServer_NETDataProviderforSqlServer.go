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

// Win32_PerfRawData_NETDataProviderforSqlServer_NETDataProviderforSqlServer struct
type Win32_PerfRawData_NETDataProviderforSqlServer_NETDataProviderforSqlServer struct {
	*Win32_PerfRawData

	//
	HardConnectsPerSecond uint32

	//
	HardDisconnectsPerSecond uint32

	//
	NumberOfActiveConnectionPoolGroups uint32

	//
	NumberOfActiveConnectionPools uint32

	//
	NumberOfActiveConnections uint32

	//
	NumberOfFreeConnections uint32

	//
	NumberOfInactiveConnectionPoolGroups uint32

	//
	NumberOfInactiveConnectionPools uint32

	//
	NumberOfNonPooledConnections uint32

	//
	NumberOfPooledConnections uint32

	//
	NumberOfReclaimedConnections uint32

	//
	NumberOfStasisConnections uint32

	//
	SoftConnectsPerSecond uint32

	//
	SoftDisconnectsPerSecond uint32
}

func NewWin32_PerfRawData_NETDataProviderforSqlServer_NETDataProviderforSqlServerEx1(instance *cim.WmiInstance) (newInstance *Win32_PerfRawData_NETDataProviderforSqlServer_NETDataProviderforSqlServer, err error) {
	tmp, err := NewWin32_PerfRawDataEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfRawData_NETDataProviderforSqlServer_NETDataProviderforSqlServer{
		Win32_PerfRawData: tmp,
	}
	return
}

func NewWin32_PerfRawData_NETDataProviderforSqlServer_NETDataProviderforSqlServerEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_PerfRawData_NETDataProviderforSqlServer_NETDataProviderforSqlServer, err error) {
	tmp, err := NewWin32_PerfRawDataEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfRawData_NETDataProviderforSqlServer_NETDataProviderforSqlServer{
		Win32_PerfRawData: tmp,
	}
	return
}

// SetHardConnectsPerSecond sets the value of HardConnectsPerSecond for the instance
func (instance *Win32_PerfRawData_NETDataProviderforSqlServer_NETDataProviderforSqlServer) SetPropertyHardConnectsPerSecond(value uint32) (err error) {
	return instance.SetProperty("HardConnectsPerSecond", (value))
}

// GetHardConnectsPerSecond gets the value of HardConnectsPerSecond for the instance
func (instance *Win32_PerfRawData_NETDataProviderforSqlServer_NETDataProviderforSqlServer) GetPropertyHardConnectsPerSecond() (value uint32, err error) {
	retValue, err := instance.GetProperty("HardConnectsPerSecond")
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

// SetHardDisconnectsPerSecond sets the value of HardDisconnectsPerSecond for the instance
func (instance *Win32_PerfRawData_NETDataProviderforSqlServer_NETDataProviderforSqlServer) SetPropertyHardDisconnectsPerSecond(value uint32) (err error) {
	return instance.SetProperty("HardDisconnectsPerSecond", (value))
}

// GetHardDisconnectsPerSecond gets the value of HardDisconnectsPerSecond for the instance
func (instance *Win32_PerfRawData_NETDataProviderforSqlServer_NETDataProviderforSqlServer) GetPropertyHardDisconnectsPerSecond() (value uint32, err error) {
	retValue, err := instance.GetProperty("HardDisconnectsPerSecond")
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

// SetNumberOfActiveConnectionPoolGroups sets the value of NumberOfActiveConnectionPoolGroups for the instance
func (instance *Win32_PerfRawData_NETDataProviderforSqlServer_NETDataProviderforSqlServer) SetPropertyNumberOfActiveConnectionPoolGroups(value uint32) (err error) {
	return instance.SetProperty("NumberOfActiveConnectionPoolGroups", (value))
}

// GetNumberOfActiveConnectionPoolGroups gets the value of NumberOfActiveConnectionPoolGroups for the instance
func (instance *Win32_PerfRawData_NETDataProviderforSqlServer_NETDataProviderforSqlServer) GetPropertyNumberOfActiveConnectionPoolGroups() (value uint32, err error) {
	retValue, err := instance.GetProperty("NumberOfActiveConnectionPoolGroups")
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

// SetNumberOfActiveConnectionPools sets the value of NumberOfActiveConnectionPools for the instance
func (instance *Win32_PerfRawData_NETDataProviderforSqlServer_NETDataProviderforSqlServer) SetPropertyNumberOfActiveConnectionPools(value uint32) (err error) {
	return instance.SetProperty("NumberOfActiveConnectionPools", (value))
}

// GetNumberOfActiveConnectionPools gets the value of NumberOfActiveConnectionPools for the instance
func (instance *Win32_PerfRawData_NETDataProviderforSqlServer_NETDataProviderforSqlServer) GetPropertyNumberOfActiveConnectionPools() (value uint32, err error) {
	retValue, err := instance.GetProperty("NumberOfActiveConnectionPools")
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

// SetNumberOfActiveConnections sets the value of NumberOfActiveConnections for the instance
func (instance *Win32_PerfRawData_NETDataProviderforSqlServer_NETDataProviderforSqlServer) SetPropertyNumberOfActiveConnections(value uint32) (err error) {
	return instance.SetProperty("NumberOfActiveConnections", (value))
}

// GetNumberOfActiveConnections gets the value of NumberOfActiveConnections for the instance
func (instance *Win32_PerfRawData_NETDataProviderforSqlServer_NETDataProviderforSqlServer) GetPropertyNumberOfActiveConnections() (value uint32, err error) {
	retValue, err := instance.GetProperty("NumberOfActiveConnections")
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

// SetNumberOfFreeConnections sets the value of NumberOfFreeConnections for the instance
func (instance *Win32_PerfRawData_NETDataProviderforSqlServer_NETDataProviderforSqlServer) SetPropertyNumberOfFreeConnections(value uint32) (err error) {
	return instance.SetProperty("NumberOfFreeConnections", (value))
}

// GetNumberOfFreeConnections gets the value of NumberOfFreeConnections for the instance
func (instance *Win32_PerfRawData_NETDataProviderforSqlServer_NETDataProviderforSqlServer) GetPropertyNumberOfFreeConnections() (value uint32, err error) {
	retValue, err := instance.GetProperty("NumberOfFreeConnections")
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

// SetNumberOfInactiveConnectionPoolGroups sets the value of NumberOfInactiveConnectionPoolGroups for the instance
func (instance *Win32_PerfRawData_NETDataProviderforSqlServer_NETDataProviderforSqlServer) SetPropertyNumberOfInactiveConnectionPoolGroups(value uint32) (err error) {
	return instance.SetProperty("NumberOfInactiveConnectionPoolGroups", (value))
}

// GetNumberOfInactiveConnectionPoolGroups gets the value of NumberOfInactiveConnectionPoolGroups for the instance
func (instance *Win32_PerfRawData_NETDataProviderforSqlServer_NETDataProviderforSqlServer) GetPropertyNumberOfInactiveConnectionPoolGroups() (value uint32, err error) {
	retValue, err := instance.GetProperty("NumberOfInactiveConnectionPoolGroups")
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

// SetNumberOfInactiveConnectionPools sets the value of NumberOfInactiveConnectionPools for the instance
func (instance *Win32_PerfRawData_NETDataProviderforSqlServer_NETDataProviderforSqlServer) SetPropertyNumberOfInactiveConnectionPools(value uint32) (err error) {
	return instance.SetProperty("NumberOfInactiveConnectionPools", (value))
}

// GetNumberOfInactiveConnectionPools gets the value of NumberOfInactiveConnectionPools for the instance
func (instance *Win32_PerfRawData_NETDataProviderforSqlServer_NETDataProviderforSqlServer) GetPropertyNumberOfInactiveConnectionPools() (value uint32, err error) {
	retValue, err := instance.GetProperty("NumberOfInactiveConnectionPools")
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

// SetNumberOfNonPooledConnections sets the value of NumberOfNonPooledConnections for the instance
func (instance *Win32_PerfRawData_NETDataProviderforSqlServer_NETDataProviderforSqlServer) SetPropertyNumberOfNonPooledConnections(value uint32) (err error) {
	return instance.SetProperty("NumberOfNonPooledConnections", (value))
}

// GetNumberOfNonPooledConnections gets the value of NumberOfNonPooledConnections for the instance
func (instance *Win32_PerfRawData_NETDataProviderforSqlServer_NETDataProviderforSqlServer) GetPropertyNumberOfNonPooledConnections() (value uint32, err error) {
	retValue, err := instance.GetProperty("NumberOfNonPooledConnections")
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

// SetNumberOfPooledConnections sets the value of NumberOfPooledConnections for the instance
func (instance *Win32_PerfRawData_NETDataProviderforSqlServer_NETDataProviderforSqlServer) SetPropertyNumberOfPooledConnections(value uint32) (err error) {
	return instance.SetProperty("NumberOfPooledConnections", (value))
}

// GetNumberOfPooledConnections gets the value of NumberOfPooledConnections for the instance
func (instance *Win32_PerfRawData_NETDataProviderforSqlServer_NETDataProviderforSqlServer) GetPropertyNumberOfPooledConnections() (value uint32, err error) {
	retValue, err := instance.GetProperty("NumberOfPooledConnections")
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

// SetNumberOfReclaimedConnections sets the value of NumberOfReclaimedConnections for the instance
func (instance *Win32_PerfRawData_NETDataProviderforSqlServer_NETDataProviderforSqlServer) SetPropertyNumberOfReclaimedConnections(value uint32) (err error) {
	return instance.SetProperty("NumberOfReclaimedConnections", (value))
}

// GetNumberOfReclaimedConnections gets the value of NumberOfReclaimedConnections for the instance
func (instance *Win32_PerfRawData_NETDataProviderforSqlServer_NETDataProviderforSqlServer) GetPropertyNumberOfReclaimedConnections() (value uint32, err error) {
	retValue, err := instance.GetProperty("NumberOfReclaimedConnections")
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

// SetNumberOfStasisConnections sets the value of NumberOfStasisConnections for the instance
func (instance *Win32_PerfRawData_NETDataProviderforSqlServer_NETDataProviderforSqlServer) SetPropertyNumberOfStasisConnections(value uint32) (err error) {
	return instance.SetProperty("NumberOfStasisConnections", (value))
}

// GetNumberOfStasisConnections gets the value of NumberOfStasisConnections for the instance
func (instance *Win32_PerfRawData_NETDataProviderforSqlServer_NETDataProviderforSqlServer) GetPropertyNumberOfStasisConnections() (value uint32, err error) {
	retValue, err := instance.GetProperty("NumberOfStasisConnections")
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

// SetSoftConnectsPerSecond sets the value of SoftConnectsPerSecond for the instance
func (instance *Win32_PerfRawData_NETDataProviderforSqlServer_NETDataProviderforSqlServer) SetPropertySoftConnectsPerSecond(value uint32) (err error) {
	return instance.SetProperty("SoftConnectsPerSecond", (value))
}

// GetSoftConnectsPerSecond gets the value of SoftConnectsPerSecond for the instance
func (instance *Win32_PerfRawData_NETDataProviderforSqlServer_NETDataProviderforSqlServer) GetPropertySoftConnectsPerSecond() (value uint32, err error) {
	retValue, err := instance.GetProperty("SoftConnectsPerSecond")
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

// SetSoftDisconnectsPerSecond sets the value of SoftDisconnectsPerSecond for the instance
func (instance *Win32_PerfRawData_NETDataProviderforSqlServer_NETDataProviderforSqlServer) SetPropertySoftDisconnectsPerSecond(value uint32) (err error) {
	return instance.SetProperty("SoftDisconnectsPerSecond", (value))
}

// GetSoftDisconnectsPerSecond gets the value of SoftDisconnectsPerSecond for the instance
func (instance *Win32_PerfRawData_NETDataProviderforSqlServer_NETDataProviderforSqlServer) GetPropertySoftDisconnectsPerSecond() (value uint32, err error) {
	retValue, err := instance.GetProperty("SoftDisconnectsPerSecond")
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
