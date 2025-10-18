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

// Win32_PerfRawData_ClussvcPerfProvider_ClusterCheckpointManager struct
type Win32_PerfRawData_ClussvcPerfProvider_ClusterCheckpointManager struct {
	*Win32_PerfRawData

	//
	CryptoCheckpointsRestored uint64

	//
	CryptoCheckpointsRestoredPersec uint64

	//
	CryptoCheckpointsSaved uint64

	//
	CryptoCheckpointsSavedPersec uint64

	//
	RegistryCheckpointsRestored uint64

	//
	RegistryCheckpointsRestoredPersec uint64

	//
	RegistryCheckpointsSaved uint64

	//
	RegistryCheckpointsSavedPersec uint64
}

func NewWin32_PerfRawData_ClussvcPerfProvider_ClusterCheckpointManagerEx1(instance *cim.WmiInstance) (newInstance *Win32_PerfRawData_ClussvcPerfProvider_ClusterCheckpointManager, err error) {
	tmp, err := NewWin32_PerfRawDataEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfRawData_ClussvcPerfProvider_ClusterCheckpointManager{
		Win32_PerfRawData: tmp,
	}
	return
}

func NewWin32_PerfRawData_ClussvcPerfProvider_ClusterCheckpointManagerEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_PerfRawData_ClussvcPerfProvider_ClusterCheckpointManager, err error) {
	tmp, err := NewWin32_PerfRawDataEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfRawData_ClussvcPerfProvider_ClusterCheckpointManager{
		Win32_PerfRawData: tmp,
	}
	return
}

// SetCryptoCheckpointsRestored sets the value of CryptoCheckpointsRestored for the instance
func (instance *Win32_PerfRawData_ClussvcPerfProvider_ClusterCheckpointManager) SetPropertyCryptoCheckpointsRestored(value uint64) (err error) {
	return instance.SetProperty("CryptoCheckpointsRestored", (value))
}

// GetCryptoCheckpointsRestored gets the value of CryptoCheckpointsRestored for the instance
func (instance *Win32_PerfRawData_ClussvcPerfProvider_ClusterCheckpointManager) GetPropertyCryptoCheckpointsRestored() (value uint64, err error) {
	retValue, err := instance.GetProperty("CryptoCheckpointsRestored")
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

// SetCryptoCheckpointsRestoredPersec sets the value of CryptoCheckpointsRestoredPersec for the instance
func (instance *Win32_PerfRawData_ClussvcPerfProvider_ClusterCheckpointManager) SetPropertyCryptoCheckpointsRestoredPersec(value uint64) (err error) {
	return instance.SetProperty("CryptoCheckpointsRestoredPersec", (value))
}

// GetCryptoCheckpointsRestoredPersec gets the value of CryptoCheckpointsRestoredPersec for the instance
func (instance *Win32_PerfRawData_ClussvcPerfProvider_ClusterCheckpointManager) GetPropertyCryptoCheckpointsRestoredPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("CryptoCheckpointsRestoredPersec")
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

// SetCryptoCheckpointsSaved sets the value of CryptoCheckpointsSaved for the instance
func (instance *Win32_PerfRawData_ClussvcPerfProvider_ClusterCheckpointManager) SetPropertyCryptoCheckpointsSaved(value uint64) (err error) {
	return instance.SetProperty("CryptoCheckpointsSaved", (value))
}

// GetCryptoCheckpointsSaved gets the value of CryptoCheckpointsSaved for the instance
func (instance *Win32_PerfRawData_ClussvcPerfProvider_ClusterCheckpointManager) GetPropertyCryptoCheckpointsSaved() (value uint64, err error) {
	retValue, err := instance.GetProperty("CryptoCheckpointsSaved")
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

// SetCryptoCheckpointsSavedPersec sets the value of CryptoCheckpointsSavedPersec for the instance
func (instance *Win32_PerfRawData_ClussvcPerfProvider_ClusterCheckpointManager) SetPropertyCryptoCheckpointsSavedPersec(value uint64) (err error) {
	return instance.SetProperty("CryptoCheckpointsSavedPersec", (value))
}

// GetCryptoCheckpointsSavedPersec gets the value of CryptoCheckpointsSavedPersec for the instance
func (instance *Win32_PerfRawData_ClussvcPerfProvider_ClusterCheckpointManager) GetPropertyCryptoCheckpointsSavedPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("CryptoCheckpointsSavedPersec")
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

// SetRegistryCheckpointsRestored sets the value of RegistryCheckpointsRestored for the instance
func (instance *Win32_PerfRawData_ClussvcPerfProvider_ClusterCheckpointManager) SetPropertyRegistryCheckpointsRestored(value uint64) (err error) {
	return instance.SetProperty("RegistryCheckpointsRestored", (value))
}

// GetRegistryCheckpointsRestored gets the value of RegistryCheckpointsRestored for the instance
func (instance *Win32_PerfRawData_ClussvcPerfProvider_ClusterCheckpointManager) GetPropertyRegistryCheckpointsRestored() (value uint64, err error) {
	retValue, err := instance.GetProperty("RegistryCheckpointsRestored")
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

// SetRegistryCheckpointsRestoredPersec sets the value of RegistryCheckpointsRestoredPersec for the instance
func (instance *Win32_PerfRawData_ClussvcPerfProvider_ClusterCheckpointManager) SetPropertyRegistryCheckpointsRestoredPersec(value uint64) (err error) {
	return instance.SetProperty("RegistryCheckpointsRestoredPersec", (value))
}

// GetRegistryCheckpointsRestoredPersec gets the value of RegistryCheckpointsRestoredPersec for the instance
func (instance *Win32_PerfRawData_ClussvcPerfProvider_ClusterCheckpointManager) GetPropertyRegistryCheckpointsRestoredPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("RegistryCheckpointsRestoredPersec")
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

// SetRegistryCheckpointsSaved sets the value of RegistryCheckpointsSaved for the instance
func (instance *Win32_PerfRawData_ClussvcPerfProvider_ClusterCheckpointManager) SetPropertyRegistryCheckpointsSaved(value uint64) (err error) {
	return instance.SetProperty("RegistryCheckpointsSaved", (value))
}

// GetRegistryCheckpointsSaved gets the value of RegistryCheckpointsSaved for the instance
func (instance *Win32_PerfRawData_ClussvcPerfProvider_ClusterCheckpointManager) GetPropertyRegistryCheckpointsSaved() (value uint64, err error) {
	retValue, err := instance.GetProperty("RegistryCheckpointsSaved")
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

// SetRegistryCheckpointsSavedPersec sets the value of RegistryCheckpointsSavedPersec for the instance
func (instance *Win32_PerfRawData_ClussvcPerfProvider_ClusterCheckpointManager) SetPropertyRegistryCheckpointsSavedPersec(value uint64) (err error) {
	return instance.SetProperty("RegistryCheckpointsSavedPersec", (value))
}

// GetRegistryCheckpointsSavedPersec gets the value of RegistryCheckpointsSavedPersec for the instance
func (instance *Win32_PerfRawData_ClussvcPerfProvider_ClusterCheckpointManager) GetPropertyRegistryCheckpointsSavedPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("RegistryCheckpointsSavedPersec")
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
