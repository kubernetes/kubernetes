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

// Win32_PerfFormattedData_ClusBfltPerfProvider_ClusterStorageCacheStores struct
type Win32_PerfFormattedData_ClusBfltPerfProvider_ClusterStorageCacheStores struct {
	*Win32_PerfFormattedData

	//
	BindingsActive uint64

	//
	BindingsEnabled uint64

	//
	CachePages uint64

	//
	CachePagesBytes uint64

	//
	CachePagesDirty uint64

	//
	CachePagesFree uint64

	//
	CachePagesStandBy uint64

	//
	CachePagesStandByL0 uint64

	//
	CachePagesStandByL1 uint64

	//
	CachePagesStandByL2 uint64

	//
	CachePagesStandByOldestL1 uint64

	//
	CacheStores uint64

	//
	CacheUsageEfficiencyPercent uint64

	//
	CacheUsagePercent uint64

	//
	DestageBytes uint64

	//
	DestageBytesPersec uint64

	//
	DestagedAtLowPriPercent uint64

	//
	DestagedAtNormalPriPercent uint64

	//
	DestageTransfers uint64

	//
	DestageTransfersPersec uint64

	//
	DevicesBlocked uint64

	//
	DevicesHybrid uint64

	//
	DevicesMaintenance uint64

	//
	DevicesNotConfigured uint64

	//
	DevicesOrphan uint64

	//
	MultiPageFragments uint64

	//
	MultiPageFragmentsRate uint64

	//
	MultiPageReMap uint64

	//
	PageHit uint64

	//
	PageHitPersec uint64

	//
	PageReMap uint64

	//
	PageReMapPersec uint64

	//
	ReadErrorsMedia uint64

	//
	ReadErrorsTimeout uint64

	//
	ReadErrorsTotal uint64

	//
	UpdateBytes uint64

	//
	UpdateBytesPersec uint64

	//
	UpdatesCritical uint64

	//
	UpdatesCriticalLogFull uint64

	//
	UpdatesCriticalPersec uint64

	//
	UpdatesNonCritical uint64

	//
	UpdatesNonCriticalLogFull uint64

	//
	UpdatesNonCriticalPersec uint64

	//
	UpdatesNotCommitted uint64

	//
	UpdateTransfers uint64

	//
	UpdateTransfersPersec uint64

	//
	WriteErrorsMedia uint64

	//
	WriteErrorsTimeout uint64

	//
	WriteErrorsTotal uint64
}

func NewWin32_PerfFormattedData_ClusBfltPerfProvider_ClusterStorageCacheStoresEx1(instance *cim.WmiInstance) (newInstance *Win32_PerfFormattedData_ClusBfltPerfProvider_ClusterStorageCacheStores, err error) {
	tmp, err := NewWin32_PerfFormattedDataEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfFormattedData_ClusBfltPerfProvider_ClusterStorageCacheStores{
		Win32_PerfFormattedData: tmp,
	}
	return
}

func NewWin32_PerfFormattedData_ClusBfltPerfProvider_ClusterStorageCacheStoresEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_PerfFormattedData_ClusBfltPerfProvider_ClusterStorageCacheStores, err error) {
	tmp, err := NewWin32_PerfFormattedDataEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfFormattedData_ClusBfltPerfProvider_ClusterStorageCacheStores{
		Win32_PerfFormattedData: tmp,
	}
	return
}

// SetBindingsActive sets the value of BindingsActive for the instance
func (instance *Win32_PerfFormattedData_ClusBfltPerfProvider_ClusterStorageCacheStores) SetPropertyBindingsActive(value uint64) (err error) {
	return instance.SetProperty("BindingsActive", (value))
}

// GetBindingsActive gets the value of BindingsActive for the instance
func (instance *Win32_PerfFormattedData_ClusBfltPerfProvider_ClusterStorageCacheStores) GetPropertyBindingsActive() (value uint64, err error) {
	retValue, err := instance.GetProperty("BindingsActive")
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

// SetBindingsEnabled sets the value of BindingsEnabled for the instance
func (instance *Win32_PerfFormattedData_ClusBfltPerfProvider_ClusterStorageCacheStores) SetPropertyBindingsEnabled(value uint64) (err error) {
	return instance.SetProperty("BindingsEnabled", (value))
}

// GetBindingsEnabled gets the value of BindingsEnabled for the instance
func (instance *Win32_PerfFormattedData_ClusBfltPerfProvider_ClusterStorageCacheStores) GetPropertyBindingsEnabled() (value uint64, err error) {
	retValue, err := instance.GetProperty("BindingsEnabled")
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

// SetCachePages sets the value of CachePages for the instance
func (instance *Win32_PerfFormattedData_ClusBfltPerfProvider_ClusterStorageCacheStores) SetPropertyCachePages(value uint64) (err error) {
	return instance.SetProperty("CachePages", (value))
}

// GetCachePages gets the value of CachePages for the instance
func (instance *Win32_PerfFormattedData_ClusBfltPerfProvider_ClusterStorageCacheStores) GetPropertyCachePages() (value uint64, err error) {
	retValue, err := instance.GetProperty("CachePages")
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

// SetCachePagesBytes sets the value of CachePagesBytes for the instance
func (instance *Win32_PerfFormattedData_ClusBfltPerfProvider_ClusterStorageCacheStores) SetPropertyCachePagesBytes(value uint64) (err error) {
	return instance.SetProperty("CachePagesBytes", (value))
}

// GetCachePagesBytes gets the value of CachePagesBytes for the instance
func (instance *Win32_PerfFormattedData_ClusBfltPerfProvider_ClusterStorageCacheStores) GetPropertyCachePagesBytes() (value uint64, err error) {
	retValue, err := instance.GetProperty("CachePagesBytes")
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

// SetCachePagesDirty sets the value of CachePagesDirty for the instance
func (instance *Win32_PerfFormattedData_ClusBfltPerfProvider_ClusterStorageCacheStores) SetPropertyCachePagesDirty(value uint64) (err error) {
	return instance.SetProperty("CachePagesDirty", (value))
}

// GetCachePagesDirty gets the value of CachePagesDirty for the instance
func (instance *Win32_PerfFormattedData_ClusBfltPerfProvider_ClusterStorageCacheStores) GetPropertyCachePagesDirty() (value uint64, err error) {
	retValue, err := instance.GetProperty("CachePagesDirty")
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

// SetCachePagesFree sets the value of CachePagesFree for the instance
func (instance *Win32_PerfFormattedData_ClusBfltPerfProvider_ClusterStorageCacheStores) SetPropertyCachePagesFree(value uint64) (err error) {
	return instance.SetProperty("CachePagesFree", (value))
}

// GetCachePagesFree gets the value of CachePagesFree for the instance
func (instance *Win32_PerfFormattedData_ClusBfltPerfProvider_ClusterStorageCacheStores) GetPropertyCachePagesFree() (value uint64, err error) {
	retValue, err := instance.GetProperty("CachePagesFree")
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

// SetCachePagesStandBy sets the value of CachePagesStandBy for the instance
func (instance *Win32_PerfFormattedData_ClusBfltPerfProvider_ClusterStorageCacheStores) SetPropertyCachePagesStandBy(value uint64) (err error) {
	return instance.SetProperty("CachePagesStandBy", (value))
}

// GetCachePagesStandBy gets the value of CachePagesStandBy for the instance
func (instance *Win32_PerfFormattedData_ClusBfltPerfProvider_ClusterStorageCacheStores) GetPropertyCachePagesStandBy() (value uint64, err error) {
	retValue, err := instance.GetProperty("CachePagesStandBy")
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

// SetCachePagesStandByL0 sets the value of CachePagesStandByL0 for the instance
func (instance *Win32_PerfFormattedData_ClusBfltPerfProvider_ClusterStorageCacheStores) SetPropertyCachePagesStandByL0(value uint64) (err error) {
	return instance.SetProperty("CachePagesStandByL0", (value))
}

// GetCachePagesStandByL0 gets the value of CachePagesStandByL0 for the instance
func (instance *Win32_PerfFormattedData_ClusBfltPerfProvider_ClusterStorageCacheStores) GetPropertyCachePagesStandByL0() (value uint64, err error) {
	retValue, err := instance.GetProperty("CachePagesStandByL0")
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

// SetCachePagesStandByL1 sets the value of CachePagesStandByL1 for the instance
func (instance *Win32_PerfFormattedData_ClusBfltPerfProvider_ClusterStorageCacheStores) SetPropertyCachePagesStandByL1(value uint64) (err error) {
	return instance.SetProperty("CachePagesStandByL1", (value))
}

// GetCachePagesStandByL1 gets the value of CachePagesStandByL1 for the instance
func (instance *Win32_PerfFormattedData_ClusBfltPerfProvider_ClusterStorageCacheStores) GetPropertyCachePagesStandByL1() (value uint64, err error) {
	retValue, err := instance.GetProperty("CachePagesStandByL1")
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

// SetCachePagesStandByL2 sets the value of CachePagesStandByL2 for the instance
func (instance *Win32_PerfFormattedData_ClusBfltPerfProvider_ClusterStorageCacheStores) SetPropertyCachePagesStandByL2(value uint64) (err error) {
	return instance.SetProperty("CachePagesStandByL2", (value))
}

// GetCachePagesStandByL2 gets the value of CachePagesStandByL2 for the instance
func (instance *Win32_PerfFormattedData_ClusBfltPerfProvider_ClusterStorageCacheStores) GetPropertyCachePagesStandByL2() (value uint64, err error) {
	retValue, err := instance.GetProperty("CachePagesStandByL2")
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

// SetCachePagesStandByOldestL1 sets the value of CachePagesStandByOldestL1 for the instance
func (instance *Win32_PerfFormattedData_ClusBfltPerfProvider_ClusterStorageCacheStores) SetPropertyCachePagesStandByOldestL1(value uint64) (err error) {
	return instance.SetProperty("CachePagesStandByOldestL1", (value))
}

// GetCachePagesStandByOldestL1 gets the value of CachePagesStandByOldestL1 for the instance
func (instance *Win32_PerfFormattedData_ClusBfltPerfProvider_ClusterStorageCacheStores) GetPropertyCachePagesStandByOldestL1() (value uint64, err error) {
	retValue, err := instance.GetProperty("CachePagesStandByOldestL1")
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

// SetCacheStores sets the value of CacheStores for the instance
func (instance *Win32_PerfFormattedData_ClusBfltPerfProvider_ClusterStorageCacheStores) SetPropertyCacheStores(value uint64) (err error) {
	return instance.SetProperty("CacheStores", (value))
}

// GetCacheStores gets the value of CacheStores for the instance
func (instance *Win32_PerfFormattedData_ClusBfltPerfProvider_ClusterStorageCacheStores) GetPropertyCacheStores() (value uint64, err error) {
	retValue, err := instance.GetProperty("CacheStores")
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

// SetCacheUsageEfficiencyPercent sets the value of CacheUsageEfficiencyPercent for the instance
func (instance *Win32_PerfFormattedData_ClusBfltPerfProvider_ClusterStorageCacheStores) SetPropertyCacheUsageEfficiencyPercent(value uint64) (err error) {
	return instance.SetProperty("CacheUsageEfficiencyPercent", (value))
}

// GetCacheUsageEfficiencyPercent gets the value of CacheUsageEfficiencyPercent for the instance
func (instance *Win32_PerfFormattedData_ClusBfltPerfProvider_ClusterStorageCacheStores) GetPropertyCacheUsageEfficiencyPercent() (value uint64, err error) {
	retValue, err := instance.GetProperty("CacheUsageEfficiencyPercent")
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

// SetCacheUsagePercent sets the value of CacheUsagePercent for the instance
func (instance *Win32_PerfFormattedData_ClusBfltPerfProvider_ClusterStorageCacheStores) SetPropertyCacheUsagePercent(value uint64) (err error) {
	return instance.SetProperty("CacheUsagePercent", (value))
}

// GetCacheUsagePercent gets the value of CacheUsagePercent for the instance
func (instance *Win32_PerfFormattedData_ClusBfltPerfProvider_ClusterStorageCacheStores) GetPropertyCacheUsagePercent() (value uint64, err error) {
	retValue, err := instance.GetProperty("CacheUsagePercent")
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

// SetDestageBytes sets the value of DestageBytes for the instance
func (instance *Win32_PerfFormattedData_ClusBfltPerfProvider_ClusterStorageCacheStores) SetPropertyDestageBytes(value uint64) (err error) {
	return instance.SetProperty("DestageBytes", (value))
}

// GetDestageBytes gets the value of DestageBytes for the instance
func (instance *Win32_PerfFormattedData_ClusBfltPerfProvider_ClusterStorageCacheStores) GetPropertyDestageBytes() (value uint64, err error) {
	retValue, err := instance.GetProperty("DestageBytes")
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

// SetDestageBytesPersec sets the value of DestageBytesPersec for the instance
func (instance *Win32_PerfFormattedData_ClusBfltPerfProvider_ClusterStorageCacheStores) SetPropertyDestageBytesPersec(value uint64) (err error) {
	return instance.SetProperty("DestageBytesPersec", (value))
}

// GetDestageBytesPersec gets the value of DestageBytesPersec for the instance
func (instance *Win32_PerfFormattedData_ClusBfltPerfProvider_ClusterStorageCacheStores) GetPropertyDestageBytesPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("DestageBytesPersec")
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

// SetDestagedAtLowPriPercent sets the value of DestagedAtLowPriPercent for the instance
func (instance *Win32_PerfFormattedData_ClusBfltPerfProvider_ClusterStorageCacheStores) SetPropertyDestagedAtLowPriPercent(value uint64) (err error) {
	return instance.SetProperty("DestagedAtLowPriPercent", (value))
}

// GetDestagedAtLowPriPercent gets the value of DestagedAtLowPriPercent for the instance
func (instance *Win32_PerfFormattedData_ClusBfltPerfProvider_ClusterStorageCacheStores) GetPropertyDestagedAtLowPriPercent() (value uint64, err error) {
	retValue, err := instance.GetProperty("DestagedAtLowPriPercent")
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

// SetDestagedAtNormalPriPercent sets the value of DestagedAtNormalPriPercent for the instance
func (instance *Win32_PerfFormattedData_ClusBfltPerfProvider_ClusterStorageCacheStores) SetPropertyDestagedAtNormalPriPercent(value uint64) (err error) {
	return instance.SetProperty("DestagedAtNormalPriPercent", (value))
}

// GetDestagedAtNormalPriPercent gets the value of DestagedAtNormalPriPercent for the instance
func (instance *Win32_PerfFormattedData_ClusBfltPerfProvider_ClusterStorageCacheStores) GetPropertyDestagedAtNormalPriPercent() (value uint64, err error) {
	retValue, err := instance.GetProperty("DestagedAtNormalPriPercent")
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

// SetDestageTransfers sets the value of DestageTransfers for the instance
func (instance *Win32_PerfFormattedData_ClusBfltPerfProvider_ClusterStorageCacheStores) SetPropertyDestageTransfers(value uint64) (err error) {
	return instance.SetProperty("DestageTransfers", (value))
}

// GetDestageTransfers gets the value of DestageTransfers for the instance
func (instance *Win32_PerfFormattedData_ClusBfltPerfProvider_ClusterStorageCacheStores) GetPropertyDestageTransfers() (value uint64, err error) {
	retValue, err := instance.GetProperty("DestageTransfers")
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

// SetDestageTransfersPersec sets the value of DestageTransfersPersec for the instance
func (instance *Win32_PerfFormattedData_ClusBfltPerfProvider_ClusterStorageCacheStores) SetPropertyDestageTransfersPersec(value uint64) (err error) {
	return instance.SetProperty("DestageTransfersPersec", (value))
}

// GetDestageTransfersPersec gets the value of DestageTransfersPersec for the instance
func (instance *Win32_PerfFormattedData_ClusBfltPerfProvider_ClusterStorageCacheStores) GetPropertyDestageTransfersPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("DestageTransfersPersec")
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

// SetDevicesBlocked sets the value of DevicesBlocked for the instance
func (instance *Win32_PerfFormattedData_ClusBfltPerfProvider_ClusterStorageCacheStores) SetPropertyDevicesBlocked(value uint64) (err error) {
	return instance.SetProperty("DevicesBlocked", (value))
}

// GetDevicesBlocked gets the value of DevicesBlocked for the instance
func (instance *Win32_PerfFormattedData_ClusBfltPerfProvider_ClusterStorageCacheStores) GetPropertyDevicesBlocked() (value uint64, err error) {
	retValue, err := instance.GetProperty("DevicesBlocked")
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

// SetDevicesHybrid sets the value of DevicesHybrid for the instance
func (instance *Win32_PerfFormattedData_ClusBfltPerfProvider_ClusterStorageCacheStores) SetPropertyDevicesHybrid(value uint64) (err error) {
	return instance.SetProperty("DevicesHybrid", (value))
}

// GetDevicesHybrid gets the value of DevicesHybrid for the instance
func (instance *Win32_PerfFormattedData_ClusBfltPerfProvider_ClusterStorageCacheStores) GetPropertyDevicesHybrid() (value uint64, err error) {
	retValue, err := instance.GetProperty("DevicesHybrid")
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

// SetDevicesMaintenance sets the value of DevicesMaintenance for the instance
func (instance *Win32_PerfFormattedData_ClusBfltPerfProvider_ClusterStorageCacheStores) SetPropertyDevicesMaintenance(value uint64) (err error) {
	return instance.SetProperty("DevicesMaintenance", (value))
}

// GetDevicesMaintenance gets the value of DevicesMaintenance for the instance
func (instance *Win32_PerfFormattedData_ClusBfltPerfProvider_ClusterStorageCacheStores) GetPropertyDevicesMaintenance() (value uint64, err error) {
	retValue, err := instance.GetProperty("DevicesMaintenance")
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

// SetDevicesNotConfigured sets the value of DevicesNotConfigured for the instance
func (instance *Win32_PerfFormattedData_ClusBfltPerfProvider_ClusterStorageCacheStores) SetPropertyDevicesNotConfigured(value uint64) (err error) {
	return instance.SetProperty("DevicesNotConfigured", (value))
}

// GetDevicesNotConfigured gets the value of DevicesNotConfigured for the instance
func (instance *Win32_PerfFormattedData_ClusBfltPerfProvider_ClusterStorageCacheStores) GetPropertyDevicesNotConfigured() (value uint64, err error) {
	retValue, err := instance.GetProperty("DevicesNotConfigured")
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

// SetDevicesOrphan sets the value of DevicesOrphan for the instance
func (instance *Win32_PerfFormattedData_ClusBfltPerfProvider_ClusterStorageCacheStores) SetPropertyDevicesOrphan(value uint64) (err error) {
	return instance.SetProperty("DevicesOrphan", (value))
}

// GetDevicesOrphan gets the value of DevicesOrphan for the instance
func (instance *Win32_PerfFormattedData_ClusBfltPerfProvider_ClusterStorageCacheStores) GetPropertyDevicesOrphan() (value uint64, err error) {
	retValue, err := instance.GetProperty("DevicesOrphan")
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

// SetMultiPageFragments sets the value of MultiPageFragments for the instance
func (instance *Win32_PerfFormattedData_ClusBfltPerfProvider_ClusterStorageCacheStores) SetPropertyMultiPageFragments(value uint64) (err error) {
	return instance.SetProperty("MultiPageFragments", (value))
}

// GetMultiPageFragments gets the value of MultiPageFragments for the instance
func (instance *Win32_PerfFormattedData_ClusBfltPerfProvider_ClusterStorageCacheStores) GetPropertyMultiPageFragments() (value uint64, err error) {
	retValue, err := instance.GetProperty("MultiPageFragments")
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

// SetMultiPageFragmentsRate sets the value of MultiPageFragmentsRate for the instance
func (instance *Win32_PerfFormattedData_ClusBfltPerfProvider_ClusterStorageCacheStores) SetPropertyMultiPageFragmentsRate(value uint64) (err error) {
	return instance.SetProperty("MultiPageFragmentsRate", (value))
}

// GetMultiPageFragmentsRate gets the value of MultiPageFragmentsRate for the instance
func (instance *Win32_PerfFormattedData_ClusBfltPerfProvider_ClusterStorageCacheStores) GetPropertyMultiPageFragmentsRate() (value uint64, err error) {
	retValue, err := instance.GetProperty("MultiPageFragmentsRate")
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

// SetMultiPageReMap sets the value of MultiPageReMap for the instance
func (instance *Win32_PerfFormattedData_ClusBfltPerfProvider_ClusterStorageCacheStores) SetPropertyMultiPageReMap(value uint64) (err error) {
	return instance.SetProperty("MultiPageReMap", (value))
}

// GetMultiPageReMap gets the value of MultiPageReMap for the instance
func (instance *Win32_PerfFormattedData_ClusBfltPerfProvider_ClusterStorageCacheStores) GetPropertyMultiPageReMap() (value uint64, err error) {
	retValue, err := instance.GetProperty("MultiPageReMap")
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

// SetPageHit sets the value of PageHit for the instance
func (instance *Win32_PerfFormattedData_ClusBfltPerfProvider_ClusterStorageCacheStores) SetPropertyPageHit(value uint64) (err error) {
	return instance.SetProperty("PageHit", (value))
}

// GetPageHit gets the value of PageHit for the instance
func (instance *Win32_PerfFormattedData_ClusBfltPerfProvider_ClusterStorageCacheStores) GetPropertyPageHit() (value uint64, err error) {
	retValue, err := instance.GetProperty("PageHit")
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

// SetPageHitPersec sets the value of PageHitPersec for the instance
func (instance *Win32_PerfFormattedData_ClusBfltPerfProvider_ClusterStorageCacheStores) SetPropertyPageHitPersec(value uint64) (err error) {
	return instance.SetProperty("PageHitPersec", (value))
}

// GetPageHitPersec gets the value of PageHitPersec for the instance
func (instance *Win32_PerfFormattedData_ClusBfltPerfProvider_ClusterStorageCacheStores) GetPropertyPageHitPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("PageHitPersec")
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

// SetPageReMap sets the value of PageReMap for the instance
func (instance *Win32_PerfFormattedData_ClusBfltPerfProvider_ClusterStorageCacheStores) SetPropertyPageReMap(value uint64) (err error) {
	return instance.SetProperty("PageReMap", (value))
}

// GetPageReMap gets the value of PageReMap for the instance
func (instance *Win32_PerfFormattedData_ClusBfltPerfProvider_ClusterStorageCacheStores) GetPropertyPageReMap() (value uint64, err error) {
	retValue, err := instance.GetProperty("PageReMap")
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

// SetPageReMapPersec sets the value of PageReMapPersec for the instance
func (instance *Win32_PerfFormattedData_ClusBfltPerfProvider_ClusterStorageCacheStores) SetPropertyPageReMapPersec(value uint64) (err error) {
	return instance.SetProperty("PageReMapPersec", (value))
}

// GetPageReMapPersec gets the value of PageReMapPersec for the instance
func (instance *Win32_PerfFormattedData_ClusBfltPerfProvider_ClusterStorageCacheStores) GetPropertyPageReMapPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("PageReMapPersec")
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

// SetReadErrorsMedia sets the value of ReadErrorsMedia for the instance
func (instance *Win32_PerfFormattedData_ClusBfltPerfProvider_ClusterStorageCacheStores) SetPropertyReadErrorsMedia(value uint64) (err error) {
	return instance.SetProperty("ReadErrorsMedia", (value))
}

// GetReadErrorsMedia gets the value of ReadErrorsMedia for the instance
func (instance *Win32_PerfFormattedData_ClusBfltPerfProvider_ClusterStorageCacheStores) GetPropertyReadErrorsMedia() (value uint64, err error) {
	retValue, err := instance.GetProperty("ReadErrorsMedia")
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

// SetReadErrorsTimeout sets the value of ReadErrorsTimeout for the instance
func (instance *Win32_PerfFormattedData_ClusBfltPerfProvider_ClusterStorageCacheStores) SetPropertyReadErrorsTimeout(value uint64) (err error) {
	return instance.SetProperty("ReadErrorsTimeout", (value))
}

// GetReadErrorsTimeout gets the value of ReadErrorsTimeout for the instance
func (instance *Win32_PerfFormattedData_ClusBfltPerfProvider_ClusterStorageCacheStores) GetPropertyReadErrorsTimeout() (value uint64, err error) {
	retValue, err := instance.GetProperty("ReadErrorsTimeout")
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

// SetReadErrorsTotal sets the value of ReadErrorsTotal for the instance
func (instance *Win32_PerfFormattedData_ClusBfltPerfProvider_ClusterStorageCacheStores) SetPropertyReadErrorsTotal(value uint64) (err error) {
	return instance.SetProperty("ReadErrorsTotal", (value))
}

// GetReadErrorsTotal gets the value of ReadErrorsTotal for the instance
func (instance *Win32_PerfFormattedData_ClusBfltPerfProvider_ClusterStorageCacheStores) GetPropertyReadErrorsTotal() (value uint64, err error) {
	retValue, err := instance.GetProperty("ReadErrorsTotal")
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

// SetUpdateBytes sets the value of UpdateBytes for the instance
func (instance *Win32_PerfFormattedData_ClusBfltPerfProvider_ClusterStorageCacheStores) SetPropertyUpdateBytes(value uint64) (err error) {
	return instance.SetProperty("UpdateBytes", (value))
}

// GetUpdateBytes gets the value of UpdateBytes for the instance
func (instance *Win32_PerfFormattedData_ClusBfltPerfProvider_ClusterStorageCacheStores) GetPropertyUpdateBytes() (value uint64, err error) {
	retValue, err := instance.GetProperty("UpdateBytes")
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

// SetUpdateBytesPersec sets the value of UpdateBytesPersec for the instance
func (instance *Win32_PerfFormattedData_ClusBfltPerfProvider_ClusterStorageCacheStores) SetPropertyUpdateBytesPersec(value uint64) (err error) {
	return instance.SetProperty("UpdateBytesPersec", (value))
}

// GetUpdateBytesPersec gets the value of UpdateBytesPersec for the instance
func (instance *Win32_PerfFormattedData_ClusBfltPerfProvider_ClusterStorageCacheStores) GetPropertyUpdateBytesPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("UpdateBytesPersec")
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

// SetUpdatesCritical sets the value of UpdatesCritical for the instance
func (instance *Win32_PerfFormattedData_ClusBfltPerfProvider_ClusterStorageCacheStores) SetPropertyUpdatesCritical(value uint64) (err error) {
	return instance.SetProperty("UpdatesCritical", (value))
}

// GetUpdatesCritical gets the value of UpdatesCritical for the instance
func (instance *Win32_PerfFormattedData_ClusBfltPerfProvider_ClusterStorageCacheStores) GetPropertyUpdatesCritical() (value uint64, err error) {
	retValue, err := instance.GetProperty("UpdatesCritical")
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

// SetUpdatesCriticalLogFull sets the value of UpdatesCriticalLogFull for the instance
func (instance *Win32_PerfFormattedData_ClusBfltPerfProvider_ClusterStorageCacheStores) SetPropertyUpdatesCriticalLogFull(value uint64) (err error) {
	return instance.SetProperty("UpdatesCriticalLogFull", (value))
}

// GetUpdatesCriticalLogFull gets the value of UpdatesCriticalLogFull for the instance
func (instance *Win32_PerfFormattedData_ClusBfltPerfProvider_ClusterStorageCacheStores) GetPropertyUpdatesCriticalLogFull() (value uint64, err error) {
	retValue, err := instance.GetProperty("UpdatesCriticalLogFull")
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

// SetUpdatesCriticalPersec sets the value of UpdatesCriticalPersec for the instance
func (instance *Win32_PerfFormattedData_ClusBfltPerfProvider_ClusterStorageCacheStores) SetPropertyUpdatesCriticalPersec(value uint64) (err error) {
	return instance.SetProperty("UpdatesCriticalPersec", (value))
}

// GetUpdatesCriticalPersec gets the value of UpdatesCriticalPersec for the instance
func (instance *Win32_PerfFormattedData_ClusBfltPerfProvider_ClusterStorageCacheStores) GetPropertyUpdatesCriticalPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("UpdatesCriticalPersec")
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

// SetUpdatesNonCritical sets the value of UpdatesNonCritical for the instance
func (instance *Win32_PerfFormattedData_ClusBfltPerfProvider_ClusterStorageCacheStores) SetPropertyUpdatesNonCritical(value uint64) (err error) {
	return instance.SetProperty("UpdatesNonCritical", (value))
}

// GetUpdatesNonCritical gets the value of UpdatesNonCritical for the instance
func (instance *Win32_PerfFormattedData_ClusBfltPerfProvider_ClusterStorageCacheStores) GetPropertyUpdatesNonCritical() (value uint64, err error) {
	retValue, err := instance.GetProperty("UpdatesNonCritical")
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

// SetUpdatesNonCriticalLogFull sets the value of UpdatesNonCriticalLogFull for the instance
func (instance *Win32_PerfFormattedData_ClusBfltPerfProvider_ClusterStorageCacheStores) SetPropertyUpdatesNonCriticalLogFull(value uint64) (err error) {
	return instance.SetProperty("UpdatesNonCriticalLogFull", (value))
}

// GetUpdatesNonCriticalLogFull gets the value of UpdatesNonCriticalLogFull for the instance
func (instance *Win32_PerfFormattedData_ClusBfltPerfProvider_ClusterStorageCacheStores) GetPropertyUpdatesNonCriticalLogFull() (value uint64, err error) {
	retValue, err := instance.GetProperty("UpdatesNonCriticalLogFull")
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

// SetUpdatesNonCriticalPersec sets the value of UpdatesNonCriticalPersec for the instance
func (instance *Win32_PerfFormattedData_ClusBfltPerfProvider_ClusterStorageCacheStores) SetPropertyUpdatesNonCriticalPersec(value uint64) (err error) {
	return instance.SetProperty("UpdatesNonCriticalPersec", (value))
}

// GetUpdatesNonCriticalPersec gets the value of UpdatesNonCriticalPersec for the instance
func (instance *Win32_PerfFormattedData_ClusBfltPerfProvider_ClusterStorageCacheStores) GetPropertyUpdatesNonCriticalPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("UpdatesNonCriticalPersec")
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

// SetUpdatesNotCommitted sets the value of UpdatesNotCommitted for the instance
func (instance *Win32_PerfFormattedData_ClusBfltPerfProvider_ClusterStorageCacheStores) SetPropertyUpdatesNotCommitted(value uint64) (err error) {
	return instance.SetProperty("UpdatesNotCommitted", (value))
}

// GetUpdatesNotCommitted gets the value of UpdatesNotCommitted for the instance
func (instance *Win32_PerfFormattedData_ClusBfltPerfProvider_ClusterStorageCacheStores) GetPropertyUpdatesNotCommitted() (value uint64, err error) {
	retValue, err := instance.GetProperty("UpdatesNotCommitted")
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

// SetUpdateTransfers sets the value of UpdateTransfers for the instance
func (instance *Win32_PerfFormattedData_ClusBfltPerfProvider_ClusterStorageCacheStores) SetPropertyUpdateTransfers(value uint64) (err error) {
	return instance.SetProperty("UpdateTransfers", (value))
}

// GetUpdateTransfers gets the value of UpdateTransfers for the instance
func (instance *Win32_PerfFormattedData_ClusBfltPerfProvider_ClusterStorageCacheStores) GetPropertyUpdateTransfers() (value uint64, err error) {
	retValue, err := instance.GetProperty("UpdateTransfers")
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

// SetUpdateTransfersPersec sets the value of UpdateTransfersPersec for the instance
func (instance *Win32_PerfFormattedData_ClusBfltPerfProvider_ClusterStorageCacheStores) SetPropertyUpdateTransfersPersec(value uint64) (err error) {
	return instance.SetProperty("UpdateTransfersPersec", (value))
}

// GetUpdateTransfersPersec gets the value of UpdateTransfersPersec for the instance
func (instance *Win32_PerfFormattedData_ClusBfltPerfProvider_ClusterStorageCacheStores) GetPropertyUpdateTransfersPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("UpdateTransfersPersec")
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

// SetWriteErrorsMedia sets the value of WriteErrorsMedia for the instance
func (instance *Win32_PerfFormattedData_ClusBfltPerfProvider_ClusterStorageCacheStores) SetPropertyWriteErrorsMedia(value uint64) (err error) {
	return instance.SetProperty("WriteErrorsMedia", (value))
}

// GetWriteErrorsMedia gets the value of WriteErrorsMedia for the instance
func (instance *Win32_PerfFormattedData_ClusBfltPerfProvider_ClusterStorageCacheStores) GetPropertyWriteErrorsMedia() (value uint64, err error) {
	retValue, err := instance.GetProperty("WriteErrorsMedia")
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

// SetWriteErrorsTimeout sets the value of WriteErrorsTimeout for the instance
func (instance *Win32_PerfFormattedData_ClusBfltPerfProvider_ClusterStorageCacheStores) SetPropertyWriteErrorsTimeout(value uint64) (err error) {
	return instance.SetProperty("WriteErrorsTimeout", (value))
}

// GetWriteErrorsTimeout gets the value of WriteErrorsTimeout for the instance
func (instance *Win32_PerfFormattedData_ClusBfltPerfProvider_ClusterStorageCacheStores) GetPropertyWriteErrorsTimeout() (value uint64, err error) {
	retValue, err := instance.GetProperty("WriteErrorsTimeout")
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

// SetWriteErrorsTotal sets the value of WriteErrorsTotal for the instance
func (instance *Win32_PerfFormattedData_ClusBfltPerfProvider_ClusterStorageCacheStores) SetPropertyWriteErrorsTotal(value uint64) (err error) {
	return instance.SetProperty("WriteErrorsTotal", (value))
}

// GetWriteErrorsTotal gets the value of WriteErrorsTotal for the instance
func (instance *Win32_PerfFormattedData_ClusBfltPerfProvider_ClusterStorageCacheStores) GetPropertyWriteErrorsTotal() (value uint64, err error) {
	retValue, err := instance.GetProperty("WriteErrorsTotal")
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
