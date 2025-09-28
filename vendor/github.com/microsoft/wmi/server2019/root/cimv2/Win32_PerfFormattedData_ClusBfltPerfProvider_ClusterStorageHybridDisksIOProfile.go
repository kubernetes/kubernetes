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

// Win32_PerfFormattedData_ClusBfltPerfProvider_ClusterStorageHybridDisksIOProfile struct
type Win32_PerfFormattedData_ClusBfltPerfProvider_ClusterStorageHybridDisksIOProfile struct {
	*Win32_PerfFormattedData

	//
	CacheWriteBoosts uint64

	//
	CacheWriteBoostsPersec uint64

	//
	CacheWriteBoostsVeto uint64

	//
	CacheWriteBoostsVetoPersec uint64

	//
	Reads0K4K uint64

	//
	Reads1024K2048K uint64

	//
	Reads128K256K uint64

	//
	Reads16K32K uint64

	//
	Reads2048K4096K uint64

	//
	Reads256K512K uint64

	//
	Reads32K64K uint64

	//
	Reads4096Koo uint64

	//
	Reads4K8K uint64

	//
	Reads512K1024K uint64

	//
	Reads64K128K uint64

	//
	Reads8K16K uint64

	//
	Readsnotaligned uint64

	//
	ReadsPagingIO uint64

	//
	ReadsPersec0K4K uint64

	//
	ReadsPersec1024K2048K uint64

	//
	ReadsPersec128K256K uint64

	//
	ReadsPersec16K32K uint64

	//
	ReadsPersec2048K4096K uint64

	//
	ReadsPersec256K512K uint64

	//
	ReadsPersec32K64K uint64

	//
	ReadsPersec4096Koo uint64

	//
	ReadsPersec4K8K uint64

	//
	ReadsPersec512K1024K uint64

	//
	ReadsPersec64K128K uint64

	//
	ReadsPersec8K16K uint64

	//
	ReadsPersecnotaligned uint64

	//
	ReadsPersecPagingIO uint64

	//
	ReadsPersecTotal uint64

	//
	ReadsTotal uint64

	//
	Writes0K4K uint64

	//
	Writes1024K2048K uint64

	//
	Writes128K256K uint64

	//
	Writes16K32K uint64

	//
	Writes2048K4096K uint64

	//
	Writes256K512K uint64

	//
	Writes32K64K uint64

	//
	Writes4096Koo uint64

	//
	Writes4K8K uint64

	//
	Writes512K1024K uint64

	//
	Writes64K128K uint64

	//
	Writes8K16K uint64

	//
	Writesnotaligned uint64

	//
	WritesPagingIO uint64

	//
	WritesPersec0K4K uint64

	//
	WritesPersec1024K2048K uint64

	//
	WritesPersec128K256K uint64

	//
	WritesPersec16K32K uint64

	//
	WritesPersec2048K4096K uint64

	//
	WritesPersec256K512K uint64

	//
	WritesPersec32K64K uint64

	//
	WritesPersec4096Koo uint64

	//
	WritesPersec4K8K uint64

	//
	WritesPersec512K1024K uint64

	//
	WritesPersec64K128K uint64

	//
	WritesPersec8K16K uint64

	//
	WritesPersecnotaligned uint64

	//
	WritesPersecPagingIO uint64

	//
	WritesPersecTotal uint64

	//
	WritesTotal uint64
}

func NewWin32_PerfFormattedData_ClusBfltPerfProvider_ClusterStorageHybridDisksIOProfileEx1(instance *cim.WmiInstance) (newInstance *Win32_PerfFormattedData_ClusBfltPerfProvider_ClusterStorageHybridDisksIOProfile, err error) {
	tmp, err := NewWin32_PerfFormattedDataEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfFormattedData_ClusBfltPerfProvider_ClusterStorageHybridDisksIOProfile{
		Win32_PerfFormattedData: tmp,
	}
	return
}

func NewWin32_PerfFormattedData_ClusBfltPerfProvider_ClusterStorageHybridDisksIOProfileEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_PerfFormattedData_ClusBfltPerfProvider_ClusterStorageHybridDisksIOProfile, err error) {
	tmp, err := NewWin32_PerfFormattedDataEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfFormattedData_ClusBfltPerfProvider_ClusterStorageHybridDisksIOProfile{
		Win32_PerfFormattedData: tmp,
	}
	return
}

// SetCacheWriteBoosts sets the value of CacheWriteBoosts for the instance
func (instance *Win32_PerfFormattedData_ClusBfltPerfProvider_ClusterStorageHybridDisksIOProfile) SetPropertyCacheWriteBoosts(value uint64) (err error) {
	return instance.SetProperty("CacheWriteBoosts", (value))
}

// GetCacheWriteBoosts gets the value of CacheWriteBoosts for the instance
func (instance *Win32_PerfFormattedData_ClusBfltPerfProvider_ClusterStorageHybridDisksIOProfile) GetPropertyCacheWriteBoosts() (value uint64, err error) {
	retValue, err := instance.GetProperty("CacheWriteBoosts")
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

// SetCacheWriteBoostsPersec sets the value of CacheWriteBoostsPersec for the instance
func (instance *Win32_PerfFormattedData_ClusBfltPerfProvider_ClusterStorageHybridDisksIOProfile) SetPropertyCacheWriteBoostsPersec(value uint64) (err error) {
	return instance.SetProperty("CacheWriteBoostsPersec", (value))
}

// GetCacheWriteBoostsPersec gets the value of CacheWriteBoostsPersec for the instance
func (instance *Win32_PerfFormattedData_ClusBfltPerfProvider_ClusterStorageHybridDisksIOProfile) GetPropertyCacheWriteBoostsPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("CacheWriteBoostsPersec")
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

// SetCacheWriteBoostsVeto sets the value of CacheWriteBoostsVeto for the instance
func (instance *Win32_PerfFormattedData_ClusBfltPerfProvider_ClusterStorageHybridDisksIOProfile) SetPropertyCacheWriteBoostsVeto(value uint64) (err error) {
	return instance.SetProperty("CacheWriteBoostsVeto", (value))
}

// GetCacheWriteBoostsVeto gets the value of CacheWriteBoostsVeto for the instance
func (instance *Win32_PerfFormattedData_ClusBfltPerfProvider_ClusterStorageHybridDisksIOProfile) GetPropertyCacheWriteBoostsVeto() (value uint64, err error) {
	retValue, err := instance.GetProperty("CacheWriteBoostsVeto")
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

// SetCacheWriteBoostsVetoPersec sets the value of CacheWriteBoostsVetoPersec for the instance
func (instance *Win32_PerfFormattedData_ClusBfltPerfProvider_ClusterStorageHybridDisksIOProfile) SetPropertyCacheWriteBoostsVetoPersec(value uint64) (err error) {
	return instance.SetProperty("CacheWriteBoostsVetoPersec", (value))
}

// GetCacheWriteBoostsVetoPersec gets the value of CacheWriteBoostsVetoPersec for the instance
func (instance *Win32_PerfFormattedData_ClusBfltPerfProvider_ClusterStorageHybridDisksIOProfile) GetPropertyCacheWriteBoostsVetoPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("CacheWriteBoostsVetoPersec")
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

// SetReads0K4K sets the value of Reads0K4K for the instance
func (instance *Win32_PerfFormattedData_ClusBfltPerfProvider_ClusterStorageHybridDisksIOProfile) SetPropertyReads0K4K(value uint64) (err error) {
	return instance.SetProperty("Reads0K4K", (value))
}

// GetReads0K4K gets the value of Reads0K4K for the instance
func (instance *Win32_PerfFormattedData_ClusBfltPerfProvider_ClusterStorageHybridDisksIOProfile) GetPropertyReads0K4K() (value uint64, err error) {
	retValue, err := instance.GetProperty("Reads0K4K")
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

// SetReads1024K2048K sets the value of Reads1024K2048K for the instance
func (instance *Win32_PerfFormattedData_ClusBfltPerfProvider_ClusterStorageHybridDisksIOProfile) SetPropertyReads1024K2048K(value uint64) (err error) {
	return instance.SetProperty("Reads1024K2048K", (value))
}

// GetReads1024K2048K gets the value of Reads1024K2048K for the instance
func (instance *Win32_PerfFormattedData_ClusBfltPerfProvider_ClusterStorageHybridDisksIOProfile) GetPropertyReads1024K2048K() (value uint64, err error) {
	retValue, err := instance.GetProperty("Reads1024K2048K")
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

// SetReads128K256K sets the value of Reads128K256K for the instance
func (instance *Win32_PerfFormattedData_ClusBfltPerfProvider_ClusterStorageHybridDisksIOProfile) SetPropertyReads128K256K(value uint64) (err error) {
	return instance.SetProperty("Reads128K256K", (value))
}

// GetReads128K256K gets the value of Reads128K256K for the instance
func (instance *Win32_PerfFormattedData_ClusBfltPerfProvider_ClusterStorageHybridDisksIOProfile) GetPropertyReads128K256K() (value uint64, err error) {
	retValue, err := instance.GetProperty("Reads128K256K")
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

// SetReads16K32K sets the value of Reads16K32K for the instance
func (instance *Win32_PerfFormattedData_ClusBfltPerfProvider_ClusterStorageHybridDisksIOProfile) SetPropertyReads16K32K(value uint64) (err error) {
	return instance.SetProperty("Reads16K32K", (value))
}

// GetReads16K32K gets the value of Reads16K32K for the instance
func (instance *Win32_PerfFormattedData_ClusBfltPerfProvider_ClusterStorageHybridDisksIOProfile) GetPropertyReads16K32K() (value uint64, err error) {
	retValue, err := instance.GetProperty("Reads16K32K")
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

// SetReads2048K4096K sets the value of Reads2048K4096K for the instance
func (instance *Win32_PerfFormattedData_ClusBfltPerfProvider_ClusterStorageHybridDisksIOProfile) SetPropertyReads2048K4096K(value uint64) (err error) {
	return instance.SetProperty("Reads2048K4096K", (value))
}

// GetReads2048K4096K gets the value of Reads2048K4096K for the instance
func (instance *Win32_PerfFormattedData_ClusBfltPerfProvider_ClusterStorageHybridDisksIOProfile) GetPropertyReads2048K4096K() (value uint64, err error) {
	retValue, err := instance.GetProperty("Reads2048K4096K")
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

// SetReads256K512K sets the value of Reads256K512K for the instance
func (instance *Win32_PerfFormattedData_ClusBfltPerfProvider_ClusterStorageHybridDisksIOProfile) SetPropertyReads256K512K(value uint64) (err error) {
	return instance.SetProperty("Reads256K512K", (value))
}

// GetReads256K512K gets the value of Reads256K512K for the instance
func (instance *Win32_PerfFormattedData_ClusBfltPerfProvider_ClusterStorageHybridDisksIOProfile) GetPropertyReads256K512K() (value uint64, err error) {
	retValue, err := instance.GetProperty("Reads256K512K")
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

// SetReads32K64K sets the value of Reads32K64K for the instance
func (instance *Win32_PerfFormattedData_ClusBfltPerfProvider_ClusterStorageHybridDisksIOProfile) SetPropertyReads32K64K(value uint64) (err error) {
	return instance.SetProperty("Reads32K64K", (value))
}

// GetReads32K64K gets the value of Reads32K64K for the instance
func (instance *Win32_PerfFormattedData_ClusBfltPerfProvider_ClusterStorageHybridDisksIOProfile) GetPropertyReads32K64K() (value uint64, err error) {
	retValue, err := instance.GetProperty("Reads32K64K")
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

// SetReads4096Koo sets the value of Reads4096Koo for the instance
func (instance *Win32_PerfFormattedData_ClusBfltPerfProvider_ClusterStorageHybridDisksIOProfile) SetPropertyReads4096Koo(value uint64) (err error) {
	return instance.SetProperty("Reads4096Koo", (value))
}

// GetReads4096Koo gets the value of Reads4096Koo for the instance
func (instance *Win32_PerfFormattedData_ClusBfltPerfProvider_ClusterStorageHybridDisksIOProfile) GetPropertyReads4096Koo() (value uint64, err error) {
	retValue, err := instance.GetProperty("Reads4096Koo")
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

// SetReads4K8K sets the value of Reads4K8K for the instance
func (instance *Win32_PerfFormattedData_ClusBfltPerfProvider_ClusterStorageHybridDisksIOProfile) SetPropertyReads4K8K(value uint64) (err error) {
	return instance.SetProperty("Reads4K8K", (value))
}

// GetReads4K8K gets the value of Reads4K8K for the instance
func (instance *Win32_PerfFormattedData_ClusBfltPerfProvider_ClusterStorageHybridDisksIOProfile) GetPropertyReads4K8K() (value uint64, err error) {
	retValue, err := instance.GetProperty("Reads4K8K")
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

// SetReads512K1024K sets the value of Reads512K1024K for the instance
func (instance *Win32_PerfFormattedData_ClusBfltPerfProvider_ClusterStorageHybridDisksIOProfile) SetPropertyReads512K1024K(value uint64) (err error) {
	return instance.SetProperty("Reads512K1024K", (value))
}

// GetReads512K1024K gets the value of Reads512K1024K for the instance
func (instance *Win32_PerfFormattedData_ClusBfltPerfProvider_ClusterStorageHybridDisksIOProfile) GetPropertyReads512K1024K() (value uint64, err error) {
	retValue, err := instance.GetProperty("Reads512K1024K")
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

// SetReads64K128K sets the value of Reads64K128K for the instance
func (instance *Win32_PerfFormattedData_ClusBfltPerfProvider_ClusterStorageHybridDisksIOProfile) SetPropertyReads64K128K(value uint64) (err error) {
	return instance.SetProperty("Reads64K128K", (value))
}

// GetReads64K128K gets the value of Reads64K128K for the instance
func (instance *Win32_PerfFormattedData_ClusBfltPerfProvider_ClusterStorageHybridDisksIOProfile) GetPropertyReads64K128K() (value uint64, err error) {
	retValue, err := instance.GetProperty("Reads64K128K")
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

// SetReads8K16K sets the value of Reads8K16K for the instance
func (instance *Win32_PerfFormattedData_ClusBfltPerfProvider_ClusterStorageHybridDisksIOProfile) SetPropertyReads8K16K(value uint64) (err error) {
	return instance.SetProperty("Reads8K16K", (value))
}

// GetReads8K16K gets the value of Reads8K16K for the instance
func (instance *Win32_PerfFormattedData_ClusBfltPerfProvider_ClusterStorageHybridDisksIOProfile) GetPropertyReads8K16K() (value uint64, err error) {
	retValue, err := instance.GetProperty("Reads8K16K")
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

// SetReadsnotaligned sets the value of Readsnotaligned for the instance
func (instance *Win32_PerfFormattedData_ClusBfltPerfProvider_ClusterStorageHybridDisksIOProfile) SetPropertyReadsnotaligned(value uint64) (err error) {
	return instance.SetProperty("Readsnotaligned", (value))
}

// GetReadsnotaligned gets the value of Readsnotaligned for the instance
func (instance *Win32_PerfFormattedData_ClusBfltPerfProvider_ClusterStorageHybridDisksIOProfile) GetPropertyReadsnotaligned() (value uint64, err error) {
	retValue, err := instance.GetProperty("Readsnotaligned")
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

// SetReadsPagingIO sets the value of ReadsPagingIO for the instance
func (instance *Win32_PerfFormattedData_ClusBfltPerfProvider_ClusterStorageHybridDisksIOProfile) SetPropertyReadsPagingIO(value uint64) (err error) {
	return instance.SetProperty("ReadsPagingIO", (value))
}

// GetReadsPagingIO gets the value of ReadsPagingIO for the instance
func (instance *Win32_PerfFormattedData_ClusBfltPerfProvider_ClusterStorageHybridDisksIOProfile) GetPropertyReadsPagingIO() (value uint64, err error) {
	retValue, err := instance.GetProperty("ReadsPagingIO")
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

// SetReadsPersec0K4K sets the value of ReadsPersec0K4K for the instance
func (instance *Win32_PerfFormattedData_ClusBfltPerfProvider_ClusterStorageHybridDisksIOProfile) SetPropertyReadsPersec0K4K(value uint64) (err error) {
	return instance.SetProperty("ReadsPersec0K4K", (value))
}

// GetReadsPersec0K4K gets the value of ReadsPersec0K4K for the instance
func (instance *Win32_PerfFormattedData_ClusBfltPerfProvider_ClusterStorageHybridDisksIOProfile) GetPropertyReadsPersec0K4K() (value uint64, err error) {
	retValue, err := instance.GetProperty("ReadsPersec0K4K")
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

// SetReadsPersec1024K2048K sets the value of ReadsPersec1024K2048K for the instance
func (instance *Win32_PerfFormattedData_ClusBfltPerfProvider_ClusterStorageHybridDisksIOProfile) SetPropertyReadsPersec1024K2048K(value uint64) (err error) {
	return instance.SetProperty("ReadsPersec1024K2048K", (value))
}

// GetReadsPersec1024K2048K gets the value of ReadsPersec1024K2048K for the instance
func (instance *Win32_PerfFormattedData_ClusBfltPerfProvider_ClusterStorageHybridDisksIOProfile) GetPropertyReadsPersec1024K2048K() (value uint64, err error) {
	retValue, err := instance.GetProperty("ReadsPersec1024K2048K")
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

// SetReadsPersec128K256K sets the value of ReadsPersec128K256K for the instance
func (instance *Win32_PerfFormattedData_ClusBfltPerfProvider_ClusterStorageHybridDisksIOProfile) SetPropertyReadsPersec128K256K(value uint64) (err error) {
	return instance.SetProperty("ReadsPersec128K256K", (value))
}

// GetReadsPersec128K256K gets the value of ReadsPersec128K256K for the instance
func (instance *Win32_PerfFormattedData_ClusBfltPerfProvider_ClusterStorageHybridDisksIOProfile) GetPropertyReadsPersec128K256K() (value uint64, err error) {
	retValue, err := instance.GetProperty("ReadsPersec128K256K")
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

// SetReadsPersec16K32K sets the value of ReadsPersec16K32K for the instance
func (instance *Win32_PerfFormattedData_ClusBfltPerfProvider_ClusterStorageHybridDisksIOProfile) SetPropertyReadsPersec16K32K(value uint64) (err error) {
	return instance.SetProperty("ReadsPersec16K32K", (value))
}

// GetReadsPersec16K32K gets the value of ReadsPersec16K32K for the instance
func (instance *Win32_PerfFormattedData_ClusBfltPerfProvider_ClusterStorageHybridDisksIOProfile) GetPropertyReadsPersec16K32K() (value uint64, err error) {
	retValue, err := instance.GetProperty("ReadsPersec16K32K")
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

// SetReadsPersec2048K4096K sets the value of ReadsPersec2048K4096K for the instance
func (instance *Win32_PerfFormattedData_ClusBfltPerfProvider_ClusterStorageHybridDisksIOProfile) SetPropertyReadsPersec2048K4096K(value uint64) (err error) {
	return instance.SetProperty("ReadsPersec2048K4096K", (value))
}

// GetReadsPersec2048K4096K gets the value of ReadsPersec2048K4096K for the instance
func (instance *Win32_PerfFormattedData_ClusBfltPerfProvider_ClusterStorageHybridDisksIOProfile) GetPropertyReadsPersec2048K4096K() (value uint64, err error) {
	retValue, err := instance.GetProperty("ReadsPersec2048K4096K")
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

// SetReadsPersec256K512K sets the value of ReadsPersec256K512K for the instance
func (instance *Win32_PerfFormattedData_ClusBfltPerfProvider_ClusterStorageHybridDisksIOProfile) SetPropertyReadsPersec256K512K(value uint64) (err error) {
	return instance.SetProperty("ReadsPersec256K512K", (value))
}

// GetReadsPersec256K512K gets the value of ReadsPersec256K512K for the instance
func (instance *Win32_PerfFormattedData_ClusBfltPerfProvider_ClusterStorageHybridDisksIOProfile) GetPropertyReadsPersec256K512K() (value uint64, err error) {
	retValue, err := instance.GetProperty("ReadsPersec256K512K")
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

// SetReadsPersec32K64K sets the value of ReadsPersec32K64K for the instance
func (instance *Win32_PerfFormattedData_ClusBfltPerfProvider_ClusterStorageHybridDisksIOProfile) SetPropertyReadsPersec32K64K(value uint64) (err error) {
	return instance.SetProperty("ReadsPersec32K64K", (value))
}

// GetReadsPersec32K64K gets the value of ReadsPersec32K64K for the instance
func (instance *Win32_PerfFormattedData_ClusBfltPerfProvider_ClusterStorageHybridDisksIOProfile) GetPropertyReadsPersec32K64K() (value uint64, err error) {
	retValue, err := instance.GetProperty("ReadsPersec32K64K")
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

// SetReadsPersec4096Koo sets the value of ReadsPersec4096Koo for the instance
func (instance *Win32_PerfFormattedData_ClusBfltPerfProvider_ClusterStorageHybridDisksIOProfile) SetPropertyReadsPersec4096Koo(value uint64) (err error) {
	return instance.SetProperty("ReadsPersec4096Koo", (value))
}

// GetReadsPersec4096Koo gets the value of ReadsPersec4096Koo for the instance
func (instance *Win32_PerfFormattedData_ClusBfltPerfProvider_ClusterStorageHybridDisksIOProfile) GetPropertyReadsPersec4096Koo() (value uint64, err error) {
	retValue, err := instance.GetProperty("ReadsPersec4096Koo")
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

// SetReadsPersec4K8K sets the value of ReadsPersec4K8K for the instance
func (instance *Win32_PerfFormattedData_ClusBfltPerfProvider_ClusterStorageHybridDisksIOProfile) SetPropertyReadsPersec4K8K(value uint64) (err error) {
	return instance.SetProperty("ReadsPersec4K8K", (value))
}

// GetReadsPersec4K8K gets the value of ReadsPersec4K8K for the instance
func (instance *Win32_PerfFormattedData_ClusBfltPerfProvider_ClusterStorageHybridDisksIOProfile) GetPropertyReadsPersec4K8K() (value uint64, err error) {
	retValue, err := instance.GetProperty("ReadsPersec4K8K")
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

// SetReadsPersec512K1024K sets the value of ReadsPersec512K1024K for the instance
func (instance *Win32_PerfFormattedData_ClusBfltPerfProvider_ClusterStorageHybridDisksIOProfile) SetPropertyReadsPersec512K1024K(value uint64) (err error) {
	return instance.SetProperty("ReadsPersec512K1024K", (value))
}

// GetReadsPersec512K1024K gets the value of ReadsPersec512K1024K for the instance
func (instance *Win32_PerfFormattedData_ClusBfltPerfProvider_ClusterStorageHybridDisksIOProfile) GetPropertyReadsPersec512K1024K() (value uint64, err error) {
	retValue, err := instance.GetProperty("ReadsPersec512K1024K")
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

// SetReadsPersec64K128K sets the value of ReadsPersec64K128K for the instance
func (instance *Win32_PerfFormattedData_ClusBfltPerfProvider_ClusterStorageHybridDisksIOProfile) SetPropertyReadsPersec64K128K(value uint64) (err error) {
	return instance.SetProperty("ReadsPersec64K128K", (value))
}

// GetReadsPersec64K128K gets the value of ReadsPersec64K128K for the instance
func (instance *Win32_PerfFormattedData_ClusBfltPerfProvider_ClusterStorageHybridDisksIOProfile) GetPropertyReadsPersec64K128K() (value uint64, err error) {
	retValue, err := instance.GetProperty("ReadsPersec64K128K")
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

// SetReadsPersec8K16K sets the value of ReadsPersec8K16K for the instance
func (instance *Win32_PerfFormattedData_ClusBfltPerfProvider_ClusterStorageHybridDisksIOProfile) SetPropertyReadsPersec8K16K(value uint64) (err error) {
	return instance.SetProperty("ReadsPersec8K16K", (value))
}

// GetReadsPersec8K16K gets the value of ReadsPersec8K16K for the instance
func (instance *Win32_PerfFormattedData_ClusBfltPerfProvider_ClusterStorageHybridDisksIOProfile) GetPropertyReadsPersec8K16K() (value uint64, err error) {
	retValue, err := instance.GetProperty("ReadsPersec8K16K")
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

// SetReadsPersecnotaligned sets the value of ReadsPersecnotaligned for the instance
func (instance *Win32_PerfFormattedData_ClusBfltPerfProvider_ClusterStorageHybridDisksIOProfile) SetPropertyReadsPersecnotaligned(value uint64) (err error) {
	return instance.SetProperty("ReadsPersecnotaligned", (value))
}

// GetReadsPersecnotaligned gets the value of ReadsPersecnotaligned for the instance
func (instance *Win32_PerfFormattedData_ClusBfltPerfProvider_ClusterStorageHybridDisksIOProfile) GetPropertyReadsPersecnotaligned() (value uint64, err error) {
	retValue, err := instance.GetProperty("ReadsPersecnotaligned")
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

// SetReadsPersecPagingIO sets the value of ReadsPersecPagingIO for the instance
func (instance *Win32_PerfFormattedData_ClusBfltPerfProvider_ClusterStorageHybridDisksIOProfile) SetPropertyReadsPersecPagingIO(value uint64) (err error) {
	return instance.SetProperty("ReadsPersecPagingIO", (value))
}

// GetReadsPersecPagingIO gets the value of ReadsPersecPagingIO for the instance
func (instance *Win32_PerfFormattedData_ClusBfltPerfProvider_ClusterStorageHybridDisksIOProfile) GetPropertyReadsPersecPagingIO() (value uint64, err error) {
	retValue, err := instance.GetProperty("ReadsPersecPagingIO")
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

// SetReadsPersecTotal sets the value of ReadsPersecTotal for the instance
func (instance *Win32_PerfFormattedData_ClusBfltPerfProvider_ClusterStorageHybridDisksIOProfile) SetPropertyReadsPersecTotal(value uint64) (err error) {
	return instance.SetProperty("ReadsPersecTotal", (value))
}

// GetReadsPersecTotal gets the value of ReadsPersecTotal for the instance
func (instance *Win32_PerfFormattedData_ClusBfltPerfProvider_ClusterStorageHybridDisksIOProfile) GetPropertyReadsPersecTotal() (value uint64, err error) {
	retValue, err := instance.GetProperty("ReadsPersecTotal")
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

// SetReadsTotal sets the value of ReadsTotal for the instance
func (instance *Win32_PerfFormattedData_ClusBfltPerfProvider_ClusterStorageHybridDisksIOProfile) SetPropertyReadsTotal(value uint64) (err error) {
	return instance.SetProperty("ReadsTotal", (value))
}

// GetReadsTotal gets the value of ReadsTotal for the instance
func (instance *Win32_PerfFormattedData_ClusBfltPerfProvider_ClusterStorageHybridDisksIOProfile) GetPropertyReadsTotal() (value uint64, err error) {
	retValue, err := instance.GetProperty("ReadsTotal")
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

// SetWrites0K4K sets the value of Writes0K4K for the instance
func (instance *Win32_PerfFormattedData_ClusBfltPerfProvider_ClusterStorageHybridDisksIOProfile) SetPropertyWrites0K4K(value uint64) (err error) {
	return instance.SetProperty("Writes0K4K", (value))
}

// GetWrites0K4K gets the value of Writes0K4K for the instance
func (instance *Win32_PerfFormattedData_ClusBfltPerfProvider_ClusterStorageHybridDisksIOProfile) GetPropertyWrites0K4K() (value uint64, err error) {
	retValue, err := instance.GetProperty("Writes0K4K")
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

// SetWrites1024K2048K sets the value of Writes1024K2048K for the instance
func (instance *Win32_PerfFormattedData_ClusBfltPerfProvider_ClusterStorageHybridDisksIOProfile) SetPropertyWrites1024K2048K(value uint64) (err error) {
	return instance.SetProperty("Writes1024K2048K", (value))
}

// GetWrites1024K2048K gets the value of Writes1024K2048K for the instance
func (instance *Win32_PerfFormattedData_ClusBfltPerfProvider_ClusterStorageHybridDisksIOProfile) GetPropertyWrites1024K2048K() (value uint64, err error) {
	retValue, err := instance.GetProperty("Writes1024K2048K")
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

// SetWrites128K256K sets the value of Writes128K256K for the instance
func (instance *Win32_PerfFormattedData_ClusBfltPerfProvider_ClusterStorageHybridDisksIOProfile) SetPropertyWrites128K256K(value uint64) (err error) {
	return instance.SetProperty("Writes128K256K", (value))
}

// GetWrites128K256K gets the value of Writes128K256K for the instance
func (instance *Win32_PerfFormattedData_ClusBfltPerfProvider_ClusterStorageHybridDisksIOProfile) GetPropertyWrites128K256K() (value uint64, err error) {
	retValue, err := instance.GetProperty("Writes128K256K")
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

// SetWrites16K32K sets the value of Writes16K32K for the instance
func (instance *Win32_PerfFormattedData_ClusBfltPerfProvider_ClusterStorageHybridDisksIOProfile) SetPropertyWrites16K32K(value uint64) (err error) {
	return instance.SetProperty("Writes16K32K", (value))
}

// GetWrites16K32K gets the value of Writes16K32K for the instance
func (instance *Win32_PerfFormattedData_ClusBfltPerfProvider_ClusterStorageHybridDisksIOProfile) GetPropertyWrites16K32K() (value uint64, err error) {
	retValue, err := instance.GetProperty("Writes16K32K")
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

// SetWrites2048K4096K sets the value of Writes2048K4096K for the instance
func (instance *Win32_PerfFormattedData_ClusBfltPerfProvider_ClusterStorageHybridDisksIOProfile) SetPropertyWrites2048K4096K(value uint64) (err error) {
	return instance.SetProperty("Writes2048K4096K", (value))
}

// GetWrites2048K4096K gets the value of Writes2048K4096K for the instance
func (instance *Win32_PerfFormattedData_ClusBfltPerfProvider_ClusterStorageHybridDisksIOProfile) GetPropertyWrites2048K4096K() (value uint64, err error) {
	retValue, err := instance.GetProperty("Writes2048K4096K")
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

// SetWrites256K512K sets the value of Writes256K512K for the instance
func (instance *Win32_PerfFormattedData_ClusBfltPerfProvider_ClusterStorageHybridDisksIOProfile) SetPropertyWrites256K512K(value uint64) (err error) {
	return instance.SetProperty("Writes256K512K", (value))
}

// GetWrites256K512K gets the value of Writes256K512K for the instance
func (instance *Win32_PerfFormattedData_ClusBfltPerfProvider_ClusterStorageHybridDisksIOProfile) GetPropertyWrites256K512K() (value uint64, err error) {
	retValue, err := instance.GetProperty("Writes256K512K")
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

// SetWrites32K64K sets the value of Writes32K64K for the instance
func (instance *Win32_PerfFormattedData_ClusBfltPerfProvider_ClusterStorageHybridDisksIOProfile) SetPropertyWrites32K64K(value uint64) (err error) {
	return instance.SetProperty("Writes32K64K", (value))
}

// GetWrites32K64K gets the value of Writes32K64K for the instance
func (instance *Win32_PerfFormattedData_ClusBfltPerfProvider_ClusterStorageHybridDisksIOProfile) GetPropertyWrites32K64K() (value uint64, err error) {
	retValue, err := instance.GetProperty("Writes32K64K")
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

// SetWrites4096Koo sets the value of Writes4096Koo for the instance
func (instance *Win32_PerfFormattedData_ClusBfltPerfProvider_ClusterStorageHybridDisksIOProfile) SetPropertyWrites4096Koo(value uint64) (err error) {
	return instance.SetProperty("Writes4096Koo", (value))
}

// GetWrites4096Koo gets the value of Writes4096Koo for the instance
func (instance *Win32_PerfFormattedData_ClusBfltPerfProvider_ClusterStorageHybridDisksIOProfile) GetPropertyWrites4096Koo() (value uint64, err error) {
	retValue, err := instance.GetProperty("Writes4096Koo")
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

// SetWrites4K8K sets the value of Writes4K8K for the instance
func (instance *Win32_PerfFormattedData_ClusBfltPerfProvider_ClusterStorageHybridDisksIOProfile) SetPropertyWrites4K8K(value uint64) (err error) {
	return instance.SetProperty("Writes4K8K", (value))
}

// GetWrites4K8K gets the value of Writes4K8K for the instance
func (instance *Win32_PerfFormattedData_ClusBfltPerfProvider_ClusterStorageHybridDisksIOProfile) GetPropertyWrites4K8K() (value uint64, err error) {
	retValue, err := instance.GetProperty("Writes4K8K")
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

// SetWrites512K1024K sets the value of Writes512K1024K for the instance
func (instance *Win32_PerfFormattedData_ClusBfltPerfProvider_ClusterStorageHybridDisksIOProfile) SetPropertyWrites512K1024K(value uint64) (err error) {
	return instance.SetProperty("Writes512K1024K", (value))
}

// GetWrites512K1024K gets the value of Writes512K1024K for the instance
func (instance *Win32_PerfFormattedData_ClusBfltPerfProvider_ClusterStorageHybridDisksIOProfile) GetPropertyWrites512K1024K() (value uint64, err error) {
	retValue, err := instance.GetProperty("Writes512K1024K")
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

// SetWrites64K128K sets the value of Writes64K128K for the instance
func (instance *Win32_PerfFormattedData_ClusBfltPerfProvider_ClusterStorageHybridDisksIOProfile) SetPropertyWrites64K128K(value uint64) (err error) {
	return instance.SetProperty("Writes64K128K", (value))
}

// GetWrites64K128K gets the value of Writes64K128K for the instance
func (instance *Win32_PerfFormattedData_ClusBfltPerfProvider_ClusterStorageHybridDisksIOProfile) GetPropertyWrites64K128K() (value uint64, err error) {
	retValue, err := instance.GetProperty("Writes64K128K")
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

// SetWrites8K16K sets the value of Writes8K16K for the instance
func (instance *Win32_PerfFormattedData_ClusBfltPerfProvider_ClusterStorageHybridDisksIOProfile) SetPropertyWrites8K16K(value uint64) (err error) {
	return instance.SetProperty("Writes8K16K", (value))
}

// GetWrites8K16K gets the value of Writes8K16K for the instance
func (instance *Win32_PerfFormattedData_ClusBfltPerfProvider_ClusterStorageHybridDisksIOProfile) GetPropertyWrites8K16K() (value uint64, err error) {
	retValue, err := instance.GetProperty("Writes8K16K")
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

// SetWritesnotaligned sets the value of Writesnotaligned for the instance
func (instance *Win32_PerfFormattedData_ClusBfltPerfProvider_ClusterStorageHybridDisksIOProfile) SetPropertyWritesnotaligned(value uint64) (err error) {
	return instance.SetProperty("Writesnotaligned", (value))
}

// GetWritesnotaligned gets the value of Writesnotaligned for the instance
func (instance *Win32_PerfFormattedData_ClusBfltPerfProvider_ClusterStorageHybridDisksIOProfile) GetPropertyWritesnotaligned() (value uint64, err error) {
	retValue, err := instance.GetProperty("Writesnotaligned")
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

// SetWritesPagingIO sets the value of WritesPagingIO for the instance
func (instance *Win32_PerfFormattedData_ClusBfltPerfProvider_ClusterStorageHybridDisksIOProfile) SetPropertyWritesPagingIO(value uint64) (err error) {
	return instance.SetProperty("WritesPagingIO", (value))
}

// GetWritesPagingIO gets the value of WritesPagingIO for the instance
func (instance *Win32_PerfFormattedData_ClusBfltPerfProvider_ClusterStorageHybridDisksIOProfile) GetPropertyWritesPagingIO() (value uint64, err error) {
	retValue, err := instance.GetProperty("WritesPagingIO")
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

// SetWritesPersec0K4K sets the value of WritesPersec0K4K for the instance
func (instance *Win32_PerfFormattedData_ClusBfltPerfProvider_ClusterStorageHybridDisksIOProfile) SetPropertyWritesPersec0K4K(value uint64) (err error) {
	return instance.SetProperty("WritesPersec0K4K", (value))
}

// GetWritesPersec0K4K gets the value of WritesPersec0K4K for the instance
func (instance *Win32_PerfFormattedData_ClusBfltPerfProvider_ClusterStorageHybridDisksIOProfile) GetPropertyWritesPersec0K4K() (value uint64, err error) {
	retValue, err := instance.GetProperty("WritesPersec0K4K")
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

// SetWritesPersec1024K2048K sets the value of WritesPersec1024K2048K for the instance
func (instance *Win32_PerfFormattedData_ClusBfltPerfProvider_ClusterStorageHybridDisksIOProfile) SetPropertyWritesPersec1024K2048K(value uint64) (err error) {
	return instance.SetProperty("WritesPersec1024K2048K", (value))
}

// GetWritesPersec1024K2048K gets the value of WritesPersec1024K2048K for the instance
func (instance *Win32_PerfFormattedData_ClusBfltPerfProvider_ClusterStorageHybridDisksIOProfile) GetPropertyWritesPersec1024K2048K() (value uint64, err error) {
	retValue, err := instance.GetProperty("WritesPersec1024K2048K")
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

// SetWritesPersec128K256K sets the value of WritesPersec128K256K for the instance
func (instance *Win32_PerfFormattedData_ClusBfltPerfProvider_ClusterStorageHybridDisksIOProfile) SetPropertyWritesPersec128K256K(value uint64) (err error) {
	return instance.SetProperty("WritesPersec128K256K", (value))
}

// GetWritesPersec128K256K gets the value of WritesPersec128K256K for the instance
func (instance *Win32_PerfFormattedData_ClusBfltPerfProvider_ClusterStorageHybridDisksIOProfile) GetPropertyWritesPersec128K256K() (value uint64, err error) {
	retValue, err := instance.GetProperty("WritesPersec128K256K")
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

// SetWritesPersec16K32K sets the value of WritesPersec16K32K for the instance
func (instance *Win32_PerfFormattedData_ClusBfltPerfProvider_ClusterStorageHybridDisksIOProfile) SetPropertyWritesPersec16K32K(value uint64) (err error) {
	return instance.SetProperty("WritesPersec16K32K", (value))
}

// GetWritesPersec16K32K gets the value of WritesPersec16K32K for the instance
func (instance *Win32_PerfFormattedData_ClusBfltPerfProvider_ClusterStorageHybridDisksIOProfile) GetPropertyWritesPersec16K32K() (value uint64, err error) {
	retValue, err := instance.GetProperty("WritesPersec16K32K")
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

// SetWritesPersec2048K4096K sets the value of WritesPersec2048K4096K for the instance
func (instance *Win32_PerfFormattedData_ClusBfltPerfProvider_ClusterStorageHybridDisksIOProfile) SetPropertyWritesPersec2048K4096K(value uint64) (err error) {
	return instance.SetProperty("WritesPersec2048K4096K", (value))
}

// GetWritesPersec2048K4096K gets the value of WritesPersec2048K4096K for the instance
func (instance *Win32_PerfFormattedData_ClusBfltPerfProvider_ClusterStorageHybridDisksIOProfile) GetPropertyWritesPersec2048K4096K() (value uint64, err error) {
	retValue, err := instance.GetProperty("WritesPersec2048K4096K")
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

// SetWritesPersec256K512K sets the value of WritesPersec256K512K for the instance
func (instance *Win32_PerfFormattedData_ClusBfltPerfProvider_ClusterStorageHybridDisksIOProfile) SetPropertyWritesPersec256K512K(value uint64) (err error) {
	return instance.SetProperty("WritesPersec256K512K", (value))
}

// GetWritesPersec256K512K gets the value of WritesPersec256K512K for the instance
func (instance *Win32_PerfFormattedData_ClusBfltPerfProvider_ClusterStorageHybridDisksIOProfile) GetPropertyWritesPersec256K512K() (value uint64, err error) {
	retValue, err := instance.GetProperty("WritesPersec256K512K")
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

// SetWritesPersec32K64K sets the value of WritesPersec32K64K for the instance
func (instance *Win32_PerfFormattedData_ClusBfltPerfProvider_ClusterStorageHybridDisksIOProfile) SetPropertyWritesPersec32K64K(value uint64) (err error) {
	return instance.SetProperty("WritesPersec32K64K", (value))
}

// GetWritesPersec32K64K gets the value of WritesPersec32K64K for the instance
func (instance *Win32_PerfFormattedData_ClusBfltPerfProvider_ClusterStorageHybridDisksIOProfile) GetPropertyWritesPersec32K64K() (value uint64, err error) {
	retValue, err := instance.GetProperty("WritesPersec32K64K")
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

// SetWritesPersec4096Koo sets the value of WritesPersec4096Koo for the instance
func (instance *Win32_PerfFormattedData_ClusBfltPerfProvider_ClusterStorageHybridDisksIOProfile) SetPropertyWritesPersec4096Koo(value uint64) (err error) {
	return instance.SetProperty("WritesPersec4096Koo", (value))
}

// GetWritesPersec4096Koo gets the value of WritesPersec4096Koo for the instance
func (instance *Win32_PerfFormattedData_ClusBfltPerfProvider_ClusterStorageHybridDisksIOProfile) GetPropertyWritesPersec4096Koo() (value uint64, err error) {
	retValue, err := instance.GetProperty("WritesPersec4096Koo")
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

// SetWritesPersec4K8K sets the value of WritesPersec4K8K for the instance
func (instance *Win32_PerfFormattedData_ClusBfltPerfProvider_ClusterStorageHybridDisksIOProfile) SetPropertyWritesPersec4K8K(value uint64) (err error) {
	return instance.SetProperty("WritesPersec4K8K", (value))
}

// GetWritesPersec4K8K gets the value of WritesPersec4K8K for the instance
func (instance *Win32_PerfFormattedData_ClusBfltPerfProvider_ClusterStorageHybridDisksIOProfile) GetPropertyWritesPersec4K8K() (value uint64, err error) {
	retValue, err := instance.GetProperty("WritesPersec4K8K")
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

// SetWritesPersec512K1024K sets the value of WritesPersec512K1024K for the instance
func (instance *Win32_PerfFormattedData_ClusBfltPerfProvider_ClusterStorageHybridDisksIOProfile) SetPropertyWritesPersec512K1024K(value uint64) (err error) {
	return instance.SetProperty("WritesPersec512K1024K", (value))
}

// GetWritesPersec512K1024K gets the value of WritesPersec512K1024K for the instance
func (instance *Win32_PerfFormattedData_ClusBfltPerfProvider_ClusterStorageHybridDisksIOProfile) GetPropertyWritesPersec512K1024K() (value uint64, err error) {
	retValue, err := instance.GetProperty("WritesPersec512K1024K")
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

// SetWritesPersec64K128K sets the value of WritesPersec64K128K for the instance
func (instance *Win32_PerfFormattedData_ClusBfltPerfProvider_ClusterStorageHybridDisksIOProfile) SetPropertyWritesPersec64K128K(value uint64) (err error) {
	return instance.SetProperty("WritesPersec64K128K", (value))
}

// GetWritesPersec64K128K gets the value of WritesPersec64K128K for the instance
func (instance *Win32_PerfFormattedData_ClusBfltPerfProvider_ClusterStorageHybridDisksIOProfile) GetPropertyWritesPersec64K128K() (value uint64, err error) {
	retValue, err := instance.GetProperty("WritesPersec64K128K")
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

// SetWritesPersec8K16K sets the value of WritesPersec8K16K for the instance
func (instance *Win32_PerfFormattedData_ClusBfltPerfProvider_ClusterStorageHybridDisksIOProfile) SetPropertyWritesPersec8K16K(value uint64) (err error) {
	return instance.SetProperty("WritesPersec8K16K", (value))
}

// GetWritesPersec8K16K gets the value of WritesPersec8K16K for the instance
func (instance *Win32_PerfFormattedData_ClusBfltPerfProvider_ClusterStorageHybridDisksIOProfile) GetPropertyWritesPersec8K16K() (value uint64, err error) {
	retValue, err := instance.GetProperty("WritesPersec8K16K")
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

// SetWritesPersecnotaligned sets the value of WritesPersecnotaligned for the instance
func (instance *Win32_PerfFormattedData_ClusBfltPerfProvider_ClusterStorageHybridDisksIOProfile) SetPropertyWritesPersecnotaligned(value uint64) (err error) {
	return instance.SetProperty("WritesPersecnotaligned", (value))
}

// GetWritesPersecnotaligned gets the value of WritesPersecnotaligned for the instance
func (instance *Win32_PerfFormattedData_ClusBfltPerfProvider_ClusterStorageHybridDisksIOProfile) GetPropertyWritesPersecnotaligned() (value uint64, err error) {
	retValue, err := instance.GetProperty("WritesPersecnotaligned")
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

// SetWritesPersecPagingIO sets the value of WritesPersecPagingIO for the instance
func (instance *Win32_PerfFormattedData_ClusBfltPerfProvider_ClusterStorageHybridDisksIOProfile) SetPropertyWritesPersecPagingIO(value uint64) (err error) {
	return instance.SetProperty("WritesPersecPagingIO", (value))
}

// GetWritesPersecPagingIO gets the value of WritesPersecPagingIO for the instance
func (instance *Win32_PerfFormattedData_ClusBfltPerfProvider_ClusterStorageHybridDisksIOProfile) GetPropertyWritesPersecPagingIO() (value uint64, err error) {
	retValue, err := instance.GetProperty("WritesPersecPagingIO")
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

// SetWritesPersecTotal sets the value of WritesPersecTotal for the instance
func (instance *Win32_PerfFormattedData_ClusBfltPerfProvider_ClusterStorageHybridDisksIOProfile) SetPropertyWritesPersecTotal(value uint64) (err error) {
	return instance.SetProperty("WritesPersecTotal", (value))
}

// GetWritesPersecTotal gets the value of WritesPersecTotal for the instance
func (instance *Win32_PerfFormattedData_ClusBfltPerfProvider_ClusterStorageHybridDisksIOProfile) GetPropertyWritesPersecTotal() (value uint64, err error) {
	retValue, err := instance.GetProperty("WritesPersecTotal")
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

// SetWritesTotal sets the value of WritesTotal for the instance
func (instance *Win32_PerfFormattedData_ClusBfltPerfProvider_ClusterStorageHybridDisksIOProfile) SetPropertyWritesTotal(value uint64) (err error) {
	return instance.SetProperty("WritesTotal", (value))
}

// GetWritesTotal gets the value of WritesTotal for the instance
func (instance *Win32_PerfFormattedData_ClusBfltPerfProvider_ClusterStorageHybridDisksIOProfile) GetPropertyWritesTotal() (value uint64, err error) {
	retValue, err := instance.GetProperty("WritesTotal")
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
