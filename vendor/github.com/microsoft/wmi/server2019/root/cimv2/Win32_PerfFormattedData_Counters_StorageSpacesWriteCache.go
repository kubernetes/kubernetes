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

// Win32_PerfFormattedData_Counters_StorageSpacesWriteCache struct
type Win32_PerfFormattedData_Counters_StorageSpacesWriteCache struct {
	*Win32_PerfFormattedData

	//
	CacheAdvances uint32

	//
	CacheCheckpoints uint32

	//
	CacheDataBytes uint64

	//
	CacheDataPercent uint64

	//
	CacheDestagesCurrent uint32

	//
	CacheReclaimableBytes uint64

	//
	CacheReclaimablePercent uint64

	//
	CacheSize uint64

	//
	CacheUsedBytes uint64

	//
	CacheUsedPercent uint64

	//
	EvictCacheBytesPersec uint64

	//
	EvictCacheDestagedBytesPersec uint64

	//
	EvictCacheDestagedPercent uint64

	//
	EvictCacheOverwriteBytesPersec uint64

	//
	EvictCacheOverwritePercent uint64

	//
	ReadBypassBytesPersec uint64

	//
	ReadBypassPercent uint64

	//
	ReadCacheBytesPersec uint64

	//
	ReadCachePercent uint64

	//
	WriteBypassBytesPersec uint64

	//
	WriteBypassPercent uint64

	//
	WriteCacheBytesPersec uint64

	//
	WriteCacheOverlapBytesPersec uint64

	//
	WriteCacheOverlapPercent uint64

	//
	WriteCachePercent uint64

	//
	WriteCacheUnalignedBytesPersec uint64

	//
	WriteCacheUnalignedPercent uint64

	//
	WriteCacheUntrimmedBytesPersec uint64

	//
	WriteCacheUntrimmedPercent uint64
}

func NewWin32_PerfFormattedData_Counters_StorageSpacesWriteCacheEx1(instance *cim.WmiInstance) (newInstance *Win32_PerfFormattedData_Counters_StorageSpacesWriteCache, err error) {
	tmp, err := NewWin32_PerfFormattedDataEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfFormattedData_Counters_StorageSpacesWriteCache{
		Win32_PerfFormattedData: tmp,
	}
	return
}

func NewWin32_PerfFormattedData_Counters_StorageSpacesWriteCacheEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_PerfFormattedData_Counters_StorageSpacesWriteCache, err error) {
	tmp, err := NewWin32_PerfFormattedDataEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfFormattedData_Counters_StorageSpacesWriteCache{
		Win32_PerfFormattedData: tmp,
	}
	return
}

// SetCacheAdvances sets the value of CacheAdvances for the instance
func (instance *Win32_PerfFormattedData_Counters_StorageSpacesWriteCache) SetPropertyCacheAdvances(value uint32) (err error) {
	return instance.SetProperty("CacheAdvances", (value))
}

// GetCacheAdvances gets the value of CacheAdvances for the instance
func (instance *Win32_PerfFormattedData_Counters_StorageSpacesWriteCache) GetPropertyCacheAdvances() (value uint32, err error) {
	retValue, err := instance.GetProperty("CacheAdvances")
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

// SetCacheCheckpoints sets the value of CacheCheckpoints for the instance
func (instance *Win32_PerfFormattedData_Counters_StorageSpacesWriteCache) SetPropertyCacheCheckpoints(value uint32) (err error) {
	return instance.SetProperty("CacheCheckpoints", (value))
}

// GetCacheCheckpoints gets the value of CacheCheckpoints for the instance
func (instance *Win32_PerfFormattedData_Counters_StorageSpacesWriteCache) GetPropertyCacheCheckpoints() (value uint32, err error) {
	retValue, err := instance.GetProperty("CacheCheckpoints")
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

// SetCacheDataBytes sets the value of CacheDataBytes for the instance
func (instance *Win32_PerfFormattedData_Counters_StorageSpacesWriteCache) SetPropertyCacheDataBytes(value uint64) (err error) {
	return instance.SetProperty("CacheDataBytes", (value))
}

// GetCacheDataBytes gets the value of CacheDataBytes for the instance
func (instance *Win32_PerfFormattedData_Counters_StorageSpacesWriteCache) GetPropertyCacheDataBytes() (value uint64, err error) {
	retValue, err := instance.GetProperty("CacheDataBytes")
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

// SetCacheDataPercent sets the value of CacheDataPercent for the instance
func (instance *Win32_PerfFormattedData_Counters_StorageSpacesWriteCache) SetPropertyCacheDataPercent(value uint64) (err error) {
	return instance.SetProperty("CacheDataPercent", (value))
}

// GetCacheDataPercent gets the value of CacheDataPercent for the instance
func (instance *Win32_PerfFormattedData_Counters_StorageSpacesWriteCache) GetPropertyCacheDataPercent() (value uint64, err error) {
	retValue, err := instance.GetProperty("CacheDataPercent")
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

// SetCacheDestagesCurrent sets the value of CacheDestagesCurrent for the instance
func (instance *Win32_PerfFormattedData_Counters_StorageSpacesWriteCache) SetPropertyCacheDestagesCurrent(value uint32) (err error) {
	return instance.SetProperty("CacheDestagesCurrent", (value))
}

// GetCacheDestagesCurrent gets the value of CacheDestagesCurrent for the instance
func (instance *Win32_PerfFormattedData_Counters_StorageSpacesWriteCache) GetPropertyCacheDestagesCurrent() (value uint32, err error) {
	retValue, err := instance.GetProperty("CacheDestagesCurrent")
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

// SetCacheReclaimableBytes sets the value of CacheReclaimableBytes for the instance
func (instance *Win32_PerfFormattedData_Counters_StorageSpacesWriteCache) SetPropertyCacheReclaimableBytes(value uint64) (err error) {
	return instance.SetProperty("CacheReclaimableBytes", (value))
}

// GetCacheReclaimableBytes gets the value of CacheReclaimableBytes for the instance
func (instance *Win32_PerfFormattedData_Counters_StorageSpacesWriteCache) GetPropertyCacheReclaimableBytes() (value uint64, err error) {
	retValue, err := instance.GetProperty("CacheReclaimableBytes")
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

// SetCacheReclaimablePercent sets the value of CacheReclaimablePercent for the instance
func (instance *Win32_PerfFormattedData_Counters_StorageSpacesWriteCache) SetPropertyCacheReclaimablePercent(value uint64) (err error) {
	return instance.SetProperty("CacheReclaimablePercent", (value))
}

// GetCacheReclaimablePercent gets the value of CacheReclaimablePercent for the instance
func (instance *Win32_PerfFormattedData_Counters_StorageSpacesWriteCache) GetPropertyCacheReclaimablePercent() (value uint64, err error) {
	retValue, err := instance.GetProperty("CacheReclaimablePercent")
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

// SetCacheSize sets the value of CacheSize for the instance
func (instance *Win32_PerfFormattedData_Counters_StorageSpacesWriteCache) SetPropertyCacheSize(value uint64) (err error) {
	return instance.SetProperty("CacheSize", (value))
}

// GetCacheSize gets the value of CacheSize for the instance
func (instance *Win32_PerfFormattedData_Counters_StorageSpacesWriteCache) GetPropertyCacheSize() (value uint64, err error) {
	retValue, err := instance.GetProperty("CacheSize")
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

// SetCacheUsedBytes sets the value of CacheUsedBytes for the instance
func (instance *Win32_PerfFormattedData_Counters_StorageSpacesWriteCache) SetPropertyCacheUsedBytes(value uint64) (err error) {
	return instance.SetProperty("CacheUsedBytes", (value))
}

// GetCacheUsedBytes gets the value of CacheUsedBytes for the instance
func (instance *Win32_PerfFormattedData_Counters_StorageSpacesWriteCache) GetPropertyCacheUsedBytes() (value uint64, err error) {
	retValue, err := instance.GetProperty("CacheUsedBytes")
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

// SetCacheUsedPercent sets the value of CacheUsedPercent for the instance
func (instance *Win32_PerfFormattedData_Counters_StorageSpacesWriteCache) SetPropertyCacheUsedPercent(value uint64) (err error) {
	return instance.SetProperty("CacheUsedPercent", (value))
}

// GetCacheUsedPercent gets the value of CacheUsedPercent for the instance
func (instance *Win32_PerfFormattedData_Counters_StorageSpacesWriteCache) GetPropertyCacheUsedPercent() (value uint64, err error) {
	retValue, err := instance.GetProperty("CacheUsedPercent")
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

// SetEvictCacheBytesPersec sets the value of EvictCacheBytesPersec for the instance
func (instance *Win32_PerfFormattedData_Counters_StorageSpacesWriteCache) SetPropertyEvictCacheBytesPersec(value uint64) (err error) {
	return instance.SetProperty("EvictCacheBytesPersec", (value))
}

// GetEvictCacheBytesPersec gets the value of EvictCacheBytesPersec for the instance
func (instance *Win32_PerfFormattedData_Counters_StorageSpacesWriteCache) GetPropertyEvictCacheBytesPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("EvictCacheBytesPersec")
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

// SetEvictCacheDestagedBytesPersec sets the value of EvictCacheDestagedBytesPersec for the instance
func (instance *Win32_PerfFormattedData_Counters_StorageSpacesWriteCache) SetPropertyEvictCacheDestagedBytesPersec(value uint64) (err error) {
	return instance.SetProperty("EvictCacheDestagedBytesPersec", (value))
}

// GetEvictCacheDestagedBytesPersec gets the value of EvictCacheDestagedBytesPersec for the instance
func (instance *Win32_PerfFormattedData_Counters_StorageSpacesWriteCache) GetPropertyEvictCacheDestagedBytesPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("EvictCacheDestagedBytesPersec")
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

// SetEvictCacheDestagedPercent sets the value of EvictCacheDestagedPercent for the instance
func (instance *Win32_PerfFormattedData_Counters_StorageSpacesWriteCache) SetPropertyEvictCacheDestagedPercent(value uint64) (err error) {
	return instance.SetProperty("EvictCacheDestagedPercent", (value))
}

// GetEvictCacheDestagedPercent gets the value of EvictCacheDestagedPercent for the instance
func (instance *Win32_PerfFormattedData_Counters_StorageSpacesWriteCache) GetPropertyEvictCacheDestagedPercent() (value uint64, err error) {
	retValue, err := instance.GetProperty("EvictCacheDestagedPercent")
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

// SetEvictCacheOverwriteBytesPersec sets the value of EvictCacheOverwriteBytesPersec for the instance
func (instance *Win32_PerfFormattedData_Counters_StorageSpacesWriteCache) SetPropertyEvictCacheOverwriteBytesPersec(value uint64) (err error) {
	return instance.SetProperty("EvictCacheOverwriteBytesPersec", (value))
}

// GetEvictCacheOverwriteBytesPersec gets the value of EvictCacheOverwriteBytesPersec for the instance
func (instance *Win32_PerfFormattedData_Counters_StorageSpacesWriteCache) GetPropertyEvictCacheOverwriteBytesPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("EvictCacheOverwriteBytesPersec")
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

// SetEvictCacheOverwritePercent sets the value of EvictCacheOverwritePercent for the instance
func (instance *Win32_PerfFormattedData_Counters_StorageSpacesWriteCache) SetPropertyEvictCacheOverwritePercent(value uint64) (err error) {
	return instance.SetProperty("EvictCacheOverwritePercent", (value))
}

// GetEvictCacheOverwritePercent gets the value of EvictCacheOverwritePercent for the instance
func (instance *Win32_PerfFormattedData_Counters_StorageSpacesWriteCache) GetPropertyEvictCacheOverwritePercent() (value uint64, err error) {
	retValue, err := instance.GetProperty("EvictCacheOverwritePercent")
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

// SetReadBypassBytesPersec sets the value of ReadBypassBytesPersec for the instance
func (instance *Win32_PerfFormattedData_Counters_StorageSpacesWriteCache) SetPropertyReadBypassBytesPersec(value uint64) (err error) {
	return instance.SetProperty("ReadBypassBytesPersec", (value))
}

// GetReadBypassBytesPersec gets the value of ReadBypassBytesPersec for the instance
func (instance *Win32_PerfFormattedData_Counters_StorageSpacesWriteCache) GetPropertyReadBypassBytesPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("ReadBypassBytesPersec")
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

// SetReadBypassPercent sets the value of ReadBypassPercent for the instance
func (instance *Win32_PerfFormattedData_Counters_StorageSpacesWriteCache) SetPropertyReadBypassPercent(value uint64) (err error) {
	return instance.SetProperty("ReadBypassPercent", (value))
}

// GetReadBypassPercent gets the value of ReadBypassPercent for the instance
func (instance *Win32_PerfFormattedData_Counters_StorageSpacesWriteCache) GetPropertyReadBypassPercent() (value uint64, err error) {
	retValue, err := instance.GetProperty("ReadBypassPercent")
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

// SetReadCacheBytesPersec sets the value of ReadCacheBytesPersec for the instance
func (instance *Win32_PerfFormattedData_Counters_StorageSpacesWriteCache) SetPropertyReadCacheBytesPersec(value uint64) (err error) {
	return instance.SetProperty("ReadCacheBytesPersec", (value))
}

// GetReadCacheBytesPersec gets the value of ReadCacheBytesPersec for the instance
func (instance *Win32_PerfFormattedData_Counters_StorageSpacesWriteCache) GetPropertyReadCacheBytesPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("ReadCacheBytesPersec")
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

// SetReadCachePercent sets the value of ReadCachePercent for the instance
func (instance *Win32_PerfFormattedData_Counters_StorageSpacesWriteCache) SetPropertyReadCachePercent(value uint64) (err error) {
	return instance.SetProperty("ReadCachePercent", (value))
}

// GetReadCachePercent gets the value of ReadCachePercent for the instance
func (instance *Win32_PerfFormattedData_Counters_StorageSpacesWriteCache) GetPropertyReadCachePercent() (value uint64, err error) {
	retValue, err := instance.GetProperty("ReadCachePercent")
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

// SetWriteBypassBytesPersec sets the value of WriteBypassBytesPersec for the instance
func (instance *Win32_PerfFormattedData_Counters_StorageSpacesWriteCache) SetPropertyWriteBypassBytesPersec(value uint64) (err error) {
	return instance.SetProperty("WriteBypassBytesPersec", (value))
}

// GetWriteBypassBytesPersec gets the value of WriteBypassBytesPersec for the instance
func (instance *Win32_PerfFormattedData_Counters_StorageSpacesWriteCache) GetPropertyWriteBypassBytesPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("WriteBypassBytesPersec")
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

// SetWriteBypassPercent sets the value of WriteBypassPercent for the instance
func (instance *Win32_PerfFormattedData_Counters_StorageSpacesWriteCache) SetPropertyWriteBypassPercent(value uint64) (err error) {
	return instance.SetProperty("WriteBypassPercent", (value))
}

// GetWriteBypassPercent gets the value of WriteBypassPercent for the instance
func (instance *Win32_PerfFormattedData_Counters_StorageSpacesWriteCache) GetPropertyWriteBypassPercent() (value uint64, err error) {
	retValue, err := instance.GetProperty("WriteBypassPercent")
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

// SetWriteCacheBytesPersec sets the value of WriteCacheBytesPersec for the instance
func (instance *Win32_PerfFormattedData_Counters_StorageSpacesWriteCache) SetPropertyWriteCacheBytesPersec(value uint64) (err error) {
	return instance.SetProperty("WriteCacheBytesPersec", (value))
}

// GetWriteCacheBytesPersec gets the value of WriteCacheBytesPersec for the instance
func (instance *Win32_PerfFormattedData_Counters_StorageSpacesWriteCache) GetPropertyWriteCacheBytesPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("WriteCacheBytesPersec")
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

// SetWriteCacheOverlapBytesPersec sets the value of WriteCacheOverlapBytesPersec for the instance
func (instance *Win32_PerfFormattedData_Counters_StorageSpacesWriteCache) SetPropertyWriteCacheOverlapBytesPersec(value uint64) (err error) {
	return instance.SetProperty("WriteCacheOverlapBytesPersec", (value))
}

// GetWriteCacheOverlapBytesPersec gets the value of WriteCacheOverlapBytesPersec for the instance
func (instance *Win32_PerfFormattedData_Counters_StorageSpacesWriteCache) GetPropertyWriteCacheOverlapBytesPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("WriteCacheOverlapBytesPersec")
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

// SetWriteCacheOverlapPercent sets the value of WriteCacheOverlapPercent for the instance
func (instance *Win32_PerfFormattedData_Counters_StorageSpacesWriteCache) SetPropertyWriteCacheOverlapPercent(value uint64) (err error) {
	return instance.SetProperty("WriteCacheOverlapPercent", (value))
}

// GetWriteCacheOverlapPercent gets the value of WriteCacheOverlapPercent for the instance
func (instance *Win32_PerfFormattedData_Counters_StorageSpacesWriteCache) GetPropertyWriteCacheOverlapPercent() (value uint64, err error) {
	retValue, err := instance.GetProperty("WriteCacheOverlapPercent")
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

// SetWriteCachePercent sets the value of WriteCachePercent for the instance
func (instance *Win32_PerfFormattedData_Counters_StorageSpacesWriteCache) SetPropertyWriteCachePercent(value uint64) (err error) {
	return instance.SetProperty("WriteCachePercent", (value))
}

// GetWriteCachePercent gets the value of WriteCachePercent for the instance
func (instance *Win32_PerfFormattedData_Counters_StorageSpacesWriteCache) GetPropertyWriteCachePercent() (value uint64, err error) {
	retValue, err := instance.GetProperty("WriteCachePercent")
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

// SetWriteCacheUnalignedBytesPersec sets the value of WriteCacheUnalignedBytesPersec for the instance
func (instance *Win32_PerfFormattedData_Counters_StorageSpacesWriteCache) SetPropertyWriteCacheUnalignedBytesPersec(value uint64) (err error) {
	return instance.SetProperty("WriteCacheUnalignedBytesPersec", (value))
}

// GetWriteCacheUnalignedBytesPersec gets the value of WriteCacheUnalignedBytesPersec for the instance
func (instance *Win32_PerfFormattedData_Counters_StorageSpacesWriteCache) GetPropertyWriteCacheUnalignedBytesPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("WriteCacheUnalignedBytesPersec")
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

// SetWriteCacheUnalignedPercent sets the value of WriteCacheUnalignedPercent for the instance
func (instance *Win32_PerfFormattedData_Counters_StorageSpacesWriteCache) SetPropertyWriteCacheUnalignedPercent(value uint64) (err error) {
	return instance.SetProperty("WriteCacheUnalignedPercent", (value))
}

// GetWriteCacheUnalignedPercent gets the value of WriteCacheUnalignedPercent for the instance
func (instance *Win32_PerfFormattedData_Counters_StorageSpacesWriteCache) GetPropertyWriteCacheUnalignedPercent() (value uint64, err error) {
	retValue, err := instance.GetProperty("WriteCacheUnalignedPercent")
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

// SetWriteCacheUntrimmedBytesPersec sets the value of WriteCacheUntrimmedBytesPersec for the instance
func (instance *Win32_PerfFormattedData_Counters_StorageSpacesWriteCache) SetPropertyWriteCacheUntrimmedBytesPersec(value uint64) (err error) {
	return instance.SetProperty("WriteCacheUntrimmedBytesPersec", (value))
}

// GetWriteCacheUntrimmedBytesPersec gets the value of WriteCacheUntrimmedBytesPersec for the instance
func (instance *Win32_PerfFormattedData_Counters_StorageSpacesWriteCache) GetPropertyWriteCacheUntrimmedBytesPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("WriteCacheUntrimmedBytesPersec")
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

// SetWriteCacheUntrimmedPercent sets the value of WriteCacheUntrimmedPercent for the instance
func (instance *Win32_PerfFormattedData_Counters_StorageSpacesWriteCache) SetPropertyWriteCacheUntrimmedPercent(value uint64) (err error) {
	return instance.SetProperty("WriteCacheUntrimmedPercent", (value))
}

// GetWriteCacheUntrimmedPercent gets the value of WriteCacheUntrimmedPercent for the instance
func (instance *Win32_PerfFormattedData_Counters_StorageSpacesWriteCache) GetPropertyWriteCacheUntrimmedPercent() (value uint64, err error) {
	retValue, err := instance.GetProperty("WriteCacheUntrimmedPercent")
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
