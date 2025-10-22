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

// Win32_PerfRawData_CsvVolumePerfProvider_ClusterCSVVolumeCache struct
type Win32_PerfRawData_CsvVolumePerfProvider_ClusterCSVVolumeCache struct {
	*Win32_PerfRawData

	//
	CacheIOReadBytes uint64

	//
	CacheIOReadBytesPersec uint64

	//
	CacheRead uint64

	//
	CacheReadPerSec uint64

	//
	CacheSizeConfigured uint64

	//
	CacheSizeCurrent uint64

	//
	CacheState uint64

	//
	DiskIOReadBytes uint64

	//
	DiskIOReadBytesPerSec uint64

	//
	DiskIOReads uint64

	//
	DiskIOReadsPerSec uint64

	//
	IOReadBytes uint64

	//
	IOReadBytesPerSec uint64

	//
	IOReads uint64

	//
	IOReadsPerSec uint64

	//
	LRUCacheSizeCurrent uint64

	//
	LRUCacheSizeTarget uint64

	//
	PartialRead uint64

	//
	PartialReadPersec uint64

	//
	PercentCacheValid uint64

	//
	PercentCacheValid_Base uint64

	//
	ValidCacheSize uint64
}

func NewWin32_PerfRawData_CsvVolumePerfProvider_ClusterCSVVolumeCacheEx1(instance *cim.WmiInstance) (newInstance *Win32_PerfRawData_CsvVolumePerfProvider_ClusterCSVVolumeCache, err error) {
	tmp, err := NewWin32_PerfRawDataEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfRawData_CsvVolumePerfProvider_ClusterCSVVolumeCache{
		Win32_PerfRawData: tmp,
	}
	return
}

func NewWin32_PerfRawData_CsvVolumePerfProvider_ClusterCSVVolumeCacheEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_PerfRawData_CsvVolumePerfProvider_ClusterCSVVolumeCache, err error) {
	tmp, err := NewWin32_PerfRawDataEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfRawData_CsvVolumePerfProvider_ClusterCSVVolumeCache{
		Win32_PerfRawData: tmp,
	}
	return
}

// SetCacheIOReadBytes sets the value of CacheIOReadBytes for the instance
func (instance *Win32_PerfRawData_CsvVolumePerfProvider_ClusterCSVVolumeCache) SetPropertyCacheIOReadBytes(value uint64) (err error) {
	return instance.SetProperty("CacheIOReadBytes", (value))
}

// GetCacheIOReadBytes gets the value of CacheIOReadBytes for the instance
func (instance *Win32_PerfRawData_CsvVolumePerfProvider_ClusterCSVVolumeCache) GetPropertyCacheIOReadBytes() (value uint64, err error) {
	retValue, err := instance.GetProperty("CacheIOReadBytes")
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

// SetCacheIOReadBytesPersec sets the value of CacheIOReadBytesPersec for the instance
func (instance *Win32_PerfRawData_CsvVolumePerfProvider_ClusterCSVVolumeCache) SetPropertyCacheIOReadBytesPersec(value uint64) (err error) {
	return instance.SetProperty("CacheIOReadBytesPersec", (value))
}

// GetCacheIOReadBytesPersec gets the value of CacheIOReadBytesPersec for the instance
func (instance *Win32_PerfRawData_CsvVolumePerfProvider_ClusterCSVVolumeCache) GetPropertyCacheIOReadBytesPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("CacheIOReadBytesPersec")
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

// SetCacheRead sets the value of CacheRead for the instance
func (instance *Win32_PerfRawData_CsvVolumePerfProvider_ClusterCSVVolumeCache) SetPropertyCacheRead(value uint64) (err error) {
	return instance.SetProperty("CacheRead", (value))
}

// GetCacheRead gets the value of CacheRead for the instance
func (instance *Win32_PerfRawData_CsvVolumePerfProvider_ClusterCSVVolumeCache) GetPropertyCacheRead() (value uint64, err error) {
	retValue, err := instance.GetProperty("CacheRead")
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

// SetCacheReadPerSec sets the value of CacheReadPerSec for the instance
func (instance *Win32_PerfRawData_CsvVolumePerfProvider_ClusterCSVVolumeCache) SetPropertyCacheReadPerSec(value uint64) (err error) {
	return instance.SetProperty("CacheReadPerSec", (value))
}

// GetCacheReadPerSec gets the value of CacheReadPerSec for the instance
func (instance *Win32_PerfRawData_CsvVolumePerfProvider_ClusterCSVVolumeCache) GetPropertyCacheReadPerSec() (value uint64, err error) {
	retValue, err := instance.GetProperty("CacheReadPerSec")
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

// SetCacheSizeConfigured sets the value of CacheSizeConfigured for the instance
func (instance *Win32_PerfRawData_CsvVolumePerfProvider_ClusterCSVVolumeCache) SetPropertyCacheSizeConfigured(value uint64) (err error) {
	return instance.SetProperty("CacheSizeConfigured", (value))
}

// GetCacheSizeConfigured gets the value of CacheSizeConfigured for the instance
func (instance *Win32_PerfRawData_CsvVolumePerfProvider_ClusterCSVVolumeCache) GetPropertyCacheSizeConfigured() (value uint64, err error) {
	retValue, err := instance.GetProperty("CacheSizeConfigured")
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

// SetCacheSizeCurrent sets the value of CacheSizeCurrent for the instance
func (instance *Win32_PerfRawData_CsvVolumePerfProvider_ClusterCSVVolumeCache) SetPropertyCacheSizeCurrent(value uint64) (err error) {
	return instance.SetProperty("CacheSizeCurrent", (value))
}

// GetCacheSizeCurrent gets the value of CacheSizeCurrent for the instance
func (instance *Win32_PerfRawData_CsvVolumePerfProvider_ClusterCSVVolumeCache) GetPropertyCacheSizeCurrent() (value uint64, err error) {
	retValue, err := instance.GetProperty("CacheSizeCurrent")
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

// SetCacheState sets the value of CacheState for the instance
func (instance *Win32_PerfRawData_CsvVolumePerfProvider_ClusterCSVVolumeCache) SetPropertyCacheState(value uint64) (err error) {
	return instance.SetProperty("CacheState", (value))
}

// GetCacheState gets the value of CacheState for the instance
func (instance *Win32_PerfRawData_CsvVolumePerfProvider_ClusterCSVVolumeCache) GetPropertyCacheState() (value uint64, err error) {
	retValue, err := instance.GetProperty("CacheState")
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

// SetDiskIOReadBytes sets the value of DiskIOReadBytes for the instance
func (instance *Win32_PerfRawData_CsvVolumePerfProvider_ClusterCSVVolumeCache) SetPropertyDiskIOReadBytes(value uint64) (err error) {
	return instance.SetProperty("DiskIOReadBytes", (value))
}

// GetDiskIOReadBytes gets the value of DiskIOReadBytes for the instance
func (instance *Win32_PerfRawData_CsvVolumePerfProvider_ClusterCSVVolumeCache) GetPropertyDiskIOReadBytes() (value uint64, err error) {
	retValue, err := instance.GetProperty("DiskIOReadBytes")
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

// SetDiskIOReadBytesPerSec sets the value of DiskIOReadBytesPerSec for the instance
func (instance *Win32_PerfRawData_CsvVolumePerfProvider_ClusterCSVVolumeCache) SetPropertyDiskIOReadBytesPerSec(value uint64) (err error) {
	return instance.SetProperty("DiskIOReadBytesPerSec", (value))
}

// GetDiskIOReadBytesPerSec gets the value of DiskIOReadBytesPerSec for the instance
func (instance *Win32_PerfRawData_CsvVolumePerfProvider_ClusterCSVVolumeCache) GetPropertyDiskIOReadBytesPerSec() (value uint64, err error) {
	retValue, err := instance.GetProperty("DiskIOReadBytesPerSec")
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

// SetDiskIOReads sets the value of DiskIOReads for the instance
func (instance *Win32_PerfRawData_CsvVolumePerfProvider_ClusterCSVVolumeCache) SetPropertyDiskIOReads(value uint64) (err error) {
	return instance.SetProperty("DiskIOReads", (value))
}

// GetDiskIOReads gets the value of DiskIOReads for the instance
func (instance *Win32_PerfRawData_CsvVolumePerfProvider_ClusterCSVVolumeCache) GetPropertyDiskIOReads() (value uint64, err error) {
	retValue, err := instance.GetProperty("DiskIOReads")
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

// SetDiskIOReadsPerSec sets the value of DiskIOReadsPerSec for the instance
func (instance *Win32_PerfRawData_CsvVolumePerfProvider_ClusterCSVVolumeCache) SetPropertyDiskIOReadsPerSec(value uint64) (err error) {
	return instance.SetProperty("DiskIOReadsPerSec", (value))
}

// GetDiskIOReadsPerSec gets the value of DiskIOReadsPerSec for the instance
func (instance *Win32_PerfRawData_CsvVolumePerfProvider_ClusterCSVVolumeCache) GetPropertyDiskIOReadsPerSec() (value uint64, err error) {
	retValue, err := instance.GetProperty("DiskIOReadsPerSec")
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

// SetIOReadBytes sets the value of IOReadBytes for the instance
func (instance *Win32_PerfRawData_CsvVolumePerfProvider_ClusterCSVVolumeCache) SetPropertyIOReadBytes(value uint64) (err error) {
	return instance.SetProperty("IOReadBytes", (value))
}

// GetIOReadBytes gets the value of IOReadBytes for the instance
func (instance *Win32_PerfRawData_CsvVolumePerfProvider_ClusterCSVVolumeCache) GetPropertyIOReadBytes() (value uint64, err error) {
	retValue, err := instance.GetProperty("IOReadBytes")
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

// SetIOReadBytesPerSec sets the value of IOReadBytesPerSec for the instance
func (instance *Win32_PerfRawData_CsvVolumePerfProvider_ClusterCSVVolumeCache) SetPropertyIOReadBytesPerSec(value uint64) (err error) {
	return instance.SetProperty("IOReadBytesPerSec", (value))
}

// GetIOReadBytesPerSec gets the value of IOReadBytesPerSec for the instance
func (instance *Win32_PerfRawData_CsvVolumePerfProvider_ClusterCSVVolumeCache) GetPropertyIOReadBytesPerSec() (value uint64, err error) {
	retValue, err := instance.GetProperty("IOReadBytesPerSec")
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

// SetIOReads sets the value of IOReads for the instance
func (instance *Win32_PerfRawData_CsvVolumePerfProvider_ClusterCSVVolumeCache) SetPropertyIOReads(value uint64) (err error) {
	return instance.SetProperty("IOReads", (value))
}

// GetIOReads gets the value of IOReads for the instance
func (instance *Win32_PerfRawData_CsvVolumePerfProvider_ClusterCSVVolumeCache) GetPropertyIOReads() (value uint64, err error) {
	retValue, err := instance.GetProperty("IOReads")
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

// SetIOReadsPerSec sets the value of IOReadsPerSec for the instance
func (instance *Win32_PerfRawData_CsvVolumePerfProvider_ClusterCSVVolumeCache) SetPropertyIOReadsPerSec(value uint64) (err error) {
	return instance.SetProperty("IOReadsPerSec", (value))
}

// GetIOReadsPerSec gets the value of IOReadsPerSec for the instance
func (instance *Win32_PerfRawData_CsvVolumePerfProvider_ClusterCSVVolumeCache) GetPropertyIOReadsPerSec() (value uint64, err error) {
	retValue, err := instance.GetProperty("IOReadsPerSec")
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

// SetLRUCacheSizeCurrent sets the value of LRUCacheSizeCurrent for the instance
func (instance *Win32_PerfRawData_CsvVolumePerfProvider_ClusterCSVVolumeCache) SetPropertyLRUCacheSizeCurrent(value uint64) (err error) {
	return instance.SetProperty("LRUCacheSizeCurrent", (value))
}

// GetLRUCacheSizeCurrent gets the value of LRUCacheSizeCurrent for the instance
func (instance *Win32_PerfRawData_CsvVolumePerfProvider_ClusterCSVVolumeCache) GetPropertyLRUCacheSizeCurrent() (value uint64, err error) {
	retValue, err := instance.GetProperty("LRUCacheSizeCurrent")
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

// SetLRUCacheSizeTarget sets the value of LRUCacheSizeTarget for the instance
func (instance *Win32_PerfRawData_CsvVolumePerfProvider_ClusterCSVVolumeCache) SetPropertyLRUCacheSizeTarget(value uint64) (err error) {
	return instance.SetProperty("LRUCacheSizeTarget", (value))
}

// GetLRUCacheSizeTarget gets the value of LRUCacheSizeTarget for the instance
func (instance *Win32_PerfRawData_CsvVolumePerfProvider_ClusterCSVVolumeCache) GetPropertyLRUCacheSizeTarget() (value uint64, err error) {
	retValue, err := instance.GetProperty("LRUCacheSizeTarget")
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

// SetPartialRead sets the value of PartialRead for the instance
func (instance *Win32_PerfRawData_CsvVolumePerfProvider_ClusterCSVVolumeCache) SetPropertyPartialRead(value uint64) (err error) {
	return instance.SetProperty("PartialRead", (value))
}

// GetPartialRead gets the value of PartialRead for the instance
func (instance *Win32_PerfRawData_CsvVolumePerfProvider_ClusterCSVVolumeCache) GetPropertyPartialRead() (value uint64, err error) {
	retValue, err := instance.GetProperty("PartialRead")
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

// SetPartialReadPersec sets the value of PartialReadPersec for the instance
func (instance *Win32_PerfRawData_CsvVolumePerfProvider_ClusterCSVVolumeCache) SetPropertyPartialReadPersec(value uint64) (err error) {
	return instance.SetProperty("PartialReadPersec", (value))
}

// GetPartialReadPersec gets the value of PartialReadPersec for the instance
func (instance *Win32_PerfRawData_CsvVolumePerfProvider_ClusterCSVVolumeCache) GetPropertyPartialReadPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("PartialReadPersec")
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

// SetPercentCacheValid sets the value of PercentCacheValid for the instance
func (instance *Win32_PerfRawData_CsvVolumePerfProvider_ClusterCSVVolumeCache) SetPropertyPercentCacheValid(value uint64) (err error) {
	return instance.SetProperty("PercentCacheValid", (value))
}

// GetPercentCacheValid gets the value of PercentCacheValid for the instance
func (instance *Win32_PerfRawData_CsvVolumePerfProvider_ClusterCSVVolumeCache) GetPropertyPercentCacheValid() (value uint64, err error) {
	retValue, err := instance.GetProperty("PercentCacheValid")
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

// SetPercentCacheValid_Base sets the value of PercentCacheValid_Base for the instance
func (instance *Win32_PerfRawData_CsvVolumePerfProvider_ClusterCSVVolumeCache) SetPropertyPercentCacheValid_Base(value uint64) (err error) {
	return instance.SetProperty("PercentCacheValid_Base", (value))
}

// GetPercentCacheValid_Base gets the value of PercentCacheValid_Base for the instance
func (instance *Win32_PerfRawData_CsvVolumePerfProvider_ClusterCSVVolumeCache) GetPropertyPercentCacheValid_Base() (value uint64, err error) {
	retValue, err := instance.GetProperty("PercentCacheValid_Base")
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

// SetValidCacheSize sets the value of ValidCacheSize for the instance
func (instance *Win32_PerfRawData_CsvVolumePerfProvider_ClusterCSVVolumeCache) SetPropertyValidCacheSize(value uint64) (err error) {
	return instance.SetProperty("ValidCacheSize", (value))
}

// GetValidCacheSize gets the value of ValidCacheSize for the instance
func (instance *Win32_PerfRawData_CsvVolumePerfProvider_ClusterCSVVolumeCache) GetPropertyValidCacheSize() (value uint64, err error) {
	retValue, err := instance.GetProperty("ValidCacheSize")
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
