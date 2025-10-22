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

// Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageHybridDisks struct
type Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageHybridDisks struct {
	*Win32_PerfRawData

	//
	BindingAttributes uint64

	//
	CacheFirstHitPopulatedBytes uint64

	//
	CacheFirstHitPopulatedBytesPersec uint64

	//
	CacheFirstHitWrittenBytes uint64

	//
	CacheFirstHitWrittenBytesPersec uint64

	//
	CacheHitReadBytes uint64

	//
	CacheHitReadBytesPersec uint64

	//
	CacheHitReads uint64

	//
	CacheHitReadsPersec uint64

	//
	CacheMissReadBytes uint64

	//
	CacheMissReadBytesPersec uint64

	//
	CacheMissReads uint64

	//
	CacheMissReadsPersec uint64

	//
	CachePages uint64

	//
	CachePagesDirty uint64

	//
	CachePagesDirtyHot uint64

	//
	CachePagesDiscardIgnored uint64

	//
	CachePagesL2 uint64

	//
	CachePopulateBytes uint64

	//
	CachePopulateBytesPersec uint64

	//
	CacheWriteBytes uint64

	//
	CacheWriteBytesPersec uint64

	//
	CacheWrites uint64

	//
	CacheWritesPersec uint64

	//
	DestageBytes uint64

	//
	DestageBytesPersec uint64

	//
	DestageTransfers uint64

	//
	DestageTransfersPersec uint64

	//
	DirectReadBytes uint64

	//
	DirectReadBytesPersec uint64

	//
	DirectReads uint64

	//
	DirectReadsPersec uint64

	//
	DirectWriteBytes uint64

	//
	DirectWriteBytesPersec uint64

	//
	DirectWrites uint64

	//
	DirectWritesPersec uint64

	//
	DirtyReadBytes uint64

	//
	DirtyReadBytesPersec uint64

	//
	DirtySlots uint64

	//
	DirtySlotsExpands uint64

	//
	DirtySlotsExpandsPersec uint64

	//
	DiskBytes uint64

	//
	DiskBytesPersec uint64

	//
	DiskReadBytes uint64

	//
	DiskReadBytesPersec uint64

	//
	DiskReads uint64

	//
	DiskReadsPersec uint64

	//
	DiskTransfers uint64

	//
	DiskTransfersPersec uint64

	//
	DiskWriteBytes uint64

	//
	DiskWriteBytesPersec uint64

	//
	DiskWrites uint64

	//
	DiskWritesPersec uint64

	//
	MissingSlots uint64

	//
	RateDiskCacheReads uint64

	//
	RateDiskCacheReads_Base uint32

	//
	RateDiskCacheWrites uint64

	//
	RateDiskCacheWrites_Base uint32

	//
	ReadErrorsMedia uint64

	//
	ReadErrorsTimeout uint64

	//
	ReadErrorsTotal uint64

	//
	WriteErrorsMedia uint64

	//
	WriteErrorsTimeout uint64

	//
	WriteErrorsTotal uint64
}

func NewWin32_PerfRawData_ClusBfltPerfProvider_ClusterStorageHybridDisksEx1(instance *cim.WmiInstance) (newInstance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageHybridDisks, err error) {
	tmp, err := NewWin32_PerfRawDataEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageHybridDisks{
		Win32_PerfRawData: tmp,
	}
	return
}

func NewWin32_PerfRawData_ClusBfltPerfProvider_ClusterStorageHybridDisksEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageHybridDisks, err error) {
	tmp, err := NewWin32_PerfRawDataEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageHybridDisks{
		Win32_PerfRawData: tmp,
	}
	return
}

// SetBindingAttributes sets the value of BindingAttributes for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageHybridDisks) SetPropertyBindingAttributes(value uint64) (err error) {
	return instance.SetProperty("BindingAttributes", (value))
}

// GetBindingAttributes gets the value of BindingAttributes for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageHybridDisks) GetPropertyBindingAttributes() (value uint64, err error) {
	retValue, err := instance.GetProperty("BindingAttributes")
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

// SetCacheFirstHitPopulatedBytes sets the value of CacheFirstHitPopulatedBytes for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageHybridDisks) SetPropertyCacheFirstHitPopulatedBytes(value uint64) (err error) {
	return instance.SetProperty("CacheFirstHitPopulatedBytes", (value))
}

// GetCacheFirstHitPopulatedBytes gets the value of CacheFirstHitPopulatedBytes for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageHybridDisks) GetPropertyCacheFirstHitPopulatedBytes() (value uint64, err error) {
	retValue, err := instance.GetProperty("CacheFirstHitPopulatedBytes")
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

// SetCacheFirstHitPopulatedBytesPersec sets the value of CacheFirstHitPopulatedBytesPersec for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageHybridDisks) SetPropertyCacheFirstHitPopulatedBytesPersec(value uint64) (err error) {
	return instance.SetProperty("CacheFirstHitPopulatedBytesPersec", (value))
}

// GetCacheFirstHitPopulatedBytesPersec gets the value of CacheFirstHitPopulatedBytesPersec for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageHybridDisks) GetPropertyCacheFirstHitPopulatedBytesPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("CacheFirstHitPopulatedBytesPersec")
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

// SetCacheFirstHitWrittenBytes sets the value of CacheFirstHitWrittenBytes for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageHybridDisks) SetPropertyCacheFirstHitWrittenBytes(value uint64) (err error) {
	return instance.SetProperty("CacheFirstHitWrittenBytes", (value))
}

// GetCacheFirstHitWrittenBytes gets the value of CacheFirstHitWrittenBytes for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageHybridDisks) GetPropertyCacheFirstHitWrittenBytes() (value uint64, err error) {
	retValue, err := instance.GetProperty("CacheFirstHitWrittenBytes")
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

// SetCacheFirstHitWrittenBytesPersec sets the value of CacheFirstHitWrittenBytesPersec for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageHybridDisks) SetPropertyCacheFirstHitWrittenBytesPersec(value uint64) (err error) {
	return instance.SetProperty("CacheFirstHitWrittenBytesPersec", (value))
}

// GetCacheFirstHitWrittenBytesPersec gets the value of CacheFirstHitWrittenBytesPersec for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageHybridDisks) GetPropertyCacheFirstHitWrittenBytesPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("CacheFirstHitWrittenBytesPersec")
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

// SetCacheHitReadBytes sets the value of CacheHitReadBytes for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageHybridDisks) SetPropertyCacheHitReadBytes(value uint64) (err error) {
	return instance.SetProperty("CacheHitReadBytes", (value))
}

// GetCacheHitReadBytes gets the value of CacheHitReadBytes for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageHybridDisks) GetPropertyCacheHitReadBytes() (value uint64, err error) {
	retValue, err := instance.GetProperty("CacheHitReadBytes")
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

// SetCacheHitReadBytesPersec sets the value of CacheHitReadBytesPersec for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageHybridDisks) SetPropertyCacheHitReadBytesPersec(value uint64) (err error) {
	return instance.SetProperty("CacheHitReadBytesPersec", (value))
}

// GetCacheHitReadBytesPersec gets the value of CacheHitReadBytesPersec for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageHybridDisks) GetPropertyCacheHitReadBytesPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("CacheHitReadBytesPersec")
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

// SetCacheHitReads sets the value of CacheHitReads for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageHybridDisks) SetPropertyCacheHitReads(value uint64) (err error) {
	return instance.SetProperty("CacheHitReads", (value))
}

// GetCacheHitReads gets the value of CacheHitReads for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageHybridDisks) GetPropertyCacheHitReads() (value uint64, err error) {
	retValue, err := instance.GetProperty("CacheHitReads")
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

// SetCacheHitReadsPersec sets the value of CacheHitReadsPersec for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageHybridDisks) SetPropertyCacheHitReadsPersec(value uint64) (err error) {
	return instance.SetProperty("CacheHitReadsPersec", (value))
}

// GetCacheHitReadsPersec gets the value of CacheHitReadsPersec for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageHybridDisks) GetPropertyCacheHitReadsPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("CacheHitReadsPersec")
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

// SetCacheMissReadBytes sets the value of CacheMissReadBytes for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageHybridDisks) SetPropertyCacheMissReadBytes(value uint64) (err error) {
	return instance.SetProperty("CacheMissReadBytes", (value))
}

// GetCacheMissReadBytes gets the value of CacheMissReadBytes for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageHybridDisks) GetPropertyCacheMissReadBytes() (value uint64, err error) {
	retValue, err := instance.GetProperty("CacheMissReadBytes")
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

// SetCacheMissReadBytesPersec sets the value of CacheMissReadBytesPersec for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageHybridDisks) SetPropertyCacheMissReadBytesPersec(value uint64) (err error) {
	return instance.SetProperty("CacheMissReadBytesPersec", (value))
}

// GetCacheMissReadBytesPersec gets the value of CacheMissReadBytesPersec for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageHybridDisks) GetPropertyCacheMissReadBytesPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("CacheMissReadBytesPersec")
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

// SetCacheMissReads sets the value of CacheMissReads for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageHybridDisks) SetPropertyCacheMissReads(value uint64) (err error) {
	return instance.SetProperty("CacheMissReads", (value))
}

// GetCacheMissReads gets the value of CacheMissReads for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageHybridDisks) GetPropertyCacheMissReads() (value uint64, err error) {
	retValue, err := instance.GetProperty("CacheMissReads")
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

// SetCacheMissReadsPersec sets the value of CacheMissReadsPersec for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageHybridDisks) SetPropertyCacheMissReadsPersec(value uint64) (err error) {
	return instance.SetProperty("CacheMissReadsPersec", (value))
}

// GetCacheMissReadsPersec gets the value of CacheMissReadsPersec for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageHybridDisks) GetPropertyCacheMissReadsPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("CacheMissReadsPersec")
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
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageHybridDisks) SetPropertyCachePages(value uint64) (err error) {
	return instance.SetProperty("CachePages", (value))
}

// GetCachePages gets the value of CachePages for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageHybridDisks) GetPropertyCachePages() (value uint64, err error) {
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

// SetCachePagesDirty sets the value of CachePagesDirty for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageHybridDisks) SetPropertyCachePagesDirty(value uint64) (err error) {
	return instance.SetProperty("CachePagesDirty", (value))
}

// GetCachePagesDirty gets the value of CachePagesDirty for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageHybridDisks) GetPropertyCachePagesDirty() (value uint64, err error) {
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

// SetCachePagesDirtyHot sets the value of CachePagesDirtyHot for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageHybridDisks) SetPropertyCachePagesDirtyHot(value uint64) (err error) {
	return instance.SetProperty("CachePagesDirtyHot", (value))
}

// GetCachePagesDirtyHot gets the value of CachePagesDirtyHot for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageHybridDisks) GetPropertyCachePagesDirtyHot() (value uint64, err error) {
	retValue, err := instance.GetProperty("CachePagesDirtyHot")
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

// SetCachePagesDiscardIgnored sets the value of CachePagesDiscardIgnored for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageHybridDisks) SetPropertyCachePagesDiscardIgnored(value uint64) (err error) {
	return instance.SetProperty("CachePagesDiscardIgnored", (value))
}

// GetCachePagesDiscardIgnored gets the value of CachePagesDiscardIgnored for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageHybridDisks) GetPropertyCachePagesDiscardIgnored() (value uint64, err error) {
	retValue, err := instance.GetProperty("CachePagesDiscardIgnored")
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

// SetCachePagesL2 sets the value of CachePagesL2 for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageHybridDisks) SetPropertyCachePagesL2(value uint64) (err error) {
	return instance.SetProperty("CachePagesL2", (value))
}

// GetCachePagesL2 gets the value of CachePagesL2 for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageHybridDisks) GetPropertyCachePagesL2() (value uint64, err error) {
	retValue, err := instance.GetProperty("CachePagesL2")
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

// SetCachePopulateBytes sets the value of CachePopulateBytes for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageHybridDisks) SetPropertyCachePopulateBytes(value uint64) (err error) {
	return instance.SetProperty("CachePopulateBytes", (value))
}

// GetCachePopulateBytes gets the value of CachePopulateBytes for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageHybridDisks) GetPropertyCachePopulateBytes() (value uint64, err error) {
	retValue, err := instance.GetProperty("CachePopulateBytes")
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

// SetCachePopulateBytesPersec sets the value of CachePopulateBytesPersec for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageHybridDisks) SetPropertyCachePopulateBytesPersec(value uint64) (err error) {
	return instance.SetProperty("CachePopulateBytesPersec", (value))
}

// GetCachePopulateBytesPersec gets the value of CachePopulateBytesPersec for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageHybridDisks) GetPropertyCachePopulateBytesPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("CachePopulateBytesPersec")
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

// SetCacheWriteBytes sets the value of CacheWriteBytes for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageHybridDisks) SetPropertyCacheWriteBytes(value uint64) (err error) {
	return instance.SetProperty("CacheWriteBytes", (value))
}

// GetCacheWriteBytes gets the value of CacheWriteBytes for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageHybridDisks) GetPropertyCacheWriteBytes() (value uint64, err error) {
	retValue, err := instance.GetProperty("CacheWriteBytes")
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

// SetCacheWriteBytesPersec sets the value of CacheWriteBytesPersec for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageHybridDisks) SetPropertyCacheWriteBytesPersec(value uint64) (err error) {
	return instance.SetProperty("CacheWriteBytesPersec", (value))
}

// GetCacheWriteBytesPersec gets the value of CacheWriteBytesPersec for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageHybridDisks) GetPropertyCacheWriteBytesPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("CacheWriteBytesPersec")
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

// SetCacheWrites sets the value of CacheWrites for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageHybridDisks) SetPropertyCacheWrites(value uint64) (err error) {
	return instance.SetProperty("CacheWrites", (value))
}

// GetCacheWrites gets the value of CacheWrites for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageHybridDisks) GetPropertyCacheWrites() (value uint64, err error) {
	retValue, err := instance.GetProperty("CacheWrites")
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

// SetCacheWritesPersec sets the value of CacheWritesPersec for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageHybridDisks) SetPropertyCacheWritesPersec(value uint64) (err error) {
	return instance.SetProperty("CacheWritesPersec", (value))
}

// GetCacheWritesPersec gets the value of CacheWritesPersec for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageHybridDisks) GetPropertyCacheWritesPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("CacheWritesPersec")
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
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageHybridDisks) SetPropertyDestageBytes(value uint64) (err error) {
	return instance.SetProperty("DestageBytes", (value))
}

// GetDestageBytes gets the value of DestageBytes for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageHybridDisks) GetPropertyDestageBytes() (value uint64, err error) {
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
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageHybridDisks) SetPropertyDestageBytesPersec(value uint64) (err error) {
	return instance.SetProperty("DestageBytesPersec", (value))
}

// GetDestageBytesPersec gets the value of DestageBytesPersec for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageHybridDisks) GetPropertyDestageBytesPersec() (value uint64, err error) {
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

// SetDestageTransfers sets the value of DestageTransfers for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageHybridDisks) SetPropertyDestageTransfers(value uint64) (err error) {
	return instance.SetProperty("DestageTransfers", (value))
}

// GetDestageTransfers gets the value of DestageTransfers for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageHybridDisks) GetPropertyDestageTransfers() (value uint64, err error) {
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
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageHybridDisks) SetPropertyDestageTransfersPersec(value uint64) (err error) {
	return instance.SetProperty("DestageTransfersPersec", (value))
}

// GetDestageTransfersPersec gets the value of DestageTransfersPersec for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageHybridDisks) GetPropertyDestageTransfersPersec() (value uint64, err error) {
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

// SetDirectReadBytes sets the value of DirectReadBytes for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageHybridDisks) SetPropertyDirectReadBytes(value uint64) (err error) {
	return instance.SetProperty("DirectReadBytes", (value))
}

// GetDirectReadBytes gets the value of DirectReadBytes for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageHybridDisks) GetPropertyDirectReadBytes() (value uint64, err error) {
	retValue, err := instance.GetProperty("DirectReadBytes")
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

// SetDirectReadBytesPersec sets the value of DirectReadBytesPersec for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageHybridDisks) SetPropertyDirectReadBytesPersec(value uint64) (err error) {
	return instance.SetProperty("DirectReadBytesPersec", (value))
}

// GetDirectReadBytesPersec gets the value of DirectReadBytesPersec for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageHybridDisks) GetPropertyDirectReadBytesPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("DirectReadBytesPersec")
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

// SetDirectReads sets the value of DirectReads for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageHybridDisks) SetPropertyDirectReads(value uint64) (err error) {
	return instance.SetProperty("DirectReads", (value))
}

// GetDirectReads gets the value of DirectReads for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageHybridDisks) GetPropertyDirectReads() (value uint64, err error) {
	retValue, err := instance.GetProperty("DirectReads")
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

// SetDirectReadsPersec sets the value of DirectReadsPersec for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageHybridDisks) SetPropertyDirectReadsPersec(value uint64) (err error) {
	return instance.SetProperty("DirectReadsPersec", (value))
}

// GetDirectReadsPersec gets the value of DirectReadsPersec for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageHybridDisks) GetPropertyDirectReadsPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("DirectReadsPersec")
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

// SetDirectWriteBytes sets the value of DirectWriteBytes for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageHybridDisks) SetPropertyDirectWriteBytes(value uint64) (err error) {
	return instance.SetProperty("DirectWriteBytes", (value))
}

// GetDirectWriteBytes gets the value of DirectWriteBytes for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageHybridDisks) GetPropertyDirectWriteBytes() (value uint64, err error) {
	retValue, err := instance.GetProperty("DirectWriteBytes")
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

// SetDirectWriteBytesPersec sets the value of DirectWriteBytesPersec for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageHybridDisks) SetPropertyDirectWriteBytesPersec(value uint64) (err error) {
	return instance.SetProperty("DirectWriteBytesPersec", (value))
}

// GetDirectWriteBytesPersec gets the value of DirectWriteBytesPersec for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageHybridDisks) GetPropertyDirectWriteBytesPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("DirectWriteBytesPersec")
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

// SetDirectWrites sets the value of DirectWrites for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageHybridDisks) SetPropertyDirectWrites(value uint64) (err error) {
	return instance.SetProperty("DirectWrites", (value))
}

// GetDirectWrites gets the value of DirectWrites for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageHybridDisks) GetPropertyDirectWrites() (value uint64, err error) {
	retValue, err := instance.GetProperty("DirectWrites")
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

// SetDirectWritesPersec sets the value of DirectWritesPersec for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageHybridDisks) SetPropertyDirectWritesPersec(value uint64) (err error) {
	return instance.SetProperty("DirectWritesPersec", (value))
}

// GetDirectWritesPersec gets the value of DirectWritesPersec for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageHybridDisks) GetPropertyDirectWritesPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("DirectWritesPersec")
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

// SetDirtyReadBytes sets the value of DirtyReadBytes for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageHybridDisks) SetPropertyDirtyReadBytes(value uint64) (err error) {
	return instance.SetProperty("DirtyReadBytes", (value))
}

// GetDirtyReadBytes gets the value of DirtyReadBytes for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageHybridDisks) GetPropertyDirtyReadBytes() (value uint64, err error) {
	retValue, err := instance.GetProperty("DirtyReadBytes")
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

// SetDirtyReadBytesPersec sets the value of DirtyReadBytesPersec for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageHybridDisks) SetPropertyDirtyReadBytesPersec(value uint64) (err error) {
	return instance.SetProperty("DirtyReadBytesPersec", (value))
}

// GetDirtyReadBytesPersec gets the value of DirtyReadBytesPersec for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageHybridDisks) GetPropertyDirtyReadBytesPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("DirtyReadBytesPersec")
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

// SetDirtySlots sets the value of DirtySlots for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageHybridDisks) SetPropertyDirtySlots(value uint64) (err error) {
	return instance.SetProperty("DirtySlots", (value))
}

// GetDirtySlots gets the value of DirtySlots for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageHybridDisks) GetPropertyDirtySlots() (value uint64, err error) {
	retValue, err := instance.GetProperty("DirtySlots")
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

// SetDirtySlotsExpands sets the value of DirtySlotsExpands for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageHybridDisks) SetPropertyDirtySlotsExpands(value uint64) (err error) {
	return instance.SetProperty("DirtySlotsExpands", (value))
}

// GetDirtySlotsExpands gets the value of DirtySlotsExpands for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageHybridDisks) GetPropertyDirtySlotsExpands() (value uint64, err error) {
	retValue, err := instance.GetProperty("DirtySlotsExpands")
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

// SetDirtySlotsExpandsPersec sets the value of DirtySlotsExpandsPersec for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageHybridDisks) SetPropertyDirtySlotsExpandsPersec(value uint64) (err error) {
	return instance.SetProperty("DirtySlotsExpandsPersec", (value))
}

// GetDirtySlotsExpandsPersec gets the value of DirtySlotsExpandsPersec for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageHybridDisks) GetPropertyDirtySlotsExpandsPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("DirtySlotsExpandsPersec")
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

// SetDiskBytes sets the value of DiskBytes for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageHybridDisks) SetPropertyDiskBytes(value uint64) (err error) {
	return instance.SetProperty("DiskBytes", (value))
}

// GetDiskBytes gets the value of DiskBytes for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageHybridDisks) GetPropertyDiskBytes() (value uint64, err error) {
	retValue, err := instance.GetProperty("DiskBytes")
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

// SetDiskBytesPersec sets the value of DiskBytesPersec for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageHybridDisks) SetPropertyDiskBytesPersec(value uint64) (err error) {
	return instance.SetProperty("DiskBytesPersec", (value))
}

// GetDiskBytesPersec gets the value of DiskBytesPersec for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageHybridDisks) GetPropertyDiskBytesPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("DiskBytesPersec")
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

// SetDiskReadBytes sets the value of DiskReadBytes for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageHybridDisks) SetPropertyDiskReadBytes(value uint64) (err error) {
	return instance.SetProperty("DiskReadBytes", (value))
}

// GetDiskReadBytes gets the value of DiskReadBytes for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageHybridDisks) GetPropertyDiskReadBytes() (value uint64, err error) {
	retValue, err := instance.GetProperty("DiskReadBytes")
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

// SetDiskReadBytesPersec sets the value of DiskReadBytesPersec for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageHybridDisks) SetPropertyDiskReadBytesPersec(value uint64) (err error) {
	return instance.SetProperty("DiskReadBytesPersec", (value))
}

// GetDiskReadBytesPersec gets the value of DiskReadBytesPersec for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageHybridDisks) GetPropertyDiskReadBytesPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("DiskReadBytesPersec")
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

// SetDiskReads sets the value of DiskReads for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageHybridDisks) SetPropertyDiskReads(value uint64) (err error) {
	return instance.SetProperty("DiskReads", (value))
}

// GetDiskReads gets the value of DiskReads for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageHybridDisks) GetPropertyDiskReads() (value uint64, err error) {
	retValue, err := instance.GetProperty("DiskReads")
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

// SetDiskReadsPersec sets the value of DiskReadsPersec for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageHybridDisks) SetPropertyDiskReadsPersec(value uint64) (err error) {
	return instance.SetProperty("DiskReadsPersec", (value))
}

// GetDiskReadsPersec gets the value of DiskReadsPersec for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageHybridDisks) GetPropertyDiskReadsPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("DiskReadsPersec")
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

// SetDiskTransfers sets the value of DiskTransfers for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageHybridDisks) SetPropertyDiskTransfers(value uint64) (err error) {
	return instance.SetProperty("DiskTransfers", (value))
}

// GetDiskTransfers gets the value of DiskTransfers for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageHybridDisks) GetPropertyDiskTransfers() (value uint64, err error) {
	retValue, err := instance.GetProperty("DiskTransfers")
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

// SetDiskTransfersPersec sets the value of DiskTransfersPersec for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageHybridDisks) SetPropertyDiskTransfersPersec(value uint64) (err error) {
	return instance.SetProperty("DiskTransfersPersec", (value))
}

// GetDiskTransfersPersec gets the value of DiskTransfersPersec for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageHybridDisks) GetPropertyDiskTransfersPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("DiskTransfersPersec")
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

// SetDiskWriteBytes sets the value of DiskWriteBytes for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageHybridDisks) SetPropertyDiskWriteBytes(value uint64) (err error) {
	return instance.SetProperty("DiskWriteBytes", (value))
}

// GetDiskWriteBytes gets the value of DiskWriteBytes for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageHybridDisks) GetPropertyDiskWriteBytes() (value uint64, err error) {
	retValue, err := instance.GetProperty("DiskWriteBytes")
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

// SetDiskWriteBytesPersec sets the value of DiskWriteBytesPersec for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageHybridDisks) SetPropertyDiskWriteBytesPersec(value uint64) (err error) {
	return instance.SetProperty("DiskWriteBytesPersec", (value))
}

// GetDiskWriteBytesPersec gets the value of DiskWriteBytesPersec for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageHybridDisks) GetPropertyDiskWriteBytesPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("DiskWriteBytesPersec")
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

// SetDiskWrites sets the value of DiskWrites for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageHybridDisks) SetPropertyDiskWrites(value uint64) (err error) {
	return instance.SetProperty("DiskWrites", (value))
}

// GetDiskWrites gets the value of DiskWrites for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageHybridDisks) GetPropertyDiskWrites() (value uint64, err error) {
	retValue, err := instance.GetProperty("DiskWrites")
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

// SetDiskWritesPersec sets the value of DiskWritesPersec for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageHybridDisks) SetPropertyDiskWritesPersec(value uint64) (err error) {
	return instance.SetProperty("DiskWritesPersec", (value))
}

// GetDiskWritesPersec gets the value of DiskWritesPersec for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageHybridDisks) GetPropertyDiskWritesPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("DiskWritesPersec")
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

// SetMissingSlots sets the value of MissingSlots for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageHybridDisks) SetPropertyMissingSlots(value uint64) (err error) {
	return instance.SetProperty("MissingSlots", (value))
}

// GetMissingSlots gets the value of MissingSlots for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageHybridDisks) GetPropertyMissingSlots() (value uint64, err error) {
	retValue, err := instance.GetProperty("MissingSlots")
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

// SetRateDiskCacheReads sets the value of RateDiskCacheReads for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageHybridDisks) SetPropertyRateDiskCacheReads(value uint64) (err error) {
	return instance.SetProperty("RateDiskCacheReads", (value))
}

// GetRateDiskCacheReads gets the value of RateDiskCacheReads for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageHybridDisks) GetPropertyRateDiskCacheReads() (value uint64, err error) {
	retValue, err := instance.GetProperty("RateDiskCacheReads")
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

// SetRateDiskCacheReads_Base sets the value of RateDiskCacheReads_Base for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageHybridDisks) SetPropertyRateDiskCacheReads_Base(value uint32) (err error) {
	return instance.SetProperty("RateDiskCacheReads_Base", (value))
}

// GetRateDiskCacheReads_Base gets the value of RateDiskCacheReads_Base for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageHybridDisks) GetPropertyRateDiskCacheReads_Base() (value uint32, err error) {
	retValue, err := instance.GetProperty("RateDiskCacheReads_Base")
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

// SetRateDiskCacheWrites sets the value of RateDiskCacheWrites for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageHybridDisks) SetPropertyRateDiskCacheWrites(value uint64) (err error) {
	return instance.SetProperty("RateDiskCacheWrites", (value))
}

// GetRateDiskCacheWrites gets the value of RateDiskCacheWrites for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageHybridDisks) GetPropertyRateDiskCacheWrites() (value uint64, err error) {
	retValue, err := instance.GetProperty("RateDiskCacheWrites")
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

// SetRateDiskCacheWrites_Base sets the value of RateDiskCacheWrites_Base for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageHybridDisks) SetPropertyRateDiskCacheWrites_Base(value uint32) (err error) {
	return instance.SetProperty("RateDiskCacheWrites_Base", (value))
}

// GetRateDiskCacheWrites_Base gets the value of RateDiskCacheWrites_Base for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageHybridDisks) GetPropertyRateDiskCacheWrites_Base() (value uint32, err error) {
	retValue, err := instance.GetProperty("RateDiskCacheWrites_Base")
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

// SetReadErrorsMedia sets the value of ReadErrorsMedia for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageHybridDisks) SetPropertyReadErrorsMedia(value uint64) (err error) {
	return instance.SetProperty("ReadErrorsMedia", (value))
}

// GetReadErrorsMedia gets the value of ReadErrorsMedia for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageHybridDisks) GetPropertyReadErrorsMedia() (value uint64, err error) {
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
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageHybridDisks) SetPropertyReadErrorsTimeout(value uint64) (err error) {
	return instance.SetProperty("ReadErrorsTimeout", (value))
}

// GetReadErrorsTimeout gets the value of ReadErrorsTimeout for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageHybridDisks) GetPropertyReadErrorsTimeout() (value uint64, err error) {
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
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageHybridDisks) SetPropertyReadErrorsTotal(value uint64) (err error) {
	return instance.SetProperty("ReadErrorsTotal", (value))
}

// GetReadErrorsTotal gets the value of ReadErrorsTotal for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageHybridDisks) GetPropertyReadErrorsTotal() (value uint64, err error) {
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

// SetWriteErrorsMedia sets the value of WriteErrorsMedia for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageHybridDisks) SetPropertyWriteErrorsMedia(value uint64) (err error) {
	return instance.SetProperty("WriteErrorsMedia", (value))
}

// GetWriteErrorsMedia gets the value of WriteErrorsMedia for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageHybridDisks) GetPropertyWriteErrorsMedia() (value uint64, err error) {
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
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageHybridDisks) SetPropertyWriteErrorsTimeout(value uint64) (err error) {
	return instance.SetProperty("WriteErrorsTimeout", (value))
}

// GetWriteErrorsTimeout gets the value of WriteErrorsTimeout for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageHybridDisks) GetPropertyWriteErrorsTimeout() (value uint64, err error) {
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
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageHybridDisks) SetPropertyWriteErrorsTotal(value uint64) (err error) {
	return instance.SetProperty("WriteErrorsTotal", (value))
}

// GetWriteErrorsTotal gets the value of WriteErrorsTotal for the instance
func (instance *Win32_PerfRawData_ClusBfltPerfProvider_ClusterStorageHybridDisks) GetPropertyWriteErrorsTotal() (value uint64, err error) {
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
