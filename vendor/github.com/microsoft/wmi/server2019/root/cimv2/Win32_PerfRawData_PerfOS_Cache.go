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

// Win32_PerfRawData_PerfOS_Cache struct
type Win32_PerfRawData_PerfOS_Cache struct {
	*Win32_PerfRawData

	//
	AsyncCopyReadsPersec uint32

	//
	AsyncDataMapsPersec uint32

	//
	AsyncFastReadsPersec uint32

	//
	AsyncMDLReadsPersec uint32

	//
	AsyncPinReadsPersec uint32

	//
	CopyReadHitsPercent uint32

	//
	CopyReadHitsPercent_Base uint32

	//
	CopyReadsPersec uint32

	//
	DataFlushesPersec uint32

	//
	DataFlushPagesPersec uint32

	//
	DataMapHitsPercent uint32

	//
	DataMapHitsPercent_Base uint32

	//
	DataMapPinsPersec uint32

	//
	DataMapPinsPersec_Base uint32

	//
	DataMapsPersec uint32

	//
	DirtyPages uint64

	//
	DirtyPageThreshold uint64

	//
	FastReadNotPossiblesPersec uint32

	//
	FastReadResourceMissesPersec uint32

	//
	FastReadsPersec uint32

	//
	LazyWriteFlushesPersec uint32

	//
	LazyWritePagesPersec uint32

	//
	MDLReadHitsPercent uint32

	//
	MDLReadHitsPercent_Base uint32

	//
	MDLReadsPersec uint32

	//
	PinReadHitsPercent uint32

	//
	PinReadHitsPercent_Base uint32

	//
	PinReadsPersec uint32

	//
	ReadAheadsPersec uint32

	//
	SyncCopyReadsPersec uint32

	//
	SyncDataMapsPersec uint32

	//
	SyncFastReadsPersec uint32

	//
	SyncMDLReadsPersec uint32

	//
	SyncPinReadsPersec uint32
}

func NewWin32_PerfRawData_PerfOS_CacheEx1(instance *cim.WmiInstance) (newInstance *Win32_PerfRawData_PerfOS_Cache, err error) {
	tmp, err := NewWin32_PerfRawDataEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfRawData_PerfOS_Cache{
		Win32_PerfRawData: tmp,
	}
	return
}

func NewWin32_PerfRawData_PerfOS_CacheEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_PerfRawData_PerfOS_Cache, err error) {
	tmp, err := NewWin32_PerfRawDataEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfRawData_PerfOS_Cache{
		Win32_PerfRawData: tmp,
	}
	return
}

// SetAsyncCopyReadsPersec sets the value of AsyncCopyReadsPersec for the instance
func (instance *Win32_PerfRawData_PerfOS_Cache) SetPropertyAsyncCopyReadsPersec(value uint32) (err error) {
	return instance.SetProperty("AsyncCopyReadsPersec", (value))
}

// GetAsyncCopyReadsPersec gets the value of AsyncCopyReadsPersec for the instance
func (instance *Win32_PerfRawData_PerfOS_Cache) GetPropertyAsyncCopyReadsPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("AsyncCopyReadsPersec")
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

// SetAsyncDataMapsPersec sets the value of AsyncDataMapsPersec for the instance
func (instance *Win32_PerfRawData_PerfOS_Cache) SetPropertyAsyncDataMapsPersec(value uint32) (err error) {
	return instance.SetProperty("AsyncDataMapsPersec", (value))
}

// GetAsyncDataMapsPersec gets the value of AsyncDataMapsPersec for the instance
func (instance *Win32_PerfRawData_PerfOS_Cache) GetPropertyAsyncDataMapsPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("AsyncDataMapsPersec")
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

// SetAsyncFastReadsPersec sets the value of AsyncFastReadsPersec for the instance
func (instance *Win32_PerfRawData_PerfOS_Cache) SetPropertyAsyncFastReadsPersec(value uint32) (err error) {
	return instance.SetProperty("AsyncFastReadsPersec", (value))
}

// GetAsyncFastReadsPersec gets the value of AsyncFastReadsPersec for the instance
func (instance *Win32_PerfRawData_PerfOS_Cache) GetPropertyAsyncFastReadsPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("AsyncFastReadsPersec")
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

// SetAsyncMDLReadsPersec sets the value of AsyncMDLReadsPersec for the instance
func (instance *Win32_PerfRawData_PerfOS_Cache) SetPropertyAsyncMDLReadsPersec(value uint32) (err error) {
	return instance.SetProperty("AsyncMDLReadsPersec", (value))
}

// GetAsyncMDLReadsPersec gets the value of AsyncMDLReadsPersec for the instance
func (instance *Win32_PerfRawData_PerfOS_Cache) GetPropertyAsyncMDLReadsPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("AsyncMDLReadsPersec")
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

// SetAsyncPinReadsPersec sets the value of AsyncPinReadsPersec for the instance
func (instance *Win32_PerfRawData_PerfOS_Cache) SetPropertyAsyncPinReadsPersec(value uint32) (err error) {
	return instance.SetProperty("AsyncPinReadsPersec", (value))
}

// GetAsyncPinReadsPersec gets the value of AsyncPinReadsPersec for the instance
func (instance *Win32_PerfRawData_PerfOS_Cache) GetPropertyAsyncPinReadsPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("AsyncPinReadsPersec")
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

// SetCopyReadHitsPercent sets the value of CopyReadHitsPercent for the instance
func (instance *Win32_PerfRawData_PerfOS_Cache) SetPropertyCopyReadHitsPercent(value uint32) (err error) {
	return instance.SetProperty("CopyReadHitsPercent", (value))
}

// GetCopyReadHitsPercent gets the value of CopyReadHitsPercent for the instance
func (instance *Win32_PerfRawData_PerfOS_Cache) GetPropertyCopyReadHitsPercent() (value uint32, err error) {
	retValue, err := instance.GetProperty("CopyReadHitsPercent")
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

// SetCopyReadHitsPercent_Base sets the value of CopyReadHitsPercent_Base for the instance
func (instance *Win32_PerfRawData_PerfOS_Cache) SetPropertyCopyReadHitsPercent_Base(value uint32) (err error) {
	return instance.SetProperty("CopyReadHitsPercent_Base", (value))
}

// GetCopyReadHitsPercent_Base gets the value of CopyReadHitsPercent_Base for the instance
func (instance *Win32_PerfRawData_PerfOS_Cache) GetPropertyCopyReadHitsPercent_Base() (value uint32, err error) {
	retValue, err := instance.GetProperty("CopyReadHitsPercent_Base")
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

// SetCopyReadsPersec sets the value of CopyReadsPersec for the instance
func (instance *Win32_PerfRawData_PerfOS_Cache) SetPropertyCopyReadsPersec(value uint32) (err error) {
	return instance.SetProperty("CopyReadsPersec", (value))
}

// GetCopyReadsPersec gets the value of CopyReadsPersec for the instance
func (instance *Win32_PerfRawData_PerfOS_Cache) GetPropertyCopyReadsPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("CopyReadsPersec")
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

// SetDataFlushesPersec sets the value of DataFlushesPersec for the instance
func (instance *Win32_PerfRawData_PerfOS_Cache) SetPropertyDataFlushesPersec(value uint32) (err error) {
	return instance.SetProperty("DataFlushesPersec", (value))
}

// GetDataFlushesPersec gets the value of DataFlushesPersec for the instance
func (instance *Win32_PerfRawData_PerfOS_Cache) GetPropertyDataFlushesPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("DataFlushesPersec")
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

// SetDataFlushPagesPersec sets the value of DataFlushPagesPersec for the instance
func (instance *Win32_PerfRawData_PerfOS_Cache) SetPropertyDataFlushPagesPersec(value uint32) (err error) {
	return instance.SetProperty("DataFlushPagesPersec", (value))
}

// GetDataFlushPagesPersec gets the value of DataFlushPagesPersec for the instance
func (instance *Win32_PerfRawData_PerfOS_Cache) GetPropertyDataFlushPagesPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("DataFlushPagesPersec")
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

// SetDataMapHitsPercent sets the value of DataMapHitsPercent for the instance
func (instance *Win32_PerfRawData_PerfOS_Cache) SetPropertyDataMapHitsPercent(value uint32) (err error) {
	return instance.SetProperty("DataMapHitsPercent", (value))
}

// GetDataMapHitsPercent gets the value of DataMapHitsPercent for the instance
func (instance *Win32_PerfRawData_PerfOS_Cache) GetPropertyDataMapHitsPercent() (value uint32, err error) {
	retValue, err := instance.GetProperty("DataMapHitsPercent")
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

// SetDataMapHitsPercent_Base sets the value of DataMapHitsPercent_Base for the instance
func (instance *Win32_PerfRawData_PerfOS_Cache) SetPropertyDataMapHitsPercent_Base(value uint32) (err error) {
	return instance.SetProperty("DataMapHitsPercent_Base", (value))
}

// GetDataMapHitsPercent_Base gets the value of DataMapHitsPercent_Base for the instance
func (instance *Win32_PerfRawData_PerfOS_Cache) GetPropertyDataMapHitsPercent_Base() (value uint32, err error) {
	retValue, err := instance.GetProperty("DataMapHitsPercent_Base")
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

// SetDataMapPinsPersec sets the value of DataMapPinsPersec for the instance
func (instance *Win32_PerfRawData_PerfOS_Cache) SetPropertyDataMapPinsPersec(value uint32) (err error) {
	return instance.SetProperty("DataMapPinsPersec", (value))
}

// GetDataMapPinsPersec gets the value of DataMapPinsPersec for the instance
func (instance *Win32_PerfRawData_PerfOS_Cache) GetPropertyDataMapPinsPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("DataMapPinsPersec")
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

// SetDataMapPinsPersec_Base sets the value of DataMapPinsPersec_Base for the instance
func (instance *Win32_PerfRawData_PerfOS_Cache) SetPropertyDataMapPinsPersec_Base(value uint32) (err error) {
	return instance.SetProperty("DataMapPinsPersec_Base", (value))
}

// GetDataMapPinsPersec_Base gets the value of DataMapPinsPersec_Base for the instance
func (instance *Win32_PerfRawData_PerfOS_Cache) GetPropertyDataMapPinsPersec_Base() (value uint32, err error) {
	retValue, err := instance.GetProperty("DataMapPinsPersec_Base")
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

// SetDataMapsPersec sets the value of DataMapsPersec for the instance
func (instance *Win32_PerfRawData_PerfOS_Cache) SetPropertyDataMapsPersec(value uint32) (err error) {
	return instance.SetProperty("DataMapsPersec", (value))
}

// GetDataMapsPersec gets the value of DataMapsPersec for the instance
func (instance *Win32_PerfRawData_PerfOS_Cache) GetPropertyDataMapsPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("DataMapsPersec")
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

// SetDirtyPages sets the value of DirtyPages for the instance
func (instance *Win32_PerfRawData_PerfOS_Cache) SetPropertyDirtyPages(value uint64) (err error) {
	return instance.SetProperty("DirtyPages", (value))
}

// GetDirtyPages gets the value of DirtyPages for the instance
func (instance *Win32_PerfRawData_PerfOS_Cache) GetPropertyDirtyPages() (value uint64, err error) {
	retValue, err := instance.GetProperty("DirtyPages")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint64)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint64 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint64(valuetmp)

	return
}

// SetDirtyPageThreshold sets the value of DirtyPageThreshold for the instance
func (instance *Win32_PerfRawData_PerfOS_Cache) SetPropertyDirtyPageThreshold(value uint64) (err error) {
	return instance.SetProperty("DirtyPageThreshold", (value))
}

// GetDirtyPageThreshold gets the value of DirtyPageThreshold for the instance
func (instance *Win32_PerfRawData_PerfOS_Cache) GetPropertyDirtyPageThreshold() (value uint64, err error) {
	retValue, err := instance.GetProperty("DirtyPageThreshold")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint64)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint64 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint64(valuetmp)

	return
}

// SetFastReadNotPossiblesPersec sets the value of FastReadNotPossiblesPersec for the instance
func (instance *Win32_PerfRawData_PerfOS_Cache) SetPropertyFastReadNotPossiblesPersec(value uint32) (err error) {
	return instance.SetProperty("FastReadNotPossiblesPersec", (value))
}

// GetFastReadNotPossiblesPersec gets the value of FastReadNotPossiblesPersec for the instance
func (instance *Win32_PerfRawData_PerfOS_Cache) GetPropertyFastReadNotPossiblesPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("FastReadNotPossiblesPersec")
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

// SetFastReadResourceMissesPersec sets the value of FastReadResourceMissesPersec for the instance
func (instance *Win32_PerfRawData_PerfOS_Cache) SetPropertyFastReadResourceMissesPersec(value uint32) (err error) {
	return instance.SetProperty("FastReadResourceMissesPersec", (value))
}

// GetFastReadResourceMissesPersec gets the value of FastReadResourceMissesPersec for the instance
func (instance *Win32_PerfRawData_PerfOS_Cache) GetPropertyFastReadResourceMissesPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("FastReadResourceMissesPersec")
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

// SetFastReadsPersec sets the value of FastReadsPersec for the instance
func (instance *Win32_PerfRawData_PerfOS_Cache) SetPropertyFastReadsPersec(value uint32) (err error) {
	return instance.SetProperty("FastReadsPersec", (value))
}

// GetFastReadsPersec gets the value of FastReadsPersec for the instance
func (instance *Win32_PerfRawData_PerfOS_Cache) GetPropertyFastReadsPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("FastReadsPersec")
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

// SetLazyWriteFlushesPersec sets the value of LazyWriteFlushesPersec for the instance
func (instance *Win32_PerfRawData_PerfOS_Cache) SetPropertyLazyWriteFlushesPersec(value uint32) (err error) {
	return instance.SetProperty("LazyWriteFlushesPersec", (value))
}

// GetLazyWriteFlushesPersec gets the value of LazyWriteFlushesPersec for the instance
func (instance *Win32_PerfRawData_PerfOS_Cache) GetPropertyLazyWriteFlushesPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("LazyWriteFlushesPersec")
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

// SetLazyWritePagesPersec sets the value of LazyWritePagesPersec for the instance
func (instance *Win32_PerfRawData_PerfOS_Cache) SetPropertyLazyWritePagesPersec(value uint32) (err error) {
	return instance.SetProperty("LazyWritePagesPersec", (value))
}

// GetLazyWritePagesPersec gets the value of LazyWritePagesPersec for the instance
func (instance *Win32_PerfRawData_PerfOS_Cache) GetPropertyLazyWritePagesPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("LazyWritePagesPersec")
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

// SetMDLReadHitsPercent sets the value of MDLReadHitsPercent for the instance
func (instance *Win32_PerfRawData_PerfOS_Cache) SetPropertyMDLReadHitsPercent(value uint32) (err error) {
	return instance.SetProperty("MDLReadHitsPercent", (value))
}

// GetMDLReadHitsPercent gets the value of MDLReadHitsPercent for the instance
func (instance *Win32_PerfRawData_PerfOS_Cache) GetPropertyMDLReadHitsPercent() (value uint32, err error) {
	retValue, err := instance.GetProperty("MDLReadHitsPercent")
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

// SetMDLReadHitsPercent_Base sets the value of MDLReadHitsPercent_Base for the instance
func (instance *Win32_PerfRawData_PerfOS_Cache) SetPropertyMDLReadHitsPercent_Base(value uint32) (err error) {
	return instance.SetProperty("MDLReadHitsPercent_Base", (value))
}

// GetMDLReadHitsPercent_Base gets the value of MDLReadHitsPercent_Base for the instance
func (instance *Win32_PerfRawData_PerfOS_Cache) GetPropertyMDLReadHitsPercent_Base() (value uint32, err error) {
	retValue, err := instance.GetProperty("MDLReadHitsPercent_Base")
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

// SetMDLReadsPersec sets the value of MDLReadsPersec for the instance
func (instance *Win32_PerfRawData_PerfOS_Cache) SetPropertyMDLReadsPersec(value uint32) (err error) {
	return instance.SetProperty("MDLReadsPersec", (value))
}

// GetMDLReadsPersec gets the value of MDLReadsPersec for the instance
func (instance *Win32_PerfRawData_PerfOS_Cache) GetPropertyMDLReadsPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("MDLReadsPersec")
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

// SetPinReadHitsPercent sets the value of PinReadHitsPercent for the instance
func (instance *Win32_PerfRawData_PerfOS_Cache) SetPropertyPinReadHitsPercent(value uint32) (err error) {
	return instance.SetProperty("PinReadHitsPercent", (value))
}

// GetPinReadHitsPercent gets the value of PinReadHitsPercent for the instance
func (instance *Win32_PerfRawData_PerfOS_Cache) GetPropertyPinReadHitsPercent() (value uint32, err error) {
	retValue, err := instance.GetProperty("PinReadHitsPercent")
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

// SetPinReadHitsPercent_Base sets the value of PinReadHitsPercent_Base for the instance
func (instance *Win32_PerfRawData_PerfOS_Cache) SetPropertyPinReadHitsPercent_Base(value uint32) (err error) {
	return instance.SetProperty("PinReadHitsPercent_Base", (value))
}

// GetPinReadHitsPercent_Base gets the value of PinReadHitsPercent_Base for the instance
func (instance *Win32_PerfRawData_PerfOS_Cache) GetPropertyPinReadHitsPercent_Base() (value uint32, err error) {
	retValue, err := instance.GetProperty("PinReadHitsPercent_Base")
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

// SetPinReadsPersec sets the value of PinReadsPersec for the instance
func (instance *Win32_PerfRawData_PerfOS_Cache) SetPropertyPinReadsPersec(value uint32) (err error) {
	return instance.SetProperty("PinReadsPersec", (value))
}

// GetPinReadsPersec gets the value of PinReadsPersec for the instance
func (instance *Win32_PerfRawData_PerfOS_Cache) GetPropertyPinReadsPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("PinReadsPersec")
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

// SetReadAheadsPersec sets the value of ReadAheadsPersec for the instance
func (instance *Win32_PerfRawData_PerfOS_Cache) SetPropertyReadAheadsPersec(value uint32) (err error) {
	return instance.SetProperty("ReadAheadsPersec", (value))
}

// GetReadAheadsPersec gets the value of ReadAheadsPersec for the instance
func (instance *Win32_PerfRawData_PerfOS_Cache) GetPropertyReadAheadsPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("ReadAheadsPersec")
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

// SetSyncCopyReadsPersec sets the value of SyncCopyReadsPersec for the instance
func (instance *Win32_PerfRawData_PerfOS_Cache) SetPropertySyncCopyReadsPersec(value uint32) (err error) {
	return instance.SetProperty("SyncCopyReadsPersec", (value))
}

// GetSyncCopyReadsPersec gets the value of SyncCopyReadsPersec for the instance
func (instance *Win32_PerfRawData_PerfOS_Cache) GetPropertySyncCopyReadsPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("SyncCopyReadsPersec")
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

// SetSyncDataMapsPersec sets the value of SyncDataMapsPersec for the instance
func (instance *Win32_PerfRawData_PerfOS_Cache) SetPropertySyncDataMapsPersec(value uint32) (err error) {
	return instance.SetProperty("SyncDataMapsPersec", (value))
}

// GetSyncDataMapsPersec gets the value of SyncDataMapsPersec for the instance
func (instance *Win32_PerfRawData_PerfOS_Cache) GetPropertySyncDataMapsPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("SyncDataMapsPersec")
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

// SetSyncFastReadsPersec sets the value of SyncFastReadsPersec for the instance
func (instance *Win32_PerfRawData_PerfOS_Cache) SetPropertySyncFastReadsPersec(value uint32) (err error) {
	return instance.SetProperty("SyncFastReadsPersec", (value))
}

// GetSyncFastReadsPersec gets the value of SyncFastReadsPersec for the instance
func (instance *Win32_PerfRawData_PerfOS_Cache) GetPropertySyncFastReadsPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("SyncFastReadsPersec")
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

// SetSyncMDLReadsPersec sets the value of SyncMDLReadsPersec for the instance
func (instance *Win32_PerfRawData_PerfOS_Cache) SetPropertySyncMDLReadsPersec(value uint32) (err error) {
	return instance.SetProperty("SyncMDLReadsPersec", (value))
}

// GetSyncMDLReadsPersec gets the value of SyncMDLReadsPersec for the instance
func (instance *Win32_PerfRawData_PerfOS_Cache) GetPropertySyncMDLReadsPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("SyncMDLReadsPersec")
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

// SetSyncPinReadsPersec sets the value of SyncPinReadsPersec for the instance
func (instance *Win32_PerfRawData_PerfOS_Cache) SetPropertySyncPinReadsPersec(value uint32) (err error) {
	return instance.SetProperty("SyncPinReadsPersec", (value))
}

// GetSyncPinReadsPersec gets the value of SyncPinReadsPersec for the instance
func (instance *Win32_PerfRawData_PerfOS_Cache) GetPropertySyncPinReadsPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("SyncPinReadsPersec")
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
