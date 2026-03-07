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

// Win32_PerfRawData_PerfOS_Memory struct
type Win32_PerfRawData_PerfOS_Memory struct {
	*Win32_PerfRawData

	//
	AvailableBytes uint64

	//
	AvailableKBytes uint64

	//
	AvailableMBytes uint64

	//
	CacheBytes uint64

	//
	CacheBytesPeak uint64

	//
	CacheFaultsPersec uint32

	//
	CommitLimit uint64

	//
	CommittedBytes uint64

	//
	DemandZeroFaultsPersec uint32

	//
	FreeAndZeroPageListBytes uint64

	//
	FreeSystemPageTableEntries uint32

	//
	LongTermAverageStandbyCacheLifetimes uint32

	//
	ModifiedPageListBytes uint64

	//
	PageFaultsPersec uint32

	//
	PageReadsPersec uint32

	//
	PagesInputPersec uint32

	//
	PagesOutputPersec uint32

	//
	PagesPersec uint32

	//
	PageWritesPersec uint32

	//
	PercentCommittedBytesInUse uint32

	//
	PercentCommittedBytesInUse_Base uint32

	//
	PoolNonpagedAllocs uint32

	//
	PoolNonpagedBytes uint64

	//
	PoolPagedAllocs uint32

	//
	PoolPagedBytes uint64

	//
	PoolPagedResidentBytes uint64

	//
	StandbyCacheCoreBytes uint64

	//
	StandbyCacheNormalPriorityBytes uint64

	//
	StandbyCacheReserveBytes uint64

	//
	SystemCacheResidentBytes uint64

	//
	SystemCodeResidentBytes uint64

	//
	SystemCodeTotalBytes uint64

	//
	SystemDriverResidentBytes uint64

	//
	SystemDriverTotalBytes uint64

	//
	TransitionFaultsPersec uint32

	//
	TransitionPagesRePurposedPersec uint32

	//
	WriteCopiesPersec uint32
}

func NewWin32_PerfRawData_PerfOS_MemoryEx1(instance *cim.WmiInstance) (newInstance *Win32_PerfRawData_PerfOS_Memory, err error) {
	tmp, err := NewWin32_PerfRawDataEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfRawData_PerfOS_Memory{
		Win32_PerfRawData: tmp,
	}
	return
}

func NewWin32_PerfRawData_PerfOS_MemoryEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_PerfRawData_PerfOS_Memory, err error) {
	tmp, err := NewWin32_PerfRawDataEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfRawData_PerfOS_Memory{
		Win32_PerfRawData: tmp,
	}
	return
}

// SetAvailableBytes sets the value of AvailableBytes for the instance
func (instance *Win32_PerfRawData_PerfOS_Memory) SetPropertyAvailableBytes(value uint64) (err error) {
	return instance.SetProperty("AvailableBytes", (value))
}

// GetAvailableBytes gets the value of AvailableBytes for the instance
func (instance *Win32_PerfRawData_PerfOS_Memory) GetPropertyAvailableBytes() (value uint64, err error) {
	retValue, err := instance.GetProperty("AvailableBytes")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint64)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint64 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint64(valuetmp)

	return
}

// SetAvailableKBytes sets the value of AvailableKBytes for the instance
func (instance *Win32_PerfRawData_PerfOS_Memory) SetPropertyAvailableKBytes(value uint64) (err error) {
	return instance.SetProperty("AvailableKBytes", (value))
}

// GetAvailableKBytes gets the value of AvailableKBytes for the instance
func (instance *Win32_PerfRawData_PerfOS_Memory) GetPropertyAvailableKBytes() (value uint64, err error) {
	retValue, err := instance.GetProperty("AvailableKBytes")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint64)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint64 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint64(valuetmp)

	return
}

// SetAvailableMBytes sets the value of AvailableMBytes for the instance
func (instance *Win32_PerfRawData_PerfOS_Memory) SetPropertyAvailableMBytes(value uint64) (err error) {
	return instance.SetProperty("AvailableMBytes", (value))
}

// GetAvailableMBytes gets the value of AvailableMBytes for the instance
func (instance *Win32_PerfRawData_PerfOS_Memory) GetPropertyAvailableMBytes() (value uint64, err error) {
	retValue, err := instance.GetProperty("AvailableMBytes")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint64)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint64 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint64(valuetmp)

	return
}

// SetCacheBytes sets the value of CacheBytes for the instance
func (instance *Win32_PerfRawData_PerfOS_Memory) SetPropertyCacheBytes(value uint64) (err error) {
	return instance.SetProperty("CacheBytes", (value))
}

// GetCacheBytes gets the value of CacheBytes for the instance
func (instance *Win32_PerfRawData_PerfOS_Memory) GetPropertyCacheBytes() (value uint64, err error) {
	retValue, err := instance.GetProperty("CacheBytes")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint64)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint64 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint64(valuetmp)

	return
}

// SetCacheBytesPeak sets the value of CacheBytesPeak for the instance
func (instance *Win32_PerfRawData_PerfOS_Memory) SetPropertyCacheBytesPeak(value uint64) (err error) {
	return instance.SetProperty("CacheBytesPeak", (value))
}

// GetCacheBytesPeak gets the value of CacheBytesPeak for the instance
func (instance *Win32_PerfRawData_PerfOS_Memory) GetPropertyCacheBytesPeak() (value uint64, err error) {
	retValue, err := instance.GetProperty("CacheBytesPeak")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint64)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint64 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint64(valuetmp)

	return
}

// SetCacheFaultsPersec sets the value of CacheFaultsPersec for the instance
func (instance *Win32_PerfRawData_PerfOS_Memory) SetPropertyCacheFaultsPersec(value uint32) (err error) {
	return instance.SetProperty("CacheFaultsPersec", (value))
}

// GetCacheFaultsPersec gets the value of CacheFaultsPersec for the instance
func (instance *Win32_PerfRawData_PerfOS_Memory) GetPropertyCacheFaultsPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("CacheFaultsPersec")
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

// SetCommitLimit sets the value of CommitLimit for the instance
func (instance *Win32_PerfRawData_PerfOS_Memory) SetPropertyCommitLimit(value uint64) (err error) {
	return instance.SetProperty("CommitLimit", (value))
}

// GetCommitLimit gets the value of CommitLimit for the instance
func (instance *Win32_PerfRawData_PerfOS_Memory) GetPropertyCommitLimit() (value uint64, err error) {
	retValue, err := instance.GetProperty("CommitLimit")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint64)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint64 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint64(valuetmp)

	return
}

// SetCommittedBytes sets the value of CommittedBytes for the instance
func (instance *Win32_PerfRawData_PerfOS_Memory) SetPropertyCommittedBytes(value uint64) (err error) {
	return instance.SetProperty("CommittedBytes", (value))
}

// GetCommittedBytes gets the value of CommittedBytes for the instance
func (instance *Win32_PerfRawData_PerfOS_Memory) GetPropertyCommittedBytes() (value uint64, err error) {
	retValue, err := instance.GetProperty("CommittedBytes")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint64)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint64 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint64(valuetmp)

	return
}

// SetDemandZeroFaultsPersec sets the value of DemandZeroFaultsPersec for the instance
func (instance *Win32_PerfRawData_PerfOS_Memory) SetPropertyDemandZeroFaultsPersec(value uint32) (err error) {
	return instance.SetProperty("DemandZeroFaultsPersec", (value))
}

// GetDemandZeroFaultsPersec gets the value of DemandZeroFaultsPersec for the instance
func (instance *Win32_PerfRawData_PerfOS_Memory) GetPropertyDemandZeroFaultsPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("DemandZeroFaultsPersec")
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

// SetFreeAndZeroPageListBytes sets the value of FreeAndZeroPageListBytes for the instance
func (instance *Win32_PerfRawData_PerfOS_Memory) SetPropertyFreeAndZeroPageListBytes(value uint64) (err error) {
	return instance.SetProperty("FreeAndZeroPageListBytes", (value))
}

// GetFreeAndZeroPageListBytes gets the value of FreeAndZeroPageListBytes for the instance
func (instance *Win32_PerfRawData_PerfOS_Memory) GetPropertyFreeAndZeroPageListBytes() (value uint64, err error) {
	retValue, err := instance.GetProperty("FreeAndZeroPageListBytes")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint64)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint64 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint64(valuetmp)

	return
}

// SetFreeSystemPageTableEntries sets the value of FreeSystemPageTableEntries for the instance
func (instance *Win32_PerfRawData_PerfOS_Memory) SetPropertyFreeSystemPageTableEntries(value uint32) (err error) {
	return instance.SetProperty("FreeSystemPageTableEntries", (value))
}

// GetFreeSystemPageTableEntries gets the value of FreeSystemPageTableEntries for the instance
func (instance *Win32_PerfRawData_PerfOS_Memory) GetPropertyFreeSystemPageTableEntries() (value uint32, err error) {
	retValue, err := instance.GetProperty("FreeSystemPageTableEntries")
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

// SetLongTermAverageStandbyCacheLifetimes sets the value of LongTermAverageStandbyCacheLifetimes for the instance
func (instance *Win32_PerfRawData_PerfOS_Memory) SetPropertyLongTermAverageStandbyCacheLifetimes(value uint32) (err error) {
	return instance.SetProperty("LongTermAverageStandbyCacheLifetimes", (value))
}

// GetLongTermAverageStandbyCacheLifetimes gets the value of LongTermAverageStandbyCacheLifetimes for the instance
func (instance *Win32_PerfRawData_PerfOS_Memory) GetPropertyLongTermAverageStandbyCacheLifetimes() (value uint32, err error) {
	retValue, err := instance.GetProperty("LongTermAverageStandbyCacheLifetimes")
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

// SetModifiedPageListBytes sets the value of ModifiedPageListBytes for the instance
func (instance *Win32_PerfRawData_PerfOS_Memory) SetPropertyModifiedPageListBytes(value uint64) (err error) {
	return instance.SetProperty("ModifiedPageListBytes", (value))
}

// GetModifiedPageListBytes gets the value of ModifiedPageListBytes for the instance
func (instance *Win32_PerfRawData_PerfOS_Memory) GetPropertyModifiedPageListBytes() (value uint64, err error) {
	retValue, err := instance.GetProperty("ModifiedPageListBytes")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint64)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint64 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint64(valuetmp)

	return
}

// SetPageFaultsPersec sets the value of PageFaultsPersec for the instance
func (instance *Win32_PerfRawData_PerfOS_Memory) SetPropertyPageFaultsPersec(value uint32) (err error) {
	return instance.SetProperty("PageFaultsPersec", (value))
}

// GetPageFaultsPersec gets the value of PageFaultsPersec for the instance
func (instance *Win32_PerfRawData_PerfOS_Memory) GetPropertyPageFaultsPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("PageFaultsPersec")
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

// SetPageReadsPersec sets the value of PageReadsPersec for the instance
func (instance *Win32_PerfRawData_PerfOS_Memory) SetPropertyPageReadsPersec(value uint32) (err error) {
	return instance.SetProperty("PageReadsPersec", (value))
}

// GetPageReadsPersec gets the value of PageReadsPersec for the instance
func (instance *Win32_PerfRawData_PerfOS_Memory) GetPropertyPageReadsPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("PageReadsPersec")
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

// SetPagesInputPersec sets the value of PagesInputPersec for the instance
func (instance *Win32_PerfRawData_PerfOS_Memory) SetPropertyPagesInputPersec(value uint32) (err error) {
	return instance.SetProperty("PagesInputPersec", (value))
}

// GetPagesInputPersec gets the value of PagesInputPersec for the instance
func (instance *Win32_PerfRawData_PerfOS_Memory) GetPropertyPagesInputPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("PagesInputPersec")
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

// SetPagesOutputPersec sets the value of PagesOutputPersec for the instance
func (instance *Win32_PerfRawData_PerfOS_Memory) SetPropertyPagesOutputPersec(value uint32) (err error) {
	return instance.SetProperty("PagesOutputPersec", (value))
}

// GetPagesOutputPersec gets the value of PagesOutputPersec for the instance
func (instance *Win32_PerfRawData_PerfOS_Memory) GetPropertyPagesOutputPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("PagesOutputPersec")
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

// SetPagesPersec sets the value of PagesPersec for the instance
func (instance *Win32_PerfRawData_PerfOS_Memory) SetPropertyPagesPersec(value uint32) (err error) {
	return instance.SetProperty("PagesPersec", (value))
}

// GetPagesPersec gets the value of PagesPersec for the instance
func (instance *Win32_PerfRawData_PerfOS_Memory) GetPropertyPagesPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("PagesPersec")
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

// SetPageWritesPersec sets the value of PageWritesPersec for the instance
func (instance *Win32_PerfRawData_PerfOS_Memory) SetPropertyPageWritesPersec(value uint32) (err error) {
	return instance.SetProperty("PageWritesPersec", (value))
}

// GetPageWritesPersec gets the value of PageWritesPersec for the instance
func (instance *Win32_PerfRawData_PerfOS_Memory) GetPropertyPageWritesPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("PageWritesPersec")
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

// SetPercentCommittedBytesInUse sets the value of PercentCommittedBytesInUse for the instance
func (instance *Win32_PerfRawData_PerfOS_Memory) SetPropertyPercentCommittedBytesInUse(value uint32) (err error) {
	return instance.SetProperty("PercentCommittedBytesInUse", (value))
}

// GetPercentCommittedBytesInUse gets the value of PercentCommittedBytesInUse for the instance
func (instance *Win32_PerfRawData_PerfOS_Memory) GetPropertyPercentCommittedBytesInUse() (value uint32, err error) {
	retValue, err := instance.GetProperty("PercentCommittedBytesInUse")
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

// SetPercentCommittedBytesInUse_Base sets the value of PercentCommittedBytesInUse_Base for the instance
func (instance *Win32_PerfRawData_PerfOS_Memory) SetPropertyPercentCommittedBytesInUse_Base(value uint32) (err error) {
	return instance.SetProperty("PercentCommittedBytesInUse_Base", (value))
}

// GetPercentCommittedBytesInUse_Base gets the value of PercentCommittedBytesInUse_Base for the instance
func (instance *Win32_PerfRawData_PerfOS_Memory) GetPropertyPercentCommittedBytesInUse_Base() (value uint32, err error) {
	retValue, err := instance.GetProperty("PercentCommittedBytesInUse_Base")
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

// SetPoolNonpagedAllocs sets the value of PoolNonpagedAllocs for the instance
func (instance *Win32_PerfRawData_PerfOS_Memory) SetPropertyPoolNonpagedAllocs(value uint32) (err error) {
	return instance.SetProperty("PoolNonpagedAllocs", (value))
}

// GetPoolNonpagedAllocs gets the value of PoolNonpagedAllocs for the instance
func (instance *Win32_PerfRawData_PerfOS_Memory) GetPropertyPoolNonpagedAllocs() (value uint32, err error) {
	retValue, err := instance.GetProperty("PoolNonpagedAllocs")
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

// SetPoolNonpagedBytes sets the value of PoolNonpagedBytes for the instance
func (instance *Win32_PerfRawData_PerfOS_Memory) SetPropertyPoolNonpagedBytes(value uint64) (err error) {
	return instance.SetProperty("PoolNonpagedBytes", (value))
}

// GetPoolNonpagedBytes gets the value of PoolNonpagedBytes for the instance
func (instance *Win32_PerfRawData_PerfOS_Memory) GetPropertyPoolNonpagedBytes() (value uint64, err error) {
	retValue, err := instance.GetProperty("PoolNonpagedBytes")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint64)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint64 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint64(valuetmp)

	return
}

// SetPoolPagedAllocs sets the value of PoolPagedAllocs for the instance
func (instance *Win32_PerfRawData_PerfOS_Memory) SetPropertyPoolPagedAllocs(value uint32) (err error) {
	return instance.SetProperty("PoolPagedAllocs", (value))
}

// GetPoolPagedAllocs gets the value of PoolPagedAllocs for the instance
func (instance *Win32_PerfRawData_PerfOS_Memory) GetPropertyPoolPagedAllocs() (value uint32, err error) {
	retValue, err := instance.GetProperty("PoolPagedAllocs")
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

// SetPoolPagedBytes sets the value of PoolPagedBytes for the instance
func (instance *Win32_PerfRawData_PerfOS_Memory) SetPropertyPoolPagedBytes(value uint64) (err error) {
	return instance.SetProperty("PoolPagedBytes", (value))
}

// GetPoolPagedBytes gets the value of PoolPagedBytes for the instance
func (instance *Win32_PerfRawData_PerfOS_Memory) GetPropertyPoolPagedBytes() (value uint64, err error) {
	retValue, err := instance.GetProperty("PoolPagedBytes")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint64)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint64 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint64(valuetmp)

	return
}

// SetPoolPagedResidentBytes sets the value of PoolPagedResidentBytes for the instance
func (instance *Win32_PerfRawData_PerfOS_Memory) SetPropertyPoolPagedResidentBytes(value uint64) (err error) {
	return instance.SetProperty("PoolPagedResidentBytes", (value))
}

// GetPoolPagedResidentBytes gets the value of PoolPagedResidentBytes for the instance
func (instance *Win32_PerfRawData_PerfOS_Memory) GetPropertyPoolPagedResidentBytes() (value uint64, err error) {
	retValue, err := instance.GetProperty("PoolPagedResidentBytes")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint64)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint64 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint64(valuetmp)

	return
}

// SetStandbyCacheCoreBytes sets the value of StandbyCacheCoreBytes for the instance
func (instance *Win32_PerfRawData_PerfOS_Memory) SetPropertyStandbyCacheCoreBytes(value uint64) (err error) {
	return instance.SetProperty("StandbyCacheCoreBytes", (value))
}

// GetStandbyCacheCoreBytes gets the value of StandbyCacheCoreBytes for the instance
func (instance *Win32_PerfRawData_PerfOS_Memory) GetPropertyStandbyCacheCoreBytes() (value uint64, err error) {
	retValue, err := instance.GetProperty("StandbyCacheCoreBytes")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint64)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint64 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint64(valuetmp)

	return
}

// SetStandbyCacheNormalPriorityBytes sets the value of StandbyCacheNormalPriorityBytes for the instance
func (instance *Win32_PerfRawData_PerfOS_Memory) SetPropertyStandbyCacheNormalPriorityBytes(value uint64) (err error) {
	return instance.SetProperty("StandbyCacheNormalPriorityBytes", (value))
}

// GetStandbyCacheNormalPriorityBytes gets the value of StandbyCacheNormalPriorityBytes for the instance
func (instance *Win32_PerfRawData_PerfOS_Memory) GetPropertyStandbyCacheNormalPriorityBytes() (value uint64, err error) {
	retValue, err := instance.GetProperty("StandbyCacheNormalPriorityBytes")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint64)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint64 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint64(valuetmp)

	return
}

// SetStandbyCacheReserveBytes sets the value of StandbyCacheReserveBytes for the instance
func (instance *Win32_PerfRawData_PerfOS_Memory) SetPropertyStandbyCacheReserveBytes(value uint64) (err error) {
	return instance.SetProperty("StandbyCacheReserveBytes", (value))
}

// GetStandbyCacheReserveBytes gets the value of StandbyCacheReserveBytes for the instance
func (instance *Win32_PerfRawData_PerfOS_Memory) GetPropertyStandbyCacheReserveBytes() (value uint64, err error) {
	retValue, err := instance.GetProperty("StandbyCacheReserveBytes")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint64)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint64 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint64(valuetmp)

	return
}

// SetSystemCacheResidentBytes sets the value of SystemCacheResidentBytes for the instance
func (instance *Win32_PerfRawData_PerfOS_Memory) SetPropertySystemCacheResidentBytes(value uint64) (err error) {
	return instance.SetProperty("SystemCacheResidentBytes", (value))
}

// GetSystemCacheResidentBytes gets the value of SystemCacheResidentBytes for the instance
func (instance *Win32_PerfRawData_PerfOS_Memory) GetPropertySystemCacheResidentBytes() (value uint64, err error) {
	retValue, err := instance.GetProperty("SystemCacheResidentBytes")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint64)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint64 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint64(valuetmp)

	return
}

// SetSystemCodeResidentBytes sets the value of SystemCodeResidentBytes for the instance
func (instance *Win32_PerfRawData_PerfOS_Memory) SetPropertySystemCodeResidentBytes(value uint64) (err error) {
	return instance.SetProperty("SystemCodeResidentBytes", (value))
}

// GetSystemCodeResidentBytes gets the value of SystemCodeResidentBytes for the instance
func (instance *Win32_PerfRawData_PerfOS_Memory) GetPropertySystemCodeResidentBytes() (value uint64, err error) {
	retValue, err := instance.GetProperty("SystemCodeResidentBytes")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint64)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint64 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint64(valuetmp)

	return
}

// SetSystemCodeTotalBytes sets the value of SystemCodeTotalBytes for the instance
func (instance *Win32_PerfRawData_PerfOS_Memory) SetPropertySystemCodeTotalBytes(value uint64) (err error) {
	return instance.SetProperty("SystemCodeTotalBytes", (value))
}

// GetSystemCodeTotalBytes gets the value of SystemCodeTotalBytes for the instance
func (instance *Win32_PerfRawData_PerfOS_Memory) GetPropertySystemCodeTotalBytes() (value uint64, err error) {
	retValue, err := instance.GetProperty("SystemCodeTotalBytes")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint64)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint64 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint64(valuetmp)

	return
}

// SetSystemDriverResidentBytes sets the value of SystemDriverResidentBytes for the instance
func (instance *Win32_PerfRawData_PerfOS_Memory) SetPropertySystemDriverResidentBytes(value uint64) (err error) {
	return instance.SetProperty("SystemDriverResidentBytes", (value))
}

// GetSystemDriverResidentBytes gets the value of SystemDriverResidentBytes for the instance
func (instance *Win32_PerfRawData_PerfOS_Memory) GetPropertySystemDriverResidentBytes() (value uint64, err error) {
	retValue, err := instance.GetProperty("SystemDriverResidentBytes")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint64)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint64 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint64(valuetmp)

	return
}

// SetSystemDriverTotalBytes sets the value of SystemDriverTotalBytes for the instance
func (instance *Win32_PerfRawData_PerfOS_Memory) SetPropertySystemDriverTotalBytes(value uint64) (err error) {
	return instance.SetProperty("SystemDriverTotalBytes", (value))
}

// GetSystemDriverTotalBytes gets the value of SystemDriverTotalBytes for the instance
func (instance *Win32_PerfRawData_PerfOS_Memory) GetPropertySystemDriverTotalBytes() (value uint64, err error) {
	retValue, err := instance.GetProperty("SystemDriverTotalBytes")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint64)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint64 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint64(valuetmp)

	return
}

// SetTransitionFaultsPersec sets the value of TransitionFaultsPersec for the instance
func (instance *Win32_PerfRawData_PerfOS_Memory) SetPropertyTransitionFaultsPersec(value uint32) (err error) {
	return instance.SetProperty("TransitionFaultsPersec", (value))
}

// GetTransitionFaultsPersec gets the value of TransitionFaultsPersec for the instance
func (instance *Win32_PerfRawData_PerfOS_Memory) GetPropertyTransitionFaultsPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("TransitionFaultsPersec")
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

// SetTransitionPagesRePurposedPersec sets the value of TransitionPagesRePurposedPersec for the instance
func (instance *Win32_PerfRawData_PerfOS_Memory) SetPropertyTransitionPagesRePurposedPersec(value uint32) (err error) {
	return instance.SetProperty("TransitionPagesRePurposedPersec", (value))
}

// GetTransitionPagesRePurposedPersec gets the value of TransitionPagesRePurposedPersec for the instance
func (instance *Win32_PerfRawData_PerfOS_Memory) GetPropertyTransitionPagesRePurposedPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("TransitionPagesRePurposedPersec")
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

// SetWriteCopiesPersec sets the value of WriteCopiesPersec for the instance
func (instance *Win32_PerfRawData_PerfOS_Memory) SetPropertyWriteCopiesPersec(value uint32) (err error) {
	return instance.SetProperty("WriteCopiesPersec", (value))
}

// GetWriteCopiesPersec gets the value of WriteCopiesPersec for the instance
func (instance *Win32_PerfRawData_PerfOS_Memory) GetPropertyWriteCopiesPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("WriteCopiesPersec")
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
