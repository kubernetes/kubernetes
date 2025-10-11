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

// Win32_PerfRawData_ESENT_Database struct
type Win32_PerfRawData_ESENT_Database struct {
	*Win32_PerfRawData

	//
	DatabaseCacheMemoryCommitted uint64

	//
	DatabaseCacheMemoryCommittedMB uint64

	//
	DatabaseCacheMemoryReserved uint64

	//
	DatabaseCacheMemoryReservedMB uint64

	//
	DatabaseCacheMissAttachedAverageLatency uint64

	//
	DatabaseCacheMissAttachedAverageLatency_Base uint32

	//
	DatabaseCacheMissesPersec uint32

	//
	DatabaseCachePercentDehydrated uint32

	//
	DatabaseCachePercentDehydrated_Base uint32

	//
	DatabaseCachePercentHit uint32

	//
	DatabaseCachePercentHit_Base uint32

	//
	DatabaseCachePercentHitUnique uint32

	//
	DatabaseCachePercentHitUnique_Base uint32

	//
	DatabaseCacheRequestsPersec uint32

	//
	DatabaseCacheRequestsPersecUnique uint32

	//
	DatabaseCacheSize uint64

	//
	DatabaseCacheSizeEffective uint64

	//
	DatabaseCacheSizeEffectiveMB uint64

	//
	DatabaseCacheSizeMB uint64

	//
	DatabaseCacheSizeResident uint64

	//
	DatabaseCacheSizeResidentMB uint64

	//
	DatabaseMaintenanceDuration uint32

	//
	DatabasePageEvictionsPersec uint32

	//
	DatabasePageFaultsPersec uint32

	//
	DatabasePageFaultStallsPersec uint32

	//
	DefragmentationTasks uint32

	//
	DefragmentationTasksPending uint32

	//
	IODatabaseReadsAttachedAverageLatency uint64

	//
	IODatabaseReadsAttachedAverageLatency_Base uint32

	//
	IODatabaseReadsAttachedPersec uint32

	//
	IODatabaseReadsAverageLatency uint64

	//
	IODatabaseReadsAverageLatency_Base uint32

	//
	IODatabaseReadsPersec uint32

	//
	IODatabaseReadsRecoveryAverageLatency uint64

	//
	IODatabaseReadsRecoveryAverageLatency_Base uint32

	//
	IODatabaseReadsRecoveryPersec uint32

	//
	IODatabaseWritesAttachedAverageLatency uint64

	//
	IODatabaseWritesAttachedAverageLatency_Base uint32

	//
	IODatabaseWritesAttachedPersec uint32

	//
	IODatabaseWritesAverageLatency uint64

	//
	IODatabaseWritesAverageLatency_Base uint32

	//
	IODatabaseWritesPersec uint32

	//
	IODatabaseWritesRecoveryAverageLatency uint64

	//
	IODatabaseWritesRecoveryAverageLatency_Base uint32

	//
	IODatabaseWritesRecoveryPersec uint32

	//
	IOFlushMapWritesAverageLatency uint64

	//
	IOFlushMapWritesAverageLatency_Base uint32

	//
	IOFlushMapWritesPersec uint32

	//
	IOLogReadsAverageLatency uint64

	//
	IOLogReadsAverageLatency_Base uint32

	//
	IOLogReadsPersec uint32

	//
	IOLogWritesAverageLatency uint64

	//
	IOLogWritesAverageLatency_Base uint32

	//
	IOLogWritesPersec uint32

	//
	LogBytesGeneratedPersec uint32

	//
	LogBytesWritePersec uint32

	//
	LogRecordStallsPersec uint32

	//
	LogThreadsWaiting uint32

	//
	LogWritesPersec uint32

	//
	SessionsInUse uint32

	//
	SessionsPercentUsed uint32

	//
	SessionsPercentUsed_Base uint32

	//
	TableClosesPersec uint32

	//
	TableOpenCacheHitsPersec uint32

	//
	TableOpenCacheMissesPersec uint32

	//
	TableOpenCachePercentHit uint32

	//
	TableOpenCachePercentHit_Base uint32

	//
	TableOpensPersec uint32

	//
	TablesOpen uint32

	//
	VersionBucketsAllocated uint32
}

func NewWin32_PerfRawData_ESENT_DatabaseEx1(instance *cim.WmiInstance) (newInstance *Win32_PerfRawData_ESENT_Database, err error) {
	tmp, err := NewWin32_PerfRawDataEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfRawData_ESENT_Database{
		Win32_PerfRawData: tmp,
	}
	return
}

func NewWin32_PerfRawData_ESENT_DatabaseEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_PerfRawData_ESENT_Database, err error) {
	tmp, err := NewWin32_PerfRawDataEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfRawData_ESENT_Database{
		Win32_PerfRawData: tmp,
	}
	return
}

// SetDatabaseCacheMemoryCommitted sets the value of DatabaseCacheMemoryCommitted for the instance
func (instance *Win32_PerfRawData_ESENT_Database) SetPropertyDatabaseCacheMemoryCommitted(value uint64) (err error) {
	return instance.SetProperty("DatabaseCacheMemoryCommitted", (value))
}

// GetDatabaseCacheMemoryCommitted gets the value of DatabaseCacheMemoryCommitted for the instance
func (instance *Win32_PerfRawData_ESENT_Database) GetPropertyDatabaseCacheMemoryCommitted() (value uint64, err error) {
	retValue, err := instance.GetProperty("DatabaseCacheMemoryCommitted")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint64)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint64 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint64(valuetmp)

	return
}

// SetDatabaseCacheMemoryCommittedMB sets the value of DatabaseCacheMemoryCommittedMB for the instance
func (instance *Win32_PerfRawData_ESENT_Database) SetPropertyDatabaseCacheMemoryCommittedMB(value uint64) (err error) {
	return instance.SetProperty("DatabaseCacheMemoryCommittedMB", (value))
}

// GetDatabaseCacheMemoryCommittedMB gets the value of DatabaseCacheMemoryCommittedMB for the instance
func (instance *Win32_PerfRawData_ESENT_Database) GetPropertyDatabaseCacheMemoryCommittedMB() (value uint64, err error) {
	retValue, err := instance.GetProperty("DatabaseCacheMemoryCommittedMB")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint64)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint64 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint64(valuetmp)

	return
}

// SetDatabaseCacheMemoryReserved sets the value of DatabaseCacheMemoryReserved for the instance
func (instance *Win32_PerfRawData_ESENT_Database) SetPropertyDatabaseCacheMemoryReserved(value uint64) (err error) {
	return instance.SetProperty("DatabaseCacheMemoryReserved", (value))
}

// GetDatabaseCacheMemoryReserved gets the value of DatabaseCacheMemoryReserved for the instance
func (instance *Win32_PerfRawData_ESENT_Database) GetPropertyDatabaseCacheMemoryReserved() (value uint64, err error) {
	retValue, err := instance.GetProperty("DatabaseCacheMemoryReserved")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint64)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint64 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint64(valuetmp)

	return
}

// SetDatabaseCacheMemoryReservedMB sets the value of DatabaseCacheMemoryReservedMB for the instance
func (instance *Win32_PerfRawData_ESENT_Database) SetPropertyDatabaseCacheMemoryReservedMB(value uint64) (err error) {
	return instance.SetProperty("DatabaseCacheMemoryReservedMB", (value))
}

// GetDatabaseCacheMemoryReservedMB gets the value of DatabaseCacheMemoryReservedMB for the instance
func (instance *Win32_PerfRawData_ESENT_Database) GetPropertyDatabaseCacheMemoryReservedMB() (value uint64, err error) {
	retValue, err := instance.GetProperty("DatabaseCacheMemoryReservedMB")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint64)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint64 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint64(valuetmp)

	return
}

// SetDatabaseCacheMissAttachedAverageLatency sets the value of DatabaseCacheMissAttachedAverageLatency for the instance
func (instance *Win32_PerfRawData_ESENT_Database) SetPropertyDatabaseCacheMissAttachedAverageLatency(value uint64) (err error) {
	return instance.SetProperty("DatabaseCacheMissAttachedAverageLatency", (value))
}

// GetDatabaseCacheMissAttachedAverageLatency gets the value of DatabaseCacheMissAttachedAverageLatency for the instance
func (instance *Win32_PerfRawData_ESENT_Database) GetPropertyDatabaseCacheMissAttachedAverageLatency() (value uint64, err error) {
	retValue, err := instance.GetProperty("DatabaseCacheMissAttachedAverageLatency")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint64)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint64 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint64(valuetmp)

	return
}

// SetDatabaseCacheMissAttachedAverageLatency_Base sets the value of DatabaseCacheMissAttachedAverageLatency_Base for the instance
func (instance *Win32_PerfRawData_ESENT_Database) SetPropertyDatabaseCacheMissAttachedAverageLatency_Base(value uint32) (err error) {
	return instance.SetProperty("DatabaseCacheMissAttachedAverageLatency_Base", (value))
}

// GetDatabaseCacheMissAttachedAverageLatency_Base gets the value of DatabaseCacheMissAttachedAverageLatency_Base for the instance
func (instance *Win32_PerfRawData_ESENT_Database) GetPropertyDatabaseCacheMissAttachedAverageLatency_Base() (value uint32, err error) {
	retValue, err := instance.GetProperty("DatabaseCacheMissAttachedAverageLatency_Base")
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

// SetDatabaseCacheMissesPersec sets the value of DatabaseCacheMissesPersec for the instance
func (instance *Win32_PerfRawData_ESENT_Database) SetPropertyDatabaseCacheMissesPersec(value uint32) (err error) {
	return instance.SetProperty("DatabaseCacheMissesPersec", (value))
}

// GetDatabaseCacheMissesPersec gets the value of DatabaseCacheMissesPersec for the instance
func (instance *Win32_PerfRawData_ESENT_Database) GetPropertyDatabaseCacheMissesPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("DatabaseCacheMissesPersec")
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

// SetDatabaseCachePercentDehydrated sets the value of DatabaseCachePercentDehydrated for the instance
func (instance *Win32_PerfRawData_ESENT_Database) SetPropertyDatabaseCachePercentDehydrated(value uint32) (err error) {
	return instance.SetProperty("DatabaseCachePercentDehydrated", (value))
}

// GetDatabaseCachePercentDehydrated gets the value of DatabaseCachePercentDehydrated for the instance
func (instance *Win32_PerfRawData_ESENT_Database) GetPropertyDatabaseCachePercentDehydrated() (value uint32, err error) {
	retValue, err := instance.GetProperty("DatabaseCachePercentDehydrated")
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

// SetDatabaseCachePercentDehydrated_Base sets the value of DatabaseCachePercentDehydrated_Base for the instance
func (instance *Win32_PerfRawData_ESENT_Database) SetPropertyDatabaseCachePercentDehydrated_Base(value uint32) (err error) {
	return instance.SetProperty("DatabaseCachePercentDehydrated_Base", (value))
}

// GetDatabaseCachePercentDehydrated_Base gets the value of DatabaseCachePercentDehydrated_Base for the instance
func (instance *Win32_PerfRawData_ESENT_Database) GetPropertyDatabaseCachePercentDehydrated_Base() (value uint32, err error) {
	retValue, err := instance.GetProperty("DatabaseCachePercentDehydrated_Base")
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

// SetDatabaseCachePercentHit sets the value of DatabaseCachePercentHit for the instance
func (instance *Win32_PerfRawData_ESENT_Database) SetPropertyDatabaseCachePercentHit(value uint32) (err error) {
	return instance.SetProperty("DatabaseCachePercentHit", (value))
}

// GetDatabaseCachePercentHit gets the value of DatabaseCachePercentHit for the instance
func (instance *Win32_PerfRawData_ESENT_Database) GetPropertyDatabaseCachePercentHit() (value uint32, err error) {
	retValue, err := instance.GetProperty("DatabaseCachePercentHit")
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

// SetDatabaseCachePercentHit_Base sets the value of DatabaseCachePercentHit_Base for the instance
func (instance *Win32_PerfRawData_ESENT_Database) SetPropertyDatabaseCachePercentHit_Base(value uint32) (err error) {
	return instance.SetProperty("DatabaseCachePercentHit_Base", (value))
}

// GetDatabaseCachePercentHit_Base gets the value of DatabaseCachePercentHit_Base for the instance
func (instance *Win32_PerfRawData_ESENT_Database) GetPropertyDatabaseCachePercentHit_Base() (value uint32, err error) {
	retValue, err := instance.GetProperty("DatabaseCachePercentHit_Base")
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

// SetDatabaseCachePercentHitUnique sets the value of DatabaseCachePercentHitUnique for the instance
func (instance *Win32_PerfRawData_ESENT_Database) SetPropertyDatabaseCachePercentHitUnique(value uint32) (err error) {
	return instance.SetProperty("DatabaseCachePercentHitUnique", (value))
}

// GetDatabaseCachePercentHitUnique gets the value of DatabaseCachePercentHitUnique for the instance
func (instance *Win32_PerfRawData_ESENT_Database) GetPropertyDatabaseCachePercentHitUnique() (value uint32, err error) {
	retValue, err := instance.GetProperty("DatabaseCachePercentHitUnique")
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

// SetDatabaseCachePercentHitUnique_Base sets the value of DatabaseCachePercentHitUnique_Base for the instance
func (instance *Win32_PerfRawData_ESENT_Database) SetPropertyDatabaseCachePercentHitUnique_Base(value uint32) (err error) {
	return instance.SetProperty("DatabaseCachePercentHitUnique_Base", (value))
}

// GetDatabaseCachePercentHitUnique_Base gets the value of DatabaseCachePercentHitUnique_Base for the instance
func (instance *Win32_PerfRawData_ESENT_Database) GetPropertyDatabaseCachePercentHitUnique_Base() (value uint32, err error) {
	retValue, err := instance.GetProperty("DatabaseCachePercentHitUnique_Base")
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

// SetDatabaseCacheRequestsPersec sets the value of DatabaseCacheRequestsPersec for the instance
func (instance *Win32_PerfRawData_ESENT_Database) SetPropertyDatabaseCacheRequestsPersec(value uint32) (err error) {
	return instance.SetProperty("DatabaseCacheRequestsPersec", (value))
}

// GetDatabaseCacheRequestsPersec gets the value of DatabaseCacheRequestsPersec for the instance
func (instance *Win32_PerfRawData_ESENT_Database) GetPropertyDatabaseCacheRequestsPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("DatabaseCacheRequestsPersec")
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

// SetDatabaseCacheRequestsPersecUnique sets the value of DatabaseCacheRequestsPersecUnique for the instance
func (instance *Win32_PerfRawData_ESENT_Database) SetPropertyDatabaseCacheRequestsPersecUnique(value uint32) (err error) {
	return instance.SetProperty("DatabaseCacheRequestsPersecUnique", (value))
}

// GetDatabaseCacheRequestsPersecUnique gets the value of DatabaseCacheRequestsPersecUnique for the instance
func (instance *Win32_PerfRawData_ESENT_Database) GetPropertyDatabaseCacheRequestsPersecUnique() (value uint32, err error) {
	retValue, err := instance.GetProperty("DatabaseCacheRequestsPersecUnique")
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

// SetDatabaseCacheSize sets the value of DatabaseCacheSize for the instance
func (instance *Win32_PerfRawData_ESENT_Database) SetPropertyDatabaseCacheSize(value uint64) (err error) {
	return instance.SetProperty("DatabaseCacheSize", (value))
}

// GetDatabaseCacheSize gets the value of DatabaseCacheSize for the instance
func (instance *Win32_PerfRawData_ESENT_Database) GetPropertyDatabaseCacheSize() (value uint64, err error) {
	retValue, err := instance.GetProperty("DatabaseCacheSize")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint64)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint64 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint64(valuetmp)

	return
}

// SetDatabaseCacheSizeEffective sets the value of DatabaseCacheSizeEffective for the instance
func (instance *Win32_PerfRawData_ESENT_Database) SetPropertyDatabaseCacheSizeEffective(value uint64) (err error) {
	return instance.SetProperty("DatabaseCacheSizeEffective", (value))
}

// GetDatabaseCacheSizeEffective gets the value of DatabaseCacheSizeEffective for the instance
func (instance *Win32_PerfRawData_ESENT_Database) GetPropertyDatabaseCacheSizeEffective() (value uint64, err error) {
	retValue, err := instance.GetProperty("DatabaseCacheSizeEffective")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint64)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint64 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint64(valuetmp)

	return
}

// SetDatabaseCacheSizeEffectiveMB sets the value of DatabaseCacheSizeEffectiveMB for the instance
func (instance *Win32_PerfRawData_ESENT_Database) SetPropertyDatabaseCacheSizeEffectiveMB(value uint64) (err error) {
	return instance.SetProperty("DatabaseCacheSizeEffectiveMB", (value))
}

// GetDatabaseCacheSizeEffectiveMB gets the value of DatabaseCacheSizeEffectiveMB for the instance
func (instance *Win32_PerfRawData_ESENT_Database) GetPropertyDatabaseCacheSizeEffectiveMB() (value uint64, err error) {
	retValue, err := instance.GetProperty("DatabaseCacheSizeEffectiveMB")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint64)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint64 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint64(valuetmp)

	return
}

// SetDatabaseCacheSizeMB sets the value of DatabaseCacheSizeMB for the instance
func (instance *Win32_PerfRawData_ESENT_Database) SetPropertyDatabaseCacheSizeMB(value uint64) (err error) {
	return instance.SetProperty("DatabaseCacheSizeMB", (value))
}

// GetDatabaseCacheSizeMB gets the value of DatabaseCacheSizeMB for the instance
func (instance *Win32_PerfRawData_ESENT_Database) GetPropertyDatabaseCacheSizeMB() (value uint64, err error) {
	retValue, err := instance.GetProperty("DatabaseCacheSizeMB")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint64)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint64 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint64(valuetmp)

	return
}

// SetDatabaseCacheSizeResident sets the value of DatabaseCacheSizeResident for the instance
func (instance *Win32_PerfRawData_ESENT_Database) SetPropertyDatabaseCacheSizeResident(value uint64) (err error) {
	return instance.SetProperty("DatabaseCacheSizeResident", (value))
}

// GetDatabaseCacheSizeResident gets the value of DatabaseCacheSizeResident for the instance
func (instance *Win32_PerfRawData_ESENT_Database) GetPropertyDatabaseCacheSizeResident() (value uint64, err error) {
	retValue, err := instance.GetProperty("DatabaseCacheSizeResident")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint64)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint64 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint64(valuetmp)

	return
}

// SetDatabaseCacheSizeResidentMB sets the value of DatabaseCacheSizeResidentMB for the instance
func (instance *Win32_PerfRawData_ESENT_Database) SetPropertyDatabaseCacheSizeResidentMB(value uint64) (err error) {
	return instance.SetProperty("DatabaseCacheSizeResidentMB", (value))
}

// GetDatabaseCacheSizeResidentMB gets the value of DatabaseCacheSizeResidentMB for the instance
func (instance *Win32_PerfRawData_ESENT_Database) GetPropertyDatabaseCacheSizeResidentMB() (value uint64, err error) {
	retValue, err := instance.GetProperty("DatabaseCacheSizeResidentMB")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint64)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint64 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint64(valuetmp)

	return
}

// SetDatabaseMaintenanceDuration sets the value of DatabaseMaintenanceDuration for the instance
func (instance *Win32_PerfRawData_ESENT_Database) SetPropertyDatabaseMaintenanceDuration(value uint32) (err error) {
	return instance.SetProperty("DatabaseMaintenanceDuration", (value))
}

// GetDatabaseMaintenanceDuration gets the value of DatabaseMaintenanceDuration for the instance
func (instance *Win32_PerfRawData_ESENT_Database) GetPropertyDatabaseMaintenanceDuration() (value uint32, err error) {
	retValue, err := instance.GetProperty("DatabaseMaintenanceDuration")
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

// SetDatabasePageEvictionsPersec sets the value of DatabasePageEvictionsPersec for the instance
func (instance *Win32_PerfRawData_ESENT_Database) SetPropertyDatabasePageEvictionsPersec(value uint32) (err error) {
	return instance.SetProperty("DatabasePageEvictionsPersec", (value))
}

// GetDatabasePageEvictionsPersec gets the value of DatabasePageEvictionsPersec for the instance
func (instance *Win32_PerfRawData_ESENT_Database) GetPropertyDatabasePageEvictionsPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("DatabasePageEvictionsPersec")
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

// SetDatabasePageFaultsPersec sets the value of DatabasePageFaultsPersec for the instance
func (instance *Win32_PerfRawData_ESENT_Database) SetPropertyDatabasePageFaultsPersec(value uint32) (err error) {
	return instance.SetProperty("DatabasePageFaultsPersec", (value))
}

// GetDatabasePageFaultsPersec gets the value of DatabasePageFaultsPersec for the instance
func (instance *Win32_PerfRawData_ESENT_Database) GetPropertyDatabasePageFaultsPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("DatabasePageFaultsPersec")
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

// SetDatabasePageFaultStallsPersec sets the value of DatabasePageFaultStallsPersec for the instance
func (instance *Win32_PerfRawData_ESENT_Database) SetPropertyDatabasePageFaultStallsPersec(value uint32) (err error) {
	return instance.SetProperty("DatabasePageFaultStallsPersec", (value))
}

// GetDatabasePageFaultStallsPersec gets the value of DatabasePageFaultStallsPersec for the instance
func (instance *Win32_PerfRawData_ESENT_Database) GetPropertyDatabasePageFaultStallsPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("DatabasePageFaultStallsPersec")
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

// SetDefragmentationTasks sets the value of DefragmentationTasks for the instance
func (instance *Win32_PerfRawData_ESENT_Database) SetPropertyDefragmentationTasks(value uint32) (err error) {
	return instance.SetProperty("DefragmentationTasks", (value))
}

// GetDefragmentationTasks gets the value of DefragmentationTasks for the instance
func (instance *Win32_PerfRawData_ESENT_Database) GetPropertyDefragmentationTasks() (value uint32, err error) {
	retValue, err := instance.GetProperty("DefragmentationTasks")
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

// SetDefragmentationTasksPending sets the value of DefragmentationTasksPending for the instance
func (instance *Win32_PerfRawData_ESENT_Database) SetPropertyDefragmentationTasksPending(value uint32) (err error) {
	return instance.SetProperty("DefragmentationTasksPending", (value))
}

// GetDefragmentationTasksPending gets the value of DefragmentationTasksPending for the instance
func (instance *Win32_PerfRawData_ESENT_Database) GetPropertyDefragmentationTasksPending() (value uint32, err error) {
	retValue, err := instance.GetProperty("DefragmentationTasksPending")
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

// SetIODatabaseReadsAttachedAverageLatency sets the value of IODatabaseReadsAttachedAverageLatency for the instance
func (instance *Win32_PerfRawData_ESENT_Database) SetPropertyIODatabaseReadsAttachedAverageLatency(value uint64) (err error) {
	return instance.SetProperty("IODatabaseReadsAttachedAverageLatency", (value))
}

// GetIODatabaseReadsAttachedAverageLatency gets the value of IODatabaseReadsAttachedAverageLatency for the instance
func (instance *Win32_PerfRawData_ESENT_Database) GetPropertyIODatabaseReadsAttachedAverageLatency() (value uint64, err error) {
	retValue, err := instance.GetProperty("IODatabaseReadsAttachedAverageLatency")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint64)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint64 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint64(valuetmp)

	return
}

// SetIODatabaseReadsAttachedAverageLatency_Base sets the value of IODatabaseReadsAttachedAverageLatency_Base for the instance
func (instance *Win32_PerfRawData_ESENT_Database) SetPropertyIODatabaseReadsAttachedAverageLatency_Base(value uint32) (err error) {
	return instance.SetProperty("IODatabaseReadsAttachedAverageLatency_Base", (value))
}

// GetIODatabaseReadsAttachedAverageLatency_Base gets the value of IODatabaseReadsAttachedAverageLatency_Base for the instance
func (instance *Win32_PerfRawData_ESENT_Database) GetPropertyIODatabaseReadsAttachedAverageLatency_Base() (value uint32, err error) {
	retValue, err := instance.GetProperty("IODatabaseReadsAttachedAverageLatency_Base")
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

// SetIODatabaseReadsAttachedPersec sets the value of IODatabaseReadsAttachedPersec for the instance
func (instance *Win32_PerfRawData_ESENT_Database) SetPropertyIODatabaseReadsAttachedPersec(value uint32) (err error) {
	return instance.SetProperty("IODatabaseReadsAttachedPersec", (value))
}

// GetIODatabaseReadsAttachedPersec gets the value of IODatabaseReadsAttachedPersec for the instance
func (instance *Win32_PerfRawData_ESENT_Database) GetPropertyIODatabaseReadsAttachedPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("IODatabaseReadsAttachedPersec")
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

// SetIODatabaseReadsAverageLatency sets the value of IODatabaseReadsAverageLatency for the instance
func (instance *Win32_PerfRawData_ESENT_Database) SetPropertyIODatabaseReadsAverageLatency(value uint64) (err error) {
	return instance.SetProperty("IODatabaseReadsAverageLatency", (value))
}

// GetIODatabaseReadsAverageLatency gets the value of IODatabaseReadsAverageLatency for the instance
func (instance *Win32_PerfRawData_ESENT_Database) GetPropertyIODatabaseReadsAverageLatency() (value uint64, err error) {
	retValue, err := instance.GetProperty("IODatabaseReadsAverageLatency")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint64)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint64 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint64(valuetmp)

	return
}

// SetIODatabaseReadsAverageLatency_Base sets the value of IODatabaseReadsAverageLatency_Base for the instance
func (instance *Win32_PerfRawData_ESENT_Database) SetPropertyIODatabaseReadsAverageLatency_Base(value uint32) (err error) {
	return instance.SetProperty("IODatabaseReadsAverageLatency_Base", (value))
}

// GetIODatabaseReadsAverageLatency_Base gets the value of IODatabaseReadsAverageLatency_Base for the instance
func (instance *Win32_PerfRawData_ESENT_Database) GetPropertyIODatabaseReadsAverageLatency_Base() (value uint32, err error) {
	retValue, err := instance.GetProperty("IODatabaseReadsAverageLatency_Base")
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

// SetIODatabaseReadsPersec sets the value of IODatabaseReadsPersec for the instance
func (instance *Win32_PerfRawData_ESENT_Database) SetPropertyIODatabaseReadsPersec(value uint32) (err error) {
	return instance.SetProperty("IODatabaseReadsPersec", (value))
}

// GetIODatabaseReadsPersec gets the value of IODatabaseReadsPersec for the instance
func (instance *Win32_PerfRawData_ESENT_Database) GetPropertyIODatabaseReadsPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("IODatabaseReadsPersec")
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

// SetIODatabaseReadsRecoveryAverageLatency sets the value of IODatabaseReadsRecoveryAverageLatency for the instance
func (instance *Win32_PerfRawData_ESENT_Database) SetPropertyIODatabaseReadsRecoveryAverageLatency(value uint64) (err error) {
	return instance.SetProperty("IODatabaseReadsRecoveryAverageLatency", (value))
}

// GetIODatabaseReadsRecoveryAverageLatency gets the value of IODatabaseReadsRecoveryAverageLatency for the instance
func (instance *Win32_PerfRawData_ESENT_Database) GetPropertyIODatabaseReadsRecoveryAverageLatency() (value uint64, err error) {
	retValue, err := instance.GetProperty("IODatabaseReadsRecoveryAverageLatency")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint64)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint64 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint64(valuetmp)

	return
}

// SetIODatabaseReadsRecoveryAverageLatency_Base sets the value of IODatabaseReadsRecoveryAverageLatency_Base for the instance
func (instance *Win32_PerfRawData_ESENT_Database) SetPropertyIODatabaseReadsRecoveryAverageLatency_Base(value uint32) (err error) {
	return instance.SetProperty("IODatabaseReadsRecoveryAverageLatency_Base", (value))
}

// GetIODatabaseReadsRecoveryAverageLatency_Base gets the value of IODatabaseReadsRecoveryAverageLatency_Base for the instance
func (instance *Win32_PerfRawData_ESENT_Database) GetPropertyIODatabaseReadsRecoveryAverageLatency_Base() (value uint32, err error) {
	retValue, err := instance.GetProperty("IODatabaseReadsRecoveryAverageLatency_Base")
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

// SetIODatabaseReadsRecoveryPersec sets the value of IODatabaseReadsRecoveryPersec for the instance
func (instance *Win32_PerfRawData_ESENT_Database) SetPropertyIODatabaseReadsRecoveryPersec(value uint32) (err error) {
	return instance.SetProperty("IODatabaseReadsRecoveryPersec", (value))
}

// GetIODatabaseReadsRecoveryPersec gets the value of IODatabaseReadsRecoveryPersec for the instance
func (instance *Win32_PerfRawData_ESENT_Database) GetPropertyIODatabaseReadsRecoveryPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("IODatabaseReadsRecoveryPersec")
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

// SetIODatabaseWritesAttachedAverageLatency sets the value of IODatabaseWritesAttachedAverageLatency for the instance
func (instance *Win32_PerfRawData_ESENT_Database) SetPropertyIODatabaseWritesAttachedAverageLatency(value uint64) (err error) {
	return instance.SetProperty("IODatabaseWritesAttachedAverageLatency", (value))
}

// GetIODatabaseWritesAttachedAverageLatency gets the value of IODatabaseWritesAttachedAverageLatency for the instance
func (instance *Win32_PerfRawData_ESENT_Database) GetPropertyIODatabaseWritesAttachedAverageLatency() (value uint64, err error) {
	retValue, err := instance.GetProperty("IODatabaseWritesAttachedAverageLatency")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint64)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint64 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint64(valuetmp)

	return
}

// SetIODatabaseWritesAttachedAverageLatency_Base sets the value of IODatabaseWritesAttachedAverageLatency_Base for the instance
func (instance *Win32_PerfRawData_ESENT_Database) SetPropertyIODatabaseWritesAttachedAverageLatency_Base(value uint32) (err error) {
	return instance.SetProperty("IODatabaseWritesAttachedAverageLatency_Base", (value))
}

// GetIODatabaseWritesAttachedAverageLatency_Base gets the value of IODatabaseWritesAttachedAverageLatency_Base for the instance
func (instance *Win32_PerfRawData_ESENT_Database) GetPropertyIODatabaseWritesAttachedAverageLatency_Base() (value uint32, err error) {
	retValue, err := instance.GetProperty("IODatabaseWritesAttachedAverageLatency_Base")
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

// SetIODatabaseWritesAttachedPersec sets the value of IODatabaseWritesAttachedPersec for the instance
func (instance *Win32_PerfRawData_ESENT_Database) SetPropertyIODatabaseWritesAttachedPersec(value uint32) (err error) {
	return instance.SetProperty("IODatabaseWritesAttachedPersec", (value))
}

// GetIODatabaseWritesAttachedPersec gets the value of IODatabaseWritesAttachedPersec for the instance
func (instance *Win32_PerfRawData_ESENT_Database) GetPropertyIODatabaseWritesAttachedPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("IODatabaseWritesAttachedPersec")
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

// SetIODatabaseWritesAverageLatency sets the value of IODatabaseWritesAverageLatency for the instance
func (instance *Win32_PerfRawData_ESENT_Database) SetPropertyIODatabaseWritesAverageLatency(value uint64) (err error) {
	return instance.SetProperty("IODatabaseWritesAverageLatency", (value))
}

// GetIODatabaseWritesAverageLatency gets the value of IODatabaseWritesAverageLatency for the instance
func (instance *Win32_PerfRawData_ESENT_Database) GetPropertyIODatabaseWritesAverageLatency() (value uint64, err error) {
	retValue, err := instance.GetProperty("IODatabaseWritesAverageLatency")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint64)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint64 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint64(valuetmp)

	return
}

// SetIODatabaseWritesAverageLatency_Base sets the value of IODatabaseWritesAverageLatency_Base for the instance
func (instance *Win32_PerfRawData_ESENT_Database) SetPropertyIODatabaseWritesAverageLatency_Base(value uint32) (err error) {
	return instance.SetProperty("IODatabaseWritesAverageLatency_Base", (value))
}

// GetIODatabaseWritesAverageLatency_Base gets the value of IODatabaseWritesAverageLatency_Base for the instance
func (instance *Win32_PerfRawData_ESENT_Database) GetPropertyIODatabaseWritesAverageLatency_Base() (value uint32, err error) {
	retValue, err := instance.GetProperty("IODatabaseWritesAverageLatency_Base")
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

// SetIODatabaseWritesPersec sets the value of IODatabaseWritesPersec for the instance
func (instance *Win32_PerfRawData_ESENT_Database) SetPropertyIODatabaseWritesPersec(value uint32) (err error) {
	return instance.SetProperty("IODatabaseWritesPersec", (value))
}

// GetIODatabaseWritesPersec gets the value of IODatabaseWritesPersec for the instance
func (instance *Win32_PerfRawData_ESENT_Database) GetPropertyIODatabaseWritesPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("IODatabaseWritesPersec")
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

// SetIODatabaseWritesRecoveryAverageLatency sets the value of IODatabaseWritesRecoveryAverageLatency for the instance
func (instance *Win32_PerfRawData_ESENT_Database) SetPropertyIODatabaseWritesRecoveryAverageLatency(value uint64) (err error) {
	return instance.SetProperty("IODatabaseWritesRecoveryAverageLatency", (value))
}

// GetIODatabaseWritesRecoveryAverageLatency gets the value of IODatabaseWritesRecoveryAverageLatency for the instance
func (instance *Win32_PerfRawData_ESENT_Database) GetPropertyIODatabaseWritesRecoveryAverageLatency() (value uint64, err error) {
	retValue, err := instance.GetProperty("IODatabaseWritesRecoveryAverageLatency")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint64)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint64 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint64(valuetmp)

	return
}

// SetIODatabaseWritesRecoveryAverageLatency_Base sets the value of IODatabaseWritesRecoveryAverageLatency_Base for the instance
func (instance *Win32_PerfRawData_ESENT_Database) SetPropertyIODatabaseWritesRecoveryAverageLatency_Base(value uint32) (err error) {
	return instance.SetProperty("IODatabaseWritesRecoveryAverageLatency_Base", (value))
}

// GetIODatabaseWritesRecoveryAverageLatency_Base gets the value of IODatabaseWritesRecoveryAverageLatency_Base for the instance
func (instance *Win32_PerfRawData_ESENT_Database) GetPropertyIODatabaseWritesRecoveryAverageLatency_Base() (value uint32, err error) {
	retValue, err := instance.GetProperty("IODatabaseWritesRecoveryAverageLatency_Base")
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

// SetIODatabaseWritesRecoveryPersec sets the value of IODatabaseWritesRecoveryPersec for the instance
func (instance *Win32_PerfRawData_ESENT_Database) SetPropertyIODatabaseWritesRecoveryPersec(value uint32) (err error) {
	return instance.SetProperty("IODatabaseWritesRecoveryPersec", (value))
}

// GetIODatabaseWritesRecoveryPersec gets the value of IODatabaseWritesRecoveryPersec for the instance
func (instance *Win32_PerfRawData_ESENT_Database) GetPropertyIODatabaseWritesRecoveryPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("IODatabaseWritesRecoveryPersec")
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

// SetIOFlushMapWritesAverageLatency sets the value of IOFlushMapWritesAverageLatency for the instance
func (instance *Win32_PerfRawData_ESENT_Database) SetPropertyIOFlushMapWritesAverageLatency(value uint64) (err error) {
	return instance.SetProperty("IOFlushMapWritesAverageLatency", (value))
}

// GetIOFlushMapWritesAverageLatency gets the value of IOFlushMapWritesAverageLatency for the instance
func (instance *Win32_PerfRawData_ESENT_Database) GetPropertyIOFlushMapWritesAverageLatency() (value uint64, err error) {
	retValue, err := instance.GetProperty("IOFlushMapWritesAverageLatency")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint64)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint64 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint64(valuetmp)

	return
}

// SetIOFlushMapWritesAverageLatency_Base sets the value of IOFlushMapWritesAverageLatency_Base for the instance
func (instance *Win32_PerfRawData_ESENT_Database) SetPropertyIOFlushMapWritesAverageLatency_Base(value uint32) (err error) {
	return instance.SetProperty("IOFlushMapWritesAverageLatency_Base", (value))
}

// GetIOFlushMapWritesAverageLatency_Base gets the value of IOFlushMapWritesAverageLatency_Base for the instance
func (instance *Win32_PerfRawData_ESENT_Database) GetPropertyIOFlushMapWritesAverageLatency_Base() (value uint32, err error) {
	retValue, err := instance.GetProperty("IOFlushMapWritesAverageLatency_Base")
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

// SetIOFlushMapWritesPersec sets the value of IOFlushMapWritesPersec for the instance
func (instance *Win32_PerfRawData_ESENT_Database) SetPropertyIOFlushMapWritesPersec(value uint32) (err error) {
	return instance.SetProperty("IOFlushMapWritesPersec", (value))
}

// GetIOFlushMapWritesPersec gets the value of IOFlushMapWritesPersec for the instance
func (instance *Win32_PerfRawData_ESENT_Database) GetPropertyIOFlushMapWritesPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("IOFlushMapWritesPersec")
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

// SetIOLogReadsAverageLatency sets the value of IOLogReadsAverageLatency for the instance
func (instance *Win32_PerfRawData_ESENT_Database) SetPropertyIOLogReadsAverageLatency(value uint64) (err error) {
	return instance.SetProperty("IOLogReadsAverageLatency", (value))
}

// GetIOLogReadsAverageLatency gets the value of IOLogReadsAverageLatency for the instance
func (instance *Win32_PerfRawData_ESENT_Database) GetPropertyIOLogReadsAverageLatency() (value uint64, err error) {
	retValue, err := instance.GetProperty("IOLogReadsAverageLatency")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint64)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint64 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint64(valuetmp)

	return
}

// SetIOLogReadsAverageLatency_Base sets the value of IOLogReadsAverageLatency_Base for the instance
func (instance *Win32_PerfRawData_ESENT_Database) SetPropertyIOLogReadsAverageLatency_Base(value uint32) (err error) {
	return instance.SetProperty("IOLogReadsAverageLatency_Base", (value))
}

// GetIOLogReadsAverageLatency_Base gets the value of IOLogReadsAverageLatency_Base for the instance
func (instance *Win32_PerfRawData_ESENT_Database) GetPropertyIOLogReadsAverageLatency_Base() (value uint32, err error) {
	retValue, err := instance.GetProperty("IOLogReadsAverageLatency_Base")
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

// SetIOLogReadsPersec sets the value of IOLogReadsPersec for the instance
func (instance *Win32_PerfRawData_ESENT_Database) SetPropertyIOLogReadsPersec(value uint32) (err error) {
	return instance.SetProperty("IOLogReadsPersec", (value))
}

// GetIOLogReadsPersec gets the value of IOLogReadsPersec for the instance
func (instance *Win32_PerfRawData_ESENT_Database) GetPropertyIOLogReadsPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("IOLogReadsPersec")
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

// SetIOLogWritesAverageLatency sets the value of IOLogWritesAverageLatency for the instance
func (instance *Win32_PerfRawData_ESENT_Database) SetPropertyIOLogWritesAverageLatency(value uint64) (err error) {
	return instance.SetProperty("IOLogWritesAverageLatency", (value))
}

// GetIOLogWritesAverageLatency gets the value of IOLogWritesAverageLatency for the instance
func (instance *Win32_PerfRawData_ESENT_Database) GetPropertyIOLogWritesAverageLatency() (value uint64, err error) {
	retValue, err := instance.GetProperty("IOLogWritesAverageLatency")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint64)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint64 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint64(valuetmp)

	return
}

// SetIOLogWritesAverageLatency_Base sets the value of IOLogWritesAverageLatency_Base for the instance
func (instance *Win32_PerfRawData_ESENT_Database) SetPropertyIOLogWritesAverageLatency_Base(value uint32) (err error) {
	return instance.SetProperty("IOLogWritesAverageLatency_Base", (value))
}

// GetIOLogWritesAverageLatency_Base gets the value of IOLogWritesAverageLatency_Base for the instance
func (instance *Win32_PerfRawData_ESENT_Database) GetPropertyIOLogWritesAverageLatency_Base() (value uint32, err error) {
	retValue, err := instance.GetProperty("IOLogWritesAverageLatency_Base")
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

// SetIOLogWritesPersec sets the value of IOLogWritesPersec for the instance
func (instance *Win32_PerfRawData_ESENT_Database) SetPropertyIOLogWritesPersec(value uint32) (err error) {
	return instance.SetProperty("IOLogWritesPersec", (value))
}

// GetIOLogWritesPersec gets the value of IOLogWritesPersec for the instance
func (instance *Win32_PerfRawData_ESENT_Database) GetPropertyIOLogWritesPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("IOLogWritesPersec")
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

// SetLogBytesGeneratedPersec sets the value of LogBytesGeneratedPersec for the instance
func (instance *Win32_PerfRawData_ESENT_Database) SetPropertyLogBytesGeneratedPersec(value uint32) (err error) {
	return instance.SetProperty("LogBytesGeneratedPersec", (value))
}

// GetLogBytesGeneratedPersec gets the value of LogBytesGeneratedPersec for the instance
func (instance *Win32_PerfRawData_ESENT_Database) GetPropertyLogBytesGeneratedPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("LogBytesGeneratedPersec")
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

// SetLogBytesWritePersec sets the value of LogBytesWritePersec for the instance
func (instance *Win32_PerfRawData_ESENT_Database) SetPropertyLogBytesWritePersec(value uint32) (err error) {
	return instance.SetProperty("LogBytesWritePersec", (value))
}

// GetLogBytesWritePersec gets the value of LogBytesWritePersec for the instance
func (instance *Win32_PerfRawData_ESENT_Database) GetPropertyLogBytesWritePersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("LogBytesWritePersec")
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

// SetLogRecordStallsPersec sets the value of LogRecordStallsPersec for the instance
func (instance *Win32_PerfRawData_ESENT_Database) SetPropertyLogRecordStallsPersec(value uint32) (err error) {
	return instance.SetProperty("LogRecordStallsPersec", (value))
}

// GetLogRecordStallsPersec gets the value of LogRecordStallsPersec for the instance
func (instance *Win32_PerfRawData_ESENT_Database) GetPropertyLogRecordStallsPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("LogRecordStallsPersec")
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

// SetLogThreadsWaiting sets the value of LogThreadsWaiting for the instance
func (instance *Win32_PerfRawData_ESENT_Database) SetPropertyLogThreadsWaiting(value uint32) (err error) {
	return instance.SetProperty("LogThreadsWaiting", (value))
}

// GetLogThreadsWaiting gets the value of LogThreadsWaiting for the instance
func (instance *Win32_PerfRawData_ESENT_Database) GetPropertyLogThreadsWaiting() (value uint32, err error) {
	retValue, err := instance.GetProperty("LogThreadsWaiting")
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

// SetLogWritesPersec sets the value of LogWritesPersec for the instance
func (instance *Win32_PerfRawData_ESENT_Database) SetPropertyLogWritesPersec(value uint32) (err error) {
	return instance.SetProperty("LogWritesPersec", (value))
}

// GetLogWritesPersec gets the value of LogWritesPersec for the instance
func (instance *Win32_PerfRawData_ESENT_Database) GetPropertyLogWritesPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("LogWritesPersec")
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

// SetSessionsInUse sets the value of SessionsInUse for the instance
func (instance *Win32_PerfRawData_ESENT_Database) SetPropertySessionsInUse(value uint32) (err error) {
	return instance.SetProperty("SessionsInUse", (value))
}

// GetSessionsInUse gets the value of SessionsInUse for the instance
func (instance *Win32_PerfRawData_ESENT_Database) GetPropertySessionsInUse() (value uint32, err error) {
	retValue, err := instance.GetProperty("SessionsInUse")
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

// SetSessionsPercentUsed sets the value of SessionsPercentUsed for the instance
func (instance *Win32_PerfRawData_ESENT_Database) SetPropertySessionsPercentUsed(value uint32) (err error) {
	return instance.SetProperty("SessionsPercentUsed", (value))
}

// GetSessionsPercentUsed gets the value of SessionsPercentUsed for the instance
func (instance *Win32_PerfRawData_ESENT_Database) GetPropertySessionsPercentUsed() (value uint32, err error) {
	retValue, err := instance.GetProperty("SessionsPercentUsed")
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

// SetSessionsPercentUsed_Base sets the value of SessionsPercentUsed_Base for the instance
func (instance *Win32_PerfRawData_ESENT_Database) SetPropertySessionsPercentUsed_Base(value uint32) (err error) {
	return instance.SetProperty("SessionsPercentUsed_Base", (value))
}

// GetSessionsPercentUsed_Base gets the value of SessionsPercentUsed_Base for the instance
func (instance *Win32_PerfRawData_ESENT_Database) GetPropertySessionsPercentUsed_Base() (value uint32, err error) {
	retValue, err := instance.GetProperty("SessionsPercentUsed_Base")
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

// SetTableClosesPersec sets the value of TableClosesPersec for the instance
func (instance *Win32_PerfRawData_ESENT_Database) SetPropertyTableClosesPersec(value uint32) (err error) {
	return instance.SetProperty("TableClosesPersec", (value))
}

// GetTableClosesPersec gets the value of TableClosesPersec for the instance
func (instance *Win32_PerfRawData_ESENT_Database) GetPropertyTableClosesPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("TableClosesPersec")
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

// SetTableOpenCacheHitsPersec sets the value of TableOpenCacheHitsPersec for the instance
func (instance *Win32_PerfRawData_ESENT_Database) SetPropertyTableOpenCacheHitsPersec(value uint32) (err error) {
	return instance.SetProperty("TableOpenCacheHitsPersec", (value))
}

// GetTableOpenCacheHitsPersec gets the value of TableOpenCacheHitsPersec for the instance
func (instance *Win32_PerfRawData_ESENT_Database) GetPropertyTableOpenCacheHitsPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("TableOpenCacheHitsPersec")
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

// SetTableOpenCacheMissesPersec sets the value of TableOpenCacheMissesPersec for the instance
func (instance *Win32_PerfRawData_ESENT_Database) SetPropertyTableOpenCacheMissesPersec(value uint32) (err error) {
	return instance.SetProperty("TableOpenCacheMissesPersec", (value))
}

// GetTableOpenCacheMissesPersec gets the value of TableOpenCacheMissesPersec for the instance
func (instance *Win32_PerfRawData_ESENT_Database) GetPropertyTableOpenCacheMissesPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("TableOpenCacheMissesPersec")
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

// SetTableOpenCachePercentHit sets the value of TableOpenCachePercentHit for the instance
func (instance *Win32_PerfRawData_ESENT_Database) SetPropertyTableOpenCachePercentHit(value uint32) (err error) {
	return instance.SetProperty("TableOpenCachePercentHit", (value))
}

// GetTableOpenCachePercentHit gets the value of TableOpenCachePercentHit for the instance
func (instance *Win32_PerfRawData_ESENT_Database) GetPropertyTableOpenCachePercentHit() (value uint32, err error) {
	retValue, err := instance.GetProperty("TableOpenCachePercentHit")
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

// SetTableOpenCachePercentHit_Base sets the value of TableOpenCachePercentHit_Base for the instance
func (instance *Win32_PerfRawData_ESENT_Database) SetPropertyTableOpenCachePercentHit_Base(value uint32) (err error) {
	return instance.SetProperty("TableOpenCachePercentHit_Base", (value))
}

// GetTableOpenCachePercentHit_Base gets the value of TableOpenCachePercentHit_Base for the instance
func (instance *Win32_PerfRawData_ESENT_Database) GetPropertyTableOpenCachePercentHit_Base() (value uint32, err error) {
	retValue, err := instance.GetProperty("TableOpenCachePercentHit_Base")
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

// SetTableOpensPersec sets the value of TableOpensPersec for the instance
func (instance *Win32_PerfRawData_ESENT_Database) SetPropertyTableOpensPersec(value uint32) (err error) {
	return instance.SetProperty("TableOpensPersec", (value))
}

// GetTableOpensPersec gets the value of TableOpensPersec for the instance
func (instance *Win32_PerfRawData_ESENT_Database) GetPropertyTableOpensPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("TableOpensPersec")
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

// SetTablesOpen sets the value of TablesOpen for the instance
func (instance *Win32_PerfRawData_ESENT_Database) SetPropertyTablesOpen(value uint32) (err error) {
	return instance.SetProperty("TablesOpen", (value))
}

// GetTablesOpen gets the value of TablesOpen for the instance
func (instance *Win32_PerfRawData_ESENT_Database) GetPropertyTablesOpen() (value uint32, err error) {
	retValue, err := instance.GetProperty("TablesOpen")
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

// SetVersionBucketsAllocated sets the value of VersionBucketsAllocated for the instance
func (instance *Win32_PerfRawData_ESENT_Database) SetPropertyVersionBucketsAllocated(value uint32) (err error) {
	return instance.SetProperty("VersionBucketsAllocated", (value))
}

// GetVersionBucketsAllocated gets the value of VersionBucketsAllocated for the instance
func (instance *Win32_PerfRawData_ESENT_Database) GetPropertyVersionBucketsAllocated() (value uint32, err error) {
	retValue, err := instance.GetProperty("VersionBucketsAllocated")
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
