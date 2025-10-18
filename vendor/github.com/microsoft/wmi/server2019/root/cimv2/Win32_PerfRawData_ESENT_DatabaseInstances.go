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

// Win32_PerfRawData_ESENT_DatabaseInstances struct
type Win32_PerfRawData_ESENT_DatabaseInstances struct {
	*Win32_PerfRawData

	//
	DatabaseCacheMissAttachedAverageLatency uint64

	//
	DatabaseCacheMissAttachedAverageLatency_Base uint32

	//
	DatabaseCacheMissesPersec uint32

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
	DatabaseCacheSizeMB uint64

	//
	DatabaseMaintenanceDuration uint32

	//
	DatabaseOldestTransaction uint64

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
	LogCheckpointDepthasaPercentofTarget uint32

	//
	LogCheckpointDepthasaPercentofTarget_Base uint32

	//
	LogFileCurrentGeneration uint32

	//
	LogFilesGenerated uint32

	//
	LogFilesGeneratedPrematurely uint32

	//
	LogGenerationCheckpointDepth uint32

	//
	LogGenerationCheckpointDepthMax uint32

	//
	LogGenerationCheckpointDepthTarget uint32

	//
	LogGenerationLossResiliencyDepth uint32

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
	StreamingBackupPagesReadPersec uint32

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
	Versionbucketsallocated uint32
}

func NewWin32_PerfRawData_ESENT_DatabaseInstancesEx1(instance *cim.WmiInstance) (newInstance *Win32_PerfRawData_ESENT_DatabaseInstances, err error) {
	tmp, err := NewWin32_PerfRawDataEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfRawData_ESENT_DatabaseInstances{
		Win32_PerfRawData: tmp,
	}
	return
}

func NewWin32_PerfRawData_ESENT_DatabaseInstancesEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_PerfRawData_ESENT_DatabaseInstances, err error) {
	tmp, err := NewWin32_PerfRawDataEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfRawData_ESENT_DatabaseInstances{
		Win32_PerfRawData: tmp,
	}
	return
}

// SetDatabaseCacheMissAttachedAverageLatency sets the value of DatabaseCacheMissAttachedAverageLatency for the instance
func (instance *Win32_PerfRawData_ESENT_DatabaseInstances) SetPropertyDatabaseCacheMissAttachedAverageLatency(value uint64) (err error) {
	return instance.SetProperty("DatabaseCacheMissAttachedAverageLatency", (value))
}

// GetDatabaseCacheMissAttachedAverageLatency gets the value of DatabaseCacheMissAttachedAverageLatency for the instance
func (instance *Win32_PerfRawData_ESENT_DatabaseInstances) GetPropertyDatabaseCacheMissAttachedAverageLatency() (value uint64, err error) {
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
func (instance *Win32_PerfRawData_ESENT_DatabaseInstances) SetPropertyDatabaseCacheMissAttachedAverageLatency_Base(value uint32) (err error) {
	return instance.SetProperty("DatabaseCacheMissAttachedAverageLatency_Base", (value))
}

// GetDatabaseCacheMissAttachedAverageLatency_Base gets the value of DatabaseCacheMissAttachedAverageLatency_Base for the instance
func (instance *Win32_PerfRawData_ESENT_DatabaseInstances) GetPropertyDatabaseCacheMissAttachedAverageLatency_Base() (value uint32, err error) {
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
func (instance *Win32_PerfRawData_ESENT_DatabaseInstances) SetPropertyDatabaseCacheMissesPersec(value uint32) (err error) {
	return instance.SetProperty("DatabaseCacheMissesPersec", (value))
}

// GetDatabaseCacheMissesPersec gets the value of DatabaseCacheMissesPersec for the instance
func (instance *Win32_PerfRawData_ESENT_DatabaseInstances) GetPropertyDatabaseCacheMissesPersec() (value uint32, err error) {
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

// SetDatabaseCachePercentHit sets the value of DatabaseCachePercentHit for the instance
func (instance *Win32_PerfRawData_ESENT_DatabaseInstances) SetPropertyDatabaseCachePercentHit(value uint32) (err error) {
	return instance.SetProperty("DatabaseCachePercentHit", (value))
}

// GetDatabaseCachePercentHit gets the value of DatabaseCachePercentHit for the instance
func (instance *Win32_PerfRawData_ESENT_DatabaseInstances) GetPropertyDatabaseCachePercentHit() (value uint32, err error) {
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
func (instance *Win32_PerfRawData_ESENT_DatabaseInstances) SetPropertyDatabaseCachePercentHit_Base(value uint32) (err error) {
	return instance.SetProperty("DatabaseCachePercentHit_Base", (value))
}

// GetDatabaseCachePercentHit_Base gets the value of DatabaseCachePercentHit_Base for the instance
func (instance *Win32_PerfRawData_ESENT_DatabaseInstances) GetPropertyDatabaseCachePercentHit_Base() (value uint32, err error) {
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
func (instance *Win32_PerfRawData_ESENT_DatabaseInstances) SetPropertyDatabaseCachePercentHitUnique(value uint32) (err error) {
	return instance.SetProperty("DatabaseCachePercentHitUnique", (value))
}

// GetDatabaseCachePercentHitUnique gets the value of DatabaseCachePercentHitUnique for the instance
func (instance *Win32_PerfRawData_ESENT_DatabaseInstances) GetPropertyDatabaseCachePercentHitUnique() (value uint32, err error) {
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
func (instance *Win32_PerfRawData_ESENT_DatabaseInstances) SetPropertyDatabaseCachePercentHitUnique_Base(value uint32) (err error) {
	return instance.SetProperty("DatabaseCachePercentHitUnique_Base", (value))
}

// GetDatabaseCachePercentHitUnique_Base gets the value of DatabaseCachePercentHitUnique_Base for the instance
func (instance *Win32_PerfRawData_ESENT_DatabaseInstances) GetPropertyDatabaseCachePercentHitUnique_Base() (value uint32, err error) {
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
func (instance *Win32_PerfRawData_ESENT_DatabaseInstances) SetPropertyDatabaseCacheRequestsPersec(value uint32) (err error) {
	return instance.SetProperty("DatabaseCacheRequestsPersec", (value))
}

// GetDatabaseCacheRequestsPersec gets the value of DatabaseCacheRequestsPersec for the instance
func (instance *Win32_PerfRawData_ESENT_DatabaseInstances) GetPropertyDatabaseCacheRequestsPersec() (value uint32, err error) {
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
func (instance *Win32_PerfRawData_ESENT_DatabaseInstances) SetPropertyDatabaseCacheRequestsPersecUnique(value uint32) (err error) {
	return instance.SetProperty("DatabaseCacheRequestsPersecUnique", (value))
}

// GetDatabaseCacheRequestsPersecUnique gets the value of DatabaseCacheRequestsPersecUnique for the instance
func (instance *Win32_PerfRawData_ESENT_DatabaseInstances) GetPropertyDatabaseCacheRequestsPersecUnique() (value uint32, err error) {
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

// SetDatabaseCacheSizeMB sets the value of DatabaseCacheSizeMB for the instance
func (instance *Win32_PerfRawData_ESENT_DatabaseInstances) SetPropertyDatabaseCacheSizeMB(value uint64) (err error) {
	return instance.SetProperty("DatabaseCacheSizeMB", (value))
}

// GetDatabaseCacheSizeMB gets the value of DatabaseCacheSizeMB for the instance
func (instance *Win32_PerfRawData_ESENT_DatabaseInstances) GetPropertyDatabaseCacheSizeMB() (value uint64, err error) {
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

// SetDatabaseMaintenanceDuration sets the value of DatabaseMaintenanceDuration for the instance
func (instance *Win32_PerfRawData_ESENT_DatabaseInstances) SetPropertyDatabaseMaintenanceDuration(value uint32) (err error) {
	return instance.SetProperty("DatabaseMaintenanceDuration", (value))
}

// GetDatabaseMaintenanceDuration gets the value of DatabaseMaintenanceDuration for the instance
func (instance *Win32_PerfRawData_ESENT_DatabaseInstances) GetPropertyDatabaseMaintenanceDuration() (value uint32, err error) {
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

// SetDatabaseOldestTransaction sets the value of DatabaseOldestTransaction for the instance
func (instance *Win32_PerfRawData_ESENT_DatabaseInstances) SetPropertyDatabaseOldestTransaction(value uint64) (err error) {
	return instance.SetProperty("DatabaseOldestTransaction", (value))
}

// GetDatabaseOldestTransaction gets the value of DatabaseOldestTransaction for the instance
func (instance *Win32_PerfRawData_ESENT_DatabaseInstances) GetPropertyDatabaseOldestTransaction() (value uint64, err error) {
	retValue, err := instance.GetProperty("DatabaseOldestTransaction")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint64)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint64 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint64(valuetmp)

	return
}

// SetDefragmentationTasks sets the value of DefragmentationTasks for the instance
func (instance *Win32_PerfRawData_ESENT_DatabaseInstances) SetPropertyDefragmentationTasks(value uint32) (err error) {
	return instance.SetProperty("DefragmentationTasks", (value))
}

// GetDefragmentationTasks gets the value of DefragmentationTasks for the instance
func (instance *Win32_PerfRawData_ESENT_DatabaseInstances) GetPropertyDefragmentationTasks() (value uint32, err error) {
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
func (instance *Win32_PerfRawData_ESENT_DatabaseInstances) SetPropertyDefragmentationTasksPending(value uint32) (err error) {
	return instance.SetProperty("DefragmentationTasksPending", (value))
}

// GetDefragmentationTasksPending gets the value of DefragmentationTasksPending for the instance
func (instance *Win32_PerfRawData_ESENT_DatabaseInstances) GetPropertyDefragmentationTasksPending() (value uint32, err error) {
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
func (instance *Win32_PerfRawData_ESENT_DatabaseInstances) SetPropertyIODatabaseReadsAttachedAverageLatency(value uint64) (err error) {
	return instance.SetProperty("IODatabaseReadsAttachedAverageLatency", (value))
}

// GetIODatabaseReadsAttachedAverageLatency gets the value of IODatabaseReadsAttachedAverageLatency for the instance
func (instance *Win32_PerfRawData_ESENT_DatabaseInstances) GetPropertyIODatabaseReadsAttachedAverageLatency() (value uint64, err error) {
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
func (instance *Win32_PerfRawData_ESENT_DatabaseInstances) SetPropertyIODatabaseReadsAttachedAverageLatency_Base(value uint32) (err error) {
	return instance.SetProperty("IODatabaseReadsAttachedAverageLatency_Base", (value))
}

// GetIODatabaseReadsAttachedAverageLatency_Base gets the value of IODatabaseReadsAttachedAverageLatency_Base for the instance
func (instance *Win32_PerfRawData_ESENT_DatabaseInstances) GetPropertyIODatabaseReadsAttachedAverageLatency_Base() (value uint32, err error) {
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
func (instance *Win32_PerfRawData_ESENT_DatabaseInstances) SetPropertyIODatabaseReadsAttachedPersec(value uint32) (err error) {
	return instance.SetProperty("IODatabaseReadsAttachedPersec", (value))
}

// GetIODatabaseReadsAttachedPersec gets the value of IODatabaseReadsAttachedPersec for the instance
func (instance *Win32_PerfRawData_ESENT_DatabaseInstances) GetPropertyIODatabaseReadsAttachedPersec() (value uint32, err error) {
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
func (instance *Win32_PerfRawData_ESENT_DatabaseInstances) SetPropertyIODatabaseReadsAverageLatency(value uint64) (err error) {
	return instance.SetProperty("IODatabaseReadsAverageLatency", (value))
}

// GetIODatabaseReadsAverageLatency gets the value of IODatabaseReadsAverageLatency for the instance
func (instance *Win32_PerfRawData_ESENT_DatabaseInstances) GetPropertyIODatabaseReadsAverageLatency() (value uint64, err error) {
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
func (instance *Win32_PerfRawData_ESENT_DatabaseInstances) SetPropertyIODatabaseReadsAverageLatency_Base(value uint32) (err error) {
	return instance.SetProperty("IODatabaseReadsAverageLatency_Base", (value))
}

// GetIODatabaseReadsAverageLatency_Base gets the value of IODatabaseReadsAverageLatency_Base for the instance
func (instance *Win32_PerfRawData_ESENT_DatabaseInstances) GetPropertyIODatabaseReadsAverageLatency_Base() (value uint32, err error) {
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
func (instance *Win32_PerfRawData_ESENT_DatabaseInstances) SetPropertyIODatabaseReadsPersec(value uint32) (err error) {
	return instance.SetProperty("IODatabaseReadsPersec", (value))
}

// GetIODatabaseReadsPersec gets the value of IODatabaseReadsPersec for the instance
func (instance *Win32_PerfRawData_ESENT_DatabaseInstances) GetPropertyIODatabaseReadsPersec() (value uint32, err error) {
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
func (instance *Win32_PerfRawData_ESENT_DatabaseInstances) SetPropertyIODatabaseReadsRecoveryAverageLatency(value uint64) (err error) {
	return instance.SetProperty("IODatabaseReadsRecoveryAverageLatency", (value))
}

// GetIODatabaseReadsRecoveryAverageLatency gets the value of IODatabaseReadsRecoveryAverageLatency for the instance
func (instance *Win32_PerfRawData_ESENT_DatabaseInstances) GetPropertyIODatabaseReadsRecoveryAverageLatency() (value uint64, err error) {
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
func (instance *Win32_PerfRawData_ESENT_DatabaseInstances) SetPropertyIODatabaseReadsRecoveryAverageLatency_Base(value uint32) (err error) {
	return instance.SetProperty("IODatabaseReadsRecoveryAverageLatency_Base", (value))
}

// GetIODatabaseReadsRecoveryAverageLatency_Base gets the value of IODatabaseReadsRecoveryAverageLatency_Base for the instance
func (instance *Win32_PerfRawData_ESENT_DatabaseInstances) GetPropertyIODatabaseReadsRecoveryAverageLatency_Base() (value uint32, err error) {
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
func (instance *Win32_PerfRawData_ESENT_DatabaseInstances) SetPropertyIODatabaseReadsRecoveryPersec(value uint32) (err error) {
	return instance.SetProperty("IODatabaseReadsRecoveryPersec", (value))
}

// GetIODatabaseReadsRecoveryPersec gets the value of IODatabaseReadsRecoveryPersec for the instance
func (instance *Win32_PerfRawData_ESENT_DatabaseInstances) GetPropertyIODatabaseReadsRecoveryPersec() (value uint32, err error) {
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
func (instance *Win32_PerfRawData_ESENT_DatabaseInstances) SetPropertyIODatabaseWritesAttachedAverageLatency(value uint64) (err error) {
	return instance.SetProperty("IODatabaseWritesAttachedAverageLatency", (value))
}

// GetIODatabaseWritesAttachedAverageLatency gets the value of IODatabaseWritesAttachedAverageLatency for the instance
func (instance *Win32_PerfRawData_ESENT_DatabaseInstances) GetPropertyIODatabaseWritesAttachedAverageLatency() (value uint64, err error) {
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
func (instance *Win32_PerfRawData_ESENT_DatabaseInstances) SetPropertyIODatabaseWritesAttachedAverageLatency_Base(value uint32) (err error) {
	return instance.SetProperty("IODatabaseWritesAttachedAverageLatency_Base", (value))
}

// GetIODatabaseWritesAttachedAverageLatency_Base gets the value of IODatabaseWritesAttachedAverageLatency_Base for the instance
func (instance *Win32_PerfRawData_ESENT_DatabaseInstances) GetPropertyIODatabaseWritesAttachedAverageLatency_Base() (value uint32, err error) {
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
func (instance *Win32_PerfRawData_ESENT_DatabaseInstances) SetPropertyIODatabaseWritesAttachedPersec(value uint32) (err error) {
	return instance.SetProperty("IODatabaseWritesAttachedPersec", (value))
}

// GetIODatabaseWritesAttachedPersec gets the value of IODatabaseWritesAttachedPersec for the instance
func (instance *Win32_PerfRawData_ESENT_DatabaseInstances) GetPropertyIODatabaseWritesAttachedPersec() (value uint32, err error) {
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
func (instance *Win32_PerfRawData_ESENT_DatabaseInstances) SetPropertyIODatabaseWritesAverageLatency(value uint64) (err error) {
	return instance.SetProperty("IODatabaseWritesAverageLatency", (value))
}

// GetIODatabaseWritesAverageLatency gets the value of IODatabaseWritesAverageLatency for the instance
func (instance *Win32_PerfRawData_ESENT_DatabaseInstances) GetPropertyIODatabaseWritesAverageLatency() (value uint64, err error) {
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
func (instance *Win32_PerfRawData_ESENT_DatabaseInstances) SetPropertyIODatabaseWritesAverageLatency_Base(value uint32) (err error) {
	return instance.SetProperty("IODatabaseWritesAverageLatency_Base", (value))
}

// GetIODatabaseWritesAverageLatency_Base gets the value of IODatabaseWritesAverageLatency_Base for the instance
func (instance *Win32_PerfRawData_ESENT_DatabaseInstances) GetPropertyIODatabaseWritesAverageLatency_Base() (value uint32, err error) {
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
func (instance *Win32_PerfRawData_ESENT_DatabaseInstances) SetPropertyIODatabaseWritesPersec(value uint32) (err error) {
	return instance.SetProperty("IODatabaseWritesPersec", (value))
}

// GetIODatabaseWritesPersec gets the value of IODatabaseWritesPersec for the instance
func (instance *Win32_PerfRawData_ESENT_DatabaseInstances) GetPropertyIODatabaseWritesPersec() (value uint32, err error) {
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
func (instance *Win32_PerfRawData_ESENT_DatabaseInstances) SetPropertyIODatabaseWritesRecoveryAverageLatency(value uint64) (err error) {
	return instance.SetProperty("IODatabaseWritesRecoveryAverageLatency", (value))
}

// GetIODatabaseWritesRecoveryAverageLatency gets the value of IODatabaseWritesRecoveryAverageLatency for the instance
func (instance *Win32_PerfRawData_ESENT_DatabaseInstances) GetPropertyIODatabaseWritesRecoveryAverageLatency() (value uint64, err error) {
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
func (instance *Win32_PerfRawData_ESENT_DatabaseInstances) SetPropertyIODatabaseWritesRecoveryAverageLatency_Base(value uint32) (err error) {
	return instance.SetProperty("IODatabaseWritesRecoveryAverageLatency_Base", (value))
}

// GetIODatabaseWritesRecoveryAverageLatency_Base gets the value of IODatabaseWritesRecoveryAverageLatency_Base for the instance
func (instance *Win32_PerfRawData_ESENT_DatabaseInstances) GetPropertyIODatabaseWritesRecoveryAverageLatency_Base() (value uint32, err error) {
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
func (instance *Win32_PerfRawData_ESENT_DatabaseInstances) SetPropertyIODatabaseWritesRecoveryPersec(value uint32) (err error) {
	return instance.SetProperty("IODatabaseWritesRecoveryPersec", (value))
}

// GetIODatabaseWritesRecoveryPersec gets the value of IODatabaseWritesRecoveryPersec for the instance
func (instance *Win32_PerfRawData_ESENT_DatabaseInstances) GetPropertyIODatabaseWritesRecoveryPersec() (value uint32, err error) {
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
func (instance *Win32_PerfRawData_ESENT_DatabaseInstances) SetPropertyIOFlushMapWritesAverageLatency(value uint64) (err error) {
	return instance.SetProperty("IOFlushMapWritesAverageLatency", (value))
}

// GetIOFlushMapWritesAverageLatency gets the value of IOFlushMapWritesAverageLatency for the instance
func (instance *Win32_PerfRawData_ESENT_DatabaseInstances) GetPropertyIOFlushMapWritesAverageLatency() (value uint64, err error) {
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
func (instance *Win32_PerfRawData_ESENT_DatabaseInstances) SetPropertyIOFlushMapWritesAverageLatency_Base(value uint32) (err error) {
	return instance.SetProperty("IOFlushMapWritesAverageLatency_Base", (value))
}

// GetIOFlushMapWritesAverageLatency_Base gets the value of IOFlushMapWritesAverageLatency_Base for the instance
func (instance *Win32_PerfRawData_ESENT_DatabaseInstances) GetPropertyIOFlushMapWritesAverageLatency_Base() (value uint32, err error) {
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
func (instance *Win32_PerfRawData_ESENT_DatabaseInstances) SetPropertyIOFlushMapWritesPersec(value uint32) (err error) {
	return instance.SetProperty("IOFlushMapWritesPersec", (value))
}

// GetIOFlushMapWritesPersec gets the value of IOFlushMapWritesPersec for the instance
func (instance *Win32_PerfRawData_ESENT_DatabaseInstances) GetPropertyIOFlushMapWritesPersec() (value uint32, err error) {
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
func (instance *Win32_PerfRawData_ESENT_DatabaseInstances) SetPropertyIOLogReadsAverageLatency(value uint64) (err error) {
	return instance.SetProperty("IOLogReadsAverageLatency", (value))
}

// GetIOLogReadsAverageLatency gets the value of IOLogReadsAverageLatency for the instance
func (instance *Win32_PerfRawData_ESENT_DatabaseInstances) GetPropertyIOLogReadsAverageLatency() (value uint64, err error) {
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
func (instance *Win32_PerfRawData_ESENT_DatabaseInstances) SetPropertyIOLogReadsAverageLatency_Base(value uint32) (err error) {
	return instance.SetProperty("IOLogReadsAverageLatency_Base", (value))
}

// GetIOLogReadsAverageLatency_Base gets the value of IOLogReadsAverageLatency_Base for the instance
func (instance *Win32_PerfRawData_ESENT_DatabaseInstances) GetPropertyIOLogReadsAverageLatency_Base() (value uint32, err error) {
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
func (instance *Win32_PerfRawData_ESENT_DatabaseInstances) SetPropertyIOLogReadsPersec(value uint32) (err error) {
	return instance.SetProperty("IOLogReadsPersec", (value))
}

// GetIOLogReadsPersec gets the value of IOLogReadsPersec for the instance
func (instance *Win32_PerfRawData_ESENT_DatabaseInstances) GetPropertyIOLogReadsPersec() (value uint32, err error) {
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
func (instance *Win32_PerfRawData_ESENT_DatabaseInstances) SetPropertyIOLogWritesAverageLatency(value uint64) (err error) {
	return instance.SetProperty("IOLogWritesAverageLatency", (value))
}

// GetIOLogWritesAverageLatency gets the value of IOLogWritesAverageLatency for the instance
func (instance *Win32_PerfRawData_ESENT_DatabaseInstances) GetPropertyIOLogWritesAverageLatency() (value uint64, err error) {
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
func (instance *Win32_PerfRawData_ESENT_DatabaseInstances) SetPropertyIOLogWritesAverageLatency_Base(value uint32) (err error) {
	return instance.SetProperty("IOLogWritesAverageLatency_Base", (value))
}

// GetIOLogWritesAverageLatency_Base gets the value of IOLogWritesAverageLatency_Base for the instance
func (instance *Win32_PerfRawData_ESENT_DatabaseInstances) GetPropertyIOLogWritesAverageLatency_Base() (value uint32, err error) {
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
func (instance *Win32_PerfRawData_ESENT_DatabaseInstances) SetPropertyIOLogWritesPersec(value uint32) (err error) {
	return instance.SetProperty("IOLogWritesPersec", (value))
}

// GetIOLogWritesPersec gets the value of IOLogWritesPersec for the instance
func (instance *Win32_PerfRawData_ESENT_DatabaseInstances) GetPropertyIOLogWritesPersec() (value uint32, err error) {
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
func (instance *Win32_PerfRawData_ESENT_DatabaseInstances) SetPropertyLogBytesGeneratedPersec(value uint32) (err error) {
	return instance.SetProperty("LogBytesGeneratedPersec", (value))
}

// GetLogBytesGeneratedPersec gets the value of LogBytesGeneratedPersec for the instance
func (instance *Win32_PerfRawData_ESENT_DatabaseInstances) GetPropertyLogBytesGeneratedPersec() (value uint32, err error) {
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
func (instance *Win32_PerfRawData_ESENT_DatabaseInstances) SetPropertyLogBytesWritePersec(value uint32) (err error) {
	return instance.SetProperty("LogBytesWritePersec", (value))
}

// GetLogBytesWritePersec gets the value of LogBytesWritePersec for the instance
func (instance *Win32_PerfRawData_ESENT_DatabaseInstances) GetPropertyLogBytesWritePersec() (value uint32, err error) {
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

// SetLogCheckpointDepthasaPercentofTarget sets the value of LogCheckpointDepthasaPercentofTarget for the instance
func (instance *Win32_PerfRawData_ESENT_DatabaseInstances) SetPropertyLogCheckpointDepthasaPercentofTarget(value uint32) (err error) {
	return instance.SetProperty("LogCheckpointDepthasaPercentofTarget", (value))
}

// GetLogCheckpointDepthasaPercentofTarget gets the value of LogCheckpointDepthasaPercentofTarget for the instance
func (instance *Win32_PerfRawData_ESENT_DatabaseInstances) GetPropertyLogCheckpointDepthasaPercentofTarget() (value uint32, err error) {
	retValue, err := instance.GetProperty("LogCheckpointDepthasaPercentofTarget")
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

// SetLogCheckpointDepthasaPercentofTarget_Base sets the value of LogCheckpointDepthasaPercentofTarget_Base for the instance
func (instance *Win32_PerfRawData_ESENT_DatabaseInstances) SetPropertyLogCheckpointDepthasaPercentofTarget_Base(value uint32) (err error) {
	return instance.SetProperty("LogCheckpointDepthasaPercentofTarget_Base", (value))
}

// GetLogCheckpointDepthasaPercentofTarget_Base gets the value of LogCheckpointDepthasaPercentofTarget_Base for the instance
func (instance *Win32_PerfRawData_ESENT_DatabaseInstances) GetPropertyLogCheckpointDepthasaPercentofTarget_Base() (value uint32, err error) {
	retValue, err := instance.GetProperty("LogCheckpointDepthasaPercentofTarget_Base")
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

// SetLogFileCurrentGeneration sets the value of LogFileCurrentGeneration for the instance
func (instance *Win32_PerfRawData_ESENT_DatabaseInstances) SetPropertyLogFileCurrentGeneration(value uint32) (err error) {
	return instance.SetProperty("LogFileCurrentGeneration", (value))
}

// GetLogFileCurrentGeneration gets the value of LogFileCurrentGeneration for the instance
func (instance *Win32_PerfRawData_ESENT_DatabaseInstances) GetPropertyLogFileCurrentGeneration() (value uint32, err error) {
	retValue, err := instance.GetProperty("LogFileCurrentGeneration")
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

// SetLogFilesGenerated sets the value of LogFilesGenerated for the instance
func (instance *Win32_PerfRawData_ESENT_DatabaseInstances) SetPropertyLogFilesGenerated(value uint32) (err error) {
	return instance.SetProperty("LogFilesGenerated", (value))
}

// GetLogFilesGenerated gets the value of LogFilesGenerated for the instance
func (instance *Win32_PerfRawData_ESENT_DatabaseInstances) GetPropertyLogFilesGenerated() (value uint32, err error) {
	retValue, err := instance.GetProperty("LogFilesGenerated")
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

// SetLogFilesGeneratedPrematurely sets the value of LogFilesGeneratedPrematurely for the instance
func (instance *Win32_PerfRawData_ESENT_DatabaseInstances) SetPropertyLogFilesGeneratedPrematurely(value uint32) (err error) {
	return instance.SetProperty("LogFilesGeneratedPrematurely", (value))
}

// GetLogFilesGeneratedPrematurely gets the value of LogFilesGeneratedPrematurely for the instance
func (instance *Win32_PerfRawData_ESENT_DatabaseInstances) GetPropertyLogFilesGeneratedPrematurely() (value uint32, err error) {
	retValue, err := instance.GetProperty("LogFilesGeneratedPrematurely")
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

// SetLogGenerationCheckpointDepth sets the value of LogGenerationCheckpointDepth for the instance
func (instance *Win32_PerfRawData_ESENT_DatabaseInstances) SetPropertyLogGenerationCheckpointDepth(value uint32) (err error) {
	return instance.SetProperty("LogGenerationCheckpointDepth", (value))
}

// GetLogGenerationCheckpointDepth gets the value of LogGenerationCheckpointDepth for the instance
func (instance *Win32_PerfRawData_ESENT_DatabaseInstances) GetPropertyLogGenerationCheckpointDepth() (value uint32, err error) {
	retValue, err := instance.GetProperty("LogGenerationCheckpointDepth")
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

// SetLogGenerationCheckpointDepthMax sets the value of LogGenerationCheckpointDepthMax for the instance
func (instance *Win32_PerfRawData_ESENT_DatabaseInstances) SetPropertyLogGenerationCheckpointDepthMax(value uint32) (err error) {
	return instance.SetProperty("LogGenerationCheckpointDepthMax", (value))
}

// GetLogGenerationCheckpointDepthMax gets the value of LogGenerationCheckpointDepthMax for the instance
func (instance *Win32_PerfRawData_ESENT_DatabaseInstances) GetPropertyLogGenerationCheckpointDepthMax() (value uint32, err error) {
	retValue, err := instance.GetProperty("LogGenerationCheckpointDepthMax")
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

// SetLogGenerationCheckpointDepthTarget sets the value of LogGenerationCheckpointDepthTarget for the instance
func (instance *Win32_PerfRawData_ESENT_DatabaseInstances) SetPropertyLogGenerationCheckpointDepthTarget(value uint32) (err error) {
	return instance.SetProperty("LogGenerationCheckpointDepthTarget", (value))
}

// GetLogGenerationCheckpointDepthTarget gets the value of LogGenerationCheckpointDepthTarget for the instance
func (instance *Win32_PerfRawData_ESENT_DatabaseInstances) GetPropertyLogGenerationCheckpointDepthTarget() (value uint32, err error) {
	retValue, err := instance.GetProperty("LogGenerationCheckpointDepthTarget")
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

// SetLogGenerationLossResiliencyDepth sets the value of LogGenerationLossResiliencyDepth for the instance
func (instance *Win32_PerfRawData_ESENT_DatabaseInstances) SetPropertyLogGenerationLossResiliencyDepth(value uint32) (err error) {
	return instance.SetProperty("LogGenerationLossResiliencyDepth", (value))
}

// GetLogGenerationLossResiliencyDepth gets the value of LogGenerationLossResiliencyDepth for the instance
func (instance *Win32_PerfRawData_ESENT_DatabaseInstances) GetPropertyLogGenerationLossResiliencyDepth() (value uint32, err error) {
	retValue, err := instance.GetProperty("LogGenerationLossResiliencyDepth")
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
func (instance *Win32_PerfRawData_ESENT_DatabaseInstances) SetPropertyLogRecordStallsPersec(value uint32) (err error) {
	return instance.SetProperty("LogRecordStallsPersec", (value))
}

// GetLogRecordStallsPersec gets the value of LogRecordStallsPersec for the instance
func (instance *Win32_PerfRawData_ESENT_DatabaseInstances) GetPropertyLogRecordStallsPersec() (value uint32, err error) {
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
func (instance *Win32_PerfRawData_ESENT_DatabaseInstances) SetPropertyLogThreadsWaiting(value uint32) (err error) {
	return instance.SetProperty("LogThreadsWaiting", (value))
}

// GetLogThreadsWaiting gets the value of LogThreadsWaiting for the instance
func (instance *Win32_PerfRawData_ESENT_DatabaseInstances) GetPropertyLogThreadsWaiting() (value uint32, err error) {
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
func (instance *Win32_PerfRawData_ESENT_DatabaseInstances) SetPropertyLogWritesPersec(value uint32) (err error) {
	return instance.SetProperty("LogWritesPersec", (value))
}

// GetLogWritesPersec gets the value of LogWritesPersec for the instance
func (instance *Win32_PerfRawData_ESENT_DatabaseInstances) GetPropertyLogWritesPersec() (value uint32, err error) {
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
func (instance *Win32_PerfRawData_ESENT_DatabaseInstances) SetPropertySessionsInUse(value uint32) (err error) {
	return instance.SetProperty("SessionsInUse", (value))
}

// GetSessionsInUse gets the value of SessionsInUse for the instance
func (instance *Win32_PerfRawData_ESENT_DatabaseInstances) GetPropertySessionsInUse() (value uint32, err error) {
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
func (instance *Win32_PerfRawData_ESENT_DatabaseInstances) SetPropertySessionsPercentUsed(value uint32) (err error) {
	return instance.SetProperty("SessionsPercentUsed", (value))
}

// GetSessionsPercentUsed gets the value of SessionsPercentUsed for the instance
func (instance *Win32_PerfRawData_ESENT_DatabaseInstances) GetPropertySessionsPercentUsed() (value uint32, err error) {
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
func (instance *Win32_PerfRawData_ESENT_DatabaseInstances) SetPropertySessionsPercentUsed_Base(value uint32) (err error) {
	return instance.SetProperty("SessionsPercentUsed_Base", (value))
}

// GetSessionsPercentUsed_Base gets the value of SessionsPercentUsed_Base for the instance
func (instance *Win32_PerfRawData_ESENT_DatabaseInstances) GetPropertySessionsPercentUsed_Base() (value uint32, err error) {
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

// SetStreamingBackupPagesReadPersec sets the value of StreamingBackupPagesReadPersec for the instance
func (instance *Win32_PerfRawData_ESENT_DatabaseInstances) SetPropertyStreamingBackupPagesReadPersec(value uint32) (err error) {
	return instance.SetProperty("StreamingBackupPagesReadPersec", (value))
}

// GetStreamingBackupPagesReadPersec gets the value of StreamingBackupPagesReadPersec for the instance
func (instance *Win32_PerfRawData_ESENT_DatabaseInstances) GetPropertyStreamingBackupPagesReadPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("StreamingBackupPagesReadPersec")
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
func (instance *Win32_PerfRawData_ESENT_DatabaseInstances) SetPropertyTableClosesPersec(value uint32) (err error) {
	return instance.SetProperty("TableClosesPersec", (value))
}

// GetTableClosesPersec gets the value of TableClosesPersec for the instance
func (instance *Win32_PerfRawData_ESENT_DatabaseInstances) GetPropertyTableClosesPersec() (value uint32, err error) {
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
func (instance *Win32_PerfRawData_ESENT_DatabaseInstances) SetPropertyTableOpenCacheHitsPersec(value uint32) (err error) {
	return instance.SetProperty("TableOpenCacheHitsPersec", (value))
}

// GetTableOpenCacheHitsPersec gets the value of TableOpenCacheHitsPersec for the instance
func (instance *Win32_PerfRawData_ESENT_DatabaseInstances) GetPropertyTableOpenCacheHitsPersec() (value uint32, err error) {
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
func (instance *Win32_PerfRawData_ESENT_DatabaseInstances) SetPropertyTableOpenCacheMissesPersec(value uint32) (err error) {
	return instance.SetProperty("TableOpenCacheMissesPersec", (value))
}

// GetTableOpenCacheMissesPersec gets the value of TableOpenCacheMissesPersec for the instance
func (instance *Win32_PerfRawData_ESENT_DatabaseInstances) GetPropertyTableOpenCacheMissesPersec() (value uint32, err error) {
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
func (instance *Win32_PerfRawData_ESENT_DatabaseInstances) SetPropertyTableOpenCachePercentHit(value uint32) (err error) {
	return instance.SetProperty("TableOpenCachePercentHit", (value))
}

// GetTableOpenCachePercentHit gets the value of TableOpenCachePercentHit for the instance
func (instance *Win32_PerfRawData_ESENT_DatabaseInstances) GetPropertyTableOpenCachePercentHit() (value uint32, err error) {
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
func (instance *Win32_PerfRawData_ESENT_DatabaseInstances) SetPropertyTableOpenCachePercentHit_Base(value uint32) (err error) {
	return instance.SetProperty("TableOpenCachePercentHit_Base", (value))
}

// GetTableOpenCachePercentHit_Base gets the value of TableOpenCachePercentHit_Base for the instance
func (instance *Win32_PerfRawData_ESENT_DatabaseInstances) GetPropertyTableOpenCachePercentHit_Base() (value uint32, err error) {
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
func (instance *Win32_PerfRawData_ESENT_DatabaseInstances) SetPropertyTableOpensPersec(value uint32) (err error) {
	return instance.SetProperty("TableOpensPersec", (value))
}

// GetTableOpensPersec gets the value of TableOpensPersec for the instance
func (instance *Win32_PerfRawData_ESENT_DatabaseInstances) GetPropertyTableOpensPersec() (value uint32, err error) {
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
func (instance *Win32_PerfRawData_ESENT_DatabaseInstances) SetPropertyTablesOpen(value uint32) (err error) {
	return instance.SetProperty("TablesOpen", (value))
}

// GetTablesOpen gets the value of TablesOpen for the instance
func (instance *Win32_PerfRawData_ESENT_DatabaseInstances) GetPropertyTablesOpen() (value uint32, err error) {
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

// SetVersionbucketsallocated sets the value of Versionbucketsallocated for the instance
func (instance *Win32_PerfRawData_ESENT_DatabaseInstances) SetPropertyVersionbucketsallocated(value uint32) (err error) {
	return instance.SetProperty("Versionbucketsallocated", (value))
}

// GetVersionbucketsallocated gets the value of Versionbucketsallocated for the instance
func (instance *Win32_PerfRawData_ESENT_DatabaseInstances) GetPropertyVersionbucketsallocated() (value uint32, err error) {
	retValue, err := instance.GetProperty("Versionbucketsallocated")
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
