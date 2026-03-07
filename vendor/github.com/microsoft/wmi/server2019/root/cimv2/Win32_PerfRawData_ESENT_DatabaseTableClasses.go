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

// Win32_PerfRawData_ESENT_DatabaseTableClasses struct
type Win32_PerfRawData_ESENT_DatabaseTableClasses struct {
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
	DatabaseCacheSize uint64

	//
	DatabaseCacheSizeMB uint64
}

func NewWin32_PerfRawData_ESENT_DatabaseTableClassesEx1(instance *cim.WmiInstance) (newInstance *Win32_PerfRawData_ESENT_DatabaseTableClasses, err error) {
	tmp, err := NewWin32_PerfRawDataEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfRawData_ESENT_DatabaseTableClasses{
		Win32_PerfRawData: tmp,
	}
	return
}

func NewWin32_PerfRawData_ESENT_DatabaseTableClassesEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_PerfRawData_ESENT_DatabaseTableClasses, err error) {
	tmp, err := NewWin32_PerfRawDataEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfRawData_ESENT_DatabaseTableClasses{
		Win32_PerfRawData: tmp,
	}
	return
}

// SetDatabaseCacheMissAttachedAverageLatency sets the value of DatabaseCacheMissAttachedAverageLatency for the instance
func (instance *Win32_PerfRawData_ESENT_DatabaseTableClasses) SetPropertyDatabaseCacheMissAttachedAverageLatency(value uint64) (err error) {
	return instance.SetProperty("DatabaseCacheMissAttachedAverageLatency", (value))
}

// GetDatabaseCacheMissAttachedAverageLatency gets the value of DatabaseCacheMissAttachedAverageLatency for the instance
func (instance *Win32_PerfRawData_ESENT_DatabaseTableClasses) GetPropertyDatabaseCacheMissAttachedAverageLatency() (value uint64, err error) {
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
func (instance *Win32_PerfRawData_ESENT_DatabaseTableClasses) SetPropertyDatabaseCacheMissAttachedAverageLatency_Base(value uint32) (err error) {
	return instance.SetProperty("DatabaseCacheMissAttachedAverageLatency_Base", (value))
}

// GetDatabaseCacheMissAttachedAverageLatency_Base gets the value of DatabaseCacheMissAttachedAverageLatency_Base for the instance
func (instance *Win32_PerfRawData_ESENT_DatabaseTableClasses) GetPropertyDatabaseCacheMissAttachedAverageLatency_Base() (value uint32, err error) {
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
func (instance *Win32_PerfRawData_ESENT_DatabaseTableClasses) SetPropertyDatabaseCacheMissesPersec(value uint32) (err error) {
	return instance.SetProperty("DatabaseCacheMissesPersec", (value))
}

// GetDatabaseCacheMissesPersec gets the value of DatabaseCacheMissesPersec for the instance
func (instance *Win32_PerfRawData_ESENT_DatabaseTableClasses) GetPropertyDatabaseCacheMissesPersec() (value uint32, err error) {
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
func (instance *Win32_PerfRawData_ESENT_DatabaseTableClasses) SetPropertyDatabaseCachePercentHit(value uint32) (err error) {
	return instance.SetProperty("DatabaseCachePercentHit", (value))
}

// GetDatabaseCachePercentHit gets the value of DatabaseCachePercentHit for the instance
func (instance *Win32_PerfRawData_ESENT_DatabaseTableClasses) GetPropertyDatabaseCachePercentHit() (value uint32, err error) {
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
func (instance *Win32_PerfRawData_ESENT_DatabaseTableClasses) SetPropertyDatabaseCachePercentHit_Base(value uint32) (err error) {
	return instance.SetProperty("DatabaseCachePercentHit_Base", (value))
}

// GetDatabaseCachePercentHit_Base gets the value of DatabaseCachePercentHit_Base for the instance
func (instance *Win32_PerfRawData_ESENT_DatabaseTableClasses) GetPropertyDatabaseCachePercentHit_Base() (value uint32, err error) {
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
func (instance *Win32_PerfRawData_ESENT_DatabaseTableClasses) SetPropertyDatabaseCachePercentHitUnique(value uint32) (err error) {
	return instance.SetProperty("DatabaseCachePercentHitUnique", (value))
}

// GetDatabaseCachePercentHitUnique gets the value of DatabaseCachePercentHitUnique for the instance
func (instance *Win32_PerfRawData_ESENT_DatabaseTableClasses) GetPropertyDatabaseCachePercentHitUnique() (value uint32, err error) {
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
func (instance *Win32_PerfRawData_ESENT_DatabaseTableClasses) SetPropertyDatabaseCachePercentHitUnique_Base(value uint32) (err error) {
	return instance.SetProperty("DatabaseCachePercentHitUnique_Base", (value))
}

// GetDatabaseCachePercentHitUnique_Base gets the value of DatabaseCachePercentHitUnique_Base for the instance
func (instance *Win32_PerfRawData_ESENT_DatabaseTableClasses) GetPropertyDatabaseCachePercentHitUnique_Base() (value uint32, err error) {
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
func (instance *Win32_PerfRawData_ESENT_DatabaseTableClasses) SetPropertyDatabaseCacheRequestsPersec(value uint32) (err error) {
	return instance.SetProperty("DatabaseCacheRequestsPersec", (value))
}

// GetDatabaseCacheRequestsPersec gets the value of DatabaseCacheRequestsPersec for the instance
func (instance *Win32_PerfRawData_ESENT_DatabaseTableClasses) GetPropertyDatabaseCacheRequestsPersec() (value uint32, err error) {
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
func (instance *Win32_PerfRawData_ESENT_DatabaseTableClasses) SetPropertyDatabaseCacheRequestsPersecUnique(value uint32) (err error) {
	return instance.SetProperty("DatabaseCacheRequestsPersecUnique", (value))
}

// GetDatabaseCacheRequestsPersecUnique gets the value of DatabaseCacheRequestsPersecUnique for the instance
func (instance *Win32_PerfRawData_ESENT_DatabaseTableClasses) GetPropertyDatabaseCacheRequestsPersecUnique() (value uint32, err error) {
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
func (instance *Win32_PerfRawData_ESENT_DatabaseTableClasses) SetPropertyDatabaseCacheSize(value uint64) (err error) {
	return instance.SetProperty("DatabaseCacheSize", (value))
}

// GetDatabaseCacheSize gets the value of DatabaseCacheSize for the instance
func (instance *Win32_PerfRawData_ESENT_DatabaseTableClasses) GetPropertyDatabaseCacheSize() (value uint64, err error) {
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

// SetDatabaseCacheSizeMB sets the value of DatabaseCacheSizeMB for the instance
func (instance *Win32_PerfRawData_ESENT_DatabaseTableClasses) SetPropertyDatabaseCacheSizeMB(value uint64) (err error) {
	return instance.SetProperty("DatabaseCacheSizeMB", (value))
}

// GetDatabaseCacheSizeMB gets the value of DatabaseCacheSizeMB for the instance
func (instance *Win32_PerfRawData_ESENT_DatabaseTableClasses) GetPropertyDatabaseCacheSizeMB() (value uint64, err error) {
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
