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

// Win32_PerfRawData_Counters_StorageSpacesDrt struct
type Win32_PerfRawData_Counters_StorageSpacesDrt struct {
	*Win32_PerfRawData

	//
	CleanBytes uint64

	//
	CleanCandidateBytes uint64

	//
	CleanCandidateCount uint64

	//
	CleanCount uint64

	//
	DirtyBytes uint64

	//
	DirtyCount uint64

	//
	FlushingBytes uint64

	//
	FlushingCount uint64

	//
	Limit uint32

	//
	LockedBytes uint64

	//
	LockedCount uint64

	//
	NotTrackingBytes uint64

	//
	NotTrackingCount uint64

	//
	Status uint32

	//
	SynchronizingBytes uint64

	//
	SynchronizingCount uint64
}

func NewWin32_PerfRawData_Counters_StorageSpacesDrtEx1(instance *cim.WmiInstance) (newInstance *Win32_PerfRawData_Counters_StorageSpacesDrt, err error) {
	tmp, err := NewWin32_PerfRawDataEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfRawData_Counters_StorageSpacesDrt{
		Win32_PerfRawData: tmp,
	}
	return
}

func NewWin32_PerfRawData_Counters_StorageSpacesDrtEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_PerfRawData_Counters_StorageSpacesDrt, err error) {
	tmp, err := NewWin32_PerfRawDataEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfRawData_Counters_StorageSpacesDrt{
		Win32_PerfRawData: tmp,
	}
	return
}

// SetCleanBytes sets the value of CleanBytes for the instance
func (instance *Win32_PerfRawData_Counters_StorageSpacesDrt) SetPropertyCleanBytes(value uint64) (err error) {
	return instance.SetProperty("CleanBytes", (value))
}

// GetCleanBytes gets the value of CleanBytes for the instance
func (instance *Win32_PerfRawData_Counters_StorageSpacesDrt) GetPropertyCleanBytes() (value uint64, err error) {
	retValue, err := instance.GetProperty("CleanBytes")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint64)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint64 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint64(valuetmp)

	return
}

// SetCleanCandidateBytes sets the value of CleanCandidateBytes for the instance
func (instance *Win32_PerfRawData_Counters_StorageSpacesDrt) SetPropertyCleanCandidateBytes(value uint64) (err error) {
	return instance.SetProperty("CleanCandidateBytes", (value))
}

// GetCleanCandidateBytes gets the value of CleanCandidateBytes for the instance
func (instance *Win32_PerfRawData_Counters_StorageSpacesDrt) GetPropertyCleanCandidateBytes() (value uint64, err error) {
	retValue, err := instance.GetProperty("CleanCandidateBytes")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint64)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint64 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint64(valuetmp)

	return
}

// SetCleanCandidateCount sets the value of CleanCandidateCount for the instance
func (instance *Win32_PerfRawData_Counters_StorageSpacesDrt) SetPropertyCleanCandidateCount(value uint64) (err error) {
	return instance.SetProperty("CleanCandidateCount", (value))
}

// GetCleanCandidateCount gets the value of CleanCandidateCount for the instance
func (instance *Win32_PerfRawData_Counters_StorageSpacesDrt) GetPropertyCleanCandidateCount() (value uint64, err error) {
	retValue, err := instance.GetProperty("CleanCandidateCount")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint64)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint64 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint64(valuetmp)

	return
}

// SetCleanCount sets the value of CleanCount for the instance
func (instance *Win32_PerfRawData_Counters_StorageSpacesDrt) SetPropertyCleanCount(value uint64) (err error) {
	return instance.SetProperty("CleanCount", (value))
}

// GetCleanCount gets the value of CleanCount for the instance
func (instance *Win32_PerfRawData_Counters_StorageSpacesDrt) GetPropertyCleanCount() (value uint64, err error) {
	retValue, err := instance.GetProperty("CleanCount")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint64)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint64 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint64(valuetmp)

	return
}

// SetDirtyBytes sets the value of DirtyBytes for the instance
func (instance *Win32_PerfRawData_Counters_StorageSpacesDrt) SetPropertyDirtyBytes(value uint64) (err error) {
	return instance.SetProperty("DirtyBytes", (value))
}

// GetDirtyBytes gets the value of DirtyBytes for the instance
func (instance *Win32_PerfRawData_Counters_StorageSpacesDrt) GetPropertyDirtyBytes() (value uint64, err error) {
	retValue, err := instance.GetProperty("DirtyBytes")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint64)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint64 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint64(valuetmp)

	return
}

// SetDirtyCount sets the value of DirtyCount for the instance
func (instance *Win32_PerfRawData_Counters_StorageSpacesDrt) SetPropertyDirtyCount(value uint64) (err error) {
	return instance.SetProperty("DirtyCount", (value))
}

// GetDirtyCount gets the value of DirtyCount for the instance
func (instance *Win32_PerfRawData_Counters_StorageSpacesDrt) GetPropertyDirtyCount() (value uint64, err error) {
	retValue, err := instance.GetProperty("DirtyCount")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint64)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint64 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint64(valuetmp)

	return
}

// SetFlushingBytes sets the value of FlushingBytes for the instance
func (instance *Win32_PerfRawData_Counters_StorageSpacesDrt) SetPropertyFlushingBytes(value uint64) (err error) {
	return instance.SetProperty("FlushingBytes", (value))
}

// GetFlushingBytes gets the value of FlushingBytes for the instance
func (instance *Win32_PerfRawData_Counters_StorageSpacesDrt) GetPropertyFlushingBytes() (value uint64, err error) {
	retValue, err := instance.GetProperty("FlushingBytes")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint64)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint64 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint64(valuetmp)

	return
}

// SetFlushingCount sets the value of FlushingCount for the instance
func (instance *Win32_PerfRawData_Counters_StorageSpacesDrt) SetPropertyFlushingCount(value uint64) (err error) {
	return instance.SetProperty("FlushingCount", (value))
}

// GetFlushingCount gets the value of FlushingCount for the instance
func (instance *Win32_PerfRawData_Counters_StorageSpacesDrt) GetPropertyFlushingCount() (value uint64, err error) {
	retValue, err := instance.GetProperty("FlushingCount")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint64)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint64 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint64(valuetmp)

	return
}

// SetLimit sets the value of Limit for the instance
func (instance *Win32_PerfRawData_Counters_StorageSpacesDrt) SetPropertyLimit(value uint32) (err error) {
	return instance.SetProperty("Limit", (value))
}

// GetLimit gets the value of Limit for the instance
func (instance *Win32_PerfRawData_Counters_StorageSpacesDrt) GetPropertyLimit() (value uint32, err error) {
	retValue, err := instance.GetProperty("Limit")
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

// SetLockedBytes sets the value of LockedBytes for the instance
func (instance *Win32_PerfRawData_Counters_StorageSpacesDrt) SetPropertyLockedBytes(value uint64) (err error) {
	return instance.SetProperty("LockedBytes", (value))
}

// GetLockedBytes gets the value of LockedBytes for the instance
func (instance *Win32_PerfRawData_Counters_StorageSpacesDrt) GetPropertyLockedBytes() (value uint64, err error) {
	retValue, err := instance.GetProperty("LockedBytes")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint64)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint64 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint64(valuetmp)

	return
}

// SetLockedCount sets the value of LockedCount for the instance
func (instance *Win32_PerfRawData_Counters_StorageSpacesDrt) SetPropertyLockedCount(value uint64) (err error) {
	return instance.SetProperty("LockedCount", (value))
}

// GetLockedCount gets the value of LockedCount for the instance
func (instance *Win32_PerfRawData_Counters_StorageSpacesDrt) GetPropertyLockedCount() (value uint64, err error) {
	retValue, err := instance.GetProperty("LockedCount")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint64)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint64 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint64(valuetmp)

	return
}

// SetNotTrackingBytes sets the value of NotTrackingBytes for the instance
func (instance *Win32_PerfRawData_Counters_StorageSpacesDrt) SetPropertyNotTrackingBytes(value uint64) (err error) {
	return instance.SetProperty("NotTrackingBytes", (value))
}

// GetNotTrackingBytes gets the value of NotTrackingBytes for the instance
func (instance *Win32_PerfRawData_Counters_StorageSpacesDrt) GetPropertyNotTrackingBytes() (value uint64, err error) {
	retValue, err := instance.GetProperty("NotTrackingBytes")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint64)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint64 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint64(valuetmp)

	return
}

// SetNotTrackingCount sets the value of NotTrackingCount for the instance
func (instance *Win32_PerfRawData_Counters_StorageSpacesDrt) SetPropertyNotTrackingCount(value uint64) (err error) {
	return instance.SetProperty("NotTrackingCount", (value))
}

// GetNotTrackingCount gets the value of NotTrackingCount for the instance
func (instance *Win32_PerfRawData_Counters_StorageSpacesDrt) GetPropertyNotTrackingCount() (value uint64, err error) {
	retValue, err := instance.GetProperty("NotTrackingCount")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint64)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint64 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint64(valuetmp)

	return
}

// SetStatus sets the value of Status for the instance
func (instance *Win32_PerfRawData_Counters_StorageSpacesDrt) SetPropertyStatus(value uint32) (err error) {
	return instance.SetProperty("Status", (value))
}

// GetStatus gets the value of Status for the instance
func (instance *Win32_PerfRawData_Counters_StorageSpacesDrt) GetPropertyStatus() (value uint32, err error) {
	retValue, err := instance.GetProperty("Status")
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

// SetSynchronizingBytes sets the value of SynchronizingBytes for the instance
func (instance *Win32_PerfRawData_Counters_StorageSpacesDrt) SetPropertySynchronizingBytes(value uint64) (err error) {
	return instance.SetProperty("SynchronizingBytes", (value))
}

// GetSynchronizingBytes gets the value of SynchronizingBytes for the instance
func (instance *Win32_PerfRawData_Counters_StorageSpacesDrt) GetPropertySynchronizingBytes() (value uint64, err error) {
	retValue, err := instance.GetProperty("SynchronizingBytes")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint64)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint64 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint64(valuetmp)

	return
}

// SetSynchronizingCount sets the value of SynchronizingCount for the instance
func (instance *Win32_PerfRawData_Counters_StorageSpacesDrt) SetPropertySynchronizingCount(value uint64) (err error) {
	return instance.SetProperty("SynchronizingCount", (value))
}

// GetSynchronizingCount gets the value of SynchronizingCount for the instance
func (instance *Win32_PerfRawData_Counters_StorageSpacesDrt) GetPropertySynchronizingCount() (value uint64, err error) {
	retValue, err := instance.GetProperty("SynchronizingCount")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint64)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint64 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint64(valuetmp)

	return
}
