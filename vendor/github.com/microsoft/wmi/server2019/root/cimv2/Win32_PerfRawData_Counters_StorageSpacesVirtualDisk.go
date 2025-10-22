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

// Win32_PerfRawData_Counters_StorageSpacesVirtualDisk struct
type Win32_PerfRawData_Counters_StorageSpacesVirtualDisk struct {
	*Win32_PerfRawData

	//
	VirtualDiskActive uint64

	//
	VirtualDiskActiveBytes uint64

	//
	VirtualDiskFailedReplacementBytes uint64

	//
	VirtualDiskFailedReplacementCount uint64

	//
	VirtualDiskMissing uint64

	//
	VirtualDiskMissingBytes uint64

	//
	VirtualDiskNeedReallocation uint64

	//
	VirtualDiskNeedReallocationBytes uint64

	//
	VirtualDiskNeedRegeneration uint64

	//
	VirtualDiskNeedRegenerationBytes uint64

	//
	VirtualDiskPendingDeletion uint64

	//
	VirtualDiskPendingDeletionBytes uint64

	//
	VirtualDiskReasonFailure uint64

	//
	VirtualDiskReasonFailureBytes uint64

	//
	VirtualDiskReasonHardwareError uint64

	//
	VirtualDiskReasonHardwareErrorBytes uint64

	//
	VirtualDiskReasonIoError uint64

	//
	VirtualDiskReasonIoErrorBytes uint64

	//
	VirtualDiskReasonMissing uint64

	//
	VirtualDiskReasonMissingBytes uint64

	//
	VirtualDiskReasonNew uint64

	//
	VirtualDiskReasonNewBytes uint64

	//
	VirtualDiskReasonRegenReadError uint64

	//
	VirtualDiskReasonRegenReadErrorBytes uint64

	//
	VirtualDiskReasonRegenWriteError uint64

	//
	VirtualDiskReasonRegenWriteErrorBytes uint64

	//
	VirtualDiskReasonRetired uint64

	//
	VirtualDiskReasonRetiredBytes uint64

	//
	VirtualDiskRebalanceReplacementBytes uint64

	//
	VirtualDiskRebalanceReplacementCount uint64

	//
	VirtualDiskRegenerating uint64

	//
	VirtualDiskRegeneratingBytes uint64

	//
	VirtualDiskRepairNeedPhase2Count uint64

	//
	VirtualDiskRepairNeedPhase6Count uint64

	//
	VirtualDiskRepairPhase1Count uint64

	//
	VirtualDiskRepairPhase1Status uint64

	//
	VirtualDiskRepairPhase2Count uint64

	//
	VirtualDiskRepairPhase2Status uint64

	//
	VirtualDiskRepairPhase3Count uint64

	//
	VirtualDiskRepairPhase3Status uint64

	//
	VirtualDiskRepairPhase4Count uint64

	//
	VirtualDiskRepairPhase4Status uint64

	//
	VirtualDiskRepairPhase5Count uint64

	//
	VirtualDiskRepairPhase5Status uint64

	//
	VirtualDiskRepairPhase6Count uint64

	//
	VirtualDiskRepairPhase6Status uint64

	//
	VirtualDiskRepairReplacementBytes uint64

	//
	VirtualDiskRepairReplacementCount uint64

	//
	VirtualDiskScopeRegenerationBytes uint64

	//
	VirtualDiskScopeRegenerationCount uint64

	//
	VirtualDiskStale uint64

	//
	VirtualDiskStaleBytes uint64

	//
	VirtualDiskTotal uint64

	//
	VirtualDiskTotalBytes uint64
}

func NewWin32_PerfRawData_Counters_StorageSpacesVirtualDiskEx1(instance *cim.WmiInstance) (newInstance *Win32_PerfRawData_Counters_StorageSpacesVirtualDisk, err error) {
	tmp, err := NewWin32_PerfRawDataEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfRawData_Counters_StorageSpacesVirtualDisk{
		Win32_PerfRawData: tmp,
	}
	return
}

func NewWin32_PerfRawData_Counters_StorageSpacesVirtualDiskEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_PerfRawData_Counters_StorageSpacesVirtualDisk, err error) {
	tmp, err := NewWin32_PerfRawDataEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfRawData_Counters_StorageSpacesVirtualDisk{
		Win32_PerfRawData: tmp,
	}
	return
}

// SetVirtualDiskActive sets the value of VirtualDiskActive for the instance
func (instance *Win32_PerfRawData_Counters_StorageSpacesVirtualDisk) SetPropertyVirtualDiskActive(value uint64) (err error) {
	return instance.SetProperty("VirtualDiskActive", (value))
}

// GetVirtualDiskActive gets the value of VirtualDiskActive for the instance
func (instance *Win32_PerfRawData_Counters_StorageSpacesVirtualDisk) GetPropertyVirtualDiskActive() (value uint64, err error) {
	retValue, err := instance.GetProperty("VirtualDiskActive")
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

// SetVirtualDiskActiveBytes sets the value of VirtualDiskActiveBytes for the instance
func (instance *Win32_PerfRawData_Counters_StorageSpacesVirtualDisk) SetPropertyVirtualDiskActiveBytes(value uint64) (err error) {
	return instance.SetProperty("VirtualDiskActiveBytes", (value))
}

// GetVirtualDiskActiveBytes gets the value of VirtualDiskActiveBytes for the instance
func (instance *Win32_PerfRawData_Counters_StorageSpacesVirtualDisk) GetPropertyVirtualDiskActiveBytes() (value uint64, err error) {
	retValue, err := instance.GetProperty("VirtualDiskActiveBytes")
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

// SetVirtualDiskFailedReplacementBytes sets the value of VirtualDiskFailedReplacementBytes for the instance
func (instance *Win32_PerfRawData_Counters_StorageSpacesVirtualDisk) SetPropertyVirtualDiskFailedReplacementBytes(value uint64) (err error) {
	return instance.SetProperty("VirtualDiskFailedReplacementBytes", (value))
}

// GetVirtualDiskFailedReplacementBytes gets the value of VirtualDiskFailedReplacementBytes for the instance
func (instance *Win32_PerfRawData_Counters_StorageSpacesVirtualDisk) GetPropertyVirtualDiskFailedReplacementBytes() (value uint64, err error) {
	retValue, err := instance.GetProperty("VirtualDiskFailedReplacementBytes")
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

// SetVirtualDiskFailedReplacementCount sets the value of VirtualDiskFailedReplacementCount for the instance
func (instance *Win32_PerfRawData_Counters_StorageSpacesVirtualDisk) SetPropertyVirtualDiskFailedReplacementCount(value uint64) (err error) {
	return instance.SetProperty("VirtualDiskFailedReplacementCount", (value))
}

// GetVirtualDiskFailedReplacementCount gets the value of VirtualDiskFailedReplacementCount for the instance
func (instance *Win32_PerfRawData_Counters_StorageSpacesVirtualDisk) GetPropertyVirtualDiskFailedReplacementCount() (value uint64, err error) {
	retValue, err := instance.GetProperty("VirtualDiskFailedReplacementCount")
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

// SetVirtualDiskMissing sets the value of VirtualDiskMissing for the instance
func (instance *Win32_PerfRawData_Counters_StorageSpacesVirtualDisk) SetPropertyVirtualDiskMissing(value uint64) (err error) {
	return instance.SetProperty("VirtualDiskMissing", (value))
}

// GetVirtualDiskMissing gets the value of VirtualDiskMissing for the instance
func (instance *Win32_PerfRawData_Counters_StorageSpacesVirtualDisk) GetPropertyVirtualDiskMissing() (value uint64, err error) {
	retValue, err := instance.GetProperty("VirtualDiskMissing")
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

// SetVirtualDiskMissingBytes sets the value of VirtualDiskMissingBytes for the instance
func (instance *Win32_PerfRawData_Counters_StorageSpacesVirtualDisk) SetPropertyVirtualDiskMissingBytes(value uint64) (err error) {
	return instance.SetProperty("VirtualDiskMissingBytes", (value))
}

// GetVirtualDiskMissingBytes gets the value of VirtualDiskMissingBytes for the instance
func (instance *Win32_PerfRawData_Counters_StorageSpacesVirtualDisk) GetPropertyVirtualDiskMissingBytes() (value uint64, err error) {
	retValue, err := instance.GetProperty("VirtualDiskMissingBytes")
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

// SetVirtualDiskNeedReallocation sets the value of VirtualDiskNeedReallocation for the instance
func (instance *Win32_PerfRawData_Counters_StorageSpacesVirtualDisk) SetPropertyVirtualDiskNeedReallocation(value uint64) (err error) {
	return instance.SetProperty("VirtualDiskNeedReallocation", (value))
}

// GetVirtualDiskNeedReallocation gets the value of VirtualDiskNeedReallocation for the instance
func (instance *Win32_PerfRawData_Counters_StorageSpacesVirtualDisk) GetPropertyVirtualDiskNeedReallocation() (value uint64, err error) {
	retValue, err := instance.GetProperty("VirtualDiskNeedReallocation")
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

// SetVirtualDiskNeedReallocationBytes sets the value of VirtualDiskNeedReallocationBytes for the instance
func (instance *Win32_PerfRawData_Counters_StorageSpacesVirtualDisk) SetPropertyVirtualDiskNeedReallocationBytes(value uint64) (err error) {
	return instance.SetProperty("VirtualDiskNeedReallocationBytes", (value))
}

// GetVirtualDiskNeedReallocationBytes gets the value of VirtualDiskNeedReallocationBytes for the instance
func (instance *Win32_PerfRawData_Counters_StorageSpacesVirtualDisk) GetPropertyVirtualDiskNeedReallocationBytes() (value uint64, err error) {
	retValue, err := instance.GetProperty("VirtualDiskNeedReallocationBytes")
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

// SetVirtualDiskNeedRegeneration sets the value of VirtualDiskNeedRegeneration for the instance
func (instance *Win32_PerfRawData_Counters_StorageSpacesVirtualDisk) SetPropertyVirtualDiskNeedRegeneration(value uint64) (err error) {
	return instance.SetProperty("VirtualDiskNeedRegeneration", (value))
}

// GetVirtualDiskNeedRegeneration gets the value of VirtualDiskNeedRegeneration for the instance
func (instance *Win32_PerfRawData_Counters_StorageSpacesVirtualDisk) GetPropertyVirtualDiskNeedRegeneration() (value uint64, err error) {
	retValue, err := instance.GetProperty("VirtualDiskNeedRegeneration")
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

// SetVirtualDiskNeedRegenerationBytes sets the value of VirtualDiskNeedRegenerationBytes for the instance
func (instance *Win32_PerfRawData_Counters_StorageSpacesVirtualDisk) SetPropertyVirtualDiskNeedRegenerationBytes(value uint64) (err error) {
	return instance.SetProperty("VirtualDiskNeedRegenerationBytes", (value))
}

// GetVirtualDiskNeedRegenerationBytes gets the value of VirtualDiskNeedRegenerationBytes for the instance
func (instance *Win32_PerfRawData_Counters_StorageSpacesVirtualDisk) GetPropertyVirtualDiskNeedRegenerationBytes() (value uint64, err error) {
	retValue, err := instance.GetProperty("VirtualDiskNeedRegenerationBytes")
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

// SetVirtualDiskPendingDeletion sets the value of VirtualDiskPendingDeletion for the instance
func (instance *Win32_PerfRawData_Counters_StorageSpacesVirtualDisk) SetPropertyVirtualDiskPendingDeletion(value uint64) (err error) {
	return instance.SetProperty("VirtualDiskPendingDeletion", (value))
}

// GetVirtualDiskPendingDeletion gets the value of VirtualDiskPendingDeletion for the instance
func (instance *Win32_PerfRawData_Counters_StorageSpacesVirtualDisk) GetPropertyVirtualDiskPendingDeletion() (value uint64, err error) {
	retValue, err := instance.GetProperty("VirtualDiskPendingDeletion")
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

// SetVirtualDiskPendingDeletionBytes sets the value of VirtualDiskPendingDeletionBytes for the instance
func (instance *Win32_PerfRawData_Counters_StorageSpacesVirtualDisk) SetPropertyVirtualDiskPendingDeletionBytes(value uint64) (err error) {
	return instance.SetProperty("VirtualDiskPendingDeletionBytes", (value))
}

// GetVirtualDiskPendingDeletionBytes gets the value of VirtualDiskPendingDeletionBytes for the instance
func (instance *Win32_PerfRawData_Counters_StorageSpacesVirtualDisk) GetPropertyVirtualDiskPendingDeletionBytes() (value uint64, err error) {
	retValue, err := instance.GetProperty("VirtualDiskPendingDeletionBytes")
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

// SetVirtualDiskReasonFailure sets the value of VirtualDiskReasonFailure for the instance
func (instance *Win32_PerfRawData_Counters_StorageSpacesVirtualDisk) SetPropertyVirtualDiskReasonFailure(value uint64) (err error) {
	return instance.SetProperty("VirtualDiskReasonFailure", (value))
}

// GetVirtualDiskReasonFailure gets the value of VirtualDiskReasonFailure for the instance
func (instance *Win32_PerfRawData_Counters_StorageSpacesVirtualDisk) GetPropertyVirtualDiskReasonFailure() (value uint64, err error) {
	retValue, err := instance.GetProperty("VirtualDiskReasonFailure")
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

// SetVirtualDiskReasonFailureBytes sets the value of VirtualDiskReasonFailureBytes for the instance
func (instance *Win32_PerfRawData_Counters_StorageSpacesVirtualDisk) SetPropertyVirtualDiskReasonFailureBytes(value uint64) (err error) {
	return instance.SetProperty("VirtualDiskReasonFailureBytes", (value))
}

// GetVirtualDiskReasonFailureBytes gets the value of VirtualDiskReasonFailureBytes for the instance
func (instance *Win32_PerfRawData_Counters_StorageSpacesVirtualDisk) GetPropertyVirtualDiskReasonFailureBytes() (value uint64, err error) {
	retValue, err := instance.GetProperty("VirtualDiskReasonFailureBytes")
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

// SetVirtualDiskReasonHardwareError sets the value of VirtualDiskReasonHardwareError for the instance
func (instance *Win32_PerfRawData_Counters_StorageSpacesVirtualDisk) SetPropertyVirtualDiskReasonHardwareError(value uint64) (err error) {
	return instance.SetProperty("VirtualDiskReasonHardwareError", (value))
}

// GetVirtualDiskReasonHardwareError gets the value of VirtualDiskReasonHardwareError for the instance
func (instance *Win32_PerfRawData_Counters_StorageSpacesVirtualDisk) GetPropertyVirtualDiskReasonHardwareError() (value uint64, err error) {
	retValue, err := instance.GetProperty("VirtualDiskReasonHardwareError")
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

// SetVirtualDiskReasonHardwareErrorBytes sets the value of VirtualDiskReasonHardwareErrorBytes for the instance
func (instance *Win32_PerfRawData_Counters_StorageSpacesVirtualDisk) SetPropertyVirtualDiskReasonHardwareErrorBytes(value uint64) (err error) {
	return instance.SetProperty("VirtualDiskReasonHardwareErrorBytes", (value))
}

// GetVirtualDiskReasonHardwareErrorBytes gets the value of VirtualDiskReasonHardwareErrorBytes for the instance
func (instance *Win32_PerfRawData_Counters_StorageSpacesVirtualDisk) GetPropertyVirtualDiskReasonHardwareErrorBytes() (value uint64, err error) {
	retValue, err := instance.GetProperty("VirtualDiskReasonHardwareErrorBytes")
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

// SetVirtualDiskReasonIoError sets the value of VirtualDiskReasonIoError for the instance
func (instance *Win32_PerfRawData_Counters_StorageSpacesVirtualDisk) SetPropertyVirtualDiskReasonIoError(value uint64) (err error) {
	return instance.SetProperty("VirtualDiskReasonIoError", (value))
}

// GetVirtualDiskReasonIoError gets the value of VirtualDiskReasonIoError for the instance
func (instance *Win32_PerfRawData_Counters_StorageSpacesVirtualDisk) GetPropertyVirtualDiskReasonIoError() (value uint64, err error) {
	retValue, err := instance.GetProperty("VirtualDiskReasonIoError")
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

// SetVirtualDiskReasonIoErrorBytes sets the value of VirtualDiskReasonIoErrorBytes for the instance
func (instance *Win32_PerfRawData_Counters_StorageSpacesVirtualDisk) SetPropertyVirtualDiskReasonIoErrorBytes(value uint64) (err error) {
	return instance.SetProperty("VirtualDiskReasonIoErrorBytes", (value))
}

// GetVirtualDiskReasonIoErrorBytes gets the value of VirtualDiskReasonIoErrorBytes for the instance
func (instance *Win32_PerfRawData_Counters_StorageSpacesVirtualDisk) GetPropertyVirtualDiskReasonIoErrorBytes() (value uint64, err error) {
	retValue, err := instance.GetProperty("VirtualDiskReasonIoErrorBytes")
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

// SetVirtualDiskReasonMissing sets the value of VirtualDiskReasonMissing for the instance
func (instance *Win32_PerfRawData_Counters_StorageSpacesVirtualDisk) SetPropertyVirtualDiskReasonMissing(value uint64) (err error) {
	return instance.SetProperty("VirtualDiskReasonMissing", (value))
}

// GetVirtualDiskReasonMissing gets the value of VirtualDiskReasonMissing for the instance
func (instance *Win32_PerfRawData_Counters_StorageSpacesVirtualDisk) GetPropertyVirtualDiskReasonMissing() (value uint64, err error) {
	retValue, err := instance.GetProperty("VirtualDiskReasonMissing")
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

// SetVirtualDiskReasonMissingBytes sets the value of VirtualDiskReasonMissingBytes for the instance
func (instance *Win32_PerfRawData_Counters_StorageSpacesVirtualDisk) SetPropertyVirtualDiskReasonMissingBytes(value uint64) (err error) {
	return instance.SetProperty("VirtualDiskReasonMissingBytes", (value))
}

// GetVirtualDiskReasonMissingBytes gets the value of VirtualDiskReasonMissingBytes for the instance
func (instance *Win32_PerfRawData_Counters_StorageSpacesVirtualDisk) GetPropertyVirtualDiskReasonMissingBytes() (value uint64, err error) {
	retValue, err := instance.GetProperty("VirtualDiskReasonMissingBytes")
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

// SetVirtualDiskReasonNew sets the value of VirtualDiskReasonNew for the instance
func (instance *Win32_PerfRawData_Counters_StorageSpacesVirtualDisk) SetPropertyVirtualDiskReasonNew(value uint64) (err error) {
	return instance.SetProperty("VirtualDiskReasonNew", (value))
}

// GetVirtualDiskReasonNew gets the value of VirtualDiskReasonNew for the instance
func (instance *Win32_PerfRawData_Counters_StorageSpacesVirtualDisk) GetPropertyVirtualDiskReasonNew() (value uint64, err error) {
	retValue, err := instance.GetProperty("VirtualDiskReasonNew")
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

// SetVirtualDiskReasonNewBytes sets the value of VirtualDiskReasonNewBytes for the instance
func (instance *Win32_PerfRawData_Counters_StorageSpacesVirtualDisk) SetPropertyVirtualDiskReasonNewBytes(value uint64) (err error) {
	return instance.SetProperty("VirtualDiskReasonNewBytes", (value))
}

// GetVirtualDiskReasonNewBytes gets the value of VirtualDiskReasonNewBytes for the instance
func (instance *Win32_PerfRawData_Counters_StorageSpacesVirtualDisk) GetPropertyVirtualDiskReasonNewBytes() (value uint64, err error) {
	retValue, err := instance.GetProperty("VirtualDiskReasonNewBytes")
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

// SetVirtualDiskReasonRegenReadError sets the value of VirtualDiskReasonRegenReadError for the instance
func (instance *Win32_PerfRawData_Counters_StorageSpacesVirtualDisk) SetPropertyVirtualDiskReasonRegenReadError(value uint64) (err error) {
	return instance.SetProperty("VirtualDiskReasonRegenReadError", (value))
}

// GetVirtualDiskReasonRegenReadError gets the value of VirtualDiskReasonRegenReadError for the instance
func (instance *Win32_PerfRawData_Counters_StorageSpacesVirtualDisk) GetPropertyVirtualDiskReasonRegenReadError() (value uint64, err error) {
	retValue, err := instance.GetProperty("VirtualDiskReasonRegenReadError")
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

// SetVirtualDiskReasonRegenReadErrorBytes sets the value of VirtualDiskReasonRegenReadErrorBytes for the instance
func (instance *Win32_PerfRawData_Counters_StorageSpacesVirtualDisk) SetPropertyVirtualDiskReasonRegenReadErrorBytes(value uint64) (err error) {
	return instance.SetProperty("VirtualDiskReasonRegenReadErrorBytes", (value))
}

// GetVirtualDiskReasonRegenReadErrorBytes gets the value of VirtualDiskReasonRegenReadErrorBytes for the instance
func (instance *Win32_PerfRawData_Counters_StorageSpacesVirtualDisk) GetPropertyVirtualDiskReasonRegenReadErrorBytes() (value uint64, err error) {
	retValue, err := instance.GetProperty("VirtualDiskReasonRegenReadErrorBytes")
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

// SetVirtualDiskReasonRegenWriteError sets the value of VirtualDiskReasonRegenWriteError for the instance
func (instance *Win32_PerfRawData_Counters_StorageSpacesVirtualDisk) SetPropertyVirtualDiskReasonRegenWriteError(value uint64) (err error) {
	return instance.SetProperty("VirtualDiskReasonRegenWriteError", (value))
}

// GetVirtualDiskReasonRegenWriteError gets the value of VirtualDiskReasonRegenWriteError for the instance
func (instance *Win32_PerfRawData_Counters_StorageSpacesVirtualDisk) GetPropertyVirtualDiskReasonRegenWriteError() (value uint64, err error) {
	retValue, err := instance.GetProperty("VirtualDiskReasonRegenWriteError")
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

// SetVirtualDiskReasonRegenWriteErrorBytes sets the value of VirtualDiskReasonRegenWriteErrorBytes for the instance
func (instance *Win32_PerfRawData_Counters_StorageSpacesVirtualDisk) SetPropertyVirtualDiskReasonRegenWriteErrorBytes(value uint64) (err error) {
	return instance.SetProperty("VirtualDiskReasonRegenWriteErrorBytes", (value))
}

// GetVirtualDiskReasonRegenWriteErrorBytes gets the value of VirtualDiskReasonRegenWriteErrorBytes for the instance
func (instance *Win32_PerfRawData_Counters_StorageSpacesVirtualDisk) GetPropertyVirtualDiskReasonRegenWriteErrorBytes() (value uint64, err error) {
	retValue, err := instance.GetProperty("VirtualDiskReasonRegenWriteErrorBytes")
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

// SetVirtualDiskReasonRetired sets the value of VirtualDiskReasonRetired for the instance
func (instance *Win32_PerfRawData_Counters_StorageSpacesVirtualDisk) SetPropertyVirtualDiskReasonRetired(value uint64) (err error) {
	return instance.SetProperty("VirtualDiskReasonRetired", (value))
}

// GetVirtualDiskReasonRetired gets the value of VirtualDiskReasonRetired for the instance
func (instance *Win32_PerfRawData_Counters_StorageSpacesVirtualDisk) GetPropertyVirtualDiskReasonRetired() (value uint64, err error) {
	retValue, err := instance.GetProperty("VirtualDiskReasonRetired")
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

// SetVirtualDiskReasonRetiredBytes sets the value of VirtualDiskReasonRetiredBytes for the instance
func (instance *Win32_PerfRawData_Counters_StorageSpacesVirtualDisk) SetPropertyVirtualDiskReasonRetiredBytes(value uint64) (err error) {
	return instance.SetProperty("VirtualDiskReasonRetiredBytes", (value))
}

// GetVirtualDiskReasonRetiredBytes gets the value of VirtualDiskReasonRetiredBytes for the instance
func (instance *Win32_PerfRawData_Counters_StorageSpacesVirtualDisk) GetPropertyVirtualDiskReasonRetiredBytes() (value uint64, err error) {
	retValue, err := instance.GetProperty("VirtualDiskReasonRetiredBytes")
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

// SetVirtualDiskRebalanceReplacementBytes sets the value of VirtualDiskRebalanceReplacementBytes for the instance
func (instance *Win32_PerfRawData_Counters_StorageSpacesVirtualDisk) SetPropertyVirtualDiskRebalanceReplacementBytes(value uint64) (err error) {
	return instance.SetProperty("VirtualDiskRebalanceReplacementBytes", (value))
}

// GetVirtualDiskRebalanceReplacementBytes gets the value of VirtualDiskRebalanceReplacementBytes for the instance
func (instance *Win32_PerfRawData_Counters_StorageSpacesVirtualDisk) GetPropertyVirtualDiskRebalanceReplacementBytes() (value uint64, err error) {
	retValue, err := instance.GetProperty("VirtualDiskRebalanceReplacementBytes")
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

// SetVirtualDiskRebalanceReplacementCount sets the value of VirtualDiskRebalanceReplacementCount for the instance
func (instance *Win32_PerfRawData_Counters_StorageSpacesVirtualDisk) SetPropertyVirtualDiskRebalanceReplacementCount(value uint64) (err error) {
	return instance.SetProperty("VirtualDiskRebalanceReplacementCount", (value))
}

// GetVirtualDiskRebalanceReplacementCount gets the value of VirtualDiskRebalanceReplacementCount for the instance
func (instance *Win32_PerfRawData_Counters_StorageSpacesVirtualDisk) GetPropertyVirtualDiskRebalanceReplacementCount() (value uint64, err error) {
	retValue, err := instance.GetProperty("VirtualDiskRebalanceReplacementCount")
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

// SetVirtualDiskRegenerating sets the value of VirtualDiskRegenerating for the instance
func (instance *Win32_PerfRawData_Counters_StorageSpacesVirtualDisk) SetPropertyVirtualDiskRegenerating(value uint64) (err error) {
	return instance.SetProperty("VirtualDiskRegenerating", (value))
}

// GetVirtualDiskRegenerating gets the value of VirtualDiskRegenerating for the instance
func (instance *Win32_PerfRawData_Counters_StorageSpacesVirtualDisk) GetPropertyVirtualDiskRegenerating() (value uint64, err error) {
	retValue, err := instance.GetProperty("VirtualDiskRegenerating")
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

// SetVirtualDiskRegeneratingBytes sets the value of VirtualDiskRegeneratingBytes for the instance
func (instance *Win32_PerfRawData_Counters_StorageSpacesVirtualDisk) SetPropertyVirtualDiskRegeneratingBytes(value uint64) (err error) {
	return instance.SetProperty("VirtualDiskRegeneratingBytes", (value))
}

// GetVirtualDiskRegeneratingBytes gets the value of VirtualDiskRegeneratingBytes for the instance
func (instance *Win32_PerfRawData_Counters_StorageSpacesVirtualDisk) GetPropertyVirtualDiskRegeneratingBytes() (value uint64, err error) {
	retValue, err := instance.GetProperty("VirtualDiskRegeneratingBytes")
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

// SetVirtualDiskRepairNeedPhase2Count sets the value of VirtualDiskRepairNeedPhase2Count for the instance
func (instance *Win32_PerfRawData_Counters_StorageSpacesVirtualDisk) SetPropertyVirtualDiskRepairNeedPhase2Count(value uint64) (err error) {
	return instance.SetProperty("VirtualDiskRepairNeedPhase2Count", (value))
}

// GetVirtualDiskRepairNeedPhase2Count gets the value of VirtualDiskRepairNeedPhase2Count for the instance
func (instance *Win32_PerfRawData_Counters_StorageSpacesVirtualDisk) GetPropertyVirtualDiskRepairNeedPhase2Count() (value uint64, err error) {
	retValue, err := instance.GetProperty("VirtualDiskRepairNeedPhase2Count")
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

// SetVirtualDiskRepairNeedPhase6Count sets the value of VirtualDiskRepairNeedPhase6Count for the instance
func (instance *Win32_PerfRawData_Counters_StorageSpacesVirtualDisk) SetPropertyVirtualDiskRepairNeedPhase6Count(value uint64) (err error) {
	return instance.SetProperty("VirtualDiskRepairNeedPhase6Count", (value))
}

// GetVirtualDiskRepairNeedPhase6Count gets the value of VirtualDiskRepairNeedPhase6Count for the instance
func (instance *Win32_PerfRawData_Counters_StorageSpacesVirtualDisk) GetPropertyVirtualDiskRepairNeedPhase6Count() (value uint64, err error) {
	retValue, err := instance.GetProperty("VirtualDiskRepairNeedPhase6Count")
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

// SetVirtualDiskRepairPhase1Count sets the value of VirtualDiskRepairPhase1Count for the instance
func (instance *Win32_PerfRawData_Counters_StorageSpacesVirtualDisk) SetPropertyVirtualDiskRepairPhase1Count(value uint64) (err error) {
	return instance.SetProperty("VirtualDiskRepairPhase1Count", (value))
}

// GetVirtualDiskRepairPhase1Count gets the value of VirtualDiskRepairPhase1Count for the instance
func (instance *Win32_PerfRawData_Counters_StorageSpacesVirtualDisk) GetPropertyVirtualDiskRepairPhase1Count() (value uint64, err error) {
	retValue, err := instance.GetProperty("VirtualDiskRepairPhase1Count")
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

// SetVirtualDiskRepairPhase1Status sets the value of VirtualDiskRepairPhase1Status for the instance
func (instance *Win32_PerfRawData_Counters_StorageSpacesVirtualDisk) SetPropertyVirtualDiskRepairPhase1Status(value uint64) (err error) {
	return instance.SetProperty("VirtualDiskRepairPhase1Status", (value))
}

// GetVirtualDiskRepairPhase1Status gets the value of VirtualDiskRepairPhase1Status for the instance
func (instance *Win32_PerfRawData_Counters_StorageSpacesVirtualDisk) GetPropertyVirtualDiskRepairPhase1Status() (value uint64, err error) {
	retValue, err := instance.GetProperty("VirtualDiskRepairPhase1Status")
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

// SetVirtualDiskRepairPhase2Count sets the value of VirtualDiskRepairPhase2Count for the instance
func (instance *Win32_PerfRawData_Counters_StorageSpacesVirtualDisk) SetPropertyVirtualDiskRepairPhase2Count(value uint64) (err error) {
	return instance.SetProperty("VirtualDiskRepairPhase2Count", (value))
}

// GetVirtualDiskRepairPhase2Count gets the value of VirtualDiskRepairPhase2Count for the instance
func (instance *Win32_PerfRawData_Counters_StorageSpacesVirtualDisk) GetPropertyVirtualDiskRepairPhase2Count() (value uint64, err error) {
	retValue, err := instance.GetProperty("VirtualDiskRepairPhase2Count")
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

// SetVirtualDiskRepairPhase2Status sets the value of VirtualDiskRepairPhase2Status for the instance
func (instance *Win32_PerfRawData_Counters_StorageSpacesVirtualDisk) SetPropertyVirtualDiskRepairPhase2Status(value uint64) (err error) {
	return instance.SetProperty("VirtualDiskRepairPhase2Status", (value))
}

// GetVirtualDiskRepairPhase2Status gets the value of VirtualDiskRepairPhase2Status for the instance
func (instance *Win32_PerfRawData_Counters_StorageSpacesVirtualDisk) GetPropertyVirtualDiskRepairPhase2Status() (value uint64, err error) {
	retValue, err := instance.GetProperty("VirtualDiskRepairPhase2Status")
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

// SetVirtualDiskRepairPhase3Count sets the value of VirtualDiskRepairPhase3Count for the instance
func (instance *Win32_PerfRawData_Counters_StorageSpacesVirtualDisk) SetPropertyVirtualDiskRepairPhase3Count(value uint64) (err error) {
	return instance.SetProperty("VirtualDiskRepairPhase3Count", (value))
}

// GetVirtualDiskRepairPhase3Count gets the value of VirtualDiskRepairPhase3Count for the instance
func (instance *Win32_PerfRawData_Counters_StorageSpacesVirtualDisk) GetPropertyVirtualDiskRepairPhase3Count() (value uint64, err error) {
	retValue, err := instance.GetProperty("VirtualDiskRepairPhase3Count")
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

// SetVirtualDiskRepairPhase3Status sets the value of VirtualDiskRepairPhase3Status for the instance
func (instance *Win32_PerfRawData_Counters_StorageSpacesVirtualDisk) SetPropertyVirtualDiskRepairPhase3Status(value uint64) (err error) {
	return instance.SetProperty("VirtualDiskRepairPhase3Status", (value))
}

// GetVirtualDiskRepairPhase3Status gets the value of VirtualDiskRepairPhase3Status for the instance
func (instance *Win32_PerfRawData_Counters_StorageSpacesVirtualDisk) GetPropertyVirtualDiskRepairPhase3Status() (value uint64, err error) {
	retValue, err := instance.GetProperty("VirtualDiskRepairPhase3Status")
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

// SetVirtualDiskRepairPhase4Count sets the value of VirtualDiskRepairPhase4Count for the instance
func (instance *Win32_PerfRawData_Counters_StorageSpacesVirtualDisk) SetPropertyVirtualDiskRepairPhase4Count(value uint64) (err error) {
	return instance.SetProperty("VirtualDiskRepairPhase4Count", (value))
}

// GetVirtualDiskRepairPhase4Count gets the value of VirtualDiskRepairPhase4Count for the instance
func (instance *Win32_PerfRawData_Counters_StorageSpacesVirtualDisk) GetPropertyVirtualDiskRepairPhase4Count() (value uint64, err error) {
	retValue, err := instance.GetProperty("VirtualDiskRepairPhase4Count")
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

// SetVirtualDiskRepairPhase4Status sets the value of VirtualDiskRepairPhase4Status for the instance
func (instance *Win32_PerfRawData_Counters_StorageSpacesVirtualDisk) SetPropertyVirtualDiskRepairPhase4Status(value uint64) (err error) {
	return instance.SetProperty("VirtualDiskRepairPhase4Status", (value))
}

// GetVirtualDiskRepairPhase4Status gets the value of VirtualDiskRepairPhase4Status for the instance
func (instance *Win32_PerfRawData_Counters_StorageSpacesVirtualDisk) GetPropertyVirtualDiskRepairPhase4Status() (value uint64, err error) {
	retValue, err := instance.GetProperty("VirtualDiskRepairPhase4Status")
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

// SetVirtualDiskRepairPhase5Count sets the value of VirtualDiskRepairPhase5Count for the instance
func (instance *Win32_PerfRawData_Counters_StorageSpacesVirtualDisk) SetPropertyVirtualDiskRepairPhase5Count(value uint64) (err error) {
	return instance.SetProperty("VirtualDiskRepairPhase5Count", (value))
}

// GetVirtualDiskRepairPhase5Count gets the value of VirtualDiskRepairPhase5Count for the instance
func (instance *Win32_PerfRawData_Counters_StorageSpacesVirtualDisk) GetPropertyVirtualDiskRepairPhase5Count() (value uint64, err error) {
	retValue, err := instance.GetProperty("VirtualDiskRepairPhase5Count")
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

// SetVirtualDiskRepairPhase5Status sets the value of VirtualDiskRepairPhase5Status for the instance
func (instance *Win32_PerfRawData_Counters_StorageSpacesVirtualDisk) SetPropertyVirtualDiskRepairPhase5Status(value uint64) (err error) {
	return instance.SetProperty("VirtualDiskRepairPhase5Status", (value))
}

// GetVirtualDiskRepairPhase5Status gets the value of VirtualDiskRepairPhase5Status for the instance
func (instance *Win32_PerfRawData_Counters_StorageSpacesVirtualDisk) GetPropertyVirtualDiskRepairPhase5Status() (value uint64, err error) {
	retValue, err := instance.GetProperty("VirtualDiskRepairPhase5Status")
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

// SetVirtualDiskRepairPhase6Count sets the value of VirtualDiskRepairPhase6Count for the instance
func (instance *Win32_PerfRawData_Counters_StorageSpacesVirtualDisk) SetPropertyVirtualDiskRepairPhase6Count(value uint64) (err error) {
	return instance.SetProperty("VirtualDiskRepairPhase6Count", (value))
}

// GetVirtualDiskRepairPhase6Count gets the value of VirtualDiskRepairPhase6Count for the instance
func (instance *Win32_PerfRawData_Counters_StorageSpacesVirtualDisk) GetPropertyVirtualDiskRepairPhase6Count() (value uint64, err error) {
	retValue, err := instance.GetProperty("VirtualDiskRepairPhase6Count")
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

// SetVirtualDiskRepairPhase6Status sets the value of VirtualDiskRepairPhase6Status for the instance
func (instance *Win32_PerfRawData_Counters_StorageSpacesVirtualDisk) SetPropertyVirtualDiskRepairPhase6Status(value uint64) (err error) {
	return instance.SetProperty("VirtualDiskRepairPhase6Status", (value))
}

// GetVirtualDiskRepairPhase6Status gets the value of VirtualDiskRepairPhase6Status for the instance
func (instance *Win32_PerfRawData_Counters_StorageSpacesVirtualDisk) GetPropertyVirtualDiskRepairPhase6Status() (value uint64, err error) {
	retValue, err := instance.GetProperty("VirtualDiskRepairPhase6Status")
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

// SetVirtualDiskRepairReplacementBytes sets the value of VirtualDiskRepairReplacementBytes for the instance
func (instance *Win32_PerfRawData_Counters_StorageSpacesVirtualDisk) SetPropertyVirtualDiskRepairReplacementBytes(value uint64) (err error) {
	return instance.SetProperty("VirtualDiskRepairReplacementBytes", (value))
}

// GetVirtualDiskRepairReplacementBytes gets the value of VirtualDiskRepairReplacementBytes for the instance
func (instance *Win32_PerfRawData_Counters_StorageSpacesVirtualDisk) GetPropertyVirtualDiskRepairReplacementBytes() (value uint64, err error) {
	retValue, err := instance.GetProperty("VirtualDiskRepairReplacementBytes")
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

// SetVirtualDiskRepairReplacementCount sets the value of VirtualDiskRepairReplacementCount for the instance
func (instance *Win32_PerfRawData_Counters_StorageSpacesVirtualDisk) SetPropertyVirtualDiskRepairReplacementCount(value uint64) (err error) {
	return instance.SetProperty("VirtualDiskRepairReplacementCount", (value))
}

// GetVirtualDiskRepairReplacementCount gets the value of VirtualDiskRepairReplacementCount for the instance
func (instance *Win32_PerfRawData_Counters_StorageSpacesVirtualDisk) GetPropertyVirtualDiskRepairReplacementCount() (value uint64, err error) {
	retValue, err := instance.GetProperty("VirtualDiskRepairReplacementCount")
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

// SetVirtualDiskScopeRegenerationBytes sets the value of VirtualDiskScopeRegenerationBytes for the instance
func (instance *Win32_PerfRawData_Counters_StorageSpacesVirtualDisk) SetPropertyVirtualDiskScopeRegenerationBytes(value uint64) (err error) {
	return instance.SetProperty("VirtualDiskScopeRegenerationBytes", (value))
}

// GetVirtualDiskScopeRegenerationBytes gets the value of VirtualDiskScopeRegenerationBytes for the instance
func (instance *Win32_PerfRawData_Counters_StorageSpacesVirtualDisk) GetPropertyVirtualDiskScopeRegenerationBytes() (value uint64, err error) {
	retValue, err := instance.GetProperty("VirtualDiskScopeRegenerationBytes")
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

// SetVirtualDiskScopeRegenerationCount sets the value of VirtualDiskScopeRegenerationCount for the instance
func (instance *Win32_PerfRawData_Counters_StorageSpacesVirtualDisk) SetPropertyVirtualDiskScopeRegenerationCount(value uint64) (err error) {
	return instance.SetProperty("VirtualDiskScopeRegenerationCount", (value))
}

// GetVirtualDiskScopeRegenerationCount gets the value of VirtualDiskScopeRegenerationCount for the instance
func (instance *Win32_PerfRawData_Counters_StorageSpacesVirtualDisk) GetPropertyVirtualDiskScopeRegenerationCount() (value uint64, err error) {
	retValue, err := instance.GetProperty("VirtualDiskScopeRegenerationCount")
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

// SetVirtualDiskStale sets the value of VirtualDiskStale for the instance
func (instance *Win32_PerfRawData_Counters_StorageSpacesVirtualDisk) SetPropertyVirtualDiskStale(value uint64) (err error) {
	return instance.SetProperty("VirtualDiskStale", (value))
}

// GetVirtualDiskStale gets the value of VirtualDiskStale for the instance
func (instance *Win32_PerfRawData_Counters_StorageSpacesVirtualDisk) GetPropertyVirtualDiskStale() (value uint64, err error) {
	retValue, err := instance.GetProperty("VirtualDiskStale")
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

// SetVirtualDiskStaleBytes sets the value of VirtualDiskStaleBytes for the instance
func (instance *Win32_PerfRawData_Counters_StorageSpacesVirtualDisk) SetPropertyVirtualDiskStaleBytes(value uint64) (err error) {
	return instance.SetProperty("VirtualDiskStaleBytes", (value))
}

// GetVirtualDiskStaleBytes gets the value of VirtualDiskStaleBytes for the instance
func (instance *Win32_PerfRawData_Counters_StorageSpacesVirtualDisk) GetPropertyVirtualDiskStaleBytes() (value uint64, err error) {
	retValue, err := instance.GetProperty("VirtualDiskStaleBytes")
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

// SetVirtualDiskTotal sets the value of VirtualDiskTotal for the instance
func (instance *Win32_PerfRawData_Counters_StorageSpacesVirtualDisk) SetPropertyVirtualDiskTotal(value uint64) (err error) {
	return instance.SetProperty("VirtualDiskTotal", (value))
}

// GetVirtualDiskTotal gets the value of VirtualDiskTotal for the instance
func (instance *Win32_PerfRawData_Counters_StorageSpacesVirtualDisk) GetPropertyVirtualDiskTotal() (value uint64, err error) {
	retValue, err := instance.GetProperty("VirtualDiskTotal")
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

// SetVirtualDiskTotalBytes sets the value of VirtualDiskTotalBytes for the instance
func (instance *Win32_PerfRawData_Counters_StorageSpacesVirtualDisk) SetPropertyVirtualDiskTotalBytes(value uint64) (err error) {
	return instance.SetProperty("VirtualDiskTotalBytes", (value))
}

// GetVirtualDiskTotalBytes gets the value of VirtualDiskTotalBytes for the instance
func (instance *Win32_PerfRawData_Counters_StorageSpacesVirtualDisk) GetPropertyVirtualDiskTotalBytes() (value uint64, err error) {
	retValue, err := instance.GetProperty("VirtualDiskTotalBytes")
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
