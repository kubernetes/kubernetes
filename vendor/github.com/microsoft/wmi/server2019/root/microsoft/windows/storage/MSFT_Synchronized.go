// Copyright 2019 (c) Microsoft Corporation.
// Licensed under the MIT license.

//
// Author:
//      Auto Generated on 9/18/2020 using wmigen
//      Source root.Microsoft.Windows.Storage
//////////////////////////////////////////////
package storage

import (
	"github.com/microsoft/wmi/pkg/base/instance"
	"github.com/microsoft/wmi/pkg/base/query"
	"github.com/microsoft/wmi/pkg/errors"
	cim "github.com/microsoft/wmi/pkg/wmiinstance"
	"reflect"
)

// MSFT_Synchronized struct
type MSFT_Synchronized struct {
	*cim.WmiInstance

	//
	CopyMethodology uint16

	//
	CopyPriority uint16

	//
	CopyState uint16

	//
	CopyType uint16

	//
	PercentSynced uint16

	//
	ProgressStatus uint16

	//
	RecoveryPointObjective uint32

	//
	ReplicaType uint16

	//
	RequestedCopyState uint16

	//
	SyncMaintained bool

	//
	SyncMode uint16

	//
	SyncState uint16

	//
	SyncTime string

	//
	SyncType uint16
}

func NewMSFT_SynchronizedEx1(instance *cim.WmiInstance) (newInstance *MSFT_Synchronized, err error) {
	tmp, err := instance, nil

	if err != nil {
		return
	}
	newInstance = &MSFT_Synchronized{
		WmiInstance: tmp,
	}
	return
}

func NewMSFT_SynchronizedEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *MSFT_Synchronized, err error) {
	tmp, err := instance.GetWmiInstance(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &MSFT_Synchronized{
		WmiInstance: tmp,
	}
	return
}

// SetCopyMethodology sets the value of CopyMethodology for the instance
func (instance *MSFT_Synchronized) SetPropertyCopyMethodology(value uint16) (err error) {
	return instance.SetProperty("CopyMethodology", (value))
}

// GetCopyMethodology gets the value of CopyMethodology for the instance
func (instance *MSFT_Synchronized) GetPropertyCopyMethodology() (value uint16, err error) {
	retValue, err := instance.GetProperty("CopyMethodology")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint16)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint16 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint16(valuetmp)

	return
}

// SetCopyPriority sets the value of CopyPriority for the instance
func (instance *MSFT_Synchronized) SetPropertyCopyPriority(value uint16) (err error) {
	return instance.SetProperty("CopyPriority", (value))
}

// GetCopyPriority gets the value of CopyPriority for the instance
func (instance *MSFT_Synchronized) GetPropertyCopyPriority() (value uint16, err error) {
	retValue, err := instance.GetProperty("CopyPriority")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint16)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint16 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint16(valuetmp)

	return
}

// SetCopyState sets the value of CopyState for the instance
func (instance *MSFT_Synchronized) SetPropertyCopyState(value uint16) (err error) {
	return instance.SetProperty("CopyState", (value))
}

// GetCopyState gets the value of CopyState for the instance
func (instance *MSFT_Synchronized) GetPropertyCopyState() (value uint16, err error) {
	retValue, err := instance.GetProperty("CopyState")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint16)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint16 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint16(valuetmp)

	return
}

// SetCopyType sets the value of CopyType for the instance
func (instance *MSFT_Synchronized) SetPropertyCopyType(value uint16) (err error) {
	return instance.SetProperty("CopyType", (value))
}

// GetCopyType gets the value of CopyType for the instance
func (instance *MSFT_Synchronized) GetPropertyCopyType() (value uint16, err error) {
	retValue, err := instance.GetProperty("CopyType")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint16)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint16 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint16(valuetmp)

	return
}

// SetPercentSynced sets the value of PercentSynced for the instance
func (instance *MSFT_Synchronized) SetPropertyPercentSynced(value uint16) (err error) {
	return instance.SetProperty("PercentSynced", (value))
}

// GetPercentSynced gets the value of PercentSynced for the instance
func (instance *MSFT_Synchronized) GetPropertyPercentSynced() (value uint16, err error) {
	retValue, err := instance.GetProperty("PercentSynced")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint16)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint16 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint16(valuetmp)

	return
}

// SetProgressStatus sets the value of ProgressStatus for the instance
func (instance *MSFT_Synchronized) SetPropertyProgressStatus(value uint16) (err error) {
	return instance.SetProperty("ProgressStatus", (value))
}

// GetProgressStatus gets the value of ProgressStatus for the instance
func (instance *MSFT_Synchronized) GetPropertyProgressStatus() (value uint16, err error) {
	retValue, err := instance.GetProperty("ProgressStatus")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint16)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint16 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint16(valuetmp)

	return
}

// SetRecoveryPointObjective sets the value of RecoveryPointObjective for the instance
func (instance *MSFT_Synchronized) SetPropertyRecoveryPointObjective(value uint32) (err error) {
	return instance.SetProperty("RecoveryPointObjective", (value))
}

// GetRecoveryPointObjective gets the value of RecoveryPointObjective for the instance
func (instance *MSFT_Synchronized) GetPropertyRecoveryPointObjective() (value uint32, err error) {
	retValue, err := instance.GetProperty("RecoveryPointObjective")
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

// SetReplicaType sets the value of ReplicaType for the instance
func (instance *MSFT_Synchronized) SetPropertyReplicaType(value uint16) (err error) {
	return instance.SetProperty("ReplicaType", (value))
}

// GetReplicaType gets the value of ReplicaType for the instance
func (instance *MSFT_Synchronized) GetPropertyReplicaType() (value uint16, err error) {
	retValue, err := instance.GetProperty("ReplicaType")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint16)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint16 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint16(valuetmp)

	return
}

// SetRequestedCopyState sets the value of RequestedCopyState for the instance
func (instance *MSFT_Synchronized) SetPropertyRequestedCopyState(value uint16) (err error) {
	return instance.SetProperty("RequestedCopyState", (value))
}

// GetRequestedCopyState gets the value of RequestedCopyState for the instance
func (instance *MSFT_Synchronized) GetPropertyRequestedCopyState() (value uint16, err error) {
	retValue, err := instance.GetProperty("RequestedCopyState")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint16)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint16 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint16(valuetmp)

	return
}

// SetSyncMaintained sets the value of SyncMaintained for the instance
func (instance *MSFT_Synchronized) SetPropertySyncMaintained(value bool) (err error) {
	return instance.SetProperty("SyncMaintained", (value))
}

// GetSyncMaintained gets the value of SyncMaintained for the instance
func (instance *MSFT_Synchronized) GetPropertySyncMaintained() (value bool, err error) {
	retValue, err := instance.GetProperty("SyncMaintained")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(bool)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " bool is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = bool(valuetmp)

	return
}

// SetSyncMode sets the value of SyncMode for the instance
func (instance *MSFT_Synchronized) SetPropertySyncMode(value uint16) (err error) {
	return instance.SetProperty("SyncMode", (value))
}

// GetSyncMode gets the value of SyncMode for the instance
func (instance *MSFT_Synchronized) GetPropertySyncMode() (value uint16, err error) {
	retValue, err := instance.GetProperty("SyncMode")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint16)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint16 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint16(valuetmp)

	return
}

// SetSyncState sets the value of SyncState for the instance
func (instance *MSFT_Synchronized) SetPropertySyncState(value uint16) (err error) {
	return instance.SetProperty("SyncState", (value))
}

// GetSyncState gets the value of SyncState for the instance
func (instance *MSFT_Synchronized) GetPropertySyncState() (value uint16, err error) {
	retValue, err := instance.GetProperty("SyncState")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint16)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint16 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint16(valuetmp)

	return
}

// SetSyncTime sets the value of SyncTime for the instance
func (instance *MSFT_Synchronized) SetPropertySyncTime(value string) (err error) {
	return instance.SetProperty("SyncTime", (value))
}

// GetSyncTime gets the value of SyncTime for the instance
func (instance *MSFT_Synchronized) GetPropertySyncTime() (value string, err error) {
	retValue, err := instance.GetProperty("SyncTime")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(string)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " string is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = string(valuetmp)

	return
}

// SetSyncType sets the value of SyncType for the instance
func (instance *MSFT_Synchronized) SetPropertySyncType(value uint16) (err error) {
	return instance.SetProperty("SyncType", (value))
}

// GetSyncType gets the value of SyncType for the instance
func (instance *MSFT_Synchronized) GetPropertySyncType() (value uint16, err error) {
	retValue, err := instance.GetProperty("SyncType")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint16)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint16 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint16(valuetmp)

	return
}
