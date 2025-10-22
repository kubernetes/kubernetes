// Copyright 2019 (c) Microsoft Corporation.
// Licensed under the MIT license.

//
// Author:
//      Auto Generated on 9/18/2020 using wmigen
//      Source root.CIMV2
//////////////////////////////////////////////
package cimv2

import (
	"github.com/microsoft/wmi/pkg/base/instance"
	"github.com/microsoft/wmi/pkg/base/query"
	"github.com/microsoft/wmi/pkg/errors"
	cim "github.com/microsoft/wmi/pkg/wmiinstance"
	"reflect"
)

// Win32_OfflineFilesBackgroundSync struct
type Win32_OfflineFilesBackgroundSync struct {
	*cim.WmiInstance

	//
	BackgroundSyncWorkOfflineSharesEnabled bool

	//
	BlockOutDurationMin uint16

	//
	BlockOutStartTimeHoursMinutes uint16

	//
	MaxTimeBetweenSyncs uint16

	//
	SyncInterval uint16

	//
	SyncVariance uint16
}

func NewWin32_OfflineFilesBackgroundSyncEx1(instance *cim.WmiInstance) (newInstance *Win32_OfflineFilesBackgroundSync, err error) {
	tmp, err := instance, nil

	if err != nil {
		return
	}
	newInstance = &Win32_OfflineFilesBackgroundSync{
		WmiInstance: tmp,
	}
	return
}

func NewWin32_OfflineFilesBackgroundSyncEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_OfflineFilesBackgroundSync, err error) {
	tmp, err := instance.GetWmiInstance(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_OfflineFilesBackgroundSync{
		WmiInstance: tmp,
	}
	return
}

// SetBackgroundSyncWorkOfflineSharesEnabled sets the value of BackgroundSyncWorkOfflineSharesEnabled for the instance
func (instance *Win32_OfflineFilesBackgroundSync) SetPropertyBackgroundSyncWorkOfflineSharesEnabled(value bool) (err error) {
	return instance.SetProperty("BackgroundSyncWorkOfflineSharesEnabled", (value))
}

// GetBackgroundSyncWorkOfflineSharesEnabled gets the value of BackgroundSyncWorkOfflineSharesEnabled for the instance
func (instance *Win32_OfflineFilesBackgroundSync) GetPropertyBackgroundSyncWorkOfflineSharesEnabled() (value bool, err error) {
	retValue, err := instance.GetProperty("BackgroundSyncWorkOfflineSharesEnabled")
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

// SetBlockOutDurationMin sets the value of BlockOutDurationMin for the instance
func (instance *Win32_OfflineFilesBackgroundSync) SetPropertyBlockOutDurationMin(value uint16) (err error) {
	return instance.SetProperty("BlockOutDurationMin", (value))
}

// GetBlockOutDurationMin gets the value of BlockOutDurationMin for the instance
func (instance *Win32_OfflineFilesBackgroundSync) GetPropertyBlockOutDurationMin() (value uint16, err error) {
	retValue, err := instance.GetProperty("BlockOutDurationMin")
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

// SetBlockOutStartTimeHoursMinutes sets the value of BlockOutStartTimeHoursMinutes for the instance
func (instance *Win32_OfflineFilesBackgroundSync) SetPropertyBlockOutStartTimeHoursMinutes(value uint16) (err error) {
	return instance.SetProperty("BlockOutStartTimeHoursMinutes", (value))
}

// GetBlockOutStartTimeHoursMinutes gets the value of BlockOutStartTimeHoursMinutes for the instance
func (instance *Win32_OfflineFilesBackgroundSync) GetPropertyBlockOutStartTimeHoursMinutes() (value uint16, err error) {
	retValue, err := instance.GetProperty("BlockOutStartTimeHoursMinutes")
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

// SetMaxTimeBetweenSyncs sets the value of MaxTimeBetweenSyncs for the instance
func (instance *Win32_OfflineFilesBackgroundSync) SetPropertyMaxTimeBetweenSyncs(value uint16) (err error) {
	return instance.SetProperty("MaxTimeBetweenSyncs", (value))
}

// GetMaxTimeBetweenSyncs gets the value of MaxTimeBetweenSyncs for the instance
func (instance *Win32_OfflineFilesBackgroundSync) GetPropertyMaxTimeBetweenSyncs() (value uint16, err error) {
	retValue, err := instance.GetProperty("MaxTimeBetweenSyncs")
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

// SetSyncInterval sets the value of SyncInterval for the instance
func (instance *Win32_OfflineFilesBackgroundSync) SetPropertySyncInterval(value uint16) (err error) {
	return instance.SetProperty("SyncInterval", (value))
}

// GetSyncInterval gets the value of SyncInterval for the instance
func (instance *Win32_OfflineFilesBackgroundSync) GetPropertySyncInterval() (value uint16, err error) {
	retValue, err := instance.GetProperty("SyncInterval")
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

// SetSyncVariance sets the value of SyncVariance for the instance
func (instance *Win32_OfflineFilesBackgroundSync) SetPropertySyncVariance(value uint16) (err error) {
	return instance.SetProperty("SyncVariance", (value))
}

// GetSyncVariance gets the value of SyncVariance for the instance
func (instance *Win32_OfflineFilesBackgroundSync) GetPropertySyncVariance() (value uint16, err error) {
	retValue, err := instance.GetProperty("SyncVariance")
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
