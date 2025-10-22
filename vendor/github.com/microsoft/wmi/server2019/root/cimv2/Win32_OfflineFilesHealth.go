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

// Win32_OfflineFilesHealth struct
type Win32_OfflineFilesHealth struct {
	*cim.WmiInstance

	// A DATETIME value, in string format, that represents the last time this folder was successfully synchronized to the Offline Files cache.
	LastSuccessfulSyncTime string

	// The status of the last attempt to synchronize this folder to the Offline Files cache.
	LastSyncStatus uint8

	// A DATETIME value, in string format, that represents the last time an attempt was made to synchronized this folder to the Offline Files cache, even if it was unsuccessful.
	LastSyncTime string

	// If true, the Offline Files feature is enabled for this folder.
	OfflineAccessEnabled bool

	// If true, the share is working in Online mode
	OnlineMode bool
}

func NewWin32_OfflineFilesHealthEx1(instance *cim.WmiInstance) (newInstance *Win32_OfflineFilesHealth, err error) {
	tmp, err := instance, nil

	if err != nil {
		return
	}
	newInstance = &Win32_OfflineFilesHealth{
		WmiInstance: tmp,
	}
	return
}

func NewWin32_OfflineFilesHealthEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_OfflineFilesHealth, err error) {
	tmp, err := instance.GetWmiInstance(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_OfflineFilesHealth{
		WmiInstance: tmp,
	}
	return
}

// SetLastSuccessfulSyncTime sets the value of LastSuccessfulSyncTime for the instance
func (instance *Win32_OfflineFilesHealth) SetPropertyLastSuccessfulSyncTime(value string) (err error) {
	return instance.SetProperty("LastSuccessfulSyncTime", (value))
}

// GetLastSuccessfulSyncTime gets the value of LastSuccessfulSyncTime for the instance
func (instance *Win32_OfflineFilesHealth) GetPropertyLastSuccessfulSyncTime() (value string, err error) {
	retValue, err := instance.GetProperty("LastSuccessfulSyncTime")
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

// SetLastSyncStatus sets the value of LastSyncStatus for the instance
func (instance *Win32_OfflineFilesHealth) SetPropertyLastSyncStatus(value uint8) (err error) {
	return instance.SetProperty("LastSyncStatus", (value))
}

// GetLastSyncStatus gets the value of LastSyncStatus for the instance
func (instance *Win32_OfflineFilesHealth) GetPropertyLastSyncStatus() (value uint8, err error) {
	retValue, err := instance.GetProperty("LastSyncStatus")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint8)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint8 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint8(valuetmp)

	return
}

// SetLastSyncTime sets the value of LastSyncTime for the instance
func (instance *Win32_OfflineFilesHealth) SetPropertyLastSyncTime(value string) (err error) {
	return instance.SetProperty("LastSyncTime", (value))
}

// GetLastSyncTime gets the value of LastSyncTime for the instance
func (instance *Win32_OfflineFilesHealth) GetPropertyLastSyncTime() (value string, err error) {
	retValue, err := instance.GetProperty("LastSyncTime")
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

// SetOfflineAccessEnabled sets the value of OfflineAccessEnabled for the instance
func (instance *Win32_OfflineFilesHealth) SetPropertyOfflineAccessEnabled(value bool) (err error) {
	return instance.SetProperty("OfflineAccessEnabled", (value))
}

// GetOfflineAccessEnabled gets the value of OfflineAccessEnabled for the instance
func (instance *Win32_OfflineFilesHealth) GetPropertyOfflineAccessEnabled() (value bool, err error) {
	retValue, err := instance.GetProperty("OfflineAccessEnabled")
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

// SetOnlineMode sets the value of OnlineMode for the instance
func (instance *Win32_OfflineFilesHealth) SetPropertyOnlineMode(value bool) (err error) {
	return instance.SetProperty("OnlineMode", (value))
}

// GetOnlineMode gets the value of OnlineMode for the instance
func (instance *Win32_OfflineFilesHealth) GetPropertyOnlineMode() (value bool, err error) {
	retValue, err := instance.GetProperty("OnlineMode")
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
