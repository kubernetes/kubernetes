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

// Win32_FolderRedirectionHealth struct
type Win32_FolderRedirectionHealth struct {
	*cim.WmiInstance

	// The health status of this folder, based on the values that were set in the Win32_FolderRedirectionHealthConfiguration properties.
	HealthStatus FolderRedirectionHealth_HealthStatus

	// The last time this folder was successfully synchronized to the Offline Files cache.
	LastSuccessfulSyncTime string

	// The status of the last attempt to synchronize this folder to the Offline Files cache.
	LastSyncStatus FolderRedirectionHealth_LastSyncStatus

	// The last time an attempt was made to synchronized this folder to the Offline Files cache, even if it was unsuccessful.
	LastSyncTime string

	// If true, the Offline Files feature is enabled for this folder.
	OfflineAccessEnabled bool

	// known folder unique id (guid)
	OfflineFileNameFolderGUID string

	// If true, indicate if this folder is being redirected.
	Redirected bool
}

func NewWin32_FolderRedirectionHealthEx1(instance *cim.WmiInstance) (newInstance *Win32_FolderRedirectionHealth, err error) {
	tmp, err := instance, nil

	if err != nil {
		return
	}
	newInstance = &Win32_FolderRedirectionHealth{
		WmiInstance: tmp,
	}
	return
}

func NewWin32_FolderRedirectionHealthEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_FolderRedirectionHealth, err error) {
	tmp, err := instance.GetWmiInstance(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_FolderRedirectionHealth{
		WmiInstance: tmp,
	}
	return
}

// SetHealthStatus sets the value of HealthStatus for the instance
func (instance *Win32_FolderRedirectionHealth) SetPropertyHealthStatus(value FolderRedirectionHealth_HealthStatus) (err error) {
	return instance.SetProperty("HealthStatus", (value))
}

// GetHealthStatus gets the value of HealthStatus for the instance
func (instance *Win32_FolderRedirectionHealth) GetPropertyHealthStatus() (value FolderRedirectionHealth_HealthStatus, err error) {
	retValue, err := instance.GetProperty("HealthStatus")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(int32)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " int32 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = FolderRedirectionHealth_HealthStatus(valuetmp)

	return
}

// SetLastSuccessfulSyncTime sets the value of LastSuccessfulSyncTime for the instance
func (instance *Win32_FolderRedirectionHealth) SetPropertyLastSuccessfulSyncTime(value string) (err error) {
	return instance.SetProperty("LastSuccessfulSyncTime", (value))
}

// GetLastSuccessfulSyncTime gets the value of LastSuccessfulSyncTime for the instance
func (instance *Win32_FolderRedirectionHealth) GetPropertyLastSuccessfulSyncTime() (value string, err error) {
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
func (instance *Win32_FolderRedirectionHealth) SetPropertyLastSyncStatus(value FolderRedirectionHealth_LastSyncStatus) (err error) {
	return instance.SetProperty("LastSyncStatus", (value))
}

// GetLastSyncStatus gets the value of LastSyncStatus for the instance
func (instance *Win32_FolderRedirectionHealth) GetPropertyLastSyncStatus() (value FolderRedirectionHealth_LastSyncStatus, err error) {
	retValue, err := instance.GetProperty("LastSyncStatus")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(int32)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " int32 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = FolderRedirectionHealth_LastSyncStatus(valuetmp)

	return
}

// SetLastSyncTime sets the value of LastSyncTime for the instance
func (instance *Win32_FolderRedirectionHealth) SetPropertyLastSyncTime(value string) (err error) {
	return instance.SetProperty("LastSyncTime", (value))
}

// GetLastSyncTime gets the value of LastSyncTime for the instance
func (instance *Win32_FolderRedirectionHealth) GetPropertyLastSyncTime() (value string, err error) {
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
func (instance *Win32_FolderRedirectionHealth) SetPropertyOfflineAccessEnabled(value bool) (err error) {
	return instance.SetProperty("OfflineAccessEnabled", (value))
}

// GetOfflineAccessEnabled gets the value of OfflineAccessEnabled for the instance
func (instance *Win32_FolderRedirectionHealth) GetPropertyOfflineAccessEnabled() (value bool, err error) {
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

// SetOfflineFileNameFolderGUID sets the value of OfflineFileNameFolderGUID for the instance
func (instance *Win32_FolderRedirectionHealth) SetPropertyOfflineFileNameFolderGUID(value string) (err error) {
	return instance.SetProperty("OfflineFileNameFolderGUID", (value))
}

// GetOfflineFileNameFolderGUID gets the value of OfflineFileNameFolderGUID for the instance
func (instance *Win32_FolderRedirectionHealth) GetPropertyOfflineFileNameFolderGUID() (value string, err error) {
	retValue, err := instance.GetProperty("OfflineFileNameFolderGUID")
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

// SetRedirected sets the value of Redirected for the instance
func (instance *Win32_FolderRedirectionHealth) SetPropertyRedirected(value bool) (err error) {
	return instance.SetProperty("Redirected", (value))
}

// GetRedirected gets the value of Redirected for the instance
func (instance *Win32_FolderRedirectionHealth) GetPropertyRedirected() (value bool, err error) {
	retValue, err := instance.GetProperty("Redirected")
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
