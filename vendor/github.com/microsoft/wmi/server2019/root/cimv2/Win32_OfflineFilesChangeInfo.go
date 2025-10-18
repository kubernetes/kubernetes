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

// Win32_OfflineFilesChangeInfo struct
type Win32_OfflineFilesChangeInfo struct {
	*cim.WmiInstance

	//
	CreatedOffline bool

	//
	DeletedOffline bool

	//
	Dirty bool

	//
	ModifiedAttributes bool

	//
	ModifiedData bool

	//
	ModifiedTime bool
}

func NewWin32_OfflineFilesChangeInfoEx1(instance *cim.WmiInstance) (newInstance *Win32_OfflineFilesChangeInfo, err error) {
	tmp, err := instance, nil

	if err != nil {
		return
	}
	newInstance = &Win32_OfflineFilesChangeInfo{
		WmiInstance: tmp,
	}
	return
}

func NewWin32_OfflineFilesChangeInfoEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_OfflineFilesChangeInfo, err error) {
	tmp, err := instance.GetWmiInstance(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_OfflineFilesChangeInfo{
		WmiInstance: tmp,
	}
	return
}

// SetCreatedOffline sets the value of CreatedOffline for the instance
func (instance *Win32_OfflineFilesChangeInfo) SetPropertyCreatedOffline(value bool) (err error) {
	return instance.SetProperty("CreatedOffline", (value))
}

// GetCreatedOffline gets the value of CreatedOffline for the instance
func (instance *Win32_OfflineFilesChangeInfo) GetPropertyCreatedOffline() (value bool, err error) {
	retValue, err := instance.GetProperty("CreatedOffline")
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

// SetDeletedOffline sets the value of DeletedOffline for the instance
func (instance *Win32_OfflineFilesChangeInfo) SetPropertyDeletedOffline(value bool) (err error) {
	return instance.SetProperty("DeletedOffline", (value))
}

// GetDeletedOffline gets the value of DeletedOffline for the instance
func (instance *Win32_OfflineFilesChangeInfo) GetPropertyDeletedOffline() (value bool, err error) {
	retValue, err := instance.GetProperty("DeletedOffline")
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

// SetDirty sets the value of Dirty for the instance
func (instance *Win32_OfflineFilesChangeInfo) SetPropertyDirty(value bool) (err error) {
	return instance.SetProperty("Dirty", (value))
}

// GetDirty gets the value of Dirty for the instance
func (instance *Win32_OfflineFilesChangeInfo) GetPropertyDirty() (value bool, err error) {
	retValue, err := instance.GetProperty("Dirty")
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

// SetModifiedAttributes sets the value of ModifiedAttributes for the instance
func (instance *Win32_OfflineFilesChangeInfo) SetPropertyModifiedAttributes(value bool) (err error) {
	return instance.SetProperty("ModifiedAttributes", (value))
}

// GetModifiedAttributes gets the value of ModifiedAttributes for the instance
func (instance *Win32_OfflineFilesChangeInfo) GetPropertyModifiedAttributes() (value bool, err error) {
	retValue, err := instance.GetProperty("ModifiedAttributes")
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

// SetModifiedData sets the value of ModifiedData for the instance
func (instance *Win32_OfflineFilesChangeInfo) SetPropertyModifiedData(value bool) (err error) {
	return instance.SetProperty("ModifiedData", (value))
}

// GetModifiedData gets the value of ModifiedData for the instance
func (instance *Win32_OfflineFilesChangeInfo) GetPropertyModifiedData() (value bool, err error) {
	retValue, err := instance.GetProperty("ModifiedData")
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

// SetModifiedTime sets the value of ModifiedTime for the instance
func (instance *Win32_OfflineFilesChangeInfo) SetPropertyModifiedTime(value bool) (err error) {
	return instance.SetProperty("ModifiedTime", (value))
}

// GetModifiedTime gets the value of ModifiedTime for the instance
func (instance *Win32_OfflineFilesChangeInfo) GetPropertyModifiedTime() (value bool, err error) {
	retValue, err := instance.GetProperty("ModifiedTime")
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
