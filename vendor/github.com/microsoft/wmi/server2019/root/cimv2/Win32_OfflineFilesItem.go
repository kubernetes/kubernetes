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

// Win32_OfflineFilesItem struct
type Win32_OfflineFilesItem struct {
	*cim.WmiInstance

	//
	ChangeInfo Win32_OfflineFilesChangeInfo

	//
	ConnectionInfo Win32_OfflineFilesConnectionInfo

	//
	DirtyInfo Win32_OfflineFilesDirtyInfo

	//
	Encrypted bool

	//
	FileSysInfo Win32_OfflineFilesFileSysInfo

	//
	ItemName string

	//
	ItemPath string

	//
	ItemType uint32

	//
	ParentItemPath string

	//
	PinInfo Win32_OfflineFilesPinInfo

	//
	Sparse bool

	//
	SuspendInfo Win32_OfflineFilesSuspendInfo
}

func NewWin32_OfflineFilesItemEx1(instance *cim.WmiInstance) (newInstance *Win32_OfflineFilesItem, err error) {
	tmp, err := instance, nil

	if err != nil {
		return
	}
	newInstance = &Win32_OfflineFilesItem{
		WmiInstance: tmp,
	}
	return
}

func NewWin32_OfflineFilesItemEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_OfflineFilesItem, err error) {
	tmp, err := instance.GetWmiInstance(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_OfflineFilesItem{
		WmiInstance: tmp,
	}
	return
}

// SetChangeInfo sets the value of ChangeInfo for the instance
func (instance *Win32_OfflineFilesItem) SetPropertyChangeInfo(value Win32_OfflineFilesChangeInfo) (err error) {
	return instance.SetProperty("ChangeInfo", (value))
}

// GetChangeInfo gets the value of ChangeInfo for the instance
func (instance *Win32_OfflineFilesItem) GetPropertyChangeInfo() (value Win32_OfflineFilesChangeInfo, err error) {
	retValue, err := instance.GetProperty("ChangeInfo")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(Win32_OfflineFilesChangeInfo)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " Win32_OfflineFilesChangeInfo is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = Win32_OfflineFilesChangeInfo(valuetmp)

	return
}

// SetConnectionInfo sets the value of ConnectionInfo for the instance
func (instance *Win32_OfflineFilesItem) SetPropertyConnectionInfo(value Win32_OfflineFilesConnectionInfo) (err error) {
	return instance.SetProperty("ConnectionInfo", (value))
}

// GetConnectionInfo gets the value of ConnectionInfo for the instance
func (instance *Win32_OfflineFilesItem) GetPropertyConnectionInfo() (value Win32_OfflineFilesConnectionInfo, err error) {
	retValue, err := instance.GetProperty("ConnectionInfo")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(Win32_OfflineFilesConnectionInfo)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " Win32_OfflineFilesConnectionInfo is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = Win32_OfflineFilesConnectionInfo(valuetmp)

	return
}

// SetDirtyInfo sets the value of DirtyInfo for the instance
func (instance *Win32_OfflineFilesItem) SetPropertyDirtyInfo(value Win32_OfflineFilesDirtyInfo) (err error) {
	return instance.SetProperty("DirtyInfo", (value))
}

// GetDirtyInfo gets the value of DirtyInfo for the instance
func (instance *Win32_OfflineFilesItem) GetPropertyDirtyInfo() (value Win32_OfflineFilesDirtyInfo, err error) {
	retValue, err := instance.GetProperty("DirtyInfo")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(Win32_OfflineFilesDirtyInfo)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " Win32_OfflineFilesDirtyInfo is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = Win32_OfflineFilesDirtyInfo(valuetmp)

	return
}

// SetEncrypted sets the value of Encrypted for the instance
func (instance *Win32_OfflineFilesItem) SetPropertyEncrypted(value bool) (err error) {
	return instance.SetProperty("Encrypted", (value))
}

// GetEncrypted gets the value of Encrypted for the instance
func (instance *Win32_OfflineFilesItem) GetPropertyEncrypted() (value bool, err error) {
	retValue, err := instance.GetProperty("Encrypted")
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

// SetFileSysInfo sets the value of FileSysInfo for the instance
func (instance *Win32_OfflineFilesItem) SetPropertyFileSysInfo(value Win32_OfflineFilesFileSysInfo) (err error) {
	return instance.SetProperty("FileSysInfo", (value))
}

// GetFileSysInfo gets the value of FileSysInfo for the instance
func (instance *Win32_OfflineFilesItem) GetPropertyFileSysInfo() (value Win32_OfflineFilesFileSysInfo, err error) {
	retValue, err := instance.GetProperty("FileSysInfo")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(Win32_OfflineFilesFileSysInfo)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " Win32_OfflineFilesFileSysInfo is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = Win32_OfflineFilesFileSysInfo(valuetmp)

	return
}

// SetItemName sets the value of ItemName for the instance
func (instance *Win32_OfflineFilesItem) SetPropertyItemName(value string) (err error) {
	return instance.SetProperty("ItemName", (value))
}

// GetItemName gets the value of ItemName for the instance
func (instance *Win32_OfflineFilesItem) GetPropertyItemName() (value string, err error) {
	retValue, err := instance.GetProperty("ItemName")
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

// SetItemPath sets the value of ItemPath for the instance
func (instance *Win32_OfflineFilesItem) SetPropertyItemPath(value string) (err error) {
	return instance.SetProperty("ItemPath", (value))
}

// GetItemPath gets the value of ItemPath for the instance
func (instance *Win32_OfflineFilesItem) GetPropertyItemPath() (value string, err error) {
	retValue, err := instance.GetProperty("ItemPath")
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

// SetItemType sets the value of ItemType for the instance
func (instance *Win32_OfflineFilesItem) SetPropertyItemType(value uint32) (err error) {
	return instance.SetProperty("ItemType", (value))
}

// GetItemType gets the value of ItemType for the instance
func (instance *Win32_OfflineFilesItem) GetPropertyItemType() (value uint32, err error) {
	retValue, err := instance.GetProperty("ItemType")
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

// SetParentItemPath sets the value of ParentItemPath for the instance
func (instance *Win32_OfflineFilesItem) SetPropertyParentItemPath(value string) (err error) {
	return instance.SetProperty("ParentItemPath", (value))
}

// GetParentItemPath gets the value of ParentItemPath for the instance
func (instance *Win32_OfflineFilesItem) GetPropertyParentItemPath() (value string, err error) {
	retValue, err := instance.GetProperty("ParentItemPath")
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

// SetPinInfo sets the value of PinInfo for the instance
func (instance *Win32_OfflineFilesItem) SetPropertyPinInfo(value Win32_OfflineFilesPinInfo) (err error) {
	return instance.SetProperty("PinInfo", (value))
}

// GetPinInfo gets the value of PinInfo for the instance
func (instance *Win32_OfflineFilesItem) GetPropertyPinInfo() (value Win32_OfflineFilesPinInfo, err error) {
	retValue, err := instance.GetProperty("PinInfo")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(Win32_OfflineFilesPinInfo)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " Win32_OfflineFilesPinInfo is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = Win32_OfflineFilesPinInfo(valuetmp)

	return
}

// SetSparse sets the value of Sparse for the instance
func (instance *Win32_OfflineFilesItem) SetPropertySparse(value bool) (err error) {
	return instance.SetProperty("Sparse", (value))
}

// GetSparse gets the value of Sparse for the instance
func (instance *Win32_OfflineFilesItem) GetPropertySparse() (value bool, err error) {
	retValue, err := instance.GetProperty("Sparse")
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

// SetSuspendInfo sets the value of SuspendInfo for the instance
func (instance *Win32_OfflineFilesItem) SetPropertySuspendInfo(value Win32_OfflineFilesSuspendInfo) (err error) {
	return instance.SetProperty("SuspendInfo", (value))
}

// GetSuspendInfo gets the value of SuspendInfo for the instance
func (instance *Win32_OfflineFilesItem) GetPropertySuspendInfo() (value Win32_OfflineFilesSuspendInfo, err error) {
	retValue, err := instance.GetProperty("SuspendInfo")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(Win32_OfflineFilesSuspendInfo)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " Win32_OfflineFilesSuspendInfo is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = Win32_OfflineFilesSuspendInfo(valuetmp)

	return
}
