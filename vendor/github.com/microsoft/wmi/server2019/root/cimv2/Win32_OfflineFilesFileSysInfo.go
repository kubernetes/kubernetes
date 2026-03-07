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

// Win32_OfflineFilesFileSysInfo struct
type Win32_OfflineFilesFileSysInfo struct {
	*cim.WmiInstance

	//
	LocalAttributes uint32

	//
	LocalChangeTime string

	//
	LocalCreationTime string

	//
	LocalLastAccessTime string

	//
	LocalLastWriteTime string

	//
	LocalSize int64

	//
	OriginalAttributes uint32

	//
	OriginalChangeTime string

	//
	OriginalCreationTime string

	//
	OriginalLastAccessTime string

	//
	OriginalLastWriteTime string

	//
	OriginalSize int64

	//
	RemoteAttributes uint32

	//
	RemoteChangeTime string

	//
	RemoteCreationTime string

	//
	RemoteLastAccessTime string

	//
	RemoteLastWriteTime string

	//
	RemoteSize int64
}

func NewWin32_OfflineFilesFileSysInfoEx1(instance *cim.WmiInstance) (newInstance *Win32_OfflineFilesFileSysInfo, err error) {
	tmp, err := instance, nil

	if err != nil {
		return
	}
	newInstance = &Win32_OfflineFilesFileSysInfo{
		WmiInstance: tmp,
	}
	return
}

func NewWin32_OfflineFilesFileSysInfoEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_OfflineFilesFileSysInfo, err error) {
	tmp, err := instance.GetWmiInstance(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_OfflineFilesFileSysInfo{
		WmiInstance: tmp,
	}
	return
}

// SetLocalAttributes sets the value of LocalAttributes for the instance
func (instance *Win32_OfflineFilesFileSysInfo) SetPropertyLocalAttributes(value uint32) (err error) {
	return instance.SetProperty("LocalAttributes", (value))
}

// GetLocalAttributes gets the value of LocalAttributes for the instance
func (instance *Win32_OfflineFilesFileSysInfo) GetPropertyLocalAttributes() (value uint32, err error) {
	retValue, err := instance.GetProperty("LocalAttributes")
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

// SetLocalChangeTime sets the value of LocalChangeTime for the instance
func (instance *Win32_OfflineFilesFileSysInfo) SetPropertyLocalChangeTime(value string) (err error) {
	return instance.SetProperty("LocalChangeTime", (value))
}

// GetLocalChangeTime gets the value of LocalChangeTime for the instance
func (instance *Win32_OfflineFilesFileSysInfo) GetPropertyLocalChangeTime() (value string, err error) {
	retValue, err := instance.GetProperty("LocalChangeTime")
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

// SetLocalCreationTime sets the value of LocalCreationTime for the instance
func (instance *Win32_OfflineFilesFileSysInfo) SetPropertyLocalCreationTime(value string) (err error) {
	return instance.SetProperty("LocalCreationTime", (value))
}

// GetLocalCreationTime gets the value of LocalCreationTime for the instance
func (instance *Win32_OfflineFilesFileSysInfo) GetPropertyLocalCreationTime() (value string, err error) {
	retValue, err := instance.GetProperty("LocalCreationTime")
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

// SetLocalLastAccessTime sets the value of LocalLastAccessTime for the instance
func (instance *Win32_OfflineFilesFileSysInfo) SetPropertyLocalLastAccessTime(value string) (err error) {
	return instance.SetProperty("LocalLastAccessTime", (value))
}

// GetLocalLastAccessTime gets the value of LocalLastAccessTime for the instance
func (instance *Win32_OfflineFilesFileSysInfo) GetPropertyLocalLastAccessTime() (value string, err error) {
	retValue, err := instance.GetProperty("LocalLastAccessTime")
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

// SetLocalLastWriteTime sets the value of LocalLastWriteTime for the instance
func (instance *Win32_OfflineFilesFileSysInfo) SetPropertyLocalLastWriteTime(value string) (err error) {
	return instance.SetProperty("LocalLastWriteTime", (value))
}

// GetLocalLastWriteTime gets the value of LocalLastWriteTime for the instance
func (instance *Win32_OfflineFilesFileSysInfo) GetPropertyLocalLastWriteTime() (value string, err error) {
	retValue, err := instance.GetProperty("LocalLastWriteTime")
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

// SetLocalSize sets the value of LocalSize for the instance
func (instance *Win32_OfflineFilesFileSysInfo) SetPropertyLocalSize(value int64) (err error) {
	return instance.SetProperty("LocalSize", (value))
}

// GetLocalSize gets the value of LocalSize for the instance
func (instance *Win32_OfflineFilesFileSysInfo) GetPropertyLocalSize() (value int64, err error) {
	retValue, err := instance.GetProperty("LocalSize")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(int64)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " int64 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = int64(valuetmp)

	return
}

// SetOriginalAttributes sets the value of OriginalAttributes for the instance
func (instance *Win32_OfflineFilesFileSysInfo) SetPropertyOriginalAttributes(value uint32) (err error) {
	return instance.SetProperty("OriginalAttributes", (value))
}

// GetOriginalAttributes gets the value of OriginalAttributes for the instance
func (instance *Win32_OfflineFilesFileSysInfo) GetPropertyOriginalAttributes() (value uint32, err error) {
	retValue, err := instance.GetProperty("OriginalAttributes")
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

// SetOriginalChangeTime sets the value of OriginalChangeTime for the instance
func (instance *Win32_OfflineFilesFileSysInfo) SetPropertyOriginalChangeTime(value string) (err error) {
	return instance.SetProperty("OriginalChangeTime", (value))
}

// GetOriginalChangeTime gets the value of OriginalChangeTime for the instance
func (instance *Win32_OfflineFilesFileSysInfo) GetPropertyOriginalChangeTime() (value string, err error) {
	retValue, err := instance.GetProperty("OriginalChangeTime")
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

// SetOriginalCreationTime sets the value of OriginalCreationTime for the instance
func (instance *Win32_OfflineFilesFileSysInfo) SetPropertyOriginalCreationTime(value string) (err error) {
	return instance.SetProperty("OriginalCreationTime", (value))
}

// GetOriginalCreationTime gets the value of OriginalCreationTime for the instance
func (instance *Win32_OfflineFilesFileSysInfo) GetPropertyOriginalCreationTime() (value string, err error) {
	retValue, err := instance.GetProperty("OriginalCreationTime")
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

// SetOriginalLastAccessTime sets the value of OriginalLastAccessTime for the instance
func (instance *Win32_OfflineFilesFileSysInfo) SetPropertyOriginalLastAccessTime(value string) (err error) {
	return instance.SetProperty("OriginalLastAccessTime", (value))
}

// GetOriginalLastAccessTime gets the value of OriginalLastAccessTime for the instance
func (instance *Win32_OfflineFilesFileSysInfo) GetPropertyOriginalLastAccessTime() (value string, err error) {
	retValue, err := instance.GetProperty("OriginalLastAccessTime")
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

// SetOriginalLastWriteTime sets the value of OriginalLastWriteTime for the instance
func (instance *Win32_OfflineFilesFileSysInfo) SetPropertyOriginalLastWriteTime(value string) (err error) {
	return instance.SetProperty("OriginalLastWriteTime", (value))
}

// GetOriginalLastWriteTime gets the value of OriginalLastWriteTime for the instance
func (instance *Win32_OfflineFilesFileSysInfo) GetPropertyOriginalLastWriteTime() (value string, err error) {
	retValue, err := instance.GetProperty("OriginalLastWriteTime")
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

// SetOriginalSize sets the value of OriginalSize for the instance
func (instance *Win32_OfflineFilesFileSysInfo) SetPropertyOriginalSize(value int64) (err error) {
	return instance.SetProperty("OriginalSize", (value))
}

// GetOriginalSize gets the value of OriginalSize for the instance
func (instance *Win32_OfflineFilesFileSysInfo) GetPropertyOriginalSize() (value int64, err error) {
	retValue, err := instance.GetProperty("OriginalSize")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(int64)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " int64 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = int64(valuetmp)

	return
}

// SetRemoteAttributes sets the value of RemoteAttributes for the instance
func (instance *Win32_OfflineFilesFileSysInfo) SetPropertyRemoteAttributes(value uint32) (err error) {
	return instance.SetProperty("RemoteAttributes", (value))
}

// GetRemoteAttributes gets the value of RemoteAttributes for the instance
func (instance *Win32_OfflineFilesFileSysInfo) GetPropertyRemoteAttributes() (value uint32, err error) {
	retValue, err := instance.GetProperty("RemoteAttributes")
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

// SetRemoteChangeTime sets the value of RemoteChangeTime for the instance
func (instance *Win32_OfflineFilesFileSysInfo) SetPropertyRemoteChangeTime(value string) (err error) {
	return instance.SetProperty("RemoteChangeTime", (value))
}

// GetRemoteChangeTime gets the value of RemoteChangeTime for the instance
func (instance *Win32_OfflineFilesFileSysInfo) GetPropertyRemoteChangeTime() (value string, err error) {
	retValue, err := instance.GetProperty("RemoteChangeTime")
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

// SetRemoteCreationTime sets the value of RemoteCreationTime for the instance
func (instance *Win32_OfflineFilesFileSysInfo) SetPropertyRemoteCreationTime(value string) (err error) {
	return instance.SetProperty("RemoteCreationTime", (value))
}

// GetRemoteCreationTime gets the value of RemoteCreationTime for the instance
func (instance *Win32_OfflineFilesFileSysInfo) GetPropertyRemoteCreationTime() (value string, err error) {
	retValue, err := instance.GetProperty("RemoteCreationTime")
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

// SetRemoteLastAccessTime sets the value of RemoteLastAccessTime for the instance
func (instance *Win32_OfflineFilesFileSysInfo) SetPropertyRemoteLastAccessTime(value string) (err error) {
	return instance.SetProperty("RemoteLastAccessTime", (value))
}

// GetRemoteLastAccessTime gets the value of RemoteLastAccessTime for the instance
func (instance *Win32_OfflineFilesFileSysInfo) GetPropertyRemoteLastAccessTime() (value string, err error) {
	retValue, err := instance.GetProperty("RemoteLastAccessTime")
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

// SetRemoteLastWriteTime sets the value of RemoteLastWriteTime for the instance
func (instance *Win32_OfflineFilesFileSysInfo) SetPropertyRemoteLastWriteTime(value string) (err error) {
	return instance.SetProperty("RemoteLastWriteTime", (value))
}

// GetRemoteLastWriteTime gets the value of RemoteLastWriteTime for the instance
func (instance *Win32_OfflineFilesFileSysInfo) GetPropertyRemoteLastWriteTime() (value string, err error) {
	retValue, err := instance.GetProperty("RemoteLastWriteTime")
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

// SetRemoteSize sets the value of RemoteSize for the instance
func (instance *Win32_OfflineFilesFileSysInfo) SetPropertyRemoteSize(value int64) (err error) {
	return instance.SetProperty("RemoteSize", (value))
}

// GetRemoteSize gets the value of RemoteSize for the instance
func (instance *Win32_OfflineFilesFileSysInfo) GetPropertyRemoteSize() (value int64, err error) {
	retValue, err := instance.GetProperty("RemoteSize")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(int64)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " int64 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = int64(valuetmp)

	return
}
