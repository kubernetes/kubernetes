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

// Win32_OfflineFilesDirtyInfo struct
type Win32_OfflineFilesDirtyInfo struct {
	*cim.WmiInstance

	//
	LocalDirtyByteCount int64

	//
	RemoteDirtyByteCount int64
}

func NewWin32_OfflineFilesDirtyInfoEx1(instance *cim.WmiInstance) (newInstance *Win32_OfflineFilesDirtyInfo, err error) {
	tmp, err := instance, nil

	if err != nil {
		return
	}
	newInstance = &Win32_OfflineFilesDirtyInfo{
		WmiInstance: tmp,
	}
	return
}

func NewWin32_OfflineFilesDirtyInfoEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_OfflineFilesDirtyInfo, err error) {
	tmp, err := instance.GetWmiInstance(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_OfflineFilesDirtyInfo{
		WmiInstance: tmp,
	}
	return
}

// SetLocalDirtyByteCount sets the value of LocalDirtyByteCount for the instance
func (instance *Win32_OfflineFilesDirtyInfo) SetPropertyLocalDirtyByteCount(value int64) (err error) {
	return instance.SetProperty("LocalDirtyByteCount", (value))
}

// GetLocalDirtyByteCount gets the value of LocalDirtyByteCount for the instance
func (instance *Win32_OfflineFilesDirtyInfo) GetPropertyLocalDirtyByteCount() (value int64, err error) {
	retValue, err := instance.GetProperty("LocalDirtyByteCount")
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

// SetRemoteDirtyByteCount sets the value of RemoteDirtyByteCount for the instance
func (instance *Win32_OfflineFilesDirtyInfo) SetPropertyRemoteDirtyByteCount(value int64) (err error) {
	return instance.SetProperty("RemoteDirtyByteCount", (value))
}

// GetRemoteDirtyByteCount gets the value of RemoteDirtyByteCount for the instance
func (instance *Win32_OfflineFilesDirtyInfo) GetPropertyRemoteDirtyByteCount() (value int64, err error) {
	retValue, err := instance.GetProperty("RemoteDirtyByteCount")
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
