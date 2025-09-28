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

// Win32_OfflineFilesConnectionInfo struct
type Win32_OfflineFilesConnectionInfo struct {
	*cim.WmiInstance

	//
	ConnectState uint32

	//
	OfflineReason uint32
}

func NewWin32_OfflineFilesConnectionInfoEx1(instance *cim.WmiInstance) (newInstance *Win32_OfflineFilesConnectionInfo, err error) {
	tmp, err := instance, nil

	if err != nil {
		return
	}
	newInstance = &Win32_OfflineFilesConnectionInfo{
		WmiInstance: tmp,
	}
	return
}

func NewWin32_OfflineFilesConnectionInfoEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_OfflineFilesConnectionInfo, err error) {
	tmp, err := instance.GetWmiInstance(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_OfflineFilesConnectionInfo{
		WmiInstance: tmp,
	}
	return
}

// SetConnectState sets the value of ConnectState for the instance
func (instance *Win32_OfflineFilesConnectionInfo) SetPropertyConnectState(value uint32) (err error) {
	return instance.SetProperty("ConnectState", (value))
}

// GetConnectState gets the value of ConnectState for the instance
func (instance *Win32_OfflineFilesConnectionInfo) GetPropertyConnectState() (value uint32, err error) {
	retValue, err := instance.GetProperty("ConnectState")
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

// SetOfflineReason sets the value of OfflineReason for the instance
func (instance *Win32_OfflineFilesConnectionInfo) SetPropertyOfflineReason(value uint32) (err error) {
	return instance.SetProperty("OfflineReason", (value))
}

// GetOfflineReason gets the value of OfflineReason for the instance
func (instance *Win32_OfflineFilesConnectionInfo) GetPropertyOfflineReason() (value uint32, err error) {
	retValue, err := instance.GetProperty("OfflineReason")
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
