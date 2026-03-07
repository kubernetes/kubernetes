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

// Win32_OfflineFilesPinInfo struct
type Win32_OfflineFilesPinInfo struct {
	*cim.WmiInstance

	//
	Pinned bool

	//
	PinnedForComputer uint32

	//
	PinnedForFolderRedirection uint32

	//
	PinnedForUser uint32

	//
	PinnedForUserByPolicy uint32
}

func NewWin32_OfflineFilesPinInfoEx1(instance *cim.WmiInstance) (newInstance *Win32_OfflineFilesPinInfo, err error) {
	tmp, err := instance, nil

	if err != nil {
		return
	}
	newInstance = &Win32_OfflineFilesPinInfo{
		WmiInstance: tmp,
	}
	return
}

func NewWin32_OfflineFilesPinInfoEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_OfflineFilesPinInfo, err error) {
	tmp, err := instance.GetWmiInstance(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_OfflineFilesPinInfo{
		WmiInstance: tmp,
	}
	return
}

// SetPinned sets the value of Pinned for the instance
func (instance *Win32_OfflineFilesPinInfo) SetPropertyPinned(value bool) (err error) {
	return instance.SetProperty("Pinned", (value))
}

// GetPinned gets the value of Pinned for the instance
func (instance *Win32_OfflineFilesPinInfo) GetPropertyPinned() (value bool, err error) {
	retValue, err := instance.GetProperty("Pinned")
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

// SetPinnedForComputer sets the value of PinnedForComputer for the instance
func (instance *Win32_OfflineFilesPinInfo) SetPropertyPinnedForComputer(value uint32) (err error) {
	return instance.SetProperty("PinnedForComputer", (value))
}

// GetPinnedForComputer gets the value of PinnedForComputer for the instance
func (instance *Win32_OfflineFilesPinInfo) GetPropertyPinnedForComputer() (value uint32, err error) {
	retValue, err := instance.GetProperty("PinnedForComputer")
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

// SetPinnedForFolderRedirection sets the value of PinnedForFolderRedirection for the instance
func (instance *Win32_OfflineFilesPinInfo) SetPropertyPinnedForFolderRedirection(value uint32) (err error) {
	return instance.SetProperty("PinnedForFolderRedirection", (value))
}

// GetPinnedForFolderRedirection gets the value of PinnedForFolderRedirection for the instance
func (instance *Win32_OfflineFilesPinInfo) GetPropertyPinnedForFolderRedirection() (value uint32, err error) {
	retValue, err := instance.GetProperty("PinnedForFolderRedirection")
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

// SetPinnedForUser sets the value of PinnedForUser for the instance
func (instance *Win32_OfflineFilesPinInfo) SetPropertyPinnedForUser(value uint32) (err error) {
	return instance.SetProperty("PinnedForUser", (value))
}

// GetPinnedForUser gets the value of PinnedForUser for the instance
func (instance *Win32_OfflineFilesPinInfo) GetPropertyPinnedForUser() (value uint32, err error) {
	retValue, err := instance.GetProperty("PinnedForUser")
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

// SetPinnedForUserByPolicy sets the value of PinnedForUserByPolicy for the instance
func (instance *Win32_OfflineFilesPinInfo) SetPropertyPinnedForUserByPolicy(value uint32) (err error) {
	return instance.SetProperty("PinnedForUserByPolicy", (value))
}

// GetPinnedForUserByPolicy gets the value of PinnedForUserByPolicy for the instance
func (instance *Win32_OfflineFilesPinInfo) GetPropertyPinnedForUserByPolicy() (value uint32, err error) {
	retValue, err := instance.GetProperty("PinnedForUserByPolicy")
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
