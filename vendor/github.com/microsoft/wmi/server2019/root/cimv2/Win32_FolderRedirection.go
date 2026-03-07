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

// Win32_FolderRedirection struct
type Win32_FolderRedirection struct {
	*cim.WmiInstance

	// Move the contents of local <folder> to the new location. This will copy the redirected folder data into the local UNC location. Then this content will be synced with the server share content. Effectively, moving the content from the local location to the share
	ContentsMoved bool

	// When the redirection policy is removed, the folder's content will be moved to the local profileIf true, the folder will be moved back to the local user profile location when policy is removed.If false, the folder will remain in the redirected location after the redirection policy is removed.
	ContentsMovedOnPolicyRemoval bool

	// Content is renamed from old to new location in Offline Files cache; assumes data on server is moved between names through other means
	ContentsRenamedInLocalCache bool

	// Grant the user exclusive rights to <folder>
	ExclusiveRightsGranted bool

	// known folder unique id (guid)
	FolderId string

	// Do not automatically make redirected folders available offline
	MakeFolderAvailableOfflineDisabled bool

	// Redirection Path [may be used when RedirectionType == {0,1}
	RedirectionPath string

	// The type of folder redirection to be performed.
	RedirectionType FolderRedirection_RedirectionType
}

func NewWin32_FolderRedirectionEx1(instance *cim.WmiInstance) (newInstance *Win32_FolderRedirection, err error) {
	tmp, err := instance, nil

	if err != nil {
		return
	}
	newInstance = &Win32_FolderRedirection{
		WmiInstance: tmp,
	}
	return
}

func NewWin32_FolderRedirectionEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_FolderRedirection, err error) {
	tmp, err := instance.GetWmiInstance(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_FolderRedirection{
		WmiInstance: tmp,
	}
	return
}

// SetContentsMoved sets the value of ContentsMoved for the instance
func (instance *Win32_FolderRedirection) SetPropertyContentsMoved(value bool) (err error) {
	return instance.SetProperty("ContentsMoved", (value))
}

// GetContentsMoved gets the value of ContentsMoved for the instance
func (instance *Win32_FolderRedirection) GetPropertyContentsMoved() (value bool, err error) {
	retValue, err := instance.GetProperty("ContentsMoved")
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

// SetContentsMovedOnPolicyRemoval sets the value of ContentsMovedOnPolicyRemoval for the instance
func (instance *Win32_FolderRedirection) SetPropertyContentsMovedOnPolicyRemoval(value bool) (err error) {
	return instance.SetProperty("ContentsMovedOnPolicyRemoval", (value))
}

// GetContentsMovedOnPolicyRemoval gets the value of ContentsMovedOnPolicyRemoval for the instance
func (instance *Win32_FolderRedirection) GetPropertyContentsMovedOnPolicyRemoval() (value bool, err error) {
	retValue, err := instance.GetProperty("ContentsMovedOnPolicyRemoval")
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

// SetContentsRenamedInLocalCache sets the value of ContentsRenamedInLocalCache for the instance
func (instance *Win32_FolderRedirection) SetPropertyContentsRenamedInLocalCache(value bool) (err error) {
	return instance.SetProperty("ContentsRenamedInLocalCache", (value))
}

// GetContentsRenamedInLocalCache gets the value of ContentsRenamedInLocalCache for the instance
func (instance *Win32_FolderRedirection) GetPropertyContentsRenamedInLocalCache() (value bool, err error) {
	retValue, err := instance.GetProperty("ContentsRenamedInLocalCache")
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

// SetExclusiveRightsGranted sets the value of ExclusiveRightsGranted for the instance
func (instance *Win32_FolderRedirection) SetPropertyExclusiveRightsGranted(value bool) (err error) {
	return instance.SetProperty("ExclusiveRightsGranted", (value))
}

// GetExclusiveRightsGranted gets the value of ExclusiveRightsGranted for the instance
func (instance *Win32_FolderRedirection) GetPropertyExclusiveRightsGranted() (value bool, err error) {
	retValue, err := instance.GetProperty("ExclusiveRightsGranted")
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

// SetFolderId sets the value of FolderId for the instance
func (instance *Win32_FolderRedirection) SetPropertyFolderId(value string) (err error) {
	return instance.SetProperty("FolderId", (value))
}

// GetFolderId gets the value of FolderId for the instance
func (instance *Win32_FolderRedirection) GetPropertyFolderId() (value string, err error) {
	retValue, err := instance.GetProperty("FolderId")
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

// SetMakeFolderAvailableOfflineDisabled sets the value of MakeFolderAvailableOfflineDisabled for the instance
func (instance *Win32_FolderRedirection) SetPropertyMakeFolderAvailableOfflineDisabled(value bool) (err error) {
	return instance.SetProperty("MakeFolderAvailableOfflineDisabled", (value))
}

// GetMakeFolderAvailableOfflineDisabled gets the value of MakeFolderAvailableOfflineDisabled for the instance
func (instance *Win32_FolderRedirection) GetPropertyMakeFolderAvailableOfflineDisabled() (value bool, err error) {
	retValue, err := instance.GetProperty("MakeFolderAvailableOfflineDisabled")
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

// SetRedirectionPath sets the value of RedirectionPath for the instance
func (instance *Win32_FolderRedirection) SetPropertyRedirectionPath(value string) (err error) {
	return instance.SetProperty("RedirectionPath", (value))
}

// GetRedirectionPath gets the value of RedirectionPath for the instance
func (instance *Win32_FolderRedirection) GetPropertyRedirectionPath() (value string, err error) {
	retValue, err := instance.GetProperty("RedirectionPath")
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

// SetRedirectionType sets the value of RedirectionType for the instance
func (instance *Win32_FolderRedirection) SetPropertyRedirectionType(value FolderRedirection_RedirectionType) (err error) {
	return instance.SetProperty("RedirectionType", (value))
}

// GetRedirectionType gets the value of RedirectionType for the instance
func (instance *Win32_FolderRedirection) GetPropertyRedirectionType() (value FolderRedirection_RedirectionType, err error) {
	retValue, err := instance.GetProperty("RedirectionType")
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

	value = FolderRedirection_RedirectionType(valuetmp)

	return
}
