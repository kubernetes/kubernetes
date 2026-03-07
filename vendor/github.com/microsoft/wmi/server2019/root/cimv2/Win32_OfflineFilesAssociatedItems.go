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

// Win32_OfflineFilesAssociatedItems struct
type Win32_OfflineFilesAssociatedItems struct {
	*cim.WmiInstance

	//
	Antecedent Win32_OfflineFilesCache

	//
	Dependent Win32_OfflineFilesItem
}

func NewWin32_OfflineFilesAssociatedItemsEx1(instance *cim.WmiInstance) (newInstance *Win32_OfflineFilesAssociatedItems, err error) {
	tmp, err := instance, nil

	if err != nil {
		return
	}
	newInstance = &Win32_OfflineFilesAssociatedItems{
		WmiInstance: tmp,
	}
	return
}

func NewWin32_OfflineFilesAssociatedItemsEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_OfflineFilesAssociatedItems, err error) {
	tmp, err := instance.GetWmiInstance(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_OfflineFilesAssociatedItems{
		WmiInstance: tmp,
	}
	return
}

// SetAntecedent sets the value of Antecedent for the instance
func (instance *Win32_OfflineFilesAssociatedItems) SetPropertyAntecedent(value Win32_OfflineFilesCache) (err error) {
	return instance.SetProperty("Antecedent", (value))
}

// GetAntecedent gets the value of Antecedent for the instance
func (instance *Win32_OfflineFilesAssociatedItems) GetPropertyAntecedent() (value Win32_OfflineFilesCache, err error) {
	retValue, err := instance.GetProperty("Antecedent")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(Win32_OfflineFilesCache)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " Win32_OfflineFilesCache is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = Win32_OfflineFilesCache(valuetmp)

	return
}

// SetDependent sets the value of Dependent for the instance
func (instance *Win32_OfflineFilesAssociatedItems) SetPropertyDependent(value Win32_OfflineFilesItem) (err error) {
	return instance.SetProperty("Dependent", (value))
}

// GetDependent gets the value of Dependent for the instance
func (instance *Win32_OfflineFilesAssociatedItems) GetPropertyDependent() (value Win32_OfflineFilesItem, err error) {
	retValue, err := instance.GetProperty("Dependent")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(Win32_OfflineFilesItem)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " Win32_OfflineFilesItem is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = Win32_OfflineFilesItem(valuetmp)

	return
}
