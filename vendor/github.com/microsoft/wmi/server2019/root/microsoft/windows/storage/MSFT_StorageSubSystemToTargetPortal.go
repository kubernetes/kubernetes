// Copyright 2019 (c) Microsoft Corporation.
// Licensed under the MIT license.

//
// Author:
//      Auto Generated on 9/18/2020 using wmigen
//      Source root.Microsoft.Windows.Storage
//////////////////////////////////////////////
package storage

import (
	"github.com/microsoft/wmi/pkg/base/instance"
	"github.com/microsoft/wmi/pkg/base/query"
	"github.com/microsoft/wmi/pkg/errors"
	cim "github.com/microsoft/wmi/pkg/wmiinstance"
	"reflect"
)

// MSFT_StorageSubSystemToTargetPortal struct
type MSFT_StorageSubSystemToTargetPortal struct {
	*cim.WmiInstance

	//
	StorageSubSystem MSFT_StorageSubSystem

	//
	TargetPortal MSFT_TargetPortal
}

func NewMSFT_StorageSubSystemToTargetPortalEx1(instance *cim.WmiInstance) (newInstance *MSFT_StorageSubSystemToTargetPortal, err error) {
	tmp, err := instance, nil

	if err != nil {
		return
	}
	newInstance = &MSFT_StorageSubSystemToTargetPortal{
		WmiInstance: tmp,
	}
	return
}

func NewMSFT_StorageSubSystemToTargetPortalEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *MSFT_StorageSubSystemToTargetPortal, err error) {
	tmp, err := instance.GetWmiInstance(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &MSFT_StorageSubSystemToTargetPortal{
		WmiInstance: tmp,
	}
	return
}

// SetStorageSubSystem sets the value of StorageSubSystem for the instance
func (instance *MSFT_StorageSubSystemToTargetPortal) SetPropertyStorageSubSystem(value MSFT_StorageSubSystem) (err error) {
	return instance.SetProperty("StorageSubSystem", (value))
}

// GetStorageSubSystem gets the value of StorageSubSystem for the instance
func (instance *MSFT_StorageSubSystemToTargetPortal) GetPropertyStorageSubSystem() (value MSFT_StorageSubSystem, err error) {
	retValue, err := instance.GetProperty("StorageSubSystem")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(MSFT_StorageSubSystem)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " MSFT_StorageSubSystem is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = MSFT_StorageSubSystem(valuetmp)

	return
}

// SetTargetPortal sets the value of TargetPortal for the instance
func (instance *MSFT_StorageSubSystemToTargetPortal) SetPropertyTargetPortal(value MSFT_TargetPortal) (err error) {
	return instance.SetProperty("TargetPortal", (value))
}

// GetTargetPortal gets the value of TargetPortal for the instance
func (instance *MSFT_StorageSubSystemToTargetPortal) GetPropertyTargetPortal() (value MSFT_TargetPortal, err error) {
	retValue, err := instance.GetProperty("TargetPortal")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(MSFT_TargetPortal)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " MSFT_TargetPortal is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = MSFT_TargetPortal(valuetmp)

	return
}
