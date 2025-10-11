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

// MSFT_StorageSubSystemToStorageEnclosure struct
type MSFT_StorageSubSystemToStorageEnclosure struct {
	*cim.WmiInstance

	//
	StorageEnclosure MSFT_StorageEnclosure

	//
	StorageSubSystem MSFT_StorageSubSystem
}

func NewMSFT_StorageSubSystemToStorageEnclosureEx1(instance *cim.WmiInstance) (newInstance *MSFT_StorageSubSystemToStorageEnclosure, err error) {
	tmp, err := instance, nil

	if err != nil {
		return
	}
	newInstance = &MSFT_StorageSubSystemToStorageEnclosure{
		WmiInstance: tmp,
	}
	return
}

func NewMSFT_StorageSubSystemToStorageEnclosureEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *MSFT_StorageSubSystemToStorageEnclosure, err error) {
	tmp, err := instance.GetWmiInstance(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &MSFT_StorageSubSystemToStorageEnclosure{
		WmiInstance: tmp,
	}
	return
}

// SetStorageEnclosure sets the value of StorageEnclosure for the instance
func (instance *MSFT_StorageSubSystemToStorageEnclosure) SetPropertyStorageEnclosure(value MSFT_StorageEnclosure) (err error) {
	return instance.SetProperty("StorageEnclosure", (value))
}

// GetStorageEnclosure gets the value of StorageEnclosure for the instance
func (instance *MSFT_StorageSubSystemToStorageEnclosure) GetPropertyStorageEnclosure() (value MSFT_StorageEnclosure, err error) {
	retValue, err := instance.GetProperty("StorageEnclosure")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(MSFT_StorageEnclosure)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " MSFT_StorageEnclosure is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = MSFT_StorageEnclosure(valuetmp)

	return
}

// SetStorageSubSystem sets the value of StorageSubSystem for the instance
func (instance *MSFT_StorageSubSystemToStorageEnclosure) SetPropertyStorageSubSystem(value MSFT_StorageSubSystem) (err error) {
	return instance.SetProperty("StorageSubSystem", (value))
}

// GetStorageSubSystem gets the value of StorageSubSystem for the instance
func (instance *MSFT_StorageSubSystemToStorageEnclosure) GetPropertyStorageSubSystem() (value MSFT_StorageSubSystem, err error) {
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
