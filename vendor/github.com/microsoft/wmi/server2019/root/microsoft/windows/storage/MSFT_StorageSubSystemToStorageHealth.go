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

// MSFT_StorageSubSystemToStorageHealth struct
type MSFT_StorageSubSystemToStorageHealth struct {
	*cim.WmiInstance

	//
	StorageHealth MSFT_StorageHealth

	//
	StorageSubSystem MSFT_StorageSubSystem
}

func NewMSFT_StorageSubSystemToStorageHealthEx1(instance *cim.WmiInstance) (newInstance *MSFT_StorageSubSystemToStorageHealth, err error) {
	tmp, err := instance, nil

	if err != nil {
		return
	}
	newInstance = &MSFT_StorageSubSystemToStorageHealth{
		WmiInstance: tmp,
	}
	return
}

func NewMSFT_StorageSubSystemToStorageHealthEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *MSFT_StorageSubSystemToStorageHealth, err error) {
	tmp, err := instance.GetWmiInstance(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &MSFT_StorageSubSystemToStorageHealth{
		WmiInstance: tmp,
	}
	return
}

// SetStorageHealth sets the value of StorageHealth for the instance
func (instance *MSFT_StorageSubSystemToStorageHealth) SetPropertyStorageHealth(value MSFT_StorageHealth) (err error) {
	return instance.SetProperty("StorageHealth", (value))
}

// GetStorageHealth gets the value of StorageHealth for the instance
func (instance *MSFT_StorageSubSystemToStorageHealth) GetPropertyStorageHealth() (value MSFT_StorageHealth, err error) {
	retValue, err := instance.GetProperty("StorageHealth")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(MSFT_StorageHealth)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " MSFT_StorageHealth is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = MSFT_StorageHealth(valuetmp)

	return
}

// SetStorageSubSystem sets the value of StorageSubSystem for the instance
func (instance *MSFT_StorageSubSystemToStorageHealth) SetPropertyStorageSubSystem(value MSFT_StorageSubSystem) (err error) {
	return instance.SetProperty("StorageSubSystem", (value))
}

// GetStorageSubSystem gets the value of StorageSubSystem for the instance
func (instance *MSFT_StorageSubSystemToStorageHealth) GetPropertyStorageSubSystem() (value MSFT_StorageSubSystem, err error) {
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
