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

// MSFT_StorageProviderToStorageSubSystem struct
type MSFT_StorageProviderToStorageSubSystem struct {
	*cim.WmiInstance

	//
	StorageProvider MSFT_StorageProvider

	//
	StorageSubSystem MSFT_StorageSubSystem
}

func NewMSFT_StorageProviderToStorageSubSystemEx1(instance *cim.WmiInstance) (newInstance *MSFT_StorageProviderToStorageSubSystem, err error) {
	tmp, err := instance, nil

	if err != nil {
		return
	}
	newInstance = &MSFT_StorageProviderToStorageSubSystem{
		WmiInstance: tmp,
	}
	return
}

func NewMSFT_StorageProviderToStorageSubSystemEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *MSFT_StorageProviderToStorageSubSystem, err error) {
	tmp, err := instance.GetWmiInstance(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &MSFT_StorageProviderToStorageSubSystem{
		WmiInstance: tmp,
	}
	return
}

// SetStorageProvider sets the value of StorageProvider for the instance
func (instance *MSFT_StorageProviderToStorageSubSystem) SetPropertyStorageProvider(value MSFT_StorageProvider) (err error) {
	return instance.SetProperty("StorageProvider", (value))
}

// GetStorageProvider gets the value of StorageProvider for the instance
func (instance *MSFT_StorageProviderToStorageSubSystem) GetPropertyStorageProvider() (value MSFT_StorageProvider, err error) {
	retValue, err := instance.GetProperty("StorageProvider")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(MSFT_StorageProvider)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " MSFT_StorageProvider is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = MSFT_StorageProvider(valuetmp)

	return
}

// SetStorageSubSystem sets the value of StorageSubSystem for the instance
func (instance *MSFT_StorageProviderToStorageSubSystem) SetPropertyStorageSubSystem(value MSFT_StorageSubSystem) (err error) {
	return instance.SetProperty("StorageSubSystem", (value))
}

// GetStorageSubSystem gets the value of StorageSubSystem for the instance
func (instance *MSFT_StorageProviderToStorageSubSystem) GetPropertyStorageSubSystem() (value MSFT_StorageSubSystem, err error) {
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
