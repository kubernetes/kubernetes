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

// MSFT_StoragePoolToStorageTier struct
type MSFT_StoragePoolToStorageTier struct {
	*cim.WmiInstance

	//
	StoragePool MSFT_StoragePool

	//
	StorageTier MSFT_StorageTier
}

func NewMSFT_StoragePoolToStorageTierEx1(instance *cim.WmiInstance) (newInstance *MSFT_StoragePoolToStorageTier, err error) {
	tmp, err := instance, nil

	if err != nil {
		return
	}
	newInstance = &MSFT_StoragePoolToStorageTier{
		WmiInstance: tmp,
	}
	return
}

func NewMSFT_StoragePoolToStorageTierEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *MSFT_StoragePoolToStorageTier, err error) {
	tmp, err := instance.GetWmiInstance(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &MSFT_StoragePoolToStorageTier{
		WmiInstance: tmp,
	}
	return
}

// SetStoragePool sets the value of StoragePool for the instance
func (instance *MSFT_StoragePoolToStorageTier) SetPropertyStoragePool(value MSFT_StoragePool) (err error) {
	return instance.SetProperty("StoragePool", (value))
}

// GetStoragePool gets the value of StoragePool for the instance
func (instance *MSFT_StoragePoolToStorageTier) GetPropertyStoragePool() (value MSFT_StoragePool, err error) {
	retValue, err := instance.GetProperty("StoragePool")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(MSFT_StoragePool)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " MSFT_StoragePool is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = MSFT_StoragePool(valuetmp)

	return
}

// SetStorageTier sets the value of StorageTier for the instance
func (instance *MSFT_StoragePoolToStorageTier) SetPropertyStorageTier(value MSFT_StorageTier) (err error) {
	return instance.SetProperty("StorageTier", (value))
}

// GetStorageTier gets the value of StorageTier for the instance
func (instance *MSFT_StoragePoolToStorageTier) GetPropertyStorageTier() (value MSFT_StorageTier, err error) {
	retValue, err := instance.GetProperty("StorageTier")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(MSFT_StorageTier)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " MSFT_StorageTier is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = MSFT_StorageTier(valuetmp)

	return
}
