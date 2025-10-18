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

// MSFT_StoragePoolToVirtualDisk struct
type MSFT_StoragePoolToVirtualDisk struct {
	*cim.WmiInstance

	//
	StoragePool MSFT_StoragePool

	//
	VirtualDisk MSFT_VirtualDisk
}

func NewMSFT_StoragePoolToVirtualDiskEx1(instance *cim.WmiInstance) (newInstance *MSFT_StoragePoolToVirtualDisk, err error) {
	tmp, err := instance, nil

	if err != nil {
		return
	}
	newInstance = &MSFT_StoragePoolToVirtualDisk{
		WmiInstance: tmp,
	}
	return
}

func NewMSFT_StoragePoolToVirtualDiskEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *MSFT_StoragePoolToVirtualDisk, err error) {
	tmp, err := instance.GetWmiInstance(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &MSFT_StoragePoolToVirtualDisk{
		WmiInstance: tmp,
	}
	return
}

// SetStoragePool sets the value of StoragePool for the instance
func (instance *MSFT_StoragePoolToVirtualDisk) SetPropertyStoragePool(value MSFT_StoragePool) (err error) {
	return instance.SetProperty("StoragePool", (value))
}

// GetStoragePool gets the value of StoragePool for the instance
func (instance *MSFT_StoragePoolToVirtualDisk) GetPropertyStoragePool() (value MSFT_StoragePool, err error) {
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

// SetVirtualDisk sets the value of VirtualDisk for the instance
func (instance *MSFT_StoragePoolToVirtualDisk) SetPropertyVirtualDisk(value MSFT_VirtualDisk) (err error) {
	return instance.SetProperty("VirtualDisk", (value))
}

// GetVirtualDisk gets the value of VirtualDisk for the instance
func (instance *MSFT_StoragePoolToVirtualDisk) GetPropertyVirtualDisk() (value MSFT_VirtualDisk, err error) {
	retValue, err := instance.GetProperty("VirtualDisk")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(MSFT_VirtualDisk)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " MSFT_VirtualDisk is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = MSFT_VirtualDisk(valuetmp)

	return
}
