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

// MSFT_VirtualDiskToStorageTier struct
type MSFT_VirtualDiskToStorageTier struct {
	*cim.WmiInstance

	//
	StorageTier MSFT_StorageTier

	//
	VirtualDisk MSFT_VirtualDisk
}

func NewMSFT_VirtualDiskToStorageTierEx1(instance *cim.WmiInstance) (newInstance *MSFT_VirtualDiskToStorageTier, err error) {
	tmp, err := instance, nil

	if err != nil {
		return
	}
	newInstance = &MSFT_VirtualDiskToStorageTier{
		WmiInstance: tmp,
	}
	return
}

func NewMSFT_VirtualDiskToStorageTierEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *MSFT_VirtualDiskToStorageTier, err error) {
	tmp, err := instance.GetWmiInstance(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &MSFT_VirtualDiskToStorageTier{
		WmiInstance: tmp,
	}
	return
}

// SetStorageTier sets the value of StorageTier for the instance
func (instance *MSFT_VirtualDiskToStorageTier) SetPropertyStorageTier(value MSFT_StorageTier) (err error) {
	return instance.SetProperty("StorageTier", (value))
}

// GetStorageTier gets the value of StorageTier for the instance
func (instance *MSFT_VirtualDiskToStorageTier) GetPropertyStorageTier() (value MSFT_StorageTier, err error) {
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

// SetVirtualDisk sets the value of VirtualDisk for the instance
func (instance *MSFT_VirtualDiskToStorageTier) SetPropertyVirtualDisk(value MSFT_VirtualDisk) (err error) {
	return instance.SetProperty("VirtualDisk", (value))
}

// GetVirtualDisk gets the value of VirtualDisk for the instance
func (instance *MSFT_VirtualDiskToStorageTier) GetPropertyVirtualDisk() (value MSFT_VirtualDisk, err error) {
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
