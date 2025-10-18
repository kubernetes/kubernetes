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

// MSFT_StorageSubSystemToDisk struct
type MSFT_StorageSubSystemToDisk struct {
	*cim.WmiInstance

	//
	Disk MSFT_Disk

	//
	StorageSubSystem MSFT_StorageSubSystem
}

func NewMSFT_StorageSubSystemToDiskEx1(instance *cim.WmiInstance) (newInstance *MSFT_StorageSubSystemToDisk, err error) {
	tmp, err := instance, nil

	if err != nil {
		return
	}
	newInstance = &MSFT_StorageSubSystemToDisk{
		WmiInstance: tmp,
	}
	return
}

func NewMSFT_StorageSubSystemToDiskEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *MSFT_StorageSubSystemToDisk, err error) {
	tmp, err := instance.GetWmiInstance(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &MSFT_StorageSubSystemToDisk{
		WmiInstance: tmp,
	}
	return
}

// SetDisk sets the value of Disk for the instance
func (instance *MSFT_StorageSubSystemToDisk) SetPropertyDisk(value MSFT_Disk) (err error) {
	return instance.SetProperty("Disk", (value))
}

// GetDisk gets the value of Disk for the instance
func (instance *MSFT_StorageSubSystemToDisk) GetPropertyDisk() (value MSFT_Disk, err error) {
	retValue, err := instance.GetProperty("Disk")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(MSFT_Disk)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " MSFT_Disk is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = MSFT_Disk(valuetmp)

	return
}

// SetStorageSubSystem sets the value of StorageSubSystem for the instance
func (instance *MSFT_StorageSubSystemToDisk) SetPropertyStorageSubSystem(value MSFT_StorageSubSystem) (err error) {
	return instance.SetProperty("StorageSubSystem", (value))
}

// GetStorageSubSystem gets the value of StorageSubSystem for the instance
func (instance *MSFT_StorageSubSystemToDisk) GetPropertyStorageSubSystem() (value MSFT_StorageSubSystem, err error) {
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
