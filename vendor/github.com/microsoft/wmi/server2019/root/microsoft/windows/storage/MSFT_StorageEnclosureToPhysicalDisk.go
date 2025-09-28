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

// MSFT_StorageEnclosureToPhysicalDisk struct
type MSFT_StorageEnclosureToPhysicalDisk struct {
	*cim.WmiInstance

	//
	PhysicalDisk MSFT_PhysicalDisk

	//
	StorageEnclosure MSFT_StorageEnclosure
}

func NewMSFT_StorageEnclosureToPhysicalDiskEx1(instance *cim.WmiInstance) (newInstance *MSFT_StorageEnclosureToPhysicalDisk, err error) {
	tmp, err := instance, nil

	if err != nil {
		return
	}
	newInstance = &MSFT_StorageEnclosureToPhysicalDisk{
		WmiInstance: tmp,
	}
	return
}

func NewMSFT_StorageEnclosureToPhysicalDiskEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *MSFT_StorageEnclosureToPhysicalDisk, err error) {
	tmp, err := instance.GetWmiInstance(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &MSFT_StorageEnclosureToPhysicalDisk{
		WmiInstance: tmp,
	}
	return
}

// SetPhysicalDisk sets the value of PhysicalDisk for the instance
func (instance *MSFT_StorageEnclosureToPhysicalDisk) SetPropertyPhysicalDisk(value MSFT_PhysicalDisk) (err error) {
	return instance.SetProperty("PhysicalDisk", (value))
}

// GetPhysicalDisk gets the value of PhysicalDisk for the instance
func (instance *MSFT_StorageEnclosureToPhysicalDisk) GetPropertyPhysicalDisk() (value MSFT_PhysicalDisk, err error) {
	retValue, err := instance.GetProperty("PhysicalDisk")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(MSFT_PhysicalDisk)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " MSFT_PhysicalDisk is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = MSFT_PhysicalDisk(valuetmp)

	return
}

// SetStorageEnclosure sets the value of StorageEnclosure for the instance
func (instance *MSFT_StorageEnclosureToPhysicalDisk) SetPropertyStorageEnclosure(value MSFT_StorageEnclosure) (err error) {
	return instance.SetProperty("StorageEnclosure", (value))
}

// GetStorageEnclosure gets the value of StorageEnclosure for the instance
func (instance *MSFT_StorageEnclosureToPhysicalDisk) GetPropertyStorageEnclosure() (value MSFT_StorageEnclosure, err error) {
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
