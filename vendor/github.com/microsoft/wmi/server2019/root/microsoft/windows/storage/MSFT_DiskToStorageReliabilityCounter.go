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

// MSFT_DiskToStorageReliabilityCounter struct
type MSFT_DiskToStorageReliabilityCounter struct {
	*cim.WmiInstance

	//
	Disk MSFT_Disk

	//
	StorageReliabilityCounter MSFT_StorageReliabilityCounter
}

func NewMSFT_DiskToStorageReliabilityCounterEx1(instance *cim.WmiInstance) (newInstance *MSFT_DiskToStorageReliabilityCounter, err error) {
	tmp, err := instance, nil

	if err != nil {
		return
	}
	newInstance = &MSFT_DiskToStorageReliabilityCounter{
		WmiInstance: tmp,
	}
	return
}

func NewMSFT_DiskToStorageReliabilityCounterEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *MSFT_DiskToStorageReliabilityCounter, err error) {
	tmp, err := instance.GetWmiInstance(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &MSFT_DiskToStorageReliabilityCounter{
		WmiInstance: tmp,
	}
	return
}

// SetDisk sets the value of Disk for the instance
func (instance *MSFT_DiskToStorageReliabilityCounter) SetPropertyDisk(value MSFT_Disk) (err error) {
	return instance.SetProperty("Disk", (value))
}

// GetDisk gets the value of Disk for the instance
func (instance *MSFT_DiskToStorageReliabilityCounter) GetPropertyDisk() (value MSFT_Disk, err error) {
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

// SetStorageReliabilityCounter sets the value of StorageReliabilityCounter for the instance
func (instance *MSFT_DiskToStorageReliabilityCounter) SetPropertyStorageReliabilityCounter(value MSFT_StorageReliabilityCounter) (err error) {
	return instance.SetProperty("StorageReliabilityCounter", (value))
}

// GetStorageReliabilityCounter gets the value of StorageReliabilityCounter for the instance
func (instance *MSFT_DiskToStorageReliabilityCounter) GetPropertyStorageReliabilityCounter() (value MSFT_StorageReliabilityCounter, err error) {
	retValue, err := instance.GetProperty("StorageReliabilityCounter")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(MSFT_StorageReliabilityCounter)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " MSFT_StorageReliabilityCounter is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = MSFT_StorageReliabilityCounter(valuetmp)

	return
}
