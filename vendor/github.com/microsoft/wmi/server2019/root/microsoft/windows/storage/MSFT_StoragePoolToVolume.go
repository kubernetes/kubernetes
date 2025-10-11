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

// MSFT_StoragePoolToVolume struct
type MSFT_StoragePoolToVolume struct {
	*cim.WmiInstance

	//
	StoragePool MSFT_StoragePool

	//
	Volume MSFT_Volume
}

func NewMSFT_StoragePoolToVolumeEx1(instance *cim.WmiInstance) (newInstance *MSFT_StoragePoolToVolume, err error) {
	tmp, err := instance, nil

	if err != nil {
		return
	}
	newInstance = &MSFT_StoragePoolToVolume{
		WmiInstance: tmp,
	}
	return
}

func NewMSFT_StoragePoolToVolumeEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *MSFT_StoragePoolToVolume, err error) {
	tmp, err := instance.GetWmiInstance(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &MSFT_StoragePoolToVolume{
		WmiInstance: tmp,
	}
	return
}

// SetStoragePool sets the value of StoragePool for the instance
func (instance *MSFT_StoragePoolToVolume) SetPropertyStoragePool(value MSFT_StoragePool) (err error) {
	return instance.SetProperty("StoragePool", (value))
}

// GetStoragePool gets the value of StoragePool for the instance
func (instance *MSFT_StoragePoolToVolume) GetPropertyStoragePool() (value MSFT_StoragePool, err error) {
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

// SetVolume sets the value of Volume for the instance
func (instance *MSFT_StoragePoolToVolume) SetPropertyVolume(value MSFT_Volume) (err error) {
	return instance.SetProperty("Volume", (value))
}

// GetVolume gets the value of Volume for the instance
func (instance *MSFT_StoragePoolToVolume) GetPropertyVolume() (value MSFT_Volume, err error) {
	retValue, err := instance.GetProperty("Volume")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(MSFT_Volume)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " MSFT_Volume is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = MSFT_Volume(valuetmp)

	return
}
