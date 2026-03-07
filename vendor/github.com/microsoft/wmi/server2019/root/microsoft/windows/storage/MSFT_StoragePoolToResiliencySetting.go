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

// MSFT_StoragePoolToResiliencySetting struct
type MSFT_StoragePoolToResiliencySetting struct {
	*cim.WmiInstance

	//
	ResiliencySetting MSFT_ResiliencySetting

	//
	StoragePool MSFT_StoragePool
}

func NewMSFT_StoragePoolToResiliencySettingEx1(instance *cim.WmiInstance) (newInstance *MSFT_StoragePoolToResiliencySetting, err error) {
	tmp, err := instance, nil

	if err != nil {
		return
	}
	newInstance = &MSFT_StoragePoolToResiliencySetting{
		WmiInstance: tmp,
	}
	return
}

func NewMSFT_StoragePoolToResiliencySettingEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *MSFT_StoragePoolToResiliencySetting, err error) {
	tmp, err := instance.GetWmiInstance(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &MSFT_StoragePoolToResiliencySetting{
		WmiInstance: tmp,
	}
	return
}

// SetResiliencySetting sets the value of ResiliencySetting for the instance
func (instance *MSFT_StoragePoolToResiliencySetting) SetPropertyResiliencySetting(value MSFT_ResiliencySetting) (err error) {
	return instance.SetProperty("ResiliencySetting", (value))
}

// GetResiliencySetting gets the value of ResiliencySetting for the instance
func (instance *MSFT_StoragePoolToResiliencySetting) GetPropertyResiliencySetting() (value MSFT_ResiliencySetting, err error) {
	retValue, err := instance.GetProperty("ResiliencySetting")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(MSFT_ResiliencySetting)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " MSFT_ResiliencySetting is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = MSFT_ResiliencySetting(valuetmp)

	return
}

// SetStoragePool sets the value of StoragePool for the instance
func (instance *MSFT_StoragePoolToResiliencySetting) SetPropertyStoragePool(value MSFT_StoragePool) (err error) {
	return instance.SetProperty("StoragePool", (value))
}

// GetStoragePool gets the value of StoragePool for the instance
func (instance *MSFT_StoragePoolToResiliencySetting) GetPropertyStoragePool() (value MSFT_StoragePool, err error) {
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
