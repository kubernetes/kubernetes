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

// MSFT_StorageSubSystemToOffloadDataTransferSetting struct
type MSFT_StorageSubSystemToOffloadDataTransferSetting struct {
	*cim.WmiInstance

	//
	OffloadDataTransferSetting MSFT_OffloadDataTransferSetting

	//
	StorageSubSystem MSFT_StorageSubSystem
}

func NewMSFT_StorageSubSystemToOffloadDataTransferSettingEx1(instance *cim.WmiInstance) (newInstance *MSFT_StorageSubSystemToOffloadDataTransferSetting, err error) {
	tmp, err := instance, nil

	if err != nil {
		return
	}
	newInstance = &MSFT_StorageSubSystemToOffloadDataTransferSetting{
		WmiInstance: tmp,
	}
	return
}

func NewMSFT_StorageSubSystemToOffloadDataTransferSettingEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *MSFT_StorageSubSystemToOffloadDataTransferSetting, err error) {
	tmp, err := instance.GetWmiInstance(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &MSFT_StorageSubSystemToOffloadDataTransferSetting{
		WmiInstance: tmp,
	}
	return
}

// SetOffloadDataTransferSetting sets the value of OffloadDataTransferSetting for the instance
func (instance *MSFT_StorageSubSystemToOffloadDataTransferSetting) SetPropertyOffloadDataTransferSetting(value MSFT_OffloadDataTransferSetting) (err error) {
	return instance.SetProperty("OffloadDataTransferSetting", (value))
}

// GetOffloadDataTransferSetting gets the value of OffloadDataTransferSetting for the instance
func (instance *MSFT_StorageSubSystemToOffloadDataTransferSetting) GetPropertyOffloadDataTransferSetting() (value MSFT_OffloadDataTransferSetting, err error) {
	retValue, err := instance.GetProperty("OffloadDataTransferSetting")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(MSFT_OffloadDataTransferSetting)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " MSFT_OffloadDataTransferSetting is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = MSFT_OffloadDataTransferSetting(valuetmp)

	return
}

// SetStorageSubSystem sets the value of StorageSubSystem for the instance
func (instance *MSFT_StorageSubSystemToOffloadDataTransferSetting) SetPropertyStorageSubSystem(value MSFT_StorageSubSystem) (err error) {
	return instance.SetProperty("StorageSubSystem", (value))
}

// GetStorageSubSystem gets the value of StorageSubSystem for the instance
func (instance *MSFT_StorageSubSystemToOffloadDataTransferSetting) GetPropertyStorageSubSystem() (value MSFT_StorageSubSystem, err error) {
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
