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

// MSFT_StorageSubSystemToFileShare struct
type MSFT_StorageSubSystemToFileShare struct {
	*cim.WmiInstance

	//
	FileShare MSFT_FileShare

	//
	StorageSubSystem MSFT_StorageSubSystem
}

func NewMSFT_StorageSubSystemToFileShareEx1(instance *cim.WmiInstance) (newInstance *MSFT_StorageSubSystemToFileShare, err error) {
	tmp, err := instance, nil

	if err != nil {
		return
	}
	newInstance = &MSFT_StorageSubSystemToFileShare{
		WmiInstance: tmp,
	}
	return
}

func NewMSFT_StorageSubSystemToFileShareEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *MSFT_StorageSubSystemToFileShare, err error) {
	tmp, err := instance.GetWmiInstance(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &MSFT_StorageSubSystemToFileShare{
		WmiInstance: tmp,
	}
	return
}

// SetFileShare sets the value of FileShare for the instance
func (instance *MSFT_StorageSubSystemToFileShare) SetPropertyFileShare(value MSFT_FileShare) (err error) {
	return instance.SetProperty("FileShare", (value))
}

// GetFileShare gets the value of FileShare for the instance
func (instance *MSFT_StorageSubSystemToFileShare) GetPropertyFileShare() (value MSFT_FileShare, err error) {
	retValue, err := instance.GetProperty("FileShare")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(MSFT_FileShare)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " MSFT_FileShare is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = MSFT_FileShare(valuetmp)

	return
}

// SetStorageSubSystem sets the value of StorageSubSystem for the instance
func (instance *MSFT_StorageSubSystemToFileShare) SetPropertyStorageSubSystem(value MSFT_StorageSubSystem) (err error) {
	return instance.SetProperty("StorageSubSystem", (value))
}

// GetStorageSubSystem gets the value of StorageSubSystem for the instance
func (instance *MSFT_StorageSubSystemToFileShare) GetPropertyStorageSubSystem() (value MSFT_StorageSubSystem, err error) {
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
