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

// MSFT_StorageSubSystemToStorageFaultDomain struct
type MSFT_StorageSubSystemToStorageFaultDomain struct {
	*cim.WmiInstance

	//
	StorageFaultDomain MSFT_StorageFaultDomain

	//
	StorageSubSystem MSFT_StorageSubSystem
}

func NewMSFT_StorageSubSystemToStorageFaultDomainEx1(instance *cim.WmiInstance) (newInstance *MSFT_StorageSubSystemToStorageFaultDomain, err error) {
	tmp, err := instance, nil

	if err != nil {
		return
	}
	newInstance = &MSFT_StorageSubSystemToStorageFaultDomain{
		WmiInstance: tmp,
	}
	return
}

func NewMSFT_StorageSubSystemToStorageFaultDomainEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *MSFT_StorageSubSystemToStorageFaultDomain, err error) {
	tmp, err := instance.GetWmiInstance(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &MSFT_StorageSubSystemToStorageFaultDomain{
		WmiInstance: tmp,
	}
	return
}

// SetStorageFaultDomain sets the value of StorageFaultDomain for the instance
func (instance *MSFT_StorageSubSystemToStorageFaultDomain) SetPropertyStorageFaultDomain(value MSFT_StorageFaultDomain) (err error) {
	return instance.SetProperty("StorageFaultDomain", (value))
}

// GetStorageFaultDomain gets the value of StorageFaultDomain for the instance
func (instance *MSFT_StorageSubSystemToStorageFaultDomain) GetPropertyStorageFaultDomain() (value MSFT_StorageFaultDomain, err error) {
	retValue, err := instance.GetProperty("StorageFaultDomain")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(MSFT_StorageFaultDomain)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " MSFT_StorageFaultDomain is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = MSFT_StorageFaultDomain(valuetmp)

	return
}

// SetStorageSubSystem sets the value of StorageSubSystem for the instance
func (instance *MSFT_StorageSubSystemToStorageFaultDomain) SetPropertyStorageSubSystem(value MSFT_StorageSubSystem) (err error) {
	return instance.SetProperty("StorageSubSystem", (value))
}

// GetStorageSubSystem gets the value of StorageSubSystem for the instance
func (instance *MSFT_StorageSubSystemToStorageFaultDomain) GetPropertyStorageSubSystem() (value MSFT_StorageSubSystem, err error) {
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
