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

// MSFT_VirtualDiskToStorageFaultDomain struct
type MSFT_VirtualDiskToStorageFaultDomain struct {
	*cim.WmiInstance

	//
	StorageFaultDomain MSFT_StorageFaultDomain

	//
	VirtualDisk MSFT_VirtualDisk
}

func NewMSFT_VirtualDiskToStorageFaultDomainEx1(instance *cim.WmiInstance) (newInstance *MSFT_VirtualDiskToStorageFaultDomain, err error) {
	tmp, err := instance, nil

	if err != nil {
		return
	}
	newInstance = &MSFT_VirtualDiskToStorageFaultDomain{
		WmiInstance: tmp,
	}
	return
}

func NewMSFT_VirtualDiskToStorageFaultDomainEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *MSFT_VirtualDiskToStorageFaultDomain, err error) {
	tmp, err := instance.GetWmiInstance(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &MSFT_VirtualDiskToStorageFaultDomain{
		WmiInstance: tmp,
	}
	return
}

// SetStorageFaultDomain sets the value of StorageFaultDomain for the instance
func (instance *MSFT_VirtualDiskToStorageFaultDomain) SetPropertyStorageFaultDomain(value MSFT_StorageFaultDomain) (err error) {
	return instance.SetProperty("StorageFaultDomain", (value))
}

// GetStorageFaultDomain gets the value of StorageFaultDomain for the instance
func (instance *MSFT_VirtualDiskToStorageFaultDomain) GetPropertyStorageFaultDomain() (value MSFT_StorageFaultDomain, err error) {
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

// SetVirtualDisk sets the value of VirtualDisk for the instance
func (instance *MSFT_VirtualDiskToStorageFaultDomain) SetPropertyVirtualDisk(value MSFT_VirtualDisk) (err error) {
	return instance.SetProperty("VirtualDisk", (value))
}

// GetVirtualDisk gets the value of VirtualDisk for the instance
func (instance *MSFT_VirtualDiskToStorageFaultDomain) GetPropertyVirtualDisk() (value MSFT_VirtualDisk, err error) {
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
