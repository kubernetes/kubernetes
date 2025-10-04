// Copyright 2019 (c) Microsoft Corporation.
// Licensed under the MIT license.

//
// Author:
//      Auto Generated on 9/18/2020 using wmigen
//      Source root.Microsoft.Windows.Storage
//////////////////////////////////////////////
package storage

import (
	"github.com/microsoft/wmi/pkg/base/query"
	"github.com/microsoft/wmi/pkg/errors"
	cim "github.com/microsoft/wmi/pkg/wmiinstance"
	"reflect"
)

// MSFT_VirtualDiskToReplicaPeer struct
type MSFT_VirtualDiskToReplicaPeer struct {
	*MSFT_Synchronized

	//
	ReplicaPeer MSFT_ReplicaPeer

	//
	VirtualDisk MSFT_VirtualDisk
}

func NewMSFT_VirtualDiskToReplicaPeerEx1(instance *cim.WmiInstance) (newInstance *MSFT_VirtualDiskToReplicaPeer, err error) {
	tmp, err := NewMSFT_SynchronizedEx1(instance)

	if err != nil {
		return
	}
	newInstance = &MSFT_VirtualDiskToReplicaPeer{
		MSFT_Synchronized: tmp,
	}
	return
}

func NewMSFT_VirtualDiskToReplicaPeerEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *MSFT_VirtualDiskToReplicaPeer, err error) {
	tmp, err := NewMSFT_SynchronizedEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &MSFT_VirtualDiskToReplicaPeer{
		MSFT_Synchronized: tmp,
	}
	return
}

// SetReplicaPeer sets the value of ReplicaPeer for the instance
func (instance *MSFT_VirtualDiskToReplicaPeer) SetPropertyReplicaPeer(value MSFT_ReplicaPeer) (err error) {
	return instance.SetProperty("ReplicaPeer", (value))
}

// GetReplicaPeer gets the value of ReplicaPeer for the instance
func (instance *MSFT_VirtualDiskToReplicaPeer) GetPropertyReplicaPeer() (value MSFT_ReplicaPeer, err error) {
	retValue, err := instance.GetProperty("ReplicaPeer")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(MSFT_ReplicaPeer)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " MSFT_ReplicaPeer is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = MSFT_ReplicaPeer(valuetmp)

	return
}

// SetVirtualDisk sets the value of VirtualDisk for the instance
func (instance *MSFT_VirtualDiskToReplicaPeer) SetPropertyVirtualDisk(value MSFT_VirtualDisk) (err error) {
	return instance.SetProperty("VirtualDisk", (value))
}

// GetVirtualDisk gets the value of VirtualDisk for the instance
func (instance *MSFT_VirtualDiskToReplicaPeer) GetPropertyVirtualDisk() (value MSFT_VirtualDisk, err error) {
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
