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

// MSFT_StorageSubSystemToReplicaPeer struct
type MSFT_StorageSubSystemToReplicaPeer struct {
	*cim.WmiInstance

	//
	ReplicaPeer MSFT_ReplicaPeer

	//
	StorageSubSystem MSFT_StorageSubSystem
}

func NewMSFT_StorageSubSystemToReplicaPeerEx1(instance *cim.WmiInstance) (newInstance *MSFT_StorageSubSystemToReplicaPeer, err error) {
	tmp, err := instance, nil

	if err != nil {
		return
	}
	newInstance = &MSFT_StorageSubSystemToReplicaPeer{
		WmiInstance: tmp,
	}
	return
}

func NewMSFT_StorageSubSystemToReplicaPeerEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *MSFT_StorageSubSystemToReplicaPeer, err error) {
	tmp, err := instance.GetWmiInstance(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &MSFT_StorageSubSystemToReplicaPeer{
		WmiInstance: tmp,
	}
	return
}

// SetReplicaPeer sets the value of ReplicaPeer for the instance
func (instance *MSFT_StorageSubSystemToReplicaPeer) SetPropertyReplicaPeer(value MSFT_ReplicaPeer) (err error) {
	return instance.SetProperty("ReplicaPeer", (value))
}

// GetReplicaPeer gets the value of ReplicaPeer for the instance
func (instance *MSFT_StorageSubSystemToReplicaPeer) GetPropertyReplicaPeer() (value MSFT_ReplicaPeer, err error) {
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

// SetStorageSubSystem sets the value of StorageSubSystem for the instance
func (instance *MSFT_StorageSubSystemToReplicaPeer) SetPropertyStorageSubSystem(value MSFT_StorageSubSystem) (err error) {
	return instance.SetProperty("StorageSubSystem", (value))
}

// GetStorageSubSystem gets the value of StorageSubSystem for the instance
func (instance *MSFT_StorageSubSystemToReplicaPeer) GetPropertyStorageSubSystem() (value MSFT_StorageSubSystem, err error) {
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
