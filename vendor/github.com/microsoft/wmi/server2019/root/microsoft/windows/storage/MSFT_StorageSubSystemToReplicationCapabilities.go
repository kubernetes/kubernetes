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

// MSFT_StorageSubSystemToReplicationCapabilities struct
type MSFT_StorageSubSystemToReplicationCapabilities struct {
	*cim.WmiInstance

	//
	ReplicationCapabilities MSFT_ReplicationCapabilities

	//
	StorageSubSystem MSFT_StorageSubSystem
}

func NewMSFT_StorageSubSystemToReplicationCapabilitiesEx1(instance *cim.WmiInstance) (newInstance *MSFT_StorageSubSystemToReplicationCapabilities, err error) {
	tmp, err := instance, nil

	if err != nil {
		return
	}
	newInstance = &MSFT_StorageSubSystemToReplicationCapabilities{
		WmiInstance: tmp,
	}
	return
}

func NewMSFT_StorageSubSystemToReplicationCapabilitiesEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *MSFT_StorageSubSystemToReplicationCapabilities, err error) {
	tmp, err := instance.GetWmiInstance(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &MSFT_StorageSubSystemToReplicationCapabilities{
		WmiInstance: tmp,
	}
	return
}

// SetReplicationCapabilities sets the value of ReplicationCapabilities for the instance
func (instance *MSFT_StorageSubSystemToReplicationCapabilities) SetPropertyReplicationCapabilities(value MSFT_ReplicationCapabilities) (err error) {
	return instance.SetProperty("ReplicationCapabilities", (value))
}

// GetReplicationCapabilities gets the value of ReplicationCapabilities for the instance
func (instance *MSFT_StorageSubSystemToReplicationCapabilities) GetPropertyReplicationCapabilities() (value MSFT_ReplicationCapabilities, err error) {
	retValue, err := instance.GetProperty("ReplicationCapabilities")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(MSFT_ReplicationCapabilities)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " MSFT_ReplicationCapabilities is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = MSFT_ReplicationCapabilities(valuetmp)

	return
}

// SetStorageSubSystem sets the value of StorageSubSystem for the instance
func (instance *MSFT_StorageSubSystemToReplicationCapabilities) SetPropertyStorageSubSystem(value MSFT_StorageSubSystem) (err error) {
	return instance.SetProperty("StorageSubSystem", (value))
}

// GetStorageSubSystem gets the value of StorageSubSystem for the instance
func (instance *MSFT_StorageSubSystemToReplicationCapabilities) GetPropertyStorageSubSystem() (value MSFT_StorageSubSystem, err error) {
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
