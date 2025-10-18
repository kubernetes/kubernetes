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

// MSFT_StorageSubSystemToReplicationGroup struct
type MSFT_StorageSubSystemToReplicationGroup struct {
	*cim.WmiInstance

	//
	ReplicationGroup MSFT_ReplicationGroup

	//
	StorageSubSystem MSFT_StorageSubSystem
}

func NewMSFT_StorageSubSystemToReplicationGroupEx1(instance *cim.WmiInstance) (newInstance *MSFT_StorageSubSystemToReplicationGroup, err error) {
	tmp, err := instance, nil

	if err != nil {
		return
	}
	newInstance = &MSFT_StorageSubSystemToReplicationGroup{
		WmiInstance: tmp,
	}
	return
}

func NewMSFT_StorageSubSystemToReplicationGroupEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *MSFT_StorageSubSystemToReplicationGroup, err error) {
	tmp, err := instance.GetWmiInstance(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &MSFT_StorageSubSystemToReplicationGroup{
		WmiInstance: tmp,
	}
	return
}

// SetReplicationGroup sets the value of ReplicationGroup for the instance
func (instance *MSFT_StorageSubSystemToReplicationGroup) SetPropertyReplicationGroup(value MSFT_ReplicationGroup) (err error) {
	return instance.SetProperty("ReplicationGroup", (value))
}

// GetReplicationGroup gets the value of ReplicationGroup for the instance
func (instance *MSFT_StorageSubSystemToReplicationGroup) GetPropertyReplicationGroup() (value MSFT_ReplicationGroup, err error) {
	retValue, err := instance.GetProperty("ReplicationGroup")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(MSFT_ReplicationGroup)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " MSFT_ReplicationGroup is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = MSFT_ReplicationGroup(valuetmp)

	return
}

// SetStorageSubSystem sets the value of StorageSubSystem for the instance
func (instance *MSFT_StorageSubSystemToReplicationGroup) SetPropertyStorageSubSystem(value MSFT_StorageSubSystem) (err error) {
	return instance.SetProperty("StorageSubSystem", (value))
}

// GetStorageSubSystem gets the value of StorageSubSystem for the instance
func (instance *MSFT_StorageSubSystemToReplicationGroup) GetPropertyStorageSubSystem() (value MSFT_StorageSubSystem, err error) {
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
