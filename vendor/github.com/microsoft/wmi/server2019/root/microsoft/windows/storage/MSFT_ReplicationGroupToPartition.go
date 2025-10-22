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

// MSFT_ReplicationGroupToPartition struct
type MSFT_ReplicationGroupToPartition struct {
	*cim.WmiInstance

	//
	Partition MSFT_Partition

	//
	ReplicationGroup MSFT_ReplicationGroup
}

func NewMSFT_ReplicationGroupToPartitionEx1(instance *cim.WmiInstance) (newInstance *MSFT_ReplicationGroupToPartition, err error) {
	tmp, err := instance, nil

	if err != nil {
		return
	}
	newInstance = &MSFT_ReplicationGroupToPartition{
		WmiInstance: tmp,
	}
	return
}

func NewMSFT_ReplicationGroupToPartitionEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *MSFT_ReplicationGroupToPartition, err error) {
	tmp, err := instance.GetWmiInstance(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &MSFT_ReplicationGroupToPartition{
		WmiInstance: tmp,
	}
	return
}

// SetPartition sets the value of Partition for the instance
func (instance *MSFT_ReplicationGroupToPartition) SetPropertyPartition(value MSFT_Partition) (err error) {
	return instance.SetProperty("Partition", (value))
}

// GetPartition gets the value of Partition for the instance
func (instance *MSFT_ReplicationGroupToPartition) GetPropertyPartition() (value MSFT_Partition, err error) {
	retValue, err := instance.GetProperty("Partition")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(MSFT_Partition)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " MSFT_Partition is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = MSFT_Partition(valuetmp)

	return
}

// SetReplicationGroup sets the value of ReplicationGroup for the instance
func (instance *MSFT_ReplicationGroupToPartition) SetPropertyReplicationGroup(value MSFT_ReplicationGroup) (err error) {
	return instance.SetProperty("ReplicationGroup", (value))
}

// GetReplicationGroup gets the value of ReplicationGroup for the instance
func (instance *MSFT_ReplicationGroupToPartition) GetPropertyReplicationGroup() (value MSFT_ReplicationGroup, err error) {
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
