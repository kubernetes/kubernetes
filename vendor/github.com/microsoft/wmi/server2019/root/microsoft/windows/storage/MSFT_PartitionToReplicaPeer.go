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

// MSFT_PartitionToReplicaPeer struct
type MSFT_PartitionToReplicaPeer struct {
	*MSFT_Synchronized

	//
	Partition MSFT_Partition

	//
	ReplicaPeer MSFT_ReplicaPeer
}

func NewMSFT_PartitionToReplicaPeerEx1(instance *cim.WmiInstance) (newInstance *MSFT_PartitionToReplicaPeer, err error) {
	tmp, err := NewMSFT_SynchronizedEx1(instance)

	if err != nil {
		return
	}
	newInstance = &MSFT_PartitionToReplicaPeer{
		MSFT_Synchronized: tmp,
	}
	return
}

func NewMSFT_PartitionToReplicaPeerEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *MSFT_PartitionToReplicaPeer, err error) {
	tmp, err := NewMSFT_SynchronizedEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &MSFT_PartitionToReplicaPeer{
		MSFT_Synchronized: tmp,
	}
	return
}

// SetPartition sets the value of Partition for the instance
func (instance *MSFT_PartitionToReplicaPeer) SetPropertyPartition(value MSFT_Partition) (err error) {
	return instance.SetProperty("Partition", (value))
}

// GetPartition gets the value of Partition for the instance
func (instance *MSFT_PartitionToReplicaPeer) GetPropertyPartition() (value MSFT_Partition, err error) {
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

// SetReplicaPeer sets the value of ReplicaPeer for the instance
func (instance *MSFT_PartitionToReplicaPeer) SetPropertyReplicaPeer(value MSFT_ReplicaPeer) (err error) {
	return instance.SetProperty("ReplicaPeer", (value))
}

// GetReplicaPeer gets the value of ReplicaPeer for the instance
func (instance *MSFT_PartitionToReplicaPeer) GetPropertyReplicaPeer() (value MSFT_ReplicaPeer, err error) {
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
