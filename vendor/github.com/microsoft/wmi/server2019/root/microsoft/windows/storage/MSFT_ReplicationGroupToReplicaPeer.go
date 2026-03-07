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

// MSFT_ReplicationGroupToReplicaPeer struct
type MSFT_ReplicationGroupToReplicaPeer struct {
	*MSFT_Synchronized

	//
	ConsistencyState uint16

	//
	ConsistencyType uint16

	//
	ReplicaPeer MSFT_ReplicaPeer

	//
	ReplicationGroup MSFT_ReplicationGroup
}

func NewMSFT_ReplicationGroupToReplicaPeerEx1(instance *cim.WmiInstance) (newInstance *MSFT_ReplicationGroupToReplicaPeer, err error) {
	tmp, err := NewMSFT_SynchronizedEx1(instance)

	if err != nil {
		return
	}
	newInstance = &MSFT_ReplicationGroupToReplicaPeer{
		MSFT_Synchronized: tmp,
	}
	return
}

func NewMSFT_ReplicationGroupToReplicaPeerEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *MSFT_ReplicationGroupToReplicaPeer, err error) {
	tmp, err := NewMSFT_SynchronizedEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &MSFT_ReplicationGroupToReplicaPeer{
		MSFT_Synchronized: tmp,
	}
	return
}

// SetConsistencyState sets the value of ConsistencyState for the instance
func (instance *MSFT_ReplicationGroupToReplicaPeer) SetPropertyConsistencyState(value uint16) (err error) {
	return instance.SetProperty("ConsistencyState", (value))
}

// GetConsistencyState gets the value of ConsistencyState for the instance
func (instance *MSFT_ReplicationGroupToReplicaPeer) GetPropertyConsistencyState() (value uint16, err error) {
	retValue, err := instance.GetProperty("ConsistencyState")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint16)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint16 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint16(valuetmp)

	return
}

// SetConsistencyType sets the value of ConsistencyType for the instance
func (instance *MSFT_ReplicationGroupToReplicaPeer) SetPropertyConsistencyType(value uint16) (err error) {
	return instance.SetProperty("ConsistencyType", (value))
}

// GetConsistencyType gets the value of ConsistencyType for the instance
func (instance *MSFT_ReplicationGroupToReplicaPeer) GetPropertyConsistencyType() (value uint16, err error) {
	retValue, err := instance.GetProperty("ConsistencyType")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint16)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint16 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint16(valuetmp)

	return
}

// SetReplicaPeer sets the value of ReplicaPeer for the instance
func (instance *MSFT_ReplicationGroupToReplicaPeer) SetPropertyReplicaPeer(value MSFT_ReplicaPeer) (err error) {
	return instance.SetProperty("ReplicaPeer", (value))
}

// GetReplicaPeer gets the value of ReplicaPeer for the instance
func (instance *MSFT_ReplicationGroupToReplicaPeer) GetPropertyReplicaPeer() (value MSFT_ReplicaPeer, err error) {
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

// SetReplicationGroup sets the value of ReplicationGroup for the instance
func (instance *MSFT_ReplicationGroupToReplicaPeer) SetPropertyReplicationGroup(value MSFT_ReplicationGroup) (err error) {
	return instance.SetProperty("ReplicationGroup", (value))
}

// GetReplicationGroup gets the value of ReplicationGroup for the instance
func (instance *MSFT_ReplicationGroupToReplicaPeer) GetPropertyReplicationGroup() (value MSFT_ReplicationGroup, err error) {
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
