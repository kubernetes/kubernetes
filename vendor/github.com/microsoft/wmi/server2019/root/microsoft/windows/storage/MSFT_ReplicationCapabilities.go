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

// MSFT_ReplicationCapabilities struct
type MSFT_ReplicationCapabilities struct {
	*MSFT_StorageObject

	//
	DefaultRecoveryPointObjective uint32

	//
	SupportedAsynchronousActions []uint16

	//
	SupportedLogVolumeFeatures []uint16

	//
	SupportedMaximumLogSize uint64

	//
	SupportedMinimumLogSize uint64

	//
	SupportedObjectTypes []uint16

	//
	SupportedReplicatedPartitionFeatures []uint16

	//
	SupportedReplicationTypes []uint16

	//
	SupportedSynchronousActions []uint16

	//
	SupportsCreateReplicationRelationshipMethod bool

	//
	SupportsEmptyReplicationGroup bool

	//
	SupportsFullDiscovery bool

	//
	SupportsReplicationGroup bool
}

func NewMSFT_ReplicationCapabilitiesEx1(instance *cim.WmiInstance) (newInstance *MSFT_ReplicationCapabilities, err error) {
	tmp, err := NewMSFT_StorageObjectEx1(instance)

	if err != nil {
		return
	}
	newInstance = &MSFT_ReplicationCapabilities{
		MSFT_StorageObject: tmp,
	}
	return
}

func NewMSFT_ReplicationCapabilitiesEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *MSFT_ReplicationCapabilities, err error) {
	tmp, err := NewMSFT_StorageObjectEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &MSFT_ReplicationCapabilities{
		MSFT_StorageObject: tmp,
	}
	return
}

// SetDefaultRecoveryPointObjective sets the value of DefaultRecoveryPointObjective for the instance
func (instance *MSFT_ReplicationCapabilities) SetPropertyDefaultRecoveryPointObjective(value uint32) (err error) {
	return instance.SetProperty("DefaultRecoveryPointObjective", (value))
}

// GetDefaultRecoveryPointObjective gets the value of DefaultRecoveryPointObjective for the instance
func (instance *MSFT_ReplicationCapabilities) GetPropertyDefaultRecoveryPointObjective() (value uint32, err error) {
	retValue, err := instance.GetProperty("DefaultRecoveryPointObjective")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint32)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint32 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint32(valuetmp)

	return
}

// SetSupportedAsynchronousActions sets the value of SupportedAsynchronousActions for the instance
func (instance *MSFT_ReplicationCapabilities) SetPropertySupportedAsynchronousActions(value []uint16) (err error) {
	return instance.SetProperty("SupportedAsynchronousActions", (value))
}

// GetSupportedAsynchronousActions gets the value of SupportedAsynchronousActions for the instance
func (instance *MSFT_ReplicationCapabilities) GetPropertySupportedAsynchronousActions() (value []uint16, err error) {
	retValue, err := instance.GetProperty("SupportedAsynchronousActions")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	for _, interfaceValue := range retValue.([]interface{}) {
		valuetmp, ok := interfaceValue.(uint16)
		if !ok {
			err = errors.Wrapf(errors.InvalidType, " uint16 is Invalid. Expected %s", reflect.TypeOf(interfaceValue))
			return
		}
		value = append(value, uint16(valuetmp))
	}

	return
}

// SetSupportedLogVolumeFeatures sets the value of SupportedLogVolumeFeatures for the instance
func (instance *MSFT_ReplicationCapabilities) SetPropertySupportedLogVolumeFeatures(value []uint16) (err error) {
	return instance.SetProperty("SupportedLogVolumeFeatures", (value))
}

// GetSupportedLogVolumeFeatures gets the value of SupportedLogVolumeFeatures for the instance
func (instance *MSFT_ReplicationCapabilities) GetPropertySupportedLogVolumeFeatures() (value []uint16, err error) {
	retValue, err := instance.GetProperty("SupportedLogVolumeFeatures")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	for _, interfaceValue := range retValue.([]interface{}) {
		valuetmp, ok := interfaceValue.(uint16)
		if !ok {
			err = errors.Wrapf(errors.InvalidType, " uint16 is Invalid. Expected %s", reflect.TypeOf(interfaceValue))
			return
		}
		value = append(value, uint16(valuetmp))
	}

	return
}

// SetSupportedMaximumLogSize sets the value of SupportedMaximumLogSize for the instance
func (instance *MSFT_ReplicationCapabilities) SetPropertySupportedMaximumLogSize(value uint64) (err error) {
	return instance.SetProperty("SupportedMaximumLogSize", (value))
}

// GetSupportedMaximumLogSize gets the value of SupportedMaximumLogSize for the instance
func (instance *MSFT_ReplicationCapabilities) GetPropertySupportedMaximumLogSize() (value uint64, err error) {
	retValue, err := instance.GetProperty("SupportedMaximumLogSize")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint64)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint64 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint64(valuetmp)

	return
}

// SetSupportedMinimumLogSize sets the value of SupportedMinimumLogSize for the instance
func (instance *MSFT_ReplicationCapabilities) SetPropertySupportedMinimumLogSize(value uint64) (err error) {
	return instance.SetProperty("SupportedMinimumLogSize", (value))
}

// GetSupportedMinimumLogSize gets the value of SupportedMinimumLogSize for the instance
func (instance *MSFT_ReplicationCapabilities) GetPropertySupportedMinimumLogSize() (value uint64, err error) {
	retValue, err := instance.GetProperty("SupportedMinimumLogSize")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint64)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint64 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint64(valuetmp)

	return
}

// SetSupportedObjectTypes sets the value of SupportedObjectTypes for the instance
func (instance *MSFT_ReplicationCapabilities) SetPropertySupportedObjectTypes(value []uint16) (err error) {
	return instance.SetProperty("SupportedObjectTypes", (value))
}

// GetSupportedObjectTypes gets the value of SupportedObjectTypes for the instance
func (instance *MSFT_ReplicationCapabilities) GetPropertySupportedObjectTypes() (value []uint16, err error) {
	retValue, err := instance.GetProperty("SupportedObjectTypes")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	for _, interfaceValue := range retValue.([]interface{}) {
		valuetmp, ok := interfaceValue.(uint16)
		if !ok {
			err = errors.Wrapf(errors.InvalidType, " uint16 is Invalid. Expected %s", reflect.TypeOf(interfaceValue))
			return
		}
		value = append(value, uint16(valuetmp))
	}

	return
}

// SetSupportedReplicatedPartitionFeatures sets the value of SupportedReplicatedPartitionFeatures for the instance
func (instance *MSFT_ReplicationCapabilities) SetPropertySupportedReplicatedPartitionFeatures(value []uint16) (err error) {
	return instance.SetProperty("SupportedReplicatedPartitionFeatures", (value))
}

// GetSupportedReplicatedPartitionFeatures gets the value of SupportedReplicatedPartitionFeatures for the instance
func (instance *MSFT_ReplicationCapabilities) GetPropertySupportedReplicatedPartitionFeatures() (value []uint16, err error) {
	retValue, err := instance.GetProperty("SupportedReplicatedPartitionFeatures")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	for _, interfaceValue := range retValue.([]interface{}) {
		valuetmp, ok := interfaceValue.(uint16)
		if !ok {
			err = errors.Wrapf(errors.InvalidType, " uint16 is Invalid. Expected %s", reflect.TypeOf(interfaceValue))
			return
		}
		value = append(value, uint16(valuetmp))
	}

	return
}

// SetSupportedReplicationTypes sets the value of SupportedReplicationTypes for the instance
func (instance *MSFT_ReplicationCapabilities) SetPropertySupportedReplicationTypes(value []uint16) (err error) {
	return instance.SetProperty("SupportedReplicationTypes", (value))
}

// GetSupportedReplicationTypes gets the value of SupportedReplicationTypes for the instance
func (instance *MSFT_ReplicationCapabilities) GetPropertySupportedReplicationTypes() (value []uint16, err error) {
	retValue, err := instance.GetProperty("SupportedReplicationTypes")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	for _, interfaceValue := range retValue.([]interface{}) {
		valuetmp, ok := interfaceValue.(uint16)
		if !ok {
			err = errors.Wrapf(errors.InvalidType, " uint16 is Invalid. Expected %s", reflect.TypeOf(interfaceValue))
			return
		}
		value = append(value, uint16(valuetmp))
	}

	return
}

// SetSupportedSynchronousActions sets the value of SupportedSynchronousActions for the instance
func (instance *MSFT_ReplicationCapabilities) SetPropertySupportedSynchronousActions(value []uint16) (err error) {
	return instance.SetProperty("SupportedSynchronousActions", (value))
}

// GetSupportedSynchronousActions gets the value of SupportedSynchronousActions for the instance
func (instance *MSFT_ReplicationCapabilities) GetPropertySupportedSynchronousActions() (value []uint16, err error) {
	retValue, err := instance.GetProperty("SupportedSynchronousActions")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	for _, interfaceValue := range retValue.([]interface{}) {
		valuetmp, ok := interfaceValue.(uint16)
		if !ok {
			err = errors.Wrapf(errors.InvalidType, " uint16 is Invalid. Expected %s", reflect.TypeOf(interfaceValue))
			return
		}
		value = append(value, uint16(valuetmp))
	}

	return
}

// SetSupportsCreateReplicationRelationshipMethod sets the value of SupportsCreateReplicationRelationshipMethod for the instance
func (instance *MSFT_ReplicationCapabilities) SetPropertySupportsCreateReplicationRelationshipMethod(value bool) (err error) {
	return instance.SetProperty("SupportsCreateReplicationRelationshipMethod", (value))
}

// GetSupportsCreateReplicationRelationshipMethod gets the value of SupportsCreateReplicationRelationshipMethod for the instance
func (instance *MSFT_ReplicationCapabilities) GetPropertySupportsCreateReplicationRelationshipMethod() (value bool, err error) {
	retValue, err := instance.GetProperty("SupportsCreateReplicationRelationshipMethod")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(bool)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " bool is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = bool(valuetmp)

	return
}

// SetSupportsEmptyReplicationGroup sets the value of SupportsEmptyReplicationGroup for the instance
func (instance *MSFT_ReplicationCapabilities) SetPropertySupportsEmptyReplicationGroup(value bool) (err error) {
	return instance.SetProperty("SupportsEmptyReplicationGroup", (value))
}

// GetSupportsEmptyReplicationGroup gets the value of SupportsEmptyReplicationGroup for the instance
func (instance *MSFT_ReplicationCapabilities) GetPropertySupportsEmptyReplicationGroup() (value bool, err error) {
	retValue, err := instance.GetProperty("SupportsEmptyReplicationGroup")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(bool)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " bool is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = bool(valuetmp)

	return
}

// SetSupportsFullDiscovery sets the value of SupportsFullDiscovery for the instance
func (instance *MSFT_ReplicationCapabilities) SetPropertySupportsFullDiscovery(value bool) (err error) {
	return instance.SetProperty("SupportsFullDiscovery", (value))
}

// GetSupportsFullDiscovery gets the value of SupportsFullDiscovery for the instance
func (instance *MSFT_ReplicationCapabilities) GetPropertySupportsFullDiscovery() (value bool, err error) {
	retValue, err := instance.GetProperty("SupportsFullDiscovery")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(bool)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " bool is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = bool(valuetmp)

	return
}

// SetSupportsReplicationGroup sets the value of SupportsReplicationGroup for the instance
func (instance *MSFT_ReplicationCapabilities) SetPropertySupportsReplicationGroup(value bool) (err error) {
	return instance.SetProperty("SupportsReplicationGroup", (value))
}

// GetSupportsReplicationGroup gets the value of SupportsReplicationGroup for the instance
func (instance *MSFT_ReplicationCapabilities) GetPropertySupportsReplicationGroup() (value bool, err error) {
	retValue, err := instance.GetProperty("SupportsReplicationGroup")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(bool)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " bool is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = bool(valuetmp)

	return
}

//

// <param name="ReplicationType" type="uint16 "></param>

// <param name="ExtendedStatus" type="MSFT_StorageExtendedStatus "></param>
// <param name="ReturnValue" type="uint32 "></param>
// <param name="SupportedOperations" type="uint16 []"></param>
func (instance *MSFT_ReplicationCapabilities) GetSupportedOperations( /* IN */ ReplicationType uint16,
	/* OUT */ SupportedOperations []uint16,
	/* OUT */ ExtendedStatus MSFT_StorageExtendedStatus) (result uint32, err error) {
	retVal, err := instance.InvokeMethod("GetSupportedOperations", ReplicationType)
	if err != nil {
		return
	}
	retValue := retVal[0].(int32)
	result = uint32(retValue)
	return

}

//

// <param name="ReplicationType" type="uint16 "></param>

// <param name="ExtendedStatus" type="MSFT_StorageExtendedStatus "></param>
// <param name="ReturnValue" type="uint32 "></param>
// <param name="SupportedGroupOperations" type="uint16 []"></param>
func (instance *MSFT_ReplicationCapabilities) GetSupportedGroupOperations( /* IN */ ReplicationType uint16,
	/* OUT */ SupportedGroupOperations []uint16,
	/* OUT */ ExtendedStatus MSFT_StorageExtendedStatus) (result uint32, err error) {
	retVal, err := instance.InvokeMethod("GetSupportedGroupOperations", ReplicationType)
	if err != nil {
		return
	}
	retValue := retVal[0].(int32)
	result = uint32(retValue)
	return

}

//

// <param name="ReplicationType" type="uint16 "></param>

// <param name="ExtendedStatus" type="MSFT_StorageExtendedStatus "></param>
// <param name="Features" type="uint16 []"></param>
// <param name="ReturnValue" type="uint32 "></param>
func (instance *MSFT_ReplicationCapabilities) GetSupportedFeatures( /* IN */ ReplicationType uint16,
	/* OUT */ Features []uint16,
	/* OUT */ ExtendedStatus MSFT_StorageExtendedStatus) (result uint32, err error) {
	retVal, err := instance.InvokeMethod("GetSupportedFeatures", ReplicationType)
	if err != nil {
		return
	}
	retValue := retVal[0].(int32)
	result = uint32(retValue)
	return

}

//

// <param name="ReplicationType" type="uint16 "></param>

// <param name="ExtendedStatus" type="MSFT_StorageExtendedStatus "></param>
// <param name="GroupFeatures" type="uint16 []"></param>
// <param name="ReturnValue" type="uint32 "></param>
func (instance *MSFT_ReplicationCapabilities) GetSupportedGroupFeatures( /* IN */ ReplicationType uint16,
	/* OUT */ GroupFeatures []uint16,
	/* OUT */ ExtendedStatus MSFT_StorageExtendedStatus) (result uint32, err error) {
	retVal, err := instance.InvokeMethod("GetSupportedGroupFeatures", ReplicationType)
	if err != nil {
		return
	}
	retValue := retVal[0].(int32)
	result = uint32(retValue)
	return

}

//

// <param name="ReplicationType" type="uint16 "></param>

// <param name="ExtendedStatus" type="MSFT_StorageExtendedStatus "></param>
// <param name="ReturnValue" type="uint32 "></param>
// <param name="SupportedCopyStates" type="uint16 []"></param>
func (instance *MSFT_ReplicationCapabilities) GetSupportedCopyStates( /* IN */ ReplicationType uint16,
	/* OUT */ SupportedCopyStates []uint16,
	/* OUT */ ExtendedStatus MSFT_StorageExtendedStatus) (result uint32, err error) {
	retVal, err := instance.InvokeMethod("GetSupportedCopyStates", ReplicationType)
	if err != nil {
		return
	}
	retValue := retVal[0].(int32)
	result = uint32(retValue)
	return

}

//

// <param name="ReplicationType" type="uint16 "></param>

// <param name="ExtendedStatus" type="MSFT_StorageExtendedStatus "></param>
// <param name="ReturnValue" type="uint32 "></param>
// <param name="SupportedCopyStates" type="uint16 []"></param>
func (instance *MSFT_ReplicationCapabilities) GetSupportedGroupCopyStates( /* IN */ ReplicationType uint16,
	/* OUT */ SupportedCopyStates []uint16,
	/* OUT */ ExtendedStatus MSFT_StorageExtendedStatus) (result uint32, err error) {
	retVal, err := instance.InvokeMethod("GetSupportedGroupCopyStates", ReplicationType)
	if err != nil {
		return
	}
	retValue := retVal[0].(int32)
	result = uint32(retValue)
	return

}

//

// <param name="ReplicationType" type="uint16 "></param>

// <param name="DefaultRecoveryPoint" type="uint32 "></param>
// <param name="ExtendedStatus" type="MSFT_StorageExtendedStatus "></param>
// <param name="RecoveryPointIndicator" type="uint16 "></param>
// <param name="RecoveryPointValues" type="uint32 []"></param>
// <param name="ReturnValue" type="uint32 "></param>
func (instance *MSFT_ReplicationCapabilities) GetRecoveryPointData( /* IN */ ReplicationType uint16,
	/* OUT */ DefaultRecoveryPoint uint32,
	/* OUT */ RecoveryPointValues []uint32,
	/* OUT */ RecoveryPointIndicator uint16,
	/* OUT */ ExtendedStatus MSFT_StorageExtendedStatus) (result uint32, err error) {
	retVal, err := instance.InvokeMethod("GetRecoveryPointData", ReplicationType)
	if err != nil {
		return
	}
	retValue := retVal[0].(int32)
	result = uint32(retValue)
	return

}
