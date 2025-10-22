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

// MSFT_StorageNodeToPhysicalDisk struct
type MSFT_StorageNodeToPhysicalDisk struct {
	*cim.WmiInstance

	//
	DiskNumber uint32

	//
	HealthStatus uint16

	//
	IsMpioEnabled bool

	//
	IsPhysicallyConnected bool

	//
	LoadBalancePolicy uint16

	//
	OperationalStatus []uint16

	//
	PathId []string

	//
	PathState []uint16

	//
	PhysicalDisk MSFT_PhysicalDisk

	//
	StorageNode MSFT_StorageNode
}

func NewMSFT_StorageNodeToPhysicalDiskEx1(instance *cim.WmiInstance) (newInstance *MSFT_StorageNodeToPhysicalDisk, err error) {
	tmp, err := instance, nil

	if err != nil {
		return
	}
	newInstance = &MSFT_StorageNodeToPhysicalDisk{
		WmiInstance: tmp,
	}
	return
}

func NewMSFT_StorageNodeToPhysicalDiskEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *MSFT_StorageNodeToPhysicalDisk, err error) {
	tmp, err := instance.GetWmiInstance(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &MSFT_StorageNodeToPhysicalDisk{
		WmiInstance: tmp,
	}
	return
}

// SetDiskNumber sets the value of DiskNumber for the instance
func (instance *MSFT_StorageNodeToPhysicalDisk) SetPropertyDiskNumber(value uint32) (err error) {
	return instance.SetProperty("DiskNumber", (value))
}

// GetDiskNumber gets the value of DiskNumber for the instance
func (instance *MSFT_StorageNodeToPhysicalDisk) GetPropertyDiskNumber() (value uint32, err error) {
	retValue, err := instance.GetProperty("DiskNumber")
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

// SetHealthStatus sets the value of HealthStatus for the instance
func (instance *MSFT_StorageNodeToPhysicalDisk) SetPropertyHealthStatus(value uint16) (err error) {
	return instance.SetProperty("HealthStatus", (value))
}

// GetHealthStatus gets the value of HealthStatus for the instance
func (instance *MSFT_StorageNodeToPhysicalDisk) GetPropertyHealthStatus() (value uint16, err error) {
	retValue, err := instance.GetProperty("HealthStatus")
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

// SetIsMpioEnabled sets the value of IsMpioEnabled for the instance
func (instance *MSFT_StorageNodeToPhysicalDisk) SetPropertyIsMpioEnabled(value bool) (err error) {
	return instance.SetProperty("IsMpioEnabled", (value))
}

// GetIsMpioEnabled gets the value of IsMpioEnabled for the instance
func (instance *MSFT_StorageNodeToPhysicalDisk) GetPropertyIsMpioEnabled() (value bool, err error) {
	retValue, err := instance.GetProperty("IsMpioEnabled")
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

// SetIsPhysicallyConnected sets the value of IsPhysicallyConnected for the instance
func (instance *MSFT_StorageNodeToPhysicalDisk) SetPropertyIsPhysicallyConnected(value bool) (err error) {
	return instance.SetProperty("IsPhysicallyConnected", (value))
}

// GetIsPhysicallyConnected gets the value of IsPhysicallyConnected for the instance
func (instance *MSFT_StorageNodeToPhysicalDisk) GetPropertyIsPhysicallyConnected() (value bool, err error) {
	retValue, err := instance.GetProperty("IsPhysicallyConnected")
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

// SetLoadBalancePolicy sets the value of LoadBalancePolicy for the instance
func (instance *MSFT_StorageNodeToPhysicalDisk) SetPropertyLoadBalancePolicy(value uint16) (err error) {
	return instance.SetProperty("LoadBalancePolicy", (value))
}

// GetLoadBalancePolicy gets the value of LoadBalancePolicy for the instance
func (instance *MSFT_StorageNodeToPhysicalDisk) GetPropertyLoadBalancePolicy() (value uint16, err error) {
	retValue, err := instance.GetProperty("LoadBalancePolicy")
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

// SetOperationalStatus sets the value of OperationalStatus for the instance
func (instance *MSFT_StorageNodeToPhysicalDisk) SetPropertyOperationalStatus(value []uint16) (err error) {
	return instance.SetProperty("OperationalStatus", (value))
}

// GetOperationalStatus gets the value of OperationalStatus for the instance
func (instance *MSFT_StorageNodeToPhysicalDisk) GetPropertyOperationalStatus() (value []uint16, err error) {
	retValue, err := instance.GetProperty("OperationalStatus")
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

// SetPathId sets the value of PathId for the instance
func (instance *MSFT_StorageNodeToPhysicalDisk) SetPropertyPathId(value []string) (err error) {
	return instance.SetProperty("PathId", (value))
}

// GetPathId gets the value of PathId for the instance
func (instance *MSFT_StorageNodeToPhysicalDisk) GetPropertyPathId() (value []string, err error) {
	retValue, err := instance.GetProperty("PathId")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	for _, interfaceValue := range retValue.([]interface{}) {
		valuetmp, ok := interfaceValue.(string)
		if !ok {
			err = errors.Wrapf(errors.InvalidType, " string is Invalid. Expected %s", reflect.TypeOf(interfaceValue))
			return
		}
		value = append(value, string(valuetmp))
	}

	return
}

// SetPathState sets the value of PathState for the instance
func (instance *MSFT_StorageNodeToPhysicalDisk) SetPropertyPathState(value []uint16) (err error) {
	return instance.SetProperty("PathState", (value))
}

// GetPathState gets the value of PathState for the instance
func (instance *MSFT_StorageNodeToPhysicalDisk) GetPropertyPathState() (value []uint16, err error) {
	retValue, err := instance.GetProperty("PathState")
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

// SetPhysicalDisk sets the value of PhysicalDisk for the instance
func (instance *MSFT_StorageNodeToPhysicalDisk) SetPropertyPhysicalDisk(value MSFT_PhysicalDisk) (err error) {
	return instance.SetProperty("PhysicalDisk", (value))
}

// GetPhysicalDisk gets the value of PhysicalDisk for the instance
func (instance *MSFT_StorageNodeToPhysicalDisk) GetPropertyPhysicalDisk() (value MSFT_PhysicalDisk, err error) {
	retValue, err := instance.GetProperty("PhysicalDisk")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(MSFT_PhysicalDisk)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " MSFT_PhysicalDisk is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = MSFT_PhysicalDisk(valuetmp)

	return
}

// SetStorageNode sets the value of StorageNode for the instance
func (instance *MSFT_StorageNodeToPhysicalDisk) SetPropertyStorageNode(value MSFT_StorageNode) (err error) {
	return instance.SetProperty("StorageNode", (value))
}

// GetStorageNode gets the value of StorageNode for the instance
func (instance *MSFT_StorageNodeToPhysicalDisk) GetPropertyStorageNode() (value MSFT_StorageNode, err error) {
	retValue, err := instance.GetProperty("StorageNode")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(MSFT_StorageNode)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " MSFT_StorageNode is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = MSFT_StorageNode(valuetmp)

	return
}
