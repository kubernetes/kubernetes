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

// MSFT_StorageNodeToStorageEnclosure struct
type MSFT_StorageNodeToStorageEnclosure struct {
	*cim.WmiInstance

	//
	CurrentSensorOperationalStatus []uint16

	//
	EnclosureNumber uint32

	//
	FanOperationalStatus []uint16

	//
	HealthStatus uint16

	//
	IOControllerOperationalStatus []uint16

	//
	IsPhysicallyConnected bool

	//
	PowerSupplyOperationalStatus []uint16

	//
	SlotOperationalStatus []uint16

	//
	StorageEnclosure MSFT_StorageEnclosure

	//
	StorageNode MSFT_StorageNode

	//
	TemperatureSensorOperationalStatus []uint16

	//
	VoltageSensorOperationalStatus []uint16
}

func NewMSFT_StorageNodeToStorageEnclosureEx1(instance *cim.WmiInstance) (newInstance *MSFT_StorageNodeToStorageEnclosure, err error) {
	tmp, err := instance, nil

	if err != nil {
		return
	}
	newInstance = &MSFT_StorageNodeToStorageEnclosure{
		WmiInstance: tmp,
	}
	return
}

func NewMSFT_StorageNodeToStorageEnclosureEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *MSFT_StorageNodeToStorageEnclosure, err error) {
	tmp, err := instance.GetWmiInstance(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &MSFT_StorageNodeToStorageEnclosure{
		WmiInstance: tmp,
	}
	return
}

// SetCurrentSensorOperationalStatus sets the value of CurrentSensorOperationalStatus for the instance
func (instance *MSFT_StorageNodeToStorageEnclosure) SetPropertyCurrentSensorOperationalStatus(value []uint16) (err error) {
	return instance.SetProperty("CurrentSensorOperationalStatus", (value))
}

// GetCurrentSensorOperationalStatus gets the value of CurrentSensorOperationalStatus for the instance
func (instance *MSFT_StorageNodeToStorageEnclosure) GetPropertyCurrentSensorOperationalStatus() (value []uint16, err error) {
	retValue, err := instance.GetProperty("CurrentSensorOperationalStatus")
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

// SetEnclosureNumber sets the value of EnclosureNumber for the instance
func (instance *MSFT_StorageNodeToStorageEnclosure) SetPropertyEnclosureNumber(value uint32) (err error) {
	return instance.SetProperty("EnclosureNumber", (value))
}

// GetEnclosureNumber gets the value of EnclosureNumber for the instance
func (instance *MSFT_StorageNodeToStorageEnclosure) GetPropertyEnclosureNumber() (value uint32, err error) {
	retValue, err := instance.GetProperty("EnclosureNumber")
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

// SetFanOperationalStatus sets the value of FanOperationalStatus for the instance
func (instance *MSFT_StorageNodeToStorageEnclosure) SetPropertyFanOperationalStatus(value []uint16) (err error) {
	return instance.SetProperty("FanOperationalStatus", (value))
}

// GetFanOperationalStatus gets the value of FanOperationalStatus for the instance
func (instance *MSFT_StorageNodeToStorageEnclosure) GetPropertyFanOperationalStatus() (value []uint16, err error) {
	retValue, err := instance.GetProperty("FanOperationalStatus")
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

// SetHealthStatus sets the value of HealthStatus for the instance
func (instance *MSFT_StorageNodeToStorageEnclosure) SetPropertyHealthStatus(value uint16) (err error) {
	return instance.SetProperty("HealthStatus", (value))
}

// GetHealthStatus gets the value of HealthStatus for the instance
func (instance *MSFT_StorageNodeToStorageEnclosure) GetPropertyHealthStatus() (value uint16, err error) {
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

// SetIOControllerOperationalStatus sets the value of IOControllerOperationalStatus for the instance
func (instance *MSFT_StorageNodeToStorageEnclosure) SetPropertyIOControllerOperationalStatus(value []uint16) (err error) {
	return instance.SetProperty("IOControllerOperationalStatus", (value))
}

// GetIOControllerOperationalStatus gets the value of IOControllerOperationalStatus for the instance
func (instance *MSFT_StorageNodeToStorageEnclosure) GetPropertyIOControllerOperationalStatus() (value []uint16, err error) {
	retValue, err := instance.GetProperty("IOControllerOperationalStatus")
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

// SetIsPhysicallyConnected sets the value of IsPhysicallyConnected for the instance
func (instance *MSFT_StorageNodeToStorageEnclosure) SetPropertyIsPhysicallyConnected(value bool) (err error) {
	return instance.SetProperty("IsPhysicallyConnected", (value))
}

// GetIsPhysicallyConnected gets the value of IsPhysicallyConnected for the instance
func (instance *MSFT_StorageNodeToStorageEnclosure) GetPropertyIsPhysicallyConnected() (value bool, err error) {
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

// SetPowerSupplyOperationalStatus sets the value of PowerSupplyOperationalStatus for the instance
func (instance *MSFT_StorageNodeToStorageEnclosure) SetPropertyPowerSupplyOperationalStatus(value []uint16) (err error) {
	return instance.SetProperty("PowerSupplyOperationalStatus", (value))
}

// GetPowerSupplyOperationalStatus gets the value of PowerSupplyOperationalStatus for the instance
func (instance *MSFT_StorageNodeToStorageEnclosure) GetPropertyPowerSupplyOperationalStatus() (value []uint16, err error) {
	retValue, err := instance.GetProperty("PowerSupplyOperationalStatus")
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

// SetSlotOperationalStatus sets the value of SlotOperationalStatus for the instance
func (instance *MSFT_StorageNodeToStorageEnclosure) SetPropertySlotOperationalStatus(value []uint16) (err error) {
	return instance.SetProperty("SlotOperationalStatus", (value))
}

// GetSlotOperationalStatus gets the value of SlotOperationalStatus for the instance
func (instance *MSFT_StorageNodeToStorageEnclosure) GetPropertySlotOperationalStatus() (value []uint16, err error) {
	retValue, err := instance.GetProperty("SlotOperationalStatus")
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

// SetStorageEnclosure sets the value of StorageEnclosure for the instance
func (instance *MSFT_StorageNodeToStorageEnclosure) SetPropertyStorageEnclosure(value MSFT_StorageEnclosure) (err error) {
	return instance.SetProperty("StorageEnclosure", (value))
}

// GetStorageEnclosure gets the value of StorageEnclosure for the instance
func (instance *MSFT_StorageNodeToStorageEnclosure) GetPropertyStorageEnclosure() (value MSFT_StorageEnclosure, err error) {
	retValue, err := instance.GetProperty("StorageEnclosure")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(MSFT_StorageEnclosure)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " MSFT_StorageEnclosure is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = MSFT_StorageEnclosure(valuetmp)

	return
}

// SetStorageNode sets the value of StorageNode for the instance
func (instance *MSFT_StorageNodeToStorageEnclosure) SetPropertyStorageNode(value MSFT_StorageNode) (err error) {
	return instance.SetProperty("StorageNode", (value))
}

// GetStorageNode gets the value of StorageNode for the instance
func (instance *MSFT_StorageNodeToStorageEnclosure) GetPropertyStorageNode() (value MSFT_StorageNode, err error) {
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

// SetTemperatureSensorOperationalStatus sets the value of TemperatureSensorOperationalStatus for the instance
func (instance *MSFT_StorageNodeToStorageEnclosure) SetPropertyTemperatureSensorOperationalStatus(value []uint16) (err error) {
	return instance.SetProperty("TemperatureSensorOperationalStatus", (value))
}

// GetTemperatureSensorOperationalStatus gets the value of TemperatureSensorOperationalStatus for the instance
func (instance *MSFT_StorageNodeToStorageEnclosure) GetPropertyTemperatureSensorOperationalStatus() (value []uint16, err error) {
	retValue, err := instance.GetProperty("TemperatureSensorOperationalStatus")
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

// SetVoltageSensorOperationalStatus sets the value of VoltageSensorOperationalStatus for the instance
func (instance *MSFT_StorageNodeToStorageEnclosure) SetPropertyVoltageSensorOperationalStatus(value []uint16) (err error) {
	return instance.SetProperty("VoltageSensorOperationalStatus", (value))
}

// GetVoltageSensorOperationalStatus gets the value of VoltageSensorOperationalStatus for the instance
func (instance *MSFT_StorageNodeToStorageEnclosure) GetPropertyVoltageSensorOperationalStatus() (value []uint16, err error) {
	retValue, err := instance.GetProperty("VoltageSensorOperationalStatus")
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
