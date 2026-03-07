// Copyright 2019 (c) Microsoft Corporation.
// Licensed under the MIT license.

//
// Author:
//      Auto Generated on 9/18/2020 using wmigen
//      Source root.CIMV2
//////////////////////////////////////////////
package cimv2

import (
	"github.com/microsoft/wmi/pkg/base/query"
	"github.com/microsoft/wmi/pkg/errors"
	cim "github.com/microsoft/wmi/pkg/wmiinstance"
	"reflect"
)

// CIM_UninterruptiblePowerSupply struct
type CIM_UninterruptiblePowerSupply struct {
	*CIM_PowerSupply

	//
	EstimatedChargeRemaining uint16

	//
	EstimatedRunTime uint32

	//
	RemainingCapacityStatus uint16

	//
	TimeOnBackup uint32
}

func NewCIM_UninterruptiblePowerSupplyEx1(instance *cim.WmiInstance) (newInstance *CIM_UninterruptiblePowerSupply, err error) {
	tmp, err := NewCIM_PowerSupplyEx1(instance)

	if err != nil {
		return
	}
	newInstance = &CIM_UninterruptiblePowerSupply{
		CIM_PowerSupply: tmp,
	}
	return
}

func NewCIM_UninterruptiblePowerSupplyEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *CIM_UninterruptiblePowerSupply, err error) {
	tmp, err := NewCIM_PowerSupplyEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &CIM_UninterruptiblePowerSupply{
		CIM_PowerSupply: tmp,
	}
	return
}

// SetEstimatedChargeRemaining sets the value of EstimatedChargeRemaining for the instance
func (instance *CIM_UninterruptiblePowerSupply) SetPropertyEstimatedChargeRemaining(value uint16) (err error) {
	return instance.SetProperty("EstimatedChargeRemaining", (value))
}

// GetEstimatedChargeRemaining gets the value of EstimatedChargeRemaining for the instance
func (instance *CIM_UninterruptiblePowerSupply) GetPropertyEstimatedChargeRemaining() (value uint16, err error) {
	retValue, err := instance.GetProperty("EstimatedChargeRemaining")
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

// SetEstimatedRunTime sets the value of EstimatedRunTime for the instance
func (instance *CIM_UninterruptiblePowerSupply) SetPropertyEstimatedRunTime(value uint32) (err error) {
	return instance.SetProperty("EstimatedRunTime", (value))
}

// GetEstimatedRunTime gets the value of EstimatedRunTime for the instance
func (instance *CIM_UninterruptiblePowerSupply) GetPropertyEstimatedRunTime() (value uint32, err error) {
	retValue, err := instance.GetProperty("EstimatedRunTime")
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

// SetRemainingCapacityStatus sets the value of RemainingCapacityStatus for the instance
func (instance *CIM_UninterruptiblePowerSupply) SetPropertyRemainingCapacityStatus(value uint16) (err error) {
	return instance.SetProperty("RemainingCapacityStatus", (value))
}

// GetRemainingCapacityStatus gets the value of RemainingCapacityStatus for the instance
func (instance *CIM_UninterruptiblePowerSupply) GetPropertyRemainingCapacityStatus() (value uint16, err error) {
	retValue, err := instance.GetProperty("RemainingCapacityStatus")
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

// SetTimeOnBackup sets the value of TimeOnBackup for the instance
func (instance *CIM_UninterruptiblePowerSupply) SetPropertyTimeOnBackup(value uint32) (err error) {
	return instance.SetProperty("TimeOnBackup", (value))
}

// GetTimeOnBackup gets the value of TimeOnBackup for the instance
func (instance *CIM_UninterruptiblePowerSupply) GetPropertyTimeOnBackup() (value uint32, err error) {
	retValue, err := instance.GetProperty("TimeOnBackup")
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
