// Copyright 2019 (c) Microsoft Corporation.
// Licensed under the MIT license.

//
// Author:
//      Auto Generated on 9/18/2020 using wmigen
//      Source root.CIMV2
//////////////////////////////////////////////
package cimv2

import (
	"github.com/microsoft/wmi/pkg/base/instance"
	"github.com/microsoft/wmi/pkg/base/query"
	"github.com/microsoft/wmi/pkg/errors"
	cim "github.com/microsoft/wmi/pkg/wmiinstance"
	"reflect"
)

// CIM_AdjacentSlots struct
type CIM_AdjacentSlots struct {
	*cim.WmiInstance

	//
	DistanceBetweenSlots float32

	//
	SharedSlots bool

	//
	SlotA CIM_Slot

	//
	SlotB CIM_Slot
}

func NewCIM_AdjacentSlotsEx1(instance *cim.WmiInstance) (newInstance *CIM_AdjacentSlots, err error) {
	tmp, err := instance, nil

	if err != nil {
		return
	}
	newInstance = &CIM_AdjacentSlots{
		WmiInstance: tmp,
	}
	return
}

func NewCIM_AdjacentSlotsEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *CIM_AdjacentSlots, err error) {
	tmp, err := instance.GetWmiInstance(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &CIM_AdjacentSlots{
		WmiInstance: tmp,
	}
	return
}

// SetDistanceBetweenSlots sets the value of DistanceBetweenSlots for the instance
func (instance *CIM_AdjacentSlots) SetPropertyDistanceBetweenSlots(value float32) (err error) {
	return instance.SetProperty("DistanceBetweenSlots", (value))
}

// GetDistanceBetweenSlots gets the value of DistanceBetweenSlots for the instance
func (instance *CIM_AdjacentSlots) GetPropertyDistanceBetweenSlots() (value float32, err error) {
	retValue, err := instance.GetProperty("DistanceBetweenSlots")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(float32)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " float32 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = float32(valuetmp)

	return
}

// SetSharedSlots sets the value of SharedSlots for the instance
func (instance *CIM_AdjacentSlots) SetPropertySharedSlots(value bool) (err error) {
	return instance.SetProperty("SharedSlots", (value))
}

// GetSharedSlots gets the value of SharedSlots for the instance
func (instance *CIM_AdjacentSlots) GetPropertySharedSlots() (value bool, err error) {
	retValue, err := instance.GetProperty("SharedSlots")
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

// SetSlotA sets the value of SlotA for the instance
func (instance *CIM_AdjacentSlots) SetPropertySlotA(value CIM_Slot) (err error) {
	return instance.SetProperty("SlotA", (value))
}

// GetSlotA gets the value of SlotA for the instance
func (instance *CIM_AdjacentSlots) GetPropertySlotA() (value CIM_Slot, err error) {
	retValue, err := instance.GetProperty("SlotA")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(CIM_Slot)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " CIM_Slot is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = CIM_Slot(valuetmp)

	return
}

// SetSlotB sets the value of SlotB for the instance
func (instance *CIM_AdjacentSlots) SetPropertySlotB(value CIM_Slot) (err error) {
	return instance.SetProperty("SlotB", (value))
}

// GetSlotB gets the value of SlotB for the instance
func (instance *CIM_AdjacentSlots) GetPropertySlotB() (value CIM_Slot, err error) {
	retValue, err := instance.GetProperty("SlotB")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(CIM_Slot)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " CIM_Slot is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = CIM_Slot(valuetmp)

	return
}
