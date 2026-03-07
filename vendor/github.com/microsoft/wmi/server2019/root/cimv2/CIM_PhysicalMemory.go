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

// CIM_PhysicalMemory struct
type CIM_PhysicalMemory struct {
	*CIM_Chip

	//
	BankLabel string

	//
	Capacity uint64

	//
	DataWidth uint16

	//
	InterleavePosition uint32

	//
	MemoryType uint16

	//
	PositionInRow uint32

	//
	Speed uint32

	//
	TotalWidth uint16
}

func NewCIM_PhysicalMemoryEx1(instance *cim.WmiInstance) (newInstance *CIM_PhysicalMemory, err error) {
	tmp, err := NewCIM_ChipEx1(instance)

	if err != nil {
		return
	}
	newInstance = &CIM_PhysicalMemory{
		CIM_Chip: tmp,
	}
	return
}

func NewCIM_PhysicalMemoryEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *CIM_PhysicalMemory, err error) {
	tmp, err := NewCIM_ChipEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &CIM_PhysicalMemory{
		CIM_Chip: tmp,
	}
	return
}

// SetBankLabel sets the value of BankLabel for the instance
func (instance *CIM_PhysicalMemory) SetPropertyBankLabel(value string) (err error) {
	return instance.SetProperty("BankLabel", (value))
}

// GetBankLabel gets the value of BankLabel for the instance
func (instance *CIM_PhysicalMemory) GetPropertyBankLabel() (value string, err error) {
	retValue, err := instance.GetProperty("BankLabel")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(string)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " string is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = string(valuetmp)

	return
}

// SetCapacity sets the value of Capacity for the instance
func (instance *CIM_PhysicalMemory) SetPropertyCapacity(value uint64) (err error) {
	return instance.SetProperty("Capacity", (value))
}

// GetCapacity gets the value of Capacity for the instance
func (instance *CIM_PhysicalMemory) GetPropertyCapacity() (value uint64, err error) {
	retValue, err := instance.GetProperty("Capacity")
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

// SetDataWidth sets the value of DataWidth for the instance
func (instance *CIM_PhysicalMemory) SetPropertyDataWidth(value uint16) (err error) {
	return instance.SetProperty("DataWidth", (value))
}

// GetDataWidth gets the value of DataWidth for the instance
func (instance *CIM_PhysicalMemory) GetPropertyDataWidth() (value uint16, err error) {
	retValue, err := instance.GetProperty("DataWidth")
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

// SetInterleavePosition sets the value of InterleavePosition for the instance
func (instance *CIM_PhysicalMemory) SetPropertyInterleavePosition(value uint32) (err error) {
	return instance.SetProperty("InterleavePosition", (value))
}

// GetInterleavePosition gets the value of InterleavePosition for the instance
func (instance *CIM_PhysicalMemory) GetPropertyInterleavePosition() (value uint32, err error) {
	retValue, err := instance.GetProperty("InterleavePosition")
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

// SetMemoryType sets the value of MemoryType for the instance
func (instance *CIM_PhysicalMemory) SetPropertyMemoryType(value uint16) (err error) {
	return instance.SetProperty("MemoryType", (value))
}

// GetMemoryType gets the value of MemoryType for the instance
func (instance *CIM_PhysicalMemory) GetPropertyMemoryType() (value uint16, err error) {
	retValue, err := instance.GetProperty("MemoryType")
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

// SetPositionInRow sets the value of PositionInRow for the instance
func (instance *CIM_PhysicalMemory) SetPropertyPositionInRow(value uint32) (err error) {
	return instance.SetProperty("PositionInRow", (value))
}

// GetPositionInRow gets the value of PositionInRow for the instance
func (instance *CIM_PhysicalMemory) GetPropertyPositionInRow() (value uint32, err error) {
	retValue, err := instance.GetProperty("PositionInRow")
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

// SetSpeed sets the value of Speed for the instance
func (instance *CIM_PhysicalMemory) SetPropertySpeed(value uint32) (err error) {
	return instance.SetProperty("Speed", (value))
}

// GetSpeed gets the value of Speed for the instance
func (instance *CIM_PhysicalMemory) GetPropertySpeed() (value uint32, err error) {
	retValue, err := instance.GetProperty("Speed")
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

// SetTotalWidth sets the value of TotalWidth for the instance
func (instance *CIM_PhysicalMemory) SetPropertyTotalWidth(value uint16) (err error) {
	return instance.SetProperty("TotalWidth", (value))
}

// GetTotalWidth gets the value of TotalWidth for the instance
func (instance *CIM_PhysicalMemory) GetPropertyTotalWidth() (value uint16, err error) {
	retValue, err := instance.GetProperty("TotalWidth")
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
