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

// CIM_DMA struct
type CIM_DMA struct {
	*CIM_SystemResource

	//
	AddressSize uint16

	//
	Availability uint16

	//
	BurstMode bool

	//
	ByteMode uint16

	//
	ChannelTiming uint16

	//
	CreationClassName string

	//
	CSCreationClassName string

	//
	CSName string

	//
	DMAChannel uint32

	//
	MaxTransferSize uint32

	//
	TransferWidths []uint16

	//
	TypeCTiming uint16

	//
	WordMode uint16
}

func NewCIM_DMAEx1(instance *cim.WmiInstance) (newInstance *CIM_DMA, err error) {
	tmp, err := NewCIM_SystemResourceEx1(instance)

	if err != nil {
		return
	}
	newInstance = &CIM_DMA{
		CIM_SystemResource: tmp,
	}
	return
}

func NewCIM_DMAEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *CIM_DMA, err error) {
	tmp, err := NewCIM_SystemResourceEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &CIM_DMA{
		CIM_SystemResource: tmp,
	}
	return
}

// SetAddressSize sets the value of AddressSize for the instance
func (instance *CIM_DMA) SetPropertyAddressSize(value uint16) (err error) {
	return instance.SetProperty("AddressSize", (value))
}

// GetAddressSize gets the value of AddressSize for the instance
func (instance *CIM_DMA) GetPropertyAddressSize() (value uint16, err error) {
	retValue, err := instance.GetProperty("AddressSize")
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

// SetAvailability sets the value of Availability for the instance
func (instance *CIM_DMA) SetPropertyAvailability(value uint16) (err error) {
	return instance.SetProperty("Availability", (value))
}

// GetAvailability gets the value of Availability for the instance
func (instance *CIM_DMA) GetPropertyAvailability() (value uint16, err error) {
	retValue, err := instance.GetProperty("Availability")
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

// SetBurstMode sets the value of BurstMode for the instance
func (instance *CIM_DMA) SetPropertyBurstMode(value bool) (err error) {
	return instance.SetProperty("BurstMode", (value))
}

// GetBurstMode gets the value of BurstMode for the instance
func (instance *CIM_DMA) GetPropertyBurstMode() (value bool, err error) {
	retValue, err := instance.GetProperty("BurstMode")
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

// SetByteMode sets the value of ByteMode for the instance
func (instance *CIM_DMA) SetPropertyByteMode(value uint16) (err error) {
	return instance.SetProperty("ByteMode", (value))
}

// GetByteMode gets the value of ByteMode for the instance
func (instance *CIM_DMA) GetPropertyByteMode() (value uint16, err error) {
	retValue, err := instance.GetProperty("ByteMode")
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

// SetChannelTiming sets the value of ChannelTiming for the instance
func (instance *CIM_DMA) SetPropertyChannelTiming(value uint16) (err error) {
	return instance.SetProperty("ChannelTiming", (value))
}

// GetChannelTiming gets the value of ChannelTiming for the instance
func (instance *CIM_DMA) GetPropertyChannelTiming() (value uint16, err error) {
	retValue, err := instance.GetProperty("ChannelTiming")
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

// SetCreationClassName sets the value of CreationClassName for the instance
func (instance *CIM_DMA) SetPropertyCreationClassName(value string) (err error) {
	return instance.SetProperty("CreationClassName", (value))
}

// GetCreationClassName gets the value of CreationClassName for the instance
func (instance *CIM_DMA) GetPropertyCreationClassName() (value string, err error) {
	retValue, err := instance.GetProperty("CreationClassName")
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

// SetCSCreationClassName sets the value of CSCreationClassName for the instance
func (instance *CIM_DMA) SetPropertyCSCreationClassName(value string) (err error) {
	return instance.SetProperty("CSCreationClassName", (value))
}

// GetCSCreationClassName gets the value of CSCreationClassName for the instance
func (instance *CIM_DMA) GetPropertyCSCreationClassName() (value string, err error) {
	retValue, err := instance.GetProperty("CSCreationClassName")
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

// SetCSName sets the value of CSName for the instance
func (instance *CIM_DMA) SetPropertyCSName(value string) (err error) {
	return instance.SetProperty("CSName", (value))
}

// GetCSName gets the value of CSName for the instance
func (instance *CIM_DMA) GetPropertyCSName() (value string, err error) {
	retValue, err := instance.GetProperty("CSName")
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

// SetDMAChannel sets the value of DMAChannel for the instance
func (instance *CIM_DMA) SetPropertyDMAChannel(value uint32) (err error) {
	return instance.SetProperty("DMAChannel", (value))
}

// GetDMAChannel gets the value of DMAChannel for the instance
func (instance *CIM_DMA) GetPropertyDMAChannel() (value uint32, err error) {
	retValue, err := instance.GetProperty("DMAChannel")
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

// SetMaxTransferSize sets the value of MaxTransferSize for the instance
func (instance *CIM_DMA) SetPropertyMaxTransferSize(value uint32) (err error) {
	return instance.SetProperty("MaxTransferSize", (value))
}

// GetMaxTransferSize gets the value of MaxTransferSize for the instance
func (instance *CIM_DMA) GetPropertyMaxTransferSize() (value uint32, err error) {
	retValue, err := instance.GetProperty("MaxTransferSize")
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

// SetTransferWidths sets the value of TransferWidths for the instance
func (instance *CIM_DMA) SetPropertyTransferWidths(value []uint16) (err error) {
	return instance.SetProperty("TransferWidths", (value))
}

// GetTransferWidths gets the value of TransferWidths for the instance
func (instance *CIM_DMA) GetPropertyTransferWidths() (value []uint16, err error) {
	retValue, err := instance.GetProperty("TransferWidths")
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

// SetTypeCTiming sets the value of TypeCTiming for the instance
func (instance *CIM_DMA) SetPropertyTypeCTiming(value uint16) (err error) {
	return instance.SetProperty("TypeCTiming", (value))
}

// GetTypeCTiming gets the value of TypeCTiming for the instance
func (instance *CIM_DMA) GetPropertyTypeCTiming() (value uint16, err error) {
	retValue, err := instance.GetProperty("TypeCTiming")
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

// SetWordMode sets the value of WordMode for the instance
func (instance *CIM_DMA) SetPropertyWordMode(value uint16) (err error) {
	return instance.SetProperty("WordMode", (value))
}

// GetWordMode gets the value of WordMode for the instance
func (instance *CIM_DMA) GetPropertyWordMode() (value uint16, err error) {
	retValue, err := instance.GetProperty("WordMode")
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
