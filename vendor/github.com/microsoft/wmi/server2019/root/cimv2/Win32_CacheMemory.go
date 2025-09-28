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

// Win32_CacheMemory struct
type Win32_CacheMemory struct {
	*CIM_CacheMemory

	//
	CacheSpeed uint32

	//
	CurrentSRAM []uint16

	//
	ErrorCorrectType uint16

	//
	InstalledSize uint32

	//
	Location uint16

	//
	MaxCacheSize uint32

	//
	SupportedSRAM []uint16
}

func NewWin32_CacheMemoryEx1(instance *cim.WmiInstance) (newInstance *Win32_CacheMemory, err error) {
	tmp, err := NewCIM_CacheMemoryEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_CacheMemory{
		CIM_CacheMemory: tmp,
	}
	return
}

func NewWin32_CacheMemoryEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_CacheMemory, err error) {
	tmp, err := NewCIM_CacheMemoryEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_CacheMemory{
		CIM_CacheMemory: tmp,
	}
	return
}

// SetCacheSpeed sets the value of CacheSpeed for the instance
func (instance *Win32_CacheMemory) SetPropertyCacheSpeed(value uint32) (err error) {
	return instance.SetProperty("CacheSpeed", (value))
}

// GetCacheSpeed gets the value of CacheSpeed for the instance
func (instance *Win32_CacheMemory) GetPropertyCacheSpeed() (value uint32, err error) {
	retValue, err := instance.GetProperty("CacheSpeed")
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

// SetCurrentSRAM sets the value of CurrentSRAM for the instance
func (instance *Win32_CacheMemory) SetPropertyCurrentSRAM(value []uint16) (err error) {
	return instance.SetProperty("CurrentSRAM", (value))
}

// GetCurrentSRAM gets the value of CurrentSRAM for the instance
func (instance *Win32_CacheMemory) GetPropertyCurrentSRAM() (value []uint16, err error) {
	retValue, err := instance.GetProperty("CurrentSRAM")
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

// SetErrorCorrectType sets the value of ErrorCorrectType for the instance
func (instance *Win32_CacheMemory) SetPropertyErrorCorrectType(value uint16) (err error) {
	return instance.SetProperty("ErrorCorrectType", (value))
}

// GetErrorCorrectType gets the value of ErrorCorrectType for the instance
func (instance *Win32_CacheMemory) GetPropertyErrorCorrectType() (value uint16, err error) {
	retValue, err := instance.GetProperty("ErrorCorrectType")
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

// SetInstalledSize sets the value of InstalledSize for the instance
func (instance *Win32_CacheMemory) SetPropertyInstalledSize(value uint32) (err error) {
	return instance.SetProperty("InstalledSize", (value))
}

// GetInstalledSize gets the value of InstalledSize for the instance
func (instance *Win32_CacheMemory) GetPropertyInstalledSize() (value uint32, err error) {
	retValue, err := instance.GetProperty("InstalledSize")
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

// SetLocation sets the value of Location for the instance
func (instance *Win32_CacheMemory) SetPropertyLocation(value uint16) (err error) {
	return instance.SetProperty("Location", (value))
}

// GetLocation gets the value of Location for the instance
func (instance *Win32_CacheMemory) GetPropertyLocation() (value uint16, err error) {
	retValue, err := instance.GetProperty("Location")
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

// SetMaxCacheSize sets the value of MaxCacheSize for the instance
func (instance *Win32_CacheMemory) SetPropertyMaxCacheSize(value uint32) (err error) {
	return instance.SetProperty("MaxCacheSize", (value))
}

// GetMaxCacheSize gets the value of MaxCacheSize for the instance
func (instance *Win32_CacheMemory) GetPropertyMaxCacheSize() (value uint32, err error) {
	retValue, err := instance.GetProperty("MaxCacheSize")
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

// SetSupportedSRAM sets the value of SupportedSRAM for the instance
func (instance *Win32_CacheMemory) SetPropertySupportedSRAM(value []uint16) (err error) {
	return instance.SetProperty("SupportedSRAM", (value))
}

// GetSupportedSRAM gets the value of SupportedSRAM for the instance
func (instance *Win32_CacheMemory) GetPropertySupportedSRAM() (value []uint16, err error) {
	retValue, err := instance.GetProperty("SupportedSRAM")
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
