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

// CIM_StorageExtent struct
type CIM_StorageExtent struct {
	*CIM_LogicalDevice

	//
	Access uint16

	//
	BlockSize uint64

	//
	ErrorMethodology string

	//
	NumberOfBlocks uint64

	//
	Purpose string
}

func NewCIM_StorageExtentEx1(instance *cim.WmiInstance) (newInstance *CIM_StorageExtent, err error) {
	tmp, err := NewCIM_LogicalDeviceEx1(instance)

	if err != nil {
		return
	}
	newInstance = &CIM_StorageExtent{
		CIM_LogicalDevice: tmp,
	}
	return
}

func NewCIM_StorageExtentEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *CIM_StorageExtent, err error) {
	tmp, err := NewCIM_LogicalDeviceEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &CIM_StorageExtent{
		CIM_LogicalDevice: tmp,
	}
	return
}

// SetAccess sets the value of Access for the instance
func (instance *CIM_StorageExtent) SetPropertyAccess(value uint16) (err error) {
	return instance.SetProperty("Access", (value))
}

// GetAccess gets the value of Access for the instance
func (instance *CIM_StorageExtent) GetPropertyAccess() (value uint16, err error) {
	retValue, err := instance.GetProperty("Access")
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

// SetBlockSize sets the value of BlockSize for the instance
func (instance *CIM_StorageExtent) SetPropertyBlockSize(value uint64) (err error) {
	return instance.SetProperty("BlockSize", (value))
}

// GetBlockSize gets the value of BlockSize for the instance
func (instance *CIM_StorageExtent) GetPropertyBlockSize() (value uint64, err error) {
	retValue, err := instance.GetProperty("BlockSize")
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

// SetErrorMethodology sets the value of ErrorMethodology for the instance
func (instance *CIM_StorageExtent) SetPropertyErrorMethodology(value string) (err error) {
	return instance.SetProperty("ErrorMethodology", (value))
}

// GetErrorMethodology gets the value of ErrorMethodology for the instance
func (instance *CIM_StorageExtent) GetPropertyErrorMethodology() (value string, err error) {
	retValue, err := instance.GetProperty("ErrorMethodology")
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

// SetNumberOfBlocks sets the value of NumberOfBlocks for the instance
func (instance *CIM_StorageExtent) SetPropertyNumberOfBlocks(value uint64) (err error) {
	return instance.SetProperty("NumberOfBlocks", (value))
}

// GetNumberOfBlocks gets the value of NumberOfBlocks for the instance
func (instance *CIM_StorageExtent) GetPropertyNumberOfBlocks() (value uint64, err error) {
	retValue, err := instance.GetProperty("NumberOfBlocks")
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

// SetPurpose sets the value of Purpose for the instance
func (instance *CIM_StorageExtent) SetPropertyPurpose(value string) (err error) {
	return instance.SetProperty("Purpose", (value))
}

// GetPurpose gets the value of Purpose for the instance
func (instance *CIM_StorageExtent) GetPropertyPurpose() (value string, err error) {
	retValue, err := instance.GetProperty("Purpose")
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
