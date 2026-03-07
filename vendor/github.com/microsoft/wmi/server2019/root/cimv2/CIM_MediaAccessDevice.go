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

// CIM_MediaAccessDevice struct
type CIM_MediaAccessDevice struct {
	*CIM_LogicalDevice

	//
	Capabilities []uint16

	//
	CapabilityDescriptions []string

	//
	CompressionMethod string

	//
	DefaultBlockSize uint64

	//
	ErrorMethodology string

	//
	MaxBlockSize uint64

	//
	MaxMediaSize uint64

	//
	MinBlockSize uint64

	//
	NeedsCleaning bool

	//
	NumberOfMediaSupported uint32
}

func NewCIM_MediaAccessDeviceEx1(instance *cim.WmiInstance) (newInstance *CIM_MediaAccessDevice, err error) {
	tmp, err := NewCIM_LogicalDeviceEx1(instance)

	if err != nil {
		return
	}
	newInstance = &CIM_MediaAccessDevice{
		CIM_LogicalDevice: tmp,
	}
	return
}

func NewCIM_MediaAccessDeviceEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *CIM_MediaAccessDevice, err error) {
	tmp, err := NewCIM_LogicalDeviceEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &CIM_MediaAccessDevice{
		CIM_LogicalDevice: tmp,
	}
	return
}

// SetCapabilities sets the value of Capabilities for the instance
func (instance *CIM_MediaAccessDevice) SetPropertyCapabilities(value []uint16) (err error) {
	return instance.SetProperty("Capabilities", (value))
}

// GetCapabilities gets the value of Capabilities for the instance
func (instance *CIM_MediaAccessDevice) GetPropertyCapabilities() (value []uint16, err error) {
	retValue, err := instance.GetProperty("Capabilities")
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

// SetCapabilityDescriptions sets the value of CapabilityDescriptions for the instance
func (instance *CIM_MediaAccessDevice) SetPropertyCapabilityDescriptions(value []string) (err error) {
	return instance.SetProperty("CapabilityDescriptions", (value))
}

// GetCapabilityDescriptions gets the value of CapabilityDescriptions for the instance
func (instance *CIM_MediaAccessDevice) GetPropertyCapabilityDescriptions() (value []string, err error) {
	retValue, err := instance.GetProperty("CapabilityDescriptions")
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

// SetCompressionMethod sets the value of CompressionMethod for the instance
func (instance *CIM_MediaAccessDevice) SetPropertyCompressionMethod(value string) (err error) {
	return instance.SetProperty("CompressionMethod", (value))
}

// GetCompressionMethod gets the value of CompressionMethod for the instance
func (instance *CIM_MediaAccessDevice) GetPropertyCompressionMethod() (value string, err error) {
	retValue, err := instance.GetProperty("CompressionMethod")
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

// SetDefaultBlockSize sets the value of DefaultBlockSize for the instance
func (instance *CIM_MediaAccessDevice) SetPropertyDefaultBlockSize(value uint64) (err error) {
	return instance.SetProperty("DefaultBlockSize", (value))
}

// GetDefaultBlockSize gets the value of DefaultBlockSize for the instance
func (instance *CIM_MediaAccessDevice) GetPropertyDefaultBlockSize() (value uint64, err error) {
	retValue, err := instance.GetProperty("DefaultBlockSize")
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
func (instance *CIM_MediaAccessDevice) SetPropertyErrorMethodology(value string) (err error) {
	return instance.SetProperty("ErrorMethodology", (value))
}

// GetErrorMethodology gets the value of ErrorMethodology for the instance
func (instance *CIM_MediaAccessDevice) GetPropertyErrorMethodology() (value string, err error) {
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

// SetMaxBlockSize sets the value of MaxBlockSize for the instance
func (instance *CIM_MediaAccessDevice) SetPropertyMaxBlockSize(value uint64) (err error) {
	return instance.SetProperty("MaxBlockSize", (value))
}

// GetMaxBlockSize gets the value of MaxBlockSize for the instance
func (instance *CIM_MediaAccessDevice) GetPropertyMaxBlockSize() (value uint64, err error) {
	retValue, err := instance.GetProperty("MaxBlockSize")
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

// SetMaxMediaSize sets the value of MaxMediaSize for the instance
func (instance *CIM_MediaAccessDevice) SetPropertyMaxMediaSize(value uint64) (err error) {
	return instance.SetProperty("MaxMediaSize", (value))
}

// GetMaxMediaSize gets the value of MaxMediaSize for the instance
func (instance *CIM_MediaAccessDevice) GetPropertyMaxMediaSize() (value uint64, err error) {
	retValue, err := instance.GetProperty("MaxMediaSize")
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

// SetMinBlockSize sets the value of MinBlockSize for the instance
func (instance *CIM_MediaAccessDevice) SetPropertyMinBlockSize(value uint64) (err error) {
	return instance.SetProperty("MinBlockSize", (value))
}

// GetMinBlockSize gets the value of MinBlockSize for the instance
func (instance *CIM_MediaAccessDevice) GetPropertyMinBlockSize() (value uint64, err error) {
	retValue, err := instance.GetProperty("MinBlockSize")
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

// SetNeedsCleaning sets the value of NeedsCleaning for the instance
func (instance *CIM_MediaAccessDevice) SetPropertyNeedsCleaning(value bool) (err error) {
	return instance.SetProperty("NeedsCleaning", (value))
}

// GetNeedsCleaning gets the value of NeedsCleaning for the instance
func (instance *CIM_MediaAccessDevice) GetPropertyNeedsCleaning() (value bool, err error) {
	retValue, err := instance.GetProperty("NeedsCleaning")
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

// SetNumberOfMediaSupported sets the value of NumberOfMediaSupported for the instance
func (instance *CIM_MediaAccessDevice) SetPropertyNumberOfMediaSupported(value uint32) (err error) {
	return instance.SetProperty("NumberOfMediaSupported", (value))
}

// GetNumberOfMediaSupported gets the value of NumberOfMediaSupported for the instance
func (instance *CIM_MediaAccessDevice) GetPropertyNumberOfMediaSupported() (value uint32, err error) {
	retValue, err := instance.GetProperty("NumberOfMediaSupported")
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
