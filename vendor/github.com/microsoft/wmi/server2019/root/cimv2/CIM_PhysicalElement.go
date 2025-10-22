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

// CIM_PhysicalElement struct
type CIM_PhysicalElement struct {
	*CIM_ManagedSystemElement

	//
	CreationClassName string

	//
	Manufacturer string

	//
	Model string

	//
	OtherIdentifyingInfo string

	//
	PartNumber string

	//
	PoweredOn bool

	//
	SerialNumber string

	//
	SKU string

	//
	Tag string

	//
	Version string
}

func NewCIM_PhysicalElementEx1(instance *cim.WmiInstance) (newInstance *CIM_PhysicalElement, err error) {
	tmp, err := NewCIM_ManagedSystemElementEx1(instance)

	if err != nil {
		return
	}
	newInstance = &CIM_PhysicalElement{
		CIM_ManagedSystemElement: tmp,
	}
	return
}

func NewCIM_PhysicalElementEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *CIM_PhysicalElement, err error) {
	tmp, err := NewCIM_ManagedSystemElementEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &CIM_PhysicalElement{
		CIM_ManagedSystemElement: tmp,
	}
	return
}

// SetCreationClassName sets the value of CreationClassName for the instance
func (instance *CIM_PhysicalElement) SetPropertyCreationClassName(value string) (err error) {
	return instance.SetProperty("CreationClassName", (value))
}

// GetCreationClassName gets the value of CreationClassName for the instance
func (instance *CIM_PhysicalElement) GetPropertyCreationClassName() (value string, err error) {
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

// SetManufacturer sets the value of Manufacturer for the instance
func (instance *CIM_PhysicalElement) SetPropertyManufacturer(value string) (err error) {
	return instance.SetProperty("Manufacturer", (value))
}

// GetManufacturer gets the value of Manufacturer for the instance
func (instance *CIM_PhysicalElement) GetPropertyManufacturer() (value string, err error) {
	retValue, err := instance.GetProperty("Manufacturer")
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

// SetModel sets the value of Model for the instance
func (instance *CIM_PhysicalElement) SetPropertyModel(value string) (err error) {
	return instance.SetProperty("Model", (value))
}

// GetModel gets the value of Model for the instance
func (instance *CIM_PhysicalElement) GetPropertyModel() (value string, err error) {
	retValue, err := instance.GetProperty("Model")
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

// SetOtherIdentifyingInfo sets the value of OtherIdentifyingInfo for the instance
func (instance *CIM_PhysicalElement) SetPropertyOtherIdentifyingInfo(value string) (err error) {
	return instance.SetProperty("OtherIdentifyingInfo", (value))
}

// GetOtherIdentifyingInfo gets the value of OtherIdentifyingInfo for the instance
func (instance *CIM_PhysicalElement) GetPropertyOtherIdentifyingInfo() (value string, err error) {
	retValue, err := instance.GetProperty("OtherIdentifyingInfo")
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

// SetPartNumber sets the value of PartNumber for the instance
func (instance *CIM_PhysicalElement) SetPropertyPartNumber(value string) (err error) {
	return instance.SetProperty("PartNumber", (value))
}

// GetPartNumber gets the value of PartNumber for the instance
func (instance *CIM_PhysicalElement) GetPropertyPartNumber() (value string, err error) {
	retValue, err := instance.GetProperty("PartNumber")
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

// SetPoweredOn sets the value of PoweredOn for the instance
func (instance *CIM_PhysicalElement) SetPropertyPoweredOn(value bool) (err error) {
	return instance.SetProperty("PoweredOn", (value))
}

// GetPoweredOn gets the value of PoweredOn for the instance
func (instance *CIM_PhysicalElement) GetPropertyPoweredOn() (value bool, err error) {
	retValue, err := instance.GetProperty("PoweredOn")
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

// SetSerialNumber sets the value of SerialNumber for the instance
func (instance *CIM_PhysicalElement) SetPropertySerialNumber(value string) (err error) {
	return instance.SetProperty("SerialNumber", (value))
}

// GetSerialNumber gets the value of SerialNumber for the instance
func (instance *CIM_PhysicalElement) GetPropertySerialNumber() (value string, err error) {
	retValue, err := instance.GetProperty("SerialNumber")
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

// SetSKU sets the value of SKU for the instance
func (instance *CIM_PhysicalElement) SetPropertySKU(value string) (err error) {
	return instance.SetProperty("SKU", (value))
}

// GetSKU gets the value of SKU for the instance
func (instance *CIM_PhysicalElement) GetPropertySKU() (value string, err error) {
	retValue, err := instance.GetProperty("SKU")
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

// SetTag sets the value of Tag for the instance
func (instance *CIM_PhysicalElement) SetPropertyTag(value string) (err error) {
	return instance.SetProperty("Tag", (value))
}

// GetTag gets the value of Tag for the instance
func (instance *CIM_PhysicalElement) GetPropertyTag() (value string, err error) {
	retValue, err := instance.GetProperty("Tag")
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

// SetVersion sets the value of Version for the instance
func (instance *CIM_PhysicalElement) SetPropertyVersion(value string) (err error) {
	return instance.SetProperty("Version", (value))
}

// GetVersion gets the value of Version for the instance
func (instance *CIM_PhysicalElement) GetPropertyVersion() (value string, err error) {
	retValue, err := instance.GetProperty("Version")
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
