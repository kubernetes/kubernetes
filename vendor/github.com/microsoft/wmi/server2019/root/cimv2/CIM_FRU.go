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

// CIM_FRU struct
type CIM_FRU struct {
	*cim.WmiInstance

	//
	Caption string

	//
	Description string

	//
	FRUNumber string

	//
	IdentifyingNumber string

	//
	Name string

	//
	RevisionLevel string

	//
	Vendor string
}

func NewCIM_FRUEx1(instance *cim.WmiInstance) (newInstance *CIM_FRU, err error) {
	tmp, err := instance, nil

	if err != nil {
		return
	}
	newInstance = &CIM_FRU{
		WmiInstance: tmp,
	}
	return
}

func NewCIM_FRUEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *CIM_FRU, err error) {
	tmp, err := instance.GetWmiInstance(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &CIM_FRU{
		WmiInstance: tmp,
	}
	return
}

// SetCaption sets the value of Caption for the instance
func (instance *CIM_FRU) SetPropertyCaption(value string) (err error) {
	return instance.SetProperty("Caption", (value))
}

// GetCaption gets the value of Caption for the instance
func (instance *CIM_FRU) GetPropertyCaption() (value string, err error) {
	retValue, err := instance.GetProperty("Caption")
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

// SetDescription sets the value of Description for the instance
func (instance *CIM_FRU) SetPropertyDescription(value string) (err error) {
	return instance.SetProperty("Description", (value))
}

// GetDescription gets the value of Description for the instance
func (instance *CIM_FRU) GetPropertyDescription() (value string, err error) {
	retValue, err := instance.GetProperty("Description")
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

// SetFRUNumber sets the value of FRUNumber for the instance
func (instance *CIM_FRU) SetPropertyFRUNumber(value string) (err error) {
	return instance.SetProperty("FRUNumber", (value))
}

// GetFRUNumber gets the value of FRUNumber for the instance
func (instance *CIM_FRU) GetPropertyFRUNumber() (value string, err error) {
	retValue, err := instance.GetProperty("FRUNumber")
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

// SetIdentifyingNumber sets the value of IdentifyingNumber for the instance
func (instance *CIM_FRU) SetPropertyIdentifyingNumber(value string) (err error) {
	return instance.SetProperty("IdentifyingNumber", (value))
}

// GetIdentifyingNumber gets the value of IdentifyingNumber for the instance
func (instance *CIM_FRU) GetPropertyIdentifyingNumber() (value string, err error) {
	retValue, err := instance.GetProperty("IdentifyingNumber")
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

// SetName sets the value of Name for the instance
func (instance *CIM_FRU) SetPropertyName(value string) (err error) {
	return instance.SetProperty("Name", (value))
}

// GetName gets the value of Name for the instance
func (instance *CIM_FRU) GetPropertyName() (value string, err error) {
	retValue, err := instance.GetProperty("Name")
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

// SetRevisionLevel sets the value of RevisionLevel for the instance
func (instance *CIM_FRU) SetPropertyRevisionLevel(value string) (err error) {
	return instance.SetProperty("RevisionLevel", (value))
}

// GetRevisionLevel gets the value of RevisionLevel for the instance
func (instance *CIM_FRU) GetPropertyRevisionLevel() (value string, err error) {
	retValue, err := instance.GetProperty("RevisionLevel")
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

// SetVendor sets the value of Vendor for the instance
func (instance *CIM_FRU) SetPropertyVendor(value string) (err error) {
	return instance.SetProperty("Vendor", (value))
}

// GetVendor gets the value of Vendor for the instance
func (instance *CIM_FRU) GetPropertyVendor() (value string, err error) {
	retValue, err := instance.GetProperty("Vendor")
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
