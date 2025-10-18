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

// CIM_PhysicalLink struct
type CIM_PhysicalLink struct {
	*CIM_PhysicalElement

	//
	Length float64

	//
	MaxLength float64

	//
	MediaType uint16

	//
	Wired bool
}

func NewCIM_PhysicalLinkEx1(instance *cim.WmiInstance) (newInstance *CIM_PhysicalLink, err error) {
	tmp, err := NewCIM_PhysicalElementEx1(instance)

	if err != nil {
		return
	}
	newInstance = &CIM_PhysicalLink{
		CIM_PhysicalElement: tmp,
	}
	return
}

func NewCIM_PhysicalLinkEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *CIM_PhysicalLink, err error) {
	tmp, err := NewCIM_PhysicalElementEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &CIM_PhysicalLink{
		CIM_PhysicalElement: tmp,
	}
	return
}

// SetLength sets the value of Length for the instance
func (instance *CIM_PhysicalLink) SetPropertyLength(value float64) (err error) {
	return instance.SetProperty("Length", (value))
}

// GetLength gets the value of Length for the instance
func (instance *CIM_PhysicalLink) GetPropertyLength() (value float64, err error) {
	retValue, err := instance.GetProperty("Length")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(float64)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " float64 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = float64(valuetmp)

	return
}

// SetMaxLength sets the value of MaxLength for the instance
func (instance *CIM_PhysicalLink) SetPropertyMaxLength(value float64) (err error) {
	return instance.SetProperty("MaxLength", (value))
}

// GetMaxLength gets the value of MaxLength for the instance
func (instance *CIM_PhysicalLink) GetPropertyMaxLength() (value float64, err error) {
	retValue, err := instance.GetProperty("MaxLength")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(float64)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " float64 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = float64(valuetmp)

	return
}

// SetMediaType sets the value of MediaType for the instance
func (instance *CIM_PhysicalLink) SetPropertyMediaType(value uint16) (err error) {
	return instance.SetProperty("MediaType", (value))
}

// GetMediaType gets the value of MediaType for the instance
func (instance *CIM_PhysicalLink) GetPropertyMediaType() (value uint16, err error) {
	retValue, err := instance.GetProperty("MediaType")
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

// SetWired sets the value of Wired for the instance
func (instance *CIM_PhysicalLink) SetPropertyWired(value bool) (err error) {
	return instance.SetProperty("Wired", (value))
}

// GetWired gets the value of Wired for the instance
func (instance *CIM_PhysicalLink) GetPropertyWired() (value bool, err error) {
	retValue, err := instance.GetProperty("Wired")
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
