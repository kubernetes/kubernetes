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

// CIM_FlatPanel struct
type CIM_FlatPanel struct {
	*CIM_Display

	//
	DisplayType uint16

	//
	HorizontalResolution uint32

	//
	LightSource uint16

	//
	ScanMode uint16

	//
	SupportsColor bool

	//
	VerticalResolution uint32
}

func NewCIM_FlatPanelEx1(instance *cim.WmiInstance) (newInstance *CIM_FlatPanel, err error) {
	tmp, err := NewCIM_DisplayEx1(instance)

	if err != nil {
		return
	}
	newInstance = &CIM_FlatPanel{
		CIM_Display: tmp,
	}
	return
}

func NewCIM_FlatPanelEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *CIM_FlatPanel, err error) {
	tmp, err := NewCIM_DisplayEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &CIM_FlatPanel{
		CIM_Display: tmp,
	}
	return
}

// SetDisplayType sets the value of DisplayType for the instance
func (instance *CIM_FlatPanel) SetPropertyDisplayType(value uint16) (err error) {
	return instance.SetProperty("DisplayType", (value))
}

// GetDisplayType gets the value of DisplayType for the instance
func (instance *CIM_FlatPanel) GetPropertyDisplayType() (value uint16, err error) {
	retValue, err := instance.GetProperty("DisplayType")
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

// SetHorizontalResolution sets the value of HorizontalResolution for the instance
func (instance *CIM_FlatPanel) SetPropertyHorizontalResolution(value uint32) (err error) {
	return instance.SetProperty("HorizontalResolution", (value))
}

// GetHorizontalResolution gets the value of HorizontalResolution for the instance
func (instance *CIM_FlatPanel) GetPropertyHorizontalResolution() (value uint32, err error) {
	retValue, err := instance.GetProperty("HorizontalResolution")
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

// SetLightSource sets the value of LightSource for the instance
func (instance *CIM_FlatPanel) SetPropertyLightSource(value uint16) (err error) {
	return instance.SetProperty("LightSource", (value))
}

// GetLightSource gets the value of LightSource for the instance
func (instance *CIM_FlatPanel) GetPropertyLightSource() (value uint16, err error) {
	retValue, err := instance.GetProperty("LightSource")
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

// SetScanMode sets the value of ScanMode for the instance
func (instance *CIM_FlatPanel) SetPropertyScanMode(value uint16) (err error) {
	return instance.SetProperty("ScanMode", (value))
}

// GetScanMode gets the value of ScanMode for the instance
func (instance *CIM_FlatPanel) GetPropertyScanMode() (value uint16, err error) {
	retValue, err := instance.GetProperty("ScanMode")
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

// SetSupportsColor sets the value of SupportsColor for the instance
func (instance *CIM_FlatPanel) SetPropertySupportsColor(value bool) (err error) {
	return instance.SetProperty("SupportsColor", (value))
}

// GetSupportsColor gets the value of SupportsColor for the instance
func (instance *CIM_FlatPanel) GetPropertySupportsColor() (value bool, err error) {
	retValue, err := instance.GetProperty("SupportsColor")
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

// SetVerticalResolution sets the value of VerticalResolution for the instance
func (instance *CIM_FlatPanel) SetPropertyVerticalResolution(value uint32) (err error) {
	return instance.SetProperty("VerticalResolution", (value))
}

// GetVerticalResolution gets the value of VerticalResolution for the instance
func (instance *CIM_FlatPanel) GetPropertyVerticalResolution() (value uint32, err error) {
	retValue, err := instance.GetProperty("VerticalResolution")
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
