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

// CIM_VideoControllerResolution struct
type CIM_VideoControllerResolution struct {
	*CIM_Setting

	//
	HorizontalResolution uint32

	//
	MaxRefreshRate uint32

	//
	MinRefreshRate uint32

	//
	NumberOfColors uint64

	//
	RefreshRate uint32

	//
	ScanMode uint16

	//
	VerticalResolution uint32
}

func NewCIM_VideoControllerResolutionEx1(instance *cim.WmiInstance) (newInstance *CIM_VideoControllerResolution, err error) {
	tmp, err := NewCIM_SettingEx1(instance)

	if err != nil {
		return
	}
	newInstance = &CIM_VideoControllerResolution{
		CIM_Setting: tmp,
	}
	return
}

func NewCIM_VideoControllerResolutionEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *CIM_VideoControllerResolution, err error) {
	tmp, err := NewCIM_SettingEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &CIM_VideoControllerResolution{
		CIM_Setting: tmp,
	}
	return
}

// SetHorizontalResolution sets the value of HorizontalResolution for the instance
func (instance *CIM_VideoControllerResolution) SetPropertyHorizontalResolution(value uint32) (err error) {
	return instance.SetProperty("HorizontalResolution", (value))
}

// GetHorizontalResolution gets the value of HorizontalResolution for the instance
func (instance *CIM_VideoControllerResolution) GetPropertyHorizontalResolution() (value uint32, err error) {
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

// SetMaxRefreshRate sets the value of MaxRefreshRate for the instance
func (instance *CIM_VideoControllerResolution) SetPropertyMaxRefreshRate(value uint32) (err error) {
	return instance.SetProperty("MaxRefreshRate", (value))
}

// GetMaxRefreshRate gets the value of MaxRefreshRate for the instance
func (instance *CIM_VideoControllerResolution) GetPropertyMaxRefreshRate() (value uint32, err error) {
	retValue, err := instance.GetProperty("MaxRefreshRate")
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

// SetMinRefreshRate sets the value of MinRefreshRate for the instance
func (instance *CIM_VideoControllerResolution) SetPropertyMinRefreshRate(value uint32) (err error) {
	return instance.SetProperty("MinRefreshRate", (value))
}

// GetMinRefreshRate gets the value of MinRefreshRate for the instance
func (instance *CIM_VideoControllerResolution) GetPropertyMinRefreshRate() (value uint32, err error) {
	retValue, err := instance.GetProperty("MinRefreshRate")
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

// SetNumberOfColors sets the value of NumberOfColors for the instance
func (instance *CIM_VideoControllerResolution) SetPropertyNumberOfColors(value uint64) (err error) {
	return instance.SetProperty("NumberOfColors", (value))
}

// GetNumberOfColors gets the value of NumberOfColors for the instance
func (instance *CIM_VideoControllerResolution) GetPropertyNumberOfColors() (value uint64, err error) {
	retValue, err := instance.GetProperty("NumberOfColors")
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

// SetRefreshRate sets the value of RefreshRate for the instance
func (instance *CIM_VideoControllerResolution) SetPropertyRefreshRate(value uint32) (err error) {
	return instance.SetProperty("RefreshRate", (value))
}

// GetRefreshRate gets the value of RefreshRate for the instance
func (instance *CIM_VideoControllerResolution) GetPropertyRefreshRate() (value uint32, err error) {
	retValue, err := instance.GetProperty("RefreshRate")
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

// SetScanMode sets the value of ScanMode for the instance
func (instance *CIM_VideoControllerResolution) SetPropertyScanMode(value uint16) (err error) {
	return instance.SetProperty("ScanMode", (value))
}

// GetScanMode gets the value of ScanMode for the instance
func (instance *CIM_VideoControllerResolution) GetPropertyScanMode() (value uint16, err error) {
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

// SetVerticalResolution sets the value of VerticalResolution for the instance
func (instance *CIM_VideoControllerResolution) SetPropertyVerticalResolution(value uint32) (err error) {
	return instance.SetProperty("VerticalResolution", (value))
}

// GetVerticalResolution gets the value of VerticalResolution for the instance
func (instance *CIM_VideoControllerResolution) GetPropertyVerticalResolution() (value uint32, err error) {
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
