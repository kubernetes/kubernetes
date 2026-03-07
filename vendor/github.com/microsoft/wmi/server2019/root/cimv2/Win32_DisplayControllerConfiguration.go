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

// Win32_DisplayControllerConfiguration struct
type Win32_DisplayControllerConfiguration struct {
	*CIM_Setting

	//
	BitsPerPixel uint32

	//
	ColorPlanes uint32

	//
	DeviceEntriesInAColorTable uint32

	//
	DeviceSpecificPens uint32

	//
	HorizontalResolution uint32

	//
	Name string

	//
	RefreshRate int32

	//
	ReservedSystemPaletteEntries uint32

	//
	SystemPaletteEntries uint32

	//
	VerticalResolution uint32

	//
	VideoMode string
}

func NewWin32_DisplayControllerConfigurationEx1(instance *cim.WmiInstance) (newInstance *Win32_DisplayControllerConfiguration, err error) {
	tmp, err := NewCIM_SettingEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_DisplayControllerConfiguration{
		CIM_Setting: tmp,
	}
	return
}

func NewWin32_DisplayControllerConfigurationEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_DisplayControllerConfiguration, err error) {
	tmp, err := NewCIM_SettingEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_DisplayControllerConfiguration{
		CIM_Setting: tmp,
	}
	return
}

// SetBitsPerPixel sets the value of BitsPerPixel for the instance
func (instance *Win32_DisplayControllerConfiguration) SetPropertyBitsPerPixel(value uint32) (err error) {
	return instance.SetProperty("BitsPerPixel", (value))
}

// GetBitsPerPixel gets the value of BitsPerPixel for the instance
func (instance *Win32_DisplayControllerConfiguration) GetPropertyBitsPerPixel() (value uint32, err error) {
	retValue, err := instance.GetProperty("BitsPerPixel")
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

// SetColorPlanes sets the value of ColorPlanes for the instance
func (instance *Win32_DisplayControllerConfiguration) SetPropertyColorPlanes(value uint32) (err error) {
	return instance.SetProperty("ColorPlanes", (value))
}

// GetColorPlanes gets the value of ColorPlanes for the instance
func (instance *Win32_DisplayControllerConfiguration) GetPropertyColorPlanes() (value uint32, err error) {
	retValue, err := instance.GetProperty("ColorPlanes")
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

// SetDeviceEntriesInAColorTable sets the value of DeviceEntriesInAColorTable for the instance
func (instance *Win32_DisplayControllerConfiguration) SetPropertyDeviceEntriesInAColorTable(value uint32) (err error) {
	return instance.SetProperty("DeviceEntriesInAColorTable", (value))
}

// GetDeviceEntriesInAColorTable gets the value of DeviceEntriesInAColorTable for the instance
func (instance *Win32_DisplayControllerConfiguration) GetPropertyDeviceEntriesInAColorTable() (value uint32, err error) {
	retValue, err := instance.GetProperty("DeviceEntriesInAColorTable")
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

// SetDeviceSpecificPens sets the value of DeviceSpecificPens for the instance
func (instance *Win32_DisplayControllerConfiguration) SetPropertyDeviceSpecificPens(value uint32) (err error) {
	return instance.SetProperty("DeviceSpecificPens", (value))
}

// GetDeviceSpecificPens gets the value of DeviceSpecificPens for the instance
func (instance *Win32_DisplayControllerConfiguration) GetPropertyDeviceSpecificPens() (value uint32, err error) {
	retValue, err := instance.GetProperty("DeviceSpecificPens")
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

// SetHorizontalResolution sets the value of HorizontalResolution for the instance
func (instance *Win32_DisplayControllerConfiguration) SetPropertyHorizontalResolution(value uint32) (err error) {
	return instance.SetProperty("HorizontalResolution", (value))
}

// GetHorizontalResolution gets the value of HorizontalResolution for the instance
func (instance *Win32_DisplayControllerConfiguration) GetPropertyHorizontalResolution() (value uint32, err error) {
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

// SetName sets the value of Name for the instance
func (instance *Win32_DisplayControllerConfiguration) SetPropertyName(value string) (err error) {
	return instance.SetProperty("Name", (value))
}

// GetName gets the value of Name for the instance
func (instance *Win32_DisplayControllerConfiguration) GetPropertyName() (value string, err error) {
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

// SetRefreshRate sets the value of RefreshRate for the instance
func (instance *Win32_DisplayControllerConfiguration) SetPropertyRefreshRate(value int32) (err error) {
	return instance.SetProperty("RefreshRate", (value))
}

// GetRefreshRate gets the value of RefreshRate for the instance
func (instance *Win32_DisplayControllerConfiguration) GetPropertyRefreshRate() (value int32, err error) {
	retValue, err := instance.GetProperty("RefreshRate")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(int32)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " int32 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = int32(valuetmp)

	return
}

// SetReservedSystemPaletteEntries sets the value of ReservedSystemPaletteEntries for the instance
func (instance *Win32_DisplayControllerConfiguration) SetPropertyReservedSystemPaletteEntries(value uint32) (err error) {
	return instance.SetProperty("ReservedSystemPaletteEntries", (value))
}

// GetReservedSystemPaletteEntries gets the value of ReservedSystemPaletteEntries for the instance
func (instance *Win32_DisplayControllerConfiguration) GetPropertyReservedSystemPaletteEntries() (value uint32, err error) {
	retValue, err := instance.GetProperty("ReservedSystemPaletteEntries")
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

// SetSystemPaletteEntries sets the value of SystemPaletteEntries for the instance
func (instance *Win32_DisplayControllerConfiguration) SetPropertySystemPaletteEntries(value uint32) (err error) {
	return instance.SetProperty("SystemPaletteEntries", (value))
}

// GetSystemPaletteEntries gets the value of SystemPaletteEntries for the instance
func (instance *Win32_DisplayControllerConfiguration) GetPropertySystemPaletteEntries() (value uint32, err error) {
	retValue, err := instance.GetProperty("SystemPaletteEntries")
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

// SetVerticalResolution sets the value of VerticalResolution for the instance
func (instance *Win32_DisplayControllerConfiguration) SetPropertyVerticalResolution(value uint32) (err error) {
	return instance.SetProperty("VerticalResolution", (value))
}

// GetVerticalResolution gets the value of VerticalResolution for the instance
func (instance *Win32_DisplayControllerConfiguration) GetPropertyVerticalResolution() (value uint32, err error) {
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

// SetVideoMode sets the value of VideoMode for the instance
func (instance *Win32_DisplayControllerConfiguration) SetPropertyVideoMode(value string) (err error) {
	return instance.SetProperty("VideoMode", (value))
}

// GetVideoMode gets the value of VideoMode for the instance
func (instance *Win32_DisplayControllerConfiguration) GetPropertyVideoMode() (value string, err error) {
	retValue, err := instance.GetProperty("VideoMode")
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
