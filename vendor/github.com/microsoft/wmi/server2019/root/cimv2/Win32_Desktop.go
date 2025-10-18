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

// Win32_Desktop struct
type Win32_Desktop struct {
	*CIM_Setting

	//
	BorderWidth uint32

	//
	CoolSwitch bool

	//
	CursorBlinkRate uint32

	//
	DragFullWindows bool

	//
	GridGranularity uint32

	//
	IconSpacing uint32

	//
	IconTitleFaceName string

	//
	IconTitleSize uint32

	//
	IconTitleWrap bool

	//
	Name string

	//
	Pattern string

	//
	ScreenSaverActive bool

	//
	ScreenSaverExecutable string

	//
	ScreenSaverSecure bool

	//
	ScreenSaverTimeout uint32

	//
	Wallpaper string

	//
	WallpaperStretched bool

	//
	WallpaperTiled bool
}

func NewWin32_DesktopEx1(instance *cim.WmiInstance) (newInstance *Win32_Desktop, err error) {
	tmp, err := NewCIM_SettingEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_Desktop{
		CIM_Setting: tmp,
	}
	return
}

func NewWin32_DesktopEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_Desktop, err error) {
	tmp, err := NewCIM_SettingEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_Desktop{
		CIM_Setting: tmp,
	}
	return
}

// SetBorderWidth sets the value of BorderWidth for the instance
func (instance *Win32_Desktop) SetPropertyBorderWidth(value uint32) (err error) {
	return instance.SetProperty("BorderWidth", (value))
}

// GetBorderWidth gets the value of BorderWidth for the instance
func (instance *Win32_Desktop) GetPropertyBorderWidth() (value uint32, err error) {
	retValue, err := instance.GetProperty("BorderWidth")
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

// SetCoolSwitch sets the value of CoolSwitch for the instance
func (instance *Win32_Desktop) SetPropertyCoolSwitch(value bool) (err error) {
	return instance.SetProperty("CoolSwitch", (value))
}

// GetCoolSwitch gets the value of CoolSwitch for the instance
func (instance *Win32_Desktop) GetPropertyCoolSwitch() (value bool, err error) {
	retValue, err := instance.GetProperty("CoolSwitch")
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

// SetCursorBlinkRate sets the value of CursorBlinkRate for the instance
func (instance *Win32_Desktop) SetPropertyCursorBlinkRate(value uint32) (err error) {
	return instance.SetProperty("CursorBlinkRate", (value))
}

// GetCursorBlinkRate gets the value of CursorBlinkRate for the instance
func (instance *Win32_Desktop) GetPropertyCursorBlinkRate() (value uint32, err error) {
	retValue, err := instance.GetProperty("CursorBlinkRate")
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

// SetDragFullWindows sets the value of DragFullWindows for the instance
func (instance *Win32_Desktop) SetPropertyDragFullWindows(value bool) (err error) {
	return instance.SetProperty("DragFullWindows", (value))
}

// GetDragFullWindows gets the value of DragFullWindows for the instance
func (instance *Win32_Desktop) GetPropertyDragFullWindows() (value bool, err error) {
	retValue, err := instance.GetProperty("DragFullWindows")
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

// SetGridGranularity sets the value of GridGranularity for the instance
func (instance *Win32_Desktop) SetPropertyGridGranularity(value uint32) (err error) {
	return instance.SetProperty("GridGranularity", (value))
}

// GetGridGranularity gets the value of GridGranularity for the instance
func (instance *Win32_Desktop) GetPropertyGridGranularity() (value uint32, err error) {
	retValue, err := instance.GetProperty("GridGranularity")
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

// SetIconSpacing sets the value of IconSpacing for the instance
func (instance *Win32_Desktop) SetPropertyIconSpacing(value uint32) (err error) {
	return instance.SetProperty("IconSpacing", (value))
}

// GetIconSpacing gets the value of IconSpacing for the instance
func (instance *Win32_Desktop) GetPropertyIconSpacing() (value uint32, err error) {
	retValue, err := instance.GetProperty("IconSpacing")
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

// SetIconTitleFaceName sets the value of IconTitleFaceName for the instance
func (instance *Win32_Desktop) SetPropertyIconTitleFaceName(value string) (err error) {
	return instance.SetProperty("IconTitleFaceName", (value))
}

// GetIconTitleFaceName gets the value of IconTitleFaceName for the instance
func (instance *Win32_Desktop) GetPropertyIconTitleFaceName() (value string, err error) {
	retValue, err := instance.GetProperty("IconTitleFaceName")
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

// SetIconTitleSize sets the value of IconTitleSize for the instance
func (instance *Win32_Desktop) SetPropertyIconTitleSize(value uint32) (err error) {
	return instance.SetProperty("IconTitleSize", (value))
}

// GetIconTitleSize gets the value of IconTitleSize for the instance
func (instance *Win32_Desktop) GetPropertyIconTitleSize() (value uint32, err error) {
	retValue, err := instance.GetProperty("IconTitleSize")
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

// SetIconTitleWrap sets the value of IconTitleWrap for the instance
func (instance *Win32_Desktop) SetPropertyIconTitleWrap(value bool) (err error) {
	return instance.SetProperty("IconTitleWrap", (value))
}

// GetIconTitleWrap gets the value of IconTitleWrap for the instance
func (instance *Win32_Desktop) GetPropertyIconTitleWrap() (value bool, err error) {
	retValue, err := instance.GetProperty("IconTitleWrap")
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

// SetName sets the value of Name for the instance
func (instance *Win32_Desktop) SetPropertyName(value string) (err error) {
	return instance.SetProperty("Name", (value))
}

// GetName gets the value of Name for the instance
func (instance *Win32_Desktop) GetPropertyName() (value string, err error) {
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

// SetPattern sets the value of Pattern for the instance
func (instance *Win32_Desktop) SetPropertyPattern(value string) (err error) {
	return instance.SetProperty("Pattern", (value))
}

// GetPattern gets the value of Pattern for the instance
func (instance *Win32_Desktop) GetPropertyPattern() (value string, err error) {
	retValue, err := instance.GetProperty("Pattern")
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

// SetScreenSaverActive sets the value of ScreenSaverActive for the instance
func (instance *Win32_Desktop) SetPropertyScreenSaverActive(value bool) (err error) {
	return instance.SetProperty("ScreenSaverActive", (value))
}

// GetScreenSaverActive gets the value of ScreenSaverActive for the instance
func (instance *Win32_Desktop) GetPropertyScreenSaverActive() (value bool, err error) {
	retValue, err := instance.GetProperty("ScreenSaverActive")
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

// SetScreenSaverExecutable sets the value of ScreenSaverExecutable for the instance
func (instance *Win32_Desktop) SetPropertyScreenSaverExecutable(value string) (err error) {
	return instance.SetProperty("ScreenSaverExecutable", (value))
}

// GetScreenSaverExecutable gets the value of ScreenSaverExecutable for the instance
func (instance *Win32_Desktop) GetPropertyScreenSaverExecutable() (value string, err error) {
	retValue, err := instance.GetProperty("ScreenSaverExecutable")
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

// SetScreenSaverSecure sets the value of ScreenSaverSecure for the instance
func (instance *Win32_Desktop) SetPropertyScreenSaverSecure(value bool) (err error) {
	return instance.SetProperty("ScreenSaverSecure", (value))
}

// GetScreenSaverSecure gets the value of ScreenSaverSecure for the instance
func (instance *Win32_Desktop) GetPropertyScreenSaverSecure() (value bool, err error) {
	retValue, err := instance.GetProperty("ScreenSaverSecure")
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

// SetScreenSaverTimeout sets the value of ScreenSaverTimeout for the instance
func (instance *Win32_Desktop) SetPropertyScreenSaverTimeout(value uint32) (err error) {
	return instance.SetProperty("ScreenSaverTimeout", (value))
}

// GetScreenSaverTimeout gets the value of ScreenSaverTimeout for the instance
func (instance *Win32_Desktop) GetPropertyScreenSaverTimeout() (value uint32, err error) {
	retValue, err := instance.GetProperty("ScreenSaverTimeout")
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

// SetWallpaper sets the value of Wallpaper for the instance
func (instance *Win32_Desktop) SetPropertyWallpaper(value string) (err error) {
	return instance.SetProperty("Wallpaper", (value))
}

// GetWallpaper gets the value of Wallpaper for the instance
func (instance *Win32_Desktop) GetPropertyWallpaper() (value string, err error) {
	retValue, err := instance.GetProperty("Wallpaper")
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

// SetWallpaperStretched sets the value of WallpaperStretched for the instance
func (instance *Win32_Desktop) SetPropertyWallpaperStretched(value bool) (err error) {
	return instance.SetProperty("WallpaperStretched", (value))
}

// GetWallpaperStretched gets the value of WallpaperStretched for the instance
func (instance *Win32_Desktop) GetPropertyWallpaperStretched() (value bool, err error) {
	retValue, err := instance.GetProperty("WallpaperStretched")
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

// SetWallpaperTiled sets the value of WallpaperTiled for the instance
func (instance *Win32_Desktop) SetPropertyWallpaperTiled(value bool) (err error) {
	return instance.SetProperty("WallpaperTiled", (value))
}

// GetWallpaperTiled gets the value of WallpaperTiled for the instance
func (instance *Win32_Desktop) GetPropertyWallpaperTiled() (value bool, err error) {
	retValue, err := instance.GetProperty("WallpaperTiled")
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
