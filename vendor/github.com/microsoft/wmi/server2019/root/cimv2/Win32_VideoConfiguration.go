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

// Win32_VideoConfiguration struct
type Win32_VideoConfiguration struct {
	*CIM_Setting

	//
	ActualColorResolution uint32

	//
	AdapterChipType string

	//
	AdapterCompatibility string

	//
	AdapterDACType string

	//
	AdapterDescription string

	//
	AdapterRAM uint32

	//
	AdapterType string

	//
	BitsPerPixel uint32

	//
	ColorPlanes uint32

	//
	ColorTableEntries uint32

	//
	DeviceSpecificPens uint32

	//
	DriverDate string

	//
	HorizontalResolution uint32

	//
	InfFilename string

	//
	InfSection string

	//
	InstalledDisplayDrivers string

	//
	MonitorManufacturer string

	//
	MonitorType string

	//
	Name string

	//
	PixelsPerXLogicalInch uint32

	//
	PixelsPerYLogicalInch uint32

	//
	RefreshRate uint32

	//
	ScanMode string

	//
	ScreenHeight uint32

	//
	ScreenWidth uint32

	//
	SystemPaletteEntries uint32

	//
	VerticalResolution uint32
}

func NewWin32_VideoConfigurationEx1(instance *cim.WmiInstance) (newInstance *Win32_VideoConfiguration, err error) {
	tmp, err := NewCIM_SettingEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_VideoConfiguration{
		CIM_Setting: tmp,
	}
	return
}

func NewWin32_VideoConfigurationEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_VideoConfiguration, err error) {
	tmp, err := NewCIM_SettingEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_VideoConfiguration{
		CIM_Setting: tmp,
	}
	return
}

// SetActualColorResolution sets the value of ActualColorResolution for the instance
func (instance *Win32_VideoConfiguration) SetPropertyActualColorResolution(value uint32) (err error) {
	return instance.SetProperty("ActualColorResolution", (value))
}

// GetActualColorResolution gets the value of ActualColorResolution for the instance
func (instance *Win32_VideoConfiguration) GetPropertyActualColorResolution() (value uint32, err error) {
	retValue, err := instance.GetProperty("ActualColorResolution")
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

// SetAdapterChipType sets the value of AdapterChipType for the instance
func (instance *Win32_VideoConfiguration) SetPropertyAdapterChipType(value string) (err error) {
	return instance.SetProperty("AdapterChipType", (value))
}

// GetAdapterChipType gets the value of AdapterChipType for the instance
func (instance *Win32_VideoConfiguration) GetPropertyAdapterChipType() (value string, err error) {
	retValue, err := instance.GetProperty("AdapterChipType")
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

// SetAdapterCompatibility sets the value of AdapterCompatibility for the instance
func (instance *Win32_VideoConfiguration) SetPropertyAdapterCompatibility(value string) (err error) {
	return instance.SetProperty("AdapterCompatibility", (value))
}

// GetAdapterCompatibility gets the value of AdapterCompatibility for the instance
func (instance *Win32_VideoConfiguration) GetPropertyAdapterCompatibility() (value string, err error) {
	retValue, err := instance.GetProperty("AdapterCompatibility")
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

// SetAdapterDACType sets the value of AdapterDACType for the instance
func (instance *Win32_VideoConfiguration) SetPropertyAdapterDACType(value string) (err error) {
	return instance.SetProperty("AdapterDACType", (value))
}

// GetAdapterDACType gets the value of AdapterDACType for the instance
func (instance *Win32_VideoConfiguration) GetPropertyAdapterDACType() (value string, err error) {
	retValue, err := instance.GetProperty("AdapterDACType")
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

// SetAdapterDescription sets the value of AdapterDescription for the instance
func (instance *Win32_VideoConfiguration) SetPropertyAdapterDescription(value string) (err error) {
	return instance.SetProperty("AdapterDescription", (value))
}

// GetAdapterDescription gets the value of AdapterDescription for the instance
func (instance *Win32_VideoConfiguration) GetPropertyAdapterDescription() (value string, err error) {
	retValue, err := instance.GetProperty("AdapterDescription")
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

// SetAdapterRAM sets the value of AdapterRAM for the instance
func (instance *Win32_VideoConfiguration) SetPropertyAdapterRAM(value uint32) (err error) {
	return instance.SetProperty("AdapterRAM", (value))
}

// GetAdapterRAM gets the value of AdapterRAM for the instance
func (instance *Win32_VideoConfiguration) GetPropertyAdapterRAM() (value uint32, err error) {
	retValue, err := instance.GetProperty("AdapterRAM")
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

// SetAdapterType sets the value of AdapterType for the instance
func (instance *Win32_VideoConfiguration) SetPropertyAdapterType(value string) (err error) {
	return instance.SetProperty("AdapterType", (value))
}

// GetAdapterType gets the value of AdapterType for the instance
func (instance *Win32_VideoConfiguration) GetPropertyAdapterType() (value string, err error) {
	retValue, err := instance.GetProperty("AdapterType")
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

// SetBitsPerPixel sets the value of BitsPerPixel for the instance
func (instance *Win32_VideoConfiguration) SetPropertyBitsPerPixel(value uint32) (err error) {
	return instance.SetProperty("BitsPerPixel", (value))
}

// GetBitsPerPixel gets the value of BitsPerPixel for the instance
func (instance *Win32_VideoConfiguration) GetPropertyBitsPerPixel() (value uint32, err error) {
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
func (instance *Win32_VideoConfiguration) SetPropertyColorPlanes(value uint32) (err error) {
	return instance.SetProperty("ColorPlanes", (value))
}

// GetColorPlanes gets the value of ColorPlanes for the instance
func (instance *Win32_VideoConfiguration) GetPropertyColorPlanes() (value uint32, err error) {
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

// SetColorTableEntries sets the value of ColorTableEntries for the instance
func (instance *Win32_VideoConfiguration) SetPropertyColorTableEntries(value uint32) (err error) {
	return instance.SetProperty("ColorTableEntries", (value))
}

// GetColorTableEntries gets the value of ColorTableEntries for the instance
func (instance *Win32_VideoConfiguration) GetPropertyColorTableEntries() (value uint32, err error) {
	retValue, err := instance.GetProperty("ColorTableEntries")
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
func (instance *Win32_VideoConfiguration) SetPropertyDeviceSpecificPens(value uint32) (err error) {
	return instance.SetProperty("DeviceSpecificPens", (value))
}

// GetDeviceSpecificPens gets the value of DeviceSpecificPens for the instance
func (instance *Win32_VideoConfiguration) GetPropertyDeviceSpecificPens() (value uint32, err error) {
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

// SetDriverDate sets the value of DriverDate for the instance
func (instance *Win32_VideoConfiguration) SetPropertyDriverDate(value string) (err error) {
	return instance.SetProperty("DriverDate", (value))
}

// GetDriverDate gets the value of DriverDate for the instance
func (instance *Win32_VideoConfiguration) GetPropertyDriverDate() (value string, err error) {
	retValue, err := instance.GetProperty("DriverDate")
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

// SetHorizontalResolution sets the value of HorizontalResolution for the instance
func (instance *Win32_VideoConfiguration) SetPropertyHorizontalResolution(value uint32) (err error) {
	return instance.SetProperty("HorizontalResolution", (value))
}

// GetHorizontalResolution gets the value of HorizontalResolution for the instance
func (instance *Win32_VideoConfiguration) GetPropertyHorizontalResolution() (value uint32, err error) {
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

// SetInfFilename sets the value of InfFilename for the instance
func (instance *Win32_VideoConfiguration) SetPropertyInfFilename(value string) (err error) {
	return instance.SetProperty("InfFilename", (value))
}

// GetInfFilename gets the value of InfFilename for the instance
func (instance *Win32_VideoConfiguration) GetPropertyInfFilename() (value string, err error) {
	retValue, err := instance.GetProperty("InfFilename")
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

// SetInfSection sets the value of InfSection for the instance
func (instance *Win32_VideoConfiguration) SetPropertyInfSection(value string) (err error) {
	return instance.SetProperty("InfSection", (value))
}

// GetInfSection gets the value of InfSection for the instance
func (instance *Win32_VideoConfiguration) GetPropertyInfSection() (value string, err error) {
	retValue, err := instance.GetProperty("InfSection")
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

// SetInstalledDisplayDrivers sets the value of InstalledDisplayDrivers for the instance
func (instance *Win32_VideoConfiguration) SetPropertyInstalledDisplayDrivers(value string) (err error) {
	return instance.SetProperty("InstalledDisplayDrivers", (value))
}

// GetInstalledDisplayDrivers gets the value of InstalledDisplayDrivers for the instance
func (instance *Win32_VideoConfiguration) GetPropertyInstalledDisplayDrivers() (value string, err error) {
	retValue, err := instance.GetProperty("InstalledDisplayDrivers")
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

// SetMonitorManufacturer sets the value of MonitorManufacturer for the instance
func (instance *Win32_VideoConfiguration) SetPropertyMonitorManufacturer(value string) (err error) {
	return instance.SetProperty("MonitorManufacturer", (value))
}

// GetMonitorManufacturer gets the value of MonitorManufacturer for the instance
func (instance *Win32_VideoConfiguration) GetPropertyMonitorManufacturer() (value string, err error) {
	retValue, err := instance.GetProperty("MonitorManufacturer")
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

// SetMonitorType sets the value of MonitorType for the instance
func (instance *Win32_VideoConfiguration) SetPropertyMonitorType(value string) (err error) {
	return instance.SetProperty("MonitorType", (value))
}

// GetMonitorType gets the value of MonitorType for the instance
func (instance *Win32_VideoConfiguration) GetPropertyMonitorType() (value string, err error) {
	retValue, err := instance.GetProperty("MonitorType")
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
func (instance *Win32_VideoConfiguration) SetPropertyName(value string) (err error) {
	return instance.SetProperty("Name", (value))
}

// GetName gets the value of Name for the instance
func (instance *Win32_VideoConfiguration) GetPropertyName() (value string, err error) {
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

// SetPixelsPerXLogicalInch sets the value of PixelsPerXLogicalInch for the instance
func (instance *Win32_VideoConfiguration) SetPropertyPixelsPerXLogicalInch(value uint32) (err error) {
	return instance.SetProperty("PixelsPerXLogicalInch", (value))
}

// GetPixelsPerXLogicalInch gets the value of PixelsPerXLogicalInch for the instance
func (instance *Win32_VideoConfiguration) GetPropertyPixelsPerXLogicalInch() (value uint32, err error) {
	retValue, err := instance.GetProperty("PixelsPerXLogicalInch")
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

// SetPixelsPerYLogicalInch sets the value of PixelsPerYLogicalInch for the instance
func (instance *Win32_VideoConfiguration) SetPropertyPixelsPerYLogicalInch(value uint32) (err error) {
	return instance.SetProperty("PixelsPerYLogicalInch", (value))
}

// GetPixelsPerYLogicalInch gets the value of PixelsPerYLogicalInch for the instance
func (instance *Win32_VideoConfiguration) GetPropertyPixelsPerYLogicalInch() (value uint32, err error) {
	retValue, err := instance.GetProperty("PixelsPerYLogicalInch")
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

// SetRefreshRate sets the value of RefreshRate for the instance
func (instance *Win32_VideoConfiguration) SetPropertyRefreshRate(value uint32) (err error) {
	return instance.SetProperty("RefreshRate", (value))
}

// GetRefreshRate gets the value of RefreshRate for the instance
func (instance *Win32_VideoConfiguration) GetPropertyRefreshRate() (value uint32, err error) {
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
func (instance *Win32_VideoConfiguration) SetPropertyScanMode(value string) (err error) {
	return instance.SetProperty("ScanMode", (value))
}

// GetScanMode gets the value of ScanMode for the instance
func (instance *Win32_VideoConfiguration) GetPropertyScanMode() (value string, err error) {
	retValue, err := instance.GetProperty("ScanMode")
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

// SetScreenHeight sets the value of ScreenHeight for the instance
func (instance *Win32_VideoConfiguration) SetPropertyScreenHeight(value uint32) (err error) {
	return instance.SetProperty("ScreenHeight", (value))
}

// GetScreenHeight gets the value of ScreenHeight for the instance
func (instance *Win32_VideoConfiguration) GetPropertyScreenHeight() (value uint32, err error) {
	retValue, err := instance.GetProperty("ScreenHeight")
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

// SetScreenWidth sets the value of ScreenWidth for the instance
func (instance *Win32_VideoConfiguration) SetPropertyScreenWidth(value uint32) (err error) {
	return instance.SetProperty("ScreenWidth", (value))
}

// GetScreenWidth gets the value of ScreenWidth for the instance
func (instance *Win32_VideoConfiguration) GetPropertyScreenWidth() (value uint32, err error) {
	retValue, err := instance.GetProperty("ScreenWidth")
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
func (instance *Win32_VideoConfiguration) SetPropertySystemPaletteEntries(value uint32) (err error) {
	return instance.SetProperty("SystemPaletteEntries", (value))
}

// GetSystemPaletteEntries gets the value of SystemPaletteEntries for the instance
func (instance *Win32_VideoConfiguration) GetPropertySystemPaletteEntries() (value uint32, err error) {
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
func (instance *Win32_VideoConfiguration) SetPropertyVerticalResolution(value uint32) (err error) {
	return instance.SetProperty("VerticalResolution", (value))
}

// GetVerticalResolution gets the value of VerticalResolution for the instance
func (instance *Win32_VideoConfiguration) GetPropertyVerticalResolution() (value uint32, err error) {
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
