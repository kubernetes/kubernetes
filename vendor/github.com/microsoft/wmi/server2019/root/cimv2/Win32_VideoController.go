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

// Win32_VideoController struct
type Win32_VideoController struct {
	*CIM_PCVideoController

	//
	AdapterCompatibility string

	//
	AdapterDACType string

	//
	AdapterRAM uint32

	//
	ColorTableEntries uint32

	//
	DeviceSpecificPens uint32

	//
	DitherType uint32

	//
	DriverDate string

	//
	DriverVersion string

	//
	ICMIntent uint32

	//
	ICMMethod uint32

	//
	InfFilename string

	//
	InfSection string

	//
	InstalledDisplayDrivers string

	//
	Monochrome bool

	//
	ReservedSystemPaletteEntries uint32

	//
	SpecificationVersion uint32

	//
	SystemPaletteEntries uint32

	//
	VideoModeDescription string
}

func NewWin32_VideoControllerEx1(instance *cim.WmiInstance) (newInstance *Win32_VideoController, err error) {
	tmp, err := NewCIM_PCVideoControllerEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_VideoController{
		CIM_PCVideoController: tmp,
	}
	return
}

func NewWin32_VideoControllerEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_VideoController, err error) {
	tmp, err := NewCIM_PCVideoControllerEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_VideoController{
		CIM_PCVideoController: tmp,
	}
	return
}

// SetAdapterCompatibility sets the value of AdapterCompatibility for the instance
func (instance *Win32_VideoController) SetPropertyAdapterCompatibility(value string) (err error) {
	return instance.SetProperty("AdapterCompatibility", (value))
}

// GetAdapterCompatibility gets the value of AdapterCompatibility for the instance
func (instance *Win32_VideoController) GetPropertyAdapterCompatibility() (value string, err error) {
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
func (instance *Win32_VideoController) SetPropertyAdapterDACType(value string) (err error) {
	return instance.SetProperty("AdapterDACType", (value))
}

// GetAdapterDACType gets the value of AdapterDACType for the instance
func (instance *Win32_VideoController) GetPropertyAdapterDACType() (value string, err error) {
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

// SetAdapterRAM sets the value of AdapterRAM for the instance
func (instance *Win32_VideoController) SetPropertyAdapterRAM(value uint32) (err error) {
	return instance.SetProperty("AdapterRAM", (value))
}

// GetAdapterRAM gets the value of AdapterRAM for the instance
func (instance *Win32_VideoController) GetPropertyAdapterRAM() (value uint32, err error) {
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

// SetColorTableEntries sets the value of ColorTableEntries for the instance
func (instance *Win32_VideoController) SetPropertyColorTableEntries(value uint32) (err error) {
	return instance.SetProperty("ColorTableEntries", (value))
}

// GetColorTableEntries gets the value of ColorTableEntries for the instance
func (instance *Win32_VideoController) GetPropertyColorTableEntries() (value uint32, err error) {
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
func (instance *Win32_VideoController) SetPropertyDeviceSpecificPens(value uint32) (err error) {
	return instance.SetProperty("DeviceSpecificPens", (value))
}

// GetDeviceSpecificPens gets the value of DeviceSpecificPens for the instance
func (instance *Win32_VideoController) GetPropertyDeviceSpecificPens() (value uint32, err error) {
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

// SetDitherType sets the value of DitherType for the instance
func (instance *Win32_VideoController) SetPropertyDitherType(value uint32) (err error) {
	return instance.SetProperty("DitherType", (value))
}

// GetDitherType gets the value of DitherType for the instance
func (instance *Win32_VideoController) GetPropertyDitherType() (value uint32, err error) {
	retValue, err := instance.GetProperty("DitherType")
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
func (instance *Win32_VideoController) SetPropertyDriverDate(value string) (err error) {
	return instance.SetProperty("DriverDate", (value))
}

// GetDriverDate gets the value of DriverDate for the instance
func (instance *Win32_VideoController) GetPropertyDriverDate() (value string, err error) {
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

// SetDriverVersion sets the value of DriverVersion for the instance
func (instance *Win32_VideoController) SetPropertyDriverVersion(value string) (err error) {
	return instance.SetProperty("DriverVersion", (value))
}

// GetDriverVersion gets the value of DriverVersion for the instance
func (instance *Win32_VideoController) GetPropertyDriverVersion() (value string, err error) {
	retValue, err := instance.GetProperty("DriverVersion")
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

// SetICMIntent sets the value of ICMIntent for the instance
func (instance *Win32_VideoController) SetPropertyICMIntent(value uint32) (err error) {
	return instance.SetProperty("ICMIntent", (value))
}

// GetICMIntent gets the value of ICMIntent for the instance
func (instance *Win32_VideoController) GetPropertyICMIntent() (value uint32, err error) {
	retValue, err := instance.GetProperty("ICMIntent")
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

// SetICMMethod sets the value of ICMMethod for the instance
func (instance *Win32_VideoController) SetPropertyICMMethod(value uint32) (err error) {
	return instance.SetProperty("ICMMethod", (value))
}

// GetICMMethod gets the value of ICMMethod for the instance
func (instance *Win32_VideoController) GetPropertyICMMethod() (value uint32, err error) {
	retValue, err := instance.GetProperty("ICMMethod")
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
func (instance *Win32_VideoController) SetPropertyInfFilename(value string) (err error) {
	return instance.SetProperty("InfFilename", (value))
}

// GetInfFilename gets the value of InfFilename for the instance
func (instance *Win32_VideoController) GetPropertyInfFilename() (value string, err error) {
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
func (instance *Win32_VideoController) SetPropertyInfSection(value string) (err error) {
	return instance.SetProperty("InfSection", (value))
}

// GetInfSection gets the value of InfSection for the instance
func (instance *Win32_VideoController) GetPropertyInfSection() (value string, err error) {
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
func (instance *Win32_VideoController) SetPropertyInstalledDisplayDrivers(value string) (err error) {
	return instance.SetProperty("InstalledDisplayDrivers", (value))
}

// GetInstalledDisplayDrivers gets the value of InstalledDisplayDrivers for the instance
func (instance *Win32_VideoController) GetPropertyInstalledDisplayDrivers() (value string, err error) {
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

// SetMonochrome sets the value of Monochrome for the instance
func (instance *Win32_VideoController) SetPropertyMonochrome(value bool) (err error) {
	return instance.SetProperty("Monochrome", (value))
}

// GetMonochrome gets the value of Monochrome for the instance
func (instance *Win32_VideoController) GetPropertyMonochrome() (value bool, err error) {
	retValue, err := instance.GetProperty("Monochrome")
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

// SetReservedSystemPaletteEntries sets the value of ReservedSystemPaletteEntries for the instance
func (instance *Win32_VideoController) SetPropertyReservedSystemPaletteEntries(value uint32) (err error) {
	return instance.SetProperty("ReservedSystemPaletteEntries", (value))
}

// GetReservedSystemPaletteEntries gets the value of ReservedSystemPaletteEntries for the instance
func (instance *Win32_VideoController) GetPropertyReservedSystemPaletteEntries() (value uint32, err error) {
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

// SetSpecificationVersion sets the value of SpecificationVersion for the instance
func (instance *Win32_VideoController) SetPropertySpecificationVersion(value uint32) (err error) {
	return instance.SetProperty("SpecificationVersion", (value))
}

// GetSpecificationVersion gets the value of SpecificationVersion for the instance
func (instance *Win32_VideoController) GetPropertySpecificationVersion() (value uint32, err error) {
	retValue, err := instance.GetProperty("SpecificationVersion")
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
func (instance *Win32_VideoController) SetPropertySystemPaletteEntries(value uint32) (err error) {
	return instance.SetProperty("SystemPaletteEntries", (value))
}

// GetSystemPaletteEntries gets the value of SystemPaletteEntries for the instance
func (instance *Win32_VideoController) GetPropertySystemPaletteEntries() (value uint32, err error) {
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

// SetVideoModeDescription sets the value of VideoModeDescription for the instance
func (instance *Win32_VideoController) SetPropertyVideoModeDescription(value string) (err error) {
	return instance.SetProperty("VideoModeDescription", (value))
}

// GetVideoModeDescription gets the value of VideoModeDescription for the instance
func (instance *Win32_VideoController) GetPropertyVideoModeDescription() (value string, err error) {
	retValue, err := instance.GetProperty("VideoModeDescription")
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
