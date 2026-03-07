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

// Win32_BIOS struct
type Win32_BIOS struct {
	*CIM_BIOSElement

	//
	BiosCharacteristics []uint16

	//
	BIOSVersion []string

	//
	CurrentLanguage string

	//
	EmbeddedControllerMajorVersion uint8

	//
	EmbeddedControllerMinorVersion uint8

	//
	InstallableLanguages uint16

	//
	ListOfLanguages []string

	//
	ReleaseDate string

	//
	SMBIOSBIOSVersion string

	//
	SMBIOSMajorVersion uint16

	//
	SMBIOSMinorVersion uint16

	//
	SMBIOSPresent bool

	//
	SystemBiosMajorVersion uint8

	//
	SystemBiosMinorVersion uint8
}

func NewWin32_BIOSEx1(instance *cim.WmiInstance) (newInstance *Win32_BIOS, err error) {
	tmp, err := NewCIM_BIOSElementEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_BIOS{
		CIM_BIOSElement: tmp,
	}
	return
}

func NewWin32_BIOSEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_BIOS, err error) {
	tmp, err := NewCIM_BIOSElementEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_BIOS{
		CIM_BIOSElement: tmp,
	}
	return
}

// SetBiosCharacteristics sets the value of BiosCharacteristics for the instance
func (instance *Win32_BIOS) SetPropertyBiosCharacteristics(value []uint16) (err error) {
	return instance.SetProperty("BiosCharacteristics", (value))
}

// GetBiosCharacteristics gets the value of BiosCharacteristics for the instance
func (instance *Win32_BIOS) GetPropertyBiosCharacteristics() (value []uint16, err error) {
	retValue, err := instance.GetProperty("BiosCharacteristics")
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

// SetBIOSVersion sets the value of BIOSVersion for the instance
func (instance *Win32_BIOS) SetPropertyBIOSVersion(value []string) (err error) {
	return instance.SetProperty("BIOSVersion", (value))
}

// GetBIOSVersion gets the value of BIOSVersion for the instance
func (instance *Win32_BIOS) GetPropertyBIOSVersion() (value []string, err error) {
	retValue, err := instance.GetProperty("BIOSVersion")
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

// SetCurrentLanguage sets the value of CurrentLanguage for the instance
func (instance *Win32_BIOS) SetPropertyCurrentLanguage(value string) (err error) {
	return instance.SetProperty("CurrentLanguage", (value))
}

// GetCurrentLanguage gets the value of CurrentLanguage for the instance
func (instance *Win32_BIOS) GetPropertyCurrentLanguage() (value string, err error) {
	retValue, err := instance.GetProperty("CurrentLanguage")
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

// SetEmbeddedControllerMajorVersion sets the value of EmbeddedControllerMajorVersion for the instance
func (instance *Win32_BIOS) SetPropertyEmbeddedControllerMajorVersion(value uint8) (err error) {
	return instance.SetProperty("EmbeddedControllerMajorVersion", (value))
}

// GetEmbeddedControllerMajorVersion gets the value of EmbeddedControllerMajorVersion for the instance
func (instance *Win32_BIOS) GetPropertyEmbeddedControllerMajorVersion() (value uint8, err error) {
	retValue, err := instance.GetProperty("EmbeddedControllerMajorVersion")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint8)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint8 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint8(valuetmp)

	return
}

// SetEmbeddedControllerMinorVersion sets the value of EmbeddedControllerMinorVersion for the instance
func (instance *Win32_BIOS) SetPropertyEmbeddedControllerMinorVersion(value uint8) (err error) {
	return instance.SetProperty("EmbeddedControllerMinorVersion", (value))
}

// GetEmbeddedControllerMinorVersion gets the value of EmbeddedControllerMinorVersion for the instance
func (instance *Win32_BIOS) GetPropertyEmbeddedControllerMinorVersion() (value uint8, err error) {
	retValue, err := instance.GetProperty("EmbeddedControllerMinorVersion")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint8)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint8 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint8(valuetmp)

	return
}

// SetInstallableLanguages sets the value of InstallableLanguages for the instance
func (instance *Win32_BIOS) SetPropertyInstallableLanguages(value uint16) (err error) {
	return instance.SetProperty("InstallableLanguages", (value))
}

// GetInstallableLanguages gets the value of InstallableLanguages for the instance
func (instance *Win32_BIOS) GetPropertyInstallableLanguages() (value uint16, err error) {
	retValue, err := instance.GetProperty("InstallableLanguages")
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

// SetListOfLanguages sets the value of ListOfLanguages for the instance
func (instance *Win32_BIOS) SetPropertyListOfLanguages(value []string) (err error) {
	return instance.SetProperty("ListOfLanguages", (value))
}

// GetListOfLanguages gets the value of ListOfLanguages for the instance
func (instance *Win32_BIOS) GetPropertyListOfLanguages() (value []string, err error) {
	retValue, err := instance.GetProperty("ListOfLanguages")
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

// SetReleaseDate sets the value of ReleaseDate for the instance
func (instance *Win32_BIOS) SetPropertyReleaseDate(value string) (err error) {
	return instance.SetProperty("ReleaseDate", (value))
}

// GetReleaseDate gets the value of ReleaseDate for the instance
func (instance *Win32_BIOS) GetPropertyReleaseDate() (value string, err error) {
	retValue, err := instance.GetProperty("ReleaseDate")
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

// SetSMBIOSBIOSVersion sets the value of SMBIOSBIOSVersion for the instance
func (instance *Win32_BIOS) SetPropertySMBIOSBIOSVersion(value string) (err error) {
	return instance.SetProperty("SMBIOSBIOSVersion", (value))
}

// GetSMBIOSBIOSVersion gets the value of SMBIOSBIOSVersion for the instance
func (instance *Win32_BIOS) GetPropertySMBIOSBIOSVersion() (value string, err error) {
	retValue, err := instance.GetProperty("SMBIOSBIOSVersion")
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

// SetSMBIOSMajorVersion sets the value of SMBIOSMajorVersion for the instance
func (instance *Win32_BIOS) SetPropertySMBIOSMajorVersion(value uint16) (err error) {
	return instance.SetProperty("SMBIOSMajorVersion", (value))
}

// GetSMBIOSMajorVersion gets the value of SMBIOSMajorVersion for the instance
func (instance *Win32_BIOS) GetPropertySMBIOSMajorVersion() (value uint16, err error) {
	retValue, err := instance.GetProperty("SMBIOSMajorVersion")
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

// SetSMBIOSMinorVersion sets the value of SMBIOSMinorVersion for the instance
func (instance *Win32_BIOS) SetPropertySMBIOSMinorVersion(value uint16) (err error) {
	return instance.SetProperty("SMBIOSMinorVersion", (value))
}

// GetSMBIOSMinorVersion gets the value of SMBIOSMinorVersion for the instance
func (instance *Win32_BIOS) GetPropertySMBIOSMinorVersion() (value uint16, err error) {
	retValue, err := instance.GetProperty("SMBIOSMinorVersion")
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

// SetSMBIOSPresent sets the value of SMBIOSPresent for the instance
func (instance *Win32_BIOS) SetPropertySMBIOSPresent(value bool) (err error) {
	return instance.SetProperty("SMBIOSPresent", (value))
}

// GetSMBIOSPresent gets the value of SMBIOSPresent for the instance
func (instance *Win32_BIOS) GetPropertySMBIOSPresent() (value bool, err error) {
	retValue, err := instance.GetProperty("SMBIOSPresent")
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

// SetSystemBiosMajorVersion sets the value of SystemBiosMajorVersion for the instance
func (instance *Win32_BIOS) SetPropertySystemBiosMajorVersion(value uint8) (err error) {
	return instance.SetProperty("SystemBiosMajorVersion", (value))
}

// GetSystemBiosMajorVersion gets the value of SystemBiosMajorVersion for the instance
func (instance *Win32_BIOS) GetPropertySystemBiosMajorVersion() (value uint8, err error) {
	retValue, err := instance.GetProperty("SystemBiosMajorVersion")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint8)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint8 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint8(valuetmp)

	return
}

// SetSystemBiosMinorVersion sets the value of SystemBiosMinorVersion for the instance
func (instance *Win32_BIOS) SetPropertySystemBiosMinorVersion(value uint8) (err error) {
	return instance.SetProperty("SystemBiosMinorVersion", (value))
}

// GetSystemBiosMinorVersion gets the value of SystemBiosMinorVersion for the instance
func (instance *Win32_BIOS) GetPropertySystemBiosMinorVersion() (value uint8, err error) {
	retValue, err := instance.GetProperty("SystemBiosMinorVersion")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint8)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint8 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint8(valuetmp)

	return
}
