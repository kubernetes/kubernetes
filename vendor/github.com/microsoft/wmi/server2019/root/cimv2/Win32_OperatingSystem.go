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

// Win32_OperatingSystem struct
type Win32_OperatingSystem struct {
	*CIM_OperatingSystem

	//
	BootDevice string

	//
	BuildNumber string

	//
	BuildType string

	//
	CodeSet string

	//
	CountryCode string

	//
	CSDVersion string

	//
	DataExecutionPrevention_32BitApplications bool

	//
	DataExecutionPrevention_Available bool

	//
	DataExecutionPrevention_Drivers bool

	//
	DataExecutionPrevention_SupportPolicy uint8

	//
	Debug bool

	//
	EncryptionLevel uint32

	//
	ForegroundApplicationBoost uint8

	//
	LargeSystemCache uint32

	//
	Locale string

	//
	Manufacturer string

	//
	MUILanguages []string

	//
	OperatingSystemSKU uint32

	//
	Organization string

	//
	OSArchitecture string

	//
	OSLanguage uint32

	//
	OSProductSuite uint32

	//
	PAEEnabled bool

	//
	PlusProductID string

	//
	PlusVersionNumber string

	//
	PortableOperatingSystem bool

	//
	Primary bool

	//
	ProductType uint32

	//
	RegisteredUser string

	//
	SerialNumber string

	//
	ServicePackMajorVersion uint16

	//
	ServicePackMinorVersion uint16

	//
	SuiteMask uint32

	//
	SystemDevice string

	//
	SystemDirectory string

	//
	SystemDrive string

	//
	WindowsDirectory string
}

func NewWin32_OperatingSystemEx1(instance *cim.WmiInstance) (newInstance *Win32_OperatingSystem, err error) {
	tmp, err := NewCIM_OperatingSystemEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_OperatingSystem{
		CIM_OperatingSystem: tmp,
	}
	return
}

func NewWin32_OperatingSystemEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_OperatingSystem, err error) {
	tmp, err := NewCIM_OperatingSystemEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_OperatingSystem{
		CIM_OperatingSystem: tmp,
	}
	return
}

// SetBootDevice sets the value of BootDevice for the instance
func (instance *Win32_OperatingSystem) SetPropertyBootDevice(value string) (err error) {
	return instance.SetProperty("BootDevice", (value))
}

// GetBootDevice gets the value of BootDevice for the instance
func (instance *Win32_OperatingSystem) GetPropertyBootDevice() (value string, err error) {
	retValue, err := instance.GetProperty("BootDevice")
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

// SetBuildNumber sets the value of BuildNumber for the instance
func (instance *Win32_OperatingSystem) SetPropertyBuildNumber(value string) (err error) {
	return instance.SetProperty("BuildNumber", (value))
}

// GetBuildNumber gets the value of BuildNumber for the instance
func (instance *Win32_OperatingSystem) GetPropertyBuildNumber() (value string, err error) {
	retValue, err := instance.GetProperty("BuildNumber")
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

// SetBuildType sets the value of BuildType for the instance
func (instance *Win32_OperatingSystem) SetPropertyBuildType(value string) (err error) {
	return instance.SetProperty("BuildType", (value))
}

// GetBuildType gets the value of BuildType for the instance
func (instance *Win32_OperatingSystem) GetPropertyBuildType() (value string, err error) {
	retValue, err := instance.GetProperty("BuildType")
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

// SetCodeSet sets the value of CodeSet for the instance
func (instance *Win32_OperatingSystem) SetPropertyCodeSet(value string) (err error) {
	return instance.SetProperty("CodeSet", (value))
}

// GetCodeSet gets the value of CodeSet for the instance
func (instance *Win32_OperatingSystem) GetPropertyCodeSet() (value string, err error) {
	retValue, err := instance.GetProperty("CodeSet")
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

// SetCountryCode sets the value of CountryCode for the instance
func (instance *Win32_OperatingSystem) SetPropertyCountryCode(value string) (err error) {
	return instance.SetProperty("CountryCode", (value))
}

// GetCountryCode gets the value of CountryCode for the instance
func (instance *Win32_OperatingSystem) GetPropertyCountryCode() (value string, err error) {
	retValue, err := instance.GetProperty("CountryCode")
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

// SetCSDVersion sets the value of CSDVersion for the instance
func (instance *Win32_OperatingSystem) SetPropertyCSDVersion(value string) (err error) {
	return instance.SetProperty("CSDVersion", (value))
}

// GetCSDVersion gets the value of CSDVersion for the instance
func (instance *Win32_OperatingSystem) GetPropertyCSDVersion() (value string, err error) {
	retValue, err := instance.GetProperty("CSDVersion")
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

// SetDataExecutionPrevention_32BitApplications sets the value of DataExecutionPrevention_32BitApplications for the instance
func (instance *Win32_OperatingSystem) SetPropertyDataExecutionPrevention_32BitApplications(value bool) (err error) {
	return instance.SetProperty("DataExecutionPrevention_32BitApplications", (value))
}

// GetDataExecutionPrevention_32BitApplications gets the value of DataExecutionPrevention_32BitApplications for the instance
func (instance *Win32_OperatingSystem) GetPropertyDataExecutionPrevention_32BitApplications() (value bool, err error) {
	retValue, err := instance.GetProperty("DataExecutionPrevention_32BitApplications")
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

// SetDataExecutionPrevention_Available sets the value of DataExecutionPrevention_Available for the instance
func (instance *Win32_OperatingSystem) SetPropertyDataExecutionPrevention_Available(value bool) (err error) {
	return instance.SetProperty("DataExecutionPrevention_Available", (value))
}

// GetDataExecutionPrevention_Available gets the value of DataExecutionPrevention_Available for the instance
func (instance *Win32_OperatingSystem) GetPropertyDataExecutionPrevention_Available() (value bool, err error) {
	retValue, err := instance.GetProperty("DataExecutionPrevention_Available")
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

// SetDataExecutionPrevention_Drivers sets the value of DataExecutionPrevention_Drivers for the instance
func (instance *Win32_OperatingSystem) SetPropertyDataExecutionPrevention_Drivers(value bool) (err error) {
	return instance.SetProperty("DataExecutionPrevention_Drivers", (value))
}

// GetDataExecutionPrevention_Drivers gets the value of DataExecutionPrevention_Drivers for the instance
func (instance *Win32_OperatingSystem) GetPropertyDataExecutionPrevention_Drivers() (value bool, err error) {
	retValue, err := instance.GetProperty("DataExecutionPrevention_Drivers")
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

// SetDataExecutionPrevention_SupportPolicy sets the value of DataExecutionPrevention_SupportPolicy for the instance
func (instance *Win32_OperatingSystem) SetPropertyDataExecutionPrevention_SupportPolicy(value uint8) (err error) {
	return instance.SetProperty("DataExecutionPrevention_SupportPolicy", (value))
}

// GetDataExecutionPrevention_SupportPolicy gets the value of DataExecutionPrevention_SupportPolicy for the instance
func (instance *Win32_OperatingSystem) GetPropertyDataExecutionPrevention_SupportPolicy() (value uint8, err error) {
	retValue, err := instance.GetProperty("DataExecutionPrevention_SupportPolicy")
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

// SetDebug sets the value of Debug for the instance
func (instance *Win32_OperatingSystem) SetPropertyDebug(value bool) (err error) {
	return instance.SetProperty("Debug", (value))
}

// GetDebug gets the value of Debug for the instance
func (instance *Win32_OperatingSystem) GetPropertyDebug() (value bool, err error) {
	retValue, err := instance.GetProperty("Debug")
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

// SetEncryptionLevel sets the value of EncryptionLevel for the instance
func (instance *Win32_OperatingSystem) SetPropertyEncryptionLevel(value uint32) (err error) {
	return instance.SetProperty("EncryptionLevel", (value))
}

// GetEncryptionLevel gets the value of EncryptionLevel for the instance
func (instance *Win32_OperatingSystem) GetPropertyEncryptionLevel() (value uint32, err error) {
	retValue, err := instance.GetProperty("EncryptionLevel")
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

// SetForegroundApplicationBoost sets the value of ForegroundApplicationBoost for the instance
func (instance *Win32_OperatingSystem) SetPropertyForegroundApplicationBoost(value uint8) (err error) {
	return instance.SetProperty("ForegroundApplicationBoost", (value))
}

// GetForegroundApplicationBoost gets the value of ForegroundApplicationBoost for the instance
func (instance *Win32_OperatingSystem) GetPropertyForegroundApplicationBoost() (value uint8, err error) {
	retValue, err := instance.GetProperty("ForegroundApplicationBoost")
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

// SetLargeSystemCache sets the value of LargeSystemCache for the instance
func (instance *Win32_OperatingSystem) SetPropertyLargeSystemCache(value uint32) (err error) {
	return instance.SetProperty("LargeSystemCache", (value))
}

// GetLargeSystemCache gets the value of LargeSystemCache for the instance
func (instance *Win32_OperatingSystem) GetPropertyLargeSystemCache() (value uint32, err error) {
	retValue, err := instance.GetProperty("LargeSystemCache")
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

// SetLocale sets the value of Locale for the instance
func (instance *Win32_OperatingSystem) SetPropertyLocale(value string) (err error) {
	return instance.SetProperty("Locale", (value))
}

// GetLocale gets the value of Locale for the instance
func (instance *Win32_OperatingSystem) GetPropertyLocale() (value string, err error) {
	retValue, err := instance.GetProperty("Locale")
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
func (instance *Win32_OperatingSystem) SetPropertyManufacturer(value string) (err error) {
	return instance.SetProperty("Manufacturer", (value))
}

// GetManufacturer gets the value of Manufacturer for the instance
func (instance *Win32_OperatingSystem) GetPropertyManufacturer() (value string, err error) {
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

// SetMUILanguages sets the value of MUILanguages for the instance
func (instance *Win32_OperatingSystem) SetPropertyMUILanguages(value []string) (err error) {
	return instance.SetProperty("MUILanguages", (value))
}

// GetMUILanguages gets the value of MUILanguages for the instance
func (instance *Win32_OperatingSystem) GetPropertyMUILanguages() (value []string, err error) {
	retValue, err := instance.GetProperty("MUILanguages")
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

// SetOperatingSystemSKU sets the value of OperatingSystemSKU for the instance
func (instance *Win32_OperatingSystem) SetPropertyOperatingSystemSKU(value uint32) (err error) {
	return instance.SetProperty("OperatingSystemSKU", (value))
}

// GetOperatingSystemSKU gets the value of OperatingSystemSKU for the instance
func (instance *Win32_OperatingSystem) GetPropertyOperatingSystemSKU() (value uint32, err error) {
	retValue, err := instance.GetProperty("OperatingSystemSKU")
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

// SetOrganization sets the value of Organization for the instance
func (instance *Win32_OperatingSystem) SetPropertyOrganization(value string) (err error) {
	return instance.SetProperty("Organization", (value))
}

// GetOrganization gets the value of Organization for the instance
func (instance *Win32_OperatingSystem) GetPropertyOrganization() (value string, err error) {
	retValue, err := instance.GetProperty("Organization")
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

// SetOSArchitecture sets the value of OSArchitecture for the instance
func (instance *Win32_OperatingSystem) SetPropertyOSArchitecture(value string) (err error) {
	return instance.SetProperty("OSArchitecture", (value))
}

// GetOSArchitecture gets the value of OSArchitecture for the instance
func (instance *Win32_OperatingSystem) GetPropertyOSArchitecture() (value string, err error) {
	retValue, err := instance.GetProperty("OSArchitecture")
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

// SetOSLanguage sets the value of OSLanguage for the instance
func (instance *Win32_OperatingSystem) SetPropertyOSLanguage(value uint32) (err error) {
	return instance.SetProperty("OSLanguage", (value))
}

// GetOSLanguage gets the value of OSLanguage for the instance
func (instance *Win32_OperatingSystem) GetPropertyOSLanguage() (value uint32, err error) {
	retValue, err := instance.GetProperty("OSLanguage")
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

// SetOSProductSuite sets the value of OSProductSuite for the instance
func (instance *Win32_OperatingSystem) SetPropertyOSProductSuite(value uint32) (err error) {
	return instance.SetProperty("OSProductSuite", (value))
}

// GetOSProductSuite gets the value of OSProductSuite for the instance
func (instance *Win32_OperatingSystem) GetPropertyOSProductSuite() (value uint32, err error) {
	retValue, err := instance.GetProperty("OSProductSuite")
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

// SetPAEEnabled sets the value of PAEEnabled for the instance
func (instance *Win32_OperatingSystem) SetPropertyPAEEnabled(value bool) (err error) {
	return instance.SetProperty("PAEEnabled", (value))
}

// GetPAEEnabled gets the value of PAEEnabled for the instance
func (instance *Win32_OperatingSystem) GetPropertyPAEEnabled() (value bool, err error) {
	retValue, err := instance.GetProperty("PAEEnabled")
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

// SetPlusProductID sets the value of PlusProductID for the instance
func (instance *Win32_OperatingSystem) SetPropertyPlusProductID(value string) (err error) {
	return instance.SetProperty("PlusProductID", (value))
}

// GetPlusProductID gets the value of PlusProductID for the instance
func (instance *Win32_OperatingSystem) GetPropertyPlusProductID() (value string, err error) {
	retValue, err := instance.GetProperty("PlusProductID")
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

// SetPlusVersionNumber sets the value of PlusVersionNumber for the instance
func (instance *Win32_OperatingSystem) SetPropertyPlusVersionNumber(value string) (err error) {
	return instance.SetProperty("PlusVersionNumber", (value))
}

// GetPlusVersionNumber gets the value of PlusVersionNumber for the instance
func (instance *Win32_OperatingSystem) GetPropertyPlusVersionNumber() (value string, err error) {
	retValue, err := instance.GetProperty("PlusVersionNumber")
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

// SetPortableOperatingSystem sets the value of PortableOperatingSystem for the instance
func (instance *Win32_OperatingSystem) SetPropertyPortableOperatingSystem(value bool) (err error) {
	return instance.SetProperty("PortableOperatingSystem", (value))
}

// GetPortableOperatingSystem gets the value of PortableOperatingSystem for the instance
func (instance *Win32_OperatingSystem) GetPropertyPortableOperatingSystem() (value bool, err error) {
	retValue, err := instance.GetProperty("PortableOperatingSystem")
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

// SetPrimary sets the value of Primary for the instance
func (instance *Win32_OperatingSystem) SetPropertyPrimary(value bool) (err error) {
	return instance.SetProperty("Primary", (value))
}

// GetPrimary gets the value of Primary for the instance
func (instance *Win32_OperatingSystem) GetPropertyPrimary() (value bool, err error) {
	retValue, err := instance.GetProperty("Primary")
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

// SetProductType sets the value of ProductType for the instance
func (instance *Win32_OperatingSystem) SetPropertyProductType(value uint32) (err error) {
	return instance.SetProperty("ProductType", (value))
}

// GetProductType gets the value of ProductType for the instance
func (instance *Win32_OperatingSystem) GetPropertyProductType() (value uint32, err error) {
	retValue, err := instance.GetProperty("ProductType")
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

// SetRegisteredUser sets the value of RegisteredUser for the instance
func (instance *Win32_OperatingSystem) SetPropertyRegisteredUser(value string) (err error) {
	return instance.SetProperty("RegisteredUser", (value))
}

// GetRegisteredUser gets the value of RegisteredUser for the instance
func (instance *Win32_OperatingSystem) GetPropertyRegisteredUser() (value string, err error) {
	retValue, err := instance.GetProperty("RegisteredUser")
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

// SetSerialNumber sets the value of SerialNumber for the instance
func (instance *Win32_OperatingSystem) SetPropertySerialNumber(value string) (err error) {
	return instance.SetProperty("SerialNumber", (value))
}

// GetSerialNumber gets the value of SerialNumber for the instance
func (instance *Win32_OperatingSystem) GetPropertySerialNumber() (value string, err error) {
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

// SetServicePackMajorVersion sets the value of ServicePackMajorVersion for the instance
func (instance *Win32_OperatingSystem) SetPropertyServicePackMajorVersion(value uint16) (err error) {
	return instance.SetProperty("ServicePackMajorVersion", (value))
}

// GetServicePackMajorVersion gets the value of ServicePackMajorVersion for the instance
func (instance *Win32_OperatingSystem) GetPropertyServicePackMajorVersion() (value uint16, err error) {
	retValue, err := instance.GetProperty("ServicePackMajorVersion")
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

// SetServicePackMinorVersion sets the value of ServicePackMinorVersion for the instance
func (instance *Win32_OperatingSystem) SetPropertyServicePackMinorVersion(value uint16) (err error) {
	return instance.SetProperty("ServicePackMinorVersion", (value))
}

// GetServicePackMinorVersion gets the value of ServicePackMinorVersion for the instance
func (instance *Win32_OperatingSystem) GetPropertyServicePackMinorVersion() (value uint16, err error) {
	retValue, err := instance.GetProperty("ServicePackMinorVersion")
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

// SetSuiteMask sets the value of SuiteMask for the instance
func (instance *Win32_OperatingSystem) SetPropertySuiteMask(value uint32) (err error) {
	return instance.SetProperty("SuiteMask", (value))
}

// GetSuiteMask gets the value of SuiteMask for the instance
func (instance *Win32_OperatingSystem) GetPropertySuiteMask() (value uint32, err error) {
	retValue, err := instance.GetProperty("SuiteMask")
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

// SetSystemDevice sets the value of SystemDevice for the instance
func (instance *Win32_OperatingSystem) SetPropertySystemDevice(value string) (err error) {
	return instance.SetProperty("SystemDevice", (value))
}

// GetSystemDevice gets the value of SystemDevice for the instance
func (instance *Win32_OperatingSystem) GetPropertySystemDevice() (value string, err error) {
	retValue, err := instance.GetProperty("SystemDevice")
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

// SetSystemDirectory sets the value of SystemDirectory for the instance
func (instance *Win32_OperatingSystem) SetPropertySystemDirectory(value string) (err error) {
	return instance.SetProperty("SystemDirectory", (value))
}

// GetSystemDirectory gets the value of SystemDirectory for the instance
func (instance *Win32_OperatingSystem) GetPropertySystemDirectory() (value string, err error) {
	retValue, err := instance.GetProperty("SystemDirectory")
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

// SetSystemDrive sets the value of SystemDrive for the instance
func (instance *Win32_OperatingSystem) SetPropertySystemDrive(value string) (err error) {
	return instance.SetProperty("SystemDrive", (value))
}

// GetSystemDrive gets the value of SystemDrive for the instance
func (instance *Win32_OperatingSystem) GetPropertySystemDrive() (value string, err error) {
	retValue, err := instance.GetProperty("SystemDrive")
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

// SetWindowsDirectory sets the value of WindowsDirectory for the instance
func (instance *Win32_OperatingSystem) SetPropertyWindowsDirectory(value string) (err error) {
	return instance.SetProperty("WindowsDirectory", (value))
}

// GetWindowsDirectory gets the value of WindowsDirectory for the instance
func (instance *Win32_OperatingSystem) GetPropertyWindowsDirectory() (value string, err error) {
	retValue, err := instance.GetProperty("WindowsDirectory")
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

//

// <param name="Flags" type="int32 "></param>
// <param name="Reserved" type="int32 "></param>

// <param name="ReturnValue" type="uint32 "></param>
func (instance *Win32_OperatingSystem) Win32Shutdown( /* IN */ Flags int32,
	/* IN */ Reserved int32) (result uint32, err error) {
	retVal, err := instance.InvokeMethodWithReturn("Win32Shutdown", Flags, Reserved)
	if err != nil {
		return
	}
	result = uint32(retVal)
	return

}

//

// <param name="Comment" type="string "></param>
// <param name="Flags" type="int32 "></param>
// <param name="ReasonCode" type="uint32 "></param>
// <param name="Timeout" type="uint32 "></param>

// <param name="ReturnValue" type="uint32 "></param>
func (instance *Win32_OperatingSystem) Win32ShutdownTracker( /* IN */ Timeout uint32,
	/* IN */ Comment string,
	/* IN */ ReasonCode uint32,
	/* IN */ Flags int32) (result uint32, err error) {
	retVal, err := instance.InvokeMethodWithReturn("Win32ShutdownTracker", Timeout, Comment, ReasonCode, Flags)
	if err != nil {
		return
	}
	result = uint32(retVal)
	return

}

//

// <param name="LocalDateTime" type="string "></param>

// <param name="ReturnValue" type="uint32 "></param>
func (instance *Win32_OperatingSystem) SetDateTime( /* IN */ LocalDateTime string) (result uint32, err error) {
	retVal, err := instance.InvokeMethodWithReturn("SetDateTime", LocalDateTime)
	if err != nil {
		return
	}
	result = uint32(retVal)
	return

}
